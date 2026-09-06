"""Training-state generation for the neural solver.

The one rule that matters: masking is LETTER-LEVEL, never position-level.
Hangman reveals every occurrence of a guessed letter simultaneously, so a state
like "l_tter" (one 'e' shown, the other hidden) is unreachable in a real game.
81.3% of our vocabulary contains a repeated letter, so position-level masking
would put most training examples outside the reachable state space entirely.

Each example is:
  pattern     - int8[MAX_LEN]  0=PAD, 1=BLANK, 2..27 = 'a'..'z'
  eliminated  - uint32 bitmask of letters guessed and known-absent
  target      - uint32 bitmask of letters present in the word but not yet revealed
  word_idx    - uint32 index into the source word list

`eliminated` is the negative evidence -- half the information the game gives you
-- and is a model input, not a label. `target` is the multi-label objective.

Two sampling strategies, selected with --strategy:

`uniform` (round 0, broad coverage): reveal-count uniform over 0..u-1, revealed
subset uniform, wrong-guess count uniform over 0..5, wrong guesses drawn from
genuinely-absent letters weighted by corpus frequency.

`gameplay` (round 0b, decision-dense coverage): same letter-level masking, but
the state distribution is reshaped toward the states where games are actually
decided.
  - Reveal fraction is drawn from Beta(0.45, 0.45), which is U-shaped: mass
    piles up at the opening (nothing revealed, every guess is a prior) and at
    the endgame (one or two letters left, where wins and losses are settled).
    Uniform reveal-count over-samples the easy middle.
  - Wrong-guess count is correlated with reveal fraction 65% of the time -- late
    states have burned lives, early states have not -- and drawn uniformly the
    other 35%, so the decorrelated coverage round 0 provides is not lost.
  - Elimination weighting is randomised per state via an exponent alpha on the
    corpus frequency: alpha=1 reproduces round 0's frequency-weighted draw,
    alpha=0 is uniform over absent letters, alpha=0.5 is between. Round 0 only
    ever eliminates high-frequency letters, so the model never sees the
    mid-frequency corners of the elimination space that real play reaches once
    the obvious guesses are gone.

The revealed SUBSET stays uniform in both strategies. Biasing it toward
frequent letters would match real play more closely, but the DAgger round-1
result showed that narrowing state diversity to match the on-policy
distribution costs more OOV generalisation than the distribution-match buys.

`word_idx` is stored so states can be relabelled without re-deriving the word.
"""
import argparse
import json
import os
import random
from array import array
from collections import Counter
from multiprocessing import Process

MAX_LEN = 29
PAD, BLANK, LETTER_BASE = 0, 1, 2
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
IDX = {c: i for i, c in enumerate(ALPHABET)}
MAX_WRONG = 6  # a live state has 0..5 wrong guesses


def load_words(path):
    with open(path) as f:
        return [w.strip() for w in f if w.strip()]


def letter_weights(words):
    """Fraction of words containing each letter -- how a real player picks guesses."""
    c = Counter()
    for w in words:
        c.update(set(w))
    n = len(words)
    return [c.get(ch, 0) / n for ch in ALPHABET]


def _gen_chunk(words, w_start, w_end, states_per_word, seed, out_dir, weights,
               total_rows, strategy="uniform"):
    """Generate states for words[w_start:w_end] into the preallocated slice.

    total_rows is passed explicitly: macOS spawns rather than forks workers, so
    module-level globals set in main() do not reach the child processes.
    """
    import numpy as np

    n_states = (w_end - w_start) * states_per_word
    row0 = w_start * states_per_word

    pat = np.memmap(os.path.join(out_dir, "patterns.i8"), dtype=np.int8, mode="r+",
                    shape=(total_rows, MAX_LEN))
    elim = np.memmap(os.path.join(out_dir, "eliminated.u32"), dtype=np.uint32, mode="r+",
                     shape=(total_rows,))
    tgt = np.memmap(os.path.join(out_dir, "target.u32"), dtype=np.uint32, mode="r+",
                    shape=(total_rows,))
    widx = np.memmap(os.path.join(out_dir, "word_idx.u32"), dtype=np.uint32, mode="r+",
                     shape=(total_rows,))

    rng = random.Random(seed)
    randrange, random_, sample = rng.randrange, rng.random, rng.sample
    betavariate, choices = rng.betavariate, rng.choices
    gameplay = strategy == "gameplay"
    # U-shaped over [0,1]: mass at the opening and at the endgame.
    BETA_A = BETA_B = 0.45
    ALPHAS, ALPHA_W = (1.0, 0.5, 0.0), (0.40, 0.35, 0.25)

    # Local buffers reused across states.
    buf_pat = np.zeros((n_states, MAX_LEN), dtype=np.int8)
    buf_elim = np.zeros(n_states, dtype=np.uint32)
    buf_tgt = np.zeros(n_states, dtype=np.uint32)
    buf_widx = np.zeros(n_states, dtype=np.uint32)

    row = 0
    for wi in range(w_start, w_end):
        word = words[wi]
        L = len(word)
        uniq = sorted(set(word))
        u = len(uniq)
        codes = [LETTER_BASE + IDX[c] for c in word]
        # Letters genuinely absent from this word, with their corpus weights.
        absent = [(i, weights[i]) for i in range(26) if ALPHABET[i] not in uniq]

        for _ in range(states_per_word):
            # --- reveal a uniformly-sized, uniformly-chosen subset of the
            # --- word's DISTINCT letters. Letter-level, so all occurrences go together.
            if gameplay:
                f = betavariate(BETA_A, BETA_B)
                k = int(f * u)
                if k >= u:  # f == 1.0 would mean the word is already solved
                    k = u - 1
            else:
                f = 0.0
                k = randrange(u)  # 0..u-1; u would mean the word is already solved
            revealed = set(sample(uniq, k)) if k else set()

            r = row
            prow = buf_pat[r]
            for p in range(L):
                prow[p] = codes[p] if word[p] in revealed else BLANK
            # positions L..MAX_LEN-1 stay PAD (buffer is zero-initialised)

            target_mask = 0
            for c in uniq:
                if c not in revealed:
                    target_mask |= 1 << IDX[c]

            # --- wrong guesses: frequency-weighted draw without replacement
            # --- from letters genuinely absent from the word.
            if gameplay and random_() < 0.65:
                # Correlate lives spent with progress: late states have burned
                # lives, early ones have not.
                if f >= 0.60:
                    n_wrong = randrange(2, MAX_WRONG)   # 2..5
                elif f <= 0.35:
                    n_wrong = randrange(0, 3)           # 0..2
                else:
                    n_wrong = randrange(MAX_WRONG)      # 0..5
            else:
                n_wrong = randrange(MAX_WRONG)  # 0..5
            # alpha=1 is round-0's frequency weighting, alpha=0 is uniform over
            # absent letters, so mid-frequency eliminations get covered too.
            alpha = choices(ALPHAS, ALPHA_W)[0] if gameplay else 1.0
            elim_mask = 0
            if n_wrong and absent:
                if n_wrong >= len(absent):
                    for i, _w in absent:
                        elim_mask |= 1 << i
                else:
                    # Efraimidis-Spirakis weighted reservoir: key = U^(1/w)
                    keyed = []
                    for i, wt in absent:
                        w_eff = 1.0 if alpha == 0.0 else (wt if alpha == 1.0 else wt ** alpha)
                        if w_eff <= 0.0:
                            keyed.append((0.0, i))
                        else:
                            keyed.append((random_() ** (1.0 / w_eff), i))
                    keyed.sort(reverse=True)
                    for _k, i in keyed[:n_wrong]:
                        elim_mask |= 1 << i

            buf_elim[r] = elim_mask
            buf_tgt[r] = target_mask
            buf_widx[r] = wi
            row += 1

    pat[row0:row0 + n_states] = buf_pat
    elim[row0:row0 + n_states] = buf_elim
    tgt[row0:row0 + n_states] = buf_tgt
    widx[row0:row0 + n_states] = buf_widx
    for m in (pat, elim, tgt, widx):
        m.flush()


def main():
    ap = argparse.ArgumentParser(
        description="Generate hangman training states. Defaults target Ada; "
                    "use --small for a local correctness check.")
    ap.add_argument("--words", default="data/train_split.txt")
    ap.add_argument("--out", default="data/states")
    ap.add_argument("--n-states", type=int, default=50_000_000)
    ap.add_argument("--max-words", type=int, default=None,
                    help="cap the number of source words (used by --small)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--strategy", choices=["uniform", "gameplay"], default="uniform",
                    help="uniform = round-0 broad coverage; gameplay = U-shaped "
                         "reveal fraction, correlated lives, randomised "
                         "elimination weighting")
    ap.add_argument("--small", action="store_true",
                    help="100K states from 20K words (~4 MB) for local testing")
    args = ap.parse_args()

    if args.small:
        args.n_states = 100_000
        args.max_words = 20_000
        if args.out == "data/states":
            args.out = "data/states_small"

    import numpy as np

    words = load_words(args.words)
    if args.max_words and args.max_words < len(words):
        # Deterministic subsample so --small is reproducible.
        rng = random.Random(args.seed)
        words = rng.sample(words, args.max_words)
        words.sort()
    n_words = len(words)
    states_per_word = max(1, round(args.n_states / n_words))
    total_rows = n_words * states_per_word
    weights = letter_weights(words)

    os.makedirs(args.out, exist_ok=True)
    # Preallocate the output files so workers can write disjoint slices.
    for name, dtype, shape in [
        ("patterns.i8", np.int8, (total_rows, MAX_LEN)),
        ("eliminated.u32", np.uint32, (total_rows,)),
        ("target.u32", np.uint32, (total_rows,)),
        ("word_idx.u32", np.uint32, (total_rows,)),
    ]:
        np.memmap(os.path.join(args.out, name), dtype=dtype, mode="w+", shape=shape).flush()

    gb = total_rows * (MAX_LEN + 12) / 1e9
    print(f"{n_words:,} words x {states_per_word} states = {total_rows:,} rows (~{gb:.2f} GB)")

    bounds = [round(i * n_words / args.workers) for i in range(args.workers + 1)]
    procs = []
    for w in range(args.workers):
        a, b = bounds[w], bounds[w + 1]
        if a >= b:
            continue
        p = Process(target=_gen_chunk,
                    args=(words, a, b, states_per_word, args.seed + w, args.out, weights,
                          total_rows, args.strategy))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"worker failed with exit code {p.exitcode}")

    # word_idx indexes THIS list, which may be a subsample -- persist it so the
    # dataset is self-contained and survives being rsynced to another machine.
    words_path = os.path.join(args.out, "words_used.txt")
    with open(words_path, "w") as f:
        f.write("\n".join(words) + "\n")

    meta = {
        "n_states": total_rows,
        "n_words": n_words,
        "states_per_word": states_per_word,
        "max_len": MAX_LEN,
        "encoding": {"PAD": PAD, "BLANK": BLANK, "letter_base": LETTER_BASE,
                     "vocab_size": LETTER_BASE + 26},
        "source_words": words_path,
        "source_words_original": args.words,
        "max_words": args.max_words,
        "seed": args.seed,
        "round": 0,
        "strategy": args.strategy,
        "sampling": ("uniform reveal-count, uniform subset, freq-weighted absent "
                     "eliminations") if args.strategy == "uniform" else
                    ("Beta(0.45,0.45) reveal fraction, uniform subset, lives "
                     "correlated with progress (65%), elimination weight "
                     "alpha in {1,0.5,0}"),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.out}/ (patterns.i8, eliminated.u32, target.u32, "
          f"word_idx.u32, words_used.txt, meta.json)")


if __name__ == "__main__":
    main()
