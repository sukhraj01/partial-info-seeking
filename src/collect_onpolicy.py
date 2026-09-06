"""Collect the game states a solver ACTUALLY VISITS during play (DAgger).

Round-0 states are drawn from an arbitrary distribution: uniform reveal-count,
uniform subset, frequency-weighted eliminations. A real game visits a very
different distribution -- revealed and eliminated sets are correlated (the
solver guesses frequent letters first), and the states that decide games are
late, heavily-masked, low-lives positions that uniform sampling barely touches.

This is the classic behavioural-cloning covariate shift. DAgger's fix is to
train on states the current policy induces. Normally the expensive part is
querying an expert for labels; here the label is "which letters are in the
word", and we know the word, so labelling is free.

Output matches generate_training_data.py exactly, plus won.u8 (whether the game
this state came from was eventually won) so the same dump can train a value
function later.
"""
import argparse
import json
import os
import sys
import time
from array import array

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAX_LEN = 29
PAD, BLANK, LETTER_BASE = 0, 1, 2
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
IDX = {c: i for i, c in enumerate(ALPHABET)}
MAX_WRONG = 6


def load_words(path):
    with open(path) as f:
        return [w.strip() for w in f if w.strip()]


def collect(predict_batch, words, chunk_size=4096, max_wrong=MAX_WRONG, progress_every=0):
    """Play every word; record each visited state. Returns parallel arrays."""
    pat_rows = array("b")
    elim_rows = array("I")
    tgt_rows = array("I")
    widx_rows = array("I")
    won_rows = array("B")

    stats = {"games": 0, "wins": 0, "states": 0}
    t0 = time.time()

    for c0 in range(0, len(words), chunk_size):
        chunk = words[c0:c0 + chunk_size]
        patterns = [["_"] * len(w) for w in chunk]
        guessed = [[] for _ in chunk]
        gsets = [set() for _ in chunk]
        wrong = [0] * len(chunk)
        # rows recorded for each game, so `won` can be backfilled at game end
        rows_of = [[] for _ in chunk]
        active = list(range(len(chunk)))

        while active:
            req_pat = ["".join(patterns[i]) for i in active]
            req_guess = [guessed[i] for i in active]

            # Record the state BEFORE the guess -- that is the state the policy
            # was actually asked to act on.
            for i, pat in zip(active, req_pat):
                word = chunk[i]
                revealed = set(pat) - {"_"}
                elim_mask = 0
                for ch in gsets[i]:
                    if ch not in revealed:
                        elim_mask |= 1 << IDX[ch]
                tgt_mask = 0
                for ch in set(word):
                    if ch not in revealed:
                        tgt_mask |= 1 << IDX[ch]

                row = [PAD] * MAX_LEN
                for p, ch in enumerate(pat):
                    row[p] = BLANK if ch == "_" else LETTER_BASE + IDX[ch]
                rows_of[i].append(len(widx_rows))
                pat_rows.extend(row)
                elim_rows.append(elim_mask)
                tgt_rows.append(tgt_mask)
                widx_rows.append(c0 + i)
                won_rows.append(0)  # backfilled below

            letters = predict_batch(req_pat, req_guess)
            still = []
            for i, g in zip(active, letters):
                if g in gsets[i]:
                    wrong[i] += 1
                else:
                    guessed[i].append(g)
                    gsets[i].add(g)
                    word = chunk[i]
                    if g in word:
                        pat = patterns[i]
                        for p, ch in enumerate(word):
                            if ch == g:
                                pat[p] = ch
                    else:
                        wrong[i] += 1
                if wrong[i] < max_wrong and "_" in patterns[i]:
                    still.append(i)
            active = still

        for i in range(len(chunk)):
            won = 1 if "_" not in patterns[i] else 0
            stats["games"] += 1
            stats["wins"] += won
            if won:
                for r in rows_of[i]:
                    won_rows[r] = 1

        stats["states"] = len(widx_rows)
        if progress_every and (c0 // chunk_size) % max(1, progress_every // chunk_size) == 0:
            el = time.time() - t0
            done = min(c0 + chunk_size, len(words))
            print(f"  {done:,}/{len(words):,} words | win {stats['wins']/stats['games']*100:.1f}% "
                  f"| {stats['states']:,} states | {done/el:.0f} w/s", flush=True)

    return pat_rows, elim_rows, tgt_rows, widx_rows, won_rows, stats


def write_dump(out_dir, arrays, words, meta_extra):
    import numpy as np
    pat_rows, elim_rows, tgt_rows, widx_rows, won_rows, stats = arrays
    n = len(widx_rows)
    os.makedirs(out_dir, exist_ok=True)

    np.frombuffer(pat_rows, dtype=np.int8).reshape(n, MAX_LEN).tofile(
        os.path.join(out_dir, "patterns.i8"))
    np.frombuffer(elim_rows, dtype=np.uint32).tofile(os.path.join(out_dir, "eliminated.u32"))
    np.frombuffer(tgt_rows, dtype=np.uint32).tofile(os.path.join(out_dir, "target.u32"))
    np.frombuffer(widx_rows, dtype=np.uint32).tofile(os.path.join(out_dir, "word_idx.u32"))
    np.frombuffer(won_rows, dtype=np.uint8).tofile(os.path.join(out_dir, "won.u8"))

    words_path = os.path.join(out_dir, "words_used.txt")
    with open(words_path, "w") as f:
        f.write("\n".join(words) + "\n")

    meta = {"n_states": n, "n_words": len(words), "states_per_word": None,
            "max_len": MAX_LEN,
            "encoding": {"PAD": PAD, "BLANK": BLANK, "letter_base": LETTER_BASE,
                         "vocab_size": LETTER_BASE + 26},
            "source_words": words_path, "on_policy": True,
            "has_won_labels": True,
            "sim_win_rate": stats["wins"] / max(stats["games"], 1) * 100,
            "games": stats["games"]}
    meta.update(meta_extra)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def describe(out_dir):
    """Compare the on-policy state distribution against round-0's."""
    import numpy as np
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    n = meta["n_states"]
    pat = np.memmap(os.path.join(out_dir, "patterns.i8"), dtype=np.int8, mode="r",
                    shape=(n, MAX_LEN))
    elim = np.memmap(os.path.join(out_dir, "eliminated.u32"), dtype=np.uint32,
                     mode="r", shape=(n,))
    take = min(n, 500_000)
    p, e = pat[:take], elim[:take]
    wlen = (p != PAD).sum(axis=1)
    blanks = (p == BLANK).sum(axis=1)
    n_elim = np.zeros(take, dtype=np.int32)
    for i in range(26):
        n_elim += ((e >> i) & 1).astype(np.int32)
    mask_rate = blanks / np.maximum(wlen, 1)
    print(f"  states                : {n:,}")
    print(f"  mean mask rate        : {mask_rate.mean():.3f}")
    print(f"  mean wrong-guesses    : {n_elim.mean():.3f}")
    hist = np.bincount(n_elim, minlength=7)[:7] / take * 100
    print("  wrong-guess distribution: "
          + ", ".join(f"{i}:{hist[i]:.1f}%" for i in range(6)))
    return {"mean_mask_rate": float(mask_rate.mean()),
            "mean_wrong": float(n_elim.mean()),
            "wrong_hist_pct": [float(x) for x in hist[:6]]}


def build_solver(kind, checkpoint, train_split, device=None, batch_size=4096):
    if kind == "transformer":
        from transformer_model import TransformerSolver
        s = TransformerSolver(checkpoint, device=device, batch_size=batch_size)
        return s.predict_batch, {"teacher": "transformer", "checkpoint": checkpoint}
    if kind == "ngram":
        from ngram_model import NgramModel
        m = NgramModel(train_split)
        return (lambda pats, gs: [m.predict(p, g) for p, g in zip(pats, gs)],
                {"teacher": "ngram"})
    raise ValueError(f"unknown teacher: {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", choices=["transformer", "ngram"], default="transformer")
    ap.add_argument("--checkpoint", default="models/transformer.pt")
    ap.add_argument("--words", default="data/train_split.txt")
    ap.add_argument("--train-split", default="data/train_split.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-words", type=int, default=None,
                    help="use only the first N words (keep the val slice out)")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    words = load_words(args.words)
    if args.max_words:
        words = words[:args.max_words]
    print(f"teacher={args.teacher} words={len(words):,}", flush=True)

    predict_batch, meta_extra = build_solver(args.teacher, args.checkpoint,
                                             args.train_split, args.device,
                                             args.batch_size)
    t0 = time.time()
    arrays = collect(predict_batch, words, chunk_size=args.chunk_size,
                     progress_every=40000)
    stats = arrays[5]
    print(f"collected {stats['states']:,} states from {stats['games']:,} games "
          f"in {(time.time()-t0)/60:.1f}m | sim win rate {stats['wins']/stats['games']*100:.2f}%",
          flush=True)

    meta_extra["max_words"] = args.max_words
    meta = write_dump(args.out, arrays, words, meta_extra)
    print(f"wrote {args.out}/")
    print("distribution:")
    dist = describe(args.out)
    meta.update(dist)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
