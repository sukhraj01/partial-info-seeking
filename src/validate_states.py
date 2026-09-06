"""Validate generated training states against the invariants of a real hangman game.

Reads the persisted arrays back and reconstructs each source word via word_idx,
so this checks what was actually written to disk, not a re-generation.

The invariant that matters most (#2) is letter-level masking: every distinct
letter of the word must be either fully revealed or fully blank. A single
partially-revealed repeated letter means the pipeline is producing states that
can never occur in a real game.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

MAX_LEN = 29
PAD, BLANK, LETTER_BASE = 0, 1, 2
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def mask_to_set(m):
    return {ALPHABET[i] for i in range(26) if m >> i & 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/states")
    ap.add_argument("--sample", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import numpy as np

    with open(os.path.join(args.dir, "meta.json")) as f:
        meta = json.load(f)
    n = meta["n_states"]
    words = [w.strip() for w in open(meta["source_words"]) if w.strip()]

    pat = np.memmap(os.path.join(args.dir, "patterns.i8"), dtype=np.int8, mode="r",
                    shape=(n, MAX_LEN))
    elim = np.memmap(os.path.join(args.dir, "eliminated.u32"), dtype=np.uint32, mode="r",
                     shape=(n,))
    tgt = np.memmap(os.path.join(args.dir, "target.u32"), dtype=np.uint32, mode="r",
                    shape=(n,))
    widx = np.memmap(os.path.join(args.dir, "word_idx.u32"), dtype=np.uint32, mode="r",
                     shape=(n,))

    rng = random.Random(args.seed)
    idxs = ([rng.randrange(n) for _ in range(args.sample)]
            if args.sample < n else list(range(n)))

    fails = Counter()
    examples = {}
    reveal_frac, elim_counts, blank_counts = [], [], []

    def fail(kind, i, detail):
        fails[kind] += 1
        examples.setdefault(kind, (i, detail))

    for i in idxs:
        word = words[int(widx[i])]
        L = len(word)
        row = pat[i]
        e_set = mask_to_set(int(elim[i]))
        t_set = mask_to_set(int(tgt[i]))
        wl = set(word)

        # 1. padding is exactly the tail beyond the word
        if L > MAX_LEN or any(row[p] == PAD for p in range(L)) \
                or any(row[p] != PAD for p in range(L, MAX_LEN)):
            fail("padding", i, f"{word!r} row={list(row[:L+2])}")
            continue

        # 3. revealed characters must match the word
        revealed = set()
        bad_char = False
        for p in range(L):
            v = int(row[p])
            if v == BLANK:
                continue
            ch = ALPHABET[v - LETTER_BASE]
            if ch != word[p]:
                fail("revealed_char_mismatch", i, f"{word!r} pos{p} got {ch!r}")
                bad_char = True
                break
            revealed.add(ch)
        if bad_char:
            continue

        # 2. THE critical one: letter-level masking, no partial reveals
        partial = False
        for ch in wl:
            positions = [p for p in range(L) if word[p] == ch]
            shown = [int(row[p]) != BLANK for p in positions]
            if any(shown) and not all(shown):
                fail("partial_letter_reveal", i,
                     f"{word!r} letter {ch!r} shown at {[p for p,s in zip(positions,shown) if s]} "
                     f"but hidden at {[p for p,s in zip(positions,shown) if not s]}")
                partial = True
                break
        if partial:
            continue

        # 4/5. eliminated letters must be genuinely absent, and never revealed
        if e_set & wl:
            fail("eliminated_letter_in_word", i, f"{word!r} elim={sorted(e_set & wl)}")
            continue
        if e_set & revealed:
            fail("eliminated_and_revealed", i, f"{word!r} {sorted(e_set & revealed)}")
            continue

        # 6/7. target is exactly the unrevealed letters, and is non-empty
        expected = wl - revealed
        if t_set != expected:
            fail("target_mismatch", i, f"{word!r} got {sorted(t_set)} want {sorted(expected)}")
            continue
        if not t_set:
            fail("empty_target", i, f"{word!r} fully revealed -- nothing to predict")
            continue

        # 8. a live state has fewer than 6 wrong guesses
        if len(e_set) >= 6:
            fail("too_many_wrong", i, f"{word!r} {len(e_set)} eliminated")
            continue

        reveal_frac.append(len(revealed) / len(wl))
        elim_counts.append(len(e_set))
        blank_counts.append(sum(1 for p in range(L) if int(row[p]) == BLANK))

    checked = len(idxs)
    total_fail = sum(fails.values())
    print(f"validated {checked:,} of {n:,} states from {args.dir}\n")

    checks = ["padding", "revealed_char_mismatch", "partial_letter_reveal",
              "eliminated_letter_in_word", "eliminated_and_revealed",
              "target_mismatch", "empty_target", "too_many_wrong"]
    for c in checks:
        cnt = fails.get(c, 0)
        star = "  <-- CRITICAL" if c == "partial_letter_reveal" else ""
        print(f"  {'FAIL' if cnt else 'ok  '}  {c:<26} {cnt:>8,}{star}")
        if cnt:
            i, detail = examples[c]
            print(f"          e.g. row {i}: {detail}")

    if reveal_frac:
        print(f"\n  distinct letters revealed: mean {sum(reveal_frac)/len(reveal_frac):.3f} "
              f"of the word's letter set")
        ec = Counter(elim_counts)
        print("  wrong-guess count distribution: "
              + ", ".join(f"{k}:{ec[k]/len(elim_counts)*100:.1f}%" for k in sorted(ec)))
        print(f"  mean blanks per state: {sum(blank_counts)/len(blank_counts):.2f}")

    print()
    if total_fail:
        print(f"INVALID: {total_fail:,} failing states")
        sys.exit(1)
    print("ALL INVARIANTS HOLD")


if __name__ == "__main__":
    main()
