#!/usr/bin/env python3
"""Build an augmented training vocabulary from external word lists, safely.

The entire evaluation apparatus rests on `holdout_split.txt` being words the
model has NEVER seen. A general English word list contains ~92% of that holdout,
so adding one carelessly would silently destroy every number we have.

This script therefore treats leakage as the primary concern, not an afterthought:

  * every external word is filtered to a-z and normalised
  * every word appearing in ANY --forbid file is removed
  * the result is ASSERTED disjoint from each forbid list before it is written
  * counts are reported at each stage so the filtering is auditable

The assertions are the point. Set arithmetic that looks obviously correct is
exactly the kind of thing that is quietly wrong, and a leak here would not
announce itself -- it would just make every subsequent result look good.

  python build_augmented_vocab.py \\
      --base data/train_split.txt \\
      --external /tmp/linux.words \\
      --forbid data/holdout_split.txt \\
      --out data/train_augmented.txt
"""
import argparse
import os
import random
import re
import sys
from collections import Counter

WORD_RE = re.compile(r"^[a-z]+$")
MAX_LEN = 29


def load_raw(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f]


def normalise(words, max_len=MAX_LEN):
    """Lowercase, keep pure a-z, drop anything longer than the model can encode."""
    out, dropped = set(), Counter()
    for w in words:
        w = w.strip().lower()
        if not w:
            dropped["empty"] += 1
        elif not WORD_RE.match(w):
            dropped["non_az"] += 1
        elif len(w) > max_len:
            dropped["too_long"] += 1
        else:
            out.add(w)
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/train_split.txt",
                    help="words we already train on; always kept")
    ap.add_argument("--external", nargs="+", required=True,
                    help="external word lists to merge in")
    ap.add_argument("--forbid", nargs="+", required=True,
                    help="word lists that must NEVER appear in the output "
                         "(the holdout, and anything else used for evaluation)")
    ap.add_argument("--out", default="data/train_augmented.txt")
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--seed", type=int, default=42,
                    help="shuffle seed; the output order determines which words "
                         "land in the trainer's held-out validation slice")
    args = ap.parse_args()

    base, base_drop = normalise(load_raw(args.base), args.max_len)
    print(f"base            {args.base:<34} {len(base):>8,}")
    if base_drop:
        print(f"  (dropped from base: {dict(base_drop)})")

    forbid = set()
    forbid_sets = {}
    for path in args.forbid:
        s, _ = normalise(load_raw(path), args.max_len)
        forbid_sets[path] = s
        forbid |= s
        print(f"FORBIDDEN       {path:<34} {len(s):>8,}")
    print(f"forbidden total {'':<34} {len(forbid):>8,}")

    # The base itself must already be clean, or the split was built wrong.
    overlap = base & forbid
    if overlap:
        sys.exit(f"FATAL: base vocabulary already overlaps a forbidden list on "
                 f"{len(overlap):,} words (e.g. {sorted(overlap)[:5]}). "
                 f"The train/holdout split is broken; fix that before augmenting.")
    print("check: base is disjoint from every forbidden list  OK\n")

    merged = set(base)
    for path in args.external:
        raw = load_raw(path)
        ext, drop = normalise(raw, args.max_len)
        new = ext - merged - forbid
        blocked = len(ext & forbid)
        print(f"external        {path:<34} {len(raw):>8,} raw")
        print(f"  after a-z/length filter{'':<20} {len(ext):>8,}   dropped {dict(drop)}")
        print(f"  BLOCKED as forbidden{'':<23} {blocked:>8,}   <-- would have leaked")
        print(f"  already known{'':<30} {len(ext & merged):>8,}")
        print(f"  genuinely new{'':<30} {len(new):>8,}")
        merged |= new

    # Belt and braces: re-derive the intersection on the FINAL set.
    for path, s in forbid_sets.items():
        bad = merged & s
        assert not bad, (f"LEAK: {len(bad)} forbidden words from {path} reached the "
                         f"output, e.g. {sorted(bad)[:5]}")
    print("\ncheck: final vocabulary is disjoint from every forbidden list  OK")

    # SHUFFLE, do not sort. The training pipeline holds out the last 2% of this
    # file as its validation slice, so alphabetical order would make that slice
    # all w/x/y/z -- and, far worse, would mean the model never trains on a
    # single word starting with x, y or z. `split_data.py` shuffles for exactly
    # this reason; emitting sorted output silently broke that invariant.
    words = sorted(merged)
    random.Random(args.seed).shuffle(words)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(words) + "\n")

    # Guard: the trainer holds out the tail of this file. Show what that tail
    # looks like, so a degenerate slice is visible rather than inferred later
    # from a mysteriously terrible validation score.
    tail = words[int(len(words) * 0.98):]
    head_letters = Counter(w[0] for w in tail)
    distinct = len(head_letters)
    print(f"\nvalidation slice preview (last 2% = {len(tail):,} words): "
          f"{distinct} distinct initial letters")
    print(f"  most common: {head_letters.most_common(6)}")
    if distinct < 15:
        print("  WARNING: the held-out slice is not alphabetically representative")

    lens = Counter(len(w) for w in words)
    base_lens = Counter(len(w) for w in base)
    print(f"\nwrote {args.out}: {len(words):,} words "
          f"({len(words) - len(base):+,} vs base, "
          f"{(len(words)/len(base) - 1) * 100:+.1f}%)")
    print(f"mean length: base {sum(len(w) for w in base)/len(base):.2f} "
          f"-> augmented {sum(len(w) for w in words)/len(words):.2f}")
    print("\nlength distribution shift (base % -> augmented %):")
    for k in range(2, 20):
        b = base_lens.get(k, 0) / len(base) * 100
        a = lens.get(k, 0) / len(words) * 100
        flag = "  <-- shifted" if abs(a - b) > 1.5 else ""
        print(f"  {k:>2}: {b:5.1f}% -> {a:5.1f}%{flag}")


if __name__ == "__main__":
    main()
