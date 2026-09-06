#!/usr/bin/env python3
"""Build NESTED vocabulary subsets for the scaling experiment.

To measure how disjoint win rate scales with vocabulary size, size must be the
only thing that varies. Two traps this avoids:

1. **Distribution drift.** Using `train_split.txt` for the 180K point and the
   augmented list for the 390K point would confound count with composition --
   the external words skew more obscure. Instead every subset is drawn from the
   SAME augmented pool, so all points share one distribution.

2. **Non-nested samples.** Independent random draws at each size add sampling
   noise between points. Prefixes of one shuffled list are nested by
   construction: the 50K set is inside the 100K set, and so on, so the curve
   reflects added words rather than resampling.

Every subset inherits the augmented pool's holdout-disjointness, and that is
asserted per subset rather than assumed.

  python build_vocab_subsets.py --vocab data/train_augmented.txt \\
      --sizes 50000 100000 180000 250000 --forbid data/holdout_split.txt
"""
import argparse
import os
import random


def load(path):
    return [w.strip() for w in open(path) if w.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="data/train_augmented.txt")
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[50000, 100000, 180000, 250000])
    ap.add_argument("--forbid", nargs="+", default=["data/holdout_split.txt"])
    ap.add_argument("--out-dir", default="data/vocab_subsets")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    words = load(args.vocab)
    forbid = set()
    for f in args.forbid:
        forbid |= set(load(f))
    bad = set(words) & forbid
    assert not bad, (f"source vocabulary already leaks {len(bad)} forbidden words, "
                     f"e.g. {sorted(bad)[:5]}")
    print(f"source vocabulary : {len(words):,} words  (disjoint from "
          f"{len(forbid):,} forbidden)  OK")

    # One shuffle, then prefixes => nested subsets.
    random.Random(args.seed).shuffle(words)
    os.makedirs(args.out_dir, exist_ok=True)

    prev = None
    for n in sorted(args.sizes):
        if n > len(words):
            print(f"  skip {n:,}: only {len(words):,} available")
            continue
        subset = words[:n]
        s = set(subset)
        leak = s & forbid
        assert not leak, f"LEAK at size {n}: {sorted(leak)[:5]}"
        if prev is not None:
            assert prev <= s, f"size {n} is not a superset of the previous subset"
        prev = s

        path = os.path.join(args.out_dir, f"vocab_{n // 1000}k.txt")
        with open(path, "w") as f:
            f.write("\n".join(subset) + "\n")
        mean_len = sum(len(w) for w in subset) / len(subset)
        print(f"  {path:<40} {n:>7,} words  mean_len {mean_len:.2f}  "
              f"holdout-clean OK  nested OK")

    print("\nAll subsets are nested, holdout-disjoint, and share one word "
          "distribution -- so vocabulary SIZE is the only variable.")


if __name__ == "__main__":
    main()
