#!/usr/bin/env python3
"""Does a bigger dictionary make the n-gram better on UNSEEN words?

We currently build the n-gram from `train_split.txt` (180,240 words = 80% of
train.txt), holding out 45,060 for evaluation. Rebuilding it from the full
225,300 would give it 25% more vocabulary -- but we could not measure that
directly, because the holdout would then be inside its dictionary.

So measure the SLOPE instead: build from 50%/70%/85%/100% of train_split and
score each on the same untouched holdout. Every point is clean, and the trend
says whether +25% more vocabulary is worth anything.

  python run_ngram_scaling.py --sample 8000
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from batch_sim import run_batch_benchmark  # noqa: E402
from benchmark import load_words  # noqa: E402
from ngram_model import NgramModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-split", default="data/train_split.txt")
    ap.add_argument("--holdout-split", default="data/holdout_split.txt")
    ap.add_argument("--fractions", default="0.5,0.7,0.85,1.0")
    ap.add_argument("--sample", type=int, default=0,
                    help="holdout words to score (0 = full split)")
    ap.add_argument("--out", default="results/ngram_scaling.json")
    args = ap.parse_args()

    train = load_words(args.train_split)
    holdout = load_words(args.holdout_split)
    if args.sample:
        holdout = random.Random(0).sample(holdout, min(args.sample, len(holdout)))
    print(f"train pool {len(train):,} | scoring on {len(holdout):,} holdout words\n",
          flush=True)

    rng = random.Random(42)
    shuffled = train[:]
    rng.shuffle(shuffled)          # nested subsets, so the curve is monotone in data

    rows = []
    for frac in [float(f) for f in args.fractions.split(",")]:
        n = int(len(train) * frac)
        subset = sorted(shuffled[:n])
        tmp = f"/tmp/_ngram_dict_{n}.txt"
        with open(tmp, "w") as f:
            f.write("\n".join(subset) + "\n")

        t0 = time.time()
        model = NgramModel(tmp)
        keys = sum(len(t) for t in model.tables.values())
        build_s = time.time() - t0

        # run_batch_benchmark wants a BATCH predictor; NgramModel.predict is scalar.
        def pb(pats, gs, _m=model):
            return [_m.predict(p_, g_) for p_, g_ in zip(pats, gs)]

        r = run_batch_benchmark(pb, holdout, name=f"ngram@{n:,}",
                                chunk_size=4096, verbose=False)
        rows.append({"n_words": n, "fraction": frac, "context_keys": keys,
                     "win_rate": r["win_rate"], "avg_wrong_won": r["avg_wrong_won"],
                     "build_s": build_s})
        print(f"dict {n:>7,} words ({frac*100:5.1f}%) | {keys:>8,} keys | "
              f"holdout {r['win_rate']:6.2f}% | avg_wrong {r['avg_wrong_won']:.3f}",
              flush=True)
        os.remove(tmp)

    # Extrapolate: n-gram quality scales roughly with log(corpus size).
    if len(rows) >= 2:
        import math
        (x0, y0), (x1, y1) = ((math.log(rows[0]["n_words"]), rows[0]["win_rate"]),
                              (math.log(rows[-1]["n_words"]), rows[-1]["win_rate"]))
        slope = (y1 - y0) / (x1 - x0)
        full = 225300
        pred = rows[-1]["win_rate"] + slope * (math.log(full) - math.log(rows[-1]["n_words"]))
        print(f"\nlog-linear slope: {slope:.3f} pp per e-fold of vocabulary")
        print(f"extrapolated to the full {full:,}-word train.txt: "
              f"{pred:.2f}% ({pred - rows[-1]['win_rate']:+.2f}pp vs the 180,240 dict)")
        print(f"at 0.30 blend weight that is roughly "
              f"{(pred - rows[-1]['win_rate']) * 0.30:+.2f}pp on the final model")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "n_holdout": len(holdout)}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
