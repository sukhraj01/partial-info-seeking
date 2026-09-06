#!/usr/bin/env python3
"""Full n-gram benchmark. Written to run on Ada; runs anywhere (pure stdlib, CPU).

  python run_ngram_benchmark.py                 # full split: 180,240 + 45,060 words
  python run_ngram_benchmark.py --sample 5000   # quick check

Writes results/ngram_benchmark.json for rsync back to the laptop.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from benchmark import load_words, run_benchmark  # noqa: E402
from ngram_model import NgramModel  # noqa: E402


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-split", default="data/train_split.txt")
    ap.add_argument("--holdout-split", default="data/holdout_split.txt")
    ap.add_argument("--sample", type=int, default=None,
                    help="words per split (default: all)")
    ap.add_argument("--out", default="results/ngram_benchmark.json")
    ap.add_argument("--min-count", type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    model = NgramModel(args.train_split, min_count=args.min_count)
    build_s = time.time() - t0
    n_keys = sum(len(t) for t in model.tables.values())
    print(f"built {n_keys:,} context keys in {build_s:.1f}s", flush=True)

    train_words = load_words(args.train_split)
    holdout_words = load_words(args.holdout_split)
    if args.sample:
        import random
        rng = random.Random(0)
        train_words = rng.sample(train_words, min(args.sample, len(train_words)))
        holdout_words = rng.sample(holdout_words, min(args.sample, len(holdout_words)))

    print(f"in-vocab: {len(train_words):,} words | disjoint: {len(holdout_words):,} words",
          flush=True)

    in_vocab = run_benchmark(model.predict, train_words, name="ngram [in-vocab]")
    disjoint = run_benchmark(model.predict, holdout_words, name="ngram [disjoint]")

    total_min = (in_vocab["elapsed_s"] + disjoint["elapsed_s"]) / 60
    print(f"\nngram_backoff | in-vocab: {in_vocab['win_rate']:5.1f}% "
          f"| disjoint: {disjoint['win_rate']:5.1f}% "
          f"| avg_wrong: {disjoint['avg_wrong_won']:.2f} | time: {total_min:.1f}m")

    payload = {
        "model": "ngram_backoff",
        "in_vocab": in_vocab,
        "disjoint": disjoint,
        "n_context_keys": n_keys,
        "build_seconds": build_s,
        "min_count": args.min_count,
        "sample": args.sample,
        "host": platform.node(),
        "python": platform.python_version(),
        "git_rev": git_rev(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
