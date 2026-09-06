#!/usr/bin/env python3
"""Benchmark the 26 per-letter CatBoost models by playing real hangman games.

  python run_catboost_benchmark.py --model-dir models/catboost
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from batch_sim import run_batch_benchmark, verify_agreement  # noqa: E402
from benchmark import load_words  # noqa: E402
from catboost_model import CatBoostSolver  # noqa: E402
from ngram_model import NgramModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/catboost")
    ap.add_argument("--train-split", default="data/train_split.txt")
    ap.add_argument("--holdout-split", default="data/holdout_split.txt")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--out", default="results/catboost_benchmark.json")
    ap.add_argument("--skip-agreement", action="store_true")
    ap.add_argument("--agreement-words", type=int, default=50)
    args = ap.parse_args()

    t0 = time.time()
    ngram = NgramModel(args.train_split)
    corpus_freq = np.load(os.path.join(args.model_dir, "corpus_freq.npy"))
    solver = CatBoostSolver(args.model_dir, ngram, corpus_freq)
    print(f"loaded 26 models + n-gram in {time.time()-t0:.0f}s", flush=True)

    train_words = load_words(args.train_split)
    holdout_words = load_words(args.holdout_split)
    if args.sample:
        import random
        rng = random.Random(0)
        train_words = rng.sample(train_words, min(args.sample, len(train_words)))
        holdout_words = rng.sample(holdout_words, min(args.sample, len(holdout_words)))

    if not args.skip_agreement:
        print(f"\nverifying batched vs scalar on {args.agreement_words} words...", flush=True)
        if not verify_agreement(solver, holdout_words[:args.agreement_words]):
            print("!! batched and scalar paths disagree -- benchmark aborted")
            sys.exit(1)

    print("\nplaying games...", flush=True)
    tb = time.time()
    in_vocab = run_batch_benchmark(solver.predict_batch, train_words,
                                   name="catboost [in-vocab]",
                                   chunk_size=args.chunk_size, progress_every=20000)
    disjoint = run_batch_benchmark(solver.predict_batch, holdout_words,
                                   name="catboost [disjoint]",
                                   chunk_size=args.chunk_size, progress_every=20000)
    total_min = (time.time() - tb) / 60

    gap = in_vocab["win_rate"] - disjoint["win_rate"]
    print(f"\ncatboost | in-vocab: {in_vocab['win_rate']:5.1f}% "
          f"| disjoint: {disjoint['win_rate']:5.1f}% "
          f"| avg_wrong: {disjoint['avg_wrong_won']:.2f} | time: {total_min:.1f}m")
    print(f"in-vocab/disjoint gap: {gap:.1f}pp")

    payload = {"model": "catboost", "model_dir": args.model_dir,
               "in_vocab": in_vocab, "disjoint": disjoint, "gap_pp": gap,
               "sample": args.sample, "host": platform.node(),
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
