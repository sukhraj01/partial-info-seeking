#!/usr/bin/env python3
"""Benchmark a trained transformer by playing real hangman games.

Validation loss is not the objective -- win rate is. This plays the full game
loop, and by default cross-checks that the batched fast path picks exactly the
same letters as the scalar predict() the rest of the codebase uses.

  python run_transformer_benchmark.py --checkpoint models/transformer.pt
"""
import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from batch_sim import run_batch_benchmark, verify_agreement  # noqa: E402
from benchmark import load_words  # noqa: E402
from transformer_model import TransformerSolver  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/transformer.pt")
    ap.add_argument("--train-split", default="data/train_split.txt")
    ap.add_argument("--holdout-split", default="data/holdout_split.txt")
    ap.add_argument("--sample", type=int, default=None, help="words per split (default all)")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--out", default="results/transformer_benchmark.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--positional", action="store_true",
                    help="checkpoint uses the per-position (MLM) head")
    ap.add_argument("--pretrained", action="store_true",
                    help="checkpoint is a fine-tuned pretrained char model "
                         "(train_pretrained.py), not our from-scratch transformer")
    ap.add_argument("--skip-agreement", action="store_true")
    ap.add_argument("--agreement-words", type=int, default=100)
    args = ap.parse_args()

    if args.positional:
        from transformer_positional import PositionalSolver
        solver = PositionalSolver(args.checkpoint, device=args.device,
                                  batch_size=args.batch_size)
    elif args.pretrained:
        from train_pretrained import PretrainedSolver
        solver = PretrainedSolver(args.checkpoint, device=args.device,
                                  batch_size=min(args.batch_size, 1024))
    else:
        solver = TransformerSolver(args.checkpoint, device=args.device,
                                   batch_size=args.batch_size)
    print(f"loaded {args.checkpoint} on {solver.device}", flush=True)
    for k in ("epoch", "step", "val_loss", "val_top1_hit"):
        if k in solver.meta:
            print(f"  {k}: {solver.meta[k]}")

    train_words = load_words(args.train_split)
    holdout_words = load_words(args.holdout_split)
    if args.sample:
        import random
        rng = random.Random(0)
        train_words = rng.sample(train_words, min(args.sample, len(train_words)))
        holdout_words = rng.sample(holdout_words, min(args.sample, len(holdout_words)))

    if not args.skip_agreement:
        print(f"\nverifying batched path against scalar predict() "
              f"on {args.agreement_words} words...", flush=True)
        if not verify_agreement(solver, holdout_words[:args.agreement_words]):
            print("!! batched and scalar paths disagree -- benchmark aborted")
            sys.exit(1)

    print("\nplaying games...", flush=True)
    t0 = time.time()
    in_vocab = run_batch_benchmark(solver.predict_batch, train_words,
                                   name="transformer [in-vocab]",
                                   chunk_size=args.chunk_size,
                                   progress_every=50000)
    disjoint = run_batch_benchmark(solver.predict_batch, holdout_words,
                                   name="transformer [disjoint]",
                                   chunk_size=args.chunk_size,
                                   progress_every=50000)
    total_min = (time.time() - t0) / 60

    gap = in_vocab["win_rate"] - disjoint["win_rate"]
    print(f"\ntransformer | in-vocab: {in_vocab['win_rate']:5.1f}% "
          f"| disjoint: {disjoint['win_rate']:5.1f}% "
          f"| avg_wrong: {disjoint['avg_wrong_won']:.2f} | time: {total_min:.1f}m")
    print(f"in-vocab/disjoint gap: {gap:.1f}pp "
          f"({'generalising' if gap < 10 else 'MEMORISING -- investigate'})")

    payload = {"model": "transformer", "checkpoint": args.checkpoint,
               "in_vocab": in_vocab, "disjoint": disjoint, "gap_pp": gap,
               "checkpoint_meta": {k: v for k, v in solver.meta.items()
                                   if k not in ("args", "states_meta", "history")},
               "sample": args.sample, "host": platform.node(),
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
