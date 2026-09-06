#!/usr/bin/env python3
"""Benchmark expectimax against greedy argmax on the SAME checkpoint.

Running both here, back to back, on the same word list is deliberate: the lift
we want to report is the decision rule's, isolated from whatever the DAgger
rounds contributed.

  python run_expectimax_benchmark.py --checkpoint models/transformer_r1.pt \\
      --value-fn models/value_fn.cbm
"""
import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from batch_sim import run_batch_benchmark  # noqa: E402
from benchmark import load_words  # noqa: E402
from expectimax import ExpectimaxSolver  # noqa: E402
from transformer_model import TransformerSolver  # noqa: E402
from value_function import ValueFunction, letter_multiplicity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/transformer.pt")
    ap.add_argument("--value-fn", default="models/value_fn.cbm")
    ap.add_argument("--train-split", default="data/train_split.txt")
    ap.add_argument("--holdout-split", default="data/holdout_split.txt")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--out", default="results/expectimax_benchmark.json")
    ap.add_argument("--skip-greedy", action="store_true")
    args = ap.parse_args()

    base = TransformerSolver(args.checkpoint, batch_size=args.batch_size)
    vf = ValueFunction(args.value_fn)
    mult = letter_multiplicity(load_words(args.train_split))
    solver = ExpectimaxSolver(base, vf, mult, top_k=args.top_k,
                              batch_size=args.batch_size)
    print(f"checkpoint={args.checkpoint} value_fn={args.value_fn} top_k={args.top_k}",
          flush=True)
    print("letter multiplicity: "
          + ", ".join(f"{c}={m:.2f}" for c, m in
                      zip("abcdefghijklmnopqrstuvwxyz", mult)), flush=True)

    words = load_words(args.holdout_split)
    if args.sample:
        import random
        words = random.Random(0).sample(words, min(args.sample, len(words)))

    payload = {"checkpoint": args.checkpoint, "value_fn": args.value_fn,
               "top_k": args.top_k, "n_words": len(words),
               "host": platform.node(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    if not args.skip_greedy:
        print("\n-- greedy argmax (same checkpoint) --", flush=True)
        g = run_batch_benchmark(base.predict_batch, words, name="greedy [disjoint]",
                                chunk_size=args.chunk_size, progress_every=20000)
        payload["greedy"] = g

    print("\n-- depth-1 expectimax --", flush=True)
    e = run_batch_benchmark(solver.predict_batch, words, name="expectimax [disjoint]",
                            chunk_size=args.chunk_size, progress_every=20000)
    payload["expectimax"] = e
    payload["decisions"] = solver.stats["decisions"]
    payload["changed"] = solver.stats["changed"]
    changed_pct = solver.stats["changed"] / max(solver.stats["decisions"], 1) * 100
    payload["changed_pct"] = changed_pct

    print(f"\nexpectimax changed the guess on {changed_pct:.2f}% of decisions "
          f"({solver.stats['changed']:,}/{solver.stats['decisions']:,})")
    if "greedy" in payload:
        d = e["win_rate"] - payload["greedy"]["win_rate"]
        dw = e["avg_wrong_won"] - payload["greedy"]["avg_wrong_won"]
        payload["delta_win_rate_pp"] = d
        payload["delta_avg_wrong"] = dw
        print(f"greedy     : {payload['greedy']['win_rate']:.2f}% "
              f"| avg_wrong {payload['greedy']['avg_wrong_won']:.3f}")
        print(f"expectimax : {e['win_rate']:.2f}% | avg_wrong {e['avg_wrong_won']:.3f}")
        print(f"DELTA      : {d:+.2f}pp win rate | {dw:+.3f} avg_wrong")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
