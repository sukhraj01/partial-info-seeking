#!/usr/bin/env python3
"""Decision-strategy experiments on a frozen model (no retraining).

  A1  Thompson sampling   sweep k, then extra seeds at the best k
  A2  cold-start curve    from the greedy run's per-ply trace
  A3  epsilon-greedy      sweep eps x seeds

Every run stores per-word outcomes (results/decision/<tag>.npz) so that all
breakdowns -- difficulty bins, paired flips versus greedy -- are computed after
the fact by analyze_decision_strategies.py, not baked into this script.

  python run_decision_strategies.py --checkpoint models/transformer_CAP57.pt
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from benchmark import load_words  # noqa: E402
from decision_strategies import (choose_greedy, make_epsilon,  # noqa: E402
                                 make_thompson, play)
from transformer_positional import PositionalSolver  # noqa: E402

K_SWEEP = [1, 2, 5, 10, 50, 100]
EPS_SWEEP = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]


def run(solver, words, tag, choose, seed, outdir, extra=None):
    path = os.path.join(outdir, f"{tag}.npz")
    if os.path.exists(path):
        d = np.load(path)
        print(f"[{tag}] cached: {d['won'].mean() * 100:.2f}%", flush=True)
        return float(d["won"].mean() * 100)
    t0 = time.time()
    r = play(solver, words, choose, seed=seed)
    np.savez_compressed(path, **r)
    wr = float(r["won"].mean() * 100)
    print(f"[{tag}] seed={seed} win={wr:.2f}% avg_wrong={r['wrong'].mean():.3f} "
          f"time={(time.time() - t0) / 60:.1f}m", flush=True)
    return wr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/transformer_CAP57.pt")
    ap.add_argument("--holdout-split", default="data/holdout_split.txt")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--outdir", default="results/decision")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--only", choices=["greedy", "A1", "A3"], nargs="*",
                    default=["greedy", "A1", "A3"])
    ap.add_argument("--best-k", type=float, default=None,
                    help="skip choosing k from the sweep (testing only)")
    args = ap.parse_args()

    solver = PositionalSolver(args.checkpoint, device=args.device,
                              batch_size=args.batch_size)
    words = load_words(args.holdout_split)
    if args.sample:
        import random
        words = random.Random(0).sample(words, min(args.sample, len(words)))
    os.makedirs(args.outdir, exist_ok=True)
    print(f"loaded {args.checkpoint} on {solver.device}; {len(words):,} words", flush=True)

    # The batched path must reproduce the benchmark's own greedy letters.
    from batch_sim import verify_agreement
    if not verify_agreement(solver, words[:100]):
        sys.exit("batched and scalar paths disagree")
    summary = {"checkpoint": args.checkpoint, "n_words": len(words),
               "host": platform.node(), "sample": args.sample}

    base = run(solver, words, "greedy", choose_greedy, 0, args.outdir)
    summary["greedy"] = base

    # Greedy through our own loop must equal the repo's reference simulator.
    from batch_sim import run_batch_benchmark
    ref = run_batch_benchmark(solver.predict_batch, words, name="reference",
                              verbose=False)["win_rate"]
    assert abs(ref - base) < 1e-9, f"greedy {base} != reference simulator {ref}"
    print(f"greedy matches reference simulator ({ref:.4f}%)", flush=True)

    if "A1" in args.only:
        sweep = {}
        for k in K_SWEEP:
            sweep[k] = run(solver, words, f"ts_k{k}_s0", make_thompson(k), 0, args.outdir)
        best_k = args.best_k or max(sweep, key=sweep.get)
        seeds = {s: run(solver, words, f"ts_k{best_k:g}_s{s}", make_thompson(best_k),
                        s, args.outdir) for s in range(args.seeds)}
        summary["A1"] = {"sweep_seed0": sweep, "best_k": best_k,
                         "best_k_seeds": seeds}

    if "A3" in args.only:
        grid = {}
        for eps in EPS_SWEEP:
            # eps=0 is deterministic: one run stands in for every seed
            n_seeds = 1 if eps == 0 else args.seeds
            grid[eps] = [run(solver, words, f"eps{eps:g}_s{s}", make_epsilon(eps),
                             s, args.outdir) for s in range(n_seeds)]
        summary["A3"] = {str(e): v for e, v in grid.items()}

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
