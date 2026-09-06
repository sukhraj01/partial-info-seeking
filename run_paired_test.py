#!/usr/bin/env python3
"""Paired significance test for ensemble variants against the transformer alone.

Headline effects here are fractions of a point on 45,060 words. The
unpaired standard error there is ~0.22pp, which is the same size as the effect,
so a bare win-rate comparison cannot tell a real gain from noise. The comparison
is paired -- identical words, identical rules, only the policy differs -- so the
right test is McNemar's on the discordant pairs.

  python run_paired_test.py --variants blend:0.70 nblend:0.75 route:0.45
"""
import argparse
import json
import os
import platform
import sys
import time
from math import comb, erfc, sqrt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from batch_sim import run_batch_benchmark  # noqa: E402
from benchmark import load_words  # noqa: E402
from ensemble_model import EnsembleSolver  # noqa: E402
from ngram_model import NgramModel  # noqa: E402


def mcnemar_exact(b, c):
    """Two-sided McNemar p-value. b = A-only wins, c = B-only wins.

    The exact binomial form needs 2**n, which overflows a float past n ~ 1024,
    and real discordant counts here run into the thousands. So: exact for small
    n, continuity-corrected normal approximation above it (they agree closely
    well before the switchover).
    """
    n = b + c
    if n == 0:
        return 1.0
    if n <= 900:
        k = min(b, c)
        return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n))
    z = (abs(b - c) - 1.0) / sqrt(n)
    if z <= 0:
        return 1.0
    return erfc(z / sqrt(2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/transformer.pt")
    ap.add_argument("--dictionary", default="data/train_split.txt")
    ap.add_argument("--holdout-split", default="data/holdout_split.txt")
    ap.add_argument("--variants", nargs="+", required=True,
                    help="mode:value, e.g. blend:0.70 route:0.45")
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--out", default="results/ensemble_paired.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    words = load_words(args.holdout_split)
    ngram = NgramModel(args.dictionary)
    solver = EnsembleSolver(args.checkpoint, mode="blend", weight=1.0,
                            device=args.device, ngram=ngram)
    print(f"{len(words):,} disjoint words | {solver.device}", flush=True)

    base = run_batch_benchmark(solver.predict_batch, words, name="transformer_only",
                               chunk_size=args.chunk_size, verbose=False, per_word=True)
    bflags = base["won_flags"]
    print(f"baseline transformer_only: {base['win_rate']:.2f}%\n", flush=True)

    rows = []
    for spec in args.variants:
        mode, val = spec.split(":")
        solver.mode = mode
        if mode == "route":
            solver.threshold = float(val)
        else:
            solver.weight = float(val)
        solver.reset_stats()
        r = run_batch_benchmark(solver.predict_batch, words, name=spec,
                               chunk_size=args.chunk_size, verbose=False, per_word=True)
        vflags = r["won_flags"]
        b = sum(1 for x, y in zip(bflags, vflags) if x and not y)
        c = sum(1 for x, y in zip(bflags, vflags) if y and not x)
        p = mcnemar_exact(b, c)
        delta = r["win_rate"] - base["win_rate"]
        rows.append({"variant": spec, "win_rate": r["win_rate"], "delta_pp": delta,
                     "baseline_only_wins": b, "variant_only_wins": c,
                     "discordant": b + c, "p_value": p,
                     "avg_wrong_won": r["avg_wrong_won"], **solver.stats()})
        print(f"{spec:14s} {r['win_rate']:6.2f}% ({delta:+.2f}pp) "
              f"| discordant {b + c:5d} (base-only {b}, variant-only {c}) "
              f"| McNemar p = {p:.3g} "
              f"{'SIGNIFICANT' if p < 0.05 else 'not significant'}", flush=True)

    payload = {"baseline": {k: v for k, v in base.items() if k != "won_flags"},
               "variants": rows, "n_words": len(words), "host": platform.node(),
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
