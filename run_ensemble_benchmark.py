#!/usr/bin/env python3
"""Sweep transformer + n-gram ensemble variants on the disjoint split.

The n-gram tables cost ~1 min and ~1 GB to build, and the transformer checkpoint
costs a GPU load, so both are built ONCE and reused across every variant. Pure
transformer and pure n-gram are re-run here as controls rather than quoted from
RESULTS.md, so every number in the comparison table comes off the same code path
on the same machine.

  python run_ensemble_benchmark.py --checkpoint models/transformer.pt
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
from ensemble_model import EnsembleSolver  # noqa: E402
from ngram_model import NgramModel  # noqa: E402


def frange(lo, hi, step):
    out, x = [], lo
    while x <= hi + 1e-9:
        out.append(round(x, 4))
        x += step
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/transformer.pt")
    ap.add_argument("--dictionary", default="data/train_split.txt")
    ap.add_argument("--holdout-split", default="data/holdout_split.txt")
    ap.add_argument("--train-split", default="data/train_split.txt")
    ap.add_argument("--sample", type=int, default=None,
                    help="words from the disjoint split (default: all 45,060)")
    ap.add_argument("--in-vocab", action="store_true",
                    help="also score the best variant in-vocab, for the gap check")
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--modes", default="blend,nblend,lblend,route")
    ap.add_argument("--out", default="results/ensemble_benchmark.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    words = load_words(args.holdout_split)
    if args.sample:
        import random
        words = random.Random(0).sample(words, min(args.sample, len(words)))
    print(f"disjoint words: {len(words):,}", flush=True)

    t0 = time.time()
    ngram = NgramModel(args.dictionary)
    print(f"n-gram tables built in {time.time()-t0:.0f}s "
          f"({sum(len(t) for t in ngram.tables.values()):,} keys)", flush=True)

    modes = args.modes.split(",")
    variants = []
    # Controls first: an ensemble that cannot beat its own best member is dead.
    variants.append(("transformer_only", "blend", 1.0, None))
    variants.append(("ngram_only", "blend", 0.0, None))
    if "blend" in modes:
        for w in frange(0.60, 0.95, 0.05):
            variants.append((f"blend_w{w:.2f}", "blend", w, None))
    if "nblend" in modes:
        for w in frange(0.60, 0.95, 0.05):
            variants.append((f"nblend_w{w:.2f}", "nblend", w, None))
    if "lblend" in modes:
        for w in frange(0.60, 0.95, 0.05):
            variants.append((f"lblend_w{w:.2f}", "lblend", w, None))
    if "route" in modes:
        for t in frange(0.30, 0.70, 0.05):
            variants.append((f"route_t{t:.2f}", "route", None, t))

    print(f"{len(variants)} variants to benchmark\n", flush=True)

    solver = EnsembleSolver(args.checkpoint, mode="blend", weight=1.0,
                            device=args.device, batch_size=args.batch_size,
                            ngram=ngram)
    print(f"transformer on {solver.device} "
          f"| config {solver.meta.get('config')}", flush=True)

    # The batched path is what produces every number below; check it against the
    # scalar path once, for one blend and one route variant (different code
    # branches), before trusting any of them.
    for m, kw in (("blend", dict(weight=0.8)), ("route", dict(threshold=0.5))):
        solver.mode = m
        solver.weight = kw.get("weight", 0.8)
        solver.threshold = kw.get("threshold", 0.5)
        print(f"agreement check [{m}]:", flush=True)
        if not verify_agreement(solver, words[:60]):
            print("!! batched and scalar paths disagree -- aborting")
            sys.exit(1)

    rows = []
    best = None
    for i, (name, mode, w, thr) in enumerate(variants, 1):
        solver.mode = mode
        if w is not None:
            solver.weight = w
        if thr is not None:
            solver.threshold = thr
        solver.reset_stats()
        r = run_batch_benchmark(solver.predict_batch, words, name=name,
                                chunk_size=args.chunk_size, verbose=False)
        r.update(mode=mode, weight=w, threshold=thr, **solver.stats())
        rows.append(r)
        print(f"[{i:2d}/{len(variants)}] {name:18s} | disjoint {r['win_rate']:6.2f}% "
              f"| avg_wrong {r['avg_wrong_won']:.3f} "
              f"| changed {r['changed_vs_transformer_frac']*100:5.1f}% "
              f"| routed {r['routed_frac']*100:5.1f}% "
              f"| {r['elapsed_s']:.0f}s", flush=True)
        if best is None or r["win_rate"] > best["win_rate"]:
            best = r

    print(f"\nBEST: {best['name']} -> {best['win_rate']:.2f}% disjoint", flush=True)

    in_vocab = None
    if args.in_vocab and best["mode"] is not None:
        # The gap is the memorisation diagnostic; only worth paying for on the winner.
        solver.mode = best["mode"]
        if best["weight"] is not None:
            solver.weight = best["weight"]
        if best["threshold"] is not None:
            solver.threshold = best["threshold"]
        solver.reset_stats()
        tw = load_words(args.train_split)
        in_vocab = run_batch_benchmark(solver.predict_batch, tw,
                                       name=f"{best['name']} [in-vocab]",
                                       chunk_size=args.chunk_size, verbose=True)
        print(f"gap: {in_vocab['win_rate'] - best['win_rate']:.2f}pp", flush=True)

    payload = {"checkpoint": args.checkpoint, "n_words": len(words),
               "variants": rows, "best": best, "in_vocab_best": in_vocab,
               "host": platform.node(),
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
