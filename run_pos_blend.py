#!/usr/bin/env python3
"""Sweep the per-position transformer blended with the n-gram, full disjoint split.

The n-gram lifted the pooled transformer by ~+0.5pp at w=0.70 (67.00 -> 67.46 on
round-0, 68.52 -> 69.01 on C1). P1 is a stronger base AND now uses the same
per-blank aggregation the n-gram uses, so the two are structurally closer than
before -- which cuts both ways: the blend may gain less because the members now
make more similar errors, or more because both are reasoning positionally.
Worth measuring rather than assuming.

Controls (pure P1, pure n-gram) are re-run here rather than quoted, so every
number comes off the same code path on the same machine.

  python run_pos_blend.py --checkpoint models/transformer_P1.pt
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
from multi_ensemble import MultiEnsemble  # noqa: E402
from ngram_model import NgramModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/transformer_P1.pt")
    ap.add_argument("--dictionary", default="data/train_split.txt")
    ap.add_argument("--holdout-split", default="data/holdout_split.txt")
    ap.add_argument("--weights", default="1.00,0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.00",
                    help="transformer weight; the n-gram takes the remainder")
    ap.add_argument("--log-space", action="store_true",
                    help="pool log-odds (product of experts) instead of probabilities")
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--out", default="results/pos_blend.json")
    args = ap.parse_args()

    words = load_words(args.holdout_split)
    print(f"FULL disjoint split: {len(words):,} words\n", flush=True)

    t0 = time.time()
    ngram = NgramModel(args.dictionary)
    print(f"n-gram tables built in {time.time()-t0:.0f}s", flush=True)

    ens = MultiEnsemble([("pos", args.checkpoint, 1.0), ("ngram", None, 0.0)],
                        dictionary=args.dictionary, ngram=ngram,
                        log_space=args.log_space)
    print(f"members: {[m.name for m in ens.members]}"
          f"{' [log-odds pooling]' if args.log_space else ''}\n", flush=True)

    rows, best = [], None
    for w in [float(x) for x in args.weights.split(",")]:
        ens.set_weights([w, 1.0 - w])
        name = ("P1_only" if w == 1.0 else "ngram_only" if w == 0.0
                else f"blend_w{w:.2f}")
        r = run_batch_benchmark(ens.predict_batch, words, name=name,
                                chunk_size=args.chunk_size, verbose=False)
        r["w_transformer"] = w
        rows.append(r)
        print(f"{name:14s} disjoint {r['win_rate']:6.2f}%  "
              f"avg_wrong {r['avg_wrong_won']:.3f}  {r['elapsed_s']:.0f}s", flush=True)
        if best is None or r["win_rate"] > best["win_rate"]:
            best = r

    solo = next((r["win_rate"] for r in rows if r["w_transformer"] == 1.0), None)
    print(f"\nBEST: {best['name']} -> {best['win_rate']:.2f}%", flush=True)
    if solo is not None:
        print(f"vs the transformer alone ({solo:.2f}%): "
              f"{best['win_rate'] - solo:+.2f}pp", flush=True)

    payload = {"checkpoint": args.checkpoint, "n_words": len(words),
               "log_space": args.log_space, "variants": rows, "best": best,
               "host": platform.node(),
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
