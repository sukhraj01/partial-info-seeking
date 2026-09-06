#!/usr/bin/env python3
"""Sweep transformer + CANINE + n-gram blends on the full disjoint split.

Runs the moment a CANINE checkpoint exists. The interesting hypothesis is not
"is CANINE better" -- it very likely is not -- but "does a model with a
different architecture and different pretraining make DIFFERENT mistakes, and
therefore lift the blend". Every pair in our comparison table sits at phi >=
0.68 because they are all the same from-scratch family; CANINE is the first
genuinely independent member we have.

  python run_multi_ensemble.py --transformer models/transformer_C1.pt \
      --pretrained models/canine.pt
"""
import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from batch_sim import run_batch_benchmark  # noqa: E402
from benchmark import load_words  # noqa: E402
from multi_ensemble import MultiEnsemble  # noqa: E402
from ngram_model import NgramModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformer", default="models/transformer_C1.pt")
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--dictionary", default="data/train_split.txt")
    ap.add_argument("--holdout-split", default="data/holdout_split.txt")
    ap.add_argument("--sample", type=int, default=None,
                    help="words to score; default is the FULL split")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--out", default="results/multi_ensemble.json")
    args = ap.parse_args()

    words = load_words(args.holdout_split)
    if args.sample:
        import random
        words = random.Random(0).sample(words, min(args.sample, len(words)))
    print(f"disjoint words: {len(words):,}", flush=True)

    t0 = time.time()
    ngram = NgramModel(args.dictionary)
    print(f"n-gram tables built in {time.time()-t0:.0f}s", flush=True)

    # (transformer, canine, ngram) weights. The incumbent is (0.70, 0, 0.30) at
    # 69.01%; everything else asks what a slice of CANINE buys.
    grid = [
        ("incumbent  T.70/C.00/N.30", 0.70, 0.00, 0.30),
        ("T.60/C.10/N.30", 0.60, 0.10, 0.30),
        ("T.55/C.15/N.30", 0.55, 0.15, 0.30),
        ("T.50/C.20/N.30", 0.50, 0.20, 0.30),
        ("T.45/C.25/N.30", 0.45, 0.25, 0.30),
        ("T.40/C.30/N.30", 0.40, 0.30, 0.30),
        ("T.60/C.20/N.20", 0.60, 0.20, 0.20),
        ("T.50/C.30/N.20", 0.50, 0.30, 0.20),
        ("T.70/C.30/N.00", 0.70, 0.30, 0.00),
        ("T.60/C.40/N.00", 0.60, 0.40, 0.00),
        ("T.50/C.50/N.00", 0.50, 0.50, 0.00),
        ("canine alone", 0.00, 1.00, 0.00),
    ]

    ens = MultiEnsemble(
        [("tfm", args.transformer, 0.7), ("pre", args.pretrained, 0.0),
         ("ngram", None, 0.3)],
        dictionary=args.dictionary, ngram=ngram)
    print(f"members: {[m.name for m in ens.members]}\n", flush=True)

    rows, best = [], None
    for name, wt, wc, wn in grid:
        ens.set_weights([wt, wc, wn])
        r = run_batch_benchmark(ens.predict_batch, words, name=name,
                                chunk_size=args.chunk_size, verbose=False)
        r.update(w_transformer=wt, w_canine=wc, w_ngram=wn)
        rows.append(r)
        print(f"{name:28s} disjoint {r['win_rate']:6.2f}%  "
              f"avg_wrong {r['avg_wrong_won']:.3f}  {r['elapsed_s']:.0f}s", flush=True)
        if best is None or r["win_rate"] > best["win_rate"]:
            best = r

    print(f"\nBEST: {best['name']} -> {best['win_rate']:.2f}% "
          f"(incumbent blend_best = 69.01%)", flush=True)
    payload = {"transformer": args.transformer, "pretrained": args.pretrained,
               "n_words": len(words), "variants": rows, "best": best,
               "host": platform.node(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
