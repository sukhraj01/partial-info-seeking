#!/usr/bin/env python3
"""Step 3: benchmark every candidate on the full disjoint split and measure how
CORRELATED their errors are.

Raw score difference is the wrong criterion for picking two models to keep. Two
models that score 67.4% and 67.2% but lose the SAME words are one model
with extra steps; a lower-scoring model that loses DIFFERENT words is a real
hedge against a private set drawn differently from our holdout.

So for every pair this reports:
  both_lose     words neither model can save -- the shared floor
  phi           correlation of the win/loss indicator vectors (1.0 = identical)
  union_win     fraction won by at least one model -- complementarity ceiling
  McNemar p     whether the score difference itself is real

Model specs are `name=kind:path[:mode:value]`, kind in {tfm, ens, byt5}:
  --model r0=tfm:models/transformer.pt
  --model blend=ens:models/transformer.pt:blend:0.70
  --model byt5=byt5:models/pretrained.pt
"""
import argparse, json, os, platform, sys, time
from math import comb, erfc, sqrt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from batch_sim import run_batch_benchmark  # noqa: E402
from benchmark import load_words  # noqa: E402


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


def phi(a, b):
    """Correlation of two binary win vectors."""
    n11 = sum(1 for x, y in zip(a, b) if x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n00 = sum(1 for x, y in zip(a, b) if not x and not y)
    num = n11 * n00 - n10 * n01
    den = sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return (num / den) if den else float("nan"), (n11, n10, n01, n00)


def build_solver(spec, dictionary, batch_size, ngram_cache):
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    kind, path = parts[0], parts[1]
    if kind == "tfm":
        from transformer_model import TransformerSolver
        return name, TransformerSolver(path, batch_size=batch_size).predict_batch
    if kind == "pos":
        from transformer_positional import PositionalSolver
        return name, PositionalSolver(path, batch_size=batch_size).predict_batch
    if kind == "ens":
        from ensemble_model import EnsembleSolver
        from ngram_model import NgramModel
        mode, val = parts[2], float(parts[3])
        if "ng" not in ngram_cache:
            t = time.time()
            ngram_cache["ng"] = NgramModel(dictionary)
            print(f"  (built n-gram tables in {time.time()-t:.0f}s)", flush=True)
        s = EnsembleSolver(path, dictionary, mode=mode,
                           weight=val if mode != "route" else 0.8,
                           threshold=val if mode == "route" else 0.5,
                           batch_size=batch_size, ngram=ngram_cache["ng"])
        return name, s.predict_batch
    if kind == "byt5":
        from train_pretrained import PretrainedSolver
        return name, PretrainedSolver(path, batch_size=min(batch_size, 1024)).predict_batch
    raise ValueError(f"unknown kind {kind!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, dest="models")
    ap.add_argument("--holdout-split", default="data/holdout_split.txt")
    ap.add_argument("--dictionary", default="data/train_split.txt")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--out", default="results/model_comparison.json")
    args = ap.parse_args()

    words = load_words(args.holdout_split)
    print(f"FULL disjoint split: {len(words):,} words (no subsampling)\n", flush=True)

    ngram_cache, results = {}, []
    for spec in args.models:
        name, pb = build_solver(spec, args.dictionary, args.batch_size, ngram_cache)
        r = run_batch_benchmark(pb, words, name=name, chunk_size=args.chunk_size,
                                verbose=False, per_word=True)
        results.append({"name": name, "spec": spec, "win_rate": r["win_rate"],
                        "avg_wrong_won": r["avg_wrong_won"], "wins": r["wins"],
                        "ms_per_word": r["time_per_word_ms"], "flags": r["won_flags"]})
        print(f"{name:24s} disjoint {r['win_rate']:6.2f}%  avg_wrong {r['avg_wrong_won']:.3f}"
              f"  {r['time_per_word_ms']:.2f} ms/word", flush=True)

    print("\n=== pairwise error correlation ===", flush=True)
    pairs = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            A, B = results[i], results[j]
            p, (n11, n10, n01, n00) = phi(A["flags"], B["flags"])
            n = len(words)
            pv = mcnemar_exact(n10, n01)
            union = (n11 + n10 + n01) / n * 100
            pairs.append({"a": A["name"], "b": B["name"], "phi": p,
                          "both_win": n11, "a_only": n10, "b_only": n01,
                          "both_lose": n00, "both_lose_pct": n00 / n * 100,
                          "union_win_pct": union, "mcnemar_p": pv,
                          "delta_pp": B["win_rate"] - A["win_rate"]})
            print(f"{A['name']:>20s} vs {B['name']:<20s} "
                  f"phi {p:+.3f} | both-lose {n00/n*100:5.2f}% | "
                  f"union {union:6.2f}% | only-A {n10:5d} only-B {n01:5d} | "
                  f"McNemar p={pv:.3g}", flush=True)

    best = max(results, key=lambda r: r["win_rate"])
    print(f"\nbest single: {best['name']} at {best['win_rate']:.2f}%", flush=True)
    print("When keeping two models, prefer a pair with LOW phi and HIGH union "
          "win -- that is a genuine hedge, not a duplicate.", flush=True)

    payload = {"n_words": len(words),
               "models": [{k: v for k, v in r.items() if k != "flags"} for r in results],
               "pairs": pairs, "host": platform.node(),
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    # Per-word flags are large; keep them beside the summary for later analysis.
    with open(args.out.replace(".json", "_flags.json"), "w") as f:
        json.dump({r["name"]: r["flags"] for r in results}, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
