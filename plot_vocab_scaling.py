#!/usr/bin/env python3
"""Vocabulary scaling curve: disjoint win rate vs number of training words.

Every point holds everything else constant -- same 19.2M per-position
architecture, 128 states per word, exactly 95,600 optimizer steps, same batch
and LR schedule, same 45,060-word disjoint holdout, nested word subsets drawn
from one distribution. So the x-axis is the only variable.

Fits two models, because they disagree about the thing we actually want to know:

  log-linear   y = a + b*ln(x)      never saturates; gives "pp per doubling"
  saturating   y = A - B * x^(-c)   has a ceiling A, which is the interesting number

If the saturating fit's A sits far above the largest measured point, more
vocabulary is still the best available lever. If A is close, the well is dry.

  python plot_vocab_scaling.py --out results/vocab_scaling.png
"""
import argparse
import glob
import json
import os

import numpy as np

# (tag, vocabulary size). V1 is the 390K point -- same recipe, already run.
POINTS = [("S050", 50_000), ("S100", 100_000), ("S180", 180_000),
          ("S250", 250_000), ("V1", 390_204)]


def load_points(results_dir):
    out = []
    for tag, n in POINTS:
        path = os.path.join(results_dir, f"benchmark_{tag}.json")
        if not os.path.exists(path):
            print(f"  {tag:5s} ({n:>7,} words): not finished yet")
            continue
        d = json.load(open(path))
        out.append({"tag": tag, "words": n,
                    "disjoint": d["disjoint"]["win_rate"],
                    "in_vocab": d["in_vocab"]["win_rate"],
                    "gap": d["gap_pp"],
                    "avg_wrong": d["disjoint"]["avg_wrong_won"]})
        print(f"  {tag:5s} ({n:>7,} words): disjoint {out[-1]['disjoint']:6.2f}%  "
              f"in-vocab {out[-1]['in_vocab']:6.2f}%  gap {out[-1]['gap']:5.2f}pp")
    return sorted(out, key=lambda r: r["words"])


def fit_loglinear(x, y):
    b, a = np.polyfit(np.log(x), y, 1)
    pred = a + b * np.log(x)
    ss = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return a, b, ss


def fit_saturating(x, y):
    """y = A - B * x^(-c), fitted by grid search over the ceiling A.

    Deliberately not scipy.optimize: with 4-5 points a coarse, inspectable grid
    is more honest than an optimiser that will happily report a precise-looking
    ceiling the data cannot support.
    """
    best = None
    for A in np.arange(max(y) + 0.1, max(y) + 40.0, 0.1):
        z = np.log(A - y)              # log(B) - c*log(x)
        c, logB = np.polyfit(np.log(x), z, 1)
        c = -c
        if c <= 0:
            continue
        pred = A - np.exp(logB) * x ** (-c)
        sse = ((y - pred) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, A, np.exp(logB), c)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/vocab_scaling.png")
    ap.add_argument("--json-out", default="results/vocab_scaling.json")
    args = ap.parse_args()

    print("Vocabulary scaling points:")
    rows = load_points(args.results_dir)
    if len(rows) < 2:
        print("\nNeed at least 2 finished points to fit a curve.")
        return

    x = np.array([r["words"] for r in rows], dtype=float)
    y = np.array([r["disjoint"] for r in rows], dtype=float)

    a, b, r2 = fit_loglinear(x, y)
    per_double = b * np.log(2)
    print(f"\nlog-linear fit: disjoint = {a:.2f} + {b:.3f}*ln(words)   R^2 = {r2:.4f}")
    print(f"  => {per_double:+.2f} pp per DOUBLING of vocabulary")

    sat = fit_saturating(x, y)
    summary = {"points": rows, "loglinear": {"a": a, "b": b, "r2": r2,
                                             "pp_per_doubling": per_double}}
    if sat:
        sse, A, B, c = sat
        print(f"saturating fit: disjoint = {A:.2f} - {B:.3g}*words^(-{c:.3f})   "
              f"SSE = {sse:.4f}")
        print(f"  => estimated CEILING {A:.2f}%  "
              f"({A - y[-1]:+.2f}pp above the largest measured point)")
        summary["saturating"] = {"ceiling": A, "B": B, "c": c, "sse": sse}

    print("\nExtrapolation (log-linear; treat as an upper bound -- it cannot saturate):")
    for target in (75.0, 76.0, 78.0, 80.0):
        if target <= y.max():
            continue
        need = float(np.exp((target - a) / b))
        print(f"  {target:.0f}% disjoint would need ~{need:,.0f} words "
              f"({need / x[-1]:.1f}x the current vocabulary)")
        summary.setdefault("extrapolation", {})[str(target)] = need

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    xs = np.logspace(np.log10(x.min() * 0.8), np.log10(x.max() * 3.2), 200)
    ax.plot(xs, a + b * np.log(xs), "--", color="#888",
            label=f"log-linear: {per_double:+.2f} pp/doubling (R²={r2:.3f})")
    if sat:
        _, A, B, c = sat
        ax.plot(xs, A - B * xs ** (-c), ":", color="#c0392b",
                label=f"saturating: ceiling ≈ {A:.1f}%")
        ax.axhline(A, color="#c0392b", alpha=0.25, lw=1)
    ax.plot(x, y, "o-", color="#2c3e50", lw=2, ms=8, label="measured (disjoint)")
    for r in rows:
        ax.annotate(f"{r['disjoint']:.2f}%", (r["words"], r["disjoint"]),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("training vocabulary (words, log scale)")
    ax.set_ylabel("disjoint win rate (%)")
    ax.set_title("Hangman: vocabulary size vs out-of-vocabulary win rate\n"
                 "19.2M per-position transformer · 128 states/word · 95,600 steps",
                 fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower right")

    ax2.plot(x, [r["in_vocab"] for r in rows], "s-", color="#e67e22",
             label="in-vocab", ms=7)
    ax2.plot(x, y, "o-", color="#2c3e50", label="disjoint", ms=7)
    ax2.fill_between(x, y, [r["in_vocab"] for r in rows], alpha=0.15,
                     color="#7f8c8d", label="generalisation gap")
    for r in rows:
        ax2.annotate(f"{r['gap']:.1f}pp",
                     (r["words"], (r["in_vocab"] + r["disjoint"]) / 2),
                     textcoords="offset points", xytext=(0, -4),
                     ha="center", fontsize=8, color="#555")
    ax2.set_xscale("log")
    ax2.set_xlabel("training vocabulary (words, log scale)")
    ax2.set_ylabel("win rate (%)")
    ax2.set_title("The gap closes as vocabulary grows\n"
                  "(memorisation gives way to generalisation)", fontsize=11)
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"\nwrote {args.out}")
    with open(args.json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
