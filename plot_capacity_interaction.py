#!/usr/bin/env python3
"""Capacity x diversity interaction: does more capacity help, or hurt?

Four runs on a 2x2 grid, identical in every respect except parameter count and
vocabulary size (128 states per word, exactly 95,600 optimizer steps, effective
batch 2048, same schedule, same 2080 Ti, same 45,060-word disjoint holdout, and
the same best-val-win-rate checkpoint rule).

                 180K vocab   390K vocab
  19.2M params   S180         V1
  57M params     CAP57_180K   CAP57_390K

The main effect of capacity is not a single number: it changes SIGN across the
diversity axis. The left panel is the interaction plot; the right panel is the
mechanism, because the training curves show what the endpoints cannot -- at 180K
the 57M model peaks two epochs in and then degrades, so its selected checkpoint
comes from 23% of the budget it was given.

  python plot_capacity_interaction.py --out figures/capacity_interaction.png
"""
import argparse
import json
import os

import numpy as np

# (tag, parameters, vocabulary, label). The 2x2 grid, read off the benchmarks.
CELLS = [
    ("S180",       19_232_794, 180_000, "19.2M @ 180K"),
    ("V1",         19_232_794, 390_204, "19.2M @ 390K"),
    ("CAP57_180K", 57_167_130, 180_000, "57M @ 180K"),
    ("CAP57_390K", 57_167_130, 390_204, "57M @ 390K"),
]

CAPS = [(19_232_794, "19.2M"), (57_167_130, "57M")]
VOCABS = [(180_000, "180K"), (390_204, "390K")]


def load_grid(results_dir):
    grid, missing = {}, []
    for tag, params, words, label in CELLS:
        bpath = os.path.join(results_dir, f"benchmark_{tag}.json")
        tpath = os.path.join(results_dir, f"training_{tag}.json")
        if not os.path.exists(bpath):
            missing.append(tag)
            continue
        b = json.load(open(bpath))
        cell = {"tag": tag, "params": params, "words": words, "label": label,
                "disjoint": b["disjoint"]["win_rate"],
                "in_vocab": b["in_vocab"]["win_rate"],
                "gap": b["gap_pp"],
                "avg_wrong": b["disjoint"]["avg_wrong_won"],
                "ckpt_step": b["checkpoint_meta"]["step"],
                "ckpt_epoch": b["checkpoint_meta"]["epoch"],
                "history": []}
        if os.path.exists(tpath):
            cell["history"] = json.load(open(tpath))["history"]
        grid[(params, words)] = cell
    return grid, missing


def se_pp(p1, p2, n=45060):
    """Unpaired binomial standard error on a difference of two win rates, in pp."""
    a, b = p1 / 100.0, p2 / 100.0
    return float(np.sqrt(a * (1 - a) / n + b * (1 - b) / n) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="figures/capacity_interaction.png")
    ap.add_argument("--json-out", default="figures/capacity_interaction.json")
    args = ap.parse_args()

    grid, missing = load_grid(args.results_dir)
    if missing:
        print(f"missing cells: {', '.join(missing)}")
    if len(grid) < 4:
        print("Need all four cells to draw the interaction.")
        return

    print("2x2 grid (disjoint win rate on 45,060 held-out words):")
    for params, cap in CAPS:
        for words, voc in VOCABS:
            c = grid[(params, words)]
            print(f"  {cap:5s} @ {voc:4s}  in {c['in_vocab']:6.2f}  "
                  f"disjoint {c['disjoint']:6.2f}  gap {c['gap']:5.2f}pp  "
                  f"ckpt ep{c['ckpt_epoch']} step {c['ckpt_step']:,}")

    effects, summary_cells = {}, {}
    print("\ncapacity effect (19.2M -> 57M) at each vocabulary:")
    for words, voc in VOCABS:
        lo, hi = grid[(CAPS[0][0], words)], grid[(CAPS[1][0], words)]
        d = hi["disjoint"] - lo["disjoint"]
        se = se_pp(lo["disjoint"], hi["disjoint"])
        effects[voc] = {"delta_pp": d, "se_pp": se, "sigma": abs(d) / se}
        print(f"  {voc:4s}: {lo['disjoint']:.2f} -> {hi['disjoint']:.2f} = "
              f"{d:+.2f}pp  (SE {se:.3f}pp, {abs(d)/se:.1f} sigma)")

    inter = effects["390K"]["delta_pp"] - effects["180K"]["delta_pp"]
    print(f"\nINTERACTION (390K effect minus 180K effect): {inter:+.2f}pp")
    print("  capacity changes SIGN across the diversity axis.")

    for (params, words), c in grid.items():
        summary_cells[c["tag"]] = {k: v for k, v in c.items() if k != "history"}
    summary = {"cells": summary_cells, "capacity_effect": effects,
               "interaction_pp": inter,
               "note": ("Identical 95,600-step budget and identical "
                        "best-val-win-rate checkpoint rule in every cell. "
                        "One seed per cell; SE shown is evaluation-set noise "
                        "only and does not cover training-seed variance.")}

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    c_small, c_big = "#2c3e50", "#c0392b"

    # Left: the interaction itself.
    xs = np.arange(len(VOCABS))
    for (params, cap), colour, marker in ((CAPS[0], c_small, "o"),
                                          (CAPS[1], c_big, "s")):
        ys = [grid[(params, w)]["disjoint"] for w, _ in VOCABS]
        ax.plot(xs, ys, marker=marker, color=colour, lw=2.2, ms=9,
                label=f"{cap} parameters")
        # Values to the LEFT of each marker; the effect arrows own the right.
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points",
                        xytext=(-12, 0), ha="right", va="center",
                        fontsize=9, color=colour)

    for x, (words, voc) in enumerate(VOCABS):
        d = effects[voc]["delta_pp"]
        lo = grid[(CAPS[0][0], words)]["disjoint"]
        hi = grid[(CAPS[1][0], words)]["disjoint"]
        ax.annotate("", xy=(x + 0.07, hi), xytext=(x + 0.07, lo),
                    arrowprops=dict(arrowstyle="<->", color="#555", lw=1.3))
        ax.annotate(f"{d:+.2f}pp\n({'helps' if d > 0 else 'hurts'})",
                    (x + 0.12, (lo + hi) / 2),
                    fontsize=10, color="#555", va="center",
                    fontweight="bold", linespacing=1.4)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v}\n({w:,} words)" for w, v in VOCABS])
    ax.set_xlim(-0.30, len(VOCABS) - 0.55)
    ax.set_xlabel("training vocabulary (environment diversity)")
    ax.set_ylabel("disjoint win rate (%)")
    ax.set_title("Capacity helps only once diversity is sufficient\n"
                 f"interaction = {inter:+.2f}pp · identical 95,600-step budget "
                 "in all four cells", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    # Right: the mechanism -- training dynamics, where the endpoints hide it.
    styles = {(19_232_794, 180_000): (c_small, "-", "19.2M @ 180K"),
              (19_232_794, 390_204): (c_small, "--", "19.2M @ 390K"),
              (57_167_130, 180_000): (c_big, "-", "57M @ 180K"),
              (57_167_130, 390_204): (c_big, "--", "57M @ 390K")}
    for key, (colour, ls, label) in styles.items():
        h = grid[key]["history"]
        if not h:
            continue
        steps = [x["step"] for x in h]
        wins = [x["val_win_rate"] for x in h]
        ax2.plot(steps, wins, ls, color=colour, lw=2, marker=".", ms=8,
                 label=label, alpha=0.95 if ls == "-" else 0.75)
        best = int(np.argmax(wins))
        ax2.plot(steps[best], wins[best], "*", color=colour, ms=16,
                 markeredgecolor="white", markeredgewidth=0.8, zorder=5)

    peak = grid[(57_167_130, 180_000)]
    ax2.annotate("57M @ 180K peaks at 22,050 steps\n"
                 "(23% of its budget) then degrades",
                 xy=(peak["ckpt_step"], 71.65),
                 xytext=(30_000, 66.2), fontsize=9, color=c_big,
                 arrowprops=dict(arrowstyle="->", color=c_big, lw=1.2,
                                 connectionstyle="arc3,rad=-0.25"))

    ax2.set_xlabel("optimizer steps")
    ax2.set_ylabel("validation win rate (%)")
    ax2.set_title("The mechanism: too much capacity for too few environments\n"
                  "star = selected checkpoint (best val win rate)", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"\nwrote {args.out}")
    with open(args.json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
