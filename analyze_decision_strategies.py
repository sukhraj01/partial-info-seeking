#!/usr/bin/env python3
"""Tables and figures for the decision-strategy experiments (A1-A3).

Reads results/decision/*.npz (per-word outcomes written by
run_decision_strategies.py); writes figures/ and results/decision/analysis.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from benchmark import load_words  # noqa: E402
from decision_strategies import BIN_NAMES, difficulty_bins  # noqa: E402

K_SWEEP = [1, 2, 5, 10, 50, 100]
EPS_SWEEP = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]
BLUE, ORANGE, GREEN, INK, GRID = "#0072B2", "#D55E00", "#009E73", "#222222", "#dddddd"

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/decision")
ap.add_argument("--holdout-split", default="data/holdout_split.txt")
ap.add_argument("--figdir", default="figures")
ap.add_argument("--sample", type=int, default=None)
args = ap.parse_args()

words = load_words(args.holdout_split)
if args.sample:
    import random
    words = random.Random(0).sample(words, min(args.sample, len(words)))
uniq, binid = difficulty_bins(words)
length = np.array([len(w) for w in words])
n = len(words)


def load(tag):
    p = os.path.join(args.dir, f"{tag}.npz")
    return dict(np.load(p)) if os.path.exists(p) else None


def by_bin(won):
    return [won[binid == b].mean() * 100 for b in range(3)]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def msd(rows):
    a = np.array(rows)
    return a.mean(0), a.std(0, ddof=1) if len(a) > 1 else np.zeros_like(a.mean(0))


greedy = load("greedy")
out = {"n_words": n, "greedy_win": greedy["won"].mean() * 100,
       "bin_counts": [int((binid == b).sum()) for b in range(3)],
       "greedy_by_bin": by_bin(greedy["won"])}
os.makedirs(args.figdir, exist_ok=True)

# ---- A1 Thompson sampling -------------------------------------------------
sweep = {k: load(f"ts_k{k}_s0") for k in K_SWEEP}
sweep = {k: v for k, v in sweep.items() if v is not None}
if sweep:
    out["A1_sweep"] = {k: {"win": v["won"].mean() * 100, "avg_wrong": float(v["wrong"].mean()),
                           "by_bin": by_bin(v["won"])} for k, v in sweep.items()}
    best_k = max(sweep, key=lambda k: sweep[k]["won"].mean())
    runs = [r for s in range(3) if (r := load(f"ts_k{best_k}_s{s}")) is not None]
    wins = [r["won"].mean() * 100 for r in runs]
    gw = greedy["won"]
    gained = np.mean([(r["won"] & ~gw).sum() for r in runs])
    lost = np.mean([(~r["won"] & gw).sum() for r in runs])
    mean_won = np.mean([r["won"] for r in runs], axis=0)   # per-word win prob over seeds
    bm, bs = msd([by_bin(r["won"]) for r in runs])
    rows = {}
    for name, sel in [("length<=6", length <= 6), ("length 7-9", (length >= 7) & (length <= 9)),
                      ("length 10-12", (length >= 10) & (length <= 12)), ("length>=13", length >= 13)]:
        rows[name] = {"n": int(sel.sum()), "greedy": gw[sel].mean() * 100,
                      "ts": mean_won[sel].mean() * 100}
    out["A1_best"] = {"k": best_k, "win_mean": float(np.mean(wins)),
                      "win_std": float(np.std(wins, ddof=1)) if len(wins) > 1 else 0.0,
                      "seeds": wins, "gained_vs_greedy": float(gained), "lost_vs_greedy": float(lost),
                      "by_bin_mean": bm.tolist(), "by_bin_std": bs.tolist(), "by_length": rows}
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ks = list(sweep)
    ax.plot(ks, [sweep[k]["won"].mean() * 100 for k in ks], "o-", color=BLUE, lw=1.5, ms=6,
            label="Thompson sampling (seed 0)")
    ax.axhline(out["greedy_win"], color=ORANGE, lw=1.5, ls="--", label="greedy argmax")
    ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("k  (Beta concentration; high k → greedy)"); ax.set_ylabel("disjoint win rate (%)")
    ax.legend(frameon=False); style(ax); fig.tight_layout()
    fig.savefig(os.path.join(args.figdir, "thompson_sweep.png"), dpi=160); plt.close(fig)

# ---- A2 cold-start curve --------------------------------------------------
T = 15
active = (greedy["hit"] != -1)
hit_rate, reveal = [], []
uniq_all = uniq.astype(float)
ended = np.zeros(n, bool)
for t in range(T):
    a = active[:, t]
    hit_rate.append((greedy["hit"][a, t] == 1).mean() * 100 if a.any() else np.nan)
    reveal.append((greedy["revealed"][:, t] / uniq_all).mean() * 100)
# decided by guess t = games whose last query index is <= t
last = greedy["n_guess"] - 1
decided = [(last <= t).mean() * 100 for t in range(T)]
decided_won = [((last <= t) & greedy["won"]).mean() * 100 for t in range(T)]
out["A2"] = {"guess": list(range(1, T + 1)), "hit_rate": hit_rate, "reveal_frac": reveal,
             "decided": decided, "decided_won": decided_won,
             "still_playing": [int(active[:, t].sum()) for t in range(T)]}
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(1, T + 1)
ax.plot(x, hit_rate, "o-", color=BLUE, lw=1.5, ms=5, label="hit rate (games still playing)")
ax.plot(x, reveal, "s-", color=GREEN, lw=1.5, ms=5, label="cumulative reveal (unique letters found)")
ax.plot(x, decided, "^-", color=ORANGE, lw=1.5, ms=5, label="game decided (won or lost)")
ax.set_xlabel("guess number"); ax.set_ylabel("%"); ax.set_xticks(x); ax.set_ylim(0, 102)
ax.legend(frameon=False, loc="lower right", fontsize=9); style(ax); fig.tight_layout()
fig.savefig(os.path.join(args.figdir, "cold_start_curve.png"), dpi=160); plt.close(fig)

# ---- A3 epsilon-greedy ----------------------------------------------------
grid = {}
for eps in EPS_SWEEP:
    runs = [r for s in range(3) if (r := load(f"eps{eps:g}_s{s}")) is not None]
    if not runs:
        continue
    overall = [r["won"].mean() * 100 for r in runs]
    bm, bs = msd([by_bin(r["won"]) for r in runs])
    grid[eps] = {"n_seeds": len(runs), "overall_mean": float(np.mean(overall)),
                 "overall_std": float(np.std(overall, ddof=1)) if len(runs) > 1 else 0.0,
                 "bin_mean": bm.tolist(), "bin_std": bs.tolist()}
out["A3"] = {str(e): v for e, v in grid.items()}
if grid:
    fig, ax = plt.subplots(figsize=(7, 4))
    for b, col in enumerate([GREEN, BLUE, ORANGE]):
        base = grid[0.0]["bin_mean"][b]
        es = list(grid)
        m = [grid[e]["bin_mean"][b] - base for e in es]
        s = [grid[e]["bin_std"][b] for e in es]
        ax.errorbar(es, m, yerr=s, marker="o", color=col, lw=1.5, ms=5, capsize=3, label=BIN_NAMES[b])
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("epsilon"); ax.set_ylabel("Δ win rate vs ε=0 (pp)")
    ax.legend(frameon=False, fontsize=9); style(ax); fig.tight_layout()
    fig.savefig(os.path.join(args.figdir, "epsilon_by_difficulty.png"), dpi=160); plt.close(fig)

with open(os.path.join(args.dir, "analysis.json"), "w") as f:
    json.dump(out, f, indent=2, default=float)

# ---- text report ----------------------------------------------------------
print(f"greedy {out['greedy_win']:.2f}%  bins {['%.2f' % v for v in out['greedy_by_bin']]} counts {out['bin_counts']}")
for k, v in out.get("A1_sweep", {}).items():
    print(f"TS k={k}: {v['win']:.2f}%  bins {['%.2f' % x for x in v['by_bin']]}")
if "A1_best" in out:
    b = out["A1_best"]
    print(f"TS best k={b['k']}: {b['win_mean']:.2f} ± {b['win_std']:.2f}  gained {b['gained_vs_greedy']:.0f} lost {b['lost_vs_greedy']:.0f}")
    print("  by bin", [f"{m:.2f}±{s:.2f}" for m, s in zip(b["by_bin_mean"], b["by_bin_std"])])
    for k, v in b["by_length"].items():
        print(f"  {k}: n={v['n']} greedy {v['greedy']:.2f} ts {v['ts']:.2f}")
for e, v in out["A3"].items():
    print(f"eps={e}: {v['overall_mean']:.2f} ± {v['overall_std']:.2f} | " +
          " | ".join(f"{m:.2f}±{s:.2f}" for m, s in zip(v["bin_mean"], v["bin_std"])))
for i in range(T):
    print(f"guess {i+1}: hit {hit_rate[i]:.1f} reveal {reveal[i]:.1f} decided {decided[i]:.1f}")
