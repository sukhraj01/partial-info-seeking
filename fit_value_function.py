#!/usr/bin/env python3
"""Fit V(state) -> P(win) on on-policy dumps, which already carry `won` labels.

  python fit_value_function.py --states-dir data/states_op_r1 \
      --checkpoint models/transformer_r1.pt --out models/value_fn.cbm
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from dataset import unpack_masks  # noqa: E402
from transformer_model import TransformerSolver  # noqa: E402
from value_function import FEATURE_NAMES, batch_features  # noqa: E402

MAX_LEN, PAD, BLANK, LETTER_BASE = 29, 0, 1, 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states-dir", action="append", required=True,
                    help="on-policy dump with won.u8; repeatable")
    ap.add_argument("--checkpoint", default="models/transformer.pt")
    ap.add_argument("--out", default="models/value_fn.cbm")
    ap.add_argument("--max-states", type=int, default=2_000_000)
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from catboost import CatBoostClassifier, Pool

    solver = TransformerSolver(args.checkpoint, batch_size=args.batch_size)
    model, device = solver.model, solver.device
    print(f"value fn from {args.checkpoint} on {device}", flush=True)

    feats_all, y_all = [], []
    for sd in args.states_dir:
        with open(os.path.join(sd, "meta.json")) as f:
            meta = json.load(f)
        n = meta["n_states"]
        if not meta.get("has_won_labels"):
            print(f"!! {sd} has no won labels -- skipped")
            continue
        pat = np.memmap(os.path.join(sd, "patterns.i8"), dtype=np.int8, mode="r",
                        shape=(n, MAX_LEN))
        elim = np.memmap(os.path.join(sd, "eliminated.u32"), dtype=np.uint32,
                         mode="r", shape=(n,))
        won = np.memmap(os.path.join(sd, "won.u8"), dtype=np.uint8, mode="r", shape=(n,))

        rng = np.random.default_rng(args.seed)
        take = min(args.max_states, n)
        rows = np.sort(rng.choice(n, size=take, replace=False))
        print(f"  {sd}: {take:,} of {n:,} states, win rate {won[:].mean()*100:.1f}%",
              flush=True)

        t0 = time.time()
        for i in range(0, take, args.batch_size):
            sel = rows[i:i + args.batch_size]
            tok = np.ascontiguousarray(pat[sel]).astype(np.int64)
            el = unpack_masks(np.ascontiguousarray(elim[sel]))
            tt = torch.from_numpy(tok).to(device)
            ee = torch.from_numpy(el).to(device)
            with torch.no_grad():
                logits = model(tt, ee).float()
            probs = torch.sigmoid(logits).cpu().numpy()

            revealed = np.zeros_like(el, dtype=bool)
            for j in range(26):
                revealed[:, j] = (tok == (j + LETTER_BASE)).any(axis=1)
            gmask = revealed | (el > 0)

            word_lens = (tok != PAD).sum(axis=1).astype(np.float32)
            n_blanks = (tok == BLANK).sum(axis=1).astype(np.float32)
            n_elim = el.sum(axis=1).astype(np.float32)
            n_guessed = gmask.sum(axis=1).astype(np.float32)

            feats_all.append(batch_features(word_lens, n_blanks, n_guessed,
                                            n_elim, probs, gmask))
            y_all.append(np.asarray(won[sel], dtype=np.int8))
        print(f"    featurised in {time.time()-t0:.0f}s", flush=True)

    X = np.concatenate(feats_all)
    y = np.concatenate(y_all)
    print(f"dataset {X.shape}, positive rate {y.mean()*100:.2f}%", flush=True)

    # Cap the val slice at a fifth of the data: a fixed floor larger than the
    # dataset would leave the training split empty.
    n_val = min(max(1000, int(len(y) * 0.05)), max(1, len(y) // 5))
    Xtr, ytr, Xva, yva = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]
    m = CatBoostClassifier(iterations=args.iterations, depth=6, learning_rate=0.1,
                           loss_function="Logloss", eval_metric="AUC",
                           random_seed=args.seed, verbose=100,
                           early_stopping_rounds=50)
    m.fit(Pool(Xtr, ytr, feature_names=FEATURE_NAMES),
          eval_set=Pool(Xva, yva, feature_names=FEATURE_NAMES), use_best_model=True)
    auc = m.get_best_score()["validation"]["AUC"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    m.save_model(args.out)
    print(f"\nvalue function AUC {auc:.4f} -> {args.out}")

    imp = sorted(zip(FEATURE_NAMES, m.get_feature_importance()),
                 key=lambda kv: -kv[1])
    print("top features: " + ", ".join(f"{k} {v:.1f}" for k, v in imp[:6]))
    os.makedirs("results", exist_ok=True)
    with open("results/value_function.json", "w") as f:
        json.dump({"auc": float(auc), "n": int(len(y)),
                   "positive_rate": float(y.mean()),
                   "importance": {k: float(v) for k, v in imp}}, f, indent=2)


if __name__ == "__main__":
    main()
