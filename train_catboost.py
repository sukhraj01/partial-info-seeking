#!/usr/bin/env python3
"""Train 26 per-letter CatBoost classifiers. CPU, runs on the Ada login node.

Feature extraction calls the n-gram model once per state (~70us), which is the
real cost here -- not the boosting. So this subsamples the state pool rather
than using all 50M: 26 models x 50M rows would be hours of feature extraction
for a control experiment. --n-states controls the tradeoff.

  python train_catboost.py --states-dir data/states --n-states 2000000
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from catboost_model import (ALPHABET, N_SHARED, build_shared,  # noqa: E402
                            corpus_frequencies, feature_names, letter_features)
from ngram_model import NgramModel  # noqa: E402

_G = {}


def _init(train_split, corpus_freq):
    _G["ngram"] = NgramModel(train_split)
    _G["freq"] = corpus_freq


def _extract(chunk):
    """chunk: list of (pattern, guessed_list, target_mask, guessed_mask)."""
    ngram, freq = _G["ngram"], _G["freq"]
    n = len(chunk)
    # Shared features only -- per-letter columns are derived at fit time.
    X = np.zeros((n, N_SHARED), dtype=np.float32)
    Y = np.zeros((n, 26), dtype=np.int8)
    A = np.zeros((n, 26), dtype=bool)
    for k, (pat, guessed, tgt_mask, g_mask) in enumerate(chunk):
        scores, _dead = ngram.score_letters(pat, guessed)
        X[k] = build_shared(pat, guessed, scores)
        for i in range(26):
            Y[k, i] = (tgt_mask >> i) & 1
            A[k, i] = not ((g_mask >> i) & 1)
    return X, Y, A


def decode_states(state_dir, n_take, seed):
    """Sample rows and decode them into (pattern, guessed, target_mask, guessed_mask)."""
    with open(os.path.join(state_dir, "meta.json")) as f:
        meta = json.load(f)
    n = meta["n_states"]
    words = [w.strip() for w in open(meta["source_words"]) if w.strip()]

    pat_mm = np.memmap(os.path.join(state_dir, "patterns.i8"), dtype=np.int8,
                       mode="r", shape=(n, 29))
    elim_mm = np.memmap(os.path.join(state_dir, "eliminated.u32"), dtype=np.uint32,
                        mode="r", shape=(n,))
    tgt_mm = np.memmap(os.path.join(state_dir, "target.u32"), dtype=np.uint32,
                       mode="r", shape=(n,))
    widx_mm = np.memmap(os.path.join(state_dir, "word_idx.u32"), dtype=np.uint32,
                        mode="r", shape=(n,))

    rng = np.random.default_rng(seed)
    take = min(n_take, n)
    rows = np.sort(rng.choice(n, size=take, replace=False))

    pats = pat_mm[rows]
    elims = elim_mm[rows]
    tgts = tgt_mm[rows]
    widxs = widx_mm[rows]

    out = []
    for r in range(take):
        wlen = len(words[int(widxs[r])])
        row = pats[r]
        pattern = "".join("_" if row[p] == 1 else ALPHABET[row[p] - 2] for p in range(wlen))
        e = int(elims[r])
        revealed = sorted(set(pattern) - {"_"})
        guessed = revealed + [ALPHABET[i] for i in range(26) if e >> i & 1]
        g_mask = int(tgts[r]) * 0
        for c in guessed:
            g_mask |= 1 << (ord(c) - 97)
        out.append((pattern, guessed, int(tgts[r]), g_mask))
    return out, meta, words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states-dir", default="data/states")
    ap.add_argument("--train-split", default="data/train_split.txt")
    ap.add_argument("--out-dir", default="models/catboost")
    ap.add_argument("--n-states", type=int, default=2_000_000)
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--thread-count", type=int, default=os.cpu_count())
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Aliased: catboost.Pool would shadow multiprocessing.Pool used below.
    from catboost import CatBoostClassifier
    from catboost import Pool as CBPool

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    print(f"decoding {args.n_states:,} states...", flush=True)
    states, meta, words = decode_states(args.states_dir, args.n_states, args.seed)
    corpus_freq = corpus_frequencies(words)
    print(f"decoded {len(states):,} in {time.time()-t0:.0f}s", flush=True)

    print(f"extracting features with {args.workers} workers...", flush=True)
    te = time.time()
    chunk = 20000
    chunks = [states[i:i + chunk] for i in range(0, len(states), chunk)]
    with Pool(args.workers, initializer=_init,
              initargs=(args.train_split, corpus_freq)) as pool:
        parts = pool.map(_extract, chunks)
    X = np.concatenate([p[0] for p in parts])
    Y = np.concatenate([p[1] for p in parts])
    A = np.concatenate([p[2] for p in parts])
    del parts
    print(f"features {X.shape} in {time.time()-te:.0f}s "
          f"({X.nbytes/1e9:.1f} GB)", flush=True)

    names = feature_names()
    n = X.shape[0]
    n_val = int(n * args.val_frac)
    summary = {}

    for i, c in enumerate(ALPHABET):
        rows = np.flatnonzero(A[:, i])  # only states where letter c is unguessed
        if rows.size < 1000:
            print(f"[{c}] only {rows.size} usable rows -- skipped", flush=True)
            continue
        Xi = letter_features(X[rows], i, corpus_freq)
        Yi = Y[rows, i]
        vi = min(n_val, max(1000, int(rows.size * args.val_frac)))
        Xtr, Ytr = Xi[:-vi], Yi[:-vi]
        Xva, Yva = Xi[-vi:], Yi[-vi:]

        tl = time.time()
        m = CatBoostClassifier(iterations=args.iterations, depth=args.depth,
                               learning_rate=args.lr, loss_function="Logloss",
                               eval_metric="AUC", random_seed=args.seed,
                               thread_count=args.thread_count, verbose=0,
                               early_stopping_rounds=50)
        # feature_names belongs on the Pool, not on fit().
        train_pool = CBPool(Xtr, Ytr, feature_names=names)
        val_pool = CBPool(Xva, Yva, feature_names=names)
        m.fit(train_pool, eval_set=val_pool, use_best_model=True)
        auc = m.get_best_score()["validation"]["AUC"]
        path = os.path.join(args.out_dir, f"letter_{c}.cbm")
        m.save_model(path)
        summary[c] = {"rows": int(rows.size), "positive_rate": float(Yi.mean()),
                      "val_auc": float(auc), "best_iteration": int(m.get_best_iteration()),
                      "seconds": round(time.time() - tl, 1)}
        print(f"[{c}] rows {rows.size:,} pos {Yi.mean()*100:5.1f}% "
              f"AUC {auc:.4f} iter {m.get_best_iteration()} "
              f"({time.time()-tl:.0f}s)", flush=True)

    np.save(os.path.join(args.out_dir, "corpus_freq.npy"), corpus_freq)
    total_min = (time.time() - t0) / 60
    mean_auc = float(np.mean([v["val_auc"] for v in summary.values()])) if summary else 0.0
    print(f"\ntrained {len(summary)}/26 models in {total_min:.1f}m | mean AUC {mean_auc:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/catboost_training.json", "w") as f:
        json.dump({"per_letter": summary, "mean_val_auc": mean_auc,
                   "minutes": total_min, "args": vars(args),
                   "n_states": len(states)}, f, indent=2)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({"train_split": args.train_split, "n_states": len(states),
                   "feature_names": names}, f, indent=2)


if __name__ == "__main__":
    main()
