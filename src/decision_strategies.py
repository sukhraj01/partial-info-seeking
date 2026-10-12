"""Inference-time decision strategies over a frozen hangman transformer.

The model is never retrained. It supplies P(letter present) for every unguessed
letter; a *policy* turns those probabilities into one query. Three policies:

  greedy      argmax of the model's own score -- the baseline, identical to
              PositionalSolver.predict_batch
  thompson    sample q_l ~ Beta(P_l*k, (1-P_l)*k) per unguessed letter, take the
              argmax sample. k -> inf recovers greedy; small k explores.
  epsilon     with probability eps, query a uniformly random unguessed letter

play() advances all games in lockstep (like batch_sim) but also records a
per-ply trace, so one run feeds the outcome tables and the cold-start curve.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformer_model import encode_batch  # noqa: E402

MAX_WRONG = 6
MAX_PLY = 26
P_EPS = 1e-6


@torch.no_grad()
def letter_scores(solver, patterns, guessed_lists):
    """-> (score, prob, guessed_mask), each float64/bool ndarray [N, 26].

    score is the solver's own ranking key (-log P(absent)); prob is P(present).
    Greedy uses score so ties and saturation behave exactly like the benchmark;
    only the samplers need prob.
    """
    scores, probs, masks = [], [], []
    for i in range(0, len(patterns), solver.batch_size):
        tk, el, gm = encode_batch(patterns[i:i + solver.batch_size],
                                  guessed_lists[i:i + solver.batch_size],
                                  solver.device)
        log_absent = solver.model.presence_logprob(solver.model(tk, el), tk)
        scores.append((-log_absent).double().cpu().numpy())
        probs.append((-torch.expm1(log_absent)).double().cpu().numpy())
        masks.append(gm.cpu().numpy())
    return np.concatenate(scores), np.concatenate(probs), np.concatenate(masks)


def choose_greedy(score, prob, mask, rng):
    return np.where(mask, -np.inf, score).argmax(-1)


def make_thompson(k):
    def choose(score, prob, mask, rng):
        p = np.clip(prob, P_EPS, 1 - P_EPS)
        draw = rng.beta(p * k, (1 - p) * k)
        return np.where(mask, -np.inf, draw).argmax(-1)
    return choose


def make_epsilon(eps):
    def choose(score, prob, mask, rng):
        greedy = np.where(mask, -np.inf, score).argmax(-1)
        if eps <= 0:
            return greedy
        # random key per letter, masked: argmax is a uniform pick among unguessed
        rand = np.where(mask, -np.inf, rng.random(mask.shape)).argmax(-1)
        explore = rng.random(len(greedy)) < eps
        return np.where(explore, rand, greedy)
    return choose


def play(solver, words, choose, seed=0, chunk_size=8192, max_wrong=MAX_WRONG):
    """Play every word to completion under `choose`; return per-word arrays.

    won[n]        bool
    wrong[n]      int, misses charged
    n_guess[n]    int, queries made
    hit[n, t]     int8: 1 hit, 0 miss, -1 no query at ply t (game already over)
    revealed[n,t] int16: distinct letters of the word revealed after ply t
                  (carried forward once the game ends)
    """
    rng = np.random.default_rng(seed)
    n = len(words)
    won = np.zeros(n, bool)
    wrong = np.zeros(n, np.int16)
    n_guess = np.zeros(n, np.int16)
    hit = np.full((n, MAX_PLY), -1, np.int8)
    revealed = np.zeros((n, MAX_PLY), np.int16)

    for c0 in range(0, n, chunk_size):
        chunk = words[c0:c0 + chunk_size]
        m = len(chunk)
        patterns = [["_"] * len(w) for w in chunk]
        guessed = [[] for _ in range(m)]
        uniq_found = np.zeros(m, np.int16)
        active = list(range(m))

        for t in range(MAX_PLY):
            if not active:
                revealed[c0:c0 + m, t:] = uniq_found[:, None]
                break
            score, prob, mask = letter_scores(
                solver, ["".join(patterns[i]) for i in active],
                [guessed[i] for i in active])
            picks = choose(score, prob, mask, rng)
            still = []
            for i, j in zip(active, picks):
                ch = chr(97 + int(j))
                guessed[i].append(ch)
                word = chunk[i]
                if ch in word:
                    for p, c in enumerate(word):
                        if c == ch:
                            patterns[i][p] = c
                    uniq_found[i] += 1
                    hit[c0 + i, t] = 1
                else:
                    wrong[c0 + i] += 1
                    hit[c0 + i, t] = 0
                n_guess[c0 + i] += 1
                if wrong[c0 + i] < max_wrong and "_" in patterns[i]:
                    still.append(i)
            revealed[c0:c0 + m, t] = uniq_found
            active = still

        for i in range(m):
            won[c0 + i] = "_" not in patterns[i]

    return {"won": won, "wrong": wrong, "n_guess": n_guess,
            "hit": hit, "revealed": revealed}


def difficulty_bins(words):
    """easy <=5 unique letters, medium 6-7, hard 8+."""
    u = np.array([len(set(w)) for w in words])
    return u, np.where(u <= 5, 0, np.where(u <= 7, 1, 2))


BIN_NAMES = ["easy (<=5 unique)", "medium (6-7)", "hard (8+)"]
