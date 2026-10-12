#!/usr/bin/env python3
"""Mechanical checks for src/decision_strategies.py, with a noisy-oracle stand-in
for the model so that win rates are meaningful and expectations are known.

  python tests_decision_strategies.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import decision_strategies as ds  # noqa: E402
from benchmark import load_words  # noqa: E402

words = load_words("data/holdout_split.txt")[:100]
WORD = {}


def oracle_scores(solver, patterns, guessed_lists):
    """P(present)=0.9 if the letter is in the hidden word else 0.1."""
    n = len(patterns)
    prob = np.full((n, 26), 0.1)
    mask = np.zeros((n, 26), bool)
    for i, (pat, g) in enumerate(zip(patterns, guessed_lists)):
        # recover the hidden word: unique holdout word consistent with the pattern
        w = WORD[pat, tuple(g)]
        for ch in set(w):
            prob[i, ord(ch) - 97] = 0.9
        for ch in g:
            mask[i, ord(ch) - 97] = True
    return -np.log1p(-prob), prob, mask


# the oracle needs the hidden word per game; give play() a way to see it by
# wrapping the word list: patterns identify games only with the word, so key on
# (pattern, guessed) built from the word itself.
def build_lookup():
    # every reachable (pattern, guessed) for the oracle's own greedy play is
    # not enumerable up front, so resolve lazily by matching the candidate words
    class Lazy(dict):
        def __missing__(self, key):
            pat, g = key
            for w in words:
                if len(w) == len(pat) and all(
                        (p == "_" and c not in g) or p == c for p, c in zip(pat, w)):
                    return w
            raise KeyError(key)
    return Lazy()


WORD = build_lookup()
ds.letter_scores = oracle_scores


def run(choose, seed=0):
    return ds.play(None, words, choose, seed=seed)


g = run(ds.choose_greedy)
print(f"oracle greedy        win {g['won'].mean() * 100:.0f}%")

# 1. trace invariants
uniq = np.array([len(set(w)) for w in words])
for name, r in [("greedy", g), ("eps0.2", run(ds.make_epsilon(0.2), 1)),
                ("ts_k2", run(ds.make_thompson(2), 2))]:
    hits = (r["hit"] == 1).sum(1)
    misses = (r["hit"] == 0).sum(1)
    assert (misses == r["wrong"]).all(), name
    assert (hits + misses == r["n_guess"]).all(), name
    assert (r["revealed"][:, -1] == hits).all(), name
    assert (r["revealed"][r["won"]][:, -1] == uniq[r["won"]]).all(), name
    assert (r["wrong"][r["won"]] < 6).all() and (r["wrong"][~r["won"]] == 6).all(), name
    assert (np.diff(r["revealed"], axis=1) >= 0).all(), name
    print(f"trace invariants ok  {name}")

# 2. eps=0 is exactly greedy; huge k is (nearly) greedy; eps=1 / k->0 are worse
assert (run(ds.make_epsilon(0.0))["won"] == g["won"]).all()
big = run(ds.make_thompson(1e7))["won"].mean()
print(f"TS k=1e7  {big * 100:.0f}%  (greedy {g['won'].mean() * 100:.0f}%)")
assert big >= g["won"].mean() - 0.03
e1 = run(ds.make_epsilon(1.0))["won"].mean()
t1 = run(ds.make_thompson(0.05))["won"].mean()
e2 = run(ds.make_epsilon(0.2), 3)["won"].mean()
print(f"eps=1.0   {e1 * 100:.0f}%   eps=0.2 {e2 * 100:.0f}%   TS k=0.05 {t1 * 100:.0f}%")
assert e1 < e2 < g["won"].mean() + 1e-9 and e1 < 0.5

# 3. determinism per seed, difference across seeds
a = run(ds.make_epsilon(0.2), 5)["hit"]
b = run(ds.make_epsilon(0.2), 5)["hit"]
c = run(ds.make_epsilon(0.2), 6)["hit"]
assert (a == b).all() and not (a == c).all()

# 4. epsilon exploration is uniform over unguessed letters; never repeats a guess
rng = np.random.default_rng(0)
mask = np.zeros((20000, 26), bool)
mask[:, :20] = True
picks = ds.make_epsilon(1.0)(np.zeros((20000, 26)), np.zeros((20000, 26)), mask, rng)
assert picks.min() >= 20
counts = np.bincount(picks, minlength=26)[20:]
assert counts.min() > 20000 / 6 * 0.9, counts
t = ds.make_thompson(5)(np.zeros((2000, 26)), rng.random((2000, 26)), mask[:2000], rng)
assert t.min() >= 20
print("all checks passed")
