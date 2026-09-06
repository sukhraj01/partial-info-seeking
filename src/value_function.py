"""A learned value function V(state) -> P(eventually win this game).

Expectimax needs a value for hypothetical successor states, and win probability
is exactly what on-policy collection already gives us for free: every state
recorded during a simulated game carries a `won` label from that game's outcome.
That is a Monte-Carlo return, so fitting V is a plain supervised problem on data
we already have -- no extra rollouts, no change to the transformer.

Features are deliberately cheap and model-aware: game shape (lives, blanks,
length) plus a summary of the policy's own belief (max prob, top-3 mass,
entropy). The belief features are what let V distinguish "3 blanks and the model
is certain" from "3 blanks and the model has no idea".
"""
import numpy as np

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
MAX_WRONG = 6

FEATURE_NAMES = [
    "lives_left", "n_blanks", "word_len", "frac_revealed", "n_guessed",
    "n_eliminated", "p_max", "p_top2", "p_top3_sum", "entropy",
    "n_candidates", "blanks_per_life",
]
N_FEATURES = len(FEATURE_NAMES)


def state_features(word_len, n_blanks, n_guessed, n_eliminated, probs_unguessed):
    """probs_unguessed: 1-D array of P(present) over letters not yet guessed."""
    lives = MAX_WRONG - n_eliminated
    p = np.sort(np.asarray(probs_unguessed, dtype=np.float64))[::-1]
    if p.size == 0:
        p = np.array([0.0])
    s = p.sum()
    q = p / s if s > 0 else np.full_like(p, 1.0 / p.size)
    ent = float(-(q * np.log(q + 1e-12)).sum())
    return np.array([
        lives,
        n_blanks,
        word_len,
        (word_len - n_blanks) / max(word_len, 1),
        n_guessed,
        n_eliminated,
        float(p[0]),
        float(p[1]) if p.size > 1 else 0.0,
        float(p[:3].sum()),
        ent,
        float(p.size),
        n_blanks / max(lives, 1),
    ], dtype=np.float32)


def batch_features(word_lens, n_blanks, n_guessed, n_elim, probs, guessed_mask):
    """Vectorised features for a batch. probs/guessed_mask are [B, 26]."""
    B = probs.shape[0]
    out = np.zeros((B, N_FEATURES), dtype=np.float32)
    masked = np.where(guessed_mask, -1.0, probs)
    order = np.sort(masked, axis=1)[:, ::-1]
    n_cand = (~guessed_mask).sum(axis=1).astype(np.float32)
    pos = np.clip(order, 0.0, None)
    s = pos.sum(axis=1, keepdims=True)
    q = np.where(s > 0, pos / np.maximum(s, 1e-12), 0.0)
    ent = -(q * np.log(q + 1e-12)).sum(axis=1)
    lives = (MAX_WRONG - n_elim).astype(np.float32)

    out[:, 0] = lives
    out[:, 1] = n_blanks
    out[:, 2] = word_lens
    out[:, 3] = (word_lens - n_blanks) / np.maximum(word_lens, 1)
    out[:, 4] = n_guessed
    out[:, 5] = n_elim
    out[:, 6] = np.clip(order[:, 0], 0, None)
    out[:, 7] = np.clip(order[:, 1], 0, None) if order.shape[1] > 1 else 0.0
    out[:, 8] = np.clip(order[:, :3], 0, None).sum(axis=1)
    out[:, 9] = ent
    out[:, 10] = n_cand
    out[:, 11] = n_blanks / np.maximum(lives, 1)
    return out


class ValueFunction:
    """Thin wrapper over a fitted CatBoost binary classifier."""

    def __init__(self, model_path):
        from catboost import CatBoostClassifier
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)

    def __call__(self, feats):
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim == 1:
            feats = feats[None, :]
        return self.model.predict_proba(feats)[:, 1]


def letter_multiplicity(words):
    """E[count of letter L in a word | L occurs in it], per letter.

    This is what makes the expectimax 'hit' branch letter-dependent: guessing a
    letter that typically appears twice reveals twice as much as one that
    appears once, and that information gain is exactly what greedy argmax
    ignores. Ranges from ~1.05 (q, j) to ~1.4 (e, s) on this corpus.
    """
    total = np.zeros(26, dtype=np.float64)
    docs = np.zeros(26, dtype=np.float64)
    for w in words:
        seen = {}
        for c in w:
            seen[c] = seen.get(c, 0) + 1
        for c, n in seen.items():
            i = ord(c) - 97
            total[i] += n
            docs[i] += 1
    return np.where(docs > 0, total / np.maximum(docs, 1), 1.0).astype(np.float32)


def expected_reveals(n_blanks, word_len, mult_l):
    """Blanks filled by a correct guess of L, capped by the blanks remaining.

    A candidate letter is by definition UNGUESSED, so every occurrence of it is
    still hidden -- hitting it reveals all of them, i.e. mult_l positions in
    expectation. No hidden-fraction scaling belongs here.

    (The previous version multiplied by n_blanks/word_len and then floored at
    1.0. Since mult_l maxes at 1.51 and the fraction is typically ~0.55, the
    product was always below 1.0, so the floor returned exactly 1.0 for all 26
    letters -- silently erasing the letter-dependence this term exists to
    provide, and leaving the whole hit branch constant across candidates.)
    """
    return np.minimum(mult_l, np.maximum(n_blanks, 1.0))
