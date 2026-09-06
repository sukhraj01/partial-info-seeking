"""Per-letter CatBoost classifiers: 26 models answering "is letter L in this word?"

This is the control for the 67% disjoint figure reported by the only
credibly-benchmarked GBM solution in the literature. Its job is to tell us
whether ~67% is reachable with OUR data and split, so we know whether a
transformer landing at 60% is under- or over-performing.

Feature design
--------------
Shared features are identical for all 26 models; per-letter features are
appended so each model sees its own letter's evidence explicitly. The n-gram
model supplies the heavy lifting -- its full 26-way score vector goes in as
shared context, and the target letter's own score and rank go in per-letter.
Handing the GBM the n-gram distribution means it learns to *correct* a 55.5%
model rather than rediscover sub-word structure from scratch.

Training filter: for letter L we keep only states where L has not been guessed,
because that is the only situation in which model L is ever consulted at
inference. Training on already-guessed states would skew the base rate and
teach nothing usable.
"""
import numpy as np

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
VOWELS = set("aeiou")
MAX_LEN = 29
PAD, BLANK, LETTER_BASE = 0, 1, 2
MAX_WRONG = 6

N_SHARED = 7 + 26 + MAX_LEN + 26   # scalars + guessed + blank bitmap + ngram scores
N_PER_LETTER = 4
N_FEATURES = N_SHARED + N_PER_LETTER
NGRAM_OFF = 7 + 26 + MAX_LEN       # where the 26 n-gram scores start in a shared row

IS_VOWEL = np.array([1.0 if c in VOWELS else 0.0 for c in ALPHABET], dtype=np.float32)


def letter_features(shared: np.ndarray, i: int, corpus_freq) -> np.ndarray:
    """Build the [n, N_FEATURES] matrix for letter i from shared rows.

    The four per-letter columns are all derivable from the shared row, so we
    store shared features ONCE per state instead of materialising
    [n, 26, N_FEATURES]. At 2M states that is 0.7 GB instead of 19 GB -- the
    difference between fitting the 32 GB QOS cap and not.
    """
    n = shared.shape[0]
    scores = shared[:, NGRAM_OFF:NGRAM_OFF + 26]
    self_score = scores[:, i]
    # rank 0 = highest scoring letter for that state
    rank = (scores > self_score[:, None]).sum(axis=1).astype(np.float32)
    out = np.empty((n, N_FEATURES), dtype=np.float32)
    out[:, :N_SHARED] = shared
    out[:, N_SHARED] = self_score
    out[:, N_SHARED + 1] = rank
    out[:, N_SHARED + 2] = corpus_freq[i]
    out[:, N_SHARED + 3] = IS_VOWEL[i]
    return out


def shared_feature_names():
    names = ["word_len", "n_blanks", "frac_revealed", "n_revealed_vowels",
             "n_revealed_consonants", "lives_remaining", "n_guessed"]
    names += [f"guessed_{c}" for c in ALPHABET]
    names += [f"blank_at_{i}" for i in range(MAX_LEN)]
    names += [f"ngram_score_{c}" for c in ALPHABET]
    return names


def feature_names():
    return shared_feature_names() + ["ngram_score_self", "ngram_rank_self",
                                     "corpus_freq_self", "is_vowel_self"]


def build_shared(pattern: str, guessed, ngram_scores) -> np.ndarray:
    """Shared feature row for one game state. ngram_scores is a length-26 list."""
    guessed_set = set(guessed)
    revealed = set(pattern) - {"_"}
    eliminated = guessed_set - revealed

    n_blanks = pattern.count("_")
    L = len(pattern)
    n_rev_vowel = sum(1 for c in revealed if c in VOWELS)

    f = np.zeros(N_SHARED, dtype=np.float32)
    f[0] = L
    f[1] = n_blanks
    f[2] = (L - n_blanks) / L if L else 0.0
    f[3] = n_rev_vowel
    f[4] = len(revealed) - n_rev_vowel
    f[5] = MAX_WRONG - len(eliminated)
    f[6] = len(guessed_set)
    off = 7
    for c in guessed_set:
        f[off + ord(c) - 97] = 1.0
    off += 26
    for i, ch in enumerate(pattern[:MAX_LEN]):
        if ch == "_":
            f[off + i] = 1.0
    off += MAX_LEN
    f[off:off + 26] = ngram_scores
    return f


def per_letter_block(ngram_scores, corpus_freq) -> np.ndarray:
    """[26, 4] block of per-letter features, row i for letter i."""
    scores = np.asarray(ngram_scores, dtype=np.float32)
    # rank 0 = highest score
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(26, dtype=np.float32)
    rank[order] = np.arange(26, dtype=np.float32)
    block = np.zeros((26, N_PER_LETTER), dtype=np.float32)
    block[:, 0] = scores
    block[:, 1] = rank
    block[:, 2] = corpus_freq
    block[:, 3] = [1.0 if c in VOWELS else 0.0 for c in ALPHABET]
    return block


def corpus_frequencies(words) -> np.ndarray:
    """Fraction of words containing each letter."""
    counts = np.zeros(26, dtype=np.float32)
    for w in words:
        for c in set(w):
            counts[ord(c) - 97] += 1
    return counts / max(len(words), 1)


class CatBoostSolver:
    """26 trained boosters behind the standard solver interface."""

    def __init__(self, model_dir: str, ngram_model, corpus_freq: np.ndarray):
        from catboost import CatBoostClassifier
        self.models = []
        for i, c in enumerate(ALPHABET):
            m = CatBoostClassifier()
            m.load_model(f"{model_dir}/letter_{c}.cbm")
            self.models.append(m)
        self.ngram = ngram_model
        self.corpus_freq = corpus_freq

    def _rows(self, patterns, guessed_lists):
        """-> (shared[n_states, N_SHARED], alive[n_states, 26] bool)"""
        n = len(patterns)
        shared = np.zeros((n, N_SHARED), dtype=np.float32)
        alive = np.zeros((n, 26), dtype=bool)
        for k, (pat, guessed) in enumerate(zip(patterns, guessed_lists)):
            scores, dead = self.ngram.score_letters(pat, guessed)
            shared[k] = build_shared(pat, guessed, scores)
            alive[k] = ~np.asarray(dead, dtype=bool)
        return shared, alive

    def predict_batch(self, patterns, guessed_lists):
        shared, alive = self._rows(patterns, guessed_lists)
        n = len(patterns)
        probs = np.full((n, 26), -np.inf, dtype=np.float32)
        # One batched call per letter across all states -- 26 calls total,
        # rather than 26 calls per state.
        for i in range(26):
            rows = np.flatnonzero(alive[:, i])
            if rows.size == 0:
                continue
            feats = letter_features(shared[rows], i, self.corpus_freq)
            p = self.models[i].predict_proba(feats)[:, 1]
            probs[rows, i] = p
        out = []
        for k in range(n):
            j = int(np.argmax(probs[k]))
            if not np.isfinite(probs[k, j]):  # everything guessed; shouldn't happen
                j = int(np.argmax(alive[k]))
            out.append(ALPHABET[j])
        return out

    def predict(self, pattern: str, guessed: list) -> str:
        return self.predict_batch([pattern], [list(guessed)])[0]
