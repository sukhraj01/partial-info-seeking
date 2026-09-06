"""N-gram positional model with two-sided context templates and count recalibration.

Design notes
------------
Words are padded with '^' (start) and '$' (end) so word-boundary context is a
first-class part of every key. This is what makes suffixes free: the template
R3 fired at the 't' of "-tion" looks up the key "R|ion" and, at the end of a
word, "R|on$" / "R|n$$" -- the '$' sentinels encode distance-from-word-end
without a separate suffix table.

Context is drawn from BOTH sides via explicit offset templates rather than only
left-to-right n-grams. Pinning a blank between known characters ("t?on") is far
more informative for hangman than left context alone, because the reveal order
is chosen by the solver, not by reading order.

Recalibration: at scoring time, eliminated and already-revealed letters are
zeroed out of each count vector and the remainder is RENORMALISED, so their
probability mass is redistributed over the letters still in play. This is
different from filtering the candidate list -- it changes the relative ranking
of the survivors, not just which ones are eligible.
"""
import os
import sys
from array import array
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import load_words, run_full_benchmark

PAD_L, PAD_R = 4, 4
START, END, BLANK = "^", "$", "_"
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
IDX = {c: i for i, c in enumerate(ALPHABET)}

# (name, offsets relative to the hole). Ordered specific -> general; the weight
# is how much a firing template contributes to the blended distribution.
TEMPLATES = [
    ("M4", (-2, -1, 1, 2), 24.0),
    ("L4", (-4, -3, -2, -1), 16.0),
    ("R4", (1, 2, 3, 4), 16.0),
    ("M3a", (-2, -1, 1), 12.0),
    ("M3b", (-1, 1, 2), 12.0),
    ("L3", (-3, -2, -1), 8.0),
    ("R3", (1, 2, 3), 8.0),
    ("M2", (-1, 1), 6.0),
    ("L2", (-2, -1), 3.0),
    ("R2", (1, 2), 3.0),
    ("L1", (-1,), 1.0),
    ("R1", (1,), 1.0),
]


class NgramModel:
    def __init__(self, dictionary_path="data/train_split.txt", min_count=1):
        self.tables = {name: {} for name, _, _ in TEMPLATES}
        self.min_count = min_count
        self.unigram_by_len = defaultdict(lambda: array("I", [0] * 26))
        self.global_unigram = array("I", [0] * 26)
        self._build(load_words(dictionary_path))

    def _build(self, words):
        tables = self.tables
        for w in words:
            padded = START * PAD_L + w + END * PAD_R
            L = len(w)
            ulen = self.unigram_by_len[L]
            for ch in set(w):
                ulen[IDX[ch]] += 1
                self.global_unigram[IDX[ch]] += 1
            for pos in range(L):
                p = pos + PAD_L  # index into padded
                letter = IDX[padded[p]]
                for name, offsets, _ in TEMPLATES:
                    ctx = "".join(padded[p + o] for o in offsets)
                    tbl = tables[name]
                    vec = tbl.get(ctx)
                    if vec is None:
                        vec = array("I", [0] * 26)
                        tbl[ctx] = vec
                    vec[letter] += 1
        # freeze default dict so lookups on unseen lengths don't mutate it
        self.unigram_by_len = dict(self.unigram_by_len)

    def _length_prior(self, length):
        return self.unigram_by_len.get(length, self.global_unigram)

    def score_letters(self, pattern: str, guessed: list):
        """-> (scores[26], dead[26]).

        scores[i] is P(letter i appears somewhere in the word), estimated as
        1 - prod over blanks of (1 - p_blank(i)). Dead letters score 0.
        Exposed separately from predict() so the CatBoost feature extractor can
        consume the full distribution rather than just the argmax.
        """
        guessed_set = set(guessed)
        # Anything already guessed (revealed or eliminated) is off the table.
        dead = [False] * 26
        for ch in guessed_set:
            dead[IDX[ch]] = True

        padded = START * PAD_L + pattern + END * PAD_R
        # not_present[l] = prod over blanks of (1 - p_i(l))
        not_present = [1.0] * 26
        any_blank = False

        for pos, pc in enumerate(pattern):
            if pc != BLANK:
                continue
            any_blank = True
            p = pos + PAD_L
            blended = [0.0] * 26
            weight_sum = 0.0

            for name, offsets, weight in TEMPLATES:
                ctx_chars = []
                ok = True
                for o in offsets:
                    c = padded[p + o]
                    if c == BLANK:
                        ok = False
                        break
                    ctx_chars.append(c)
                if not ok:
                    continue
                vec = self.tables[name].get("".join(ctx_chars))
                if vec is None:
                    continue
                # Recalibrate: drop dead letters, renormalise over survivors.
                total = 0
                for i in range(26):
                    if not dead[i]:
                        total += vec[i]
                if total < self.min_count:
                    continue
                inv = weight / total
                for i in range(26):
                    if not dead[i] and vec[i]:
                        blended[i] += vec[i] * inv
                weight_sum += weight

            if weight_sum == 0.0:
                prior = self._length_prior(len(pattern))
                total = sum(prior[i] for i in range(26) if not dead[i])
                if total == 0:
                    continue
                for i in range(26):
                    if not dead[i] and prior[i]:
                        blended[i] = prior[i] / total
                weight_sum = 1.0

            inv_w = 1.0 / weight_sum
            for i in range(26):
                if blended[i]:
                    not_present[i] *= 1.0 - min(blended[i] * inv_w, 1.0)

        if not any_blank:  # shouldn't happen; game would be over
            not_present = [0.0] * 26

        scores = [0.0 if dead[i] else 1.0 - not_present[i] for i in range(26)]
        return scores, dead

    def predict(self, pattern: str, guessed: list) -> str:
        scores, dead = self.score_letters(pattern, guessed)
        best, best_score = -1, -1.0
        for i in range(26):
            if dead[i]:
                continue
            if scores[i] > best_score:
                best, best_score = i, scores[i]

        if best < 0:  # everything guessed already
            for i in range(26):
                if not dead[i]:
                    return ALPHABET[i]
            raise RuntimeError("alphabet exhausted")
        return ALPHABET[best]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    import time
    t0 = time.time()
    model = NgramModel()
    n_keys = sum(len(t) for t in model.tables.values())
    print(f"built {n_keys:,} context keys in {time.time()-t0:.1f}s")
    run_full_benchmark(model.predict, "ngram_backoff", sample=n)
