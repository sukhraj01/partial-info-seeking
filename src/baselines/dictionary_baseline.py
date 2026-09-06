"""Reference baseline: dictionary candidate filtering + candidate letter frequency.

Keeps the set of dictionary words consistent with (a) word length, (b) revealed
letters at their positions, (c) absence of letters guessed-and-missed. Guesses the
letter appearing in the most surviving candidates.

This is the standard strong baseline. Its in-vocab number validates the harness
(CLAUDE.md: in-vocab < 90% means the filtering logic is buggy); its disjoint number
is the real bar any ML approach must beat, since it collapses when zero candidates
survive -- which is the normal case for unseen words.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter, defaultdict
from benchmark import load_words, run_full_benchmark


class DictionarySolver:
    def __init__(self, dictionary_path="data/train_split.txt"):
        words = load_words(dictionary_path)
        self.by_length = defaultdict(list)
        for w in words:
            self.by_length[len(w)].append(w)
        counts = Counter()
        for w in words:
            counts.update(set(w))
        self.global_order = [ch for ch, _ in counts.most_common()]
        self.no_candidate_games = 0
        self.total_guesses = 0

    def _candidates(self, pattern, guessed_set):
        # Letters guessed but not revealed in the pattern are known-absent.
        revealed = set(pattern) - {"_"}
        absent = guessed_set - revealed
        out = []
        for w in self.by_length[len(pattern)]:
            ok = True
            for i, pc in enumerate(pattern):
                wc = w[i]
                if pc == "_":
                    if wc in absent or wc in revealed:
                        ok = False
                        break
                elif wc != pc:
                    ok = False
                    break
            if ok:
                out.append(w)
        return out

    def predict(self, pattern: str, guessed: list) -> str:
        self.total_guesses += 1
        guessed_set = set(guessed)
        candidates = self._candidates(pattern, guessed_set)

        if candidates:
            counts = Counter()
            for w in candidates:
                counts.update(set(w) - guessed_set)
            if counts:
                return counts.most_common(1)[0][0]

        self.no_candidate_games += 1
        for ch in self.global_order:
            if ch not in guessed_set:
                return ch
        raise RuntimeError("exhausted alphabet")


if __name__ == "__main__":
    solver = DictionarySolver()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    run_full_benchmark(solver.predict, "dict_filter_baseline", sample=n)
    fallback_pct = solver.no_candidate_games / solver.total_guesses * 100
    print(f"  (fell back to global frequency on {fallback_pct:.1f}% of guesses)")
