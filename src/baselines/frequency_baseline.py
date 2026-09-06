"""Dumbest possible baseline: guess letters in overall corpus frequency order.

Ignores pattern length, position, and revealed letters entirely (beyond
skipping already-guessed letters). This is the floor every other approach
must beat.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from benchmark import load_words, run_full_benchmark


def build_letter_order(words):
    """Order letters by fraction of words containing them (not raw char count)."""
    counts = Counter()
    for w in words:
        counts.update(set(w))
    return [ch for ch, _ in counts.most_common()]


_LETTER_ORDER = build_letter_order(load_words("data/train_split.txt"))


def predict(pattern: str, guessed: list) -> str:
    guessed_set = set(guessed)
    for ch in _LETTER_ORDER:
        if ch not in guessed_set:
            return ch
    raise RuntimeError("exhausted alphabet without winning")


if __name__ == "__main__":
    print(f"letter order: {''.join(_LETTER_ORDER)}")
    run_full_benchmark(predict, "freq_baseline")
