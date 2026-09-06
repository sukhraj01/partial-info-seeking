"""Solver-agnostic hangman benchmark.

Any solver is a callable: predict(pattern: str, guessed: list[str]) -> str
  pattern: current reveal state, e.g. "h_ll_" (underscore = blank)
  guessed: letters already guessed (right or wrong), no duplicates
  returns: next letter to guess (must not be in `guessed`)

Usage:
  from benchmark import run_benchmark, load_words
  run_benchmark(predict_fn, load_words("data/train_split.txt"), name="freq_baseline (in-vocab)")
"""
import time
from typing import Callable, List, Tuple

MAX_WRONG = 6
Predictor = Callable[[str, List[str]], str]


def load_words(path: str) -> List[str]:
    with open(path) as f:
        return [w.strip() for w in f if w.strip()]


def play_word(word: str, predict_fn: Predictor, max_wrong: int = MAX_WRONG) -> Tuple[bool, int]:
    guessed = []
    guessed_set = set()
    wrong = 0
    pattern = ["_"] * len(word)

    while wrong < max_wrong and "_" in pattern:
        guess = predict_fn("".join(pattern), guessed)
        if guess in guessed_set:
            # Buggy solver repeating a guess -- charge a wrong guess so it doesn't stall forever.
            wrong += 1
            continue
        guessed.append(guess)
        guessed_set.add(guess)
        if guess in word:
            for i, ch in enumerate(word):
                if ch == guess:
                    pattern[i] = ch
        else:
            wrong += 1

    won = "_" not in pattern
    return won, wrong


def run_benchmark(predict_fn: Predictor, words: List[str], name: str = "solver",
                   max_wrong: int = MAX_WRONG, verbose: bool = True) -> dict:
    start = time.time()
    wins = 0
    total_wrong_won = 0
    total_wrong_all = 0

    for word in words:
        won, wrong = play_word(word, predict_fn, max_wrong=max_wrong)
        if won:
            wins += 1
            total_wrong_won += wrong
        total_wrong_all += wrong

    elapsed = time.time() - start
    n = len(words)
    win_rate = wins / n * 100
    avg_wrong_won = total_wrong_won / wins if wins else float("nan")
    time_per_word_ms = elapsed / n * 1000

    result = {
        "name": name,
        "n_words": n,
        "wins": wins,
        "win_rate": win_rate,
        "avg_wrong_won": avg_wrong_won,
        "avg_wrong_all": total_wrong_all / n,
        "elapsed_s": elapsed,
        "time_per_word_ms": time_per_word_ms,
    }

    if verbose:
        print(f"[{name}] | win_rate: {win_rate:5.1f}% | avg_wrong(won): {avg_wrong_won:.2f} "
              f"| n: {n} | time: {elapsed:.1f}s ({time_per_word_ms:.2f}ms/word)")

    return result


def run_full_benchmark(predict_fn: Predictor, name: str,
                        train_split_path: str = "data/train_split.txt",
                        holdout_split_path: str = "data/holdout_split.txt",
                        sample: int = None) -> dict:
    """Report both in-vocab and disjoint win rate in the CLAUDE.md-mandated format."""
    train_words = load_words(train_split_path)
    holdout_words = load_words(holdout_split_path)

    if sample:
        import random
        random.seed(0)
        train_words = random.sample(train_words, min(sample, len(train_words)))
        holdout_words = random.sample(holdout_words, min(sample, len(holdout_words)))

    in_vocab = run_benchmark(predict_fn, train_words, name=f"{name} [in-vocab]", verbose=False)
    disjoint = run_benchmark(predict_fn, holdout_words, name=f"{name} [disjoint]", verbose=False)

    total_time = in_vocab["elapsed_s"] + disjoint["elapsed_s"]
    print(f"{name} | in-vocab: {in_vocab['win_rate']:5.1f}% | disjoint: {disjoint['win_rate']:5.1f}% "
          f"| avg_wrong: {disjoint['avg_wrong_won']:.2f} | time: {total_time/60:.1f}m")

    return {"in_vocab": in_vocab, "disjoint": disjoint}


if __name__ == "__main__":
    # Smoke test with a trivial always-vowel-then-alphabet solver.
    ORDER = list("etaoinshrdlcumwfgypbvkjxqz")

    def dummy_predict(pattern, guessed):
        for ch in ORDER:
            if ch not in guessed:
                return ch
        raise RuntimeError("exhausted alphabet")

    run_full_benchmark(dummy_predict, "dummy_smoke_test", sample=2000)
