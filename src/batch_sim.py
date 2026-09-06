"""Batched hangman simulation.

benchmark.py plays one word at a time, which is correct but calls the model once
per guess. For a GPU model that means ~600K single-row forward passes to cover
the disjoint split -- kernel-launch bound and hours long. Here every game in a
chunk advances in lockstep so the model sees one large batch per ply.

A solver supplies:
    predict_batch(patterns: list[str], guessed: list[list[str]]) -> list[str]

run_batch_benchmark returns the same dict shape as benchmark.run_benchmark, and
verify_agreement() checks a batched solver against its own scalar predict() so
the fast path can't silently diverge from the reference one.
"""
import time
from typing import Callable, List

MAX_WRONG = 6


def run_batch_benchmark(predict_batch: Callable[[List[str], List[List[str]]], List[str]],
                        words: List[str], name: str = "solver",
                        max_wrong: int = MAX_WRONG, chunk_size: int = 4096,
                        verbose: bool = True, progress_every: int = 0,
                        per_word: bool = False) -> dict:
    """per_word=True also returns a bool list aligned to `words`.

    Two solvers differing by half a point on 45K words differ by ~200 games, and
    the unpaired standard error is ~0.22pp -- the same order as the effect. The
    comparison is paired (same words, same rules), so the per-word outcomes are
    what make a McNemar test possible and the difference interpretable.
    """
    start = time.time()
    wins = 0
    total_wrong_won = 0
    total_wrong_all = 0
    done_words = 0
    won_flags = [] if per_word else None

    for c0 in range(0, len(words), chunk_size):
        chunk = words[c0:c0 + chunk_size]
        patterns = [["_"] * len(w) for w in chunk]
        guessed = [[] for _ in chunk]
        guessed_sets = [set() for _ in chunk]
        wrong = [0] * len(chunk)
        active = [i for i in range(len(chunk)) if "_" in patterns[i]]

        while active:
            req_pat = ["".join(patterns[i]) for i in active]
            req_guess = [guessed[i] for i in active]
            letters = predict_batch(req_pat, req_guess)
            if len(letters) != len(active):
                raise RuntimeError(
                    f"predict_batch returned {len(letters)} letters for {len(active)} states")

            still = []
            for i, g in zip(active, letters):
                if g in guessed_sets[i]:
                    # Repeated guess: charge a miss so a buggy solver can't stall.
                    wrong[i] += 1
                else:
                    guessed[i].append(g)
                    guessed_sets[i].add(g)
                    word = chunk[i]
                    if g in word:
                        pat = patterns[i]
                        for p, ch in enumerate(word):
                            if ch == g:
                                pat[p] = ch
                    else:
                        wrong[i] += 1
                if wrong[i] < max_wrong and "_" in patterns[i]:
                    still.append(i)
            active = still

        for i, w in enumerate(chunk):
            won = "_" not in patterns[i]
            if won:
                wins += 1
                total_wrong_won += wrong[i]
            total_wrong_all += wrong[i]
            if per_word:
                won_flags.append(won)

        done_words += len(chunk)
        if progress_every and done_words % progress_every < chunk_size:
            el = time.time() - start
            rate = done_words / el if el else 0
            eta = (len(words) - done_words) / rate / 60 if rate else 0
            print(f"  [{name}] {done_words:,}/{len(words):,} words "
                  f"| win {wins/done_words*100:.1f}% | {rate:.0f} w/s | eta {eta:.1f}m",
                  flush=True)

    elapsed = time.time() - start
    n = len(words)
    result = {
        "name": name,
        "n_words": n,
        "wins": wins,
        "win_rate": wins / n * 100,
        "avg_wrong_won": total_wrong_won / wins if wins else float("nan"),
        "avg_wrong_all": total_wrong_all / n,
        "elapsed_s": elapsed,
        "time_per_word_ms": elapsed / n * 1000,
    }
    if per_word:
        result["won_flags"] = won_flags
    if verbose:
        print(f"[{name}] | win_rate: {result['win_rate']:5.1f}% "
              f"| avg_wrong(won): {result['avg_wrong_won']:.2f} "
              f"| n: {n} | time: {elapsed:.1f}s ({result['time_per_word_ms']:.2f}ms/word)")
    return result


def verify_agreement(solver, words: List[str], max_wrong: int = MAX_WRONG) -> bool:
    """Assert the batched path and the scalar predict() choose identical letters.

    Replays each word with the scalar solver, and at every ply asks the batched
    path for the same state. Any divergence means the two paths would produce
    different win rates, which would make our benchmarks meaningless.
    """
    mismatches = 0
    checked = 0
    for word in words:
        pattern = ["_"] * len(word)
        guessed, gset, wrong = [], set(), 0
        while wrong < max_wrong and "_" in pattern:
            pat = "".join(pattern)
            scalar = solver.predict(pat, list(guessed))
            batched = solver.predict_batch([pat], [list(guessed)])[0]
            checked += 1
            if scalar != batched:
                mismatches += 1
                if mismatches <= 3:
                    print(f"  MISMATCH {word!r} pattern={pat!r} guessed={guessed} "
                          f"scalar={scalar!r} batched={batched!r}")
            if scalar in gset:
                wrong += 1
                continue
            guessed.append(scalar)
            gset.add(scalar)
            if scalar in word:
                for p, ch in enumerate(word):
                    if ch == scalar:
                        pattern[p] = ch
            else:
                wrong += 1
    ok = mismatches == 0
    print(f"  agreement: {checked - mismatches:,}/{checked:,} plies match "
          f"({'ok' if ok else 'FAIL'})")
    return ok
