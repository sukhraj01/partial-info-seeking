"""PyTorch Dataset over the generated hangman states.

The arrays are memory-mapped, so a 200M-state set costs no resident RAM beyond
the pages actually touched -- workers page in what they read. Bitmasks are
unpacked to dense 26-dim float vectors in the collate step, which is where the
work belongs (cheap, parallel across DataLoader workers, keeps the on-disk
format compact).

Model input per example:
    tokens      int64[MAX_LEN]  0=PAD, 1=BLANK, 2..27 = 'a'..'z'
    eliminated  float32[26]     1.0 where the letter is guessed and known-absent
Target:
    target      float32[26]     1.0 where the letter is in the word but not revealed

Use with BCEWithLogitsLoss over the 26 outputs. At inference, mask out every
already-guessed letter before taking the argmax.
"""
import json
import os

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:  # encoding layer stays testable on machines without torch
    HAS_TORCH = False

    class Dataset:  # minimal stand-in so the class body still defines
        pass

MAX_LEN = 29
PAD, BLANK, LETTER_BASE = 0, 1, 2
VOCAB_SIZE = LETTER_BASE + 26  # 28

_BITS = np.arange(26, dtype=np.uint32)


def unpack_masks(masks: np.ndarray) -> np.ndarray:
    """uint32 bitmasks -> float32 [N, 26]."""
    return ((masks[:, None] >> _BITS[None, :]) & 1).astype(np.float32)


class HangmanStates(Dataset):
    """with_positions=True additionally returns per-BLANK letter targets.

    The pooled head predicts 26 numbers per state ("is letter L present"). A
    per-position head instead predicts a letter distribution at every blank,
    which is both a richer representation and ~L times more supervision from the
    same data. The true letter under each blank is not stored in the state
    arrays, but `word_idx` is -- so it is recovered from the source word list.
    """

    def __init__(self, state_dir="data/states", split=None, with_positions=False):
        with open(os.path.join(state_dir, "meta.json")) as f:
            self.meta = json.load(f)
        n = self.meta["n_states"]
        self.patterns = np.memmap(os.path.join(state_dir, "patterns.i8"),
                                  dtype=np.int8, mode="r", shape=(n, MAX_LEN))
        self.eliminated = np.memmap(os.path.join(state_dir, "eliminated.u32"),
                                    dtype=np.uint32, mode="r", shape=(n,))
        self.target = np.memmap(os.path.join(state_dir, "target.u32"),
                                dtype=np.uint32, mode="r", shape=(n,))
        self.word_idx = np.memmap(os.path.join(state_dir, "word_idx.u32"),
                                  dtype=np.uint32, mode="r", shape=(n,))
        # split: optional (start, end) row range, e.g. to hold out a val slice.
        self.start, self.end = split if split else (0, n)

        self.with_positions = with_positions
        self.word_letters = None
        if with_positions:
            # [n_words, MAX_LEN] int8 of letter indices (0-25), -1 past the word.
            words = [w.strip() for w in open(self.meta["source_words"]) if w.strip()]
            arr = np.full((len(words), MAX_LEN), -1, dtype=np.int8)
            for i, w in enumerate(words):
                for p_, ch in enumerate(w[:MAX_LEN]):
                    arr[i, p_] = ord(ch) - 97
            self.word_letters = arr

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, i):
        j = self.start + i
        return j  # indices only; batch_collate does the vectorised gather

    def batch_numpy(self, idxs):
        """Vectorised gather -> (tokens, eliminated, target) as numpy. Torch-free.

        With with_positions, also returns pos_tgt int64[N, MAX_LEN]: the letter
        index under each BLANK, and -100 elsewhere (torch's default CE
        ignore_index, so revealed and padded positions contribute no loss).
        """
        idxs = np.asarray(idxs, dtype=np.int64)
        idxs.sort()  # locality: memmap reads are much cheaper in order
        tokens = np.ascontiguousarray(self.patterns[idxs]).astype(np.int64)
        elim = unpack_masks(np.ascontiguousarray(self.eliminated[idxs]))
        tgt = unpack_masks(np.ascontiguousarray(self.target[idxs]))
        if not self.with_positions:
            return tokens, elim, tgt
        widx = np.ascontiguousarray(self.word_idx[idxs]).astype(np.int64)
        letters = self.word_letters[widx].astype(np.int64)   # [N, MAX_LEN]
        pos_tgt = np.where(tokens == BLANK, letters, -100)
        return tokens, elim, tgt, pos_tgt

    def batch_collate(self, idxs):
        out = self.batch_numpy(idxs)
        return tuple(torch.from_numpy(a) for a in out)


def make_loader(state_dir="data/states", batch_size=1024, split=None,
                num_workers=8, shuffle=True):
    ds = HangmanStates(state_dir, split=split)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=ds.batch_collate, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )


def encode_live_state_numpy(pattern: str, guessed):
    """Encode a live game state exactly as training states are encoded.

    Keeping this beside the Dataset is deliberate: any drift between the
    training encoding and the inference encoding is silent and fatal.
    """
    guessed_set = set(guessed)
    revealed = set(pattern) - {"_"}
    eliminated = guessed_set - revealed

    tokens = np.zeros(MAX_LEN, dtype=np.int64)
    for p, ch in enumerate(pattern):
        tokens[p] = BLANK if ch == "_" else LETTER_BASE + (ord(ch) - 97)
    elim = np.zeros(26, dtype=np.float32)
    for ch in eliminated:
        elim[ord(ch) - 97] = 1.0
    return tokens, elim


def encode_live_state(pattern: str, guessed):
    tokens, elim = encode_live_state_numpy(pattern, guessed)
    return torch.from_numpy(tokens), torch.from_numpy(elim)


if __name__ == "__main__":
    import sys
    state_dir = sys.argv[1] if len(sys.argv) > 1 else "data/states_small"
    ds = HangmanStates(state_dir)
    print(f"{len(ds):,} states ({ds.meta['states_per_word']} per word), torch={HAS_TORCH}")
    tokens, elim, tgt = ds.batch_numpy(list(range(8)))
    print("tokens", tokens.shape, "eliminated", elim.shape, "target", tgt.shape)
    print("row0 tokens:", tokens[0].tolist())
    print("row0 eliminated:", np.flatnonzero(elim[0]).tolist())
    print("row0 target:", np.flatnonzero(tgt[0]).tolist())

    # Round-trip: a live state derived from a training row must encode to the
    # exact same tokens/eliminated arrays the pipeline wrote for that row.
    import random
    words = [w.strip() for w in open(ds.meta["source_words"]) if w.strip()]
    alpha = "abcdefghijklmnopqrstuvwxyz"
    rng = random.Random(0)
    rows = [rng.randrange(len(ds)) for _ in range(20000)]
    tok_b, elim_b, tgt_b = ds.batch_numpy(rows)
    sorted_rows = sorted(rows)
    bad = 0
    for r, tk, el in zip(sorted_rows, tok_b, elim_b):
        word = words[int(ds.word_idx[r])]
        pattern = "".join("_" if tk[p] == BLANK else alpha[tk[p] - LETTER_BASE]
                          for p in range(len(word)))
        guessed = [alpha[i] for i in np.flatnonzero(el)] + sorted(set(pattern) - {"_"})
        tk2, el2 = encode_live_state_numpy(pattern, guessed)
        if not (np.array_equal(tk2, tk) and np.array_equal(el2, el)):
            bad += 1
    print(f"encode round-trip over {len(rows):,} rows: {bad} mismatches "
          f"({'ok' if bad == 0 else 'FAIL'})")
