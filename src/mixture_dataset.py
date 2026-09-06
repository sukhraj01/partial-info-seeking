"""Sample training batches from several state pools at a fixed ratio.

DAgger needs 50/50 (or any ratio) between round-0 states and on-policy states,
but the pools are wildly different sizes -- 50M round-0 vs ~2M per on-policy
round. Concatenating them would drown the on-policy states at 4%, which is the
opposite of the intent.

So the ratio is enforced by SAMPLING, not by concatenation: each epoch draws
`weight * epoch_size` rows from each pool (with replacement for pools smaller
than their quota). On-policy states repeat within an epoch; that is deliberate.

A (source, row) pair is packed into one int64 so the whole thing still works
with a plain DataLoader over an index array.
"""
import numpy as np

from dataset import HangmanStates, unpack_masks

SRC_SHIFT = 40  # rows per pool must stay under 2^40 (~1.1e12)


class MixtureStates:
    @staticmethod
    def _layout_key(src):
        """Identifies pools whose rows refer to the same words in the same order."""
        import hashlib
        m = src.meta
        try:
            with open(m["source_words"], "rb") as f:
                h = hashlib.sha1(f.read()).hexdigest()
        except OSError:
            h = None
        return (h, m.get("n_words"), m.get("states_per_word"))

    def __init__(self, specs, val_source=0, val_rows=None):
        """specs: list of (state_dir, weight). Weights are normalised."""
        self.dirs = [d for d, _ in specs]
        # Guard: distinct pools must be distinct paths. A shell bug once
        # collapsed all three onto one directory, which silently trained on
        # 100% round-0 data while reporting a DAgger mixture.
        import os as _os
        real = [_os.path.realpath(d).rstrip("/") for d in self.dirs]
        if len(set(real)) != len(real):
            raise ValueError(
                "mixture pools must be distinct directories, got:\n  "
                + "\n  ".join(real))
        self.sources = [HangmanStates(d) for d, _ in specs]
        w = np.array([float(x) for _, x in specs], dtype=np.float64)
        if w.sum() <= 0:
            raise ValueError("weights must sum to > 0")
        self.weights = w / w.sum()
        # Rows excluded from sampling (the validation slice of one pool), so a
        # held-out word never leaks into training through the mixture.
        self.val_source = val_source
        self.val_rows = val_rows  # (start, end) on val_source, or None

        # A pool generated from the SAME word list with the SAME states-per-word
        # is laid out word-major identically, so the held-out words sit in the
        # same row range there too. The two 50M pools are exactly this
        # case: excluding the val slice from pool 0 alone would let the second
        # pool feed those words straight back in, quietly invalidating the
        # per-epoch win-rate the checkpoint is selected on. Pools with a
        # different layout (on-policy DAgger pools) are left alone.
        self.val_excluded = set()
        if val_rows is not None:
            key0 = self._layout_key(self.sources[val_source])
            for si, src in enumerate(self.sources):
                if self._layout_key(src) == key0 and len(src.patterns) >= val_rows[1]:
                    self.val_excluded.add(si)
            shared = sorted(self.val_excluded - {val_source})
            if shared:
                print(f"MixtureStates: val rows {val_rows} also excluded from pools "
                      f"{shared} (identical word layout)", flush=True)

    def pool_sizes(self):
        return [len(s.patterns) for s in self.sources]

    def build_epoch_index(self, epoch_size, seed=0):
        rng = np.random.default_rng(seed)
        parts = []
        for si, (src, w) in enumerate(zip(self.sources, self.weights)):
            n = len(src.patterns)
            quota = int(round(epoch_size * w))
            if quota == 0:
                continue
            lo, hi = 0, n
            if self.val_rows is not None and si in self.val_excluded:
                lo, hi = 0, self.val_rows[0]  # train portion only
            span = hi - lo
            if span <= 0:
                continue
            if quota <= span:
                rows = rng.choice(span, size=quota, replace=False) + lo
            else:
                rows = rng.integers(lo, hi, size=quota)
            parts.append((np.int64(si) << SRC_SHIFT) | rows.astype(np.int64))
        idx = np.concatenate(parts)
        rng.shuffle(idx)
        return idx

    def collate(self, packed):
        """Decode packed ids, gather per source, return torch tensors."""
        import torch
        packed = np.asarray(packed, dtype=np.int64)
        src = (packed >> SRC_SHIFT).astype(np.int32)
        row = (packed & ((np.int64(1) << SRC_SHIFT) - 1)).astype(np.int64)

        n = packed.shape[0]
        tokens = np.zeros((n, 29), dtype=np.int64)
        elim = np.zeros((n, 26), dtype=np.float32)
        tgt = np.zeros((n, 26), dtype=np.float32)

        for si in np.unique(src):
            sel = np.flatnonzero(src == si)
            rows = row[sel]
            order = np.argsort(rows)          # memmap reads are cheaper in order
            sel_o, rows_o = sel[order], rows[order]
            s = self.sources[si]
            tokens[sel_o] = np.ascontiguousarray(s.patterns[rows_o]).astype(np.int64)
            elim[sel_o] = unpack_masks(np.ascontiguousarray(s.eliminated[rows_o]))
            tgt[sel_o] = unpack_masks(np.ascontiguousarray(s.target[rows_o]))

        return (torch.from_numpy(tokens), torch.from_numpy(elim), torch.from_numpy(tgt))


class IndexDataset:
    """Trivial Dataset over a precomputed int64 index array."""

    def __init__(self, index):
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        return int(self.index[i])


def parse_spec(s):
    """'dir:weight' or 'dir' (weight 1.0)."""
    if ":" in s:
        d, w = s.rsplit(":", 1)
        return d, float(w)
    return s, 1.0
