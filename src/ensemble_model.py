"""Ensemble of the transformer MLM and the n-gram backoff model.

Both models estimate the SAME quantity -- P(letter L is in the word and not yet
revealed) -- so their outputs are directly comparable without recalibration:
the transformer via sigmoid(logit), the n-gram via 1 - prod_blanks(1 - p_i).
That is what makes a plain weighted average well-founded here rather than a
hack. They are also architecturally unrelated (attention over the whole pattern
vs. fixed two-sided context templates), so their errors are only loosely
correlated -- the precondition for an ensemble to beat its best member.

Three combination modes:
  blend   p = w*p_tfm + (1-w)*p_ngram, on raw presence probabilities
  nblend  the same, after renormalising each model over the unguessed letters
          (removes any difference in how the two saturate on long words)
  lblend  weighted average in LOG-ODDS space -- a product of experts rather than
          a mixture. Linear pooling is dominated by whichever model is closer to
          0 or 1; log-odds pooling instead lets a confident model veto, which is
          the better rule when both outputs are probabilities of the same binary
          event. Different enough from `blend` to be worth its own family.
  route   transformer by default; fall back to the n-gram whenever the
          transformer's top presence probability is below a threshold

Every mode masks already-guessed letters before the argmax, so no mode can ever
repeat a guess.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ngram_model import ALPHABET, NgramModel  # noqa: E402
from transformer_model import HangmanTransformer, encode_batch  # noqa: E402

MODES = ("blend", "nblend", "lblend", "route")
EPS = 1e-6  # keeps logit() finite at p = 0 or 1


class EnsembleSolver:
    """Standard solver interface over a transformer checkpoint + an n-gram model.

    The n-gram tables take ~1 min to build and ~1 GB, so a caller sweeping many
    variants should build ONE NgramModel and pass it in via `ngram=`.
    """

    def __init__(self, checkpoint_path: str, dictionary_path: str = "data/train_split.txt",
                 mode: str = "blend", weight: float = 0.8, threshold: float = 0.5,
                 device: str = None, batch_size: int = 4096, ngram: NgramModel = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode, self.weight, self.threshold = mode, weight, threshold
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = HangmanTransformer(**ckpt["config"])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval().to(self.device)
        self.meta = {k: v for k, v in ckpt.items() if k != "model"}

        self.ngram = ngram if ngram is not None else NgramModel(dictionary_path)
        self.reset_stats()

    def reset_stats(self):
        # How often the two models disagree, and how often `route` actually
        # falls back. If the fallback rate is ~0 the threshold sweep is inert,
        # and a flat result means nothing -- this is the guard against that.
        self.n_plies = 0
        self.n_routed = 0
        self.n_changed = 0

    def stats(self):
        n = max(self.n_plies, 1)
        return {"plies": self.n_plies,
                "routed_frac": self.n_routed / n,
                "changed_vs_transformer_frac": self.n_changed / n}

    @torch.no_grad()
    def _transformer_probs(self, patterns, guessed_lists):
        """-> (probs float32[N,26], guessed_mask bool[N,26]); guessed letters zeroed."""
        out_p, out_g = [], []
        for i in range(0, len(patterns), self.batch_size):
            tokens, elim, gmask = encode_batch(patterns[i:i + self.batch_size],
                                               guessed_lists[i:i + self.batch_size],
                                               self.device)
            p = torch.sigmoid(self.model(tokens, elim).float())
            p = p.masked_fill(gmask, 0.0)
            out_p.append(p.cpu().numpy())
            out_g.append(gmask.cpu().numpy())
        return np.concatenate(out_p), np.concatenate(out_g)

    def _ngram_probs(self, patterns, guessed_lists):
        p = np.zeros((len(patterns), 26), dtype=np.float32)
        score = self.ngram.score_letters
        for i, (pat, g) in enumerate(zip(patterns, guessed_lists)):
            s, _dead = score(pat, g)
            p[i] = s  # score_letters already zeroes guessed letters
        return p

    @staticmethod
    def _logit(p):
        q = np.clip(p, EPS, 1.0 - EPS)
        return np.log(q / (1.0 - q))

    @staticmethod
    def _renorm(p, gmask):
        """Renormalise each row over the letters still in play."""
        q = np.where(gmask, 0.0, p)
        tot = q.sum(axis=1, keepdims=True)
        return np.divide(q, tot, out=np.zeros_like(q), where=tot > 0)

    def predict_batch(self, patterns, guessed_lists):
        p_t, gmask = self._transformer_probs(patterns, guessed_lists)
        p_n = self._ngram_probs(patterns, guessed_lists)

        tfm_pick = np.where(gmask, -1.0, p_t).argmax(axis=1)

        if self.mode == "route":
            conf = np.take_along_axis(p_t, tfm_pick[:, None], axis=1).ravel()
            use_ngram = conf < self.threshold
            ng_pick = np.where(gmask, -1.0, p_n).argmax(axis=1)
            pick = np.where(use_ngram, ng_pick, tfm_pick)
            self.n_routed += int(use_ngram.sum())
        elif self.mode == "lblend":
            # sigmoid is monotone, so the argmax of the pooled log-odds is the
            # argmax of the pooled probability -- no need to map back.
            blended = (self.weight * self._logit(p_t)
                       + (1.0 - self.weight) * self._logit(p_n))
            pick = np.where(gmask, -np.inf, blended).argmax(axis=1)
        else:
            if self.mode == "nblend":
                p_t_c, p_n_c = self._renorm(p_t, gmask), self._renorm(p_n, gmask)
            else:
                p_t_c, p_n_c = p_t, p_n
            blended = self.weight * p_t_c + (1.0 - self.weight) * p_n_c
            pick = np.where(gmask, -1.0, blended).argmax(axis=1)

        self.n_plies += len(patterns)
        self.n_changed += int((pick != tfm_pick).sum())
        return [ALPHABET[j] for j in pick]

    def predict(self, pattern: str, guessed) -> str:
        return self.predict_batch([pattern], [list(guessed)])[0]


if __name__ == "__main__":
    # Local smoke test: tiny dictionary, random-init transformer, no GPU needed.
    import tempfile
    words = ["hello", "world", "python", "hangman", "letter", "banana", "silent"]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(words) + "\n")
        dict_path = f.name
    ck = os.path.join(tempfile.gettempdir(), "_ens_smoke.pt")
    m = HangmanTransformer(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    torch.save({"model": m.state_dict(),
                "config": dict(d_model=64, nhead=4, num_layers=2,
                               dim_feedforward=128, dropout=0.1)}, ck)

    ng = NgramModel(dict_path)
    pats = ["h_ll_", "_o__d", "_____"]
    gs = [["h", "l", "a"], ["o", "d", "e"], []]
    for mode, kw in [("blend", dict(weight=0.8)), ("nblend", dict(weight=0.8)),
                     ("lblend", dict(weight=0.8)), ("route", dict(threshold=0.5))]:
        s = EnsembleSolver(ck, dict_path, mode=mode, device="cpu", ngram=ng, **kw)
        picks = s.predict_batch(pats, gs)
        assert all(p not in g for p, g in zip(picks, gs)), f"{mode} repeated a guess"
        assert picks == [s.predict(p, g) for p, g in zip(pats, gs)], \
            f"{mode}: batched and scalar paths disagree"
        print(f"{mode:7s} picks={picks} stats={s.stats()}")
    print("ensemble smoke test: ok")
