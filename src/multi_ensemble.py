"""Blend an arbitrary set of letter-presence predictors.

`ensemble_model.EnsembleSolver` hardcodes one transformer plus the n-gram. This
generalises it so a PRETRAINED character model (CANINE) can join the mix.

Why this is worth doing even if CANINE scores below the incumbent: every pair in
our comparison table sits at phi >= 0.68, because they are all the same
from-scratch architecture trained on overlapping data. CANINE is a different
architecture with different pretraining, so its errors should be far less
correlated -- and low correlation between members is exactly the condition under
which an ensemble beats its best member. A 67% CANINE can still lift a 69%
blend; a second 69% transformer would not.

Every member exposes the same quantity: P(letter is in the word and not yet
revealed), as a [N, 26] array with already-guessed letters zeroed. That shared
definition is what makes a weighted average meaningful rather than a hack.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ngram_model import ALPHABET, NgramModel  # noqa: E402
from transformer_model import HangmanTransformer, encode_batch  # noqa: E402

EPS = 1e-6


class _TransformerMember:
    """Our from-scratch transformer."""

    def __init__(self, checkpoint, device, batch_size=4096):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = HangmanTransformer(**ck["config"])
        self.model.load_state_dict(ck["model"])
        self.model.eval().to(device)
        self.device, self.batch_size = device, batch_size
        self.name = os.path.basename(checkpoint)

    @torch.no_grad()
    def probs(self, patterns, guessed):
        out_p, out_g = [], []
        for i in range(0, len(patterns), self.batch_size):
            tk, el, gm = encode_batch(patterns[i:i + self.batch_size],
                                      guessed[i:i + self.batch_size], self.device)
            p = torch.sigmoid(self.model(tk, el).float()).masked_fill(gm, 0.0)
            out_p.append(p.cpu().numpy())
            out_g.append(gm.cpu().numpy())
        return np.concatenate(out_p), np.concatenate(out_g)


class _PretrainedMember:
    """A fine-tuned pretrained character encoder (train_pretrained.py)."""

    def __init__(self, checkpoint, device, batch_size=2048):
        # 2048 fits inference-only on a 16 GB T4 and roughly halves the number of
        # kernel launches vs 1024. CANINE is the slowest member by far, so this
        # is the one knob that matters for inference wall time.
        from train_pretrained import PretrainedHangman
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = PretrainedHangman(ck["model_name"], freeze_layers=0)
        self.model.load_state_dict(ck["model"])
        self.model.eval().to(device)
        self.device, self.batch_size = device, batch_size
        self.name = os.path.basename(checkpoint)

    @torch.no_grad()
    def probs(self, patterns, guessed):
        out_p, out_g = [], []
        for i in range(0, len(patterns), self.batch_size):
            tk, el, gm = encode_batch(patterns[i:i + self.batch_size],
                                      guessed[i:i + self.batch_size], self.device)
            p = torch.sigmoid(self.model(tk, el).float()).masked_fill(gm, 0.0)
            out_p.append(p.cpu().numpy())
            out_g.append(gm.cpu().numpy())
        return np.concatenate(out_p), np.concatenate(out_g)


class _PositionalMember:
    """Per-position (MLM) transformer, exposing the same presence probability.

    The model predicts a letter distribution at every blank; P(letter present)
    is the same aggregation the n-gram uses, so it drops into the blend on the
    identical scale as every other member.
    """

    def __init__(self, checkpoint, device, batch_size=4096):
        from transformer_positional import PositionalHangmanTransformer
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = PositionalHangmanTransformer(**ck["config"])
        self.model.load_state_dict(ck["model"])
        self.model.eval().to(device)
        self.device, self.batch_size = device, batch_size
        self.name = os.path.basename(checkpoint)

    @torch.no_grad()
    def probs(self, patterns, guessed):
        out_p, out_g = [], []
        for i in range(0, len(patterns), self.batch_size):
            tk, el, gm = encode_batch(patterns[i:i + self.batch_size],
                                      guessed[i:i + self.batch_size], self.device)
            log_absent = self.model.presence_logprob(self.model(tk, el), tk)
            p = (1.0 - log_absent.exp()).clamp(0.0, 1.0).masked_fill(gm, 0.0)
            out_p.append(p.float().cpu().numpy())
            out_g.append(gm.cpu().numpy())
        return np.concatenate(out_p), np.concatenate(out_g)


class _NgramMember:
    def __init__(self, ngram):
        self.ngram = ngram
        self.name = "ngram"

    def probs(self, patterns, guessed):
        p = np.zeros((len(patterns), 26), dtype=np.float32)
        score = self.ngram.score_letters
        for i, (pat, g) in enumerate(zip(patterns, guessed)):
            p[i] = score(pat, g)[0]
        return p, None


class MultiEnsemble:
    """Weighted blend over any number of members.

    specs: list of (kind, path_or_None, weight); kind in {tfm, pos, pre, ngram}.
    Weights are normalised. `log_space=True` pools log-odds (product of experts)
    instead of probabilities.
    """

    def __init__(self, specs, dictionary="data/train_split.txt", device=None,
                 ngram=None, log_space=False, pre_batch=2048):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.log_space = log_space
        self.members, w = [], []
        for kind, path, weight in specs:
            if kind == "tfm":
                self.members.append(_TransformerMember(path, self.device))
            elif kind == "pos":
                self.members.append(_PositionalMember(path, self.device))
            elif kind == "pre":
                self.members.append(_PretrainedMember(path, self.device,
                                                      batch_size=pre_batch))
            elif kind == "ngram":
                self.members.append(_NgramMember(ngram or NgramModel(dictionary)))
            else:
                raise ValueError(f"unknown member kind {kind!r}")
            w.append(float(weight))
        w = np.array(w, dtype=np.float64)
        if w.sum() <= 0:
            raise ValueError("weights must sum to > 0")
        self.weights = w / w.sum()

    def set_weights(self, weights):
        w = np.asarray(weights, dtype=np.float64)
        self.weights = w / w.sum()

    def predict_batch(self, patterns, guessed_lists):
        acc, gmask = None, None
        for m, w in zip(self.members, self.weights):
            if w == 0.0:
                continue
            p, gm = m.probs(patterns, guessed_lists)
            if gm is not None and gmask is None:
                gmask = gm
            contrib = (np.log(np.clip(p, EPS, 1 - EPS) /
                              (1 - np.clip(p, EPS, 1 - EPS))) if self.log_space else p)
            acc = contrib * w if acc is None else acc + contrib * w
        if gmask is None:  # n-gram-only mixtures still need a guessed mask
            gmask = np.zeros_like(acc, dtype=bool)
            for i, g in enumerate(guessed_lists):
                for ch in g:
                    gmask[i, ord(ch) - 97] = True
        pick = np.where(gmask, -np.inf, acc).argmax(axis=1)
        return [ALPHABET[j] for j in pick]

    def predict(self, pattern, guessed):
        return self.predict_batch([pattern], [list(guessed)])[0]
