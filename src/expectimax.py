"""Depth-1 expectimax letter selection.

Greedy play picks argmax P(letter present). That maximises the chance THIS guess
lands, which is not the same as maximising the chance the GAME is won -- with
five lives in hand an informative split can beat a safe letter, and with one
life left the ordering reverses. Barbay & Subercaseaux (2020) show greedy can be
arbitrarily far from optimal.

For each candidate letter L we evaluate both branches:

  miss  (prob 1 - p_L): EXACT. The successor state is the same pattern with L
        added to the eliminated set and one fewer life, so we re-run the
        transformer on it and score the result with the value function. This
        branch differs across letters because eliminating different letters
        reshapes the model's posterior differently -- a letter whose removal
        leaves the model confident is a cheaper gamble.

  hit   (prob p_L): APPROXIMATE. The successor pattern depends on WHERE L sits,
        and our model predicts word-level presence, not per-position identity,
        so the exact pattern is unavailable. We approximate it by the value of a
        state with the same lives and E[occurrences] fewer blanks.

  EV(L) = p_L * V_hit(L) + (1 - p_L) * V_miss(L)

Only the top-K letters by p_L are expanded; the rest cannot win the argmax and
each expansion costs a forward pass. K is the compute/quality dial.
"""
import numpy as np
import torch

from transformer_model import ALPHABET, encode_batch
from value_function import batch_features, expected_reveals, letter_multiplicity

PAD, BLANK, LETTER_BASE = 0, 1, 2


class ExpectimaxSolver:
    """Wraps a TransformerSolver, replacing argmax with depth-1 expectimax."""

    def __init__(self, base_solver, value_fn, multiplicity, top_k=5, batch_size=4096):
        self.base = base_solver
        self.value_fn = value_fn
        # E[count of L | L present]; makes the hit branch letter-dependent.
        self.multiplicity = np.asarray(multiplicity, dtype=np.float32)
        self.top_k = top_k
        self.batch_size = batch_size
        self.model = base_solver.model
        self.device = base_solver.device
        self.stats = {"decisions": 0, "changed": 0}

    @torch.no_grad()
    def _probs(self, patterns, guessed_lists):
        tokens, elim, gmask = encode_batch(patterns, guessed_lists, self.device)
        logits = self.model(tokens, elim).float()
        return torch.sigmoid(logits).cpu().numpy(), tokens.cpu().numpy(), \
            elim.cpu().numpy(), gmask.cpu().numpy()

    @torch.no_grad()
    def predict_batch(self, patterns, guessed_lists):
        out = []
        for i in range(0, len(patterns), self.batch_size):
            out.extend(self._decide(patterns[i:i + self.batch_size],
                                    guessed_lists[i:i + self.batch_size]))
        return out

    @torch.no_grad()
    def _decide(self, patterns, guessed_lists):
        B = len(patterns)
        probs, tokens, elim, gmask = self._probs(patterns, guessed_lists)

        word_lens = (tokens != PAD).sum(axis=1).astype(np.float32)
        n_blanks = (tokens == BLANK).sum(axis=1).astype(np.float32)
        n_elim = elim.sum(axis=1).astype(np.float32)
        n_guessed = gmask.sum(axis=1).astype(np.float32)

        masked = np.where(gmask, -1.0, probs)
        greedy = masked.argmax(axis=1)
        k = min(self.top_k, 26)
        cand = np.argsort(-masked, axis=1)[:, :k]          # [B, k]

        # --- hit branch: same lives, fewer blanks. Letter-dependent through
        # --- multiplicity: 'e' reveals more ground than 'q' when it lands.
        # After a hit, L is GUESSED: it must leave the candidate distribution,
        # or the value function scores the successor with L still contributing
        # its (large) probability mass. Entropy is the dominant feature, so
        # leaving L in distorts the hit value badly.
        rows = np.arange(B)
        hit_vals = np.zeros((B, k), dtype=np.float32)
        for j in range(k):
            mult = self.multiplicity[cand[:, j]]
            reveals = expected_reveals(n_blanks, word_lens, mult)
            hb = np.maximum(n_blanks - reveals, 0.0)
            p_hit = probs.copy()
            p_hit[rows, cand[:, j]] = 0.0
            gm_hit = gmask.copy()
            gm_hit[rows, cand[:, j]] = True
            f = batch_features(word_lens, hb, n_guessed + 1, n_elim, p_hit, gm_hit)
            hit_vals[:, j] = self.value_fn(f)

        # --- miss branch: exact successor, one forward pass per candidate ---
        miss_vals = np.zeros((B, k), dtype=np.float32)
        tok_t = torch.from_numpy(tokens).to(self.device)
        for j in range(k):
            el2 = elim.copy()
            el2[np.arange(B), cand[:, j]] = 1.0
            gm2 = gmask.copy()
            gm2[np.arange(B), cand[:, j]] = True
            el2_t = torch.from_numpy(el2).to(self.device)
            logits2 = self.model(tok_t, el2_t).float()
            probs2 = torch.sigmoid(logits2).cpu().numpy()
            f = batch_features(word_lens, n_blanks, n_guessed + 1,
                               n_elim + 1, probs2, gm2)
            miss_vals[:, j] = self.value_fn(f)

        p_cand = np.take_along_axis(probs, cand, axis=1)
        ev = p_cand * hit_vals + (1.0 - p_cand) * miss_vals
        # A letter already guessed must never win.
        ev = np.where(np.take_along_axis(gmask, cand, axis=1), -np.inf, ev)
        best = cand[np.arange(B), ev.argmax(axis=1)]

        self.stats["decisions"] += B
        self.stats["changed"] += int((best != greedy).sum())
        return [ALPHABET[i] for i in best]

    def predict(self, pattern: str, guessed: list) -> str:
        return self.predict_batch([pattern], [list(guessed)])[0]
