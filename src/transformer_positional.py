"""Per-position (true MLM) variant of the hangman transformer.

The pooled model collapses the encoder output to one vector and predicts 26
numbers: "is letter L somewhere in this word". That discards position entirely --
it cannot represent "slot 1 is probably 'e', slot 4 is probably 'o'".

Our own n-gram, which scores 55.59% with nothing but counting, does the opposite:
it scores each blank separately and aggregates

    P(letter present) = 1 - prod over blanks (1 - p_blank(letter))

This model gives the transformer the same structure, and with it two things the
pooled version cannot have:

1. Representation -- per-slot beliefs instead of one average.
2. Supervision -- with per-position targets the loss sees 26 x (number of blanks)
   values per state rather than 26. For a 9-letter word with 5 blanks that is
   roughly 5x the gradient signal from identical data, which matters on a task
   we measured to be data-limited rather than capacity-limited.

Inference aggregates exactly as the n-gram does, then masks already-guessed
letters before the argmax, so a repeated guess remains impossible.

Kept in its own module so the pooled model used by the submitted checkpoints is
untouched.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformer_model import (ALPHABET, BLANK, MAX_LEN, PAD,  # noqa: E402
                               VOCAB_SIZE, encode_batch, sinusoidal_encoding)

NEG_INF = float("-inf")


class PositionalHangmanTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024,
                 dropout=0.1, max_len=MAX_LEN, head_hidden=512):
        super().__init__()
        self.config = dict(d_model=d_model, nhead=nhead, num_layers=num_layers,
                           dim_feedforward=dim_feedforward, dropout=dropout,
                           max_len=max_len, head_hidden=head_hidden)
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.register_buffer("pos_emb", sinusoidal_encoding(max_len, d_model),
                             persistent=False)
        self.elim_proj = nn.Linear(26, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        # Applied at EVERY position. The eliminated vector is concatenated per
        # position so each slot's prediction can condition on it directly.
        self.pos_head = nn.Sequential(
            nn.Linear(d_model + 26, head_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 26),
        )

    def forward(self, tokens, eliminated):
        """-> per-position logits float[B, L, 26]."""
        pad_mask = tokens.eq(PAD)
        x = (self.token_emb(tokens)
             + self.pos_emb[:tokens.size(1)].unsqueeze(0)
             + self.elim_proj(eliminated).unsqueeze(1))
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        e = eliminated.unsqueeze(1).expand(-1, x.size(1), -1).to(x.dtype)
        return self.pos_head(torch.cat([x, e], dim=-1))

    @staticmethod
    def presence_logprob(pos_logits, tokens):
        """Aggregate per-position distributions into P(letter present), in logs.

        Only BLANK positions contribute -- a revealed position tells us nothing
        about which letters are still hidden. Computed as

            log P(absent)  = sum over blanks of log(1 - p_pos(letter))
            P(present)     = 1 - exp(that)

        in log space, because the product runs over up to 29 terms and would
        underflow if taken directly.
        """
        logp = torch.log_softmax(pos_logits.float(), dim=-1)          # [B, L, 26]
        log1m = torch.log1p(-logp.exp().clamp(max=1 - 1e-6))          # log(1 - p)
        blank = tokens.eq(BLANK).unsqueeze(-1)                        # [B, L, 1]
        log_absent = (log1m * blank).sum(dim=1)                       # [B, 26]
        return log_absent

    @classmethod
    def presence_prob(cls, pos_logits, tokens):
        return 1.0 - cls.presence_logprob(pos_logits, tokens).exp()

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class PositionalSolver:
    """Standard solver interface, so it drops into the existing benchmarks."""

    def __init__(self, checkpoint_path, device=None, batch_size=4096):
        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = PositionalHangmanTransformer(**ck["config"])
        self.model.load_state_dict(ck["model"])
        self.model.eval().to(self.device)
        self.batch_size = batch_size
        self.meta = {k: v for k, v in ck.items() if k != "model"}

    @torch.no_grad()
    def predict_batch(self, patterns, guessed_lists):
        out = []
        for i in range(0, len(patterns), self.batch_size):
            tk, el, gm = encode_batch(patterns[i:i + self.batch_size],
                                      guessed_lists[i:i + self.batch_size],
                                      self.device)
            pos_logits = self.model(tk, el)
            # Rank by P(present); monotone in -log P(absent), so use that directly.
            score = -self.model.presence_logprob(pos_logits, tk)
            score = score.masked_fill(gm, NEG_INF)
            out.extend(ALPHABET[j] for j in score.argmax(-1).tolist())
        return out

    def predict(self, pattern, guessed):
        return self.predict_batch([pattern], [list(guessed)])[0]


if __name__ == "__main__":
    torch.manual_seed(0)
    m = PositionalHangmanTransformer(d_model=64, nhead=4, num_layers=2,
                                     dim_feedforward=128, dropout=0.0)
    print(f"parameters: {m.num_parameters():,}")
    pats = ["h_ll_", "_e_t", "_" * 29, "abc"]
    gs = [["h", "l", "a", "s"], ["e", "t", "o"], ["e"], ["a", "b", "c"]]
    tk, el, gm = encode_batch(pats, gs, torch.device("cpu"))
    m.eval()
    with torch.no_grad():
        pl = m(tk, el)
        pres = m.presence_prob(pl, tk)
    print("pos_logits", tuple(pl.shape), "| presence", tuple(pres.shape))
    assert pl.shape == (4, MAX_LEN, 26)
    assert torch.isfinite(pl).all()
    assert ((pres >= 0) & (pres <= 1)).all(), "presence must be a probability"

    # A word with no blanks can have no hidden letters: presence must be 0.
    print("no-blank row presence max:", float(pres[3].max()))
    assert float(pres[3].max()) < 1e-5, "a fully revealed word cannot hide letters"

    # More blanks must never DECREASE the chance a letter is hidden.
    t1, e1, _ = encode_batch(["h_ll_"], [["h", "l"]], torch.device("cpu"))
    t2, e2, _ = encode_batch(["h____"], [["h"]], torch.device("cpu"))
    with torch.no_grad():
        p1 = m.presence_prob(m(t1, e1), t1).sum()
        p2 = m.presence_prob(m(t2, e2), t2).sum()
    print(f"total presence mass: 2 blanks {float(p1):.3f} | 4 blanks {float(p2):.3f}")
    assert p2 > p1, "more blanks should mean more hidden letters"
    print("positional model smoke test: ok")
