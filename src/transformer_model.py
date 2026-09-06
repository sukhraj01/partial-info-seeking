"""Character-level transformer MLM for hangman.

The game state is (pattern, eliminated). The pattern goes in per-position; the
eliminated set is GLOBAL negative evidence and is injected two ways -- projected
to d_model and broadcast onto every position before the encoder, and
concatenated to the pooled vector at the head. Broadcasting lets attention
condition on it; the head concat gives the classifier a direct, unmixed view.

Output is 26 logits for "letter L is present in the word but not yet guessed",
trained with BCEWithLogitsLoss. At inference every already-guessed letter is set
to -inf before the argmax, so the model can never repeat a guess.
"""
import math

import torch
import torch.nn as nn

MAX_LEN = 29
PAD, BLANK, LETTER_BASE = 0, 1, 2
VOCAB_SIZE = LETTER_BASE + 26  # 28
ALPHABET = "abcdefghijklmnopqrstuvwxyz"

DEFAULT_CONFIG = dict(d_model=256, nhead=8, num_layers=6, dim_feedforward=1024,
                      dropout=0.1, max_len=MAX_LEN)


def sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class HangmanTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024,
                 dropout=0.1, max_len=MAX_LEN):
        super().__init__()
        self.config = dict(d_model=d_model, nhead=nhead, num_layers=num_layers,
                           dim_feedforward=dim_feedforward, dropout=dropout,
                           max_len=max_len)
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.register_buffer("pos_emb", sinusoidal_encoding(max_len, d_model),
                             persistent=False)
        self.elim_proj = nn.Linear(26, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,  # pre-LN: stabler without a long warmup
        )
        # enable_nested_tensor is a no-op with norm_first=True and only emits a warning.
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model + 26, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 26),
        )

    def forward(self, tokens: torch.Tensor, eliminated: torch.Tensor) -> torch.Tensor:
        """tokens int64[B, L], eliminated float32[B, 26] -> logits float[B, 26]."""
        pad_mask = tokens.eq(PAD)
        x = (self.token_emb(tokens)
             + self.pos_emb[:tokens.size(1)].unsqueeze(0)
             + self.elim_proj(eliminated).unsqueeze(1))
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        valid = (~pad_mask).to(x.dtype).unsqueeze(-1)
        pooled = (x * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        return self.head(torch.cat([pooled, eliminated.to(pooled.dtype)], dim=-1))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def encode_batch(patterns, guessed_lists, device, max_len=MAX_LEN):
    """(patterns, guessed) -> (tokens, eliminated, guessed_mask) on `device`.

    guessed_mask marks every letter already guessed so the caller can suppress
    it before the argmax. Encoding here mirrors dataset.encode_live_state
    exactly; the two are cross-checked in the benchmark script.
    """
    b = len(patterns)
    tokens = torch.zeros(b, max_len, dtype=torch.long)
    elim = torch.zeros(b, 26, dtype=torch.float32)
    gmask = torch.zeros(b, 26, dtype=torch.bool)
    for i, (pat, guessed) in enumerate(zip(patterns, guessed_lists)):
        revealed = set()
        for p, ch in enumerate(pat):
            if ch == "_":
                tokens[i, p] = BLANK
            else:
                tokens[i, p] = LETTER_BASE + (ord(ch) - 97)
                revealed.add(ch)
        for ch in guessed:
            j = ord(ch) - 97
            gmask[i, j] = True
            if ch not in revealed:
                elim[i, j] = 1.0
    return tokens.to(device), elim.to(device), gmask.to(device)


class TransformerSolver:
    """Wraps a trained checkpoint in the standard solver interface."""

    def __init__(self, checkpoint_path: str, device: str = None, batch_size: int = 4096):
        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", DEFAULT_CONFIG)
        self.model = HangmanTransformer(**cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval().to(self.device)
        self.batch_size = batch_size
        self.meta = {k: v for k, v in ckpt.items() if k not in ("model", "optimizer")}

    @torch.no_grad()
    def predict_batch(self, patterns, guessed_lists):
        out = []
        for i in range(0, len(patterns), self.batch_size):
            pats = patterns[i:i + self.batch_size]
            gs = guessed_lists[i:i + self.batch_size]
            tokens, elim, gmask = encode_batch(pats, gs, self.device)
            logits = self.model(tokens, elim).float()
            logits = logits.masked_fill(gmask, float("-inf"))
            idx = logits.argmax(dim=-1).tolist()
            out.extend(ALPHABET[j] for j in idx)
        return out

    def predict(self, pattern: str, guessed: list) -> str:
        return self.predict_batch([pattern], [list(guessed)])[0]


if __name__ == "__main__":
    # Shape/dtype smoke test -- no training, no data required.
    torch.manual_seed(0)
    model = HangmanTransformer()
    print(f"parameters: {model.num_parameters():,}")

    pats = ["h_ll_", "_e_t", "a", "_" * 29]
    guessed = [["h", "l", "a", "s"], ["e", "t", "o"], [], ["e"]]
    tokens, elim, gmask = encode_batch(pats, guessed, torch.device("cpu"))
    print("tokens", tuple(tokens.shape), tokens.dtype,
          "| eliminated", tuple(elim.shape), elim.dtype,
          "| guessed_mask", tuple(gmask.shape), gmask.dtype)

    model.eval()
    with torch.no_grad():
        logits = model(tokens, elim)
    print("logits", tuple(logits.shape), logits.dtype,
          "| finite:", bool(torch.isfinite(logits).all()))

    masked = logits.masked_fill(gmask, float("-inf"))
    picks = [ALPHABET[i] for i in masked.argmax(-1).tolist()]
    print("picks (untrained):", picks)
    for p, g in zip(picks, guessed):
        assert p not in g, f"picked an already-guessed letter: {p!r}"
    print("no already-guessed letter picked: ok")

    # Padding must not change the answer for a short word.
    t2, e2, _ = encode_batch(["h_ll_"], [["h", "l", "a", "s"]], torch.device("cpu"))
    with torch.no_grad():
        single = model(t2, e2)
    print("batch-vs-single max abs diff:",
          float((single[0] - logits[0]).abs().max()))
