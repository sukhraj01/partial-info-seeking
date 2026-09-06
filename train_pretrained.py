#!/usr/bin/env python3
"""Track 1: fine-tune a pretrained character model (ByT5 encoder) for hangman.

The bet: ByT5 saw billions of English characters in pretraining, so its
sub-word structure priors should transfer to guessing letters in unseen words --
which is exactly where our from-scratch model is bottlenecked.

The cost: ByT5-small's encoder is ~218M parameters, roughly 4x the largest
config we benchmarked (A3, 57M, measured at 19.7 h for 4 epochs over 50M
states). A full-data fine-tune is ~75 h and does not fit. So this trains on a
capped sample (--epoch-size) and optionally freezes the lower encoder layers,
which is standard for fine-tuning and cuts backward cost substantially.

Input mapping: our state encoding is translated to raw ByT5 byte ids --
PAD -> 0, BLANK -> the '_' byte, letters -> their ASCII bytes -- so the
pretrained embeddings are used as they were trained, not reinitialised.
The eliminated-letter vector is injected the way our own model does it:
projected to d_model and broadcast onto every position, plus concatenated at
the head. That keeps the pretrained stack untouched and adds only an adapter.

  python train_pretrained.py --epoch-size 8000000 --epochs 1 --freeze-layers 6
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from batch_sim import run_batch_benchmark  # noqa: E402
from dataset import HangmanStates  # noqa: E402

MAX_LEN = 29
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
PAD, BLANK, LETTER_BASE = 0, 1, 2

# Fixed-width character layout, so the whole encoding is vectorised:
#   [0..25]  one slot per letter: the letter itself if ELIMINATED, else '.'
#   [26]     '|' separator
#   [27..55] the revealed pattern, '_' for blanks, padding after the word
# 56 positions total, and divisible by 4, which CANINE's downsampling requires.
ELIM_SLOTS, SEP_POS, PAT_POS, SEQ_LEN = 26, 26, 27, 56
FILLER, SEP, BLANK_CH = ".", "|", "_"


def build_char_codes(tokens: torch.Tensor, elim: torch.Tensor):
    """(our tokens [B,29], eliminated [B,26]) -> (unicode codepoints [B,56], mask [B,56]).

    Eliminated letters are PREPENDED as literal characters rather than injected
    as a learned vector. A pretrained character model has never seen an
    'eliminated-letter embedding', but it has seen plenty of letters -- so the
    negative evidence is expressed in the alphabet the model was pretrained on.
    """
    B = tokens.size(0)
    dev = tokens.device
    codes = torch.empty(B, SEQ_LEN, dtype=torch.long, device=dev)

    letter_ids = torch.arange(26, device=dev) + ord("a")
    codes[:, :ELIM_SLOTS] = torch.where(elim > 0,
                                        letter_ids.unsqueeze(0).expand(B, 26),
                                        torch.full((B, 26), ord(FILLER), device=dev))
    codes[:, SEP_POS] = ord(SEP)

    pat = torch.where(tokens == BLANK, torch.full_like(tokens, ord(BLANK_CH)),
                      tokens - LETTER_BASE + ord("a"))
    pad = tokens == PAD
    pat = torch.where(pad, torch.zeros_like(pat), pat)
    codes[:, PAT_POS:PAT_POS + MAX_LEN] = pat

    mask = torch.ones(B, SEQ_LEN, dtype=torch.long, device=dev)
    mask[:, PAT_POS:PAT_POS + MAX_LEN] = (~pad).long()
    return codes, mask


class PretrainedHangman(nn.Module):
    """A pretrained character encoder + our 26-way multi-label hangman head."""

    def __init__(self, model_name="google/byt5-small", freeze_layers=0, dropout=0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(model_name)
        self.model_type = cfg.model_type
        self.model_name = model_name

        if self.model_type == "t5":
            from transformers import T5EncoderModel
            loader = T5EncoderModel
        else:
            loader = AutoModel
        try:
            self.encoder, info = loader.from_pretrained(model_name,
                                                        output_loading_info=True)
        except TypeError:
            self.encoder = loader.from_pretrained(model_name)
            info = {"missing_keys": []}

        # The whole premise of this track is PRETRAINED knowledge. Loading the
        # wrong class silently yields a randomly-initialised model that trains
        # fine and means nothing, so refuse to continue if the weights did not land.
        missing = [k for k in info.get("missing_keys", [])
                   if "position_ids" not in k and not k.startswith("head")]
        if len(missing) > 8:
            raise RuntimeError(
                f"{model_name}: {len(missing)} pretrained weights failed to load "
                f"(e.g. {missing[:4]}). Refusing to train a randomly-initialised "
                f"model and call it pretrained.")
        self.n_missing = len(missing)

        d = getattr(cfg, "d_model", None) or getattr(cfg, "hidden_size")
        self.d_model = d
        # ByT5 works on bytes offset by 3; CANINE consumes unicode codepoints.
        self.id_offset = 3 if self.model_type == "t5" else 0
        self.head = nn.Sequential(
            nn.Linear(d + 26, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 26),
        )
        self.n_frozen = 0
        if freeze_layers:
            blocks = (self.encoder.encoder.block if self.model_type == "t5"
                      else self.encoder.encoder.layer)
            emb = (self.encoder.shared if self.model_type == "t5"
                   else self.encoder.char_embeddings)
            for p_ in emb.parameters():
                p_.requires_grad = False
            for blk in blocks[:freeze_layers]:
                for p_ in blk.parameters():
                    p_.requires_grad = False
            self.n_frozen = freeze_layers

    def forward(self, tokens, eliminated):
        codes, mask = build_char_codes(tokens, eliminated)
        ids = codes + self.id_offset * (codes > 0).long()
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        valid = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        return self.head(torch.cat([pooled, eliminated.to(pooled.dtype)], dim=-1))

    def num_parameters(self, trainable_only=False):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad or not trainable_only)


class PretrainedSolver:
    """Same interface the benchmark and inference paths expect."""

    def __init__(self, checkpoint_path, device=None, batch_size=1024):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = PretrainedHangman(ck["model_name"], freeze_layers=0)
        self.model.load_state_dict(ck["model"])
        self.model.eval().to(self.device)
        self.batch_size = batch_size
        self.meta = {k: v for k, v in ck.items() if k != "model"}

    @torch.no_grad()
    def predict_batch(self, patterns, guessed_lists):
        from transformer_model import encode_batch
        out = []
        for i in range(0, len(patterns), self.batch_size):
            tk, el, gm = encode_batch(patterns[i:i + self.batch_size],
                                      guessed_lists[i:i + self.batch_size], self.device)
            lg = self.model(tk, el).float().masked_fill(gm, float("-inf"))
            out.extend(ALPHABET[j] for j in lg.argmax(-1).tolist())
        return out

    def predict(self, pattern, guessed):
        return self.predict_batch([pattern], [list(guessed)])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states-dir", default="data/states")
    ap.add_argument("--model-name", default="google/byt5-small")
    ap.add_argument("--out", default="models/pretrained.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--epoch-size", type=int, default=8_000_000,
                    help="states per epoch; the full pool does NOT fit in the window")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--freeze-layers", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-batches", type=int, default=60)
    ap.add_argument("--val-games", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--time-budget-h", type=float, default=0.0,
                    help="stop cleanly after this many hours (0 = no limit)")
    ap.add_argument("--resume", default=None,
                    help="continue from a checkpoint written by a previous run. "
                         "The LR schedule restarts, which is what we want: a "
                         "cosine-decayed run has already annealed to ~0, so "
                         "continuing it flat would learn nothing.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if device.type == "cuda":
        amp_dtype = (torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8
                     else torch.float16)
    print(f"device={device} amp={amp_dtype}", flush=True)

    with open(os.path.join(args.states_dir, "meta.json")) as f:
        meta = json.load(f)
    n_states, per_word, n_words = meta["n_states"], meta["states_per_word"], meta["n_words"]
    val_words = max(1, int(n_words * 0.02))
    cut = (n_words - val_words) * per_word
    print(f"{n_states:,} states | train rows [0,{cut:,}) | val [{cut:,},{n_states:,})",
          flush=True)

    ds_all = HangmanStates(args.states_dir)
    val_loader = torch.utils.data.DataLoader(
        HangmanStates(args.states_dir, split=(cut, n_states)),
        batch_size=args.batch_size, shuffle=False, num_workers=2,
        collate_fn=ds_all.batch_collate, pin_memory=True)

    words_used = [w.strip() for w in open(meta["source_words"]) if w.strip()]
    heldout = words_used[n_words - val_words:]
    import random as _rnd
    _rnd.Random(0).shuffle(heldout)
    heldout = heldout[:args.val_games]

    model = PretrainedHangman(args.model_name, freeze_layers=args.freeze_layers).to(device)
    resumed_from = None
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        resumed_from = {"path": args.resume, "epoch": ck.get("epoch"),
                        "val_win_rate": ck.get("val_win_rate")}
        print(f"resumed from {args.resume} "
              f"(epoch {ck.get('epoch')}, val_win_rate {ck.get('val_win_rate')})",
              flush=True)
    tot = model.num_parameters()
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{args.model_name}: {tot:,} params, {trn:,} trainable "
          f"({model.n_frozen} blocks frozen), d_model={model.d_model}", flush=True)

    accum = max(1, args.grad_accum)
    steps_per_epoch = (args.epoch_size // args.batch_size) // accum
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_frac))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)

    def lr_at(s):
        if s < warmup:
            return s / warmup
        p = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)
    crit = nn.BCEWithLogitsLoss()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"{steps_per_epoch:,} steps/epoch, {total_steps:,} total "
          f"(effective batch {args.batch_size*accum})", flush=True)

    rng = np.random.default_rng(args.seed)
    # Seed `best` with the resumed checkpoint's score so a continuation can only
    # overwrite the file if it actually beats what it started from.
    best = (resumed_from or {}).get("val_win_rate") or -1.0
    history, step, micro, t0, stop = [], 0, 0, time.time(), False
    if resumed_from:
        print(f"checkpoint will only be overwritten above {best:.2f}%", flush=True)
    opt.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        if stop:
            break
        idx = rng.choice(cut, size=min(args.epoch_size, cut), replace=False)
        loader = torch.utils.data.DataLoader(
            _Idx(idx), batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, collate_fn=ds_all.batch_collate,
            pin_memory=True, drop_last=True)
        run_loss, run_n = 0.0, 0
        for tokens, elim, tgt in loader:
            tokens, elim, tgt = (tokens.to(device, non_blocking=True),
                                 elim.to(device, non_blocking=True),
                                 tgt.to(device, non_blocking=True))
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                loss = crit(model(tokens, elim).float(), tgt)
            (scaler.scale(loss / accum) if scaler.is_enabled() else loss / accum).backward()
            run_loss += loss.item(); run_n += 1; micro += 1
            if micro % accum:
                continue
            if scaler.is_enabled():
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
            opt.zero_grad(set_to_none=True); sched.step(); step += 1

            if step % args.log_every == 0:
                el = time.time() - t0
                print(f"ep{epoch} step {step:,}/{total_steps:,} | loss {run_loss/run_n:.4f} "
                      f"| lr {sched.get_last_lr()[0]:.2e} | {step/el:.2f} it/s "
                      f"| {step*args.batch_size*accum/el:,.0f} samp/s "
                      f"| eta {(total_steps-step)/(step/el)/60:.1f}m", flush=True)
                run_loss, run_n = 0.0, 0
            if args.max_steps and step >= args.max_steps:
                stop = True; break
            if args.time_budget_h and (time.time() - t0) > args.time_budget_h * 3600:
                print(f"time budget {args.time_budget_h}h reached -- stopping", flush=True)
                stop = True; break

        vl, vh = _evaluate(model, val_loader, device, amp_dtype, args.val_batches)
        wr = float("nan")
        if heldout:
            model.eval()
            solver_pb = _make_pb(model, device)
            wr = run_batch_benchmark(solver_pb, heldout, name="epoch-eval",
                                     chunk_size=2048, verbose=False)["win_rate"]
            model.train()
        print(f"== epoch {epoch} | val_loss {vl:.4f} | val_top1_hit {vh*100:.2f}% "
              f"| val_win_rate {wr:.2f}% | elapsed {(time.time()-t0)/60:.1f}m", flush=True)
        history.append({"epoch": epoch, "step": step, "val_loss": vl,
                        "val_top1_hit": vh, "val_win_rate": wr})
        if wr == wr and wr > best:
            best = wr
            torch.save({"model": model.state_dict(), "model_name": args.model_name,
                        "resumed_from": resumed_from,
                        "epoch": epoch, "step": step, "val_loss": vl,
                        "val_top1_hit": vh, "val_win_rate": wr,
                        "args": vars(args), "history": history}, args.out)
            print(f"   saved {args.out} (best win_rate {best:.2f}%)", flush=True)

    print(f"done in {(time.time()-t0)/60:.1f}m | best val_win_rate {best:.2f}%", flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/pretrained_training.json", "w") as f:
        json.dump({"history": history, "args": vars(args), "resumed_from": resumed_from,
                   "parameters": tot, "trainable": trn}, f, indent=2)


class _Idx(torch.utils.data.Dataset):
    def __init__(self, idx): self.idx = idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i): return int(self.idx[i])


def _make_pb(model, device):
    from transformer_model import encode_batch
    @torch.no_grad()
    def pb(pats, gs):
        out = []
        for i in range(0, len(pats), 1024):
            tk, el, gm = encode_batch(pats[i:i+1024], gs[i:i+1024], device)
            lg = model(tk, el).float().masked_fill(gm, float("-inf"))
            out.extend(ALPHABET[j] for j in lg.argmax(-1).tolist())
        return out
    return pb


@torch.no_grad()
def _evaluate(model, loader, device, amp_dtype, max_batches):
    model.eval(); crit = nn.BCEWithLogitsLoss()
    tl, th, tn, nb = 0.0, 0, 0, 0
    for tokens, elim, tgt in loader:
        tokens, elim, tgt = tokens.to(device), elim.to(device), tgt.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            logits = model(tokens, elim)
            tl += crit(logits.float(), tgt).item()
        guessed = elim > 0
        revealed = torch.zeros_like(guessed)
        for j in range(26):
            revealed[:, j] = (tokens == (j + 2)).any(dim=1)
        pick = logits.float().masked_fill(guessed | revealed, float("-inf")).argmax(-1)
        th += tgt.gather(1, pick.unsqueeze(1)).sum().item(); tn += tokens.size(0); nb += 1
        if max_batches and nb >= max_batches:
            break
    model.train()
    return tl / max(nb, 1), th / max(tn, 1)


if __name__ == "__main__":
    main()
