#!/usr/bin/env python3
"""Train the hangman character transformer. Runs on Ada under SLURM.

Validation is split by WORD, not by row. States are laid out word-major, so a
random row split would put states from the same word on both sides and report a
validation loss inflated by memorisation -- exactly the failure this whole
project is trying to avoid measuring.

  python train_transformer.py --states-dir data/states --epochs 8
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from batch_sim import run_batch_benchmark  # noqa: E402
from dataset import HangmanStates  # noqa: E402
from mixture_dataset import IndexDataset, MixtureStates, parse_spec  # noqa: E402
from transformer_model import HangmanTransformer  # noqa: E402
from transformer_positional import PositionalHangmanTransformer  # noqa: E402


class WeightEMA:
    """Exponential moving average of the weights, evaluated instead of the raw ones.

    Averaging the trajectory suppresses the minibatch noise the final step
    happens to land in. It is one of the most consistently positive cheap
    additions in supervised training, costs one extra copy of the model in
    memory and no extra compute in the backward pass, and cannot hurt: the raw
    weights are still there, and checkpoint selection compares both.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model):
        """Swap EMA weights in, keeping the originals for restore()."""
        sd = model.state_dict()
        self.backup = {k: sd[k].detach().clone() for k in self.shadow}
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))

    @torch.no_grad()
    def restore(self, model):
        if self.backup is None:
            return
        sd = model.state_dict()
        for k, v in self.backup.items():
            sd[k].copy_(v)
        self.backup = None


def presence_bce(log_absent, tgt):
    """BCE on the aggregated presence probability, via its logit.

    log_absent = log P(letter is NOT under any blank), so
        P(present) = 1 - exp(log_absent)
        logit      = log(1 - exp(la)) - la
    Working in logits keeps this inside binary_cross_entropy_with_logits, which
    is autocast-safe; the plain binary_cross_entropy is not.
    """
    la = log_absent.float().clamp(max=-1e-6)
    logit_present = torch.log1p(-la.exp()) - la
    return nn.functional.binary_cross_entropy_with_logits(logit_present, tgt.float())


def build_loader(state_dir, split, batch_size, workers, shuffle, with_positions=False):
    ds = HangmanStates(state_dir, split=split, with_positions=with_positions)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        collate_fn=ds.batch_collate, pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=workers > 0,
    ), ds


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, max_batches=None, positional=False):
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    tot_loss, tot_hit, tot_n, nb = 0.0, 0, 0, 0
    for batch in loader:
        tokens, elim, tgt = batch[0], batch[1], batch[2]
        tokens, elim, tgt = (tokens.to(device, non_blocking=True),
                             elim.to(device, non_blocking=True),
                             tgt.to(device, non_blocking=True))
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            if positional:
                # Score on the SAME quantity as the pooled model so val_loss and
                # top-1 hit stay comparable across the two architectures.
                log_absent = model.presence_logprob(model(tokens, elim), tokens)
                logits = -log_absent            # monotone in P(present)
                loss = presence_bce(log_absent, tgt)
            else:
                logits = model(tokens, elim)
                loss = crit(logits.float(), tgt)
        # top-1 hit rate: after suppressing guessed letters, does the argmax land
        # on a letter actually in the word? This tracks win rate far better than BCE.
        guessed = (elim > 0)
        revealed = torch.zeros_like(guessed)
        # a letter present in the pattern is revealed => also already guessed
        for j in range(26):
            revealed[:, j] = (tokens == (j + 2)).any(dim=1)
        pick = logits.float().masked_fill(guessed | revealed, float("-inf")).argmax(-1)
        tot_hit += tgt.gather(1, pick.unsqueeze(1)).sum().item()
        tot_loss += loss.item()
        tot_n += tokens.size(0)
        nb += 1
        if max_batches and nb >= max_batches:
            break
    model.train()
    return tot_loss / max(nb, 1), tot_hit / max(tot_n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states-dir", default="data/states",
                    help="round-0 pool; also the pool the val slice comes from")
    ap.add_argument("--mix", action="append", default=[],
                    help="extra pool as 'dir:weight', repeatable. The round-0 pool "
                         "gets --round0-weight. Enables DAgger mixing.")
    ap.add_argument("--round0-weight", type=float, default=1.0)
    ap.add_argument("--epoch-size", type=int, default=0,
                    help="samples per epoch when mixing (0 = size of round-0 pool)")
    ap.add_argument("--out", default="models/transformer.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4096,
                    help="MICRO-batch. Effective batch is this x --grad-accum.")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="accumulate this many micro-batches per optimizer step. "
                         "Lets a model too big for batch 2048 keep the SAME "
                         "effective batch, and therefore the same LR schedule, "
                         "as the round-0 baseline it is being compared against.")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-word-frac", type=float, default=0.02)
    ap.add_argument("--val-batches", type=int, default=200)
    ap.add_argument("--val-games", type=int, default=2000,
                    help="held-out TRAIN words to actually play each epoch; 0 disables")
    ap.add_argument("--select-on", choices=["val_loss", "win_rate"], default="win_rate",
                    help="checkpoint selection metric. val_loss is measured on the "
                         "round-0 state distribution, which a DAgger mixture "
                         "deliberately moves away from -- so win_rate is the honest "
                         "criterion once --mix is used.")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--positional", action="store_true",
                    help="per-position (true MLM) head instead of mean-pooling. "
                         "Predicts a letter distribution at every blank and "
                         "aggregates as the n-gram does, which gives ~L times "
                         "more supervision per state.")
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="evaluate an exponential moving average of the weights "
                         "(0 disables). 0.999 is a sane default.")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="label smoothing on the per-position cross-entropy. "
                         "The target letter under a blank is genuinely uncertain "
                         "given the pattern, so a hard one-hot target is "
                         "overconfident by construction.")
    ap.add_argument("--aux-presence-weight", type=float, default=1.0,
                    help="weight on the presence BCE alongside the per-position "
                         "cross-entropy. The presence term is the objective the "
                         "pooled model used and is what inference ranks on, so "
                         "keeping it makes this strictly-more-signal rather than "
                         "a different objective.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full epochs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if device.type == "cuda":
        # is_bf16_supported() returns True on Turing via EMULATION, which is both
        # slower and more memory-hungry than fp16. Gate on real hardware support
        # (compute capability >= 8.0, Ampere and later) instead.
        major = torch.cuda.get_device_capability(0)[0]
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    print(f"device={device} amp={amp_dtype}", flush=True)
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"gpu: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} {total:.1f} GB",
              flush=True)

    with open(os.path.join(args.states_dir, "meta.json")) as f:
        meta = json.load(f)
    n_states = meta["n_states"]
    per_word = meta["states_per_word"]
    n_words = meta["n_words"]

    # Word-boundary split: rows [0, cut) train, [cut, n) validation.
    val_words = max(1, int(n_words * args.val_word_frac))
    cut = (n_words - val_words) * per_word
    print(f"{n_states:,} states | train rows [0,{cut:,}) "
          f"| val rows [{cut:,},{n_states:,}) over {val_words:,} held-out words",
          flush=True)

    words_used = [w.strip() for w in open(meta["source_words"]) if w.strip()]
    heldout_words = words_used[n_words - val_words:]
    if args.val_games:
        import random as _rnd
        _rnd.Random(0).shuffle(heldout_words)
        heldout_words = heldout_words[:args.val_games]
    print(f"per-epoch game eval on {len(heldout_words):,} held-out TRAIN words "
          f"(word-disjoint from training; holdout_split stays untouched for the "
          f"final benchmark)", flush=True)

    # Validation ALWAYS comes from the same held-out word slice of the round-0
    # pool, whatever the training mixture is, so val numbers stay comparable
    # across DAgger rounds.
    val_loader, _ = build_loader(args.states_dir, (cut, n_states), args.batch_size,
                                 max(2, args.workers // 2), shuffle=False,
                                 with_positions=args.positional)

    mixture = None
    if args.mix:
        specs = [(args.states_dir, args.round0_weight)] + [parse_spec(m) for m in args.mix]
        mixture = MixtureStates(specs, val_source=0, val_rows=(cut, n_states))
        epoch_size = args.epoch_size or cut
        wpct = ", ".join(f"{d.split('/')[-1]}={w*100:.0f}%"
                         for (d, _), w in zip(specs, mixture.weights))
        print(f"MIXTURE: {wpct} | pools={mixture.pool_sizes()} "
              f"| epoch_size={epoch_size:,}", flush=True)
        train_loader = None  # rebuilt per epoch with a fresh sample
    else:
        train_loader, _ = build_loader(args.states_dir, (0, cut), args.batch_size,
                                       args.workers, shuffle=True,
                                       with_positions=args.positional)
        epoch_size = cut

    def make_train_loader(epoch):
        if mixture is None:
            return train_loader
        idx = mixture.build_epoch_index(epoch_size, seed=args.seed * 1000 + epoch)
        return torch.utils.data.DataLoader(
            IndexDataset(idx), batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, collate_fn=mixture.collate,
            pin_memory=torch.cuda.is_available(), drop_last=True,
            persistent_workers=args.workers > 0)

    cfg = dict(d_model=args.d_model, nhead=args.nhead, num_layers=args.layers,
               dim_feedforward=args.ff, dropout=args.dropout)
    Net = PositionalHangmanTransformer if args.positional else HangmanTransformer
    model = Net(**cfg).to(device)
    if args.positional:
        print("head: PER-POSITION (true MLM) + presence aggregation", flush=True)
    print(f"parameters: {model.num_parameters():,}", flush=True)

    accum = max(1, args.grad_accum)
    eff_batch = args.batch_size * accum
    micro_per_epoch = (epoch_size // args.batch_size) if mixture is not None \
        else len(train_loader)
    steps_per_epoch = micro_per_epoch // accum  # OPTIMIZER steps
    if accum > 1:
        print(f"grad accumulation: {accum} x {args.batch_size} "
              f"= effective batch {eff_batch}", flush=True)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_frac))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.98))

    def lr_at(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)
    crit = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=args.label_smoothing)
    ema = WeightEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    if ema:
        print(f"EMA enabled (decay {args.ema_decay})", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_val = float("inf")
    step = 0
    t0 = time.time()
    history = []
    print(f"{steps_per_epoch:,} optimizer steps/epoch, {total_steps:,} total "
          f"(effective batch {eff_batch})", flush=True)

    stop = False
    micro = 0
    opt.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        if stop:
            break
        run_loss, run_n = 0.0, 0
        for batch in make_train_loader(epoch):
            tokens, elim, tgt = batch[0], batch[1], batch[2]
            pos_tgt = batch[3].to(device, non_blocking=True) if args.positional else None
            tokens = tokens.to(device, non_blocking=True)
            elim = elim.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                if args.positional:
                    pos_logits = model(tokens, elim)
                    loss = ce(pos_logits.float().reshape(-1, 26), pos_tgt.reshape(-1))
                    if args.aux_presence_weight:
                        # binary_cross_entropy is on the autocast blocklist (unsafe
                        # in fp16). Convert the aggregated probability to a LOGIT and
                        # use the *_with_logits form, which is both autocast-safe and
                        # numerically stabler than log-ing a near-zero probability.
                        loss = loss + args.aux_presence_weight * presence_bce(
                            model.presence_logprob(pos_logits, tokens), tgt)
                else:
                    logits = model(tokens, elim)
                    loss = crit(logits.float(), tgt)

            # Scale down so accumulated grads average, not sum -- otherwise the
            # effective LR would silently scale with --grad-accum.
            scaled_loss = loss / accum
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            run_loss += loss.item()
            run_n += 1
            micro += 1
            if micro % accum:
                continue  # mid-accumulation: no optimizer step yet

            if scaler.is_enabled():
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if ema is not None:
                ema.update(model)

            if step % args.log_every == 0:
                el = time.time() - t0
                rate = step / el
                eta = (total_steps - step) / rate / 60
                print(f"ep{epoch} step {step:,}/{total_steps:,} "
                      f"| loss {run_loss/run_n:.4f} | lr {sched.get_last_lr()[0]:.2e} "
                      f"| {rate:.1f} it/s | eta {eta:.1f}m", flush=True)
                run_loss, run_n = 0.0, 0

            if args.max_steps and step >= args.max_steps:
                stop = True
                break

        if ema is not None:
            ema.apply_to(model)   # evaluate the averaged weights
        val_loss, val_hit = evaluate(model, val_loader, device, amp_dtype,
                                     max_batches=args.val_batches,
                                     positional=args.positional)

        win_rate = float("nan")
        if args.val_games and heldout_words:
            from transformer_model import ALPHABET as _AL, encode_batch as _enc
            model.eval()

            @torch.no_grad()
            def _pb(pats, gs):
                out = []
                for i in range(0, len(pats), 4096):
                    tk, el, gm = _enc(pats[i:i+4096], gs[i:i+4096], device)
                    if args.positional:
                        lg = -model.presence_logprob(model(tk, el), tk)
                    else:
                        lg = model(tk, el).float()
                    lg = lg.float().masked_fill(gm, float("-inf"))
                    out.extend(_AL[j] for j in lg.argmax(-1).tolist())
                return out

            r = run_batch_benchmark(_pb, heldout_words, name="epoch-eval",
                                    chunk_size=4096, verbose=False)
            win_rate = r["win_rate"]
            model.train()

        print(f"== epoch {epoch} | val_loss {val_loss:.4f} | val_top1_hit {val_hit*100:.2f}% "
              f"| val_win_rate {win_rate:.2f}%{' [EMA]' if ema else ''} "
              f"| elapsed {(time.time()-t0)/60:.1f}m", flush=True)
        history.append({"epoch": epoch, "step": step, "val_loss": val_loss,
                        "val_top1_hit": val_hit, "val_win_rate": win_rate})

        # Lower is better for loss, higher for win rate -- normalise to "score".
        if args.select_on == "win_rate" and win_rate == win_rate:  # not NaN
            score, better = -win_rate, -win_rate < best_val
        else:
            score, better = val_loss, val_loss < best_val
        if better:
            best_val = score
            torch.save({"model": model.state_dict(), "config": cfg,
                        "mixture": [ [d, float(w)] for d, w in
                                     zip(mixture.dirs, mixture.weights)] if mixture else None,
                        "epoch": epoch, "step": step, "val_loss": val_loss,
                        "val_top1_hit": val_hit, "args": vars(args),
                        "states_meta": meta, "history": history}, args.out)
            print(f"   saved {args.out} (best {args.select_on} "
                  f"{-best_val if args.select_on == 'win_rate' else best_val:.4f})",
                  flush=True)
        if ema is not None:
            ema.restore(model)    # training continues from the raw weights

    print(f"done in {(time.time()-t0)/60:.1f}m | best val_loss {best_val:.4f}", flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/transformer_training.json", "w") as f:
        json.dump({"history": history, "best_val_loss": best_val,
                   "args": vars(args), "parameters": model.num_parameters()}, f, indent=2)


if __name__ == "__main__":
    main()
