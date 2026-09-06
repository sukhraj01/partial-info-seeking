#!/usr/bin/env python3
"""Measure ByT5 / CANINE fine-tune throughput before committing the single GPU.

Track 1 and Track 2 compete for one GPU (QOS low: gres/gpu=1 per user), so
picking wrong costs the whole remaining window. This reports samples/s and peak
memory for each candidate, and converts them into "states seen per hour" -- the
number that decides whether a fine-tune can see enough data to be worth running.
"""
import argparse, json, os, sys, time
import torch, torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import HangmanStates  # noqa: E402
from train_pretrained import PretrainedHangman  # noqa: E402


def probe(name, freeze, batch, ds, device, amp, steps=12):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        m = PretrainedHangman(name, freeze_layers=freeze).to(device)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=amp == torch.float16)
    crit = nn.BCEWithLogitsLoss()
    tok, el, tg = ds.batch_collate(list(range(batch)))
    tok, el, tg = tok.to(device), el.to(device), tg.to(device)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    try:
        for i in range(steps + 3):
            if i == 3:
                torch.cuda.synchronize(); t0 = time.time()
            with torch.autocast("cuda", dtype=amp):
                loss = crit(m(tok, el).float(), tg)
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
        torch.cuda.synchronize(); dt = time.time() - t0
    except torch.cuda.OutOfMemoryError:
        del m, opt; torch.cuda.empty_cache()
        return {"oom": True, "total": total, "trainable": trainable}
    peak = torch.cuda.max_memory_allocated() / 1e9
    del m, opt, tok, el, tg; torch.cuda.empty_cache()
    sps = steps * batch / dt
    return {"total": total, "trainable": trainable, "batch": batch, "freeze": freeze,
            "samples_s": sps, "peak_gb": peak, "states_per_hour": sps * 3600}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states-dir", default="data/states")
    ap.add_argument("--out", default="results/pretrained_probe.json")
    args = ap.parse_args()
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    amp = (torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8
           else torch.float16)
    print(f"gpu: {torch.cuda.get_device_name(0)} "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB | amp={amp}",
          flush=True)
    ds = HangmanStates(args.states_dir)

    rows = []
    for name in ("google/byt5-small", "google/canine-s"):
        for freeze, batch in ((0, 256), (6, 256), (6, 512)):
            r = probe(name, freeze, batch, ds, device, amp)
            r["model"] = name
            rows.append(r)
            if r.get("error"):
                print(f"{name:22s} frz={freeze} b={batch} | ERROR {r['error'][:80]}", flush=True)
            elif r.get("oom"):
                print(f"{name:22s} frz={freeze} b={batch} | OOM "
                      f"({r['total']/1e6:.0f}M params)", flush=True)
            else:
                print(f"{name:22s} frz={freeze} b={batch} | {r['total']/1e6:6.0f}M tot "
                      f"{r['trainable']/1e6:6.0f}M trn | {r['samples_s']:7,.0f} samp/s "
                      f"| peak {r['peak_gb']:4.1f} GB "
                      f"| {r['states_per_hour']/1e6:5.1f}M states/h", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "gpu": torch.cuda.get_device_name(0)}, f, indent=2)
    print(f"\nwrote {args.out}")
    print("\nReference: our from-scratch model saw ~196M samples (4 epochs x 49M) "
          "to reach 67.00%.")


if __name__ == "__main__":
    main()
