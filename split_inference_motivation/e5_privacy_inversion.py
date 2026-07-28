"""E5 - Privacy: can an attacker reconstruct the RGB frame from the transmitted features?

Threat model (deliberately GENEROUS to the attacker - a failed strong attack is the
only meaningful privacy evidence):
  - attacker intercepts the uplink -> has the exact wire features (per-channel u8),
  - attacker knows the encoder architecture and has a large in-distribution dataset
    of (feature, image) pairs from the same scenes,
  - attacker trains a dedicated inversion decoder with full supervision.

Reported against two reference points:
  - architecture B (full-offload): the "reconstruction" IS the transmitted JPEG -> PSNR/SSIM ceiling
  - mean-image predictor: the zero-information floor

Usage: python e5_privacy_inversion.py [--minutes 15] [--variant u8|fp32]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from torch.utils.data import DataLoader, Dataset

from common_setup import EXP_DIR, FrontWrapper, build_full_model

OUT = Path(__file__).parent / "results"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class FrameDS(Dataset):
    """Loads the 7-ch model input and the plain RGB target at model resolution.

    split_mode="manifest" (default): use the dataset's own train/val/test column.
      WARNING for inversion work - that split is randomly interleaved per sample, and
      consecutive samples are ~0.2 s apart on a moving ego, so test frames have
      near-duplicate neighbours in train. That inflates an attacker's apparent success.

    split_mode="temporal": leakage-controlled holdout. Within each experiment_id, order
      by frame_id and take train = first 65%, test = last 20%, leaving a 15% buffer so no
      training frame is temporally adjacent to a test frame. The route still repeats across
      the 8 loops, so geography is NOT disjoint - this removes adjacent-frame leakage only.
    """

    def __init__(self, split, input_size, limit=0, split_mode="manifest"):
        self.ds = EXP_DIR / "dataset"
        self.W, self.H = input_size
        with open(self.ds / "manifest.csv") as fh:
            allrows = list(csv.DictReader(fh))
        if split_mode == "manifest":
            rows = [r for r in allrows if r.get("split") == split]
        elif split_mode == "temporal":
            by_exp = {}
            for r in allrows:
                by_exp.setdefault(r["experiment_id"], []).append(r)
            rows = []
            for exp, rs in sorted(by_exp.items()):
                rs.sort(key=lambda r: int(r["frame_id"]))
                n = len(rs)
                if split == "train":
                    rows += rs[: int(0.65 * n)]
                else:                       # test: last 20%, after a 15% buffer
                    rows += rs[int(0.80 * n):]
            rows.sort(key=lambda r: (r["experiment_id"], int(r["frame_id"])))
        else:
            raise ValueError(split_mode)
        self.rows = rows[:limit] if limit else rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.ds / r["rgb_path"]).convert("RGB").resize((self.W, self.H), Image.BILINEAR)
        rgb = np.asarray(img, np.float32) / 255.0                     # target, [0,1] HWC
        norm = ((rgb - MEAN) / STD).transpose(2, 0, 1)                # model input RGB
        rad = np.load(self.ds / r["radar_tensor_path"])
        if hasattr(rad, "files"):
            rad = rad["radar"]
        rad = np.asarray(rad, np.float32)
        if rad.shape[1] != self.H or rad.shape[2] != self.W:
            import cv2
            ch = [cv2.resize(c, (self.W, self.H),
                             interpolation=cv2.INTER_NEAREST if k == 0 else cv2.INTER_LINEAR)
                  for k, c in enumerate(rad)]
            rad = np.stack(ch, 0).astype(np.float32)
        x = np.concatenate([norm, rad], 0)
        return torch.from_numpy(x), torch.from_numpy(rgb.transpose(2, 0, 1))


def quantize_per_channel_u8(t):
    """Simulate the deployed per_channel_uint8 wire codec (quantize -> dequantize)."""
    B, C = t.shape[0], t.shape[1]
    flat = t.reshape(B, C, -1)
    lo = flat.min(dim=2, keepdim=True).values
    hi = flat.max(dim=2, keepdim=True).values
    scale = (hi - lo).clamp(min=1e-8) / 255.0
    q = torch.round((flat - lo) / scale).clamp(0, 255)
    return (q * scale + lo).reshape(t.shape)


class InversionDecoder(nn.Module):
    """Attacker's decoder: transmitted features (low 40ch@1/8, high 960ch@1/16) -> RGB."""

    def __init__(self, low_ch=40, high_ch=960, base=192):
        super().__init__()
        self.high_in = nn.Sequential(nn.Conv2d(high_ch, base, 1), nn.BatchNorm2d(base), nn.ReLU(True))
        self.low_in = nn.Sequential(nn.Conv2d(low_ch, 64, 1), nn.BatchNorm2d(64), nn.ReLU(True))

        def up(cin, cout):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True))

        self.u1 = up(base, 128)          # 1/16 -> 1/8
        self.u2 = up(128 + 64, 96)       # 1/8  -> 1/4  (low feature concatenated here)
        self.u3 = up(96, 64)             # 1/4  -> 1/2
        self.u4 = up(64, 32)             # 1/2  -> 1/1
        self.out = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, low, high):
        h = self.u1(self.high_in(high))
        l = self.low_in(low)
        if h.shape[-2:] != l.shape[-2:]:
            h = F.interpolate(h, size=l.shape[-2:], mode="bilinear", align_corners=False)
        h = self.u2(torch.cat([h, l], 1))
        h = self.u4(self.u3(h))
        return torch.sigmoid(self.out(h))


@torch.no_grad()
def evaluate(dec, front, loader, dev, variant, out_hw, max_batches=40, save_samples=0):
    dec.eval()
    ps, ss = [], []
    samples = []
    for bi, (x, tgt) in enumerate(loader):
        if bi >= max_batches:
            break
        x = x.to(dev, non_blocking=True)
        f = front(x)
        low, high = f["low"], f["high"]
        if variant == "u8":
            low, high = quantize_per_channel_u8(low), quantize_per_channel_u8(high)
        rec = dec(low, high)
        if rec.shape[-2:] != out_hw:
            rec = F.interpolate(rec, size=out_hw, mode="bilinear", align_corners=False)
        rec = rec.cpu().numpy().transpose(0, 2, 3, 1)
        gt = tgt.numpy().transpose(0, 2, 3, 1)
        for k in range(rec.shape[0]):
            ps.append(psnr_fn(gt[k], rec[k], data_range=1.0))
            ss.append(ssim_fn(gt[k], rec[k], channel_axis=2, data_range=1.0))
            if len(samples) < save_samples:
                samples.append((gt[k], rec[k]))
    dec.train()
    return float(np.mean(ps)), float(np.mean(ss)), samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15.0, help="training time-box")
    ap.add_argument("--variant", default="u8", choices=("u8", "fp32"))
    ap.add_argument("--train-limit", type=int, default=6000)
    ap.add_argument("--test-limit", type=int, default=320)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    dev = torch.device("cuda")
    model, input_size, _ = build_full_model(dev)
    front = FrontWrapper(model).eval().to(dev)
    for p in front.parameters():
        p.requires_grad_(False)
    out_hw = (input_size[1], input_size[0])

    tr = DataLoader(FrameDS("train", input_size, args.train_limit), batch_size=args.batch,
                    shuffle=True, num_workers=6, pin_memory=True, drop_last=True, persistent_workers=True)
    te = DataLoader(FrameDS("test", input_size, args.test_limit), batch_size=args.batch,
                    shuffle=False, num_workers=4, pin_memory=True)
    print(f"train {len(tr.dataset)} / test {len(te.dataset)} frames; variant={args.variant}")

    dec = InversionDecoder().to(dev)
    nparam = sum(p.numel() for p in dec.parameters())
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    print(f"attacker decoder: {nparam/1e6:.2f}M params, time-box {args.minutes} min")

    t0 = time.time()
    deadline = t0 + args.minutes * 60
    step, ep = 0, 0
    hist = []
    while time.time() < deadline:
        ep += 1
        for x, tgt in tr:
            if time.time() >= deadline:
                break
            x, tgt = x.to(dev, non_blocking=True), tgt.to(dev, non_blocking=True)
            with torch.no_grad():
                f = front(x)
                low, high = f["low"], f["high"]
                if args.variant == "u8":
                    low, high = quantize_per_channel_u8(low), quantize_per_channel_u8(high)
            with torch.amp.autocast("cuda"):
                rec = dec(low, high)
                if rec.shape[-2:] != tgt.shape[-2:]:
                    rec = F.interpolate(rec, size=tgt.shape[-2:], mode="bilinear", align_corners=False)
                loss = F.l1_loss(rec.float(), tgt) + 0.5 * F.mse_loss(rec.float(), tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 100 == 0:
                print(f"  ep{ep} step{step} loss {loss.item():.4f} "
                      f"({(time.time()-t0)/60:.1f}/{args.minutes} min)")
        p, s, _ = evaluate(dec, front, te, dev, args.variant, out_hw, max_batches=10)
        hist.append({"epoch": ep, "step": step, "psnr": round(p, 3), "ssim": round(s, 4)})
        print(f"  [epoch {ep}] test PSNR {p:.2f} dB  SSIM {s:.4f}")

    psnr, ssim, samples = evaluate(dec, front, te, dev, args.variant, out_hw,
                                   max_batches=40, save_samples=3)
    print(f"\nFINAL attack: PSNR {psnr:.2f} dB  SSIM {ssim:.4f}  ({step} steps, {(time.time()-t0)/60:.1f} min)")

    # ---- reference points ----
    # floor: predict the dataset mean image (zero information from features)
    acc, n = None, 0
    for bi, (_, tgt) in enumerate(te):
        acc = tgt.sum(0) if acc is None else acc + tgt.sum(0)
        n += tgt.shape[0]
        if bi >= 20:
            break
    mean_img = (acc / n).numpy().transpose(1, 2, 0)
    fp, fs = [], []
    for bi, (_, tgt) in enumerate(te):
        if bi >= 20:
            break
        for k in range(tgt.shape[0]):
            g = tgt[k].numpy().transpose(1, 2, 0)
            fp.append(psnr_fn(g, mean_img, data_range=1.0))
            fs.append(ssim_fn(g, mean_img, channel_axis=2, data_range=1.0))
    floor_psnr, floor_ssim = float(np.mean(fp)), float(np.mean(fs))

    # ceiling: architecture B ships the JPEG itself
    ds = EXP_DIR / "dataset"
    with open(ds / "manifest.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("split") == "test"][:40]
    bp, bs = [], []
    for r in rows:
        img = Image.open(ds / r["rgb_path"]).convert("RGB").resize(input_size, Image.BILINEAR)
        g = np.asarray(img, np.float32) / 255.0
        buf = io.BytesIO()
        Image.fromarray((g * 255).astype(np.uint8)).save(buf, format="JPEG", quality=92)
        buf.seek(0)
        rec = np.asarray(Image.open(buf).convert("RGB"), np.float32) / 255.0
        bp.append(psnr_fn(g, rec, data_range=1.0))
        bs.append(ssim_fn(g, rec, channel_axis=2, data_range=1.0))

    res = {
        "variant": args.variant,
        "attack": {"psnr_db": round(psnr, 3), "ssim": round(ssim, 4),
                   "decoder_params_M": round(nparam / 1e6, 3), "steps": step,
                   "train_frames": len(tr.dataset), "minutes": round((time.time() - t0) / 60, 2)},
        "floor_mean_image": {"psnr_db": round(floor_psnr, 3), "ssim": round(floor_ssim, 4)},
        "ceiling_architecture_B_jpeg92": {"psnr_db": round(float(np.mean(bp)), 3),
                                          "ssim": round(float(np.mean(bs)), 4)},
        "history": hist,
    }
    p = OUT / f"E5_raw_{args.variant}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"floor  (mean image) PSNR {floor_psnr:.2f} SSIM {floor_ssim:.4f}")
    print(f"ceiling (arch B JPEG) PSNR {np.mean(bp):.2f} SSIM {np.mean(bs):.4f}")
    print(f"wrote {p}")

    if samples:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(samples), 2, figsize=(9, 3.1 * len(samples)))
        axes = np.atleast_2d(axes)
        for i, (g, r) in enumerate(samples):
            axes[i, 0].imshow(np.clip(g, 0, 1)); axes[i, 0].set_title(
                "original (what architecture B transmits)" if i == 0 else "", fontsize=10)
            axes[i, 1].imshow(np.clip(r, 0, 1)); axes[i, 1].set_title(
                f"attacker reconstruction from features ({args.variant})" if i == 0 else "", fontsize=10)
            for a in axes[i]:
                a.axis("off")
        fig.tight_layout()
        fp_ = OUT / f"E5_inversion_samples_{args.variant}.png"
        fig.savefig(fp_, dpi=140)
        print(f"wrote {fp_}")


if __name__ == "__main__":
    main()
