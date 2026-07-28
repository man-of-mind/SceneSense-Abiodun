"""E5 (extended) - does a SMALLER split payload also leak less?

Runs the same feature-inversion attack against several deployed split profiles taken
from rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md, chosen to hold model accuracy while
varying AE bottleneck / quantization bit-depth / ROI drop.

Wire pipeline replicated exactly as evaluate_fusion.py does it:
    backbone -> ROI gate (rank-based) -> AE.encode('high') -> per-channel uintN quantize
The attacker intercepts (low, high_bottleneck) AFTER all of that.

Usage: python e5_profile_variants.py --profile ae128__uint4__roi0.0 --minutes 15
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from torch.utils.data import DataLoader

from common_setup import AB, EXP_DIR
from e5_privacy_inversion import FrameDS, InversionDecoder

OUT = Path(__file__).parent / "results"

# Profiles from PERMODEL_KNOB_MATRIX_ZSTD.md. accuracy figures are that file's, not re-measured.
# accept = within the 2% tolerance of the ae128__clean baseline (its "accept" column).
PROFILES = {
    "noae__uint8__roi0.0": dict(model="noae", bits=8, roi=0.0, payload_kb=1050.3,
                                miou=0.840, ped_recall=0.855, loc_m=0.95, accept=False,
                                note="currently deployed (no AE)"),
    "ae128__uint4__roi0.0": dict(model="ae128", bits=4, roi=0.0, payload_kb=129.2,
                                 miou=0.819, ped_recall=0.887, loc_m=0.88, accept=True,
                                 note="PARETO PICK - min payload within accuracy tolerance"),
    "ae32__uint6__roi0.0": dict(model="ae32", bits=6, roi=0.0, payload_kb=174.7,
                                miou=0.822, ped_recall=0.865, loc_m=0.88, accept=True,
                                note="smallest bottleneck (32ch)"),
    "ae64__uint8__roi0.3": dict(model="ae64", bits=8, roi=0.3, payload_kb=195.7,
                                miou=0.805, ped_recall=0.864, loc_m=0.86, accept=True,
                                note="only accepted profile with ROI drop"),
}

CKPTS = {
    "noae": EXP_DIR / "checkpoints" / "mprime_joint_noae" / "best.pt",
    "ae32": AB / "experiments/ae_integrated_20260710/ae32/checkpoints/ae32_integrated/best.pt",
    "ae64": AB / "experiments/ae_integrated_20260710/ae64/checkpoints/ae64_integrated/best.pt",
    "ae128": AB / "experiments/ae_integrated_20260710/ae128/checkpoints/ae128_integrated/best.pt",
}


def build_variant_model(kind, device):
    """Mirror evaluate_fusion.py's build, including attaching the integrated AE before load."""
    from pole_lraspp_multimodal_fusion.common import load_config
    from pole_lraspp_multimodal_fusion.model import OBJECT_HEAD_CHANNELS, build_multitask_fusion_lraspp
    from common_setup import CONFIG

    cfg = load_config(str(CONFIG))
    tcfg, fcfg, ocfg = cfg["training"], cfg.get("fusion", {}), cfg.get("object_heads", {})
    ckpt = torch.load(CKPTS[kind], map_location=device)
    input_size = tuple(int(v) for v in ckpt.get("input_size", tcfg.get("input_size", [768, 432])))
    model = build_multitask_fusion_lraspp(
        num_classes=int(tcfg.get("num_classes", 3)),
        radar_channels=int(ckpt.get("radar_channels") or fcfg.get("radar_channels", 4)),
        pretrained=False,
        object_channels=int(ckpt.get("object_channels") or ocfg.get("output_channels", OBJECT_HEAD_CHANNELS)),
        object_hidden_channels=int(ocfg.get("hidden_channels", 128)),
        fuse_low_into_object_head=bool(ckpt.get("fuse_low_into_object_head"))
        or bool(ocfg.get("fuse_low_feature", False)),
        head_arch=str(ckpt.get("object_head_arch") or ocfg.get("head_arch", "shared")),
        use_coordconv=bool(ckpt.get("object_use_coordconv")) or bool(ocfg.get("use_coordconv", False)),
        head_depth=int(ckpt.get("object_head_depth") or ocfg.get("head_depth", 2)),
        predict_bbox2d=bool(ckpt.get("object_predict_bbox2d")) or bool(ocfg.get("predict_bbox2d", False)),
        use_groundplane_prior=bool(ckpt.get("object_use_groundplane_prior"))
        or bool(ocfg.get("use_groundplane_prior", False)),
        groundplane_params=dict(ckpt.get("object_groundplane_params") or ocfg.get("groundplane_params", {}) or {}),
        device=device,
    ).to(device)
    bn = int((ckpt.get("trial") or {}).get("ae_bottleneck", 0))
    if bn > 0:
        sys.path.insert(0, str(AB / "rl_agent" / "feature_ae"))
        from ae_model import build_ae
        hi = int(model.classifier.cbr[0].in_channels)
        model.feature_ae = build_ae(str((ckpt.get("trial") or {}).get("ae_arch", "v2")), hi, bn).to(device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    return model, input_size, bn


def quantize_per_channel(t, bits):
    """Replicates PerChannelFeatureCodec.encode/decode: per-channel min/max, 2^bits-1 levels."""
    B, C = t.shape[0], t.shape[1]
    flat = t.reshape(B, C, -1)
    lo = flat.min(dim=2, keepdim=True).values
    hi = flat.max(dim=2, keepdim=True).values
    span = (hi - lo).clamp(min=1e-12)
    maxlvl = (1 << bits) - 1
    q = (((flat - lo) / span).clamp(0, 1) * maxlvl).round()
    return (q / maxlvl * span + lo).reshape(t.shape)


@torch.no_grad()
def wire_features(model, x, prof, out_hw):
    """Exactly what the attacker intercepts for this profile."""
    feats = model.backbone(x)
    feats = OrderedDict(feats) if not isinstance(feats, torch.Tensor) else OrderedDict([("0", feats)])

    # --- ROI gate (rank-based, matching evaluate_fusion._roi_gate) ---
    q = float(prof["roi"])
    if q > 0:
        high = feats["high"]
        low = feats["low"]
        if bool(getattr(model, "fuse_low_into_object_head", False)):
            h = high
            if tuple(h.shape[-2:]) != tuple(low.shape[-2:]):
                h = F.interpolate(h, size=low.shape[-2:], mode="bilinear", align_corners=False)
            obj_in = torch.cat([low, h], dim=1)
        else:
            obj_in = high
        obj = model.object_head(obj_in)
        n_heat = int(getattr(model, "heatmap_channels", 2))
        objness = torch.sigmoid(obj[:, :n_heat]).amax(dim=1, keepdim=True)
        gated = OrderedDict()
        B = x.shape[0]
        for name, feat in feats.items():
            # rank PER SAMPLE: pooling across the batch would let one frame's objectness
            # decide another frame's dropped cells.
            pooled = F.adaptive_max_pool2d(objness, feat.shape[-2:]).reshape(B, -1).float()
            k = int(round(q * pooled.shape[1]))
            keep = torch.ones_like(pooled)
            if k > 0:
                keep.scatter_(1, pooled.argsort(dim=1)[:, :k], 0.0)
            gated[name] = feat * keep.reshape(B, 1, feat.shape[-2], feat.shape[-1]).to(feat.dtype)
        feats = gated

    # --- AE encode on 'high' only (bottleneck is what is transmitted) ---
    ae = getattr(model, "feature_ae", None)
    low, high = feats["low"], feats["high"]
    if ae is not None:
        high = ae.encode(high)

    # --- per-channel uintN quantization (the wire codec) ---
    bits = int(prof["bits"])
    return quantize_per_channel(low, bits), quantize_per_channel(high, bits)


@torch.no_grad()
def evaluate(dec, model, loader, dev, prof, out_hw, max_batches=40, save=0):
    dec.eval()
    ps, ss, samples = [], [], []
    for bi, (x, tgt) in enumerate(loader):
        if bi >= max_batches:
            break
        x = x.to(dev, non_blocking=True)
        low, high = wire_features(model, x, prof, out_hw)
        rec = dec(low, high)
        if rec.shape[-2:] != out_hw:
            rec = F.interpolate(rec, size=out_hw, mode="bilinear", align_corners=False)
        rec = rec.cpu().numpy().transpose(0, 2, 3, 1)
        gt = tgt.numpy().transpose(0, 2, 3, 1)
        for k in range(rec.shape[0]):
            ps.append(psnr_fn(gt[k], rec[k], data_range=1.0))
            ss.append(ssim_fn(gt[k], rec[k], channel_axis=2, data_range=1.0))
            if len(samples) < save:
                samples.append((gt[k], rec[k]))
    dec.train()
    return float(np.mean(ps)), float(np.mean(ss)), samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, choices=list(PROFILES))
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--train-limit", type=int, default=6000)
    ap.add_argument("--test-limit", type=int, default=320)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--split-mode", default="manifest", choices=("manifest", "temporal"))
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    prof = PROFILES[args.profile]
    dev = torch.device("cuda")
    model, input_size, bn = build_variant_model(prof["model"], dev)
    for p in model.parameters():
        p.requires_grad_(False)
    out_hw = (input_size[1], input_size[0])

    tr = DataLoader(FrameDS("train", input_size, args.train_limit, args.split_mode), batch_size=args.batch, shuffle=True,
                    num_workers=6, pin_memory=True, drop_last=True, persistent_workers=True)
    te = DataLoader(FrameDS("test", input_size, args.test_limit, args.split_mode), batch_size=args.batch, shuffle=False,
                    num_workers=4, pin_memory=True)

    # probe wire shapes
    x0, _ = next(iter(te))
    low0, high0 = wire_features(model, x0[:1].to(dev), prof, out_hw)
    lc, hc = low0.shape[1], high0.shape[1]
    wire_elems = low0.numel() + high0.numel()
    print(f"profile {args.profile}: model={prof['model']} bits={prof['bits']} roi={prof['roi']} "
          f"ae_bottleneck={bn}")
    print(f"  wire tensors: low {tuple(low0.shape)}  high {tuple(high0.shape)}  "
          f"({wire_elems} elems, knob-matrix payload {prof['payload_kb']} KB)")
    print(f"  model accuracy (knob matrix): mIoU {prof['miou']} ped-recall {prof['ped_recall']} "
          f"accept={prof['accept']}")

    dec = InversionDecoder(low_ch=lc, high_ch=hc).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    t0 = time.time()
    deadline = t0 + args.minutes * 60
    step, ep, hist = 0, 0, []
    while time.time() < deadline:
        ep += 1
        for x, tgt in tr:
            if time.time() >= deadline:
                break
            x, tgt = x.to(dev, non_blocking=True), tgt.to(dev, non_blocking=True)
            low, high = wire_features(model, x, prof, out_hw)
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
            if step % 200 == 0:
                print(f"  ep{ep} step{step} loss {loss.item():.4f} ({(time.time()-t0)/60:.1f}/{args.minutes} min)")
        p, s, _ = evaluate(dec, model, te, dev, prof, out_hw, max_batches=10)
        hist.append({"epoch": ep, "step": step, "psnr": round(p, 3), "ssim": round(s, 4)})
        print(f"  [epoch {ep}] test PSNR {p:.2f} dB SSIM {s:.4f}")

    psnr, ssim, samples = evaluate(dec, model, te, dev, prof, out_hw, max_batches=40, save=2)
    print(f"\nFINAL {args.profile}: PSNR {psnr:.2f} dB  SSIM {ssim:.4f}")

    res = {"profile": args.profile, "split_mode": args.split_mode, **{k: v for k, v in prof.items()},
           "ae_bottleneck": bn, "wire_low_ch": int(lc), "wire_high_ch": int(hc),
           "wire_elems": int(wire_elems),
           "attack": {"psnr_db": round(psnr, 3), "ssim": round(ssim, 4), "steps": step,
                      "minutes": round((time.time() - t0) / 60, 2),
                      "train_frames": len(tr.dataset)},
           "history": hist}
    suffix = "" if args.split_mode == "manifest" else f"_{args.split_mode}"
    p_ = OUT / f"E5_profile_{args.profile}{suffix}.json"
    p_.write_text(json.dumps(res, indent=2))
    print(f"wrote {p_}")

    if samples:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(samples), 2, figsize=(9, 3.1 * len(samples)))
        axes = np.atleast_2d(axes)
        for i, (g, r) in enumerate(samples):
            axes[i, 0].imshow(np.clip(g, 0, 1))
            axes[i, 1].imshow(np.clip(r, 0, 1))
            if i == 0:
                axes[i, 0].set_title("original", fontsize=10)
                axes[i, 1].set_title(f"reconstruction — {args.profile} ({prof['payload_kb']} KB)", fontsize=10)
            for a in axes[i]:
                a.axis("off")
        fig.tight_layout()
        fp = OUT / f"E5_samples_{args.profile}{suffix}.png"
        fig.savefig(fp, dpi=140)
        print(f"wrote {fp}")


if __name__ == "__main__":
    main()
