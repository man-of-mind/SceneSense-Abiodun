#!/usr/bin/env python3
"""Task-aware feature-AE trainer for the fusion split point (workstream E).

Trains a FeatureAE {128,64,32} on the frozen M' backbone's 960-ch 'high' feature so the
split-point tensor can be compressed. Objective = task-output DISTILLATION (make M''s frozen
heads produce M'-like outputs from the AE-reconstructed feature) + a small reconstruction
regularizer, all under objectness ROI-drop q~U(0,q_max) so the AE composes with the ROI action
exactly as at inference (composition: features -> ROI-drop -> AE-encode -> AE-decode -> heads).
The 'low' feature (40 ch) passes through uncompressed (it is a small fraction of payload).
The model is fully frozen; gradients flow ONLY to the AE (verified: model grad == 0).

Quantization is NOT in the training loop: the AE is trained continuous, then quant is applied to
the bottleneck at eval (the quant sweep + AE eval measure AE x quant combos on M').

Usage:
  train_ae.py --model-checkpoint <M'.pt> --bottleneck 64 [--epochs 15] [--drop-max 0.8]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ABIODUN = HERE.parent.parent
sys.path.insert(0, str(ABIODUN / "pole_lraspp_multimodal_fusion"))
sys.path.insert(0, str(HERE))
from pole_lraspp_multimodal_fusion.train_fusion import (  # noqa: E402
    read_manifest, split_rows, load_object_boxes, FusionPoleMultiTaskDataset,
    object_reg_channels, OBJECT_CLASS_NAMES)
from pole_lraspp_multimodal_fusion.model import build_multitask_fusion_lraspp  # noqa: E402
from ae_model import build_ae  # noqa: E402

# object-head config M' was trained with (must match the checkpoint for a clean load)
OBJHEADS = dict(heatmap_radius_px=4, fuse_low_feature=True, head_arch="shared", use_coordconv=False,
                head_depth=3, use_groundplane_prior=False, predict_bbox2d=True,
                adaptive_heatmap_radius=True, max_gt_distance_m=40)
DEFAULT_DS = str(ABIODUN / "fusion_training_data/moving_ego_pps200000_merged_8loops_stride2")
DEFAULT_MPRIME = str(ABIODUN / "experiments/mprime_dropaware_20260708/stage2_obj_drop/"
                                "checkpoints/mprime_stage2_obj_drop/best.pt")


def build_frozen_model(ckpt: str, device: str):
    obj_ch = len(OBJECT_CLASS_NAMES) + object_reg_channels(OBJHEADS["predict_bbox2d"])
    m = build_multitask_fusion_lraspp(
        num_classes=3, radar_channels=4, pretrained=False, object_channels=obj_ch,
        object_hidden_channels=128, fuse_low_into_object_head=OBJHEADS["fuse_low_feature"],
        head_arch=OBJHEADS["head_arch"], use_coordconv=OBJHEADS["use_coordconv"],
        head_depth=OBJHEADS["head_depth"], predict_bbox2d=OBJHEADS["predict_bbox2d"]).to(device)
    sd = torch.load(ckpt, map_location=device)
    sd = sd.get("model", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
    miss, unexp = m.load_state_dict(sd, strict=False)
    if miss or unexp:
        print(f"[warn] load_state_dict missing={len(miss)} unexpected={len(unexp)}")
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def heads_from_features(m, feats):
    seg = m.classifier(feats)
    if isinstance(seg, dict):
        seg = seg["out"]
    obj_in = m._object_input(feats)
    if m.head_arch == "decoupled":
        obj = torch.cat([m.heatmap_head(obj_in), m.reg_head(obj_in)], dim=1)
    else:
        obj = m.object_head(obj_in)
    return seg, obj


def _rel_mse(s, t):
    """Scale-free reconstruction error: MSE normalized by the teacher's own magnitude -> O(1),
    so terms of very different logit ranges (seg vs heatmap vs regression) stay comparable and the
    loss weights express real priority rather than accidental scale."""
    return F.mse_loss(s, t) / (t.detach().pow(2).mean() + 1e-6)


def distill_loss(m, seg_s, obj_s, seg_t, obj_t, temp=2.0):
    # segmentation: per-pixel temperature-softened KL (probability space, scale-robust)
    hw = seg_s.shape[-2] * seg_s.shape[-1]
    seg_kl = F.kl_div(F.log_softmax(seg_s / temp, dim=1), F.softmax(seg_t / temp, dim=1),
                      reduction="batchmean") * (temp * temp) / hw
    # object: heatmap distilled in LOGIT space (sigmoid is ~0.01 everywhere -> ignores peaks), and
    # regression, both as relative errors so neither vanishes nor dominates.
    nh = m.heatmap_channels
    heat = _rel_mse(obj_s[:, :nh], obj_t[:, :nh])
    reg = _rel_mse(obj_s[:, nh:], obj_t[:, nh:]) if obj_s.shape[1] > nh else obj_s.new_zeros(())
    return seg_kl, heat, reg


def run_epoch(m, ae, loader, device, drop_max, opt=None, seg_w=1.0, heat_w=1.0, reg_w=0.2, recon_w=0.1,
              max_batches=0):
    train = opt is not None
    ae.train(train)
    agg = {"loss": 0.0, "seg": 0.0, "heat": 0.0, "reg": 0.0, "recon": 0.0, "n": 0}
    for bi, (tensors, _masks, _obj) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        tensors = tensors.to(device, non_blocking=True)
        b = tensors.shape[0]
        q = float(torch.rand(1).item()) * drop_max if drop_max > 0 else 0.0
        with torch.no_grad():
            feats = m.backbone(tensors)
            if q > 0:
                feats = m._objectness_drop(feats, q)
            seg_t, obj_t = heads_from_features(m, feats)
            high = m._high_feature(feats)
        high_hat = ae.decode(ae.encode(high))
        feats_hat = type(feats)((k, (high_hat if k == "high" else v)) for k, v in feats.items())
        seg_s, obj_s = heads_from_features(m, feats_hat)
        seg_kl, heat_mse, reg_l1 = distill_loss(m, seg_s, obj_s, seg_t, obj_t)
        recon = _rel_mse(high_hat, high)
        loss = seg_w * seg_kl + heat_w * heat_mse + reg_w * reg_l1 + recon_w * recon
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        for k, v in (("loss", loss), ("seg", seg_kl), ("heat", heat_mse), ("reg", reg_l1), ("recon", recon)):
            agg[k] += float(v.detach()) * b
        agg["n"] += b
    n = max(1, agg["n"])
    return {k: agg[k] / n for k in ("loss", "seg", "heat", "reg", "recon")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-checkpoint", default=DEFAULT_MPRIME)
    ap.add_argument("--dataset", default=DEFAULT_DS)
    ap.add_argument("--bottleneck", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--drop-max", type=float, default=0.8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--input-size", type=int, nargs=2, default=[768, 432])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=str(HERE / "checkpoints"))
    ap.add_argument("--max-batches", type=int, default=0, help="cap batches/epoch (0=all); for smoke tests")
    # distill weights (default = seg-leaning; object-weighted preset recovers detection through the bottleneck)
    ap.add_argument("--seg-w", type=float, default=1.0)
    ap.add_argument("--heat-w", type=float, default=1.0)
    ap.add_argument("--reg-w", type=float, default=0.2)
    ap.add_argument("--recon-w", type=float, default=0.1)
    ap.add_argument("--tag", default="", help="suffix for checkpoint name, e.g. '_obj' -> ae_b64_obj.pt")
    ap.add_argument("--arch", default="v1", choices=("v1", "v2"),
                    help="v1=linear 1x1 channel projection; v2=nonlinear+spatial (recovers detection at same payload)")
    args = ap.parse_args()

    dev = args.device
    m = build_frozen_model(args.model_checkpoint, dev)
    hi_ch = int(m.classifier.cbr[0].in_channels)
    print(f"model loaded (frozen); high_channels={hi_ch}; bottleneck={args.bottleneck}")

    ds_root = Path(args.dataset)
    rows = split_rows(read_manifest(ds_root / "manifest.csv"))
    obj_rows = load_object_boxes(ds_root / "object_boxes.csv")
    wh = (int(args.input_size[0]), int(args.input_size[1]))
    tr = FusionPoleMultiTaskDataset(ds_root, rows["train"], obj_rows, wh, OBJHEADS, augment_strength="off")
    va = FusionPoleMultiTaskDataset(ds_root, rows["val"], obj_rows, wh, OBJHEADS, augment_strength="off")
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"data: train={len(tr)} val={len(va)}")

    ae = build_ae(args.arch, hi_ch, args.bottleneck).to(dev)
    print(f"AE arch={args.arch}  params={sum(p.numel() for p in ae.parameters()):,}")
    opt = torch.optim.AdamW(ae.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"ae_b{args.bottleneck}{args.tag}.pt"
    wts = dict(seg_w=args.seg_w, heat_w=args.heat_w, reg_w=args.reg_w, recon_w=args.recon_w)
    print(f"distill weights: {wts}")
    log_path = HERE / "ae_train_log.md"
    best = float("inf")
    with open(log_path, "a") as fh:
        fh.write(f"\n## AE b{args.bottleneck} on {Path(args.model_checkpoint).parent.parent.name} "
                 f"(drop_max={args.drop_max}, {args.epochs} ep)\n")
    for ep in range(args.epochs):
        t0 = time.time()
        tr_m = run_epoch(m, ae, tl, dev, args.drop_max, opt=opt, max_batches=args.max_batches, **wts)
        with torch.no_grad():
            va_m = run_epoch(m, ae, vl, dev, args.drop_max, opt=None, max_batches=args.max_batches, **wts)
        line = (f"b{args.bottleneck} ep={ep} train_loss={tr_m['loss']:.4f} val_loss={va_m['loss']:.4f} "
                f"(seg={va_m['seg']:.4f} heat={va_m['heat']:.5f} reg={va_m['reg']:.4f} recon={va_m['recon']:.4f}) "
                f"{time.time()-t0:.0f}s")
        print(line)
        with open(log_path, "a") as fh:
            fh.write(f"- {line}\n")
        if va_m["loss"] < best:
            best = va_m["loss"]
            torch.save({"ae_state": ae.state_dict(), "arch": args.arch, "bottleneck": args.bottleneck, "in_channels": hi_ch,
                        "val_loss": best, "model_checkpoint": args.model_checkpoint,
                        "drop_max": args.drop_max}, best_path)
    print(f"DONE b{args.bottleneck}: best val_loss={best:.4f} -> {best_path}")


if __name__ == "__main__":
    main()
