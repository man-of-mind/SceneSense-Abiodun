"""E3 (part 1) - measure the REAL uplink payload for each of the 3 architectures.

A. full-local + share detections -> serialized detection list
B. full-offload raw             -> compressed RGB (+ radar) of the native 1280x720 frame
C. split + feature fusion       -> entropy-coded backbone features (the deployed path)

Averaged over N real test-split frames. Emits results/E3_payloads.json.
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd
from PIL import Image

from common_setup import EXP_DIR, FrontWrapper, build_full_model, get_real_input

OUT = Path(__file__).parent / "results"


def kb(n):
    return n / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--zstd-level", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    import carla_split_inference_udp_data_collect as od
    from pole_lraspp_multimodal_fusion.common import read_manifest
    from pole_lraspp_multimodal_fusion.object_targets import decode_objects

    dev = torch.device("cuda")
    model, input_size, cfg = build_full_model(dev)
    front = FrontWrapper(model).eval()
    eval_cfg = cfg.get("evaluation", {})
    dataset_dir = EXP_DIR / "dataset"
    rows = [r for r in read_manifest(dataset_dir / "manifest.csv") if r.get("split") == "test"]
    cctx = zstd.ZstdCompressor(level=args.zstd_level)

    acc = {k: [] for k in (
        "A_det_json_raw", "A_det_json_zstd", "A_n_detections",
        "B_jpeg92", "B_jpeg75", "B_png", "B_radar_npy_zstd", "B_jpeg92_plus_radar",
        "C_feat_fp32_raw", "C_feat_u8_raw", "C_feat_u8_zstd", "C_feat_fp16_uncompressed",
    )}

    for i in range(args.frames):
        x, row, orig_hw = get_real_input(dev, input_size, index=i)

        # ---------- C: split feature payload (the deployed serialize path) ----------
        with torch.inference_mode():
            feats = front(x)
        transport = od.TransportConfig(
            quantization_mode="per_channel_uint8",
            entropy_coder_name="zstd",
            zstd_level=args.zstd_level,
            roi_objectness_threshold=0.0,
            bypass_rcnn_transform=False,
        )
        # per_level_compress_probe=True is REQUIRED: without it serialize_feature_maps
        # returns only the quantized (uncompressed) wire dict and per_level_compressed
        # is empty. The entropy-coded size is what actually goes on the wire.
        serialized, uncompressed_fp16_bytes, per_lvl_unc, per_lvl_comp = od.serialize_feature_maps(
            OrderedDict(feats), {}, quantization_mode=transport.quantization_mode,
            per_level_compress_probe=True, entropy_coder=transport.make_entropy_coder(),
        )
        assert per_lvl_comp, "per_level_compressed empty - compress probe did not run"
        n_elem = sum(v.numel() for v in feats.values())
        acc["C_feat_fp32_raw"].append(n_elem * 4)
        acc["C_feat_u8_raw"].append(int(sum(per_lvl_unc.values())))
        acc["C_feat_u8_zstd"].append(int(sum(per_lvl_comp.values())))
        acc["C_feat_fp16_uncompressed"].append(int(uncompressed_fp16_bytes))

        # ---------- A: full-local, share detections ----------
        with torch.inference_mode():
            out = model(x)
        cam = np.array(json.loads(row["camera_matrix_json"]), dtype=np.float64)
        dets = decode_objects(
            out["object"], camera_matrix=cam,
            topk=int(eval_cfg.get("topk_objects", 80)),
            score_threshold=float(eval_cfg.get("object_score_threshold", 0.03)),
            nms_radius_px=int(eval_cfg.get("object_nms_radius_px", 4)),
        )
        # Round to a sane on-wire precision (cm / 0.001 score) rather than full float repr.
        slim = [{k: (round(float(v), 3) if isinstance(v, (int, float)) else v) for k, v in d.items()}
                for d in dets]
        blob = json.dumps({"dets": slim, "frame": row["sample_id"]}, separators=(",", ":")).encode()
        acc["A_n_detections"].append(len(dets))
        acc["A_det_json_raw"].append(len(blob))
        acc["A_det_json_zstd"].append(len(cctx.compress(blob)))

        # ---------- B: full-offload raw ----------
        img = Image.open(dataset_dir / row["rgb_path"]).convert("RGB")  # native 1280x720
        for q, key in ((92, "B_jpeg92"), (75, "B_jpeg75")):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            acc[key].append(buf.tell())
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        acc["B_png"].append(buf.tell())
        # radar must also be shipped for the edge to run the fusion model
        radar = np.load(dataset_dir / row["radar_tensor_path"])
        if hasattr(radar, "files"):
            radar = radar["radar"]
        r_bytes = len(cctx.compress(np.ascontiguousarray(radar.astype(np.float16)).tobytes()))
        acc["B_radar_npy_zstd"].append(r_bytes)
        acc["B_jpeg92_plus_radar"].append(acc["B_jpeg92"][-1] + r_bytes)

        if i == 0:
            print(f"frame0 {row['sample_id']}  native {img.size}  dets={len(dets)}")

    res = {}
    for k, v in acc.items():
        res[k] = {
            "mean_bytes": round(statistics.mean(v), 1),
            "mean_KB": round(kb(statistics.mean(v)), 2),
            "p50_KB": round(kb(statistics.median(v)), 2),
            "min_KB": round(kb(min(v)), 2),
            "max_KB": round(kb(max(v)), 2),
        }
    res["A_n_detections"] = {"mean": round(statistics.mean(acc["A_n_detections"]), 2),
                             "min": min(acc["A_n_detections"]), "max": max(acc["A_n_detections"])}
    res["_meta"] = {"frames": args.frames, "zstd_level": args.zstd_level,
                    "native_frame": "1280x720", "model_input": "768x432 (7ch)"}
    (OUT / "E3_payloads.json").write_text(json.dumps(res, indent=2))

    print(f"\n== measured uplink payload per frame ({args.frames} real test frames) ==")
    order = [
        ("A: detections JSON (raw)", "A_det_json_raw"),
        ("A: detections JSON (zstd)", "A_det_json_zstd"),
        ("B: RGB JPEG q92", "B_jpeg92"),
        ("B: RGB JPEG q75", "B_jpeg75"),
        ("B: RGB PNG lossless", "B_png"),
        ("B: radar fp16 zstd", "B_radar_npy_zstd"),
        ("B: JPEG q92 + radar", "B_jpeg92_plus_radar"),
        ("C: features fp32 raw", "C_feat_fp32_raw"),
        ("C: features fp16 uncompressed", "C_feat_fp16_uncompressed"),
        ("C: features u8 raw", "C_feat_u8_raw"),
        ("C: features u8 + zstd", "C_feat_u8_zstd"),
    ]
    for label, k in order:
        r = res[k]
        print(f"  {label:30s} {r['mean_KB']:9.2f} KB  (p50 {r['p50_KB']:8.2f}, "
              f"range {r['min_KB']:.2f}-{r['max_KB']:.2f})")
    print(f"  detections/frame: mean {res['A_n_detections']['mean']}")
    print(f"\nwrote {OUT/'E3_payloads.json'}")


if __name__ == "__main__":
    main()
