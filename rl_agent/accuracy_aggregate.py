#!/usr/bin/env python3
"""Aggregate the offline accuracy-vs-compression sweep into one table (profile -> mIoU / vehicle IoU /
person IoU / object recall + per-class recall / localization MAE). Reads each profile's
evaluate_fusion metrics JSON. Writes rl_agent/analysis/accuracy_vs_compression.md."""
from __future__ import annotations
import json
from pathlib import Path

AB = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun")
SWEEP = AB / "experiments" / "rl_accuracy_sweep"
OUT = AB / "rl_agent" / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["baseline", "q_pchan_u8_zlib", "q_pchan_u8_none", "q_ptensor_u8_zlib", "q_ptensor_u8_none",
         "q_pchan_u6_zlib", "q_pchan_u6_none", "q_pchan_u4_zlib", "q_pchan_u4_none"]


def g(d, k):
    v = d.get(k)
    return round(float(v), 3) if isinstance(v, (int, float)) else "—"


def main():
    rows = []
    for name in ORDER:
        f = SWEEP / name / "metrics" / "test_fusion_evaluation_metrics.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        rows.append((name, d))
    if not rows:
        print("no accuracy-sweep metrics yet — run rl_agent/run_accuracy_sweep.sh first")
        return
    md = ["# Accuracy vs compression (offline, deterministic — 200k fusion model, test split)\n",
          "Model run through the split-inference codec round-trip at each quant profile; uncompressed"
          " baseline routes the model directly. Same eval thresholds (thr 0.10, nms 6, ≤40 m). All numbers"
          " directly comparable (same code path).\n",
          "| profile | mIoU | vehicle IoU | person IoU | obj recall | veh recall | ped recall | veh loc MAE (m) | ped loc MAE (m) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, d in rows:
        md.append(f"| {name} | {g(d,'miou')} | {g(d,'vehicle_iou')} | {g(d,'person_iou')} | "
                  f"{g(d,'learned_object_recall')} | {g(d,'learned_vehicle_object_recall')} | "
                  f"{g(d,'learned_person_object_recall')} | {g(d,'learned_vehicle_global_xy_mae_m')} | "
                  f"{g(d,'learned_person_global_xy_mae_m')} |")
    (OUT / "accuracy_vs_compression.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    # quick validation hint
    base = dict(rows).get("baseline")
    u8 = dict(rows).get("q_pchan_u8_zlib")
    if base and u8:
        dm = abs(g(base, "miou") - g(u8, "miou")) if isinstance(g(base, "miou"), float) and isinstance(g(u8, "miou"), float) else None
        print(f"\n[validation] baseline mIoU {g(base,'miou')} vs uint8 {g(u8,'miou')} "
              f"(should be close — uint8 is near-lossless). ref baseline mIoU=0.837")
    print(f"\n-> {OUT/'accuracy_vs_compression.md'}")


if __name__ == "__main__":
    main()
