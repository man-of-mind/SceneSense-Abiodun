#!/usr/bin/env python3
"""Deterministic aggregation of the static-sweep run folders (payload + front-latency per compression
profile). Safe to run unattended (pure aggregation of the metrics CSVs; no model/RL). Accuracy-vs-
compression is measured separately by the offline eval (validation-heavy; done with a human in the loop).

Outputs: rl_agent/analysis/static_sweep_summary.md + static_sweep_payload.png
"""
from __future__ import annotations
import csv, glob, math, os, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AB = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun")
SWEEP = AB / "metrics_logs" / "rl_static_sweep"
OUT = AB / "rl_agent" / "analysis"; OUT.mkdir(parents=True, exist_ok=True)


def _finite(vals):
    out = []
    for v in vals:
        try:
            f = float(v)
            if math.isfinite(f):
                out.append(f)
        except (TypeError, ValueError):
            pass
    return out


def main():
    variants = sorted(d.name for d in SWEEP.iterdir() if d.is_dir() and not d.name.startswith("_")) \
        if SWEEP.exists() else []
    rows = []
    for v in variants:
        csvs = glob.glob(str(SWEEP / v / "**" / "*_metrics.csv"), recursive=True)
        if not csvs:
            continue
        R = list(csv.DictReader(open(csvs[0])))
        if not R:
            continue
        def col(k): return _finite(r.get(k) for r in R)
        pc = col("feature_payload_bytes"); pu = col("feature_payload_bytes_uncompressed")
        fr = col("front_ms"); nres = sum(1 for r in R if str(r.get("round_trip_ms")).replace(".", "").isdigit())
        rows.append({
            "variant": v, "frames": len(R),
            "quant": (R[0].get("quantization_mode") or "?"), "entropy": (R[0].get("entropy_coder") or "?"),
            "payload_KB_comp": round(st.mean(pc) / 1024, 1) if pc else None,
            "payload_KB_uncomp": round(st.mean(pu) / 1024, 1) if pu else None,
            "compression_x": round(st.mean(pu) / st.mean(pc), 2) if pc and pu else None,
            "front_ms": round(st.mean(fr), 1) if fr else None,
            "frames_with_result": nres,
        })

    md = ["# Static sweep — payload + front-latency by compression profile\n",
          "Deterministic aggregation of the quant×entropy sweep (loopback). Accuracy-vs-compression is a"
          " separate offline eval (deterministic, human-validated). `frames_with_result` shows how often the"
          " loopback result returned (low = the ~1MB payload fragments over UDP; why live accuracy is unreliable).\n",
          "| variant | quant | entropy | payload KB (comp) | KB (uncomp) | compression× | front_ms | frames | w/result |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['variant']} | {r['quant']} | {r['entropy']} | {r['payload_KB_comp']} | "
                  f"{r['payload_KB_uncomp']} | {r['compression_x']} | {r['front_ms']} | {r['frames']} | "
                  f"{r['frames_with_result']} |")
    (OUT / "static_sweep_summary.md").write_text("\n".join(md) + "\n")

    prof = [r for r in rows if r["payload_KB_comp"] is not None]
    if prof:
        fig, ax = plt.subplots(figsize=(10, 5))
        names = [r["variant"] for r in prof]
        ax.bar(range(len(prof)), [r["payload_KB_comp"] for r in prof], color="#00d1ff")
        ax.set_xticks(range(len(prof))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("payload KB/frame (compressed)"); ax.set_title("Static sweep — compressed payload per profile")
        fig.tight_layout(); fig.savefig(OUT / "static_sweep_payload.png", bbox_inches="tight"); plt.close(fig)

    print(f"[sweep_analyze] {len(rows)} variant(s) aggregated -> {OUT}/static_sweep_summary.md")
    print("\n".join(md))


if __name__ == "__main__":
    main()
