#!/usr/bin/env python3
"""Summarize Track 2 OAI MCS-policy runs.

Example:

  python3 abiodun/oai_mcs_policy_track2/summarize_policy_runs.py \
    --run P0=downlink_oai_default106_ttracer_fps10_drivable_fast_timingfix_20260731_default106_noae \
    --run P2=downlink_oai_default106_ttracer_fps10_track2_holdfew_20260801_default106_noae
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "oai_mcs_policy_track2" / "results"


def q(series: pd.Series, p: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return float("nan")
    return float(vals.quantile(p))


def find_metrics_csv(run_group: str) -> Path:
    batch_id = run_group.removeprefix("downlink_oai_default106_ttracer_fps10_")
    direct = (
        ROOT
        / "downlink_latency_fps/runs/oai_default106_ttracer"
        / f"fps_10_{batch_id}"
        / "streams"
        / f"{run_group}_metrics.csv"
    )
    if direct.exists():
        return direct
    candidates = sorted(
        (ROOT / "downlink_latency_fps/runs/oai_default106_ttracer").glob(
            f"fps_10_*/streams/{run_group}_metrics.csv"
        )
    )
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"metrics CSV not found for {run_group}")


def grant_summary(run_group: str) -> pd.Series:
    path = (
        ROOT
        / "metrics_logs/scenesense_ttracer"
        / run_group
        / "ue/analysis/nrue_grant_summary.csv"
    )
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path)
    ul = df[df["direction_label"].eq("ul")]
    if ul.empty:
        return pd.Series(dtype=float)
    return ul.iloc[0]


def carla_summary(run_group: str) -> pd.Series:
    path = (
        ROOT
        / "metrics_logs/carla_oai_ttracer"
        / run_group
        / "CARLA10_OAI_TTRACER_SUMMARY.csv"
    )
    if not path.exists():
        return pd.Series(dtype=float)
    return pd.read_csv(path).iloc[0]


def summarize(label: str, run_group: str) -> Dict[str, object]:
    metrics_path = find_metrics_csv(run_group)
    df = pd.read_csv(metrics_path)
    received = pd.to_numeric(df.get("result_received"), errors="coerce").fillna(0.0)
    row: Dict[str, object] = {
        "label": label,
        "run_group": run_group,
        "frames": int(len(df)),
        "returned": int(received.sum()),
        "delivery_pct": 100.0 * float(received.mean()) if len(df) else float("nan"),
        "payload_p50_kib": q(df["feature_payload_bytes"], 0.50) / 1024.0
        if "feature_payload_bytes" in df
        else float("nan"),
        "payload_p95_kib": q(df["feature_payload_bytes"], 0.95) / 1024.0
        if "feature_payload_bytes" in df
        else float("nan"),
    }

    if {
        "capture_to_backbone_input_ms",
        "front_backbone_ms",
        "feature_serialize_ms",
        "capture_to_front_send_ms",
        "round_trip_result_recv_ms",
        "t_edge_recv_wall_s",
        "t_front_send_wall_s",
    }.issubset(df.columns):
        feature_build = (
            pd.to_numeric(df["capture_to_backbone_input_ms"], errors="coerce")
            + pd.to_numeric(df["front_backbone_ms"], errors="coerce")
            + pd.to_numeric(df["feature_serialize_ms"], errors="coerce")
        )
        uplink = (
            pd.to_numeric(df["t_edge_recv_wall_s"], errors="coerce")
            - pd.to_numeric(df["t_front_send_wall_s"], errors="coerce")
        ) * 1000.0
        uplink = uplink.where(uplink >= 0.0)
        capture_to_result = (
            pd.to_numeric(df["capture_to_front_send_ms"], errors="coerce")
            + pd.to_numeric(df["round_trip_result_recv_ms"], errors="coerce")
        )
        row.update(
            {
                "front_build_p50_ms": q(feature_build, 0.50),
                "front_build_p95_ms": q(feature_build, 0.95),
                "uplink_p50_ms": q(uplink, 0.50),
                "uplink_p95_ms": q(uplink, 0.95),
                "capture_to_result_p50_ms": q(capture_to_result, 0.50),
                "capture_to_result_p95_ms": q(capture_to_result, 0.95),
            }
        )
    else:
        csum = carla_summary(run_group)
        row.update(
            {
                "front_build_p50_ms": float(csum.get("front_ms_p50", float("nan"))),
                "front_build_p95_ms": float("nan"),
                "uplink_p50_ms": float(
                    csum.get("feature_upload_payload_handling_ms_p50", float("nan"))
                ),
                "uplink_p95_ms": float("nan"),
                "capture_to_result_p50_ms": float(csum.get("front_ms_p50", 0.0))
                + float(csum.get("rtt_recv_ms_p50", float("nan"))),
                "capture_to_result_p95_ms": float("nan"),
            }
        )

    row.update(
        {
            "edge_tail_p50_ms": q(df["back_ms"], 0.50) if "back_ms" in df else float("nan"),
            "downlink_p50_ms": q(df["result_send_to_recv_ms_perf"], 0.50)
            if "result_send_to_recv_ms_perf" in df
            else float("nan"),
            "rtt_p50_ms": q(df["round_trip_result_recv_ms"], 0.50)
            if "round_trip_result_recv_ms" in df
            else float("nan"),
            "send_call_p50_ms": q(df["send_call_ms"], 0.50)
            if "send_call_ms" in df
            else float("nan"),
        }
    )

    ul = grant_summary(run_group)
    row.update(
        {
            "ul_scheduled_mbps": float(ul.get("scheduled_mbps", float("nan"))),
            "ul_mcs_avg": float(ul.get("avg_mcs", float("nan"))),
            "ul_mcs_p50": float(ul.get("p50_mcs", float("nan"))),
            "ul_mcs_p95": float(ul.get("p95_mcs", float("nan"))),
            "ul_prb_p50": float(ul.get("p50_rb_size", float("nan"))),
            "ul_prb_p95": float(ul.get("p95_rb_size", float("nan"))),
            "ul_avg_tbs_bytes": float(ul.get("avg_tbs_bytes", float("nan"))),
            "ul_p95_tbs_bytes": float(ul.get("p95_tbs_bytes", float("nan"))),
            "ul_retx_rate": float(ul.get("retx_rate", float("nan"))),
        }
    )
    return row


def parse_run(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use LABEL=RUN_GROUP")
    label, run_group = value.split("=", 1)
    label = label.strip()
    run_group = run_group.strip()
    if not label or not run_group:
        raise argparse.ArgumentTypeError("use non-empty LABEL=RUN_GROUP")
    return label, run_group


def to_markdown(df: pd.DataFrame) -> str:
    rounded = df.copy()
    for col in rounded.columns:
        if pd.api.types.is_numeric_dtype(rounded[col]):
            rounded[col] = rounded[col].map(
                lambda v: "" if not math.isfinite(float(v)) else f"{float(v):.3f}"
            )
    headers = list(rounded.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in rounded.iterrows():
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--out-prefix", default="track2_policy_summary")
    args = parser.parse_args()

    rows = [summarize(label, run_group) for label, run_group in args.run]
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{args.out_prefix}.csv"
    md_path = OUT_DIR / f"{args.out_prefix}.md"
    df.to_csv(csv_path, index=False)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Track 2 policy summary\n\n")
        f.write(to_markdown(df))
        f.write("\n")
    print(csv_path)
    print(md_path)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
