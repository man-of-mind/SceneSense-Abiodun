#!/usr/bin/env python3
"""Summarize Track-2 SINR-policy uplink-only OAI ladder runs."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "uplink_only_spatial_map_pipeline" / "results"

RUN_CONFIGS = {
    "clear_sinr": {
        "profile": "clear",
        "condition": "track2_uplink_only_sinr_clear",
        "noise_power_dB": "",
    },
    "mild_sinr": {
        "profile": "mild",
        "condition": "track2_uplink_only_sinr_mild_awgn",
        "noise_power_dB": "-10",
    },
    "mid15_sinr": {
        "profile": "mid15",
        "condition": "track2_uplink_only_sinr_mid15_awgn",
        "noise_power_dB": "-8",
    },
    "strong_sinr": {
        "profile": "strong",
        "condition": "track2_uplink_only_sinr_strong_awgn",
        "noise_power_dB": "-4",
    },
}


def q(series: pd.Series, p: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return float("nan")
    return float(vals.quantile(p))


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def run_group(base_batch: str, label: str) -> str:
    cfg = RUN_CONFIGS[label]
    batch = f"{base_batch}_{label}"
    return f"track1_{cfg['condition']}_fps10_{batch}"


def run_dir(base_batch: str, label: str) -> Path:
    cfg = RUN_CONFIGS[label]
    batch = f"{base_batch}_{label}"
    return ROOT / "uplink_only_spatial_map_pipeline" / "runs" / cfg["condition"] / f"fps_10_{batch}"


def find_send_events(path: Path) -> Path:
    candidates = sorted((path / "front_metrics" / "streams").glob("*_queue_probe_send_events.csv"))
    if not candidates:
        raise FileNotFoundError(f"send-events CSV not found under {path}")
    return candidates[-1]


def grant_summary(rg: str) -> Dict[str, float]:
    path = ROOT / "metrics_logs" / "scenesense_ttracer" / rg / "ue" / "analysis" / "nrue_grant_summary.csv"
    df = safe_read_csv(path)
    if df.empty or "direction_label" not in df:
        return {}
    ul = df[df["direction_label"].eq("ul")]
    if ul.empty:
        return {}
    row = ul.iloc[0]
    return {
        "ul_sched_mbps": float(row.get("scheduled_mbps", float("nan"))),
        "ul_first_tx_mbps": float(row.get("first_tx_mbps", float("nan"))),
        "ul_retx_mbps": float(row.get("retx_mbps", float("nan"))),
        "ul_grant_rate_hz": float(row.get("grant_rate_hz", float("nan"))),
        "ul_new_data_grant_rate_hz": float(row.get("new_data_grant_rate_hz", float("nan"))),
        "mcs_avg": float(row.get("avg_mcs", float("nan"))),
        "mcs_p50": float(row.get("p50_mcs", float("nan"))),
        "mcs_p95": float(row.get("p95_mcs", float("nan"))),
        "prb_p50": float(row.get("p50_rb_size", float("nan"))),
        "tbs_avg_bytes": float(row.get("avg_tbs_bytes", float("nan"))),
        "tbs_p95_bytes": float(row.get("p95_tbs_bytes", float("nan"))),
        "retx_rate_pct": 100.0 * float(row.get("retx_rate", float("nan"))),
    }


def queue_summary(rg: str) -> Dict[str, float]:
    path = ROOT / "metrics_logs" / "scenesense_ttracer" / rg / "ue" / "analysis" / "nrue_queue_summary.csv"
    df = safe_read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0]
    return {
        "rlc_buffer_p50_kib": float(row.get("rlc_total_buffer_p50_bytes", float("nan"))) / 1024.0,
        "rlc_buffer_p95_kib": float(row.get("rlc_total_buffer_p95_bytes", float("nan"))) / 1024.0,
        "rlc_buffer_max_kib": float(row.get("rlc_total_buffer_max_bytes", float("nan"))) / 1024.0,
        "bsr_lcg_p50_kib": float(row.get("bsr_total_lcg_p50_bytes", float("nan"))) / 1024.0,
        "bsr_lcg_p95_kib": float(row.get("bsr_total_lcg_p95_bytes", float("nan"))) / 1024.0,
        "bsr_lcg_max_kib": float(row.get("bsr_total_lcg_max_bytes", float("nan"))) / 1024.0,
        "rlc_sdu_drain_mbps": float(row.get("sdu_mbps", float("nan"))),
    }


def snr_summary(rg: str) -> Dict[str, float]:
    path = ROOT / "metrics_logs" / "scenesense_ttracer" / rg / "gnb" / "csv" / "GNB_MAC_UL_MCS_DECISION.csv"
    df = safe_read_csv(path)
    if df.empty or "avg_snr_x10" not in df:
        return {}
    snr = pd.to_numeric(df["avg_snr_x10"], errors="coerce") / 10.0
    return {
        "snr_p50_db": q(snr, 0.50),
        "snr_p05_db": q(snr, 0.05),
        "snr_p95_db": q(snr, 0.95),
    }


def parse_layer_markdown(rg: str) -> Dict[str, float]:
    path = ROOT / "metrics_logs" / "scenesense_ttracer" / rg / "layer_latency" / "uplink_layer_latency.md"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "rlc_queue_little_ms": r"RLC mean queueing delay .*?:\*\*[^0-9]*([0-9.]+) ms",
        "pdcp_to_gnb_mean_ms": r"UE PDCP-ingress -> gNB PDCP-deliver .*?mean=([0-9.]+) ms",
        "pdcp_to_gnb_p50_ms": r"UE PDCP-ingress -> gNB PDCP-deliver .*?p50=([0-9.]+)",
        "pdcp_to_gnb_p95_ms": r"UE PDCP-ingress -> gNB PDCP-deliver .*?p95=([0-9.]+)",
    }
    out: Dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            out[key] = float(match.group(1))
    return out


def summarize_one(base_batch: str, label: str) -> Dict[str, object]:
    cfg = RUN_CONFIGS[label]
    rg = run_group(base_batch, label)
    rd = run_dir(base_batch, label)
    send = safe_read_csv(find_send_events(rd))
    edge = safe_read_csv(rd / "edge_uplink_metrics.csv")
    if send.empty:
        raise FileNotFoundError(f"empty send-events CSV for {label}: {rd}")
    if edge.empty:
        raise FileNotFoundError(f"empty edge metrics CSV for {label}: {rd / 'edge_uplink_metrics.csv'}")

    attempted = int(send["frame_id"].nunique()) if "frame_id" in send else int(len(send))
    edge_frames = int(edge["frame_id"].nunique()) if "frame_id" in edge else int(len(edge))
    delivery_pct = 100.0 * edge_frames / attempted if attempted else float("nan")

    row: Dict[str, object] = {
        "label": label,
        "profile": cfg["profile"],
        "policy": "sinr",
        "noise_power_dB": cfg["noise_power_dB"],
        "run_group": rg,
        "attempted_frames": attempted,
        "edge_frames": edge_frames,
        "edge_delivery_pct": delivery_pct,
        "payload_p50_kib": q(send["feature_payload_bytes"], 0.50) / 1024.0
        if "feature_payload_bytes" in send
        else float("nan"),
        "front_build_p50_ms": q(
            pd.to_numeric(send.get("capture_to_backbone_input_ms"), errors="coerce")
            + pd.to_numeric(send.get("front_backbone_ms"), errors="coerce")
            + pd.to_numeric(send.get("feature_serialize_ms"), errors="coerce"),
            0.50,
        )
        if {"capture_to_backbone_input_ms", "front_backbone_ms", "feature_serialize_ms"}.issubset(send.columns)
        else float("nan"),
        "front_to_edge_p50_ms": q(edge["front_to_edge_ms"], 0.50) if "front_to_edge_ms" in edge else float("nan"),
        "front_to_edge_p95_ms": q(edge["front_to_edge_ms"], 0.95) if "front_to_edge_ms" in edge else float("nan"),
        "tail_p50_ms": q(edge["tail_ms"], 0.50) if "tail_ms" in edge else float("nan"),
        "backbone_to_tail_p50_ms": q(edge["backbone_input_to_tail_done_ms"], 0.50)
        if "backbone_input_to_tail_done_ms" in edge
        else float("nan"),
        "capture_to_map_publish_p50_ms": q(edge["capture_to_map_publish_ms"], 0.50)
        if "capture_to_map_publish_ms" in edge
        else float("nan"),
        "udp_partial_messages_dropped_max": float(pd.to_numeric(edge.get("udp_partial_messages_dropped"), errors="coerce").max())
        if "udp_partial_messages_dropped" in edge
        else float("nan"),
        "edge_queue_dropped_max": float(pd.to_numeric(edge.get("edge_receive_queue_dropped"), errors="coerce").max())
        if "edge_receive_queue_dropped" in edge
        else float("nan"),
        "spatial_publisher_dropped_max": float(pd.to_numeric(edge.get("spatial_publisher_dropped"), errors="coerce").max())
        if "spatial_publisher_dropped" in edge
        else float("nan"),
    }
    row.update(grant_summary(rg))
    row.update(queue_summary(rg))
    row.update(snr_summary(rg))
    row.update(parse_layer_markdown(rg))
    return row


def format_float(v: object) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "N/A"
    return f"{f:.3f}"


def to_markdown(df: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    for _, row in df.iterrows():
        vals = [format_float(row[col]) if pd.api.types.is_numeric_dtype(df[col]) else str(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def parse_run(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use LABEL=RUN_GROUP for manual entries")
    label, rg = value.split("=", 1)
    return label.strip(), rg.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-batch", required=True)
    parser.add_argument("--runs", default="clear_sinr mild_sinr mid15_sinr strong_sinr")
    parser.add_argument("--out-prefix", default="")
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    for label in args.runs.split():
        if label not in RUN_CONFIGS:
            raise SystemExit(f"unknown label {label}; choices={sorted(RUN_CONFIGS)}")
        rows.append(summarize_one(args.base_batch, label))

    df = pd.DataFrame(rows)
    preferred = [
        "label",
        "profile",
        "policy",
        "noise_power_dB",
        "attempted_frames",
        "edge_frames",
        "edge_delivery_pct",
        "payload_p50_kib",
        "snr_p50_db",
        "mcs_p50",
        "mcs_p95",
        "ul_sched_mbps",
        "ul_first_tx_mbps",
        "retx_rate_pct",
        "ul_grant_rate_hz",
        "front_build_p50_ms",
        "front_to_edge_p50_ms",
        "front_to_edge_p95_ms",
        "tail_p50_ms",
        "backbone_to_tail_p50_ms",
        "capture_to_map_publish_p50_ms",
        "rlc_queue_little_ms",
        "pdcp_to_gnb_p50_ms",
        "rlc_buffer_p95_kib",
        "bsr_lcg_p95_kib",
        "rlc_sdu_drain_mbps",
        "udp_partial_messages_dropped_max",
        "edge_queue_dropped_max",
        "spatial_publisher_dropped_max",
        "run_group",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = args.out_prefix or f"track2_sinr_uplink_only_ladder_{args.base_batch}"
    csv_path = OUT_DIR / f"{suffix}.csv"
    md_path = OUT_DIR / f"{suffix}.md"
    df.to_csv(csv_path, index=False)
    md = "# Track-2 SINR uplink-only ladder summary\n\n"
    md += f"Base batch: `{args.base_batch}`\n\n"
    md += to_markdown(df) + "\n"
    md_path.write_text(md, encoding="utf-8")
    print(csv_path)
    print(md_path)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
