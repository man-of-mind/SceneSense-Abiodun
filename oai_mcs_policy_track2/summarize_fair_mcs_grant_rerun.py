#!/usr/bin/env python3
"""Summarize the fair MCS/grant-rate rerun.

This companion to run_fair_mcs_grant_rerun.sh focuses on the variables needed
to explain the confusing case where MCS can be high while latency remains high:

  - MCS and PRB allocation
  - TBS/grant
  - grants per second
  - scheduled Mbps vs first-transmission Mbps vs retransmission Mbps
  - frontend latency and RLC queue/drain summaries

The script is intentionally tied to the fair-run naming scheme so we do not
accidentally compare old clear-channel runs against newer AWGN runs with
different trace profiles.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "oai_mcs_policy_track2" / "results"


RUN_CONFIGS: Dict[str, Dict[str, str]] = {
    "clear_vanilla": {
        "channel": "clear",
        "policy": "vanilla",
        "condition": "oai_default106_fair_clear_vanilla",
        "awgn_noise_power_dB": "",
    },
    "clear_aimd_cap": {
        "channel": "clear",
        "policy": "aimd_cap",
        "condition": "oai_default106_fair_clear_aimd_cap",
        "awgn_noise_power_dB": "",
    },
    "clear_sinr": {
        "channel": "clear",
        "policy": "sinr",
        "condition": "oai_default106_fair_clear_sinr",
        "awgn_noise_power_dB": "",
    },
    "mild_vanilla": {
        "channel": "mild_awgn",
        "policy": "vanilla",
        "condition": "oai_default106_fair_mild_awgn_vanilla",
        "awgn_noise_power_dB": "-10",
    },
    "mild_aimd_cap": {
        "channel": "mild_awgn",
        "policy": "aimd_cap",
        "condition": "oai_default106_fair_mild_awgn_aimd_cap",
        "awgn_noise_power_dB": "-10",
    },
    "mild_sinr": {
        "channel": "mild_awgn",
        "policy": "sinr",
        "condition": "oai_default106_fair_mild_awgn_sinr",
        "awgn_noise_power_dB": "-10",
    },
    "medium_vanilla": {
        "channel": "medium_awgn",
        "policy": "vanilla",
        "condition": "oai_default106_fair_medium_awgn_vanilla",
        "awgn_noise_power_dB": "-5",
    },
    "medium_aimd_cap": {
        "channel": "medium_awgn",
        "policy": "aimd_cap",
        "condition": "oai_default106_fair_medium_awgn_aimd_cap",
        "awgn_noise_power_dB": "-5",
    },
    "medium_sinr": {
        "channel": "medium_awgn",
        "policy": "sinr",
        "condition": "oai_default106_fair_medium_awgn_sinr",
        "awgn_noise_power_dB": "-5",
    },
}

DEFAULT_RUNS = "clear_vanilla clear_aimd_cap mild_vanilla mild_aimd_cap"


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


def parse_hms_to_seconds(value: str) -> float:
    hour, minute, second = value.strip().split(":")
    return int(hour) * 3600.0 + int(minute) * 60.0 + float(second)


def local_iso_seconds(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.hour * 3600.0 + dt.dt.minute * 60.0 + dt.dt.second + dt.dt.microsecond / 1_000_000.0


def fmt(value: object) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return ""
    return f"{f:.3f}"


def run_group(base_batch: str, label: str) -> str:
    cfg = RUN_CONFIGS[label]
    return f"downlink_{cfg['condition']}_fps10_{base_batch}_{label}"


def metrics_csv(run_group_name: str, condition: str, batch: str) -> Path:
    return (
        ROOT
        / "downlink_latency_fps"
        / "runs"
        / condition
        / f"fps_10_{batch}"
        / "streams"
        / f"{run_group_name}_metrics.csv"
    )


def parse_layer_markdown(run_group_name: str) -> Dict[str, float]:
    path = (
        ROOT
        / "metrics_logs"
        / "scenesense_ttracer"
        / run_group_name
        / "layer_latency"
        / "uplink_layer_latency.md"
    )
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    out: Dict[str, float] = {}
    patterns = {
        "rlc_sdu_drain_mbps": r"SDU drain: .*? = ([0-9.]+) Mbps",
        "rlc_queue_little_ms": r"RLC mean queueing delay .*?:\*\*[^0-9]*([0-9.]+) ms",
        "pdcp_to_gnb_mean_ms": r"UE PDCP-ingress -> gNB PDCP-deliver .*?mean=([0-9.]+) ms",
        "pdcp_to_gnb_p50_ms": r"UE PDCP-ingress -> gNB PDCP-deliver .*?p50=([0-9.]+)",
        "pdcp_to_gnb_p95_ms": r"UE PDCP-ingress -> gNB PDCP-deliver .*?p95=([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            out[key] = float(match.group(1))
    return out


def parse_front_active_interval(run_group_name: str) -> Tuple[float, float]:
    path = ROOT / "metrics_logs" / "carla_oai_ttracer" / run_group_name / "run.log"
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    start = re.search(r"\[(\d\d:\d\d:\d\d)\] running CARLA frontend", text)
    end = re.search(r"\[(\d\d:\d\d:\d\d)\] front completed", text)
    if not start or not end:
        raise ValueError(f"could not parse CARLA active interval from {path}")
    start_s = parse_hms_to_seconds(start.group(1) + ".000000")
    end_s = parse_hms_to_seconds(end.group(1) + ".000000")
    if end_s < start_s:
        end_s += 24 * 3600.0
    return start_s, end_s


def tracer_time_seconds(series: pd.Series) -> pd.Series:
    return series.astype(str).map(parse_hms_to_seconds)


def active_window_summary(run_group_name: str, metrics_df: pd.DataFrame) -> Dict[str, float]:
    try:
        start_s, end_s = parse_front_active_interval(run_group_name)
    except (OSError, ValueError):
        return {}
    duration_s = max(end_s - start_s, 1e-9)

    out: Dict[str, float] = {
        "active_front_wall_s": duration_s,
    }

    if {"wall_time_iso", "feature_payload_bytes"}.issubset(metrics_df.columns):
        local_s = local_iso_seconds(metrics_df["wall_time_iso"])
        payload = pd.to_numeric(metrics_df["feature_payload_bytes"], errors="coerce").fillna(0.0)
        active = (local_s >= start_s) & (local_s <= end_s)
        out["active_app_mbps"] = float(payload[active].sum() * 8.0 / duration_s / 1_000_000.0)
        out["active_app_frames_per_s"] = float(active.sum() / duration_s)

    trace_base = ROOT / "metrics_logs" / "scenesense_ttracer" / run_group_name / "ue" / "csv"
    grants = safe_read_csv(trace_base / "NRUE_MAC_DCI_GRANT.csv")
    if not grants.empty and {"time", "direction", "tbs", "mcs", "rb_size"}.issubset(grants.columns):
        grants = grants.copy()
        grants["_t"] = tracer_time_seconds(grants["time"])
        ul = grants[(pd.to_numeric(grants["direction"], errors="coerce") == 1) & (grants["_t"] >= start_s) & (grants["_t"] <= end_s)].copy()
        if not ul.empty:
            tbs = pd.to_numeric(ul["tbs"], errors="coerce").fillna(0.0)
            mcs = pd.to_numeric(ul["mcs"], errors="coerce")
            rb = pd.to_numeric(ul["rb_size"], errors="coerce")
            rv = pd.to_numeric(ul.get("rv", 0), errors="coerce").fillna(0)
            harq_round = pd.to_numeric(ul.get("round", 0), errors="coerce").fillna(0)
            retx_mask = (rv > 0) | (harq_round > 0)
            first_tbs = tbs.where(~retx_mask, 0.0)
            retx_tbs = tbs.where(retx_mask, 0.0)
            out.update(
                {
                    "active_ul_grant_hz": float(len(ul) / duration_s),
                    "active_ul_scheduled_mbps": float(tbs.sum() * 8.0 / duration_s / 1_000_000.0),
                    "active_ul_first_tx_mbps": float(first_tbs.sum() * 8.0 / duration_s / 1_000_000.0),
                    "active_ul_retx_mbps": float(retx_tbs.sum() * 8.0 / duration_s / 1_000_000.0),
                    "active_ul_retx_rate_pct": float(100.0 * retx_mask.mean()),
                    "active_ul_avg_mcs": float(mcs.mean()),
                    "active_ul_p50_mcs": q(mcs, 0.50),
                    "active_ul_p95_mcs": q(mcs, 0.95),
                    "active_ul_avg_rb_size": float(rb.mean()),
                    "active_ul_p50_rb_size": q(rb, 0.50),
                    "active_ul_p95_rb_size": q(rb, 0.95),
                    "active_ul_full_prb_grant_pct": float(100.0 * (rb == 106).mean()),
                    "active_ul_avg_tbs_bytes": float(tbs.mean()),
                    "active_ul_p50_tbs_bytes": q(tbs, 0.50),
                    "active_ul_p95_tbs_bytes": q(tbs, 0.95),
                }
            )

            # 100 ms bins show the relationship during the periods where RLC
            # actually has data queued; this avoids blaming idle time.
            bin_s = 0.1
            num_bins = max(1, int(math.ceil(duration_s / bin_s)))
            ul["bin"] = ((ul["_t"] - start_s) // bin_s).astype(int)
            ul["first_tbs"] = first_tbs
            grant_bins = ul.groupby("bin").agg(
                tbs=("tbs", "sum"),
                first_tbs=("first_tbs", "sum"),
                grants=("tbs", "size"),
                mcs=("mcs", "mean"),
                tbs_avg=("tbs", "mean"),
            )

            rlc = safe_read_csv(trace_base / "NRUE_MAC_RLC_BUFFER_STATUS.csv")
            if not rlc.empty and {"time", "lcid", "bytes_in_buffer"}.issubset(rlc.columns):
                rlc = rlc.copy()
                rlc["_t"] = tracer_time_seconds(rlc["time"])
                lcid = pd.to_numeric(rlc["lcid"], errors="coerce")
                rlc4 = rlc[(lcid == 4) & (rlc["_t"] >= start_s) & (rlc["_t"] <= end_s)].copy()
                if not rlc4.empty:
                    rlc4["bin"] = ((rlc4["_t"] - start_s) // bin_s).astype(int)
                    rlc_bins = rlc4.groupby("bin")["bytes_in_buffer"].mean()
                    bins = pd.DataFrame(index=range(num_bins))
                    bins["tbs"] = grant_bins["tbs"]
                    bins["grants"] = grant_bins["grants"]
                    bins["mcs"] = grant_bins["mcs"]
                    bins["tbs_avg"] = grant_bins["tbs_avg"]
                    bins["rlc"] = rlc_bins
                    bins = bins.fillna({"tbs": 0.0, "grants": 0.0, "rlc": 0.0})
                    backlog_bins = bins[bins["rlc"] > 0]
                    if not backlog_bins.empty:
                        out.update(
                            {
                                "active_rlc_nonzero_pct_100ms": float(100.0 * len(backlog_bins) / len(bins)),
                                "active_sched_mbps_when_rlc_nonzero": float(
                                    backlog_bins["tbs"].sum() * 8.0 / (len(backlog_bins) * bin_s) / 1_000_000.0
                                ),
                                "active_grant_hz_when_rlc_nonzero": float(backlog_bins["grants"].mean() / bin_s),
                                "active_mcs_when_rlc_nonzero": float(backlog_bins["mcs"].mean()),
                                "active_tbs_when_rlc_nonzero": float(backlog_bins["tbs_avg"].mean()),
                                "active_rlc_occ_mean_kib_when_nonzero": float(backlog_bins["rlc"].mean() / 1024.0),
                                "active_rlc_occ_p95_kib_when_nonzero": float(backlog_bins["rlc"].quantile(0.95) / 1024.0),
                            }
                        )

    bsr = safe_read_csv(trace_base / "NRUE_MAC_BSR_STATUS.csv")
    if not bsr.empty and {"time", "sdu_bytes"}.issubset(bsr.columns):
        bsr = bsr.copy()
        bsr["_t"] = tracer_time_seconds(bsr["time"])
        active_bsr = bsr[(bsr["_t"] >= start_s) & (bsr["_t"] <= end_s)]
        if not active_bsr.empty:
            sdu_bytes = pd.to_numeric(active_bsr["sdu_bytes"], errors="coerce").fillna(0.0)
            out["active_rlc_sdu_drain_mbps"] = float(sdu_bytes.sum() * 8.0 / duration_s / 1_000_000.0)

    return out


def grant_summary(run_group_name: str) -> Dict[str, float]:
    path = (
        ROOT
        / "metrics_logs"
        / "scenesense_ttracer"
        / run_group_name
        / "ue"
        / "analysis"
        / "nrue_grant_summary.csv"
    )
    df = safe_read_csv(path)
    if df.empty or "direction_label" not in df:
        return {}
    ul = df[df["direction_label"].eq("ul")]
    if ul.empty:
        return {}
    row = ul.iloc[0]
    out: Dict[str, float] = {}
    for col in [
        "duration_s",
        "grant_rate_hz",
        "scheduled_mbps",
        "first_tx_grant_rate_hz",
        "first_tx_mbps",
        "retx_mbps",
        "new_data_grant_rate_hz",
        "new_data_mbps",
        "avg_mcs",
        "p50_mcs",
        "p95_mcs",
        "avg_rb_size",
        "p50_rb_size",
        "p95_rb_size",
        "full_prb_grant_pct",
        "avg_tbs_bytes",
        "p95_tbs_bytes",
        "retx_rate",
    ]:
        if col in row:
            out[f"ul_{col}"] = float(row[col])
    if "retx_rate" in row:
        out["ul_retx_rate_pct"] = 100.0 * float(row["retx_rate"])
    return out


def snr_summary(run_group_name: str) -> Dict[str, float]:
    path = (
        ROOT
        / "metrics_logs"
        / "scenesense_ttracer"
        / run_group_name
        / "gnb"
        / "csv"
        / "GNB_MAC_UL_MCS_DECISION.csv"
    )
    df = safe_read_csv(path)
    if df.empty or "avg_snr_x10" not in df:
        return {}
    snr = pd.to_numeric(df["avg_snr_x10"], errors="coerce") / 10.0
    return {
        "snr_p05_db": q(snr, 0.05),
        "snr_p50_db": q(snr, 0.50),
        "snr_p95_db": q(snr, 0.95),
    }


def summarize_one(base_batch: str, label: str) -> Dict[str, object]:
    cfg = RUN_CONFIGS[label]
    batch = f"{base_batch}_{label}"
    rg = run_group(base_batch, label)
    path = metrics_csv(rg, cfg["condition"], batch)
    if not path.exists():
        raise FileNotFoundError(f"metrics CSV not found for {label}: {path}")

    df = pd.read_csv(path)
    received = pd.to_numeric(df.get("result_received"), errors="coerce").fillna(0.0)
    row: Dict[str, object] = {
        "label": label,
        "channel": cfg["channel"],
        "policy": cfg["policy"],
        "awgn_noise_power_dB": cfg["awgn_noise_power_dB"],
        "run_group": rg,
        "frames": int(len(df)),
        "returned": int(received.sum()),
        "delivery_pct": 100.0 * float(received.mean()) if len(df) else float("nan"),
    }
    if "feature_payload_bytes" in df:
        row["payload_p50_kib"] = q(df["feature_payload_bytes"], 0.50) / 1024.0

    if {"capture_to_backbone_input_ms", "front_backbone_ms", "feature_serialize_ms"}.issubset(df.columns):
        feature_build = (
            pd.to_numeric(df["capture_to_backbone_input_ms"], errors="coerce")
            + pd.to_numeric(df["front_backbone_ms"], errors="coerce")
            + pd.to_numeric(df["feature_serialize_ms"], errors="coerce")
        )
        row["front_build_p50_ms"] = q(feature_build, 0.50)
        row["front_build_p95_ms"] = q(feature_build, 0.95)

    if {"t_edge_recv_wall_s", "t_front_send_wall_s"}.issubset(df.columns):
        uplink = (
            pd.to_numeric(df["t_edge_recv_wall_s"], errors="coerce")
            - pd.to_numeric(df["t_front_send_wall_s"], errors="coerce")
        ) * 1000.0
        uplink = uplink.where(uplink >= 0.0)
        row["uplink_p50_ms"] = q(uplink, 0.50)
        row["uplink_p95_ms"] = q(uplink, 0.95)

    if {"capture_to_front_send_ms", "round_trip_result_recv_ms"}.issubset(df.columns):
        capture_result = (
            pd.to_numeric(df["capture_to_front_send_ms"], errors="coerce")
            + pd.to_numeric(df["round_trip_result_recv_ms"], errors="coerce")
        )
        row["capture_result_p50_ms"] = q(capture_result, 0.50)
        row["capture_result_p95_ms"] = q(capture_result, 0.95)

    if "back_ms" in df:
        row["edge_tail_p50_ms"] = q(df["back_ms"], 0.50)
    if "result_send_to_recv_ms_perf" in df:
        row["downlink_p50_ms"] = q(df["result_send_to_recv_ms_perf"], 0.50)

    row.update(active_window_summary(rg, df))
    row.update(grant_summary(rg))
    row.update(snr_summary(rg))
    row.update(parse_layer_markdown(rg))
    return row


def to_markdown(df: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    for _, row in df.iterrows():
        vals = [fmt(row[col]) if pd.api.types.is_numeric_dtype(df[col]) else str(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-batch", required=True)
    parser.add_argument("--runs", default=DEFAULT_RUNS)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--out-prefix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = args.runs.split()
    unknown = [x for x in labels if x not in RUN_CONFIGS]
    if unknown:
        raise SystemExit(f"unknown run labels: {', '.join(unknown)}")

    rows: List[Dict[str, object]] = []
    missing: List[str] = []
    for label in labels:
        try:
            rows.append(summarize_one(args.base_batch, label))
        except FileNotFoundError as exc:
            missing.append(str(exc))

    if missing and not args.allow_missing:
        msg = [
            "Missing fair rerun artifacts; refusing to create a complete summary.",
            "",
            "Missing:",
            *[f"- {item}" for item in missing],
            "",
            "Use --allow-missing only for an explicit partial summary.",
        ]
        raise SystemExit("\n".join(msg))
    if not rows:
        raise SystemExit("no runs could be summarized")

    df = pd.DataFrame(rows)
    preferred = [
        "label",
        "channel",
        "policy",
        "awgn_noise_power_dB",
        "frames",
        "returned",
        "delivery_pct",
        "payload_p50_kib",
        "front_build_p50_ms",
        "uplink_p50_ms",
        "uplink_p95_ms",
        "capture_result_p50_ms",
        "capture_result_p95_ms",
        "snr_p50_db",
        "active_front_wall_s",
        "active_app_frames_per_s",
        "active_app_mbps",
        "active_ul_scheduled_mbps",
        "active_ul_first_tx_mbps",
        "active_ul_retx_mbps",
        "active_ul_grant_hz",
        "active_ul_avg_mcs",
        "active_ul_p50_mcs",
        "active_ul_avg_tbs_bytes",
        "active_ul_p95_tbs_bytes",
        "active_ul_p50_rb_size",
        "active_ul_full_prb_grant_pct",
        "active_ul_retx_rate_pct",
        "active_rlc_sdu_drain_mbps",
        "active_rlc_nonzero_pct_100ms",
        "active_sched_mbps_when_rlc_nonzero",
        "active_grant_hz_when_rlc_nonzero",
        "active_mcs_when_rlc_nonzero",
        "active_tbs_when_rlc_nonzero",
        "active_rlc_occ_mean_kib_when_nonzero",
        "active_rlc_occ_p95_kib_when_nonzero",
        "ul_scheduled_mbps",
        "ul_first_tx_mbps",
        "ul_retx_mbps",
        "ul_grant_rate_hz",
        "ul_first_tx_grant_rate_hz",
        "ul_avg_mcs",
        "ul_p50_mcs",
        "ul_p95_mcs",
        "ul_avg_rb_size",
        "ul_p50_rb_size",
        "ul_p95_rb_size",
        "ul_full_prb_grant_pct",
        "ul_avg_tbs_bytes",
        "ul_p95_tbs_bytes",
        "ul_retx_rate_pct",
        "rlc_sdu_drain_mbps",
        "rlc_queue_little_ms",
        "pdcp_to_gnb_p50_ms",
        "pdcp_to_gnb_p95_ms",
        "edge_tail_p50_ms",
        "downlink_p50_ms",
        "run_group",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"fair_mcs_grant_rerun_{args.base_batch}"
    csv_path = OUT_DIR / f"{prefix}.csv"
    md_path = OUT_DIR / f"{prefix}.md"
    df.to_csv(csv_path, index=False)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Fair MCS/grant-rate rerun summary\n\n")
        handle.write(f"Base batch: `{args.base_batch}`\n\n")
        handle.write(to_markdown(df))
        handle.write("\n")
        if missing:
            handle.write("\n## Missing runs\n\n")
            for item in missing:
                handle.write(f"- {item}\n")

    print(csv_path)
    print(md_path)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
