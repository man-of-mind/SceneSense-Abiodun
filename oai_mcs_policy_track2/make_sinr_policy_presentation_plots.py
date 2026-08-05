#!/usr/bin/env python3
"""Presentation plots for the closed-loop CARLA OAI MCS-policy comparison.

This script intentionally compares only the 106PRB / 10FPS / no-AE / ROI0 /
zstd closed-loop CARLA runs used for the Track-2 scheduler story:

  - clean channel: vanilla OAI, AIMD, SINR lookup
  - mild AWGN:     vanilla OAI, AIMD, SINR lookup

For time-series plots, samples are aligned to the CARLA frontend active window
from the run log. That avoids accidentally plotting attach/setup time as if it
were application behavior.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "oai_mcs_policy_track2" / "results" / "presentation_sinr_policy"
BIN_S = 1.0


@dataclass(frozen=True)
class RunSpec:
    channel: str
    policy: str
    abbr: str
    run_group: str


RUNS: List[RunSpec] = [
    RunSpec(
        "Clean",
        "Vanilla OAI",
        "Vanilla",
        "downlink_oai_default106_fair_clear_vanilla_fps10_track2_sinr_clear_20260803_clear_vanilla",
    ),
    RunSpec(
        "Clean",
        "AIMD",
        "AIMD",
        "downlink_oai_default106_fair_clear_aimd_cap_fps10_track2_fair_grant_20260801_clear_aimd_cap",
    ),
    RunSpec(
        "Clean",
        "SINR lookup",
        "SINR",
        "downlink_oai_default106_fair_clear_sinr_fps10_track2_sinr_clear_20260803_clear_sinr",
    ),
    RunSpec(
        "Mild AWGN",
        "Vanilla OAI",
        "Vanilla",
        "downlink_oai_default106_awgn_mild_track2_vanilla_fps10_track2_sinr_awgn_ladder_20260803_mild_vanilla",
    ),
    RunSpec(
        "Mild AWGN",
        "AIMD",
        "AIMD",
        "downlink_oai_default106_fair_mild_awgn_aimd_cap_fps10_track2_fair_grant_20260801_mild_aimd_cap",
    ),
    RunSpec(
        "Mild AWGN",
        "SINR lookup",
        "SINR",
        "downlink_oai_default106_awgn_mild_track2_sinr_fps10_track2_sinr_awgn_ladder_20260803_mild_sinr",
    ),
]

POLICIES = ["Vanilla OAI", "AIMD", "SINR lookup"]
CHANNELS = ["Clean", "Mild AWGN"]
COLORS = {
    "Vanilla OAI": "#D55E00",
    "AIMD": "#0072B2",
    "SINR lookup": "#009E73",
}
MARKERS = {
    "Vanilla OAI": "o",
    "AIMD": "s",
    "SINR lookup": "^",
}


def q(series: pd.Series, p: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return float("nan")
    return float(vals.quantile(p))


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def hms_to_seconds(value: str) -> float:
    hour, minute, second = value.strip().split(":")
    return int(hour) * 3600.0 + int(minute) * 60.0 + float(second)


def parse_front_active_interval(run_group: str) -> Tuple[float, float]:
    path = ROOT / "metrics_logs" / "carla_oai_ttracer" / run_group / "run.log"
    if not path.exists():
        raise FileNotFoundError(f"run log not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    start = re.search(r"\[(\d\d:\d\d:\d\d)\] running CARLA frontend", text)
    end = re.search(r"\[(\d\d:\d\d:\d\d)\] front completed", text)
    if not start or not end:
        raise ValueError(f"could not parse CARLA active interval from {path}")
    start_s = hms_to_seconds(start.group(1) + ".000000")
    end_s = hms_to_seconds(end.group(1) + ".000000")
    if end_s < start_s:
        end_s += 24.0 * 3600.0
    return start_s, end_s


def tracer_seconds(series: pd.Series, start_s: float, end_s: float) -> pd.Series:
    vals = series.astype(str).map(hms_to_seconds).astype(float)
    # If a run crosses midnight, the active window end is already shifted by
    # 24h. Shift trace samples after midnight into the same coordinate system.
    if end_s >= 24.0 * 3600.0:
        vals = vals.where(vals >= start_s, vals + 24.0 * 3600.0)
    return vals


def local_iso_seconds(series: pd.Series, start_s: float, end_s: float) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    vals = (
        dt.dt.hour * 3600.0
        + dt.dt.minute * 60.0
        + dt.dt.second
        + dt.dt.microsecond / 1_000_000.0
    )
    if end_s >= 24.0 * 3600.0:
        vals = vals.where(vals >= start_s, vals + 24.0 * 3600.0)
    return vals


def find_metrics_csv(run_group: str) -> Path:
    candidates = sorted((ROOT / "downlink_latency_fps" / "runs").glob(f"*/fps_10_*/streams/{run_group}_metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"frontend metrics CSV not found for {run_group}")
    return candidates[-1]


def active_bins(start_s: float, end_s: float) -> pd.DataFrame:
    n = max(1, int(math.ceil((end_s - start_s) / BIN_S)))
    return pd.DataFrame({"bin": np.arange(n, dtype=int), "t": np.arange(n, dtype=float) * BIN_S})


def load_active_app_rate(run_group: str, start_s: float, end_s: float) -> pd.DataFrame:
    df = safe_read_csv(find_metrics_csv(run_group))
    bins = active_bins(start_s, end_s)
    if df.empty or not {"wall_time_iso", "feature_payload_bytes"}.issubset(df.columns):
        bins["app_mbps"] = 0.0
        return bins
    df = df.copy()
    df["_t_abs"] = local_iso_seconds(df["wall_time_iso"], start_s, end_s)
    active = df[(df["_t_abs"] >= start_s) & (df["_t_abs"] <= end_s)].copy()
    if active.empty:
        bins["app_mbps"] = 0.0
        return bins
    active["bin"] = np.floor((active["_t_abs"] - start_s) / BIN_S).astype(int)
    active["payload"] = pd.to_numeric(active["feature_payload_bytes"], errors="coerce").fillna(0.0)
    agg = active.groupby("bin", as_index=False).agg(app_bytes=("payload", "sum"), app_frames=("payload", "size"))
    bins = bins.merge(agg, on="bin", how="left")
    bins["app_bytes"] = bins["app_bytes"].fillna(0.0)
    bins["app_frames"] = bins["app_frames"].fillna(0.0)
    bins["app_mbps"] = bins["app_bytes"] * 8.0 / BIN_S / 1_000_000.0
    return bins


def load_active_grants(run_group: str, start_s: float, end_s: float) -> pd.DataFrame:
    path = ROOT / "metrics_logs" / "scenesense_ttracer" / run_group / "ue" / "csv" / "NRUE_MAC_DCI_GRANT.csv"
    usecols = lambda c: c in {"time", "direction", "tbs", "mcs", "rb_size", "rv", "round"}
    df = safe_read_csv(path, usecols=usecols)
    bins = active_bins(start_s, end_s)
    if df.empty:
        return bins.assign(
            grant_rate_hz=0.0,
            scheduled_mbps=0.0,
            first_tx_mbps=0.0,
            retx_mbps=0.0,
            retx_rate_pct=0.0,
            mcs_p50=np.nan,
            mcs_avg=np.nan,
            rb_p50=np.nan,
            tbs_avg_bytes=np.nan,
        )
    for col in ["direction", "tbs", "mcs", "rb_size", "rv", "round"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_t_abs"] = tracer_seconds(df["time"], start_s, end_s)
    ul = df[(df["direction"] == 1) & (df["_t_abs"] >= start_s) & (df["_t_abs"] <= end_s)].copy()
    if ul.empty:
        return bins.assign(
            grant_rate_hz=0.0,
            scheduled_mbps=0.0,
            first_tx_mbps=0.0,
            retx_mbps=0.0,
            retx_rate_pct=0.0,
            mcs_p50=np.nan,
            mcs_avg=np.nan,
            rb_p50=np.nan,
            tbs_avg_bytes=np.nan,
        )
    ul["bin"] = np.floor((ul["_t_abs"] - start_s) / BIN_S).astype(int)
    ul["tbs"] = pd.to_numeric(ul["tbs"], errors="coerce").fillna(0.0)
    rv = pd.to_numeric(ul.get("rv", 0), errors="coerce").fillna(0.0)
    harq_round = pd.to_numeric(ul.get("round", 0), errors="coerce").fillna(0.0)
    ul["is_retx"] = (rv > 0) | (harq_round > 0)
    ul["first_tbs"] = ul["tbs"].where(~ul["is_retx"], 0.0)
    ul["retx_tbs"] = ul["tbs"].where(ul["is_retx"], 0.0)
    agg = ul.groupby("bin", as_index=False).agg(
        grants=("tbs", "size"),
        total_tbs=("tbs", "sum"),
        first_tbs=("first_tbs", "sum"),
        retx_tbs=("retx_tbs", "sum"),
        retx_rate=("is_retx", "mean"),
        mcs_p50=("mcs", "median"),
        mcs_avg=("mcs", "mean"),
        rb_p50=("rb_size", "median"),
        tbs_avg_bytes=("tbs", "mean"),
    )
    bins = bins.merge(agg, on="bin", how="left")
    for col in ["grants", "total_tbs", "first_tbs", "retx_tbs", "retx_rate"]:
        bins[col] = bins[col].fillna(0.0)
    bins["grant_rate_hz"] = bins["grants"] / BIN_S
    bins["scheduled_mbps"] = bins["total_tbs"] * 8.0 / BIN_S / 1_000_000.0
    bins["first_tx_mbps"] = bins["first_tbs"] * 8.0 / BIN_S / 1_000_000.0
    bins["retx_mbps"] = bins["retx_tbs"] * 8.0 / BIN_S / 1_000_000.0
    bins["retx_rate_pct"] = bins["retx_rate"] * 100.0
    return bins


def load_active_queue(run_group: str, start_s: float, end_s: float) -> pd.DataFrame:
    bins = active_bins(start_s, end_s)
    rlc_path = ROOT / "metrics_logs" / "scenesense_ttracer" / run_group / "ue" / "csv" / "NRUE_MAC_RLC_BUFFER_STATUS.csv"
    bsr_path = ROOT / "metrics_logs" / "scenesense_ttracer" / run_group / "ue" / "csv" / "NRUE_MAC_BSR_STATUS.csv"
    deq_path = ROOT / "metrics_logs" / "scenesense_ttracer" / run_group / "ue" / "csv" / "NR_RLC_TX_DEQUEUE.csv"

    rlc = safe_read_csv(rlc_path, usecols=lambda c: c in {"time", "lcid", "bytes_in_buffer"})
    if not rlc.empty:
        rlc["lcid"] = pd.to_numeric(rlc["lcid"], errors="coerce")
        rlc["bytes_in_buffer"] = pd.to_numeric(rlc["bytes_in_buffer"], errors="coerce")
        rlc["_t_abs"] = tracer_seconds(rlc["time"], start_s, end_s)
        rlc = rlc[(rlc["lcid"] == 4) & (rlc["_t_abs"] >= start_s) & (rlc["_t_abs"] <= end_s)].copy()
        if not rlc.empty:
            rlc["bin"] = np.floor((rlc["_t_abs"] - start_s) / BIN_S).astype(int)
            agg = rlc.groupby("bin", as_index=False).agg(
                rlc_p50_kib=("bytes_in_buffer", lambda s: q(s, 0.50) / 1024.0),
                rlc_p95_kib=("bytes_in_buffer", lambda s: q(s, 0.95) / 1024.0),
            )
            bins = bins.merge(agg, on="bin", how="left")

    bsr = safe_read_csv(bsr_path, usecols=lambda c: c in {"time", "lcg1_bytes", "bsr_sent"})
    if not bsr.empty:
        bsr["lcg1_bytes"] = pd.to_numeric(bsr["lcg1_bytes"], errors="coerce").fillna(0.0)
        bsr["_t_abs"] = tracer_seconds(bsr["time"], start_s, end_s)
        bsr = bsr[(bsr["_t_abs"] >= start_s) & (bsr["_t_abs"] <= end_s)].copy()
        if not bsr.empty:
            bsr["bin"] = np.floor((bsr["_t_abs"] - start_s) / BIN_S).astype(int)
            agg = bsr.groupby("bin", as_index=False).agg(
                bsr_lcg1_p50_kib=("lcg1_bytes", lambda s: q(s, 0.50) / 1024.0),
                bsr_lcg1_p95_kib=("lcg1_bytes", lambda s: q(s, 0.95) / 1024.0),
            )
            bins = bins.merge(agg, on="bin", how="left")

    deq = safe_read_csv(deq_path, usecols=lambda c: c in {"time", "lcid", "pdu_bytes"})
    if not deq.empty:
        deq["lcid"] = pd.to_numeric(deq["lcid"], errors="coerce")
        deq["pdu_bytes"] = pd.to_numeric(deq["pdu_bytes"], errors="coerce").fillna(0.0)
        deq["_t_abs"] = tracer_seconds(deq["time"], start_s, end_s)
        deq = deq[(deq["lcid"] == 4) & (deq["_t_abs"] >= start_s) & (deq["_t_abs"] <= end_s)].copy()
        if not deq.empty:
            deq["bin"] = np.floor((deq["_t_abs"] - start_s) / BIN_S).astype(int)
            agg = deq.groupby("bin", as_index=False).agg(rlc_drain_bytes=("pdu_bytes", "sum"))
            bins = bins.merge(agg, on="bin", how="left")

    for col in ["rlc_p50_kib", "rlc_p95_kib", "bsr_lcg1_p50_kib", "bsr_lcg1_p95_kib", "rlc_drain_bytes"]:
        if col not in bins:
            bins[col] = 0.0
        bins[col] = bins[col].fillna(0.0)
    bins["rlc_drain_mbps"] = bins["rlc_drain_bytes"] * 8.0 / BIN_S / 1_000_000.0
    return bins


def load_active_snr(run_group: str, start_s: float, end_s: float) -> pd.DataFrame:
    path = ROOT / "metrics_logs" / "scenesense_ttracer" / run_group / "gnb" / "csv" / "GNB_MAC_UL_MCS_DECISION.csv"
    df = safe_read_csv(path, usecols=lambda c: c in {"time", "avg_snr_x10"})
    bins = active_bins(start_s, end_s)
    if df.empty:
        bins["snr_db"] = np.nan
        return bins
    df["snr_db"] = pd.to_numeric(df["avg_snr_x10"], errors="coerce") / 10.0
    df["_t_abs"] = tracer_seconds(df["time"], start_s, end_s)
    df = df[(df["_t_abs"] >= start_s) & (df["_t_abs"] <= end_s)].copy()
    if df.empty:
        bins["snr_db"] = np.nan
        return bins
    df["bin"] = np.floor((df["_t_abs"] - start_s) / BIN_S).astype(int)
    agg = df.groupby("bin", as_index=False).agg(snr_db=("snr_db", "median"))
    return bins.merge(agg, on="bin", how="left")


def load_active_bler(run_group: str, start_s: float, end_s: float) -> pd.DataFrame:
    path = ROOT / "metrics_logs" / "scenesense_ttracer" / run_group / "gnb" / "csv" / "GNB_MAC_BLER_MCS_DECISION.csv"
    df = safe_read_csv(path)
    bins = active_bins(start_s, end_s)
    if df.empty:
        bins["filtered_bler_pct"] = np.nan
        return bins
    for col in df.columns:
        if col != "time":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "direction" in df:
        df = df[df["direction"] == 1].copy()
    if "updated" in df:
        df = df[df["updated"] == 1].copy()
    if df.empty or "bler_after_ppm" not in df:
        bins["filtered_bler_pct"] = np.nan
        return bins
    df["filtered_bler_pct"] = pd.to_numeric(df["bler_after_ppm"], errors="coerce") / 10000.0
    df["_t_abs"] = tracer_seconds(df["time"], start_s, end_s)
    df = df[(df["_t_abs"] >= start_s) & (df["_t_abs"] <= end_s)].copy()
    if df.empty:
        bins["filtered_bler_pct"] = np.nan
        return bins
    df["bin"] = np.floor((df["_t_abs"] - start_s) / BIN_S).astype(int)
    agg = df.groupby("bin", as_index=False).agg(filtered_bler_pct=("filtered_bler_pct", "median"))
    return bins.merge(agg, on="bin", how="left")


def grant_summary(run_group: str) -> Dict[str, float]:
    path = ROOT / "metrics_logs" / "scenesense_ttracer" / run_group / "ue" / "analysis" / "nrue_grant_summary.csv"
    df = safe_read_csv(path)
    if df.empty or "direction_label" not in df:
        return {}
    ul = df[df["direction_label"].eq("ul")]
    if ul.empty:
        return {}
    row = ul.iloc[0]
    return {
        "overall_ul_sched_mbps": float(row.get("scheduled_mbps", float("nan"))),
        "overall_ul_first_tx_mbps": float(row.get("first_tx_mbps", float("nan"))),
        "overall_ul_retx_mbps": float(row.get("retx_mbps", float("nan"))),
        "overall_ul_grant_rate_hz": float(row.get("grant_rate_hz", float("nan"))),
        "overall_mcs_avg": float(row.get("avg_mcs", float("nan"))),
        "overall_mcs_p50": float(row.get("p50_mcs", float("nan"))),
        "overall_mcs_p95": float(row.get("p95_mcs", float("nan"))),
        "overall_tbs_avg_bytes": float(row.get("avg_tbs_bytes", float("nan"))),
        "overall_tbs_p95_bytes": float(row.get("p95_tbs_bytes", float("nan"))),
        "overall_retx_rate_pct": 100.0 * float(row.get("retx_rate", float("nan"))),
    }


def queue_summary(run_group: str) -> Dict[str, float]:
    path = ROOT / "metrics_logs" / "scenesense_ttracer" / run_group / "ue" / "analysis" / "nrue_queue_summary.csv"
    df = safe_read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0]
    return {
        "overall_rlc_buffer_p95_kib": float(row.get("rlc_total_buffer_p95_bytes", float("nan"))) / 1024.0,
        "overall_bsr_lcg_p95_kib": float(row.get("bsr_total_lcg_p95_bytes", float("nan"))) / 1024.0,
        "overall_rlc_sdu_drain_mbps": float(row.get("sdu_mbps", float("nan"))),
    }


def snr_summary(run_group: str) -> Dict[str, float]:
    path = ROOT / "metrics_logs" / "scenesense_ttracer" / run_group / "gnb" / "csv" / "GNB_MAC_UL_MCS_DECISION.csv"
    df = safe_read_csv(path, usecols=lambda c: c in {"avg_snr_x10"})
    if df.empty or "avg_snr_x10" not in df:
        return {}
    snr = pd.to_numeric(df["avg_snr_x10"], errors="coerce") / 10.0
    return {"snr_p50_db": q(snr, 0.50), "snr_p05_db": q(snr, 0.05), "snr_p95_db": q(snr, 0.95)}


def bler_summary(run_group: str) -> Dict[str, float]:
    path = ROOT / "metrics_logs" / "scenesense_ttracer" / run_group / "gnb" / "csv" / "GNB_MAC_BLER_MCS_DECISION.csv"
    df = safe_read_csv(path)
    if df.empty:
        return {"ul_olla_bler_status": "missing"}
    for col in df.columns:
        if col != "time":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "direction" in df:
        df = df[df["direction"] == 1].copy()
    if "updated" in df:
        df = df[df["updated"] == 1].copy()
    if df.empty:
        return {"ul_olla_bler_status": "N/A"}
    bler = pd.to_numeric(df.get("bler_after_ppm"), errors="coerce") / 10000.0
    window = pd.to_numeric(df.get("bler_window_ppm"), errors="coerce") / 10000.0
    branch = pd.to_numeric(df.get("branch"), errors="coerce")
    return {
        "ul_olla_bler_status": "available",
        "filtered_bler_p50_pct": q(bler, 0.50),
        "filtered_bler_p95_pct": q(bler, 0.95),
        "window_bler_p95_pct": q(window, 0.95),
        "sparse_branch_pct": 100.0 * float((branch == 3).mean()) if len(branch) else float("nan"),
        "decrease_branch_count": float((branch == 2).sum()),
    }


def parse_layer_markdown(run_group: str) -> Dict[str, float]:
    path = ROOT / "metrics_logs" / "scenesense_ttracer" / run_group / "layer_latency" / "uplink_layer_latency.md"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "rlc_queue_little_ms": r"RLC mean queueing delay .*?:\*\*[^0-9]*([0-9.]+) ms",
        "pdcp_to_gnb_p50_ms": r"UE PDCP-ingress -> gNB PDCP-deliver .*?p50=([0-9.]+)",
        "pdcp_to_gnb_p95_ms": r"UE PDCP-ingress -> gNB PDCP-deliver .*?p95=([0-9.]+)",
    }
    out: Dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            out[key] = float(match.group(1))
    return out


def active_summary(run_group: str) -> Dict[str, float]:
    start_s, end_s = parse_front_active_interval(run_group)
    grants = load_active_grants(run_group, start_s, end_s)
    queue = load_active_queue(run_group, start_s, end_s)
    app = load_active_app_rate(run_group, start_s, end_s)
    snr = load_active_snr(run_group, start_s, end_s)
    bler = load_active_bler(run_group, start_s, end_s)
    out: Dict[str, float] = {
        "active_duration_s": end_s - start_s,
        "active_app_mbps_mean": float(app["app_mbps"].mean()) if "app_mbps" in app else float("nan"),
        "active_app_mbps_p95": q(app.get("app_mbps", pd.Series(dtype=float)), 0.95),
        "active_app_frames_per_s": float(app.get("app_frames", pd.Series(dtype=float)).sum() / max(end_s - start_s, 1e-9))
        if "app_frames" in app
        else float("nan"),
        "active_ul_sched_mbps_mean": float(grants["scheduled_mbps"].mean()),
        "active_ul_sched_mbps_p95": q(grants["scheduled_mbps"], 0.95),
        "active_ul_first_tx_mbps_mean": float(grants["first_tx_mbps"].mean()),
        "active_ul_retx_mbps_mean": float(grants["retx_mbps"].mean()),
        "active_ul_grant_rate_hz_mean": float(grants["grant_rate_hz"].mean()),
        "active_ul_retx_rate_pct_mean": float(grants["retx_rate_pct"].mean()),
        "active_mcs_p50": q(grants["mcs_p50"], 0.50),
        "active_mcs_p95": q(grants["mcs_p50"], 0.95),
        "active_tbs_avg_bytes": q(grants["tbs_avg_bytes"], 0.50),
        "active_rlc_buffer_p95_kib": q(queue["rlc_p95_kib"], 0.95),
        "active_bsr_lcg1_p95_kib": q(queue["bsr_lcg1_p95_kib"], 0.95),
        "active_rlc_drain_mbps_mean": float(queue["rlc_drain_mbps"].mean()),
        "active_snr_p50_db": q(snr["snr_db"], 0.50),
        "active_filtered_bler_p95_pct": q(bler["filtered_bler_pct"], 0.95),
    }
    return out


def summarize_run(spec: RunSpec) -> Dict[str, object]:
    df = pd.read_csv(find_metrics_csv(spec.run_group))
    received = pd.to_numeric(df.get("result_received"), errors="coerce").fillna(0.0)
    row: Dict[str, object] = {
        "channel": spec.channel,
        "policy": spec.policy,
        "abbr": spec.abbr,
        "run_group": spec.run_group,
        "frames": int(len(df)),
        "returned": int(received.sum()),
        "delivery_pct": 100.0 * float(received.mean()) if len(df) else float("nan"),
        "payload_p50_kib": q(df["feature_payload_bytes"], 0.50) / 1024.0 if "feature_payload_bytes" in df else float("nan"),
        "edge_tail_p50_ms": q(df["back_ms"], 0.50) if "back_ms" in df else float("nan"),
        "downlink_p50_ms": q(df["result_send_to_recv_ms_perf"], 0.50) if "result_send_to_recv_ms_perf" in df else float("nan"),
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
                "uplink_p50_ms": q(uplink, 0.50),
                "uplink_p95_ms": q(uplink, 0.95),
                "capture_result_p50_ms": q(capture_to_result, 0.50),
                "capture_result_p95_ms": q(capture_to_result, 0.95),
            }
        )
    row.update(grant_summary(spec.run_group))
    row.update(queue_summary(spec.run_group))
    row.update(snr_summary(spec.run_group))
    row.update(bler_summary(spec.run_group))
    row.update(parse_layer_markdown(spec.run_group))
    row.update(active_summary(spec.run_group))
    return row


def format_float(value: object) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
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


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.24, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10, width=1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def savefig(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=280)
    plt.close(fig)


def plot_summary(df: pd.DataFrame) -> None:
    metrics = [
        ("active_snr_p50_db", "Measured SNR", "dB"),
        ("active_mcs_p50", "MCS", "p50"),
        ("active_app_mbps_mean", "App send rate", "Mbps"),
        ("active_ul_sched_mbps_mean", "Scheduled UL", "Mbps"),
        ("uplink_p50_ms", "Uplink latency", "p50 ms"),
        ("active_rlc_buffer_p95_kib", "RLC backlog", "p95 KiB"),
        ("active_ul_retx_rate_pct_mean", "Retransmissions", "% grants"),
        ("delivery_pct", "Frame delivery", "%"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.9))
    x = np.arange(len(CHANNELS))
    width = 0.23
    for ax, (col, title, unit) in zip(axes.flat, metrics):
        for idx, policy in enumerate(POLICIES):
            vals = []
            for channel in CHANNELS:
                rows = df[(df["channel"] == channel) & (df["policy"] == policy)]
                vals.append(float(rows[col].iloc[0]) if not rows.empty and col in rows else np.nan)
            bars = ax.bar(x + (idx - 1) * width, vals, width=width, color=COLORS[policy], label=policy)
            for bar, val in zip(bars, vals):
                if math.isfinite(val) and col in {"active_mcs_p50", "uplink_p50_ms", "delivery_pct"}:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height(),
                        f"{val:.0f}" if abs(val) >= 10 else f"{val:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.set_ylabel(unit, fontsize=10.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(CHANNELS, fontsize=10, fontweight="bold")
        style_axes(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        prop={"weight": "bold", "size": 11},
    )
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    savefig(fig, "closedloop_policy_summary_bars")


def policy_legend() -> List[Line2D]:
    return [
        Line2D([0], [0], color=COLORS[p], marker=MARKERS[p], linewidth=2.8, label=p)
        for p in POLICIES
    ]


def plot_timeseries(channel: str, specs: List[RunSpec], max_t: float) -> None:
    fig, axes = plt.subplots(7, 1, figsize=(13.2, 15.8), sharex=True)
    for spec in specs:
        start_s, end_s = parse_front_active_interval(spec.run_group)
        color = COLORS[spec.policy]
        marker = MARKERS[spec.policy]
        grants = load_active_grants(spec.run_group, start_s, end_s)
        queue = load_active_queue(spec.run_group, start_s, end_s)
        app = load_active_app_rate(spec.run_group, start_s, end_s)
        snr = load_active_snr(spec.run_group, start_s, end_s)

        grants = grants[grants["t"] <= max_t]
        queue = queue[queue["t"] <= max_t]
        app = app[app["t"] <= max_t]
        snr = snr[snr["t"] <= max_t]
        markevery = max(1, len(grants) // 9)

        axes[0].plot(grants["t"], grants["mcs_p50"], color=color, linewidth=2.7, marker=marker, markevery=markevery)
        axes[1].plot(snr["t"], snr["snr_db"], color=color, linewidth=2.5)
        axes[2].plot(app["t"], app["app_mbps"], color=color, linewidth=2.4)
        axes[3].plot(grants["t"], grants["scheduled_mbps"], color=color, linewidth=2.7)
        axes[4].plot(queue["t"], queue["rlc_p95_kib"], color=color, linewidth=2.5)
        axes[5].plot(queue["t"], queue["bsr_lcg1_p95_kib"], color=color, linewidth=2.5)
        axes[6].plot(grants["t"], grants["grant_rate_hz"], color=color, linewidth=2.5)

    titles = [
        "MCS assigned to uplink grants",
        "gNB measured uplink SNR",
        "Application feature payload offered to OAI",
        "Scheduled uplink service rate",
        "UE RLC LCID4 buffer occupancy",
        "UE BSR LCG1 backlog reported to gNB",
        "Uplink grants per second",
    ]
    ylabels = ["MCS", "dB", "Mbps", "Mbps", "KiB", "KiB", "grants/s"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontsize=12.6, fontweight="bold", loc="left", pad=7)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        style_axes(ax)
    axes[-1].set_xlabel("Time from CARLA frontend start (s)", fontsize=12, fontweight="bold")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    axes[6].set_ylim(bottom=0)

    fig.legend(
        handles=policy_legend(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.998),
        ncol=3,
        frameon=False,
        prop={"weight": "bold", "size": 11},
    )
    axes[0].text(
        0.0,
        1.28,
        f"{channel}: closed-loop CARLA radio/queue dynamics",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.942))
    stem = "closedloop_policy_timeseries_" + channel.lower().replace(" ", "_").replace("-", "_")
    savefig(fig, stem)


def plot_latency_breakdown(df: pd.DataFrame) -> None:
    cols = [
        ("front_build_p50_ms", "front build"),
        ("uplink_p50_ms", "uplink"),
        ("edge_tail_p50_ms", "edge tail"),
        ("downlink_p50_ms", "downlink"),
    ]
    rows = df.copy()
    rows["label"] = rows["channel"] + "\n" + rows["policy"]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    left = np.zeros(len(rows))
    colors = ["#9E9E9E", "#56B4E9", "#CC79A7", "#F0E442"]
    for (col, label), color in zip(cols, colors):
        vals = pd.to_numeric(rows[col], errors="coerce").fillna(0.0).to_numpy()
        ax.barh(y, vals, left=left, height=0.68, color=color, label=label)
        for idx, val in enumerate(vals):
            if val >= 8.0:
                ax.text(left[idx] + val / 2.0, idx, f"{val:.0f}", ha="center", va="center", fontsize=8.5, fontweight="bold")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(rows["label"], fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("p50 latency component (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Closed-loop latency breakdown across MCS policies", fontsize=15, fontweight="bold", pad=10)
    ax.legend(loc="lower right", frameon=False, ncol=4, prop={"weight": "bold", "size": 10})
    style_axes(ax)
    savefig(fig, "closedloop_policy_latency_breakdown")


def choose_common_window(specs: Iterable[RunSpec], cap_s: float = 160.0) -> float:
    durations = []
    for spec in specs:
        start_s, end_s = parse_front_active_interval(spec.run_group)
        durations.append(end_s - start_s)
    if not durations:
        return cap_s
    return max(30.0, min(cap_s, math.floor(min(durations) / 10.0) * 10.0))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.1,
        }
    )

    rows = [summarize_run(spec) for spec in RUNS]
    df = pd.DataFrame(rows)
    preferred = [
        "channel",
        "policy",
        "frames",
        "returned",
        "delivery_pct",
        "payload_p50_kib",
        "active_duration_s",
        "active_app_frames_per_s",
        "active_app_mbps_mean",
        "active_snr_p50_db",
        "active_mcs_p50",
        "active_mcs_p95",
        "active_ul_sched_mbps_mean",
        "active_ul_first_tx_mbps_mean",
        "active_ul_retx_mbps_mean",
        "active_ul_retx_rate_pct_mean",
        "active_ul_grant_rate_hz_mean",
        "active_tbs_avg_bytes",
        "active_rlc_buffer_p95_kib",
        "active_bsr_lcg1_p95_kib",
        "active_rlc_drain_mbps_mean",
        "active_filtered_bler_p95_pct",
        "front_build_p50_ms",
        "uplink_p50_ms",
        "uplink_p95_ms",
        "capture_result_p50_ms",
        "capture_result_p95_ms",
        "edge_tail_p50_ms",
        "downlink_p50_ms",
        "pdcp_to_gnb_p50_ms",
        "pdcp_to_gnb_p95_ms",
        "rlc_queue_little_ms",
        "sparse_branch_pct",
        "decrease_branch_count",
        "ul_olla_bler_status",
        "run_group",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    csv_path = OUT_DIR / "closedloop_policy_presentation_summary.csv"
    md_path = OUT_DIR / "closedloop_policy_presentation_summary.md"
    df.to_csv(csv_path, index=False)
    md = "# Closed-loop CARLA MCS policy presentation summary\n\n"
    md += "Scope: 106PRB, 10FPS, no-AE, ROI0, zstd closed-loop CARLA runs.\n\n"
    md += "Note: AIMD rows are from the fair-grant batch; vanilla/SINR rows use the latest SINR validation batches.\n\n"
    md += "Active-window metrics are aligned to the CARLA frontend start/finish timestamps, not t-tracer attach/setup time.\n\n"
    md += to_markdown(df) + "\n"
    md_path.write_text(md, encoding="utf-8")

    plot_summary(df)
    plot_latency_breakdown(df)
    for channel in CHANNELS:
        specs = [spec for spec in RUNS if spec.channel == channel]
        plot_timeseries(channel, specs, max_t=choose_common_window(specs))

    print(OUT_DIR)
    print(csv_path)
    print(md_path)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
