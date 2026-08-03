#!/usr/bin/env python3
"""Summarize non-CARLA vanilla clear/AWGN OAI 106PRB traffic runs.

This is the companion to run_noncarla_vanilla_awgn_106prb.sh.  It is designed
to answer one question cleanly:

    Does mild AWGN produce the same "high-ish MCS but low useful drain / high
    queue" pattern for iperf or tractor replay as it did for CARLA traffic?
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "oai_mcs_policy_track2" / "results"

RUN_CONFIGS: Dict[str, Dict[str, str]] = {
    "iperf_clear": {
        "traffic": "iperf",
        "channel": "clear",
        "condition": "noncarla_iperf_clear_vanilla",
        "awgn_noise_power_dB": "",
    },
    "iperf_mild": {
        "traffic": "iperf",
        "channel": "mild_awgn",
        "condition": "noncarla_iperf_mild_awgn_vanilla",
        "awgn_noise_power_dB": "-10",
    },
    "tractor_clear": {
        "traffic": "tractor",
        "channel": "clear",
        "condition": "noncarla_tractor_clear_vanilla",
        "awgn_noise_power_dB": "",
    },
    "tractor_mild": {
        "traffic": "tractor",
        "channel": "mild_awgn",
        "condition": "noncarla_tractor_mild_awgn_vanilla",
        "awgn_noise_power_dB": "-10",
    },
}

DEFAULT_RUNS = "iperf_clear iperf_mild tractor_clear tractor_mild"


def q(series: pd.Series, p: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return float("nan")
    return float(vals.quantile(p))


def pct_true(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(100.0 * series.mean())


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def fmt(value: object) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return ""
    return f"{f:.3f}"


def parse_hms_to_seconds(value: str) -> float:
    value = str(value).strip()
    parts = value.split(":")
    if len(parts) != 3:
        return float("nan")
    hour, minute, second = parts
    return int(hour) * 3600.0 + int(minute) * 60.0 + float(second)


def tracer_time_seconds(series: pd.Series) -> pd.Series:
    return series.astype(str).map(parse_hms_to_seconds)


def run_group(base_batch: str, label: str) -> str:
    cfg = RUN_CONFIGS[label]
    return f"{cfg['condition']}_{base_batch}_{label}"


def cap_root(run_group_name: str) -> Path:
    return ROOT / "metrics_logs" / "noncarla_awgn" / run_group_name


def tt_root(run_group_name: str) -> Path:
    return ROOT / "metrics_logs" / "scenesense_ttracer" / run_group_name


def parse_interval(path: Path) -> Dict[str, float | str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    start_s = parse_hms_to_seconds(data.get("start_hms", ""))
    end_s = parse_hms_to_seconds(data.get("end_hms", ""))
    if math.isfinite(start_s) and math.isfinite(end_s) and end_s < start_s:
        end_s += 24 * 3600.0
    elapsed = float(data.get("elapsed_s", max(end_s - start_s, 0.0)))
    return {
        "traffic_start_sod": start_s,
        "traffic_end_sod": end_s,
        "traffic_elapsed_s": elapsed,
        "traffic_start_hms": data.get("start_hms", ""),
        "traffic_end_hms": data.get("end_hms", ""),
    }


def parse_iperf_json(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    try:
        data = json.loads(path.read_text(errors="replace"))
    except json.JSONDecodeError:
        return out
    end = data.get("end", {})
    sent = end.get("sum_sent") or {}
    recv = end.get("sum_received") or {}
    summary = end.get("sum") or {}
    if summary and summary.get("sender") is True and not sent:
        sent = summary
    elif summary and summary.get("sender") is False and not recv:
        recv = summary
    if sent:
        out["app_sent_mbps"] = float(sent.get("bits_per_second", float("nan"))) / 1e6
        out["app_sent_mbytes"] = float(sent.get("bytes", float("nan"))) / 1e6
        out["app_sent_seconds"] = float(sent.get("seconds", float("nan")))
        if "lost_percent" in sent:
            out["app_lost_percent"] = float(sent.get("lost_percent", float("nan")))
        if "jitter_ms" in sent:
            out["app_jitter_ms"] = float(sent.get("jitter_ms", float("nan")))
    if recv:
        recv_bytes = float(recv.get("bytes", float("nan")))
        recv_bps = float(recv.get("bits_per_second", float("nan")))
        if math.isfinite(recv_bytes) and recv_bytes > 0:
            out["app_recv_mbytes"] = recv_bytes / 1e6
        if math.isfinite(recv_bps) and recv_bps > 0:
            out["app_recv_mbps"] = recv_bps / 1e6
        out["app_lost_percent"] = float(recv.get("lost_percent", float("nan")))
        out["app_jitter_ms"] = float(recv.get("jitter_ms", float("nan")))
    if out.get("app_sent_mbytes", 0) and out.get("app_recv_mbytes", 0):
        out["app_byte_delivery_pct"] = 100.0 * out["app_recv_mbytes"] / out["app_sent_mbytes"]
    if "app_lost_percent" in out and math.isfinite(out["app_lost_percent"]):
        out["app_packet_delivery_pct"] = 100.0 - out["app_lost_percent"]
        if "app_recv_mbps" not in out and "app_sent_mbps" in out:
            out["app_recv_mbps"] = out["app_sent_mbps"] * (1.0 - out["app_lost_percent"] / 100.0)
        if "app_byte_delivery_pct" not in out:
            out["app_byte_delivery_pct"] = out["app_packet_delivery_pct"]
    return out


def parse_replay_sender(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    text = path.read_text(errors="replace")
    for key in [
        "udp_datagrams",
        "bytes",
        "duration_s",
        "mean_mbps",
        "sent_packets",
        "sent_bytes",
        "elapsed_s",
        "offered_mbps",
    ]:
        match = re.search(rf"{key}=([0-9.]+)", text)
        if not match:
            continue
        out[f"tractor_{key}"] = float(match.group(1))
    if "tractor_offered_mbps" in out:
        out["app_sent_mbps"] = out["tractor_offered_mbps"]
    if "tractor_sent_bytes" in out:
        out["app_sent_mbytes"] = out["tractor_sent_bytes"] / 1e6
    if "tractor_elapsed_s" in out:
        out["app_sent_seconds"] = out["tractor_elapsed_s"]
    return out


def parse_tcpdump_packets(path: Path) -> Dict[str, float]:
    out = {
        "sink_packets": float("nan"),
        "sink_udp_bytes": float("nan"),
        "sink_duration_s": float("nan"),
        "sink_mbps": float("nan"),
    }
    if not path.exists():
        return out
    times: List[float] = []
    bytes_total = 0
    packets = 0
    pattern = re.compile(r"^([0-9.]+).*UDP, length ([0-9]+)")
    with path.open(errors="replace") as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            times.append(float(match.group(1)))
            bytes_total += int(match.group(2))
            packets += 1
    if len(times) >= 2:
        duration = max(times) - min(times)
        mbps = bytes_total * 8.0 / duration / 1e6 if duration > 0 else float("nan")
    else:
        duration = float("nan")
        mbps = float("nan")
    out.update(
        {
            "sink_packets": float(packets),
            "sink_udp_bytes": float(bytes_total),
            "sink_duration_s": duration,
            "sink_mbps": mbps,
        }
    )
    return out


def add_active_grant_metrics(run_group_name: str, interval: Dict[str, float | str], row: Dict[str, object]) -> None:
    start_s = float(interval.get("traffic_start_sod", float("nan")))
    end_s = float(interval.get("traffic_end_sod", float("nan")))
    duration_s = float(interval.get("traffic_elapsed_s", float("nan")))
    if not (math.isfinite(start_s) and math.isfinite(end_s) and math.isfinite(duration_s) and duration_s > 0):
        return

    trace_base = tt_root(run_group_name) / "ue" / "csv"
    grants = safe_read_csv(trace_base / "NRUE_MAC_DCI_GRANT.csv")
    if grants.empty or not {"time", "direction", "tbs", "mcs", "rb_size"}.issubset(grants.columns):
        return

    grants = grants.copy()
    grants["_t"] = tracer_time_seconds(grants["time"])
    direction = pd.to_numeric(grants["direction"], errors="coerce")
    active = grants[(direction == 1) & (grants["_t"] >= start_s) & (grants["_t"] <= end_s)].copy()
    if active.empty:
        return

    tbs = pd.to_numeric(active["tbs"], errors="coerce").fillna(0.0)
    mcs = pd.to_numeric(active["mcs"], errors="coerce")
    rb = pd.to_numeric(active["rb_size"], errors="coerce")
    rv = pd.to_numeric(active.get("rv", 0), errors="coerce").fillna(0)
    harq_round = pd.to_numeric(active.get("round", 0), errors="coerce").fillna(0)
    retx_mask = (rv > 0) | (harq_round > 0)
    first_tbs = tbs.where(~retx_mask, 0.0)
    retx_tbs = tbs.where(retx_mask, 0.0)

    row.update(
        {
            "active_ul_grants": float(len(active)),
            "active_ul_grant_hz": float(len(active) / duration_s),
            "active_ul_scheduled_mbps": float(tbs.sum() * 8.0 / duration_s / 1e6),
            "active_ul_first_tx_mbps": float(first_tbs.sum() * 8.0 / duration_s / 1e6),
            "active_ul_retx_mbps": float(retx_tbs.sum() * 8.0 / duration_s / 1e6),
            "active_ul_retx_grant_pct": pct_true(retx_mask),
            "active_ul_avg_mcs": float(mcs.mean()),
            "active_ul_p50_mcs": q(mcs, 0.50),
            "active_ul_p95_mcs": q(mcs, 0.95),
            "active_ul_avg_tbs_bytes": float(tbs.mean()),
            "active_ul_p50_tbs_bytes": q(tbs, 0.50),
            "active_ul_p95_tbs_bytes": q(tbs, 0.95),
            "active_ul_avg_rb_size": float(rb.mean()),
            "active_ul_p50_rb_size": q(rb, 0.50),
            "active_ul_p95_rb_size": q(rb, 0.95),
            "active_ul_full_prb_grant_pct": pct_true(rb == 106),
        }
    )

    # 100 ms bins: compare scheduler service while backlog exists.
    bin_s = 0.1
    num_bins = max(1, int(math.ceil(duration_s / bin_s)))
    active["bin"] = ((active["_t"] - start_s) // bin_s).astype(int)
    active["first_tbs"] = first_tbs
    active["retx_tbs"] = retx_tbs
    grant_bins = active.groupby("bin").agg(
        tbs=("tbs", "sum"),
        first_tbs=("first_tbs", "sum"),
        retx_tbs=("retx_tbs", "sum"),
        grants=("tbs", "size"),
        mcs=("mcs", "mean"),
        tbs_avg=("tbs", "mean"),
        rb=("rb_size", "mean"),
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
            all_bins = pd.DataFrame(index=range(num_bins))
            all_bins["tbs"] = grant_bins["tbs"]
            all_bins["first_tbs"] = grant_bins["first_tbs"]
            all_bins["retx_tbs"] = grant_bins["retx_tbs"]
            all_bins["grants"] = grant_bins["grants"]
            all_bins["mcs"] = grant_bins["mcs"]
            all_bins["tbs_avg"] = grant_bins["tbs_avg"]
            all_bins["rb"] = grant_bins["rb"]
            all_bins["rlc"] = rlc_bins
            all_bins = all_bins.fillna({"tbs": 0.0, "first_tbs": 0.0, "retx_tbs": 0.0, "grants": 0.0, "rlc": 0.0})
            backlog_bins = all_bins[all_bins["rlc"] > 0]
            row.update(
                {
                    "active_rlc_lcid4_mean_kib": float(rlc4["bytes_in_buffer"].mean() / 1024.0),
                    "active_rlc_lcid4_p50_kib": q(rlc4["bytes_in_buffer"], 0.50) / 1024.0,
                    "active_rlc_lcid4_p95_kib": q(rlc4["bytes_in_buffer"], 0.95) / 1024.0,
                    "active_rlc_lcid4_max_kib": float(pd.to_numeric(rlc4["bytes_in_buffer"], errors="coerce").max() / 1024.0),
                    "active_rlc_nonzero_pct_100ms": float(100.0 * len(backlog_bins) / len(all_bins)),
                }
            )
            if not backlog_bins.empty:
                row.update(
                    {
                        "sched_mbps_when_rlc_nonzero": float(backlog_bins["tbs"].sum() * 8.0 / (len(backlog_bins) * bin_s) / 1e6),
                        "first_tx_mbps_when_rlc_nonzero": float(backlog_bins["first_tbs"].sum() * 8.0 / (len(backlog_bins) * bin_s) / 1e6),
                        "retx_mbps_when_rlc_nonzero": float(backlog_bins["retx_tbs"].sum() * 8.0 / (len(backlog_bins) * bin_s) / 1e6),
                        "grant_hz_when_rlc_nonzero": float(backlog_bins["grants"].mean() / bin_s),
                        "mcs_when_rlc_nonzero": float(backlog_bins["mcs"].mean()),
                        "tbs_when_rlc_nonzero": float(backlog_bins["tbs_avg"].mean()),
                        "rb_when_rlc_nonzero": float(backlog_bins["rb"].mean()),
                    }
                )

    bsr = safe_read_csv(trace_base / "NRUE_MAC_BSR_STATUS.csv")
    if not bsr.empty and {"time", "lcg1_bytes", "sdu_bytes"}.issubset(bsr.columns):
        bsr = bsr.copy()
        bsr["_t"] = tracer_time_seconds(bsr["time"])
        active_bsr = bsr[(bsr["_t"] >= start_s) & (bsr["_t"] <= end_s)].copy()
        if not active_bsr.empty:
            row.update(
                {
                    "active_bsr_lcg1_mean_kib": float(pd.to_numeric(active_bsr["lcg1_bytes"], errors="coerce").mean() / 1024.0),
                    "active_bsr_lcg1_p50_kib": q(active_bsr["lcg1_bytes"], 0.50) / 1024.0,
                    "active_bsr_lcg1_p95_kib": q(active_bsr["lcg1_bytes"], 0.95) / 1024.0,
                    "active_bsr_lcg1_max_kib": float(pd.to_numeric(active_bsr["lcg1_bytes"], errors="coerce").max() / 1024.0),
                    "active_bsr_sdu_bytes_mean": float(pd.to_numeric(active_bsr["sdu_bytes"], errors="coerce").mean()),
                }
            )


def add_gnb_metrics(run_group_name: str, interval: Dict[str, float | str], row: Dict[str, object]) -> None:
    start_s = float(interval.get("traffic_start_sod", float("nan")))
    end_s = float(interval.get("traffic_end_sod", float("nan")))
    trace_base = tt_root(run_group_name) / "gnb" / "csv"

    pwr = safe_read_csv(trace_base / "GNB_MAC_PUSCH_POWER_CONTROL.csv")
    if not pwr.empty and {"time", "snrx10", "mcs"}.issubset(pwr.columns):
        pwr = pwr.copy()
        pwr["_t"] = tracer_time_seconds(pwr["time"])
        if math.isfinite(start_s) and math.isfinite(end_s):
            pwr = pwr[(pwr["_t"] >= start_s) & (pwr["_t"] <= end_s)]
        if not pwr.empty:
            snr = pd.to_numeric(pwr["snrx10"], errors="coerce") / 10.0
            row.update(
                {
                    "gnb_snr_p05_db": q(snr, 0.05),
                    "gnb_snr_p50_db": q(snr, 0.50),
                    "gnb_snr_p95_db": q(snr, 0.95),
                    "gnb_pusch_mcs_p50": q(pwr["mcs"], 0.50),
                }
            )

    bler = safe_read_csv(trace_base / "GNB_MAC_BLER_MCS_DECISION.csv")
    if not bler.empty and {"time", "direction", "updated", "branch"}.issubset(bler.columns):
        bler = bler.copy()
        bler["_t"] = tracer_time_seconds(bler["time"])
        direction = pd.to_numeric(bler["direction"], errors="coerce")
        updated = pd.to_numeric(bler["updated"], errors="coerce")
        active = bler[(direction == 1) & (updated == 1)]
        if math.isfinite(start_s) and math.isfinite(end_s):
            active = active[(active["_t"] >= start_s) & (active["_t"] <= end_s)]
        if not active.empty:
            branch = pd.to_numeric(active["branch"], errors="coerce")
            row.update(
                {
                    "bler_updates": float(len(active)),
                    "branch_increase_low_bler_pct": pct_true(branch == 1),
                    "branch_decrease_high_bler_pct": pct_true(branch == 2),
                    "branch_decrease_few_samples_pct": pct_true(branch == 3),
                    "branch_hold_target_pct": pct_true(branch == 4),
                }
            )
            for col in ["num_sched", "num_retx", "bler_window_ppm", "bler_before_ppm", "bler_after_ppm"]:
                if col in active.columns:
                    vals = pd.to_numeric(active[col], errors="coerce")
                    if col.startswith("bler"):
                        row[f"{col}_p50_pct"] = q(vals, 0.50) / 10_000.0
                        row[f"{col}_p95_pct"] = q(vals, 0.95) / 10_000.0
                    else:
                        row[f"{col}_p50"] = q(vals, 0.50)
                        row[f"{col}_p95"] = q(vals, 0.95)


def summarize_one(base_batch: str, label: str) -> Dict[str, object]:
    cfg = RUN_CONFIGS[label]
    rg = run_group(base_batch, label)
    cap = cap_root(rg)
    row: Dict[str, object] = {
        "label": label,
        "traffic": cfg["traffic"],
        "channel": cfg["channel"],
        "awgn_noise_power_dB": cfg["awgn_noise_power_dB"],
        "condition": cfg["condition"],
        "run_group": rg,
        "cap_root": str(cap),
    }
    interval = parse_interval(cap / "traffic_interval.json")
    row.update(interval)

    if cfg["traffic"] == "iperf":
        row.update(parse_iperf_json(cap / "iperf3_client.json"))
        row.update({f"server_{k}": v for k, v in parse_iperf_json(cap / "iperf3_server.json").items()})
    else:
        row.update(parse_replay_sender(cap / "replay_sender.log"))
        row.update(parse_tcpdump_packets(cap / "udp_sink_packets.txt"))
        sent_packets = float(row.get("tractor_sent_packets", float("nan")))
        sent_bytes = float(row.get("tractor_sent_bytes", float("nan")))
        sink_packets = float(row.get("sink_packets", float("nan")))
        sink_bytes = float(row.get("sink_udp_bytes", float("nan")))
        if sent_packets and math.isfinite(sent_packets):
            row["app_packet_delivery_pct"] = 100.0 * sink_packets / sent_packets
        if sent_bytes and math.isfinite(sent_bytes):
            row["app_byte_delivery_pct"] = 100.0 * sink_bytes / sent_bytes
        if math.isfinite(float(row.get("sink_mbps", float("nan")))):
            row["app_recv_mbps"] = row["sink_mbps"]

    add_active_grant_metrics(rg, interval, row)
    add_gnb_metrics(rg, interval, row)
    return row


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    columns = [
        "label",
        "traffic",
        "channel",
        "app_sent_mbps",
        "app_recv_mbps",
        "app_byte_delivery_pct",
        "active_ul_first_tx_mbps",
        "active_ul_scheduled_mbps",
        "active_ul_retx_mbps",
        "active_ul_retx_grant_pct",
        "active_ul_grant_hz",
        "active_ul_avg_mcs",
        "active_ul_p50_mcs",
        "active_ul_p95_mcs",
        "active_ul_avg_tbs_bytes",
        "active_ul_p50_rb_size",
        "active_ul_full_prb_grant_pct",
        "active_rlc_nonzero_pct_100ms",
        "active_rlc_lcid4_p95_kib",
        "active_bsr_lcg1_p95_kib",
        "first_tx_mbps_when_rlc_nonzero",
        "grant_hz_when_rlc_nonzero",
        "mcs_when_rlc_nonzero",
        "tbs_when_rlc_nonzero",
        "gnb_snr_p50_db",
        "branch_decrease_high_bler_pct",
        "branch_decrease_few_samples_pct",
        "bler_window_ppm_p50_pct",
        "num_sched_p50",
        "num_retx_p50",
    ]
    existing = [c for c in columns if c in df.columns]
    out = df[existing].copy()
    for c in out.columns:
        if c not in {"label", "traffic", "channel"}:
            out[c] = out[c].map(fmt)
    lines = [
        "# Non-CARLA vanilla OAI 106PRB clear vs mild-AWGN summary",
        "",
        "This table isolates traffic/scheduler behavior without CARLA model compute or closed-loop result waiting.",
        "",
        "Note: scheduled/first-TX TBS is radio grant capacity. For small-packet traffic it can exceed app payload rate, so use it with RLC/BSR backlog and app delivery rather than treating it as received payload.",
        "",
        out.to_markdown(index=False),
        "",
        "Interpretation guide:",
        "",
        "- If mild-AWGN iperf keeps high MCS and high first-transmission drain, the CARLA mild-AWGN behavior is not a generic AWGN failure.",
        "- If tractor mild-AWGN behaves like CARLA, bursty replay traffic is enough to reproduce the issue.",
        "- If tractor behaves like iperf while CARLA remains slow, the distinctive part is the split-inference whole-frame burst/backpressure pattern.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-batch", required=True)
    parser.add_argument("--runs", default=DEFAULT_RUNS)
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    labels = args.runs.split()
    rows = [summarize_one(args.base_batch, label) for label in labels]
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = args.out_prefix or f"noncarla_vanilla_awgn106_{args.base_batch}"
    csv_path = OUT_DIR / f"{suffix}.csv"
    md_path = OUT_DIR / f"{suffix}.md"
    df.to_csv(csv_path, index=False)
    write_markdown(df, md_path)
    print(csv_path)
    print(md_path)
    display_cols = [
        c
        for c in [
            "label",
            "traffic",
            "channel",
            "app_sent_mbps",
            "app_recv_mbps",
            "active_ul_first_tx_mbps",
            "active_ul_retx_mbps",
            "active_ul_grant_hz",
            "active_ul_avg_mcs",
            "active_ul_avg_tbs_bytes",
            "active_rlc_lcid4_p95_kib",
            "gnb_snr_p50_db",
            "branch_decrease_high_bler_pct",
            "branch_decrease_few_samples_pct",
        ]
        if c in df.columns
    ]
    print(df[display_cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
