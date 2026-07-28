#!/usr/bin/env python3
"""Summarize one TRACTOR-over-OAI replay run."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


AB = Path(__file__).resolve().parents[1]


def pct(series: pd.Series, q: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.quantile(q)) if len(s) else float("nan")


def parse_sender(path: Path) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {}
    if not path.exists():
        return out
    text = path.read_text(errors="replace")
    for key in ["udp_datagrams", "bytes", "duration_s", "mean_mbps", "sent_packets", "sent_bytes", "elapsed_s", "offered_mbps"]:
        m = re.search(rf"{key}=([0-9.]+)", text)
        if m:
            val = float(m.group(1))
            out[key] = int(val) if key in {"udp_datagrams", "bytes", "sent_packets", "sent_bytes"} else val
    return out


def parse_tcpdump_packets(path: Path) -> dict[str, float | int]:
    if not path.exists():
        return {"rx_packets": 0, "rx_ip_bytes": 0, "rx_duration_s": float("nan"), "rx_mbps": float("nan")}
    times = []
    bytes_total = 0
    packets = 0
    # Example:
    # 1785200787.123456 IP 10.0.0.2.54321 > 192.168.70.135.55000: UDP, length 1400
    pat = re.compile(r"^([0-9.]+).*UDP, length ([0-9]+)")
    with path.open(errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            times.append(float(m.group(1)))
            bytes_total += int(m.group(2))
            packets += 1
    if len(times) >= 2:
        dur = max(times) - min(times)
    else:
        dur = float("nan")
    mbps = bytes_total * 8 / dur / 1e6 if dur and dur == dur and dur > 0 else float("nan")
    return {"rx_packets": packets, "rx_ip_bytes": bytes_total, "rx_duration_s": dur, "rx_mbps": mbps}


def parse_layer(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    text = path.read_text(errors="replace")
    m = re.search(r"RLC mean queueing delay .*:\*\* ([0-9.]+) ms", text)
    if m:
        out["rlc_queue_wait_mean_ms"] = float(m.group(1))
    m = re.search(r"UE PDCP-ingress -> gNB PDCP-deliver .* mean=([0-9.]+) ms\s+p50=([0-9.]+)\s+p95=([0-9.]+)", text)
    if m:
        out["ran_ul_mean_ms"] = float(m.group(1))
        out["ran_ul_p50_ms"] = float(m.group(2))
        out["ran_ul_p95_ms"] = float(m.group(3))
    return out


def summarize(run_group: str) -> dict[str, object]:
    cap = AB / "metrics_logs" / "tractor_replay" / run_group
    tt = AB / "metrics_logs" / "scenesense_ttracer" / run_group
    row: dict[str, object] = {"run_group": run_group}
    row.update(parse_sender(cap / "replay_sender.log"))
    row.update(parse_tcpdump_packets(cap / "udp_sink_packets.txt"))
    if row.get("sent_packets"):
        row["packet_delivery"] = float(row.get("rx_packets", 0)) / float(row["sent_packets"])
    if row.get("sent_bytes"):
        row["byte_delivery"] = float(row.get("rx_ip_bytes", 0)) / float(row["sent_bytes"])

    grant_path = tt / "ue/analysis/nrue_grant_windows.csv"
    if grant_path.exists():
        w = pd.read_csv(grant_path)
        if "direction" in w.columns:
            d = w["direction"].astype(str).str.lower()
            w = w[d.eq("ul") | d.eq("1")]
        active = w[pd.to_numeric(w.get("scheduled_mbps", 0), errors="coerce") > 0.1].copy()
        if len(active):
            row["active_windows"] = len(active)
            row["ul_sched_mbps_p50"] = pct(active["scheduled_mbps"], 0.50)
            row["ul_sched_mbps_p95"] = pct(active["scheduled_mbps"], 0.95)
            row["ul_avg_mcs_p50"] = pct(active["avg_mcs"], 0.50)
            row["ul_avg_mcs_p95"] = pct(active["avg_mcs"], 0.95)
            row["ul_avg_rbs_p50"] = pct(active["avg_rb_size"], 0.50)
            row["ul_retx_rate_mean"] = float(pd.to_numeric(active["retx_rate"], errors="coerce").mean())
            row["ul_retx_rate_p95"] = pct(active["retx_rate"], 0.95)

    bler_path = tt / "gnb/csv/GNB_MAC_BLER_MCS_DECISION.csv"
    if bler_path.exists():
        b = pd.read_csv(bler_path)
        upd = b[(b["direction"] == 1) & (b["updated"] == 1)].copy()
        row["bler_updates"] = len(upd)
        if len(upd):
            row["branch_increase_low_bler_pct"] = 100 * (upd["branch"] == 1).mean()
            row["branch_decrease_high_bler_pct"] = 100 * (upd["branch"] == 2).mean()
            row["branch_decrease_few_samples_pct"] = 100 * (upd["branch"] == 3).mean()
            row["branch_hold_target_pct"] = 100 * (upd["branch"] == 4).mean()
            row["num_sched_p50"] = pct(upd["num_sched"], 0.50)
            row["num_sched_p95"] = pct(upd["num_sched"], 0.95)
            row["num_retx_p95"] = pct(upd["num_retx"], 0.95)

    rlc_path = tt / "ue/csv/NRUE_MAC_RLC_BUFFER_STATUS.csv"
    if rlc_path.exists():
        r = pd.read_csv(rlc_path)
        r4 = r[r["lcid"] == 4]["bytes_in_buffer"]
        row["rlc_lcid4_kb_p50"] = pct(r4, 0.50) / 1024
        row["rlc_lcid4_kb_p95"] = pct(r4, 0.95) / 1024
        row["rlc_lcid4_kb_mean"] = float(pd.to_numeric(r4, errors="coerce").mean()) / 1024
        row["rlc_lcid4_kb_max"] = float(pd.to_numeric(r4, errors="coerce").max()) / 1024

    bsr_path = tt / "ue/csv/NRUE_MAC_BSR_STATUS.csv"
    if bsr_path.exists():
        bsr = pd.read_csv(bsr_path)
        row["bsr_lcg1_kb_p50"] = pct(bsr["lcg1_bytes"], 0.50) / 1024
        row["bsr_lcg1_kb_p95"] = pct(bsr["lcg1_bytes"], 0.95) / 1024
        row["bsr_lcg1_kb_max"] = float(pd.to_numeric(bsr["lcg1_bytes"], errors="coerce").max()) / 1024

    pwr_path = tt / "gnb/csv/GNB_MAC_PUSCH_POWER_CONTROL.csv"
    if pwr_path.exists():
        p = pd.read_csv(pwr_path)
        row["gnb_snr_db_p50"] = pct(p["snrx10"], 0.50) / 10
        row["gnb_snr_db_min"] = float(pd.to_numeric(p["snrx10"], errors="coerce").min()) / 10
        row["gnb_snr_db_max"] = float(pd.to_numeric(p["snrx10"], errors="coerce").max()) / 10
        row["gnb_pusch_mcs_p50"] = pct(p["mcs"], 0.50)

    row.update(parse_layer(tt / "layer_latency/uplink_layer_latency.md"))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-group", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    row = summarize(args.run_group)
    out = Path(args.out) if args.out else AB / "metrics_logs/tractor_replay" / args.run_group / "tractor_oai_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, index=False)
    print(out)
    print(pd.DataFrame([row]).T.to_string(header=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
