#!/usr/bin/env python3
"""Summarize a measure.py capture: command -> per-transmission PUSCH SNR -> scheduler EMA -> MAC-log SNR -> MCS."""
from __future__ import annotations

import collections
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

PUSCH = ("time", "rnti", "frame", "slot", "snrx10", "phr", "tpc", "tb_size", "txpower_calc", "rbSize", "mcs", "rssi")
MCS = ("time", "rnti", "frame", "slot", "sched_frame", "sched_slot", "avg_snr_x10", "mcs_table", "ul_bler_mcs_before",
       "selected_mcs", "pre_phr_mcs", "post_phr_mcs", "final_mcs", "estimated_ul_buffer", "sched_ul_bytes", "B",
       "min_rb", "available_rb_before", "available_rb_after", "ph", "pcmax", "rb_size_final", "tbs_final",
       "force_ul_mcs")
UL = re.compile(r"ulsch_rounds (\d+)/(\d+)/(\d+)/(\d+), ulsch_errors (\d+), ulsch_DTX (\d+), BLER ([\d.]+) "
                r"MCS \((\d+)\) (\d+) \(Qm \d+ deltaMCS (-?\d+) dB\) NPRB (\d+) SNR ([-\d.]+) \(([-+\d.]+)\) dB")
RI = re.compile(r"UL-RI (\d+)")


def trace(path, names):
    rows = []
    for line in open(path):
        wall, mono, rest = line.rstrip("\n").split(",", 2)
        if not rest[:1].isdigit():
            continue
        values = rest.split(",")
        if len(values) == len(names):
            rows.append(dict(mono_ns=int(mono), **{k: (v if k == "time" else int(v)) for k, v in zip(names, values)}))
    return rows


def stats(values, low=21.0):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return dict(n=0)
    return dict(n=int(a.size), median=float(np.median(a)), p05=float(np.percentile(a, 5)),
                p95=float(np.percentile(a, 95)), min=float(a.min()), max=float(a.max()),
                frac_below_21=float(np.mean(a < low)))


def main(root: Path) -> dict:
    meta = json.loads((root / "measurement.json").read_text())
    pusch = trace(root / "ttracer/live_pusch_with_ingest.csv", PUSCH)
    mcs = trace(root / "ttracer/live_mcs_with_ingest.csv", MCS)
    mac = [json.loads(l) for l in open(root / "gnb_stats_observed.jsonl")]
    sender = list(csv.DictReader(open(root / "traffic/sender.csv")))
    sink = list(csv.DictReader(open(root / "traffic/sink_packets.csv")))
    # Sink wall_s is CLOCK_MONOTONIC relative to its own start; anchor its first packet on the sender's first send
    # (sender wall -> monotonic via the recorded spawn pair). Error = one-way uplink delay of frame 0 (tens of ms).
    w2m = meta["sender_spawn_monotonic_ns"] - meta["sender_spawn_wall_ns"]
    sink_zero = int(float(sender[0]["wall_time_s"]) * 1e9) + w2m - int(float(sink[0]["wall_s"]) * 1e9)
    rntis = sorted({r["rnti"] for r in pusch + mcs})
    out = dict(source=str(root), rntis=rntis, offered=meta["traffic"], windows=[], commands=meta["commands"])
    for w in meta["windows"]:
        s, e = w["start_monotonic_ns"], w["end_monotonic_ns"]
        dur = (e - s) / 1e9
        p = [r for r in pusch if s <= r["mono_ns"] < e]
        m = [r for r in mcs if s <= r["mono_ns"] < e]
        lines = [r for r in mac if s <= r["mono_ns"] < e]
        ul = [UL.search(r["line"]) for r in lines]
        ul = [x for x in ul if x]
        ri = collections.Counter(int(x.group(1)) for x in (RI.search(r["line"]) for r in lines) if x)
        first, last = ul[0], ul[-1]
        rounds = [int(last.group(i)) - int(first.group(i)) for i in range(1, 5)]
        sent = [r for r in sender if s <= int(float(r["wall_time_s"]) * 1e9) + w2m < e]
        got = [r for r in sink if s <= sink_zero + int(float(r["wall_s"]) * 1e9) < e]
        out["windows"].append(dict(
            nominal_target_db=w["nominal_target_db"], noise_power_db=w["noise_power_db"], duration_s=dur,
            pusch_snr_db=stats([r["snrx10"] / 10 for r in p]),
            scheduler_avg_snr_db=stats([r["avg_snr_x10"] / 10 for r in m]),
            mac_log_snr_db=stats([float(x.group(12)) for x in ul]),
            mac_log_diff_to_target_db=stats([float(x.group(13)) for x in ul]),
            selected_mcs=dict(sorted(collections.Counter(r["selected_mcs"] for r in m).items())),
            final_mcs=dict(sorted(collections.Counter(r["final_mcs"] for r in m).items())),
            pusch_mcs=dict(sorted(collections.Counter(r["mcs"] for r in p).items())),
            mcs_table=sorted({r["mcs_table"] for r in m}), mac_log_mcs_table=sorted({int(x.group(8)) for x in ul}),
            force_ul_mcs=sorted({r["force_ul_mcs"] for r in m}),
            ul_rank_indicator=dict(ri), txpower_calc=dict(collections.Counter(r["txpower_calc"] for r in p)),
            tpc_field=dict(sorted(collections.Counter(r["tpc"] for r in p).items())),
            rb_size_median=float(np.median([r["rbSize"] for r in p])),
            mac_log_rounds_delta=rounds, mac_log_errors_delta=int(last.group(5)) - int(first.group(5)),
            mac_log_dtx_delta=int(last.group(6)) - int(first.group(6)),
            retransmission_fraction=(sum(rounds[1:]) / max(1, sum(rounds))),
            pusch_tb_bytes_per_s=sum(r["tb_size"] for r in p) / dur,
            offered_mbps=sum(int(r["chunk_bytes"]) for r in sent) * 8 / dur / 1e6,
            delivered_mbps=sum(int(r["size"]) for r in got) * 8 / dur / 1e6,
            sender_frames=len({r["frame_index"] for r in sent}),
        ))
    totals = dict(sent_bytes=sum(int(r["chunk_bytes"]) for r in sender), received_bytes=sum(int(r["size"]) for r in sink),
                  sent_chunks=len(sender), received_chunks=len(sink))
    out["traffic_totals"] = totals
    (root / "SUMMARY.json").write_text(json.dumps(out, indent=1))
    plot(root, meta, pusch, mcs, mac)
    return out


def plot(root, meta, pusch, mcs, mac):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = meta["sender_spawn_monotonic_ns"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4]))
    a.scatter([(r["mono_ns"] - t0) / 1e9 for r in pusch], [r["snrx10"] / 10 for r in pusch], s=2, alpha=0.25,
              color="#1f77b4", label="PUSCH SNR per transmission (snrx10/10; calibrated metric)")
    a.plot([(r["mono_ns"] - t0) / 1e9 for r in mcs], [r["avg_snr_x10"] / 10 for r in mcs], color="#ff7f0e", lw=1.2,
           label="scheduler avg_snr (EMA 0.975; drives MCS)")
    pts = [(r["mono_ns"], UL.search(r["line"])) for r in mac]
    pts = [(t, float(x.group(12))) for t, x in pts if x]
    a.plot([(t - t0) / 1e9 for t, _ in pts], [v for _, v in pts], "s-", ms=3, color="#d62728", lw=1,
           label="MAC-log SNR = avg_snr + tpc_in_flight (power-control view)")
    a.axhline(15.0, color="grey", ls=":", lw=1)
    a.text(1, 15.2, "pusch_TargetSNR 15.0 dB", color="grey", fontsize=8)
    a.axhline(17.0, color="grey", ls="--", lw=0.8)
    a.text(1, 17.2, "target + 2 dB (TPC 'decrease' edge)", color="grey", fontsize=8)
    for w in meta["windows"]:
        x0, x1 = (w["start_monotonic_ns"] - t0) / 1e9, (w["end_monotonic_ns"] - t0) / 1e9
        for ax in (a, b):
            ax.axvspan(x0, x1, color="#2ca02c", alpha=0.07)
        a.hlines(w["nominal_target_db"], x0, x1, color="black", lw=2)
        a.text(x0 + 1, w["nominal_target_db"] + 0.6, f"commanded {w['nominal_target_db']} dB "
               f"(noise {w['noise_power_db']} dB)", fontsize=8)
    for c in meta["commands"]:
        a.axvline((c["ack_monotonic_ns"] - t0) / 1e9, color="black", lw=0.6, ls="-.")
    a.set_ylim(12, 30)
    a.set_ylabel("dB")
    a.legend(loc="upper right", fontsize=7.5)
    a.set_title("GOOD_ALL radio-only qualification, 100 MHz/273 PRB/4D5U, 8 fps x 112.8 KB (shaded = measured)")
    b.step([(r["mono_ns"] - t0) / 1e9 for r in mcs], [r["selected_mcs"] for r in mcs], where="post", lw=1,
           label="selected MCS (SINR lookup)")
    b.step([(r["mono_ns"] - t0) / 1e9 for r in mcs], [r["final_mcs"] for r in mcs], where="post", lw=1, ls="--",
           label="final MCS")
    b.set_ylabel("MCS (table 0)")
    b.set_xlabel("seconds since traffic start")
    b.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(root / "snr_quantities_timeline.png", dpi=130)


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=1))
