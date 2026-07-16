#!/usr/bin/env python3
"""Compute OAI transport metrics for a run-group and append one row to a results TSV.
Reads the front's stream metrics CSV (transport_round_trip_ms_estimate, feature_payload_bytes,
feature_payload_chunks, result_received) + optionally the network sampler summary (ping RTT).
Usage: oai_extract_metrics.py <config_label> <run_group> <results_tsv>
Finds the newest metrics_logs/scenesense_runs/*<run_group>* run dir automatically.
"""
import csv, sys, glob, os, statistics

AB = "/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
label, run_group, tsv = sys.argv[1], sys.argv[2], sys.argv[3]

def truthy(v): return str(v).strip().lower() in ("true", "1", "yes")

# find newest run dir whose metrics CSV has this run_group
cands = []
for d in glob.glob(f"{AB}/metrics_logs/scenesense_runs/*"):
    mcs = glob.glob(d + "/streams/*metrics.csv")
    if not mcs: continue
    try:
        rows = list(csv.DictReader(open(mcs[0])))
        if rows and str(rows[0].get("run_group", "")) == run_group:
            cands.append((os.path.getmtime(d), d, mcs[0]))
    except Exception:
        pass
if not cands:
    print(f"[extract] NO run found for run_group={run_group}")
    row = dict(config=label, run_group=run_group, frames=0, rtt_mean_ms="NA", rtt_p95_ms="NA",
               payload_kb="NA", frag_per_frame="NA", delivery_pct="NA", ping_rtt_ms="NA", status="NO_RUN")
else:
    cands.sort(); _, rundir, csvp = cands[-1]
    rows = list(csv.DictReader(open(csvp)))
    rtt = [float(r["transport_round_trip_ms_estimate"]) for r in rows
           if r.get("transport_round_trip_ms_estimate", "") not in ("", "nan")]
    pay = [float(r["feature_payload_bytes"]) for r in rows if r.get("feature_payload_bytes", "") not in ("",)]
    chunks = [float(r["feature_payload_chunks"]) for r in rows if r.get("feature_payload_chunks", "") not in ("",)]
    deliv = [truthy(r.get("result_received", "")) for r in rows if "result_received" in r]
    def p95(x): return sorted(x)[max(0, int(0.95 * len(x)) - 1)] if x else float("nan")
    # network sampler ping RTT (idle-ish) if present
    ping = "NA"
    for ns in glob.glob(f"{AB}/metrics_logs/scenesense_network/{run_group}/network_summary.csv"):
        try:
            nr = list(csv.DictReader(open(ns)))
            vals = [float(r["avg_ping_rtt_ms"]) for r in nr if r.get("avg_ping_rtt_ms", "") not in ("", "nan")]
            if vals: ping = f"{statistics.mean(vals):.2f}"
        except Exception: pass
    row = dict(config=label, run_group=run_group, frames=len(rows),
               rtt_mean_ms=f"{statistics.mean(rtt):.1f}" if rtt else "NA",
               rtt_p95_ms=f"{p95(rtt):.1f}" if rtt else "NA",
               payload_kb=f"{statistics.mean(pay)/1024:.1f}" if pay else "NA",
               frag_per_frame=f"{statistics.mean(chunks):.1f}" if chunks else "NA",
               delivery_pct=f"{100*sum(deliv)/len(deliv):.1f}" if deliv else "NA",
               ping_rtt_ms=ping, status="OK")
    print(f"[extract] {label}: frames={row['frames']} RTT={row['rtt_mean_ms']}ms "
          f"payload={row['payload_kb']}KB frag={row['frag_per_frame']} delivery={row['delivery_pct']}%")

fields = ["config", "run_group", "frames", "rtt_mean_ms", "rtt_p95_ms", "payload_kb",
          "frag_per_frame", "delivery_pct", "ping_rtt_ms", "status"]
newfile = not os.path.exists(tsv)
with open(tsv, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    if newfile: w.writeheader()
    w.writerow(row)
