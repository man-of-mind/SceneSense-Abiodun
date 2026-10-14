"""Readers for the M3.5 v15 (= v11) calibrated radio telemetry (offline; the CSVs are raw T-tracer rows with ingest clocks).

Each data line is ``<host wall_ns>,<CLOCK_MONOTONIC_RAW ns>,<T time-of-day>,<event fields...>``.  The two leading
clocks are stamped by the runner-parent drain thread when the line arrives, so every row carries both the clock the
decision timeline uses (host wall) and the clock the GT worker traces use (RAW).  The T ``time`` field itself is a
local time-of-day in microseconds with no date and is kept verbatim, never used for joining.

Quantities (``experiments/splitfusion_m3_good_all_snr_qual_v1/20261008_run01``):

* ``GNB_MAC_PUSCH_POWER_CONTROL.snrx10 / 10``  = per-transmission PUSCH SNR, the Phase-14A calibrated channel SNR;
* ``GNB_MAC_UL_MCS_DECISION.avg_snr_x10 / 10`` = the scheduler EMA the SINR MCS lookup reads;
* ``selected_mcs`` / ``final_mcs`` (and the PUSCH event's own ``mcs``) = the actual MCS.

The gNB MAC-log value (``m3_radio/gnb_stats_observed.jsonl``) is a different statistic -- the power-control view
``avg_snr + tpc_in_flight``, TPC-pinned near 17 dB at any channel SNR -- and is never mixed into these series.
"""
from pathlib import Path
import statistics

from rl_agent.splitfusion_m3_route_b_v15.radio_telemetry import MCS_FIELDS, PUSCH_FIELDS

STREAMS = {"pusch": ("live_pusch_with_ingest.csv", PUSCH_FIELDS),
           "mcs": ("live_mcs_with_ingest.csv", MCS_FIELDS)}
LABELS = dict(
    pusch_snr_db="PUSCH SNR (per-transmission snrx10, Phase-14A calibrated channel SNR)",
    scheduler_avg_snr_db="scheduler avg_snr (EMA, drives the SINR MCS lookup)",
    mcs="UL MCS (scheduler selected/final)",
    commanded_target_db="commanded target (OPERATING_12DB constant, actor-visible proxy)",
    mac_log_snr_db="power-control SNR (avg_snr + tpc_in_flight) - NOT channel SNR")


def read(path, fields):
    """Parse one ingest CSV; returns rows as dicts with ``wall_ns``/``raw_ns`` plus the integer event fields."""
    rows, skipped = [], 0
    for line in Path(path).read_text(errors="replace").splitlines():
        parts = line.split(",")
        if len(parts) != len(fields) + 2 or ":" not in parts[2] or not parts[0].isdigit():
            skipped += 1
            continue
        row = dict(wall_ns=int(parts[0]), raw_ns=int(parts[1]), time=parts[2])
        try:
            for name, value in zip(fields[1:], parts[3:]):
                row[name] = int(value)
        except ValueError:
            skipped += 1
            continue
        rows.append(row)
    return rows, skipped


def load(ttracer_dir):
    """Both streams of one capture, with the derived calibrated quantities attached."""
    directory = Path(ttracer_dir)
    out = {}
    for name, (filename, fields) in STREAMS.items():
        rows, skipped = read(directory / filename, fields)
        for row in rows:
            if name == "pusch":
                row["pusch_snr_db"] = row["snrx10"] / 10.0
            else:
                row["scheduler_avg_snr_db"] = row["avg_snr_x10"] / 10.0
        out[name] = dict(rows=rows, skipped_lines=skipped, file=str(directory / filename))
    return out


def summarise(values):
    values = sorted(float(v) for v in values)
    if not values:
        return dict(n=0)
    def q(p):
        return values[min(len(values) - 1, max(0, int(round(p * (len(values) - 1)))))]
    return dict(n=len(values), min=values[0], p05=q(0.05), p50=statistics.median(values), p95=q(0.95),
                max=values[-1], mean=statistics.fmean(values))
