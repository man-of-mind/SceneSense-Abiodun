#!/usr/bin/env python3
"""Combine UE-N3E campaign directories into one CSV and one short report."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent.ue_n3e_fallback_snr_floor_v1 import RUN_FIELDS  # noqa: E402

CRITERIA = {
    "minimum_delivered_and_acked": 594,
    "maximum_ack_latency_p95_ms": 100.0,
    "maximum_outage_s": 1.0,
}


def fmt(value: object, digits: int = 2) -> str:
    if value in (None, ""):
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for directory in args.campaign_dir:
        path = Path(directory).resolve()
        for run_summary in sorted(path.glob("run_*/run_summary.json")):
            row = json.loads(run_summary.read_text(encoding="utf-8"))
            row["campaign_dir"] = path.name
            row["output_dir"] = f"{path.name}/{row.get('output_dir', run_summary.parent.name)}"
            rows.append(row)
    rows.sort(key=lambda r: (-float(r["commanded_noise_power_db"]), int(r.get("repetition_index", 1))))

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fields = ["campaign_dir"] + RUN_FIELDS
    csv_path = out / "ue_n3e_fallback_snr_floor_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    by_command: dict[float, list[dict[str, object]]] = {}
    for row in rows:
        by_command.setdefault(float(row["commanded_noise_power_db"]), []).append(row)

    passing = {
        command: runs for command, runs in by_command.items()
        if len(runs) >= 3 and all(run.get("verdict") == "PASS" for run in runs)
    }
    # Lower commanded noise == higher SNR, so the weakest replicated pass is the
    # largest (least negative) commanded value that passed all three times.
    floor_command = max(passing) if passing else None
    floor_medians = (
        [float(r["achieved_pusch_snr_db_median"]) for r in passing[floor_command]
         if r.get("achieved_pusch_snr_db_median") is not None]
        if floor_command is not None else []
    )
    floor_snr = statistics.median(floor_medians) if floor_medians else None

    lines = [
        "# UE-N3E — provisional lower operational SNR for the degraded/local fallback route",
        "",
        "**Workload:** one 2,048-byte application payload UE->DN every 100 ms with a small "
        "DN->UE ACK carrying the sequence number (~164 kbps). This is *not* the 1 Mbps "
        "workload used by UE-N3/UE-N3A.",
        "",
        "**Test type:** runtime sustain. Every run brings the RAN up at the known-good clean "
        "condition (commanded -50 dB), attaches the UE, proves the PDU tunnel and ext-DN "
        "reachability by ping, and only then applies the candidate. The good condition is "
        "restored before the next candidate. Cold attachment was not tested.",
        "",
        "**Pass criteria (all four):** >=594/600 delivered and acknowledged; ACK latency "
        "p95 <=100 ms; no UE/PDU-session disconnection; no continuous outage >=1 s.",
        "",
        "## Tested candidates",
        "",
        "| RFsim command | achieved PUSCH SNR p05/med/p95 (dB) | rep | delivered/600 | % | ACK p50/p95/max (ms) | misses >100 ms | longest outage (s) | disconnect | recovered | verdict |",
        "|---|---|---:|---:|---:|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('rfsim_command','')}` "
            f"| {fmt(row.get('achieved_pusch_snr_db_p05'),1)} / "
            f"{fmt(row.get('achieved_pusch_snr_db_median'),1)} / "
            f"{fmt(row.get('achieved_pusch_snr_db_p95'),1)} "
            f"| {row.get('repetition_index','')} "
            f"| {row.get('messages_delivered_and_acked','')} "
            f"| {fmt(row.get('delivery_ack_percent'))} "
            f"| {fmt(row.get('ack_latency_ms_p50'))} / {fmt(row.get('ack_latency_ms_p95'))} / "
            f"{fmt(row.get('ack_latency_ms_max'))} "
            f"| {row.get('deadline_misses_over_100ms','')} "
            f"| {fmt(row.get('longest_outage_s'),3)} "
            f"| {'yes: ' + str(row.get('disconnect_reason')) if row.get('ue_or_pdu_disconnected') else 'no'} "
            f"| {'yes' if row.get('recovered_after_restore') else 'no'} "
            f"| **{row.get('verdict','')}** |"
        )
    ordered = sorted(by_command)  # ascending commanded noise == descending SNR
    fails = [c for c in ordered if any(r.get("verdict") != "PASS" for r in by_command[c])]
    weakest_fail = min(fails) if fails else None

    lines += ["", "## Fail/pass boundary", ""]
    if floor_command is not None and weakest_fail is not None:
        pass_snr = fmt(floor_snr, 1)
        fail_snr = fmt(
            statistics.median([
                float(r["achieved_pusch_snr_db_median"]) for r in by_command[weakest_fail]
                if r.get("achieved_pusch_snr_db_median") is not None
            ]), 1,
        )
        lines += [
            f"- **Pass:** commanded `{floor_command}` dB -> achieved PUSCH SNR median "
            f"**{pass_snr} dB**, {len(passing[floor_command])}/"
            f"{len(passing[floor_command])} repetitions passed all four criteria.",
            f"- **Fail:** commanded `{weakest_fail}` dB -> achieved PUSCH SNR median "
            f"**{fail_snr} dB**.",
            "",
            f"The boundary is bracketed between achieved {fail_snr} dB (fail) and "
            f"{pass_snr} dB (pass).",
            "",
            "## Recommended provisional lower operational SNR",
            "",
            f"**Achieved PUSCH SNR median {pass_snr} dB** (commanded RFsim "
            f"`noise_power_dB {floor_command}`) for the degraded/local fallback route.",
        ]
    elif floor_command is None:
        lines += [
            "- No candidate passed all three repetitions.",
            "",
            "Measured fail/pass bracket only; no provisional floor is recommended.",
        ]
    lines += [
        "",
        "## Limitations",
        "",
        "- This is a **runtime sustain** bound for an already-connected UE, not a cold-attach "
        "limit and not a universal physical limit. Cold attachment at these conditions is a "
        "separate open question (UE-N3B/N3C/N3D).",
        "- The bound is workload-specific: 2 KB per 100 ms with a small ACK. The existing "
        "1 Mbps UE-N3/UE-N3A results remain valid as higher-load evidence and are not "
        "superseded.",
        "- Achieved SNR is the gNB `GNB_MAC_PUSCH_POWER_CONTROL` measurement over the exact "
        "60-second window; RFsim `noise_power_dB` is the commanded knob, not the achieved SNR.",
        "- Single UE, 106 PRB, band 78, AWGN RFsim channel, SINR-driven UL MCS, no CARLA load. "
        "Multi-UE contention, fading, and mobility are out of scope.",
        "- ACK latency is a UE-side single-clock monotonic round trip (UE app -> DN -> UE app).",
        "- **Latency margin at the floor is thin.** All three passing runs sit at p50 ~44 ms and "
        "p95 ~77 ms, but maximum ACK latency is 97.6 / 97.8 / 107.7 ms. Repetition 3 put 3 of 600 "
        "messages over 100 ms. The p95 criterion passes comfortably; the per-message worst case "
        "does not have headroom, so a hard per-message 100 ms deadline is not guaranteed at 5.5 dB.",
        "- The failing candidate degraded purely by **loss**, not by latency or by disconnection: "
        "at 5.0 dB the ACK p95 was still 82.7 ms with no outage >=1 s and no session drop, yet "
        "only 52.8% of messages were delivered. Link-quality collapse here shows up as uplink "
        "erasure, so a latency-only health check would not detect it.",
        "- The step between the tested rungs is 0.25 dB commanded, which resolved to a 0.5 dB "
        "difference in achieved median SNR (5.0 vs 5.5). The true boundary lies somewhere in "
        "that interval; it was not resolved more finely.",
        "- Achieved SNR is reported at the measurement resolution of the tracer (0.5 dB steps in "
        "`snrx10`-derived medians), so 5.0 and 5.5 dB are adjacent quantized levels.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "# 1) descent from the first weak candidate (stops at the first failure)",
        "/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \\",
        "  -m rl_agent.ue_n3e_fallback_snr_floor_v1 \\",
        "  --config rl_agent/configs/ue_n3e_fallback_snr_floor_v1.json \\",
        "  --output-dir rl_agent/experiments/ue_n3e_fallback_snr_floor_v1/20260821_live_01",
        "",
        "# 2) 0.25 dB refinement at -2.25, auto-replicated to three runs on a pass",
        "/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \\",
        "  -m rl_agent.ue_n3e_fallback_snr_floor_v1 \\",
        "  --config rl_agent/configs/ue_n3e_fallback_snr_floor_v1.json \\",
        "  --output-dir rl_agent/experiments/ue_n3e_fallback_snr_floor_v1/20260821_live_02 \\",
        "  --start-db -2.25 --stop-db -2.25 --step-db 0.25",
        "",
        "# 3) combine both campaigns into the CSV and this report",
        "/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \\",
        "  -m rl_agent.ue_n3e_fallback_snr_floor_report \\",
        "  --campaign-dir rl_agent/experiments/ue_n3e_fallback_snr_floor_v1/20260821_live_01 \\",
        "  --campaign-dir rl_agent/experiments/ue_n3e_fallback_snr_floor_v1/20260821_live_02 \\",
        "  --out-dir rl_agent/experiments/ue_n3e_fallback_snr_floor_v1",
        "```",
    ]
    (out / "UE_N3E_FALLBACK_SNR_FLOOR_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "csv": str(csv_path),
        "report": str(out / "UE_N3E_FALLBACK_SNR_FLOOR_REPORT.md"),
        "runs": len(rows),
        "floor_command_db": floor_command,
        "floor_achieved_snr_db_median": floor_snr,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
