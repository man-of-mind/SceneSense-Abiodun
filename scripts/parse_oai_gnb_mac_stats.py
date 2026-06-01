#!/usr/bin/env python3
"""Parse OAI gNB MAC stdout summaries into CSV files."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


FRAME_RE = re.compile(r"\[NR_MAC\]\s+Frame\.Slot\s+(\d+)\.(\d+)")
UE_RE = re.compile(
    r"UE RNTI ([0-9a-fA-F]+) CU-UE-ID (\S+) "
    r"(in-sync|out-of-sync) PH (-?\d+) dB PCMAX (-?\d+) dBm"
    r"(?:, average RSRP (-?\d+) \((\d+) meas\))?"
    r"(?:, average SINR (-?\d+)\.(\d+) \((\d+) meas\))?"
)
DLSCH_RE = re.compile(
    r"UE ([0-9a-fA-F]+): dlsch_rounds ([0-9/]+), "
    r"dlsch_errors (\d+), pucch0_DTX (\d+) "
    r"\(SNR ([+-]?\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?) dB\), "
    r"BLER ([0-9.]+) MCS \((\d+)\) (\d+) CCE fail (\d+)"
)
ULSCH_RE = re.compile(
    r"UE ([0-9a-fA-F]+): ulsch_rounds ([0-9/]+), "
    r"ulsch_errors (\d+), ulsch_DTX (\d+), BLER ([0-9.]+) "
    r"MCS \((\d+)\) (\d+) \(Qm (\d+) deltaMCS (-?\d+) dB\) "
    r"NPRB (\d+) SNR ([+-]?\d+(?:\.\d+)?) "
    r"\(([+-]\d+(?:\.\d+)?)\) dB CCE fail (\d+)"
)
MAC_RE = re.compile(r"UE ([0-9a-fA-F]+): MAC:\s+TX\s+(\d+)\s+RX\s+(\d+) bytes")
LCID_RE = re.compile(r"UE ([0-9a-fA-F]+): LCID (\d+): TX\s+(\d+)\s+RX\s+(\d+) bytes")


def parse_rounds(value: str, prefix: str) -> dict[str, int]:
    parts = [int(x) for x in value.split("/") if x != ""]
    return {f"{prefix}{idx}": parts[idx] if idx < len(parts) else 0 for idx in range(4)}


def maybe_int(value: str | None) -> int | None:
    return int(value) if value is not None else None


def get_row(rows: dict[tuple[int, str], dict[str, object]], sample_index: int, frame: int | None, slot: int | None, rnti_hex: str) -> dict[str, object]:
    key = (sample_index, rnti_hex.lower())
    if key not in rows:
        rows[key] = {
            "sample_index": sample_index,
            "frame": frame,
            "slot": slot,
            "rnti_hex": rnti_hex.lower(),
            "rnti_dec": int(rnti_hex, 16),
        }
    return rows[key]


def parse_lines(lines: list[str]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sample_index = -1
    current_frame: int | None = None
    current_slot: int | None = None
    rows: dict[tuple[int, str], dict[str, object]] = {}
    lcid_rows: list[dict[str, object]] = []

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        if match := FRAME_RE.search(line):
            sample_index += 1
            current_frame = int(match.group(1))
            current_slot = int(match.group(2))
            continue

        if sample_index < 0:
            continue

        if match := UE_RE.search(line):
            rnti_hex = match.group(1)
            row = get_row(rows, sample_index, current_frame, current_slot, rnti_hex)
            sinr = None
            if match.group(8) is not None and match.group(9) is not None:
                sinr = float(f"{match.group(8)}.{match.group(9)}")
            row.update(
                {
                    "line_number": line_number,
                    "cu_ue_id": match.group(2),
                    "sync_state": match.group(3),
                    "ph_db": int(match.group(4)),
                    "pcmax_dbm": int(match.group(5)),
                    "average_rsrp_dbm": maybe_int(match.group(6)),
                    "rsrp_meas_count": maybe_int(match.group(7)),
                    "average_sinr_db": sinr,
                    "sinr_meas_count": maybe_int(match.group(10)),
                }
            )
            continue

        if match := DLSCH_RE.search(line):
            row = get_row(rows, sample_index, current_frame, current_slot, match.group(1))
            row.update(parse_rounds(match.group(2), "dlsch_round"))
            row.update(
                {
                    "dlsch_errors": int(match.group(3)),
                    "pucch0_dtx": int(match.group(4)),
                    "pucch_snr_db": float(match.group(5)),
                    "pucch_snr_delta_db": float(match.group(6)),
                    "dlsch_bler": float(match.group(7)),
                    "dlsch_mcs_table": int(match.group(8)),
                    "dlsch_mcs": int(match.group(9)),
                    "dlsch_cce_fail": int(match.group(10)),
                }
            )
            continue

        if match := ULSCH_RE.search(line):
            row = get_row(rows, sample_index, current_frame, current_slot, match.group(1))
            row.update(parse_rounds(match.group(2), "ulsch_round"))
            row.update(
                {
                    "ulsch_errors": int(match.group(3)),
                    "ulsch_dtx": int(match.group(4)),
                    "ulsch_bler": float(match.group(5)),
                    "ulsch_mcs_table": int(match.group(6)),
                    "ulsch_mcs": int(match.group(7)),
                    "ulsch_qm": int(match.group(8)),
                    "ulsch_delta_mcs_db": int(match.group(9)),
                    "ulsch_nprb": int(match.group(10)),
                    "ulsch_snr_db": float(match.group(11)),
                    "ulsch_snr_delta_db": float(match.group(12)),
                    "ulsch_cce_fail": int(match.group(13)),
                }
            )
            continue

        if match := MAC_RE.search(line):
            row = get_row(rows, sample_index, current_frame, current_slot, match.group(1))
            row.update({"mac_tx_bytes": int(match.group(2)), "mac_rx_bytes": int(match.group(3))})
            continue

        if match := LCID_RE.search(line):
            rnti_hex = match.group(1).lower()
            lcid_rows.append(
                {
                    "sample_index": sample_index,
                    "frame": current_frame,
                    "slot": current_slot,
                    "rnti_hex": rnti_hex,
                    "rnti_dec": int(rnti_hex, 16),
                    "lcid": int(match.group(2)),
                    "lcid_tx_bytes": int(match.group(3)),
                    "lcid_rx_bytes": int(match.group(4)),
                }
            )

    summary_rows = sorted(rows.values(), key=lambda row: (int(row["sample_index"]), str(row["rnti_hex"])))
    return summary_rows, lcid_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="gNB stdout log path, or '-' for stdin.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults beside the input log.")
    args = parser.parse_args()

    if args.input == "-":
        lines = sys.stdin.read().splitlines()
        output_dir = Path(args.output_dir or ".").resolve()
    else:
        input_path = Path(args.input).expanduser().resolve()
        lines = input_path.read_text(errors="replace").splitlines()
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_path.with_suffix("").parent / f"{input_path.stem}_parsed"

    summary_rows, lcid_rows = parse_lines(lines)

    summary_fields = [
        "sample_index",
        "frame",
        "slot",
        "line_number",
        "rnti_hex",
        "rnti_dec",
        "cu_ue_id",
        "sync_state",
        "ph_db",
        "pcmax_dbm",
        "average_rsrp_dbm",
        "rsrp_meas_count",
        "average_sinr_db",
        "sinr_meas_count",
        "dlsch_round0",
        "dlsch_round1",
        "dlsch_round2",
        "dlsch_round3",
        "dlsch_errors",
        "pucch0_dtx",
        "pucch_snr_db",
        "pucch_snr_delta_db",
        "dlsch_bler",
        "dlsch_mcs_table",
        "dlsch_mcs",
        "dlsch_cce_fail",
        "ulsch_round0",
        "ulsch_round1",
        "ulsch_round2",
        "ulsch_round3",
        "ulsch_errors",
        "ulsch_dtx",
        "ulsch_bler",
        "ulsch_mcs_table",
        "ulsch_mcs",
        "ulsch_qm",
        "ulsch_delta_mcs_db",
        "ulsch_nprb",
        "ulsch_snr_db",
        "ulsch_snr_delta_db",
        "ulsch_cce_fail",
        "mac_tx_bytes",
        "mac_rx_bytes",
    ]
    lcid_fields = [
        "sample_index",
        "frame",
        "slot",
        "rnti_hex",
        "rnti_dec",
        "lcid",
        "lcid_tx_bytes",
        "lcid_rx_bytes",
    ]

    summary_path = output_dir / "gnb_mac_stdout_summary.csv"
    lcid_path = output_dir / "gnb_mac_stdout_lcid.csv"
    write_csv(summary_path, summary_rows, summary_fields)
    write_csv(lcid_path, lcid_rows, lcid_fields)

    print(f"[gnb-mac-parser] summary_rows={len(summary_rows)} -> {summary_path}")
    print(f"[gnb-mac-parser] lcid_rows={len(lcid_rows)} -> {lcid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
