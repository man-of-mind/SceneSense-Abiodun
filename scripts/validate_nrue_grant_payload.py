#!/usr/bin/env python3
"""Validate NRUE_MAC_DCI_GRANT UL TBS against UE_PHY_UL_PAYLOAD_TX_BITS."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple


ABIODUN_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TTRACER_ROOT = ABIODUN_DIR / "metrics_logs" / "scenesense_ttracer"

FIELDS = (
    "run_group",
    "rnti",
    "grant_ul_events",
    "payload_events",
    "matched_events",
    "grant_without_payload_events",
    "payload_without_grant_events",
    "grant_ul_bits",
    "payload_bits",
    "matched_bits",
    "grant_without_payload_bits",
    "payload_without_grant_bits",
    "payload_to_grant_bit_ratio",
)


Key = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether each decoded UE UL grant's TBS maps to OAI's existing "
            "UE_PHY_UL_PAYLOAD_TX_BITS trace. The comparison key is "
            "(rnti, scheduled frame, scheduled slot, tbs*8)."
        )
    )
    parser.add_argument("--run-group", required=True, help="T-tracer run group to validate.")
    parser.add_argument(
        "--root",
        default=str(DEFAULT_TTRACER_ROOT),
        help="T-tracer metrics root.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to <run_group>/ue/analysis.",
    )
    return parser.parse_args()


def to_int(value: object) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def fmt_ratio(num: int, den: int) -> str:
    if den == 0:
        return ""
    return f"{num / den:.9f}"


def load_grants(path: Path) -> Counter[Key]:
    events: Counter[Key] = Counter()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if to_int(row.get("direction")) != 1:
                continue
            key = (
                to_int(row.get("rnti")),
                to_int(row.get("sched_frame")),
                to_int(row.get("sched_slot")),
                to_int(row.get("tbs")) * 8,
            )
            events[key] += 1
    return events


def load_payload(path: Path) -> Counter[Key]:
    events: Counter[Key] = Counter()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (
                to_int(row.get("rnti")),
                to_int(row.get("frame")),
                to_int(row.get("slot")),
                to_int(row.get("number_of_bits")),
            )
            events[key] += 1
    return events


def bit_sum(events: Counter[Key]) -> int:
    return sum(bits * count for (_, _, _, bits), count in events.items())


def event_sum(events: Counter[Key]) -> int:
    return sum(events.values())


def split_by_rnti(events: Counter[Key]) -> Dict[int, Counter[Key]]:
    result: Dict[int, Counter[Key]] = defaultdict(Counter)
    for key, count in events.items():
        result[key[0]][key] += count
    return result


def make_rows(run_group: str, grants: Counter[Key], payload: Counter[Key]) -> list[dict[str, str]]:
    grant_by_rnti = split_by_rnti(grants)
    payload_by_rnti = split_by_rnti(payload)
    rows: list[dict[str, str]] = []
    for rnti in sorted(set(grant_by_rnti) | set(payload_by_rnti)):
        g = grant_by_rnti.get(rnti, Counter())
        p = payload_by_rnti.get(rnti, Counter())
        matched = g & p
        grant_missing_payload = g - p
        payload_missing_grant = p - g
        grant_bits = bit_sum(g)
        payload_bits = bit_sum(p)
        rows.append(
            {
                "run_group": run_group,
                "rnti": f"0x{rnti:04x}",
                "grant_ul_events": str(event_sum(g)),
                "payload_events": str(event_sum(p)),
                "matched_events": str(event_sum(matched)),
                "grant_without_payload_events": str(event_sum(grant_missing_payload)),
                "payload_without_grant_events": str(event_sum(payload_missing_grant)),
                "grant_ul_bits": str(grant_bits),
                "payload_bits": str(payload_bits),
                "matched_bits": str(bit_sum(matched)),
                "grant_without_payload_bits": str(bit_sum(grant_missing_payload)),
                "payload_without_grant_bits": str(bit_sum(payload_missing_grant)),
                "payload_to_grant_bit_ratio": fmt_ratio(payload_bits, grant_bits),
            }
        )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# NR UE Grant Payload Validation\n\n")
        handle.write("| RNTI | Grant UL events | Payload events | Matched events | Grant bits | Payload bits | Extra payload bits | Payload/grant ratio |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                "| {rnti} | {grant_ul_events} | {payload_events} | {matched_events} | "
                "{grant_ul_bits} | {payload_bits} | {payload_without_grant_bits} | "
                "{payload_to_grant_bit_ratio} |\n".format(**row)
            )
        handle.write(
            "\nA clean validation has zero `grant_without_payload_events`. "
            "`payload_without_grant_events` can appear at trace boundaries if payload "
            "events were already active just before the first decoded grant event was captured.\n"
        )


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    run_dir = root / args.run_group
    grant_csv = run_dir / "ue" / "csv" / "NRUE_MAC_DCI_GRANT.csv"
    payload_csv = run_dir / "ue" / "csv" / "UE_PHY_UL_PAYLOAD_TX_BITS.csv"
    if not grant_csv.exists():
        print(f"[validate_nrue_grant_payload] missing {grant_csv}", file=sys.stderr)
        return 1
    if not payload_csv.exists():
        print(f"[validate_nrue_grant_payload] missing {payload_csv}", file=sys.stderr)
        return 1

    grants = load_grants(grant_csv)
    payload = load_payload(payload_csv)
    rows = make_rows(args.run_group, grants, payload)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "ue" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "nrue_grant_payload_validation.csv"
    md_path = output_dir / "nrue_grant_payload_validation.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    grant_missing = sum(int(row["grant_without_payload_events"]) for row in rows)
    payload_missing = sum(int(row["payload_without_grant_events"]) for row in rows)
    print(f"[validate_nrue_grant_payload] grant_without_payload_events={grant_missing}")
    print(f"[validate_nrue_grant_payload] payload_without_grant_events={payload_missing}")
    print(f"[validate_nrue_grant_payload] wrote {csv_path}")
    print(f"[validate_nrue_grant_payload] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
