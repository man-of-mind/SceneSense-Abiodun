#!/usr/bin/env python3
"""Compare UE decoded NR grants with gNB scheduler/PHY T-tracer metrics."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Sequence, Tuple


ABIODUN_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TTRACER_ROOT = ABIODUN_DIR / "metrics_logs" / "scenesense_ttracer"

FIELDS = (
    "run_group",
    "rnti",
    "direction",
    "ue_grants",
    "ue_tbs_bytes",
    "gnb_mac_grants",
    "gnb_mac_tbs_bytes",
    "gnb_phy_rows",
    "gnb_phy_payload_bytes",
    "ue_vs_gnb_mac_tbs_ratio",
    "ue_minus_gnb_mac_tbs_bytes",
    "ue_vs_gnb_phy_payload_ratio",
    "ue_minus_gnb_phy_payload_bytes",
    "ue_avg_mcs",
    "gnb_mac_avg_mcs",
    "gnb_phy_avg_mcs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate NRUE_MAC_DCI_GRANT against gNB MAC/PHY CSVs for the same "
            "run_group. This compares aggregate per-RNTI UL/DL TBS and MCS."
        )
    )
    parser.add_argument("--run-group", required=True, help="T-tracer run group to compare.")
    parser.add_argument(
        "--root",
        default=str(DEFAULT_TTRACER_ROOT),
        help="T-tracer metrics root.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to <run_group>/analysis.",
    )
    return parser.parse_args()


def to_int(value: object) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def safe_ratio(num: float, den: float) -> float:
    return num / den if den else float("nan")


def fmt(value: float, digits: int = 6) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def fmt_count(value: float) -> str:
    if not math.isfinite(value):
        return ""
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return fmt(value)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def direction_label(direction: int) -> str:
    return "ul" if direction == 1 else "dl" if direction == 0 else "unknown"


def normalize_ue_tbs_bytes(direction: int, raw_tbs: int) -> float:
    # In this OAI tree, UE UL grant tb_size is bytes while UE DL tb_size is bits.
    # Normalize both directions to bytes before comparing against gNB MAC TBS.
    return float(raw_tbs) if direction == 1 else float(raw_tbs) / 8.0


def require_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)


def add_metric(
    data: DefaultDict[Tuple[int, int], Dict[str, object]],
    rnti: int,
    direction: int,
    prefix: str,
    tbs_bytes: float,
    mcs: int | None = None,
) -> None:
    row = data[(rnti, direction)]
    row[f"{prefix}_grants"] = int(row.get(f"{prefix}_grants", 0)) + 1
    row[f"{prefix}_tbs_bytes"] = float(row.get(f"{prefix}_tbs_bytes", 0.0)) + tbs_bytes
    if mcs is not None:
        values = row.setdefault(f"{prefix}_mcs_values", [])
        assert isinstance(values, list)
        values.append(mcs)


def load_ue(path: Path, data: DefaultDict[Tuple[int, int], Dict[str, object]]) -> None:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            direction = to_int(raw.get("direction"))
            add_metric(
                data,
                to_int(raw.get("rnti")),
                direction,
                "ue",
                normalize_ue_tbs_bytes(direction, to_int(raw.get("tbs"))),
                to_int(raw.get("mcs")),
            )


def load_gnb_mac(path: Path, direction: int, data: DefaultDict[Tuple[int, int], Dict[str, object]]) -> None:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            add_metric(
                data,
                to_int(raw.get("rnti")),
                direction,
                "gnb_mac",
                to_int(raw.get("tbs")),
                to_int(raw.get("mcs")),
            )


def load_gnb_phy_ul(path: Path, data: DefaultDict[Tuple[int, int], Dict[str, object]]) -> None:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = data[(to_int(raw.get("rnti")), 1)]
            row["gnb_phy_rows"] = int(row.get("gnb_phy_rows", 0)) + 1
            row["gnb_phy_payload_bits"] = int(row.get("gnb_phy_payload_bits", 0)) + to_int(raw.get("number_of_bits"))
            values = row.setdefault("gnb_phy_mcs_values", [])
            assert isinstance(values, list)
            values.append(to_int(raw.get("mcs_index")))


def make_rows(run_group: str, data: DefaultDict[Tuple[int, int], Dict[str, object]]) -> list[Dict[str, str]]:
    rows: list[Dict[str, str]] = []
    for (rnti, direction), raw in sorted(data.items()):
        ue_tbs = float(raw.get("ue_tbs_bytes", 0.0))
        gnb_mac_tbs = float(raw.get("gnb_mac_tbs_bytes", 0.0))
        gnb_phy_bits = int(raw.get("gnb_phy_payload_bits", 0))
        gnb_phy_bytes = gnb_phy_bits / 8.0

        ue_mcs_values = raw.get("ue_mcs_values", [])
        gnb_mac_mcs_values = raw.get("gnb_mac_mcs_values", [])
        gnb_phy_mcs_values = raw.get("gnb_phy_mcs_values", [])
        assert isinstance(ue_mcs_values, list)
        assert isinstance(gnb_mac_mcs_values, list)
        assert isinstance(gnb_phy_mcs_values, list)

        rows.append(
            {
                "run_group": run_group,
                "rnti": f"0x{rnti:04x}",
                "direction": direction_label(direction),
                "ue_grants": str(int(raw.get("ue_grants", 0))),
                "ue_tbs_bytes": fmt_count(ue_tbs),
                "gnb_mac_grants": str(int(raw.get("gnb_mac_grants", 0))),
                "gnb_mac_tbs_bytes": fmt_count(gnb_mac_tbs),
                "gnb_phy_rows": str(int(raw.get("gnb_phy_rows", 0))),
                "gnb_phy_payload_bytes": fmt(gnb_phy_bytes),
                "ue_vs_gnb_mac_tbs_ratio": fmt(safe_ratio(ue_tbs, gnb_mac_tbs)),
                "ue_minus_gnb_mac_tbs_bytes": fmt(ue_tbs - gnb_mac_tbs),
                "ue_vs_gnb_phy_payload_ratio": fmt(safe_ratio(ue_tbs, gnb_phy_bytes)),
                "ue_minus_gnb_phy_payload_bytes": fmt(ue_tbs - gnb_phy_bytes),
                "ue_avg_mcs": fmt(mean(ue_mcs_values)),
                "gnb_mac_avg_mcs": fmt(mean(gnb_mac_mcs_values)),
                "gnb_phy_avg_mcs": fmt(mean(gnb_phy_mcs_values)),
            }
        )
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# UE-vs-gNB Grant Validation\n\n")
        handle.write("| RNTI | Dir | UE grants | gNB MAC grants | UE TBS bytes | gNB MAC TBS bytes | UE/gNB MAC ratio | gNB PHY bytes | UE/gNB PHY ratio | UE MCS | gNB MAC MCS |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                "| {rnti} | {direction} | {ue_grants} | {gnb_mac_grants} | "
                "{ue_tbs_bytes} | {gnb_mac_tbs_bytes} | {ue_vs_gnb_mac_tbs_ratio} | "
                "{gnb_phy_payload_bytes} | {ue_vs_gnb_phy_payload_ratio} | "
                "{ue_avg_mcs} | {gnb_mac_avg_mcs} |\n".format(**row)
            )
        handle.write(
            "\nRatios close to 1.0 are expected when UE and gNB traces cover the same "
            "time interval. Small differences can come from starting the two recorders at "
            "slightly different times.\n"
        )


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    run_dir = root / args.run_group
    ue_csv = run_dir / "ue" / "csv" / "NRUE_MAC_DCI_GRANT.csv"
    gnb_ul_csv = run_dir / "gnb" / "csv" / "GNB_MAC_UL.csv"
    gnb_dl_csv = run_dir / "gnb" / "csv" / "GNB_MAC_DL.csv"
    gnb_phy_ul_csv = run_dir / "gnb" / "csv" / "GNB_PHY_UL_PAYLOAD_RX_BITS.csv"

    try:
        for path in (ue_csv, gnb_ul_csv, gnb_dl_csv, gnb_phy_ul_csv):
            require_file(path)
    except OSError as exc:
        print(f"[compare_nrue_gnb_grants] missing input: {exc}", file=sys.stderr)
        return 1

    data: DefaultDict[Tuple[int, int], Dict[str, object]] = defaultdict(dict)
    load_ue(ue_csv, data)
    load_gnb_mac(gnb_ul_csv, 1, data)
    load_gnb_mac(gnb_dl_csv, 0, data)
    load_gnb_phy_ul(gnb_phy_ul_csv, data)
    rows = make_rows(args.run_group, data)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ue_gnb_grant_validation.csv"
    md_path = output_dir / "ue_gnb_grant_validation.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"[compare_nrue_gnb_grants] wrote {csv_path}")
    print(f"[compare_nrue_gnb_grants] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
