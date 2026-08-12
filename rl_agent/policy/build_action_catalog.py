"""Generate the canonical seven-profile Track A catalog from the measured Markdown table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .catalog import RETAINED_PROFILES
from .config import REPO_ROOT


SOURCE = REPO_ROOT / "rl_agent" / "PERMODEL_KNOB_MATRIX_ZSTD.md"
VEHICLE_RECALL_EVAL_ROOT = (
    REPO_ROOT / "experiments" / "ae_integrated_20260710" / "sweeps_permodel_zstd"
)
OUTPUT = REPO_ROOT / "rl_agent" / "policy" / "data" / "action_catalog.csv"
META = REPO_ROOT / "rl_agent" / "policy" / "data" / "action_catalog.meta.json"


def _parse_float(value: str) -> float:
    return float(value.strip().replace("~", ""))


def _vehicle_recall_measurement(profile_id: str) -> tuple[float, Path, str]:
    """Load the class-specific recall omitted from the published Markdown table."""

    metrics_path = (
        VEHICLE_RECALL_EVAL_ROOT
        / profile_id
        / "metrics"
        / "test_fusion_evaluation_metrics.json"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    key = "learned_vehicle_object_recall"
    if key not in metrics:
        raise ValueError(f"{metrics_path} is missing {key}")
    value = float(metrics[key])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"invalid vehicle recall {value} in {metrics_path}")
    return value, metrics_path, hashlib.sha256(metrics_path.read_bytes()).hexdigest()


def parse_matrix(source: Path = SOURCE) -> pd.DataFrame:
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "__" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 17 or cells[0] not in RETAINED_PROFILES:
            continue
        vehicle_recall, vehicle_source, vehicle_source_sha256 = (
            _vehicle_recall_measurement(cells[0])
        )
        rows.append(
            {
                "profile_id": cells[0],
                "quantization": cells[1],
                "entropy_codec": cells[2],
                "roi_q": _parse_float(cells[3]),
                "ae_bottleneck": int(cells[4]),
                "payload_kib": _parse_float(cells[5]),
                "miou": _parse_float(cells[7]),
                "vehicle_iou": _parse_float(cells[8]),
                "pedestrian_recall": _parse_float(cells[9]),
                "vehicle_recall": vehicle_recall,
                "object_recall": _parse_float(cells[10]),
                "base_loc_raw_m": _parse_float(cells[11]),
                "pedestrian_loc_m": _parse_float(cells[12]),
                "front_ms": _parse_float(cells[13]),
                "back_ms": _parse_float(cells[14]),
                "loopback_transport_ms": _parse_float(cells[15]),
                "vehicle_recall_source_file": str(vehicle_source.relative_to(REPO_ROOT)),
                "vehicle_recall_source_sha256": vehicle_source_sha256,
            }
        )
    frame = pd.DataFrame(rows)
    if set(frame.get("profile_id", [])) != set(RETAINED_PROFILES):
        missing = set(RETAINED_PROFILES) - set(frame.get("profile_id", []))
        raise ValueError(f"failed to parse retained profiles: {sorted(missing)}")
    frame = frame.set_index("profile_id").loc[list(RETAINED_PROFILES)].reset_index()
    floor = float(frame["base_loc_raw_m"].min())
    frame["base_loc_calibrated_m"] = 1.10 + frame["base_loc_raw_m"] - floor
    frame["source_file"] = str(source.relative_to(REPO_ROOT))
    frame["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return frame


def build(output: Path = OUTPUT, meta_path: Path = META) -> pd.DataFrame:
    frame = parse_matrix()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, float_format="%.6f")
    metadata = {
        "schema_version": 1,
        "source_file": str(SOURCE.relative_to(REPO_ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "catalog_file": str(output.relative_to(REPO_ROOT)),
        "catalog_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "profile_count": int(len(frame)),
        "flattened_action_count": int(len(frame) * 5 + 1),
        "base_loc_calibration": "1.10 + raw - min(retained_raw)",
        "vehicle_recall_field": "learned_vehicle_object_recall",
        "vehicle_recall_reference": {
            "value": 0.927,
            "definition": "rounded best measured profile recall in sweeps_permodel_zstd",
        },
        "vehicle_recall_sources": {
            str(row["profile_id"]): {
                "file": str(row["vehicle_recall_source_file"]),
                "sha256": str(row["vehicle_recall_source_sha256"]),
            }
            for row in frame.to_dict(orient="records")
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--meta", type=Path, default=META)
    args = parser.parse_args()
    frame = build(args.output, args.meta)
    print(f"wrote {len(frame)} profiles and {len(frame) * 5 + 1} flattened actions to {args.output}")


if __name__ == "__main__":
    main()
