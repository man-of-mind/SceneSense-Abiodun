"""Build and validate the deterministic Phase-2 Suite A/B design manifest.

This is an offline design tool. It never imports CARLA, launches a process, or
authorizes collection. The output enumerates independent scenario groups and
their frozen calibration/validation/test assignment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml
from scipy.stats import nct, t


SCHEMA = "scenesense.phase2_suite_design_manifest.v1"
SPLITS = ("calibration", "validation", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_seed(master_seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1) + 1


def _deterministic_multiset(
    counts: Mapping[str, int], *, master_seed: int, namespace: str
) -> list[str]:
    tagged = []
    for value, count in counts.items():
        for occurrence in range(int(count)):
            key = hashlib.sha256(
                f"{master_seed}:{namespace}:{value}:{occurrence}".encode("utf-8")
            ).digest()
            tagged.append((key, str(value)))
    return [value for _key, value in sorted(tagged)]


def paired_t_power(
    *, sample_count: int, effect_s: float, paired_sd_s: float, alpha: float
) -> float:
    if sample_count < 2 or effect_s <= 0 or paired_sd_s <= 0 or not 0 < alpha < 1:
        raise ValueError("invalid paired-power inputs")
    degrees = sample_count - 1
    critical = float(t.ppf(1.0 - alpha / 2.0, degrees))
    noncentrality = float(effect_s / paired_sd_s * math.sqrt(sample_count))
    return float(
        nct.cdf(-critical, degrees, noncentrality)
        + 1.0
        - nct.cdf(critical, degrees, noncentrality)
    )


def _retention_tier(split: str, *, audit: bool, config: Mapping[str, object]) -> str:
    retention = config["retention"]
    if audit:
        return str(retention["audit_tier"])
    return str(retention[f"{split}_tier"])


def build_manifest(config: Mapping[str, object]) -> pd.DataFrame:
    if config.get("schema_version") != "scenesense.phase2_suite_design.v1":
        raise ValueError("unsupported suite-design config schema")
    if any(bool(value) for value in config["authorization"].values()):
        raise ValueError("suite-design config must not authorize runtime work")

    master_seed = int(config["master_seed"])
    common = config["common"]
    renderer = common.get("renderer_quality")
    expected_renderer = {
        "primary_level": "Epic",
        "required_server_launch_flag": "-quality-level=Epic",
        "provenance_source": "operator_declared_server_launch_flag",
        "rpc_introspection_available": False,
        "existing_stress_level": "Low",
        "future_low_collection_authorized": False,
    }
    if renderer != expected_renderer:
        raise ValueError(
            "Suite A/B renderer contract must lock explicit Epic primary and "
            "keep Low as an existing, non-collecting stress condition"
        )
    raw_window_s = float(config["retention"]["raw_window_duration_s"])
    rows: list[dict] = []

    suite_a = config["suite_a"]
    replicates = int(suite_a["replicates_per_factor_cell"])
    split_by_replicate = {
        split: {int(value) for value in suite_a["split_by_replicate"][split]}
        for split in SPLITS
    }
    density_cycle = [str(value) for value in suite_a["traffic_density_cycle"]]
    cell_index = 0
    for geometry in suite_a["geometries"]:
        for closing_band in suite_a["closing_speed_bands"]:
            for tth_band in suite_a["time_to_hazard_bands"]:
                for replicate in range(replicates):
                    matching = [
                        split
                        for split, indices in split_by_replicate.items()
                        if replicate in indices
                    ]
                    if len(matching) != 1:
                        raise ValueError(
                            f"Suite A replicate {replicate} maps to {matching}"
                        )
                    split = matching[0]
                    group_id = (
                        f"sa_{geometry['geometry_id']}_{closing_band}_"
                        f"{tth_band}_r{replicate:02d}"
                    )
                    audit = (
                        split == "calibration"
                        and closing_band == next(iter(suite_a["closing_speed_bands"]))
                        and tth_band == next(iter(suite_a["time_to_hazard_bands"]))
                    )
                    density = density_cycle[(cell_index + replicate) % len(density_cycle)]
                    shared = {
                        "schema": SCHEMA,
                        "design_id": config["design_id"],
                        "suite_id": "A",
                        "suite_label": suite_a["label"],
                        "split": split,
                        "group_id": group_id,
                        "matched_pair_id": group_id,
                        "geometry_or_route_id": geometry["geometry_id"],
                        "geometry_or_route_status": geometry["implementation_status"],
                        "hazard_class": geometry["hazard_class"],
                        "closing_speed_band": closing_band,
                        "time_to_hazard_band": tth_band,
                        "traffic_density": density,
                        "weather": suite_a["weather"],
                        "renderer_quality_level": renderer["primary_level"],
                        "renderer_server_launch_flag": renderer[
                            "required_server_launch_flag"
                        ],
                        "renderer_contract_role": "primary",
                        "carla_seed": _stable_seed(master_seed, f"{group_id}:carla"),
                        "traffic_seed": _stable_seed(master_seed, f"{group_id}:traffic"),
                        "sensor_seed": _stable_seed(master_seed, f"{group_id}:sensor"),
                        "raw_retention_tier": _retention_tier(
                            split, audit=audit, config=config
                        ),
                        "raw_window_duration_s": raw_window_s if split != "test" else 0.0,
                        "raw_window_anchor": (
                            config["retention"]["raw_window_anchor"]
                            if split != "test"
                            else "none"
                        ),
                        "confirmatory_locked": int(split == "test"),
                    }
                    for role, present in (
                        ("controlled_positive_occlusion", 1),
                        ("matched_benign_negative", 0),
                    ):
                        rows.append(
                            {
                                **shared,
                                "trajectory_id": f"{group_id}_{'pos' if present else 'ben'}",
                                "scenario_role": role,
                                "controlled_hazard_present": present,
                            }
                        )
                cell_index += 1

    suite_b = config["suite_b"]
    for route in suite_b["routes"]:
        route_id = str(route["route_id"])
        route_offset = 0
        for split in SPLITS:
            group_count = int(suite_b["split_counts_per_route"][split])
            density = _deterministic_multiset(
                suite_b["density_counts_per_route_split"][split],
                master_seed=master_seed,
                namespace=f"{route_id}:{split}:density",
            )
            weather = _deterministic_multiset(
                suite_b["weather_counts_per_route_split"][split],
                master_seed=master_seed,
                namespace=f"{route_id}:{split}:weather",
            )
            if len(density) != group_count or len(weather) != group_count:
                raise ValueError(f"Suite B quota mismatch for {route_id}/{split}")
            for local_index in range(group_count):
                ordinal = route_offset + local_index
                group_id = f"sb_{route_id}_r{ordinal:02d}"
                audit = split == "calibration" and local_index == 0
                rows.append(
                    {
                        "schema": SCHEMA,
                        "design_id": config["design_id"],
                        "suite_id": "B",
                        "suite_label": suite_b["label"],
                        "split": split,
                        "group_id": group_id,
                        "matched_pair_id": "",
                        "trajectory_id": f"{group_id}_natural",
                        "scenario_role": "naturalistic_operation",
                        "controlled_hazard_present": "unforced",
                        "geometry_or_route_id": route_id,
                        "geometry_or_route_status": route["implementation_status"],
                        "hazard_class": "natural_prevalence",
                        "closing_speed_band": "natural",
                        "time_to_hazard_band": "natural",
                        "traffic_density": density[local_index],
                        "weather": weather[local_index],
                        "renderer_quality_level": renderer["primary_level"],
                        "renderer_server_launch_flag": renderer[
                            "required_server_launch_flag"
                        ],
                        "renderer_contract_role": "primary",
                        "carla_seed": _stable_seed(master_seed, f"{group_id}:carla"),
                        "traffic_seed": _stable_seed(master_seed, f"{group_id}:traffic"),
                        "sensor_seed": _stable_seed(master_seed, f"{group_id}:sensor"),
                        "raw_retention_tier": _retention_tier(
                            split, audit=audit, config=config
                        ),
                        "raw_window_duration_s": raw_window_s if split != "test" else 0.0,
                        "raw_window_anchor": (
                            config["retention"]["raw_window_anchor"]
                            if split != "test"
                            else "none"
                        ),
                        "confirmatory_locked": int(split == "test"),
                    }
                )
            route_offset += group_count

    manifest = pd.DataFrame(rows)
    validate_manifest(manifest, config)
    return manifest


def validate_manifest(manifest: pd.DataFrame, config: Mapping[str, object]) -> None:
    if manifest.empty or manifest["trajectory_id"].duplicated().any():
        raise ValueError("manifest is empty or contains duplicate trajectory IDs")
    if set(manifest["suite_id"]) != {"A", "B"}:
        raise ValueError("manifest must contain Suite A and Suite B")
    if set(manifest["renderer_quality_level"].astype(str)) != {"Epic"}:
        raise ValueError("every primary Suite A/B row must declare Epic rendering")
    if set(manifest["renderer_server_launch_flag"].astype(str)) != {
        "-quality-level=Epic"
    }:
        raise ValueError("every primary Suite A/B row must declare the exact Epic flag")
    if set(manifest["renderer_contract_role"].astype(str)) != {"primary"}:
        raise ValueError("Suite A/B design rows must remain primary-renderer rows")
    labels = dict(manifest.groupby("suite_id")["suite_label"].first())
    if labels != {"A": "designed_decision_opportunities", "B": "naturalistic_operation"}:
        raise ValueError(f"Suite A/B labels drifted: {labels}")
    pilot_ids = set(str(value) for value in config["pilot_exclusion_group_ids"])
    if pilot_ids & set(manifest["group_id"].astype(str)):
        raise ValueError("excluded pilot group entered the scientific manifest")

    group_splits = manifest.groupby("group_id")["split"].nunique()
    if int(group_splits.max()) != 1:
        raise ValueError("a trajectory group crosses data splits")
    group_seeds = manifest.groupby("group_id")["carla_seed"].nunique()
    if int(group_seeds.max()) != 1:
        raise ValueError("matched group members do not share CARLA seed")
    distinct_group_seeds = manifest.groupby("group_id")["carla_seed"].first()
    if distinct_group_seeds.duplicated().any():
        raise ValueError("CARLA seed is reused across independent groups")

    suite_a = manifest[manifest["suite_id"] == "A"]
    pair_sizes = suite_a.groupby("group_id").size()
    if set(pair_sizes) != {2}:
        raise ValueError("every Suite A group must have positive and benign members")
    pair_roles = suite_a.groupby("group_id")["scenario_role"].agg(set)
    expected_roles = {"controlled_positive_occlusion", "matched_benign_negative"}
    if any(value != expected_roles for value in pair_roles):
        raise ValueError("Suite A positive/benign role pairing drifted")

    group_counts = (
        manifest.drop_duplicates("group_id").groupby(["suite_id", "split"]).size()
    )
    expected = {
        ("A", "calibration"): 24,
        ("A", "validation"): 24,
        ("A", "test"): 72,
        ("B", "calibration"): 18,
        ("B", "validation"): 18,
        ("B", "test"): 54,
    }
    if group_counts.to_dict() != expected:
        raise ValueError(f"group-count contract drifted: {group_counts.to_dict()}")

    suite_a_groups = suite_a.drop_duplicates("group_id")
    cell_columns = [
        "geometry_or_route_id",
        "closing_speed_band",
        "time_to_hazard_band",
    ]
    cell_counts = suite_a_groups.groupby(cell_columns + ["split"]).size().unstack(
        fill_value=0
    )
    if not all(
        tuple(int(row[split]) for split in SPLITS) == (1, 1, 3)
        for _, row in cell_counts.iterrows()
    ):
        raise ValueError("Suite A cells are not split 1/1/3")

    if (manifest.loc[manifest["split"] == "test", "raw_window_duration_s"] != 0).any():
        raise ValueError("confirmatory test rows must not retain heavy raw windows")


def build_power_sensitivity(config: Mapping[str, object], manifest: pd.DataFrame) -> pd.DataFrame:
    power = config["power"]
    planned_test = int(
        manifest[
            (manifest["suite_id"] == "A")
            & (manifest["split"] == "test")
            & (manifest["scenario_role"] == "controlled_positive_occlusion")
        ]["group_id"].nunique()
    )
    censor_rates = sorted(
        {0.0, float(power["planned_censor_fraction"]), 0.20}
    )
    rows = []
    for censor_fraction in censor_rates:
        effective = max(2, math.floor(planned_test * (1.0 - censor_fraction)))
        for paired_sd_s in power["paired_sd_sensitivity_s"]:
            rows.append(
                {
                    "planned_test_groups": planned_test,
                    "censor_fraction": censor_fraction,
                    "effective_numeric_pairs": effective,
                    "smallest_effect_s": float(power["smallest_effect_s"]),
                    "paired_sd_s": float(paired_sd_s),
                    "two_sided_alpha": float(power["two_sided_alpha"]),
                    "approximate_paired_t_power": paired_t_power(
                        sample_count=effective,
                        effect_s=float(power["smallest_effect_s"]),
                        paired_sd_s=float(paired_sd_s),
                        alpha=float(power["two_sided_alpha"]),
                    ),
                    "status": "sensitivity_only_not_pilot_estimated",
                }
            )
    return pd.DataFrame(rows)


def summarize(
    config: Mapping[str, object], manifest: pd.DataFrame, power_table: pd.DataFrame
) -> dict:
    retention = config["retention"]
    frames = int(
        round(
            float(retention["raw_window_duration_s"])
            * float(config["common"]["world_hz"])
        )
    )
    roles = len(config["common"]["roles"])
    estimated_bytes = 0
    for row in manifest.itertuples(index=False):
        estimated_bytes += int(retention["estimated_lightweight_bytes_per_trajectory"])
        if row.raw_retention_tier in {"inputs_only_window", "inputs_plus_logits_window"}:
            estimated_bytes += (
                frames
                * roles
                * int(retention["pilot_measured_role_input_bytes_per_frame"])
            )
        if row.raw_retention_tier == "inputs_plus_logits_window":
            estimated_bytes += (
                frames
                * roles
                * int(retention["pilot_measured_role_logits_bytes_per_frame"])
            )

    group_frame = manifest.drop_duplicates("group_id")
    trajectory_count = len(manifest)
    planned_censor = float(config["power"]["planned_censor_fraction"])
    planned_sd = 1.25
    planned_row = power_table[
        (power_table["censor_fraction"] == planned_censor)
        & (power_table["paired_sd_s"] == planned_sd)
    ].iloc[0]
    pending_statuses = sorted(
        set(
            manifest.loc[
                ~manifest["geometry_or_route_status"].astype(str).str.startswith(
                    "reviewed"
                ),
                "geometry_or_route_status",
            ].astype(str)
        )
    )
    runtime_minutes = trajectory_count * float(
        config["runtime_estimate"]["pilot_measured_minutes_per_trajectory"]
    )
    return {
        "schema": "scenesense.phase2_suite_design_summary.v1",
        "design_id": config["design_id"],
        "collection_authorized": False,
        "suite_labels": {
            "A": "designed_decision_opportunities",
            "B": "naturalistic_operation",
        },
        "renderer_contract": dict(config["common"]["renderer_quality"]),
        "independent_group_count": int(group_frame["group_id"].nunique()),
        "trajectory_count": trajectory_count,
        "group_counts": {
            f"{suite}_{split}": int(count)
            for (suite, split), count in group_frame.groupby(
                ["suite_id", "split"]
            ).size().items()
        },
        "trajectory_counts": {
            split: int(count)
            for split, count in manifest.groupby("split").size().items()
        },
        "suite_a_test_positive_group_count": int(
            manifest[
                (manifest["suite_id"] == "A")
                & (manifest["split"] == "test")
                & (manifest["scenario_role"] == "controlled_positive_occlusion")
            ]["group_id"].nunique()
        ),
        "power_status": "conditional_on_calibration_simulation_gate",
        "planned_censor_fraction": planned_censor,
        "sensitivity_power_at_sd_1_25_s": float(
            planned_row["approximate_paired_t_power"]
        ),
        "minimum_required_calibration_power": float(config["power"]["minimum_power"]),
        "estimated_storage_bytes": int(estimated_bytes),
        "design_raw_cap_bytes": int(retention["design_raw_cap_bytes"]),
        "storage_estimate_within_cap": estimated_bytes
        <= int(retention["design_raw_cap_bytes"]),
        "estimated_capture_hours": runtime_minutes / 60.0,
        "pending_manual_scenario_statuses": pending_statuses,
        "blocking_gates": [
            "author_and_visually_review_all_pending_geometry_and_route_families",
            "calibration_replay_sufficiency_capture",
            "calibration_simulation_power_at_least_0_80_for_all_registered_endpoints",
            "review_exact_local_and_oai_timestamp_byte_fields",
        ],
    }


def write_design(
    config_path: Path, output_dir: Path, *, overwrite: bool = False
) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("config root must be a mapping")
    manifest = build_manifest(config)
    power_table = build_power_sensitivity(config, manifest)
    summary = summarize(config, manifest, power_table)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"design output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=overwrite)
    manifest.to_csv(
        output_dir / "trajectory_group_manifest.csv",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    power_table.to_csv(output_dir / "power_sensitivity.csv", index=False)
    (output_dir / "design_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    module_path = Path(__file__).resolve()
    provenance = {
        "schema": "scenesense.phase2_suite_design_provenance.v1",
        "design_id": config["design_id"],
        "runtime_authorized": False,
        "config_sha256": _sha256(config_path),
        "config_semantic_sha256": _semantic_sha256(config),
        "module_sha256": _sha256(module_path),
        "deterministic_master_seed": int(config["master_seed"]),
    }
    (output_dir / "design_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = output_dir / "artifact_manifest.json"
    artifact_manifest = {
        "schema": "scenesense.phase2_suite_design_artifact_manifest.v1",
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path != manifest_path
        ],
    }
    manifest_path.write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repository_root
        / "phase2_map_sharing/configs/phase2_suite_ab_design_v1.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root
        / "phase2_map_sharing/design/phase2_suite_ab_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = write_design(
        args.config.resolve(), args.output_dir.resolve(), overwrite=args.overwrite
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
