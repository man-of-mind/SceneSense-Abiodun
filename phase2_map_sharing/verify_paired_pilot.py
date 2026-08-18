"""Fail-closed nine-gate verifier for a completed paired-causal pilot.

The verifier is intentionally performance-neutral: it checks whether C2 is
causal and computable, not whether the two-trajectory pilot shows a gain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from data_collection.phase2_causal_runtime import SourceLocalCausalTracker
from phase2_map_sharing.causal_contract import (
    CAUSAL_AUDIT_SCHEMA,
    CausalDecisionAudit,
    CausalField,
    DecisionRecord,
)
from phase2_map_sharing.schemas_v2 import (
    FORBIDDEN_RUNTIME_KEYS,
    PLACEMENT_ACTIONS,
    PUBLICATION_ACTIONS,
)


VERIFICATION_SCHEMA = "scenesense.phase2_paired_pilot_verification.v4"
ROLE_NAMES = ("helper", "recipient")
SCENARIO_ROLES = {
    "controlled_positive_occlusion",
    "matched_benign_negative",
}
TRACK_ID_PATTERN = re.compile(r"^(helper|recipient):track:[0-9]{6}$")
OUTPUT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class GateResult:
    gate: int
    name: str
    passed: bool
    evidence: Mapping[str, object]
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "gate": self.gate,
            "name": self.name,
            "pass": self.passed,
            "evidence": dict(self.evidence),
            "failures": list(self.failures),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_output_name(value: str, prefix: str) -> str:
    name = str(value).strip()
    if not OUTPUT_NAME_PATTERN.fullmatch(name) or not name.startswith(prefix):
        raise ValueError(
            f"output name must be one safe basename beginning with {prefix!r}: {value!r}"
        )
    return name


def _single(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} under {root}, found {len(matches)}")
    return matches[0]


def _role_dirs(batch_root: Path, config: Mapping[str, object]) -> list[tuple[dict, str, Path]]:
    rows = []
    for trajectory in config["trajectories"]:
        for role in ROLE_NAMES:
            rows.append(
                (
                    dict(trajectory),
                    role,
                    batch_root / str(trajectory["trajectory_id"]) / role,
                )
            )
    return rows


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _audit_record(envelope: Mapping[str, object]) -> CausalDecisionAudit:
    if envelope.get("schema") != CAUSAL_AUDIT_SCHEMA:
        raise ValueError("causal audit schema mismatch")
    payload = {
        "schema": envelope["schema"],
        "decision": envelope["decision"],
        "fields": envelope["fields"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if envelope.get("record_sha256") != digest:
        raise ValueError("causal audit record hash mismatch")
    decision = DecisionRecord(**dict(envelope["decision"]))
    fields = tuple(CausalField(**dict(item)) for item in envelope["fields"])
    audit = CausalDecisionAudit(decision=decision, fields=fields)
    audit.validate()
    return audit


def gate_causal_availability(
    batch_root: Path, config: Mapping[str, object], evaluation_name: str = "evaluation"
) -> GateResult:
    failures: list[str] = []
    counts: dict[str, int] = {}
    maximum_slack_s = 0.0
    for trajectory, role, role_dir in _role_dirs(batch_root, config):
        label = f"{trajectory['trajectory_id']}:{role}"
        path = role_dir / "runtime/causal_decisions.jsonl"
        try:
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
            audits = [_audit_record(json.loads(line)) for line in lines]
            if not audits:
                raise ValueError("causal audit is empty")
            stages = {audit.decision.decision_stage for audit in audits}
            if stages != {"placement", "publication"}:
                raise ValueError(f"decision stages incomplete: {sorted(stages)}")
            counts[label] = len(audits)
            for audit in audits:
                for field in audit.fields:
                    maximum_slack_s = max(
                        maximum_slack_s,
                        float(audit.decision.decision_at_s) - float(field.available_at_s),
                    )
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
    return GateResult(
        1,
        "causal_availability",
        not failures,
        {"audit_records_by_stream": counts, "maximum_nonnegative_slack_s": maximum_slack_s},
        tuple(failures),
    )


def _find_forbidden_keys(value: object, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text.lower() in FORBIDDEN_RUNTIME_KEYS:
                found.append(path)
            found.extend(_find_forbidden_keys(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_forbidden_keys(item, f"{prefix}[{index}]"))
    return found


def gate_representation(
    batch_root: Path, config: Mapping[str, object], evaluation_name: str = "evaluation"
) -> GateResult:
    failures: list[str] = []
    track_counts: dict[str, int] = {}
    for trajectory, role, role_dir in _role_dirs(batch_root, config):
        label = f"{trajectory['trajectory_id']}:{role}"
        runtime_dir = role_dir / "runtime"
        try:
            for path in sorted(runtime_dir.glob("*.json")):
                forbidden = _find_forbidden_keys(_load_json(path))
                if forbidden:
                    raise ValueError(f"forbidden runtime keys in {path.name}: {forbidden}")
            for path in sorted(runtime_dir.glob("*.jsonl")):
                for line_number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if not line:
                        continue
                    forbidden = _find_forbidden_keys(json.loads(line))
                    if forbidden:
                        raise ValueError(
                            f"forbidden runtime keys in {path.name}:{line_number}: {forbidden}"
                        )
            tracks = pd.read_csv(runtime_dir / "causal_tracks.csv")
            ids = tracks.get("source_track_id", pd.Series(dtype=str)).dropna().astype(str)
            invalid = sorted({track_id for track_id in ids if not TRACK_ID_PATTERN.fullmatch(track_id)})
            wrong_role = sorted({track_id for track_id in ids if not track_id.startswith(f"{role}:")})
            if invalid or wrong_role:
                raise ValueError(f"invalid source-local track IDs: {invalid or wrong_role}")
            for csv_path in runtime_dir.glob("*.csv"):
                with csv_path.open("r", encoding="utf-8", newline="") as stream:
                    fields = next(csv.reader(stream), [])
                forbidden_fields = sorted(
                    field for field in fields if field.lower() in FORBIDDEN_RUNTIME_KEYS
                )
                if forbidden_fields:
                    raise ValueError(
                        f"forbidden runtime columns in {csv_path.name}: {forbidden_fields}"
                    )
            track_counts[label] = len(ids)
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
    return GateResult(
        2,
        "source_local_representation",
        not failures,
        {"track_rows_by_stream": track_counts},
        tuple(failures),
    )


def gate_false_positive_preservation(
    batch_root: Path, config: Mapping[str, object], evaluation_name: str = "evaluation"
) -> GateResult:
    failures: list[str] = []
    inventory_summary: dict[str, dict] = {}
    for trajectory, role, role_dir in _role_dirs(batch_root, config):
        label = f"{trajectory['trajectory_id']}:{role}"
        try:
            inventory = pd.read_csv(role_dir / "runtime/raw_inference_inventory.csv")
            detections = pd.read_csv(role_dir / "runtime/final_detections.csv")
            metrics = pd.read_csv(_single(role_dir / "streams", "*_metrics.csv"))
            if inventory.empty or (inventory["object_heatmap_cells"] <= 0).any():
                raise ValueError("raw object candidate inventory is empty")
            if not (inventory["retained_logits"] == 1).all():
                raise ValueError("one or more raw logits artifacts was not retained")
            observed = detections.groupby("frame_id").size().to_dict()
            expected = metrics.set_index("frame_id")["object_count"].astype(int).to_dict()
            mismatches = {
                int(frame): (int(expected_count), int(observed.get(frame, 0)))
                for frame, expected_count in expected.items()
                if int(expected_count) != int(observed.get(frame, 0))
            }
            if mismatches:
                raise ValueError(
                    f"final candidate count mismatch for frames {list(mismatches)[:5]}"
                )
            inventory_summary[label] = {
                "raw_frames": len(inventory),
                "minimum_raw_heatmap_cells": int(inventory["object_heatmap_cells"].min()),
                "final_detection_rows": len(detections),
            }
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")

    try:
        tracker = SourceLocalCausalTracker("helper")
        tracks, associations = tracker.update(
            frame_id=999,
            timestamp_s=99.9,
            detections=[
                {
                    "class_name": "pedestrian",
                    "score": 0.051,
                    "world_x": 123.0,
                    "world_y": -77.0,
                    "world_z": 0.0,
                }
            ],
        )
        serialized = json.dumps({"tracks": tracks, "associations": associations})
        if associations[0]["association"] != "birth":
            raise ValueError("synthetic unmatched detection did not create a track birth")
        if not TRACK_ID_PATTERN.fullmatch(str(tracks[0]["source_track_id"])):
            raise ValueError("synthetic unmatched detection acquired an invalid identity")
        if _find_forbidden_keys(json.loads(serialized)):
            raise ValueError("synthetic unmatched path acquired evaluation identity")
        synthetic = {"pass": True, "source_track_id": tracks[0]["source_track_id"]}
    except Exception as exc:
        failures.append(f"synthetic_unmatched_injection: {type(exc).__name__}: {exc}")
        synthetic = {"pass": False}
    return GateResult(
        3,
        "false_positive_preservation",
        not failures,
        {"streams": inventory_summary, "synthetic_unmatched_injection": synthetic},
        tuple(failures),
    )


def _verify_artifact_manifest(role_dir: Path) -> int:
    manifest = _load_json(role_dir / "artifact_manifest.json")
    count = 0
    for item in manifest.get("files", []):
        path = role_dir / str(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"manifested artifact missing: {path}")
        if path.stat().st_size != int(item["bytes"]) or _sha256(path) != item["sha256"]:
            raise ValueError(f"artifact integrity mismatch: {path}")
        count += 1
    if count == 0:
        raise ValueError("artifact manifest is empty")
    return count


def gate_alignment_recoverability(
    batch_root: Path, config: Mapping[str, object], evaluation_name: str = "evaluation"
) -> GateResult:
    failures: list[str] = []
    sampled: dict[str, int] = {}
    manifest_counts: dict[str, int] = {}
    for trajectory in config["trajectories"]:
        trajectory_id = str(trajectory["trajectory_id"])
        frame_sets = []
        try:
            for role in ROLE_NAMES:
                role_dir = batch_root / trajectory_id / role
                inputs = {
                    int(path.stem.split("_")[1])
                    for path in (role_dir / "retained_inputs").glob("frame_*_inputs.npz")
                }
                logits = {
                    int(path.stem.split("_")[1])
                    for path in (role_dir / "retained_inputs").glob("frame_*_logits.npz")
                }
                truth = pd.read_csv(_single(role_dir / "evaluation_truth", "*_ground_truth.csv"))
                frame_sets.append(inputs & logits & set(truth["frame_id"].astype(int)))
                manifest_counts[f"{trajectory_id}:{role}"] = _verify_artifact_manifest(role_dir)
            common = set.intersection(*frame_sets)
            if not common:
                raise ValueError("no helper/recipient raw+logit+truth frame is jointly recoverable")
            sampled[trajectory_id] = min(common)
        except Exception as exc:
            failures.append(f"{trajectory_id}: {type(exc).__name__}: {exc}")
    chain_path = batch_root / evaluation_name / "capture_warning_truth_chain.json"
    chain_evidence: dict[str, object] = {}
    try:
        chain = _load_json(chain_path)
        required = {
            "capture", "inference", "tracking", "action", "transport",
            "map_install", "warning", "truth_score", "target_contract",
        }
        if not required.issubset(chain):
            raise ValueError(f"sample chain is missing stages: {sorted(required - set(chain))}")
        target_contract = chain["target_contract"]
        positive = next(
            item
            for item in config["trajectories"]
            if item["scenario_role"] == "controlled_positive_occlusion"
        )
        if str(target_contract.get("trajectory_id")) != str(positive["trajectory_id"]):
            raise ValueError("sample chain does not target the registered positive trajectory")
        if str(target_contract.get("target_truth_role_prefix")) != str(
            positive["target_truth_role_prefix"]
        ):
            raise ValueError("sample chain target role prefix differs from the registered target")

        warning = chain["warning"]
        status = str(warning.get("status"))
        event = warning.get("event")
        source_roles = {str(item) for item in chain["capture"].get("source_roles", [])}
        if not source_roles or not source_roles.issubset(set(ROLE_NAMES)):
            raise ValueError(f"invalid chained source roles: {sorted(source_roles)}")
        if status == "observed_registered_target":
            if not isinstance(event, Mapping):
                raise ValueError("observed target warning is missing its event")
            event_sources = set(json.loads(str(event.get("evidence_sources", "[]"))))
            if source_roles != event_sources:
                raise ValueError(
                    "chained source roles do not cover every warning evidence source"
                )

        artifact_paths: dict[str, object] = {}
        for stage in ("capture", "inference", "tracking", "action"):
            artifacts = chain[stage].get("artifacts_by_source", {})
            if set(artifacts) != source_roles:
                raise ValueError(f"{stage} artifacts do not cover every evidence source")
            resolved = {}
            for role, artifact in artifacts.items():
                relative = Path(str(artifact))
                path = (batch_root / relative).resolve()
                try:
                    path.relative_to(batch_root.resolve())
                except ValueError as exc:
                    raise ValueError(f"{stage}:{role} artifact escapes the batch root") from exc
                if not path.is_file():
                    raise FileNotFoundError(f"{stage}:{role} artifact is missing: {relative}")
                resolved[str(role)] = path
            artifact_paths[stage] = resolved
        for stage in ("transport", "truth_score"):
            relative = Path(str(chain[stage]["artifact"]))
            path = (batch_root / relative).resolve()
            try:
                path.relative_to(batch_root.resolve())
            except ValueError as exc:
                raise ValueError(f"{stage} artifact escapes the batch root") from exc
            if not path.is_file():
                raise FileNotFoundError(f"{stage} artifact is missing: {relative}")
            artifact_paths[stage] = path

        frame_id = int(chain["capture"]["frame_id"])
        for role in source_roles:
            if f"frame_{frame_id:08d}_inputs.npz" != artifact_paths["capture"][role].name:
                raise ValueError(f"{role} capture artifact does not match the chained frame")
            if f"frame_{frame_id:08d}_logits.npz" != artifact_paths["inference"][role].name:
                raise ValueError(f"{role} inference artifact does not match the chained frame")

        if status == "observed_registered_target":
            if int(event.get("target_hazard_match", 0)) != 1:
                raise ValueError("sample warning is not matched to the registered target")
            if str(event.get("trajectory_id")) != str(positive["trajectory_id"]):
                raise ValueError("sample warning belongs to a different trajectory")
            if int(event.get("frame_id", -1)) != frame_id:
                raise ValueError("sample warning and capture frame differ")
            truth = pd.read_csv(artifact_paths["truth_score"])
            truth_rows = truth[
                (truth["frame_id"].astype(int) == frame_id)
                & (
                    truth["actor_id"].astype(str)
                    == str(event.get("evaluation_truth_id"))
                )
            ]
            if truth_rows.empty or not truth_rows["role_name"].astype(str).str.startswith(
                str(positive["target_truth_role_prefix"])
            ).any():
                raise ValueError("sample warning truth ID is not the registered target at that frame")
        elif status == "missed_but_computable":
            if event is not None:
                raise ValueError("missed target chain unexpectedly contains a warning event")
        else:
            raise ValueError(f"unknown target-chain warning status: {status}")
        chain_evidence = {
            "status": status,
            "frame_id": frame_id,
            "target_specific": True,
            "evidence_sources_recovered": sorted(source_roles),
            "warning_arm": event.get("arm_id") if isinstance(event, Mapping) else None,
        }
    except Exception as exc:
        failures.append(f"sample_chain: {type(exc).__name__}: {exc}")
    return GateResult(
        4,
        "alignment_and_recoverability",
        not failures,
        {
            "sampled_frame_by_trajectory": sampled,
            "manifest_file_counts": manifest_counts,
            "target_specific_chain": chain_evidence,
        },
        tuple(failures),
    )


def gate_action_provenance(
    batch_root: Path, config: Mapping[str, object], evaluation_name: str = "evaluation"
) -> GateResult:
    failures: list[str] = []
    actions: dict[str, dict[str, list[str]]] = {}
    for trajectory, role, role_dir in _role_dirs(batch_root, config):
        label = f"{trajectory['trajectory_id']}:{role}"
        try:
            stage_actions = {"placement": set(), "publication": set()}
            for line in (role_dir / "runtime/causal_decisions.jsonl").read_text(
                encoding="utf-8"
            ).splitlines():
                if not line:
                    continue
                audit = _audit_record(json.loads(line))
                stage_actions[audit.decision.decision_stage].add(audit.decision.action)
                if any(field.source_stage == "shadow_inference" for field in audit.fields):
                    raise ValueError("shadow output influenced an action")
            if not stage_actions["placement"].issubset(PLACEMENT_ACTIONS):
                raise ValueError("publication action appeared in placement stage")
            if not stage_actions["publication"].issubset(PUBLICATION_ACTIONS):
                raise ValueError("placement action appeared in publication stage")
            manifest = _load_json(_single(role_dir / "manifests", "*_manifest.json"))
            phase2 = manifest["phase2_paired_causal"]
            if bool(phase2["warnings_actuated"]):
                raise ValueError("warning actuation was enabled")
            actions[label] = {
                stage: sorted(values) for stage, values in stage_actions.items()
            }
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
    return GateResult(
        5,
        "action_provenance",
        not failures,
        {"actions_by_stream": actions},
        tuple(failures),
    )


def gate_c2_computability(
    batch_root: Path, config: Mapping[str, object], evaluation_name: str = "evaluation"
) -> GateResult:
    failures: list[str] = []
    evaluation_dir = batch_root / evaluation_name
    required_columns = {
        "trajectory_id", "scenario_role", "arm_id", "first_warning_s",
        "warning_lead_s", "false_warning", "missed_hazard", "application_bytes",
        "on_wire_bytes", "capture_to_install_ms", "map_aoi_s", "evidence_provenance",
        "false_warning_definition", "capture_to_install_timing_status", "frame_count",
        "warning_event_count", "warning_frame_count", "warning_frame_rate",
        "target_warning_event_count", "target_warning_frame_count",
        "non_target_warning_event_count", "non_target_warning_frame_count",
        "non_target_warning_frame_rate", "unmatched_warning_event_count",
        "unmatched_warning_frame_count", "unmatched_warning_frame_rate",
        "false_warning_adjudication_status",
    }
    rows = 0
    diagnostics_summary: dict[str, object] = {}
    try:
        metrics = pd.read_csv(evaluation_dir / "arm_metrics.csv")
        missing = required_columns - set(metrics.columns)
        if missing:
            raise ValueError(f"arm metrics are missing columns: {sorted(missing)}")
        expected_pairs = {
            (str(item["trajectory_id"]), arm)
            for item in config["trajectories"]
            for arm in ("ego_only", "send_everything", "hazard_only")
        }
        observed_pairs = set(zip(metrics["trajectory_id"].astype(str), metrics["arm_id"].astype(str)))
        if expected_pairs != observed_pairs:
            raise ValueError("arm metrics do not contain exactly three arms per trajectory")
        for rate_column in (
            "warning_frame_rate",
            "non_target_warning_frame_rate",
            "unmatched_warning_frame_rate",
        ):
            rates = pd.to_numeric(metrics[rate_column])
            if rates.isna().any() or ((rates < 0.0) | (rates > 1.0)).any():
                raise ValueError(f"{rate_column} contains an invalid exposure rate")
        if set(metrics["false_warning_adjudication_status"].astype(str)) != {
            "provisional_non_target_proxy_not_hazard_adjudicated"
        }:
            raise ValueError("false-warning proxy is not explicitly marked provisional")
        if set(metrics["capture_to_install_timing_status"].astype(str)) != {
            "non_citable_shared_gpu_correctness_pilot"
        }:
            raise ValueError("pilot timing is not explicitly marked non-citable")
        rows = len(metrics)
        diagnostics = pd.read_csv(evaluation_dir / "warning_diagnostics.csv")
        diagnostic_pairs = set(
            zip(diagnostics["trajectory_id"].astype(str), diagnostics["arm_id"].astype(str))
        )
        if diagnostic_pairs != expected_pairs:
            raise ValueError("warning diagnostics do not contain every paired arm")
        fragmentation = pd.read_csv(evaluation_dir / "warning_fragmentation.csv")
        if fragmentation.empty:
            raise ValueError("warning-fragmentation diagnostics are empty")
        warning_events = pd.read_csv(evaluation_dir / "warning_events.csv")
        required_warning_columns = {
            "track_world_x",
            "track_world_y",
            "track_velocity_x",
            "track_velocity_y",
            "track_position_sigma_m",
            "evidence_track_ids",
            "evidence_scope",
        }
        missing_warning_columns = required_warning_columns - set(warning_events.columns)
        if missing_warning_columns:
            raise ValueError(
                "warning events cannot support independent future-truth matching: "
                f"{sorted(missing_warning_columns)}"
            )
        isolation = _load_json(evaluation_dir / "arm_state_manifest.json")
        if not bool(isolation.get("independent_state_per_arm")):
            raise ValueError("arm state isolation was not proven")
        if bool(isolation.get("shared_mutable_state_detected")):
            raise ValueError("counterfactual arms shared mutable state")
        provenance = _load_json(evaluation_dir / "analysis_provenance.json")
        if provenance.get("analysis_config", {}).get("semantic_sha256") != _semantic_sha256(config):
            raise ValueError("analysis provenance config hash does not match verifier config")
        if not bool(
            provenance.get("capture_resolved_config", {}).get(
                "semantic_match_to_analysis_config"
            )
        ):
            raise ValueError("analysis config does not match the captured resolved config")
        if not bool(
            provenance.get("detached_launch", {}).get(
                "analysis_config_file_matches_launch"
            )
        ):
            raise ValueError("analysis config file hash does not match the launch manifest")
        if provenance.get("detached_launch", {}).get("inference_timing_citable") is not False:
            raise ValueError("shared-GPU pilot timing provenance is not explicitly non-citable")

        artifact_manifest = _load_json(
            evaluation_dir / "evaluation_artifact_manifest.json"
        )
        manifested_names = set()
        for item in artifact_manifest.get("files", []):
            path = evaluation_dir / str(item["path"])
            if not path.is_file():
                raise FileNotFoundError(f"manifested evaluation artifact is missing: {path}")
            if path.stat().st_size != int(item["bytes"]) or _sha256(path) != item["sha256"]:
                raise ValueError(f"evaluation artifact integrity mismatch: {path.name}")
            manifested_names.add(path.name)
        required_artifacts = {
            "arm_metrics.csv",
            "warning_events.csv",
            "warning_diagnostics.csv",
            "warning_fragmentation.csv",
            "capture_warning_truth_chain.json",
            "analysis_provenance.json",
            "replay_summary.json",
        }
        if not required_artifacts.issubset(manifested_names):
            raise ValueError(
                f"evaluation manifest is missing artifacts: {sorted(required_artifacts - manifested_names)}"
            )
        benign = diagnostics[
            diagnostics["scenario_role"].astype(str) == "matched_benign_negative"
        ]
        diagnostics_summary = {
            "warning_diagnostic_rows": len(diagnostics),
            "fragmentation_diagnostic_rows": len(fragmentation),
            "benign_warning_frame_rate_by_arm": {
                str(row["arm_id"]): float(row["warning_frame_rate"])
                for _, row in benign.iterrows()
            },
            "false_warning_performance_evaluated_as_gate": False,
            "timing_citable": False,
        }
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    return GateResult(
        6,
        "c2_computability",
        not failures,
        {
            "arm_metric_rows": rows,
            "performance_gain_evaluated_as_gate": False,
            **diagnostics_summary,
        },
        tuple(failures),
    )


def gate_paired_semantics(
    batch_root: Path, config: Mapping[str, object], evaluation_name: str = "evaluation"
) -> GateResult:
    failures: list[str] = []
    positive = next(
        item for item in config["trajectories"] if item["scenario_role"] == "controlled_positive_occlusion"
    )
    benign = next(
        item for item in config["trajectories"] if item["scenario_role"] == "matched_benign_negative"
    )
    matched_fields = ("seed", "population_family", "matched_pair_id")
    for field in matched_fields:
        if positive[field] != benign[field]:
            failures.append(f"positive/benign mismatch in {field}")
    try:
        pairing = _load_json(batch_root / evaluation_name / "paired_semantics.json")
        if set(pairing.get("declared_arm_differences", [])) != {
            "publication_selection"
        }:
            raise ValueError("counterfactual arm differences are not isolated to publication")
        if bool(pairing.get("hidden_world_state_divergence")):
            raise ValueError("hidden world-state divergence was treated as a policy effect")
    except Exception as exc:
        failures.append(f"paired_semantics: {type(exc).__name__}: {exc}")
    return GateResult(
        7,
        "paired_semantics_and_arm_isolation",
        not failures,
        {"matched_fields": list(matched_fields)},
        tuple(failures),
    )


def gate_integrity(
    batch_root: Path, config: Mapping[str, object], evaluation_name: str = "evaluation"
) -> GateResult:
    failures: list[str] = []
    statuses: dict[str, str] = {}
    for trajectory, role, role_dir in _role_dirs(batch_root, config):
        label = f"{trajectory['trajectory_id']}:{role}"
        try:
            summary = _load_json(role_dir / "phase2_runtime_summary.json")
            statuses[label] = str(summary.get("status"))
            if summary.get("status") != "complete":
                raise ValueError(f"runtime status is {summary.get('status')}")
            if int(summary.get("raw_input_files_written", 0)) == 0 or int(
                summary.get("logits_files_written", 0)
            ) == 0:
                raise ValueError("required retained streams are empty")
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
    try:
        manifest = _load_json(batch_root / "batch_manifest.json")
        if manifest.get("status") != "complete":
            raise ValueError("batch manifest is not complete")
        for trajectory in manifest.get("trajectories", []):
            integrity = trajectory.get("integrity", {})
            if int(integrity.get("unintended_collision_count", -1)) != 0:
                raise ValueError("unintended collision occurred")
            if bool(integrity.get("persistent_gridlock")):
                raise ValueError("persistent gridlock occurred")
            if not bool(integrity.get("actor_cleanup_complete")):
                raise ValueError("actor cleanup did not complete")
            if bool(integrity.get("dropped_required_stream")):
                raise ValueError("required stream was dropped")
    except Exception as exc:
        failures.append(f"batch_manifest: {type(exc).__name__}: {exc}")
    return GateResult(
        8,
        "integrity",
        not failures,
        {"runtime_status_by_stream": statuses},
        tuple(failures),
    )


def gate_sensor_contract(
    batch_root: Path, config: Mapping[str, object], evaluation_name: str = "evaluation"
) -> GateResult:
    failures: list[str] = []
    medians: dict[str, float] = {}
    reference = float(config["verification"]["radar_density_reference_projected_median"])
    tolerance = float(config["verification"]["radar_density_relative_tolerance"])
    lower, upper = reference * (1.0 - tolerance), reference * (1.0 + tolerance)
    for trajectory, role, role_dir in _role_dirs(batch_root, config):
        label = f"{trajectory['trajectory_id']}:{role}"
        try:
            manifest = _load_json(_single(role_dir / "manifests", "*_manifest.json"))
            camera, radar, clock = manifest["camera"], manifest["radar"], manifest["clock_contract"]
            observed = (
                float(clock["world_control_hz"]),
                float(clock["sensor_detection_hz"]),
                int(camera["width"]),
                int(camera["height"]),
                float(camera["fov"]),
                int(radar["points_per_second"]),
                int(radar["raster_radius_px"]),
                int(radar["temporal_window_frames"]),
            )
            expected = (10.0, 10.0, 1280, 720, 120.0, 200000, 4, 2)
            if observed != expected:
                raise ValueError(f"sensor contract mismatch: {observed}")
            metrics = pd.read_csv(_single(role_dir / "streams", "*_metrics.csv"))
            median = float(pd.to_numeric(metrics["radar_projected_points"]).median())
            medians[label] = median
            if not lower <= median <= upper:
                raise ValueError(
                    f"radar projected-points median {median:.1f} outside [{lower:.1f}, {upper:.1f}]"
                )
            timestamps = pd.to_numeric(metrics["carla_timestamp"]).sort_values()
            if len(timestamps) < 2 or not np.allclose(
                np.diff(timestamps), 0.1, rtol=0.0, atol=0.005
            ):
                raise ValueError("sensor timestamps are not a stable 10 Hz sequence")
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
    return GateResult(
        9,
        "sensor_contract_and_radar_density",
        not failures,
        {"radar_projected_points_median_by_stream": medians, "accepted_band": [lower, upper]},
        tuple(failures),
    )


GATES: tuple[Callable[[Path, Mapping[str, object], str], GateResult], ...] = (
    gate_causal_availability,
    gate_representation,
    gate_false_positive_preservation,
    gate_alignment_recoverability,
    gate_action_provenance,
    gate_c2_computability,
    gate_paired_semantics,
    gate_integrity,
    gate_sensor_contract,
)


def verify(
    batch_root: Path,
    config: Mapping[str, object],
    *,
    evaluation_name: str = "evaluation",
) -> dict:
    evaluation_name = _validated_output_name(evaluation_name, "evaluation")
    results: list[GateResult] = []
    for gate in GATES:
        result = gate(batch_root, config, evaluation_name)
        results.append(result)
        if not result.passed:
            break
    passed = len(results) == len(GATES) and all(result.passed for result in results)
    return {
        "schema": VERIFICATION_SCHEMA,
        "verdict": "PASS" if passed else "FAIL_HOLD",
        "performance_gain_is_a_gate": False,
        "evaluation_namespace": evaluation_name,
        "first_failed_gate": next(
            (result.gate for result in results if not result.passed), None
        ),
        "gates": [result.to_dict() for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "data_collection/configs/phase2_paired_causal_pilot_integration_v1.yaml"
        ),
    )
    parser.add_argument(
        "--evaluation-name",
        default="evaluation",
        help="existing replay output directory basename (for example evaluation_v2)",
    )
    parser.add_argument(
        "--verification-name",
        default="verification",
        help="create-only verification output directory basename (for example verification_v2)",
    )
    args = parser.parse_args()
    batch_root = args.batch_root.resolve()
    with args.config.resolve().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, Mapping):
        raise ValueError("integration config root must be a mapping")
    evaluation_name = _validated_output_name(args.evaluation_name, "evaluation")
    verification_name = _validated_output_name(args.verification_name, "verification")
    result = verify(batch_root, config, evaluation_name=evaluation_name)
    verification_dir = batch_root / verification_name
    verification_dir.mkdir(parents=True, exist_ok=True)
    output_path = verification_dir / "pilot_verification.json"
    if output_path.exists():
        raise FileExistsError(f"verification output already exists: {output_path}")
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result["verdict"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
