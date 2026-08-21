"""Freeze the UE-A4 72-action technical certification registry.

This is an evidence join, not a runtime or experiment runner.  It preserves
the immutable UE-A1 operational action identity and promotes each row only
when the authoritative UE-A2 ``_02`` evidence proves the exact action passed
the strict model, wire, tail, map-schema, and localhost-UDP smoke.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent.ue_split_profile_registry import (  # noqa: E402
    REGISTRY_FIELDS as A1_REGISTRY_FIELDS,
    registry_row_fingerprint as a1_row_fingerprint,
    validate_registry_bundle as validate_a1_bundle,
)
from rl_agent.ue_split_wire_contract import (  # noqa: E402
    action_contract_sha256,
    build_launch_binding,
    load_registered_profiles,
)


DEFAULT_CONFIG = ROOT / "rl_agent/configs/ue_split_technical_registry_v1.json"
CONFIG_SCHEMA = "scenesense.ue_split_technical_registry_config.v1"
REGISTRY_SCHEMA = "scenesense.ue_split_technical_registry.v1"
MANIFEST_SCHEMA = "scenesense.ue_split_technical_registry_manifest.v1"
TERMINAL_SCHEMA = "scenesense.ue_split_technical_registry_decision.v1"

A1_COPY_FIELDS = tuple(
    field for field in A1_REGISTRY_FIELDS if field != "row_fingerprint_sha256"
)
A2_GATE_FIELDS = (
    "registry_binding_status",
    "fixture_status",
    "production_codec_status",
    "in_memory_wire_status",
    "wire_shape_status",
    "map_schema_status",
    "model_front_status",
    "model_edge_status",
    "roi_execution_status",
    "ae_execution_status",
    "tail_execution_status",
    "actual_udp_status",
)
TECHNICAL_REGISTRY_FIELDS = (
    "technical_registry_schema",
    "technical_registry_id",
    "certification_status",
    "source_registry_sha256",
    "source_row_fingerprint_sha256",
    "action_contract_sha256",
    "a2_bundle_name",
    "a2_manifest_sha256",
    "a2_profile_table_sha256",
    "a2_profile_row_sha256",
    "a2_launch_binding_sha256",
    "a2_model_smoke_sha256",
    "a2_negative_tests_sha256",
    "a2_transport_preflight_sha256",
    "a2_gate_statuses_json",
    "a2_fixture_zstd_bytes",
    "a2_fixture_udp_chunks",
    "a2_fixture_compressed_sha256",
    "a1_declared_runtime_path",
    "a1_declared_runtime_sha256",
    "a1_declared_front_profile_launch_args_json",
    "a1_declared_edge_profile_launch_args_json",
    "certified_runtime_path",
    "certified_runtime_sha256",
    "certified_launcher_path",
    "certified_launcher_sha256",
    *A1_COPY_FIELDS,
    "technical_row_fingerprint_sha256",
)


class TechnicalRegistryError(RuntimeError):
    """Raised when UE-A4 evidence cannot be joined and frozen exactly."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_scalar(value: Any) -> str:
    return "" if value is None else str(value)


def technical_row_fingerprint(row: Mapping[str, Any]) -> str:
    source = {
        field: _csv_scalar(row[field])
        for field in TECHNICAL_REGISTRY_FIELDS
        if field != "technical_row_fingerprint_sha256"
    }
    return hashlib.sha256(canonical_json_bytes(source)).hexdigest()


def _a2_row_fingerprint(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json_bytes({key: _csv_scalar(value) for key, value in row.items()})
    ).hexdigest()


def _canonical_decimal(value: Any, label: str) -> str:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise TechnicalRegistryError(f"invalid {label}: {value!r}") from exc
    if not parsed.is_finite():
        raise TechnicalRegistryError(f"non-finite {label}: {value!r}")
    if parsed == 0:
        return "0"
    rendered = format(parsed.normalize(), "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _repo_path(relative: str) -> Path:
    path = (ROOT / str(relative)).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise TechnicalRegistryError(f"path escapes repository: {relative}") from exc
    return path


def _pinned_file(path: Path, expected_sha256: str, label: str) -> Path:
    if not path.is_file():
        raise TechnicalRegistryError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != str(expected_sha256):
        raise TechnicalRegistryError(
            f"{label} hash drift: expected={expected_sha256} actual={actual}"
        )
    return path


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TechnicalRegistryError(f"{label} must be a mapping")
    return value


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise TechnicalRegistryError("invalid UE-A4 config schema")
    if config.get("registry_schema") != REGISTRY_SCHEMA:
        raise TechnicalRegistryError("invalid UE-A4 technical registry schema")
    if config.get("registry_id") != "ue_split_technical_registry_v1":
        raise TechnicalRegistryError("invalid UE-A4 technical registry ID")
    root = (path.parent / str(config.get("repository_root", ""))).resolve()
    if root != ROOT:
        raise TechnicalRegistryError(f"repository root mismatch: {root} != {ROOT}")
    expected_authority = {
        "evidence_only_successor": True,
        "runtime_retarget_authorized": False,
        "quality_filter_authorized": False,
        "carla_run_authorized": False,
        "oai_run_authorized": False,
        "policy_training_authorized": False,
    }
    if dict(_require_mapping(config.get("authority"), "authority")) != expected_authority:
        raise TechnicalRegistryError("UE-A4 authority must be evidence-only")

    a1 = _require_mapping(config.get("a1"), "a1")
    a2 = _require_mapping(config.get("a2"), "a2")
    if int(a1.get("profiles", -1)) != 72 or int(a2.get("profiles", -1)) != 72:
        raise TechnicalRegistryError("A1 and A2 profile counts must both be 72")
    if a2.get("required_bundle_name") != "20260820_cuda_model_smoke_02":
        raise TechnicalRegistryError("only the authoritative UE-A2 _02 bundle is accepted")
    if a2.get("superseded_bundle_name") != "20260820_cuda_model_smoke_01":
        raise TechnicalRegistryError("superseded UE-A2 _01 bundle must be explicit")
    if Path(str(a2.get("bundle_dir", ""))).name != a2["required_bundle_name"]:
        raise TechnicalRegistryError("UE-A2 bundle path is not the required _02 bundle")
    if int(a2.get("negative_tests", -1)) != 34:
        raise TechnicalRegistryError("UE-A2 negative-test count must be 34")

    decision = _require_mapping(config.get("transport_decision"), "transport_decision")
    if dict(_require_mapping(decision.get("ue_to_edge"), "ue_to_edge")) != {
        "payload": "quantized_intermediate_features",
        "coder": "zstd",
        "level": 3,
    }:
        raise TechnicalRegistryError("UE-to-edge feature wire must remain zstd level 3")
    if dict(_require_mapping(decision.get("edge_to_map"), "edge_to_map")) != {
        "payload": "fusion_object_spatial_map.v1_json",
        "coder": "zlib",
        "level": 1,
        "runtime_literal": "zlib.compress(encoded, level=1)",
    }:
        raise TechnicalRegistryError("edge-to-map JSON must remain zlib level 1")
    return config


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _verify_manifest_outputs(bundle: Path, manifest: Mapping[str, Any]) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise TechnicalRegistryError(f"manifest has no output seals: {bundle}")
    for record in outputs:
        item = _require_mapping(record, "manifest output")
        path = (bundle / str(item["path"])).resolve()
        try:
            path.relative_to(bundle.resolve())
        except ValueError as exc:
            raise TechnicalRegistryError(f"manifest output escapes bundle: {path}") from exc
        _pinned_file(path, str(item["sha256"]), f"sealed output {path.name}")
        if "bytes" in item and path.stat().st_size != int(item["bytes"]):
            raise TechnicalRegistryError(f"sealed output byte count changed: {path.name}")
        if "rows" in item and len(_read_csv(path)) != int(item["rows"]):
            raise TechnicalRegistryError(f"sealed output row count changed: {path.name}")


def _load_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    a1 = _require_mapping(config["a1"], "a1")
    a2 = _require_mapping(config["a2"], "a2")
    a1_bundle = _repo_path(str(a1["bundle_dir"]))
    a2_bundle = _repo_path(str(a2["bundle_dir"]))

    a1_registry = _pinned_file(
        a1_bundle / str(a1["registry_csv"]), str(a1["registry_sha256"]), "A1 registry"
    )
    a1_manifest_path = _pinned_file(
        a1_bundle / str(a1["manifest_json"]), str(a1["manifest_sha256"]), "A1 manifest"
    )
    _pinned_file(
        a1_bundle / str(a1["terminal_json"]), str(a1["terminal_sha256"]), "A1 terminal"
    )
    validate_a1_bundle(a1_bundle)
    a1_manifest = json.loads(a1_manifest_path.read_text(encoding="utf-8"))
    _verify_manifest_outputs(a1_bundle, a1_manifest)

    a2_manifest_path = _pinned_file(
        a2_bundle / str(a2["manifest_json"]), str(a2["manifest_sha256"]), "A2 manifest"
    )
    a2_manifest = json.loads(a2_manifest_path.read_text(encoding="utf-8"))
    _verify_manifest_outputs(a2_bundle, a2_manifest)
    required_a2 = {
        "terminal": ("terminal_json", "terminal_sha256"),
        "profile table": ("profile_csv", "profile_csv_sha256"),
        "model smoke": ("model_smoke_json", "model_smoke_sha256"),
        "negative tests": ("negative_tests_json", "negative_tests_sha256"),
        "transport": ("transport_json", "transport_sha256"),
    }
    pinned_a2: dict[str, Path] = {}
    for label, (path_key, sha_key) in required_a2.items():
        pinned_a2[label] = _pinned_file(
            a2_bundle / str(a2[path_key]), str(a2[sha_key]), f"A2 {label}"
        )

    # UE-A2 sealed these exact sources.  A4 verifies they have not drifted but
    # does not retarget the runtime to the successor evidence CSV.
    source_inputs: list[dict[str, Any]] = []
    for label, spec_value in _require_mapping(config["sources"], "sources").items():
        spec = _require_mapping(spec_value, f"sources.{label}")
        path = _pinned_file(_repo_path(str(spec["path"])), str(spec["sha256"]), label)
        source_inputs.append(
            {"kind": "source", "label": label, "path": str(spec["path"]),
             "sha256": sha256_file(path), "bytes": path.stat().st_size}
        )
    runtime_path = _repo_path(str(config["sources"]["runtime_v2"]["path"]))
    runtime_source = runtime_path.read_text(encoding="utf-8")
    if str(config["transport_decision"]["edge_to_map"]["runtime_literal"]) not in runtime_source:
        raise TechnicalRegistryError("pinned runtime lost the zlib-1 edge-to-map binding")

    registered_profiles = load_registered_profiles(
        a1_registry, expected_registry_sha256=str(a1["registry_sha256"])
    )
    certified_launch_bindings: dict[str, dict[str, Any]] = {}
    certified_launch_digests: dict[str, str] = {}
    for profile in registered_profiles:
        binding = build_launch_binding(profile)
        certified_launch_bindings[profile.profile_id] = binding
        certified_launch_digests[profile.profile_id] = hashlib.sha256(
            canonical_json_bytes(binding)
        ).hexdigest()
    recorded_launch_digests = _require_mapping(
        _require_mapping(a2_manifest.get("registry_audit"), "A2 registry audit").get(
            "launch_binding_sha256"
        ),
        "A2 launch-binding seals",
    )
    if dict(recorded_launch_digests) != certified_launch_digests:
        raise TechnicalRegistryError("A2 launch-binding seals do not match current A1 bindings")

    # Check current UE-A2 input seals.  The manifest environment booleans are
    # a pre-run snapshot and are intentionally not completion gates.
    for record_value in a2_manifest.get("inputs", []):
        record = _require_mapping(record_value, "A2 input")
        path = Path(str(record["path"])).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError as exc:
            raise TechnicalRegistryError(f"A2 input escapes repository: {path}") from exc
        _pinned_file(path, str(record["sha256"]), f"A2 input {record.get('label', path.name)}")

    return {
        "a1_bundle": a1_bundle,
        "a2_bundle": a2_bundle,
        "a1_registry": a1_registry,
        "a1_manifest": a1_manifest,
        "a2_manifest": a2_manifest,
        "a2_terminal": json.loads(pinned_a2["terminal"].read_text(encoding="utf-8")),
        "a2_profiles_path": pinned_a2["profile table"],
        "a2_model": json.loads(pinned_a2["model smoke"].read_text(encoding="utf-8")),
        "a2_negatives": json.loads(pinned_a2["negative tests"].read_text(encoding="utf-8")),
        "a2_transport": json.loads(pinned_a2["transport"].read_text(encoding="utf-8")),
        "certified_launch_bindings": certified_launch_bindings,
        "certified_launch_digests": certified_launch_digests,
        "source_inputs": source_inputs,
    }


def _validate_a2_bundle(config: Mapping[str, Any], loaded: Mapping[str, Any]) -> None:
    terminal = _require_mapping(loaded["a2_terminal"], "A2 terminal")
    if terminal.get("schema") != "scenesense.ue_a2_technical_smoke_terminal.v1":
        raise TechnicalRegistryError("A2 terminal schema mismatch")
    expected_terminal = {
        "status": "PASSED",
        "profile_rows": 72,
        "technical_valid_profiles": 72,
        "technical_invalid_profiles": 0,
        "infrastructure_blocked_profiles": 0,
        "quality_gate_applied": False,
        "successor_registry_emitted": False,
        "model_inference_executed": True,
        "actual_udp_executed": True,
        "carla_run": False,
        "oai_run": False,
    }
    for key, value in expected_terminal.items():
        if terminal.get(key) != value:
            raise TechnicalRegistryError(f"A2 terminal gate failed: {key}={terminal.get(key)!r}")

    manifest = _require_mapping(loaded["a2_manifest"], "A2 manifest")
    if (
        manifest.get("schema") != "scenesense.ue_a2_technical_smoke_manifest.v1"
        or manifest.get("terminal_path") != config["a2"]["terminal_json"]
        or manifest.get("terminal_status") != "PASSED"
        or manifest.get("successor_registry_emitted") is not False
    ):
        raise TechnicalRegistryError("A2 manifest terminal/successor gate failed")
    audit = _require_mapping(manifest.get("registry_audit"), "A2 registry audit")
    expected_audit = {
        "profiles": 72,
        "unique_action_contracts": 72,
        "quality_mask_applied": False,
        "status": "PASS",
    }
    for key, value in expected_audit.items():
        if audit.get(key) != value:
            raise TechnicalRegistryError(f"A2 registry audit gate failed: {key}")

    model = _require_mapping(loaded["a2_model"], "A2 model smoke")
    if model.get("schema") != "scenesense.ue_a2_cuda_model_smoke.v1":
        raise TechnicalRegistryError("A2 model-smoke schema mismatch")
    model_gates = {
        "status": "PASS",
        "strict_front_loads": 4,
        "strict_edge_loads": 4,
        "native_backbone_encodes": 4,
        "roi_ae_paths": 24,
        "tail_decodes": 72,
        "technical_valid_profiles": 72,
        "technical_invalid_profiles": 0,
        "actual_udp_executed": True,
        "model_inference_executed": True,
        "source_drift": {},
    }
    for key, value in model_gates.items():
        if model.get(key) != value:
            raise TechnicalRegistryError(f"A2 model-smoke gate failed: {key}")
    if model.get("source_seals_start") != model.get("source_seals_end"):
        raise TechnicalRegistryError("A2 model-smoke source seals changed")

    negatives = _require_mapping(loaded["a2_negatives"], "A2 negatives")
    if negatives.get("schema") != "scenesense.ue_a2_negative_contract_tests.v1":
        raise TechnicalRegistryError("A2 negative-test schema mismatch")
    if (
        negatives.get("status") != "PASS"
        or negatives.get("tests") != int(config["a2"]["negative_tests"])
        or len(negatives.get("records", [])) != int(config["a2"]["negative_tests"])
        or negatives.get("decode_or_map_calls_after_rejection") != 0
        or negatives.get("temporary_registry_copies_only") is not True
        or negatives.get("source_registry_sha256_before") != config["a1"]["registry_sha256"]
        or negatives.get("source_registry_sha256_after") != config["a1"]["registry_sha256"]
    ):
        raise TechnicalRegistryError("A2 negative-contract gate failed")

    transport = _require_mapping(loaded["a2_transport"], "A2 transport")
    if transport.get("schema") != "scenesense.ue_a2_transport_evidence.v1":
        raise TechnicalRegistryError("A2 transport schema mismatch")
    actual_udp = _require_mapping(transport.get("actual_udp"), "A2 actual UDP")
    if (
        actual_udp.get("status") != "PASS"
        or actual_udp.get("actual_udp_executed") is not True
        or actual_udp.get("messages_sent") != 72
        or actual_udp.get("messages_received") != 72
    ):
        raise TechnicalRegistryError("A2 actual-UDP gate failed")


def _expected_a2_gate_values(row: Mapping[str, str]) -> dict[str, str]:
    return {
        "registry_binding_status": "PASS",
        "fixture_status": "PASS",
        "production_codec_status": "PASS_MODEL_FEATURES",
        "in_memory_wire_status": "PASS",
        "wire_shape_status": "PASS",
        "map_schema_status": "PASS",
        "model_front_status": "PASS_STRICT",
        "model_edge_status": "PASS_STRICT",
        "roi_execution_status": (
            "PASS_NO_DROP_VERIFIED"
            if _canonical_decimal(row["roi_drop_fraction"], "roi_drop_fraction") == "0"
            else "PASS_RANK_DROP_VERIFIED"
        ),
        "ae_execution_status": (
            "PASS_NO_AE" if row["model_family"] == "noae" else "PASS_INTEGRATED_AE"
        ),
        "tail_execution_status": "PASS_FINITE",
        "actual_udp_status": "PASS",
    }


def join_and_promote(
    config: Mapping[str, Any],
    a1_rows: Sequence[Mapping[str, str]],
    a2_rows: Sequence[Mapping[str, str]],
    certified_launch_bindings: Mapping[str, Mapping[str, Any]],
    certified_launch_digests: Mapping[str, str],
) -> list[dict[str, Any]]:
    if len(a1_rows) != 72 or len(a2_rows) != 72:
        raise TechnicalRegistryError("A1 and A2 tables must each contain exactly 72 rows")
    if len({row["profile_id"] for row in a1_rows}) != 72:
        raise TechnicalRegistryError("A1 profile IDs are not unique")
    if len({row["profile_id"] for row in a2_rows}) != 72:
        raise TechnicalRegistryError("A2 profile IDs are not unique")
    if {int(row["action_index"]) for row in a1_rows} != set(range(72)):
        raise TechnicalRegistryError("A1 action-index set must be exactly 0..71")
    if {int(row["action_index"]) for row in a2_rows} != set(range(72)):
        raise TechnicalRegistryError("A2 action-index set must be exactly 0..71")
    a2_by_id = {row["profile_id"]: row for row in a2_rows}
    profile_ids = {row["profile_id"] for row in a1_rows}
    if set(a2_by_id) != profile_ids:
        raise TechnicalRegistryError("A1/A2 profile-ID sets differ")
    if set(certified_launch_bindings) != profile_ids or set(certified_launch_digests) != profile_ids:
        raise TechnicalRegistryError("A2 launch-binding profile set differs from A1")

    output: list[dict[str, Any]] = []
    for a1_row_value in sorted(a1_rows, key=lambda row: int(row["action_index"])):
        a1_row = dict(a1_row_value)
        profile_id = a1_row["profile_id"]
        a2_row = dict(a2_by_id[profile_id])
        if a1_row.get("row_fingerprint_sha256") != a1_row_fingerprint(a1_row):
            raise TechnicalRegistryError(f"A1 row fingerprint failed: {profile_id}")
        if a1_row.get("quality_mask_applied") != "False":
            raise TechnicalRegistryError(f"A1 quality mask present: {profile_id}")
        if a1_row.get("technical_validity_status") != "REGISTERED_PENDING_SMOKE":
            raise TechnicalRegistryError(f"A1 row is not pending smoke: {profile_id}")
        if a1_row.get("entropy_coder") != "zstd" or a1_row.get("entropy_level") != "3":
            raise TechnicalRegistryError(f"A1 feature codec is not zstd-3: {profile_id}")

        join_fields = (
            "action_index",
            "profile_id",
            "model_family",
            "quantization_mode",
            "quantization_bits",
            "checkpoint_sha256",
        )
        for field in join_fields:
            if a1_row[field] != a2_row[field]:
                raise TechnicalRegistryError(f"A1/A2 {field} mismatch: {profile_id}")
        if _canonical_decimal(a1_row["roi_drop_fraction"], "A1 q") != _canonical_decimal(
            a2_row["roi_drop_fraction"], "A2 q"
        ):
            raise TechnicalRegistryError(f"A1/A2 roi_drop_fraction mismatch: {profile_id}")
        contract_sha = action_contract_sha256(a1_row)
        if a2_row.get("action_contract_sha256") != contract_sha:
            raise TechnicalRegistryError(f"A1/A2 action-contract mismatch: {profile_id}")
        expected_gates = _expected_a2_gate_values(a2_row)
        for field, expected in expected_gates.items():
            if a2_row.get(field) != expected:
                raise TechnicalRegistryError(
                    f"A2 gate failed for {profile_id}: {field}={a2_row.get(field)!r}"
                )
        if a2_row.get("technical_validity_status") != "TECHNICALLY_VALID":
            raise TechnicalRegistryError(f"A2 row is not technically valid: {profile_id}")
        if a2_row.get("blocking_codes"):
            raise TechnicalRegistryError(f"A2 row has blocking codes: {profile_id}")

        binding = _require_mapping(
            certified_launch_bindings[profile_id], f"launch binding {profile_id}"
        )
        binding_digest = hashlib.sha256(canonical_json_bytes(binding)).hexdigest()
        if binding_digest != certified_launch_digests[profile_id]:
            raise TechnicalRegistryError(f"A2 launch-binding digest mismatch: {profile_id}")

        promoted = {field: a1_row[field] for field in A1_COPY_FIELDS}
        promoted.update(
            {
                "feature_shape_status": "PASS_OBSERVED_UE_A2",
                "edge_decoder_binding_status": "PASS_UE_A2_FIXED_PROFILE",
                "edge_launcher_propagation_status": "PASS_UE_A2_REGISTERED_BINDING",
                "wire_profile_identity_present": True,
                "wire_mismatch_rejection_present": True,
                "wire_smoke_status": "PASS_UE_A2",
                "technical_validity_status": "TECHNICALLY_VALID",
                "technical_invalid_reason": "",
                "runtime_path": config["sources"]["runtime_v2"]["path"],
                "runtime_sha256": config["sources"]["runtime_v2"]["sha256"],
                "front_profile_launch_args_json": json.dumps(
                    binding["front_args"], separators=(",", ":")
                ),
                "edge_profile_launch_args_json": json.dumps(
                    binding["edge_args"], separators=(",", ":")
                ),
            }
        )
        row: dict[str, Any] = {
            "technical_registry_schema": config["registry_schema"],
            "technical_registry_id": config["registry_id"],
            "certification_status": "TECHNICALLY_VALID_A1_IDENTITY_A2_SMOKE",
            "source_registry_sha256": config["a1"]["registry_sha256"],
            "source_row_fingerprint_sha256": a1_row["row_fingerprint_sha256"],
            "action_contract_sha256": contract_sha,
            "a2_bundle_name": config["a2"]["required_bundle_name"],
            "a2_manifest_sha256": config["a2"]["manifest_sha256"],
            "a2_profile_table_sha256": config["a2"]["profile_csv_sha256"],
            "a2_profile_row_sha256": _a2_row_fingerprint(a2_row),
            "a2_launch_binding_sha256": binding_digest,
            "a2_model_smoke_sha256": config["a2"]["model_smoke_sha256"],
            "a2_negative_tests_sha256": config["a2"]["negative_tests_sha256"],
            "a2_transport_preflight_sha256": config["a2"]["transport_sha256"],
            "a2_gate_statuses_json": json.dumps(expected_gates, sort_keys=True, separators=(",", ":")),
            "a2_fixture_zstd_bytes": a2_row["zstd_bytes"],
            "a2_fixture_udp_chunks": a2_row["udp_chunks"],
            "a2_fixture_compressed_sha256": a2_row["compressed_sha256"],
            "a1_declared_runtime_path": a1_row["runtime_path"],
            "a1_declared_runtime_sha256": a1_row["runtime_sha256"],
            "a1_declared_front_profile_launch_args_json": a1_row[
                "front_profile_launch_args_json"
            ],
            "a1_declared_edge_profile_launch_args_json": a1_row[
                "edge_profile_launch_args_json"
            ],
            "certified_runtime_path": config["sources"]["runtime_v2"]["path"],
            "certified_runtime_sha256": config["sources"]["runtime_v2"]["sha256"],
            "certified_launcher_path": config["sources"]["launcher_v2"]["path"],
            "certified_launcher_sha256": config["sources"]["launcher_v2"]["sha256"],
            **promoted,
        }
        if action_contract_sha256(
            {**row, "row_fingerprint_sha256": a1_row["row_fingerprint_sha256"]}
        ) != contract_sha:
            raise TechnicalRegistryError(f"A4 action identity drifted: {profile_id}")
        row["technical_row_fingerprint_sha256"] = technical_row_fingerprint(row)
        output.append(row)
    if len({row["action_contract_sha256"] for row in output}) != 72:
        raise TechnicalRegistryError("successor action-contract hashes are not unique")
    return output


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TECHNICAL_REGISTRY_FIELDS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in TECHNICAL_REGISTRY_FIELDS})
            count += 1
    return count


def _report(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    return f"""# UE split technical registry v1 — UE-A4

**Status:** FROZEN — {len(rows)}/{len(rows)} TECHNICALLY VALID

This create-only successor certifies the same 72 immutable UE-A1 action
identities using the authoritative UE-A2 `_02` CUDA/wire evidence. UE-A3
recorded zero genuine technical failures. No perception-quality or payload
filter was applied.

## Fixed codec boundary

- UE-to-edge quantized feature envelope: lossless `zstd`, level 3.
- Edge-to-spatial-map JSON packet: lossless `zlib`, level 1.
- Codec is fixed experiment infrastructure, not an agent action.

## Authority boundary

This is an evidence/certification registry. The v2 runtime and launcher remain
bound to the immutable UE-A1 operational registry and were not retargeted in
UE-A4. This registry authorizes the next checklist design task, UE-N1, but no
CARLA/OAI collection, continuous-q promotion, or policy training.
"""


def validate_bundle(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise TechnicalRegistryError("missing A4 manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("registry_id") != "ue_split_technical_registry_v1"
        or manifest.get("status") != "FROZEN"
        or manifest.get("claim_scope")
        != "A1_OPERATIONAL_IDENTITY_PLUS_A2_LOCAL_TECHNICAL_CERTIFICATION"
    ):
        raise TechnicalRegistryError("invalid A4 manifest schema/status")
    config = load_config(DEFAULT_CONFIG)
    if manifest.get("authority") != config["authority"]:
        raise TechnicalRegistryError("A4 manifest authority mismatch")
    if manifest.get("codec_boundary") != config["transport_decision"]:
        raise TechnicalRegistryError("A4 manifest codec boundary mismatch")
    expected_counts = {
        "profiles": 72,
        "technically_valid": 72,
        "technically_invalid": 0,
        "quality_masked": 0,
    }
    if manifest.get("counts") != expected_counts:
        raise TechnicalRegistryError("A4 manifest counts mismatch")
    expected_gate_values = {
        "a1_bundle_seals": "PASS",
        "a2_02_bundle_seals": "PASS",
        "exact_72_row_join": "PASS",
        "action_contract_identity": "PASS",
        "all_a2_stage_statuses": "PASS",
        "strict_model_smoke": "PASS",
        "actual_udp_72_of_72": "PASS",
        "negative_contract_34_of_34": "PASS",
        "no_quality_filter": "PASS",
        "runtime_retarget": "NOT_AUTHORIZED_NOT_PERFORMED",
    }
    if manifest.get("gates") != expected_gate_values:
        raise TechnicalRegistryError("A4 manifest gates mismatch")

    terminal_name = str(config["output"]["terminal_json"])
    expected_output_names = {
        str(config["output"]["registry_csv"]),
        str(config["output"]["report_md"]),
        str(config["output"]["resolved_config_json"]),
    }
    if manifest.get("terminal_decision_path") != terminal_name:
        raise TechnicalRegistryError("A4 terminal path mismatch")
    output_records = manifest.get("outputs", [])
    if not isinstance(output_records, list) or len(output_records) != 3:
        raise TechnicalRegistryError("A4 manifest must seal exactly three outputs")
    if {str(record.get("path")) for record in output_records} != expected_output_names:
        raise TechnicalRegistryError("A4 manifest output-name set mismatch")

    expected_files = {"manifest.json", terminal_name, *expected_output_names}
    registry_rows: list[dict[str, str]] | None = None
    for record_value in output_records:
        record = _require_mapping(record_value, "A4 output")
        relative = Path(str(record["path"]))
        if relative.is_absolute() or len(relative.parts) != 1:
            raise TechnicalRegistryError(f"A4 output path is not flat: {relative}")
        path = (output_dir / relative).resolve()
        if path.parent != output_dir:
            raise TechnicalRegistryError(f"A4 output escapes bundle: {relative}")
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise TechnicalRegistryError(f"A4 output seal mismatch: {path.name}")
        if path.stat().st_size != int(record["bytes"]):
            raise TechnicalRegistryError(f"A4 output byte-count mismatch: {path.name}")
        if "rows" in record:
            rows = _read_csv(path)
            if path.name != config["output"]["registry_csv"] or int(record["rows"]) != 72:
                raise TechnicalRegistryError("A4 row count is attached to the wrong output")
            if len(rows) != 72:
                raise TechnicalRegistryError("A4 registry row-count mismatch")
            registry_rows = rows
            if tuple(rows[0].keys()) != TECHNICAL_REGISTRY_FIELDS:
                raise TechnicalRegistryError("A4 registry column contract mismatch")
        elif path.suffix == ".csv":
            raise TechnicalRegistryError("unsealed A4 CSV row count")
    if registry_rows is None:
        raise TechnicalRegistryError("A4 registry CSV is missing")

    resolved_path = output_dir / str(config["output"]["resolved_config_json"])
    resolved_config = json.loads(resolved_path.read_text(encoding="utf-8"))
    if resolved_config != config:
        raise TechnicalRegistryError("A4 resolved config differs from current frozen config")
    if {int(row["action_index"]) for row in registry_rows} != set(range(72)):
        raise TechnicalRegistryError("A4 action-index set must be exactly 0..71")
    if len({row["profile_id"] for row in registry_rows}) != 72:
        raise TechnicalRegistryError("A4 profile IDs are not unique")
    if len({row["action_contract_sha256"] for row in registry_rows}) != 72:
        raise TechnicalRegistryError("A4 action contracts are not unique")
    expected_grid = {
        (family, quantizer, q)
        for family in ("noae", "ae32", "ae64", "ae128")
        for quantizer in ("per_channel_uint8", "per_channel_uint6", "per_channel_uint4")
        for q in ("0", "0.3", "0.5", "0.7", "0.9", "0.98")
    }
    observed_grid = {
        (
            row["model_family"],
            row["quantization_mode"],
            _canonical_decimal(row["roi_drop_fraction"], "roi_drop_fraction"),
        )
        for row in registry_rows
    }
    if observed_grid != expected_grid:
        raise TechnicalRegistryError("A4 factor grid is not exactly 4x3x6")
    a1_path = _repo_path(
        str(Path(config["a1"]["bundle_dir"]) / config["a1"]["registry_csv"])
    )
    a2_path = _repo_path(
        str(Path(config["a2"]["bundle_dir"]) / config["a2"]["profile_csv"])
    )
    a1_by_id = {row["profile_id"]: row for row in _read_csv(a1_path)}
    a2_by_id = {row["profile_id"]: row for row in _read_csv(a2_path)}
    a2_manifest = json.loads(
        _repo_path(
            str(Path(config["a2"]["bundle_dir"]) / config["a2"]["manifest_json"])
        ).read_text(encoding="utf-8")
    )
    launch_seals = _require_mapping(
        _require_mapping(a2_manifest.get("registry_audit"), "A2 registry audit").get(
            "launch_binding_sha256"
        ),
        "A2 launch seals",
    )
    profile_ids = {row["profile_id"] for row in registry_rows}
    if set(a1_by_id) != profile_ids or set(a2_by_id) != profile_ids or set(launch_seals) != profile_ids:
        raise TechnicalRegistryError("A4 source/evidence profile-ID sets differ")

    # Reconstruct the complete authoritative successor from the pinned A1/A2
    # inputs and require byte-facing CSV values to match.  This closes the
    # possibility of resealing a locally consistent row with downgraded gate
    # JSON or altered evidence annotations.
    loaded = _load_inputs(config)
    _validate_a2_bundle(config, loaded)
    expected_rows = join_and_promote(
        config,
        _read_csv(loaded["a1_registry"]),
        _read_csv(loaded["a2_profiles_path"]),
        loaded["certified_launch_bindings"],
        loaded["certified_launch_digests"],
    )
    expected_csv_rows = [
        {field: _csv_scalar(row[field]) for field in TECHNICAL_REGISTRY_FIELDS}
        for row in expected_rows
    ]
    if registry_rows != expected_csv_rows:
        raise TechnicalRegistryError("A4 registry differs from reconstructed A1/A2 authority")
    for row in registry_rows:
        profile_id = row["profile_id"]
        a1_row = a1_by_id[profile_id]
        a2_row = a2_by_id[profile_id]
        if row["technical_row_fingerprint_sha256"] != technical_row_fingerprint(row):
            raise TechnicalRegistryError(f"A4 row seal mismatch: {profile_id}")
        if row["technical_registry_schema"] != REGISTRY_SCHEMA:
            raise TechnicalRegistryError(f"A4 row schema mismatch: {profile_id}")
        if row["technical_registry_id"] != config["registry_id"]:
            raise TechnicalRegistryError(f"A4 row registry ID mismatch: {profile_id}")
        if row["certification_status"] != "TECHNICALLY_VALID_A1_IDENTITY_A2_SMOKE":
            raise TechnicalRegistryError(f"A4 row certification mismatch: {profile_id}")
        if row["source_registry_sha256"] != config["a1"]["registry_sha256"]:
            raise TechnicalRegistryError(f"A4 A1 source mismatch: {profile_id}")
        if row["a2_manifest_sha256"] != config["a2"]["manifest_sha256"]:
            raise TechnicalRegistryError(f"A4 A2 manifest mismatch: {profile_id}")
        if row["a2_profile_table_sha256"] != config["a2"]["profile_csv_sha256"]:
            raise TechnicalRegistryError(f"A4 A2 table mismatch: {profile_id}")
        if row["a2_bundle_name"] != config["a2"]["required_bundle_name"]:
            raise TechnicalRegistryError(f"A4 A2 bundle mismatch: {profile_id}")
        if row["source_row_fingerprint_sha256"] != a1_row["row_fingerprint_sha256"]:
            raise TechnicalRegistryError(f"A4 A1 row fingerprint mismatch: {profile_id}")
        if row["a2_profile_row_sha256"] != _a2_row_fingerprint(a2_row):
            raise TechnicalRegistryError(f"A4 A2 row fingerprint mismatch: {profile_id}")
        if row["a2_launch_binding_sha256"] != launch_seals[profile_id]:
            raise TechnicalRegistryError(f"A4 launch-binding seal mismatch: {profile_id}")
        if row["quality_mask_applied"] != "False":
            raise TechnicalRegistryError(f"A4 quality mask present: {profile_id}")
        if row["technical_validity_status"] != "TECHNICALLY_VALID":
            raise TechnicalRegistryError(f"A4 invalid status: {profile_id}")
        if row["technical_invalid_reason"]:
            raise TechnicalRegistryError(f"A4 invalid reason is non-empty: {profile_id}")
        if row["entropy_coder"] != "zstd" or row["entropy_level"] != "3":
            raise TechnicalRegistryError(f"A4 feature codec mismatch: {profile_id}")
        if row["runtime_path"] != config["sources"]["runtime_v2"]["path"]:
            raise TechnicalRegistryError(f"A4 certified runtime path mismatch: {profile_id}")
        if row["runtime_sha256"] != config["sources"]["runtime_v2"]["sha256"]:
            raise TechnicalRegistryError(f"A4 certified runtime hash mismatch: {profile_id}")
        if action_contract_sha256(
            {**row, "row_fingerprint_sha256": row["source_row_fingerprint_sha256"]}
        ) != row["action_contract_sha256"]:
            raise TechnicalRegistryError(f"A4 action identity mismatch: {profile_id}")
    actual_files = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise TechnicalRegistryError(
            f"A4 bundle file set mismatch: expected={sorted(expected_files)} actual={sorted(actual_files)}"
        )

    terminal_path = output_dir / terminal_name
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    payload = dict(terminal)
    back_reference = payload.pop("manifest_sha256", None)
    if back_reference != sha256_file(manifest_path):
        raise TechnicalRegistryError("A4 terminal manifest back-reference mismatch")
    if payload != manifest.get("terminal_decision_payload"):
        raise TechnicalRegistryError("A4 terminal differs from manifest commitment")
    if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != manifest.get(
        "terminal_decision_payload_sha256"
    ):
        raise TechnicalRegistryError("A4 terminal payload seal mismatch")
    expected_terminal = {
        "schema": TERMINAL_SCHEMA,
        "registry_id": config["registry_id"],
        "created_at": manifest["created_at"],
        "status": "FROZEN",
        "profile_count": 72,
        "technically_valid_profiles": 72,
        "technically_invalid_profiles": 0,
        "quality_mask_count": 0,
        "runtime_retargeted": False,
        "next_checklist_item": "UE-N1",
    }
    if payload != expected_terminal:
        raise TechnicalRegistryError("A4 terminal semantic contract mismatch")

    input_records = manifest.get("inputs")
    if not isinstance(input_records, list) or not input_records:
        raise TechnicalRegistryError("A4 manifest inputs missing")
    expected_input_paths = {
        DEFAULT_CONFIG.resolve(),
        Path(__file__).resolve(),
        _repo_path(str(Path(config["a1"]["bundle_dir"]) / config["a1"]["registry_csv"])),
        _repo_path(str(Path(config["a1"]["bundle_dir"]) / config["a1"]["manifest_json"])),
        _repo_path(str(Path(config["a1"]["bundle_dir"]) / config["a1"]["terminal_json"])),
        _repo_path(str(Path(config["a2"]["bundle_dir"]) / config["a2"]["manifest_json"])),
        _repo_path(str(Path(config["a2"]["bundle_dir"]) / config["a2"]["terminal_json"])),
        _repo_path(str(Path(config["a2"]["bundle_dir"]) / config["a2"]["profile_csv"])),
        _repo_path(str(Path(config["a2"]["bundle_dir"]) / config["a2"]["model_smoke_json"])),
        _repo_path(str(Path(config["a2"]["bundle_dir"]) / config["a2"]["negative_tests_json"])),
        _repo_path(str(Path(config["a2"]["bundle_dir"]) / config["a2"]["transport_json"])),
        *{
            _repo_path(str(spec["path"]))
            for spec in config["sources"].values()
        },
    }
    recorded_input_paths = {
        (
            Path(str(record["path"])).resolve()
            if Path(str(record["path"])).is_absolute()
            else (ROOT / str(record["path"])).resolve()
        )
        for record in input_records
    }
    if recorded_input_paths != expected_input_paths or len(input_records) != len(expected_input_paths):
        raise TechnicalRegistryError("A4 manifest input-name set mismatch")
    seen_inputs: set[str] = set()
    for record_value in input_records:
        record = _require_mapping(record_value, "A4 input")
        raw_path = Path(str(record["path"]))
        path = raw_path.resolve() if raw_path.is_absolute() else (ROOT / raw_path).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError as exc:
            raise TechnicalRegistryError(f"A4 input escapes repository: {path}") from exc
        key = str(path)
        if key in seen_inputs:
            raise TechnicalRegistryError(f"duplicate A4 input seal: {path}")
        seen_inputs.add(key)
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise TechnicalRegistryError(f"A4 input seal mismatch: {path}")
        if path.stat().st_size != int(record["bytes"]):
            raise TechnicalRegistryError(f"A4 input byte-count mismatch: {path}")
    return manifest


def assemble(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path | None = None,
    *,
    now: str | None = None,
) -> Path:
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    if output_dir is None:
        output_dir = _repo_path(str(config["output"]["root"]))
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise TechnicalRegistryError(f"refusing to overwrite existing A4 registry: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    created_at = now or utc_now()
    try:
        loaded = _load_inputs(config)
        _validate_a2_bundle(config, loaded)
        a1_rows = _read_csv(loaded["a1_registry"])
        a2_rows = _read_csv(loaded["a2_profiles_path"])
        rows = join_and_promote(
            config,
            a1_rows,
            a2_rows,
            loaded["certified_launch_bindings"],
            loaded["certified_launch_digests"],
        )
        if len(rows) != 72:
            raise TechnicalRegistryError("A4 promotion did not produce exactly 72 rows")

        names = _require_mapping(config["output"], "output")
        registry_path = temporary / str(names["registry_csv"])
        report_path = temporary / str(names["report_md"])
        resolved_path = temporary / str(names["resolved_config_json"])
        if _write_csv(registry_path, rows) != 72:
            raise TechnicalRegistryError("A4 registry changed while writing")
        report_path.write_text(_report(config, rows), encoding="utf-8")
        _write_json(resolved_path, config)

        terminal_payload = {
            "schema": TERMINAL_SCHEMA,
            "registry_id": config["registry_id"],
            "created_at": created_at,
            "status": "FROZEN",
            "profile_count": 72,
            "technically_valid_profiles": 72,
            "technically_invalid_profiles": 0,
            "quality_mask_count": 0,
            "runtime_retargeted": False,
            "next_checklist_item": "UE-N1",
        }
        output_records = []
        for path, row_count in (
            (registry_path, 72),
            (report_path, None),
            (resolved_path, None),
        ):
            record: dict[str, Any] = {
                "path": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            if row_count is not None:
                record["rows"] = row_count
            output_records.append(record)

        input_records = [
            {
                "kind": "config",
                "path": str(config_path.relative_to(ROOT)),
                "sha256": sha256_file(config_path),
                "bytes": config_path.stat().st_size,
            },
            {
                "kind": "source",
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": sha256_file(Path(__file__).resolve()),
                "bytes": Path(__file__).stat().st_size,
            },
            {
                "kind": "A1 registry",
                "path": str(loaded["a1_registry"].relative_to(ROOT)),
                "sha256": config["a1"]["registry_sha256"],
                "bytes": loaded["a1_registry"].stat().st_size,
            },
            {
                "kind": "A1 manifest",
                "path": str(
                    (loaded["a1_bundle"] / config["a1"]["manifest_json"]).relative_to(ROOT)
                ),
                "sha256": config["a1"]["manifest_sha256"],
                "bytes": (
                    loaded["a1_bundle"] / config["a1"]["manifest_json"]
                ).stat().st_size,
            },
            {
                "kind": "A1 terminal",
                "path": str(
                    (loaded["a1_bundle"] / config["a1"]["terminal_json"]).relative_to(ROOT)
                ),
                "sha256": config["a1"]["terminal_sha256"],
                "bytes": (
                    loaded["a1_bundle"] / config["a1"]["terminal_json"]
                ).stat().st_size,
            },
            {
                "kind": "A2 manifest",
                "path": str((loaded["a2_bundle"] / config["a2"]["manifest_json"]).relative_to(ROOT)),
                "sha256": config["a2"]["manifest_sha256"],
                "bytes": (loaded["a2_bundle"] / config["a2"]["manifest_json"]).stat().st_size,
            },
            *[
                {
                    "kind": f"A2 {label}",
                    "path": str((loaded["a2_bundle"] / config["a2"][path_key]).relative_to(ROOT)),
                    "sha256": config["a2"][sha_key],
                    "bytes": (
                        loaded["a2_bundle"] / config["a2"][path_key]
                    ).stat().st_size,
                }
                for label, path_key, sha_key in (
                    ("terminal", "terminal_json", "terminal_sha256"),
                    ("profile table", "profile_csv", "profile_csv_sha256"),
                    ("model smoke", "model_smoke_json", "model_smoke_sha256"),
                    ("negative tests", "negative_tests_json", "negative_tests_sha256"),
                    ("transport", "transport_json", "transport_sha256"),
                )
            ],
            *loaded["source_inputs"],
        ]
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "registry_id": config["registry_id"],
            "created_at": created_at,
            "status": "FROZEN",
            "claim_scope": "A1_OPERATIONAL_IDENTITY_PLUS_A2_LOCAL_TECHNICAL_CERTIFICATION",
            "authority": dict(config["authority"]),
            "codec_boundary": dict(config["transport_decision"]),
            "counts": {
                "profiles": 72,
                "technically_valid": 72,
                "technically_invalid": 0,
                "quality_masked": 0,
            },
            "gates": {
                "a1_bundle_seals": "PASS",
                "a2_02_bundle_seals": "PASS",
                "exact_72_row_join": "PASS",
                "action_contract_identity": "PASS",
                "all_a2_stage_statuses": "PASS",
                "strict_model_smoke": "PASS",
                "actual_udp_72_of_72": "PASS",
                "negative_contract_34_of_34": "PASS",
                "no_quality_filter": "PASS",
                "runtime_retarget": "NOT_AUTHORIZED_NOT_PERFORMED",
            },
            "inputs": input_records,
            "outputs": output_records,
            "terminal_decision_path": str(names["terminal_json"]),
            "terminal_decision_payload": terminal_payload,
            "terminal_decision_payload_sha256": hashlib.sha256(
                canonical_json_bytes(terminal_payload)
            ).hexdigest(),
        }
        manifest_path = temporary / str(names["manifest_json"])
        _write_json(manifest_path, manifest)
        terminal = {**terminal_payload, "manifest_sha256": sha256_file(manifest_path)}
        _write_json(temporary / str(names["terminal_json"]), terminal)
        validate_bundle(temporary)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--validate-bundle", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_bundle is not None:
        validate_bundle(args.validate_bundle)
        print(f"UE-A4 registry valid: {Path(args.validate_bundle).resolve()}")
        return 0
    output = assemble(args.config, args.output_dir)
    print(f"UE-A4 technical registry: {output}")
    print("Status: FROZEN (72/72 technically valid; next UE-N1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
