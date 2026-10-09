"""Run-manifest and row-schema documents for the offline grid."""

from __future__ import annotations

from typing import Any, Mapping

from ..state_reward_transition_contract import RewardSpecV1
from .contract import (
    CONTRACT_SHA256,
    DEPLOYMENT_CLAIM,
    EVIDENCE_LABEL,
    EXPECTED_GRID_ROWS,
    FAMILIES,
    Q_E4_GRID,
    QUANTIZERS,
    RUN_MANIFEST_SCHEMA_ID,
    SCHEMA_ID,
    SCHEMA_VERSION,
    UDP_CHUNK_BYTES_INCLUDING_HEADER,
    UDP_PAYLOAD_BYTES_PER_DATAGRAM,
    canonical_sha256,
)
from .quality import calibration_status
from .schema import ROW_FIELDS


def row_schema_document() -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": "splitfusion_exact_offline_quality_grid_row_schema_v1",
        "row_record": SCHEMA_ID,
        "row_schema_version": SCHEMA_VERSION,
        "fields_in_canonical_order": list(ROW_FIELDS),
        "primary_key": "row_key_sha256",
        "natural_key": [
            "episode_id",
            "sample_id",
            "frame_id",
            "grid_split",
            "mode_id",
            "family",
            "quantizer",
            "q_e4",
        ],
        "raw_sufficient_statistics": {
            "segmentation": "per-class GT/prediction/intersection/union pixel counts",
            "localization": (
                "per-class eligible GT, TP/FP/FN, ignored predictions and full "
                "matched XY-error vectors plus derived sums/mean/median/max"
            ),
            "retained_masks": False,
        },
        "quality_interpretation": (
            "Q values are exactly recomputable from the raw counts/error vectors "
            "and caller-pinned RewardSpecV1; they are not primary evidence"
        ),
        "scalar_reward_weights_used_by_extraction": False,
        "transport_byte_accounting": {
            "scientific_inner_payload": "exact frozen codec zstd bytes",
            "sfd1_outer_envelope": "exact pack_envelope v2 common header plus frame context",
            "total_transmitted_bytes": "scientific inner plus SFD1 outer envelope",
            "datagram_count": "exact deployed !IHH chunker over total transmitted bytes",
            "udp_application_bytes": "total transmitted bytes plus all !IHH chunk headers",
            "off_anchor_identity": (
                "explicit non-dispatchable uint32 sentinel for byte accounting only; "
                "no SFD1-v2 dynamic-action execution claim"
            ),
        },
        "udp": {
            "chunk_bytes_including_header": UDP_CHUNK_BYTES_INCLUDING_HEADER,
            "payload_bytes_per_datagram": UDP_PAYLOAD_BYTES_PER_DATAGRAM,
        },
        "row_schema_sha256": "",
    }
    document["row_schema_sha256"] = canonical_sha256(
        {key: value for key, value in document.items() if key != "row_schema_sha256"}
    )
    return document


def run_binding_document(
    *,
    selection_manifest: Mapping[str, Any],
    preflight: Mapping[str, Any],
    reward_spec: RewardSpecV1,
) -> dict[str, Any]:
    row_schema = row_schema_document()
    return {
        "contract_sha256": CONTRACT_SHA256,
        "selection_manifest_sha256": selection_manifest["selection_manifest_sha256"],
        "preflight_binding_sha256": preflight["preflight_binding_sha256"],
        "row_schema_sha256": row_schema["row_schema_sha256"],
        "reward_spec_sha256": reward_spec.canonical_sha256(),
        "source_module_sha256": dict(preflight["source_bindings"]),
        "source_module_roles": dict(preflight["source_roles"]),
        "evaluation_source_sha256": dict(preflight["evaluation_source_bindings"]),
        "checkpoint_sha256": {
            value["path"]: value["sha256"]
            for value in preflight["checkpoints"].values()
        },
    }


def run_manifest_document(
    *,
    selection_manifest: Mapping[str, Any],
    preflight: Mapping[str, Any],
    reward_spec: RewardSpecV1,
) -> dict[str, Any]:
    binding = run_binding_document(
        selection_manifest=selection_manifest,
        preflight=preflight,
        reward_spec=reward_spec,
    )
    run_binding_sha256 = canonical_sha256(binding)
    row_schema = row_schema_document()
    minimum = EXPECTED_GRID_ROWS * 3_000
    maximum = EXPECTED_GRID_ROWS * 8_000
    document: dict[str, Any] = {
        "schema": RUN_MANIFEST_SCHEMA_ID,
        "evidence_label": EVIDENCE_LABEL,
        "deployment_claim": DEPLOYMENT_CLAIM,
        "contract_sha256": CONTRACT_SHA256,
        "run_binding": binding,
        "run_binding_sha256": run_binding_sha256,
        "row_schema": row_schema,
        "selection_manifest_sha256": selection_manifest["selection_manifest_sha256"],
        "selected_frame_count": int(selection_manifest["selected_frame_count"]),
        "families": list(FAMILIES),
        "quantizers": list(QUANTIZERS),
        "q_e4": list(Q_E4_GRID),
        "expected_rows": EXPECTED_GRID_ROWS,
        "storage_estimate": {
            "basis": "planning range of 3-8 kB canonical JSON per row (matched-error vectors vary); SQLite indexes/WAL add overhead",
            "row_json_min_bytes": minimum,
            "row_json_max_bytes": maximum,
            "row_json_min_mib": round(minimum / 2**20, 2),
            "row_json_max_mib": round(maximum / 2**20, 2),
            "actual_bytes_recorded_after_execution": True,
        },
        "reward_spec_sha256": reward_spec.canonical_sha256(),
        "quality_calibration_status": calibration_status(reward_spec),
        "raw_sufficient_statistics_are_primary": True,
        "scalar_reward_weights_used_by_extraction": False,
        "test_episode_access": "NONE_ALLOWLIST_ONLY_05_06",
        "execution_status_at_manifest_creation": "PREPARED_NOT_STARTED",
        "durable_bundle_policy": {
            "selection_manifest": "exact caller bytes copied create-only",
            "reward_spec": "exact canonical caller bytes copied create-only",
            "metadata_preflight": "persisted create-only",
            "runtime_preflight_and_equivalence": "persisted per execution attempt",
            "completion": "final manifest and terminal only after exact expected-key reconciliation",
        },
        "completion_source": "quality_rows.sqlite3 exact expected-key reconciliation",
        "run_manifest_sha256": "",
    }
    document["run_manifest_sha256"] = canonical_sha256(
        {key: value for key, value in document.items() if key != "run_manifest_sha256"}
    )
    return document


def completion_manifest_document(
    *, initial_manifest: Mapping[str, Any], execution_result: Mapping[str, Any],
    artifact_sha256: Mapping[str, str], store_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the final immutable COMPLETE manifest from durable artifacts."""

    validate_run_manifest(initial_manifest)
    if execution_result.get("status") != "COMPLETE":
        raise ValueError("completion manifest requires a COMPLETE execution result")
    if store_audit.get("complete") is not True:
        raise ValueError("completion manifest requires exact expected-key reconciliation")
    if int(store_audit.get("rows", -1)) != EXPECTED_GRID_ROWS:
        raise ValueError("completion manifest store row count drift")
    if not artifact_sha256 or any(
        not isinstance(path, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        for path, digest in artifact_sha256.items()
    ):
        raise ValueError("completion artifact inventory is malformed")
    document = dict(initial_manifest)
    document.update(
        {
            "execution_status": "COMPLETE",
            "initial_run_manifest_sha256": initial_manifest["run_manifest_sha256"],
            "artifact_sha256": dict(sorted(artifact_sha256.items())),
            "store_audit": dict(store_audit),
            "execution_result_status": str(execution_result["status"]),
            "run_manifest_sha256": "",
        }
    )
    document["run_manifest_sha256"] = canonical_sha256(
        {key: value for key, value in document.items() if key != "run_manifest_sha256"}
    )
    return document


def validate_completion_manifest(document: Mapping[str, Any]) -> str:
    digest = validate_run_manifest(document)
    if document.get("execution_status") != "COMPLETE":
        raise ValueError("final quality-grid manifest is not COMPLETE")
    if document.get("initial_run_manifest_sha256") == document.get("run_manifest_sha256"):
        raise ValueError("final and initial run manifests must be distinct")
    audit = document.get("store_audit")
    if not isinstance(audit, Mapping) or audit.get("complete") is not True:
        raise ValueError("final quality-grid manifest lacks a complete store audit")
    if int(audit.get("rows", -1)) != EXPECTED_GRID_ROWS:
        raise ValueError("final quality-grid manifest row count drift")
    artifacts = document.get("artifact_sha256")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("final quality-grid manifest lacks artifact bindings")
    return digest


def validate_run_manifest(document: Mapping[str, Any]) -> str:
    if document.get("schema") != RUN_MANIFEST_SCHEMA_ID:
        raise ValueError("wrong quality-grid run manifest schema")
    if document.get("contract_sha256") != CONTRACT_SHA256:
        raise ValueError("quality-grid run manifest contract drift")
    binding = document.get("run_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("quality-grid run binding is missing")
    if canonical_sha256(binding) != document.get("run_binding_sha256"):
        raise ValueError("quality-grid run binding digest drift")
    if binding.get("selection_manifest_sha256") != document.get(
        "selection_manifest_sha256"
    ):
        raise ValueError("quality-grid selection binding drift")
    if binding.get("reward_spec_sha256") != document.get("reward_spec_sha256"):
        raise ValueError("quality-grid reward-spec binding drift")
    row_schema = document.get("row_schema")
    if not isinstance(row_schema, Mapping):
        raise ValueError("quality-grid row schema is missing")
    expected_row_schema = canonical_sha256(
        {key: value for key, value in row_schema.items() if key != "row_schema_sha256"}
    )
    if row_schema.get("row_schema_sha256") != expected_row_schema:
        raise ValueError("quality-grid row-schema digest drift")
    expected = canonical_sha256(
        {key: value for key, value in document.items() if key != "run_manifest_sha256"}
    )
    if document.get("run_manifest_sha256") != expected:
        raise ValueError("quality-grid run manifest self-digest drift")
    if document.get("expected_rows") != EXPECTED_GRID_ROWS:
        raise ValueError("quality-grid run manifest row count drift")
    return expected
