"""Fail-closed binder for the measured Run-4 target-radio MCS sequence.

This module binds the successful 273-PRB/100-MHz/4D5U capture.  Evidence
loading is explicit.  Importing the module performs no I/O and imports no
runtime, CUDA, or learning framework.

The model-facing surface is intentionally narrow: a policy observation is
only an explicit MCS status plus, when valid, the prior UE-decoded round-0
UL-MCS index.  A transition contains only two such observations and the
registered duration.  Profile names, target SNR, gNB evidence, timestamps,
frame/decision identifiers, provenance hashes, and partition labels are
verified while binding but never enter a policy observation or transition.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence, Tuple

__all__ = [
    "DynamicMcs273PrbEvidenceError",
    "EvidenceHashMismatch",
    "EvidenceSchemaError",
    "EvidenceIdentityError",
    "PolicyFeatureUnavailable",
    "McsStatus",
    "PolicyMcsObservationV1",
    "PolicyMcsTransitionV1",
    "PolicyMcsSequenceV1",
    "DynamicMcs273PrbEvidenceV1",
    "SOURCE_RUN_RELATIVE_PATH",
    "EXPECTED_MANIFEST_SHA256",
    "EXPECTED_SUCCESS_TERMINAL_SHA256",
    "EXPECTED_ANALYSIS_SHA256",
    "EXPECTED_OBSERVATIONS_SHA256",
    "EXPECTED_TRANSITIONS_SHA256",
    "EXPECTED_CANONICAL_EVIDENCE_SHA256",
    "load_dynamic_mcs_273prb_evidence",
]


SCHEMA_ID = "splitfusion_run4_dynamic_mcs_273prb_evidence_v1"
SCHEMA_VERSION = 1
SOURCE_RUN_RELATIVE_PATH = Path(
    "rl_agent/experiments/ue_dynamic_mcs_273prb_v1/"
    "20260924_target_radio_capture_v1_retry3"
)

EXPECTED_MANIFEST_SHA256 = (
    "2279d4b0be5861df6431f80bd0f879c0f2a43a307c585572709547429cde1943"
)
EXPECTED_SUCCESS_TERMINAL_SHA256 = (
    "e12b68d839aedc615d28ac84106f8a2dd2836a0837aebf673e0ec80e72be19ab"
)
EXPECTED_ANALYSIS_SHA256 = (
    "252437137c1cf8b3a9351c5fe92d647883c2e81e9b1d3e61b9647493b3f2c06d"
)
EXPECTED_OBSERVATIONS_SHA256 = (
    "af64fd55791e97f788fc6486d961c5d818c83f810fc51a1ed9982f4b84b33e32"
)
EXPECTED_TRANSITIONS_SHA256 = (
    "8a1dd320a3acc7510578425ae322f432e808380edf99de48faa030bea45fd7f2"
)

# Filled from the canonical, model-facing evidence payload.  It is independent
# of verifier-only row metadata but binds the five frozen source digests.
EXPECTED_CANONICAL_EVIDENCE_SHA256 = (
    "7ec3edd93c67c5938188de8a24b838b5330970e357229b8a9e3eaa5000c796d5"
)

EXPECTED_STATUS = "RUN4_DYNAMIC_MCS_273PRB_CAPTURED"
EXPECTED_CONTRACT_ID = "ue_dynamic_mcs_273prb_v1"
EXPECTED_SOURCE_SCHEMA = "scenesense.ue_dynamic_mcs_273prb.v1"
EXPECTED_ANALYSIS_SCHEMA = "scenesense.ue_dynamic_mcs_273prb.analysis.v1"
EXPECTED_DESIGN_SHA256 = (
    "95efa89259b7aa29a4658e2f2612792dfe962461d496aad87f84aba15e40442e"
)
EXPECTED_RADIO_PROFILE_ID = "OAI_N78_100MHZ_273PRB_4D5U_V1"
EXPECTED_CLAIM_BOUNDARY = (
    "TWO_REGISTERED_DYNAMIC_PROFILE_UE_DECODED_PRIOR_UL_MCS_SEQUENCE_UNDER_"
    "OAI_N78_100MHZ_273PRB_4D5U_V1_FOR_RUN4_OFFLINE_FIT_AND_INTERNAL_"
    "VALIDATION_ONLY_NOT_A_GENERALIZATION_OR_DEPLOYMENT_CLAIM"
)
EXPECTED_POLICY_BOUNDARY = (
    "ONLY policy_feature_json; target_snr_db_verifier_only, profile_id, "
    "frame/timestamps and gNB data are forbidden actor inputs"
)
EXPECTED_PROFILES = (
    ("MID_VARIABLE", "GM_V2_MID_VARIABLE_SEED_2026082102"),
    ("FADE_RECOVERY", "GM_V2_FADE_RECOVERY_SEED_2026082104"),
)
EXPECTED_RADIO = {
    "band": 78,
    "bandwidth_mhz": 100,
    "downlink_slots": 4,
    "mcs_policy": "sinr",
    "numerology": 1,
    "prb": 273,
    "ue_count": 1,
    "ue_ip": "10.0.0.2",
    "uplink_slots": 5,
}

PERIOD_NS = 100_000_000
DURATION_TENSORS = 2
SUCCESSOR_DELTA_NS = 200_000_000
OBSERVATION_COUNT = 600
TRANSITION_COUNT = 592
FIT_OBSERVATION_COUNT = 420
VALIDATION_OBSERVATION_COUNT = 180
FIT_TRANSITION_COUNT = 416
VALIDATION_TRANSITION_COUNT = 176
MCS_MIN = 0
MCS_MAX = 28

MANIFEST_NAME = "manifest.json"
TERMINAL_NAME = "RUN4_DYNAMIC_MCS_273PRB_CAPTURED.json"
ANALYSIS_NAME = "analysis/analysis.json"
OBSERVATIONS_NAME = "analysis/mcs_observations.csv"
TRANSITIONS_NAME = "analysis/duration2_transitions.csv"

OBSERVATION_FIELDS = (
    "profile_id", "trace_id", "decision_index", "partition",
    "scheduled_action_open_monotonic_ns", "actual_send_open_monotonic_ns",
    "schedule_lag_ms", "mcs_status", "prior_ul_mcs_index", "mcs_age_ms",
    "source_grant_monotonic_ns", "source_grant_rnti",
    "source_grant_dci_frame", "source_grant_dci_slot",
    "source_grant_sched_frame", "source_grant_sched_slot",
    "source_grant_mcs_table", "source_grant_round", "source_grant_ndi",
    "source_provenance_sha256", "policy_feature_json",
    "target_snr_db_verifier_only", "hidden_profile_verifier_only",
)
TRANSITION_FIELDS = (
    "profile_id", "partition", "current_decision_index",
    "successor_decision_index", "duration_tensors", "scheduled_delta_ns",
    "current_mcs_status", "current_prior_ul_mcs_index",
    "successor_mcs_status", "successor_prior_ul_mcs_index",
    "learning_eligible", "reset_required_after_current",
)


class DynamicMcs273PrbEvidenceError(ValueError):
    """Base class for target-radio evidence refusal."""


class EvidenceHashMismatch(DynamicMcs273PrbEvidenceError):
    """A pinned artifact or inventory member changed."""


class EvidenceSchemaError(DynamicMcs273PrbEvidenceError):
    """Evidence is malformed or violates a frozen structural invariant."""


class EvidenceIdentityError(DynamicMcs273PrbEvidenceError):
    """Radio, profile, trace, partition, or design identity drifted."""


class PolicyFeatureUnavailable(DynamicMcs273PrbEvidenceError):
    """Missing/stale MCS requires external fallback, never imputation."""


class McsStatus(str, Enum):
    VALID = "VALID"
    MISSING_NO_PRIOR_GRANT = "MISSING_NO_PRIOR_GRANT"
    STALE = "STALE"


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise EvidenceSchemaError("value is not canonical-JSON encodable") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise EvidenceHashMismatch(f"cannot read required evidence {path}") from exc
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    def reject_duplicates(pairs: Sequence[Tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceSchemaError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceSchemaError(f"cannot parse required JSON {path}") from exc


def _read_csv(path: Path, fields: Sequence[str]) -> tuple[dict[str, str], ...]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                raise EvidenceSchemaError(f"{path}: unexpected CSV header")
            rows: list[dict[str, str]] = []
            for line, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise EvidenceSchemaError(f"{path}:{line}: ragged CSV row")
                row["__line__"] = str(line)
                rows.append(row)
            return tuple(rows)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise EvidenceSchemaError(f"cannot parse required CSV {path}") from exc


def _exact_int(text: str, name: str, *, minimum: int = 0) -> int:
    if not isinstance(text, str) or not text:
        raise EvidenceSchemaError(f"{name} is missing")
    try:
        value = int(text, 10)
    except ValueError as exc:
        raise EvidenceSchemaError(f"{name} is not a base-10 integer") from exc
    if str(value) != text or value < minimum:
        raise EvidenceSchemaError(f"{name} is not a canonical int >= {minimum}")
    return value


def _exact_bool(text: str, name: str) -> bool:
    if text == "True":
        return True
    if text == "False":
        return False
    raise EvidenceSchemaError(f"{name} must be exactly True or False")


def _partition_for(index: int) -> str:
    if 0 <= index <= 209:
        return "FIT"
    if 210 <= index <= 299:
        return "INTERNAL_VALIDATION"
    raise EvidenceIdentityError(f"decision index {index} is outside the frozen grid")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class PolicyMcsObservationV1:
    """The complete MCS surface allowed to reach a model or policy."""

    status: McsStatus
    prior_ul_mcs_index: Optional[int]

    def __post_init__(self) -> None:
        if not isinstance(self.status, McsStatus):
            raise EvidenceSchemaError("status must be an exact McsStatus")
        if self.status is McsStatus.VALID:
            if type(self.prior_ul_mcs_index) is not int or not (
                MCS_MIN <= self.prior_ul_mcs_index <= MCS_MAX
            ):
                raise EvidenceSchemaError("valid prior MCS must be an exact table-0 index")
        elif self.prior_ul_mcs_index is not None:
            raise EvidenceSchemaError("missing/stale prior MCS must be null")

    def policy_features(self) -> dict[str, int]:
        if self.status is not McsStatus.VALID:
            raise PolicyFeatureUnavailable(
                f"{self.status.value} requires external fallback; no MCS is imputed"
            )
        return {"prior_ul_mcs_index": int(self.prior_ul_mcs_index)}

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "prior_ul_mcs_index": self.prior_ul_mcs_index,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class PolicyMcsTransitionV1:
    """Identity-free causal current->successor MCS transition."""

    current: PolicyMcsObservationV1
    successor: PolicyMcsObservationV1
    duration_tensors: int
    learning_eligible: bool

    def __post_init__(self) -> None:
        if type(self.current) is not PolicyMcsObservationV1:
            raise EvidenceSchemaError("current must be PolicyMcsObservationV1")
        if type(self.successor) is not PolicyMcsObservationV1:
            raise EvidenceSchemaError("successor must be PolicyMcsObservationV1")
        if type(self.duration_tensors) is not int or self.duration_tensors != 2:
            raise EvidenceSchemaError("target-radio evidence supports exactly duration=2")
        expected = (
            self.current.status is McsStatus.VALID
            and self.successor.status is McsStatus.VALID
        )
        if type(self.learning_eligible) is not bool or self.learning_eligible != expected:
            raise EvidenceSchemaError("learning eligibility disagrees with MCS validity")

    def policy_values(self) -> tuple[int, int]:
        return (
            self.current.policy_features()["prior_ul_mcs_index"],
            self.successor.policy_features()["prior_ul_mcs_index"],
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "current": self.current.to_canonical_dict(),
            "duration_tensors": self.duration_tensors,
            "learning_eligible": self.learning_eligible,
            "successor": self.successor.to_canonical_dict(),
        }


@dataclass(frozen=True, slots=True)
class PolicyMcsSequenceV1:
    """One trajectory with its identity intentionally erased."""

    observations: tuple[PolicyMcsObservationV1, ...]
    transitions: tuple[PolicyMcsTransitionV1, ...]

    def __post_init__(self) -> None:
        if not self.observations or any(
            type(item) is not PolicyMcsObservationV1 for item in self.observations
        ):
            raise EvidenceSchemaError("sequence observations are malformed")
        if any(type(item) is not PolicyMcsTransitionV1 for item in self.transitions):
            raise EvidenceSchemaError("sequence transitions are malformed")

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "observations": [item.to_canonical_dict() for item in self.observations],
            "transitions": [item.to_canonical_dict() for item in self.transitions],
        }


@dataclass(frozen=True, slots=True)
class DynamicMcs273PrbEvidenceV1:
    """Hash-bound splits with no verifier identity on their model surface."""

    fit_sequences: tuple[PolicyMcsSequenceV1, ...]
    internal_validation_sequences: tuple[PolicyMcsSequenceV1, ...]
    source_manifest_sha256: str
    source_terminal_sha256: str
    source_analysis_sha256: str
    source_observations_sha256: str
    source_transitions_sha256: str
    source_inventory_sha256: str
    canonical_evidence_sha256: str

    def __post_init__(self) -> None:
        if len(self.fit_sequences) != 2 or len(self.internal_validation_sequences) != 2:
            raise EvidenceSchemaError("evidence must contain two sequences per split")
        for value in (
            self.source_manifest_sha256,
            self.source_terminal_sha256,
            self.source_analysis_sha256,
            self.source_observations_sha256,
            self.source_transitions_sha256,
            self.source_inventory_sha256,
            self.canonical_evidence_sha256,
        ):
            if not _is_digest(value):
                raise EvidenceSchemaError("evidence digests must be lowercase SHA-256")

    @property
    def fit_transitions(self) -> tuple[PolicyMcsTransitionV1, ...]:
        return tuple(item for sequence in self.fit_sequences for item in sequence.transitions)

    @property
    def internal_validation_transitions(self) -> tuple[PolicyMcsTransitionV1, ...]:
        return tuple(
            item
            for sequence in self.internal_validation_sequences
            for item in sequence.transitions
        )


@dataclass(frozen=True, slots=True)
class _VerifiedObservation:
    profile_id: str
    trace_id: str
    decision_index: int
    partition: str
    scheduled_ns: int
    provenance_sha256: str
    policy: PolicyMcsObservationV1


def _verify_pinned(path: Path, expected: str) -> None:
    observed = _sha256_file(path)
    if observed != expected:
        raise EvidenceHashMismatch(f"{path}: {observed} != frozen {expected}")


def _verify_inventory(run_dir: Path, manifest: Mapping[str, Any]) -> str:
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != 93:
        raise EvidenceSchemaError("manifest must contain exactly 93 inventory rows")
    canonical_rows: list[dict[str, Any]] = []
    declared: set[str] = set()
    for number, row in enumerate(entries):
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path", "sha256", "size_bytes"
        }:
            raise EvidenceSchemaError(f"manifest inventory row {number} is malformed")
        relative = row["relative_path"]
        pure = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            pure is None
            or pure.is_absolute()
            or not pure.parts
            or any(part in ("", ".", "..") for part in pure.parts)
            or relative in declared
        ):
            raise EvidenceSchemaError(f"unsafe/duplicate inventory path {relative!r}")
        declared.add(relative)
        expected_hash = row["sha256"]
        size = row["size_bytes"]
        if not _is_digest(expected_hash) or type(size) is not int or size < 0:
            raise EvidenceSchemaError(f"inventory metadata invalid for {relative}")
        path = run_dir / relative
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            raise EvidenceHashMismatch(f"missing inventory member {path}") from exc
        if not path.is_file() or actual_size != size:
            raise EvidenceHashMismatch(f"inventory size/type mismatch for {path}")
        observed = _sha256_file(path)
        if observed != expected_hash:
            raise EvidenceHashMismatch(f"inventory digest mismatch for {path}")
        canonical_rows.append({
            "relative_path": relative,
            "sha256": expected_hash,
            "size_bytes": size,
        })
    try:
        actual = {
            path.relative_to(run_dir).as_posix()
            for path in run_dir.rglob("*")
            if path.is_file()
        }
    except OSError as exc:
        raise EvidenceHashMismatch("cannot enumerate source inventory") from exc
    expected_actual = declared | {MANIFEST_NAME, TERMINAL_NAME}
    if actual != expected_actual:
        raise EvidenceHashMismatch(
            f"source inventory differs: missing={sorted(expected_actual - actual)}, "
            f"foreign={sorted(actual - expected_actual)}"
        )
    return _canonical_sha256(canonical_rows)


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != EXPECTED_SOURCE_SCHEMA:
        raise EvidenceIdentityError("manifest schema drifted")
    if manifest.get("contract_id") != EXPECTED_CONTRACT_ID:
        raise EvidenceIdentityError("manifest contract ID drifted")
    if manifest.get("contract_version") != 1:
        raise EvidenceIdentityError("manifest contract version drifted")
    if manifest.get("status") != EXPECTED_STATUS or manifest.get("failure") is not None:
        raise EvidenceIdentityError("manifest is not the successful terminal state")
    if manifest.get("design_sha256") != EXPECTED_DESIGN_SHA256:
        raise EvidenceIdentityError("design digest drifted")
    design = manifest.get("design")
    if not isinstance(design, Mapping) or _canonical_sha256(design) != EXPECTED_DESIGN_SHA256:
        raise EvidenceIdentityError("embedded design does not match its digest")
    if manifest.get("radio_profile_id") != EXPECTED_RADIO_PROFILE_ID:
        raise EvidenceIdentityError("manifest radio identity drifted")
    if manifest.get("claim_boundary") != EXPECTED_CLAIM_BOUNDARY:
        raise EvidenceIdentityError("claim boundary drifted")
    if design.get("radio_profile_id") != EXPECTED_RADIO_PROFILE_ID:
        raise EvidenceIdentityError("design radio identity drifted")
    if design.get("radio") != EXPECTED_RADIO:
        raise EvidenceIdentityError("273-PRB/100-MHz/4D5U radio design drifted")
    grid = design.get("decision_grid")
    if grid != {
        "fit": [0, 209],
        "frames_per_profile": 300,
        "hold_duration_tensors": 2,
        "internal_validation": [210, 299],
        "period_ns": PERIOD_NS,
        "successor_delta_ns": SUCCESSOR_DELTA_NS,
    }:
        raise EvidenceIdentityError("decision grid drifted")
    profiles = design.get("profiles")
    if not isinstance(profiles, list) or tuple(
        (row.get("profile_id"), row.get("trace_id"))
        for row in profiles if isinstance(row, Mapping)
    ) != EXPECTED_PROFILES:
        raise EvidenceIdentityError("registered profile/trace identities drifted")
    source = manifest.get("source_verification")
    radio = manifest.get("radio_binding_final")
    if not isinstance(source, Mapping) or source.get("verified") is not True or source.get("problems") != []:
        raise EvidenceIdentityError("source verification did not pass")
    if not isinstance(radio, Mapping) or radio.get("verified") is not True or radio.get("problems") != []:
        raise EvidenceIdentityError("final radio sealing did not pass")
    for label, block in (("source", source), ("radio", radio)):
        files = block.get("files")
        if not isinstance(files, Mapping) or not files:
            raise EvidenceSchemaError(f"{label} verification inventory missing")
        for name, item in files.items():
            if not isinstance(item, Mapping) or item.get("matches") is not True:
                raise EvidenceIdentityError(f"{label} source {name} was not verified")
            observed = item.get("observed_sha256")
            expected = item.get("expected_sha256", observed)
            if not _is_digest(observed) or expected != observed:
                raise EvidenceIdentityError(f"{label} source {name} hash disagrees")


def _validate_analysis(analysis: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if analysis != manifest.get("analysis"):
        raise EvidenceIdentityError("manifest and standalone analysis disagree")
    expected = {
        "schema": EXPECTED_ANALYSIS_SCHEMA,
        "contract_id": EXPECTED_CONTRACT_ID,
        "claim_boundary": EXPECTED_CLAIM_BOUNDARY,
        "radio_profile_id": EXPECTED_RADIO_PROFILE_ID,
        "design_sha256": EXPECTED_DESIGN_SHA256,
        "passed": True,
        "policy_input_boundary": EXPECTED_POLICY_BOUNDARY,
    }
    for key, value in expected.items():
        if analysis.get(key) != value:
            raise EvidenceIdentityError(f"analysis {key} drifted")
    gates = analysis.get("aggregate_gates")
    if gates != {
        "all_profile_gates": True,
        "exact_observation_count": True,
        "policy_feature_excludes_hidden_metadata": True,
        "two_registered_profiles": True,
    }:
        raise EvidenceIdentityError("aggregate analysis gates did not all pass")
    profiles = analysis.get("profiles")
    if not isinstance(profiles, Mapping) or set(profiles) != {
        profile for profile, _ in EXPECTED_PROFILES
    }:
        raise EvidenceIdentityError("analysis profile set drifted")
    required_profile_gates = {
        "clock_bridge", "duration2_successors_exact", "exact_decision_grid",
        "fit_mcs_coverage", "gnb_provenance", "mcs_informative",
        "no_cross_partition_transition", "only_table0_new_data_grants",
        "rf_commands_causal_unclamped", "schedule_lag_p99",
        "sender_complete_without_backpressure", "single_ue_rnti",
        "validation_mcs_coverage",
    }
    traces = dict(EXPECTED_PROFILES)
    for profile_id, report in profiles.items():
        if not isinstance(report, Mapping):
            raise EvidenceSchemaError("profile analysis must be a mapping")
        if (
            report.get("profile_id") != profile_id
            or report.get("trace_id") != traces[profile_id]
            or report.get("radio_profile_id") != EXPECTED_RADIO_PROFILE_ID
            or report.get("rows") != 300
            or report.get("duration2_transitions") != 296
            or report.get("duration2_learning_eligible") != 296
            or report.get("reset_required_indices") != [208, 209, 298, 299]
            or report.get("passed") is not True
        ):
            raise EvidenceIdentityError(f"{profile_id} analysis identity/count drifted")
        gates = report.get("gates")
        if not isinstance(gates, Mapping) or set(gates) != required_profile_gates or not all(
            value is True for value in gates.values()
        ):
            raise EvidenceIdentityError(f"{profile_id} profile gates did not all pass")
        partitions = report.get("partitions")
        if not isinstance(partitions, Mapping):
            raise EvidenceSchemaError("profile partition summary missing")
        for partition, count in (("FIT", 210), ("INTERNAL_VALIDATION", 90)):
            row = partitions.get(partition)
            if not isinstance(row, Mapping) or (
                row.get("rows") != count
                or row.get("valid") != count
                or row.get("missing") != 0
                or row.get("stale") != 0
                or row.get("valid_fraction") != 1
            ):
                raise EvidenceIdentityError(f"{profile_id}/{partition} coverage drifted")


def _parse_status(text: str, value: str, where: str) -> PolicyMcsObservationV1:
    try:
        status = McsStatus(text)
    except ValueError as exc:
        raise EvidenceSchemaError(f"{where}: unknown MCS status {text!r}") from exc
    if status is McsStatus.VALID:
        return PolicyMcsObservationV1(status, _exact_int(value, f"{where}.mcs"))
    if value != "":
        raise EvidenceSchemaError(f"{where}: missing/stale MCS must be empty")
    return PolicyMcsObservationV1(status, None)


def _parse_observations(rows: Sequence[Mapping[str, str]]) -> tuple[_VerifiedObservation, ...]:
    if len(rows) != OBSERVATION_COUNT:
        raise EvidenceSchemaError("expected exactly 600 MCS observations")
    result: list[_VerifiedObservation] = []
    seen: set[tuple[str, int]] = set()
    trace_by_profile = dict(EXPECTED_PROFILES)
    profile_order: list[str] = []
    last_schedule: dict[str, int] = {}
    provenance_by_split: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        line = row.get("__line__", "?")
        profile = row["profile_id"]
        if profile not in trace_by_profile or row["trace_id"] != trace_by_profile[profile]:
            raise EvidenceIdentityError(f"observation line {line}: profile/trace drifted")
        if not profile_order or profile_order[-1] != profile:
            if profile in profile_order:
                raise EvidenceIdentityError("profile observation blocks are interleaved")
            profile_order.append(profile)
        index = _exact_int(row["decision_index"], f"line {line}.decision_index")
        key = (profile, index)
        if key in seen:
            raise EvidenceIdentityError(f"duplicate observation {key}")
        seen.add(key)
        partition = _partition_for(index)
        if row["partition"] != partition:
            raise EvidenceIdentityError(f"observation {key} partition drifted")
        scheduled = _exact_int(
            row["scheduled_action_open_monotonic_ns"], f"line {line}.scheduled"
        )
        previous = last_schedule.get(profile)
        if previous is not None and scheduled - previous != PERIOD_NS:
            raise EvidenceIdentityError(f"observation {key} is off the 100-ms grid")
        last_schedule[profile] = scheduled
        policy = _parse_status(
            row["mcs_status"], row["prior_ul_mcs_index"], f"observation {key}"
        )
        try:
            feature = json.loads(row["policy_feature_json"])
        except json.JSONDecodeError as exc:
            raise EvidenceSchemaError(f"observation {key}: policy JSON malformed") from exc
        expected_feature = (
            {"prior_ul_mcs_index": policy.prior_ul_mcs_index}
            if policy.status is McsStatus.VALID else {}
        )
        if feature != expected_feature or not isinstance(feature, dict):
            raise EvidenceIdentityError(f"observation {key}: hidden/imputed policy feature")
        if row["hidden_profile_verifier_only"] != profile:
            raise EvidenceIdentityError(f"observation {key}: hidden verifier profile drifted")
        if row["target_snr_db_verifier_only"] == "":
            raise EvidenceSchemaError(f"observation {key}: target-SNR verifier missing")
        provenance = row["source_provenance_sha256"]
        if policy.status is McsStatus.VALID:
            if not _is_digest(provenance):
                raise EvidenceSchemaError(f"observation {key}: provenance missing")
            source_ns = _exact_int(
                row["source_grant_monotonic_ns"], f"observation {key}.source_ns"
            )
            if not source_ns < scheduled:
                raise EvidenceIdentityError(f"observation {key}: source is not prior")
            if row["source_grant_mcs_table"] != "0" or row["source_grant_round"] != "0":
                raise EvidenceIdentityError(f"observation {key}: not round-0/table-0")
            if row["source_grant_ndi"] != "1":
                raise EvidenceIdentityError(f"observation {key}: not new data")
        else:
            if provenance != "":
                raise EvidenceSchemaError(f"observation {key}: invalid MCS retains provenance")
        provenance_by_split.setdefault((profile, partition), set()).add(provenance)
        result.append(_VerifiedObservation(
            profile_id=profile,
            trace_id=row["trace_id"],
            decision_index=index,
            partition=partition,
            scheduled_ns=scheduled,
            provenance_sha256=provenance,
            policy=policy,
        ))
    if tuple(profile_order) != tuple(profile for profile, _ in EXPECTED_PROFILES):
        raise EvidenceIdentityError("observation profile order drifted")
    if seen != {
        (profile, index)
        for profile, _ in EXPECTED_PROFILES
        for index in range(300)
    }:
        raise EvidenceIdentityError("observation grid is incomplete")
    for profile, _ in EXPECTED_PROFILES:
        if provenance_by_split[(profile, "FIT")] & provenance_by_split[
            (profile, "INTERNAL_VALIDATION")
        ]:
            raise EvidenceIdentityError("FIT/validation source grants overlap")
    return tuple(result)


def _parse_transitions(
    rows: Sequence[Mapping[str, str]],
    observations: Sequence[_VerifiedObservation],
) -> dict[tuple[str, str], tuple[PolicyMcsTransitionV1, ...]]:
    if len(rows) != TRANSITION_COUNT:
        raise EvidenceSchemaError("expected exactly 592 duration-2 transitions")
    by_key = {(row.profile_id, row.decision_index): row for row in observations}
    expected_keys = {
        (profile, current)
        for profile, _ in EXPECTED_PROFILES
        for current in tuple(range(0, 208)) + tuple(range(210, 298))
    }
    seen: set[tuple[str, int]] = set()
    grouped: dict[tuple[str, str], list[PolicyMcsTransitionV1]] = {}
    for row in rows:
        line = row.get("__line__", "?")
        profile = row["profile_id"]
        current_index = _exact_int(row["current_decision_index"], f"line {line}.current")
        successor_index = _exact_int(row["successor_decision_index"], f"line {line}.successor")
        key = (profile, current_index)
        if key not in expected_keys or key in seen:
            raise EvidenceIdentityError(f"foreign/duplicate transition {key}")
        seen.add(key)
        if successor_index != current_index + DURATION_TENSORS:
            raise EvidenceIdentityError(f"transition {key}: wrong successor")
        if row["partition"] != _partition_for(current_index) or (
            _partition_for(successor_index) != row["partition"]
        ):
            raise EvidenceIdentityError(f"transition {key}: partition boundary crossed")
        if _exact_int(row["duration_tensors"], f"transition {key}.duration") != 2:
            raise EvidenceIdentityError(f"transition {key}: duration drifted")
        if _exact_int(row["scheduled_delta_ns"], f"transition {key}.delta") != SUCCESSOR_DELTA_NS:
            raise EvidenceIdentityError(f"transition {key}: scheduled delta drifted")
        current = by_key.get((profile, current_index))
        successor = by_key.get((profile, successor_index))
        if current is None or successor is None:
            raise EvidenceIdentityError(f"transition {key}: observation missing")
        if successor.scheduled_ns - current.scheduled_ns != SUCCESSOR_DELTA_NS:
            raise EvidenceIdentityError(f"transition {key}: real timestamp delta drifted")
        parsed_current = _parse_status(
            row["current_mcs_status"], row["current_prior_ul_mcs_index"],
            f"transition {key}.current",
        )
        parsed_successor = _parse_status(
            row["successor_mcs_status"], row["successor_prior_ul_mcs_index"],
            f"transition {key}.successor",
        )
        if parsed_current != current.policy or parsed_successor != successor.policy:
            raise EvidenceIdentityError(f"transition {key}: observation values disagree")
        eligible = _exact_bool(row["learning_eligible"], f"transition {key}.eligible")
        if _exact_bool(row["reset_required_after_current"], f"transition {key}.reset"):
            raise EvidenceIdentityError(f"transition {key}: unexpected reset row")
        transition = PolicyMcsTransitionV1(
            current=parsed_current,
            successor=parsed_successor,
            duration_tensors=2,
            learning_eligible=eligible,
        )
        grouped.setdefault((profile, row["partition"]), []).append(transition)
    if seen != expected_keys:
        raise EvidenceIdentityError("duration-2 transition grid is incomplete")
    return {key: tuple(value) for key, value in grouped.items()}


def _build_sequences(
    observations: Sequence[_VerifiedObservation],
    transitions: Mapping[tuple[str, str], tuple[PolicyMcsTransitionV1, ...]],
    partition: str,
) -> tuple[PolicyMcsSequenceV1, ...]:
    sequences = []
    for profile, _ in EXPECTED_PROFILES:
        policy_observations = tuple(
            row.policy
            for row in observations
            if row.profile_id == profile and row.partition == partition
        )
        sequences.append(PolicyMcsSequenceV1(
            observations=policy_observations,
            transitions=transitions[(profile, partition)],
        ))
    return tuple(sequences)


def _canonical_evidence_payload(
    *,
    fit: Sequence[PolicyMcsSequenceV1],
    validation: Sequence[PolicyMcsSequenceV1],
    inventory_sha256: str,
) -> dict[str, Any]:
    return {
        "claim_boundary": EXPECTED_CLAIM_BOUNDARY,
        "design_sha256": EXPECTED_DESIGN_SHA256,
        "fit_sequences": [item.to_canonical_dict() for item in fit],
        "internal_validation_sequences": [
            item.to_canonical_dict() for item in validation
        ],
        "radio_profile_id": EXPECTED_RADIO_PROFILE_ID,
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "source_hashes": {
            "analysis": EXPECTED_ANALYSIS_SHA256,
            "inventory": inventory_sha256,
            "manifest": EXPECTED_MANIFEST_SHA256,
            "observations": EXPECTED_OBSERVATIONS_SHA256,
            "success_terminal": EXPECTED_SUCCESS_TERMINAL_SHA256,
            "transitions": EXPECTED_TRANSITIONS_SHA256,
        },
    }


def load_dynamic_mcs_273prb_evidence(
    *, repository_root: Optional[Path] = None,
) -> DynamicMcs273PrbEvidenceV1:
    """Load, rehash, cross-check, and identity-erase the passed capture."""
    root = _repo_root() if repository_root is None else Path(repository_root)
    run_dir = root / SOURCE_RUN_RELATIVE_PATH
    paths = {
        "manifest": run_dir / MANIFEST_NAME,
        "terminal": run_dir / TERMINAL_NAME,
        "analysis": run_dir / ANALYSIS_NAME,
        "observations": run_dir / OBSERVATIONS_NAME,
        "transitions": run_dir / TRANSITIONS_NAME,
    }
    for label, expected in (
        ("manifest", EXPECTED_MANIFEST_SHA256),
        ("terminal", EXPECTED_SUCCESS_TERMINAL_SHA256),
        ("analysis", EXPECTED_ANALYSIS_SHA256),
        ("observations", EXPECTED_OBSERVATIONS_SHA256),
        ("transitions", EXPECTED_TRANSITIONS_SHA256),
    ):
        _verify_pinned(paths[label], expected)

    manifest = _load_json(paths["manifest"])
    terminal = _load_json(paths["terminal"])
    analysis = _load_json(paths["analysis"])
    if not isinstance(manifest, Mapping) or not isinstance(terminal, Mapping) or not isinstance(analysis, Mapping):
        raise EvidenceSchemaError("manifest, terminal and analysis must be JSON objects")
    _validate_manifest(manifest)
    if set(terminal) != {"terminal", "manifest_sha256", "utc"} or (
        terminal.get("terminal") != EXPECTED_STATUS
        or terminal.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
    ):
        raise EvidenceIdentityError("success terminal does not bind the manifest")
    _validate_analysis(analysis, manifest)
    inventory_sha256 = _verify_inventory(run_dir, manifest)

    observation_rows = _read_csv(paths["observations"], OBSERVATION_FIELDS)
    transition_rows = _read_csv(paths["transitions"], TRANSITION_FIELDS)
    verified_observations = _parse_observations(observation_rows)
    verified_transitions = _parse_transitions(transition_rows, verified_observations)
    fit = _build_sequences(verified_observations, verified_transitions, "FIT")
    validation = _build_sequences(
        verified_observations, verified_transitions, "INTERNAL_VALIDATION"
    )
    if sum(len(item.observations) for item in fit) != FIT_OBSERVATION_COUNT:
        raise EvidenceSchemaError("FIT observation count drifted")
    if sum(len(item.observations) for item in validation) != VALIDATION_OBSERVATION_COUNT:
        raise EvidenceSchemaError("validation observation count drifted")
    if sum(len(item.transitions) for item in fit) != FIT_TRANSITION_COUNT:
        raise EvidenceSchemaError("FIT transition count drifted")
    if sum(len(item.transitions) for item in validation) != VALIDATION_TRANSITION_COUNT:
        raise EvidenceSchemaError("validation transition count drifted")

    canonical = _canonical_sha256(_canonical_evidence_payload(
        fit=fit,
        validation=validation,
        inventory_sha256=inventory_sha256,
    ))
    if canonical != EXPECTED_CANONICAL_EVIDENCE_SHA256:
        raise EvidenceHashMismatch(
            f"canonical evidence {canonical} != {EXPECTED_CANONICAL_EVIDENCE_SHA256}"
        )
    return DynamicMcs273PrbEvidenceV1(
        fit_sequences=fit,
        internal_validation_sequences=validation,
        source_manifest_sha256=EXPECTED_MANIFEST_SHA256,
        source_terminal_sha256=EXPECTED_SUCCESS_TERMINAL_SHA256,
        source_analysis_sha256=EXPECTED_ANALYSIS_SHA256,
        source_observations_sha256=EXPECTED_OBSERVATIONS_SHA256,
        source_transitions_sha256=EXPECTED_TRANSITIONS_SHA256,
        source_inventory_sha256=inventory_sha256,
        canonical_evidence_sha256=canonical,
    )
