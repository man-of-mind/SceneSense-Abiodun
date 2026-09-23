"""Create-only per-frame evidence schema and writer for the live pilot.

Create-only means exactly that.  The session directory is created with
``mkdir(exist_ok=False)``, every file is opened ``O_CREAT | O_EXCL``, and the
JSONL stream is append-only.  There is no overwrite flag, no ``replace=True``
and no ``force``.  A prior campaign or a prior failed run cannot be clobbered
by this module, including by passing it the wrong path twice.

The frame schema is fixed here in Phase 1 and populated progressively by the
later phases.  Every field a phase has not yet produced is written as an
explicit ``None`` with a sibling status string, never as a plausible zero.  The
distinction matters: a missing person-class metric and a person recall of zero
are different scientific statements, and only one of them is true when no
person was in frame.

Field groups, in the order the runtime produces them:

``identity``
    session, frame, tensor, decision, ticket and action-hold identities.
``decision``
    whether this frame opened a decision or reused a held action,
    ``reward_requested``, the 31 features, the raw SI/P40/SNR/MCS/BSR values
    with their source timestamps and derived ages, the training-support audit,
    the proposed mode/``q``, the exact executed identity and the policy
    inference time.
``transport``
    payload bytes, datagram count, and the send / reassembly / edge-receipt /
    tail / evaluation / ACK / timeout timestamps, plus the reassembly,
    admission, timeout, supersession and late-orphan outcomes.
``quality``
    vehicle and person segmentation IoU; vehicle and person localization
    recall, XY error and footprint IoU; a validity/status string for every
    class metric; and ``Q_seg`` / ``Q_loc`` / ``Q_perc`` with the exact
    realized-reward breakdown.
``map``
    direct publication and install timestamps and the derived AoI, kept as a
    separate diagnostic that is explicitly *not* on the reward-ACK path.

Writing evidence performs no network, CARLA, OAI or CUDA work.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from . import pilot_contract as contract

__all__ = [
    "EvidenceError",
    "FRAME_FIELD_GROUPS",
    "PilotEvidenceWriter",
    "blank_frame_record",
    "validate_frame_record",
]


class EvidenceError(RuntimeError):
    """An evidence directory, file or record violated the create-only contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


# --------------------------------------------------------------------------- #
# Frame schema
# --------------------------------------------------------------------------- #

_IDENTITY_FIELDS: Tuple[str, ...] = (
    "session_uuid",
    "controller_lineage_uuid",
    "run_id",
    "cell_id",
    "stream_id",
    "carla_frame_id",
    "tensor_seq",
    "decision_seq",
    "ticket_id",
    "action_hold_id",
    "capture_timestamp_ns",
)

_DECISION_FIELDS: Tuple[str, ...] = (
    "opened_decision",
    "reused_held_action",
    "reward_requested",
    "policy_features",
    "policy_feature_order_sha256",
    "state_sha256",
    "features_sha256",
    "normalization_spec_sha256",
    "freshness_policy_sha256",
    "raw_camera_si",
    "raw_radar_p40",
    "raw_achieved_snr_db",
    "raw_mcs_index",
    "raw_bsr_bytes",
    "scene_measured_ns",
    "snr_measured_ns",
    "mcs_measured_ns",
    "bsr_measured_ns",
    "observed_ns",
    "scene_age_ns",
    "snr_age_ns",
    "mcs_age_ns",
    "bsr_age_ns",
    "radio_evidence_path",
    "radio_provenance_status",
    "support_audit",
    "proposed_mode_id",
    "proposed_q",
    "executed_mode_id",
    "executed_q_e4",
    "executed_action_id",
    "executed_profile_id",
    "executed_measurement_status",
    "executed_keep_count",
    "executed_drop_count",
    "execution_bundle_sha256",
    "execution_identity_sha256",
    "policy_inference_ns",
    "actor_state_sha256",
)

_TRANSPORT_FIELDS: Tuple[str, ...] = (
    "payload_bytes",
    "datagram_count",
    "send_started_ns",
    "send_finished_ns",
    "reassembly_completed_ns",
    "edge_receipt_ns",
    "tail_completed_ns",
    "evaluation_enqueued_ns",
    "evaluation_started_ns",
    "evaluation_completed_ns",
    "ack_emitted_ns",
    "ack_received_ns",
    "ticket_deadline_ns",
    "timeout_fired_ns",
    "reassembly_outcome",
    "admission_outcome",
    "timeout_outcome",
    "supersession_outcome",
    "late_orphan_outcome",
    "duplicate_outcome",
    "terminal_class",
)

_QUALITY_FIELDS: Tuple[str, ...] = (
    "seg_vehicle_iou",
    "seg_vehicle_status",
    "seg_person_iou",
    "seg_person_status",
    "loc_vehicle_recall",
    "loc_vehicle_xy_error_m",
    "loc_vehicle_footprint_iou",
    "loc_vehicle_status",
    "loc_person_recall",
    "loc_person_xy_error_m",
    "loc_person_footprint_iou",
    "loc_person_status",
    "q_seg",
    "q_loc",
    "q_perc",
    "realized_latency_ms",
    "realized_reward",
    "realized_reward_breakdown",
    "reward_learning_eligible",
    "reward_spec_sha256",
)

_MAP_FIELDS: Tuple[str, ...] = (
    "map_publication_ns",
    "map_install_ns",
    "map_aoi_ms",
    "map_install_status",
    "map_path_disclosure",
)

FRAME_FIELD_GROUPS: Mapping[str, Tuple[str, ...]] = MappingProxyType(
    {
        "identity": _IDENTITY_FIELDS,
        "decision": _DECISION_FIELDS,
        "transport": _TRANSPORT_FIELDS,
        "quality": _QUALITY_FIELDS,
        "map": _MAP_FIELDS,
    }
)

_ALL_FRAME_FIELDS: Tuple[str, ...] = tuple(
    name for group in FRAME_FIELD_GROUPS.values() for name in group
)

_RESERVED_FIELDS: Tuple[str, ...] = (
    "record",
    "pilot_label",
    "pilot_contract_sha256",
    "schema",
)

#: Every class metric carries a status; ``NOT_MEASURED`` is the Phase-1 default
#: and ``NO_CLASS_SUPPORT`` is what a genuinely absent class must record.  A
#: zero is never a stand-in for either.
_DEFAULT_STATUS = "NOT_MEASURED"
_MAP_PATH_DISCLOSURE = (
    "DIRECT_EDGE_TO_MAP_PUBLICATION;NOT_ON_THE_REWARD_ACK_CRITICAL_PATH;"
    "AOI_IS_A_SEPARATE_DIAGNOSTIC"
)

_STATUS_FIELDS: Tuple[str, ...] = (
    "seg_vehicle_status",
    "seg_person_status",
    "loc_vehicle_status",
    "loc_person_status",
)


def blank_frame_record() -> Dict[str, Any]:
    """A fully populated frame record with every measurement explicitly absent.

    Callers overwrite the fields their phase actually produced.  Because the
    skeleton starts complete, a forgotten field is written as ``None`` rather
    than silently omitted from the record.
    """
    record: Dict[str, Any] = {name: None for name in _ALL_FRAME_FIELDS}
    for name in _STATUS_FIELDS:
        record[name] = _DEFAULT_STATUS
    record["map_path_disclosure"] = _MAP_PATH_DISCLOSURE
    record["record"] = contract.EVIDENCE_FRAME_SCHEMA_ID
    record["schema"] = contract.EVIDENCE_FRAME_SCHEMA_ID
    record["pilot_label"] = contract.PILOT_LABEL
    record["pilot_contract_sha256"] = contract.PILOT_CONTRACT_SHA256
    return record


def validate_frame_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Fail closed on a record with a missing, extra or mislabelled field."""
    _require(isinstance(record, Mapping), "a frame record must be a mapping")
    expected = set(_ALL_FRAME_FIELDS) | set(_RESERVED_FIELDS)
    observed = set(record)
    _require(
        observed == expected,
        f"frame record field drift: missing {sorted(expected - observed)}, "
        f"unexpected {sorted(observed - expected)}",
    )
    _require(
        record["record"] == contract.EVIDENCE_FRAME_SCHEMA_ID
        and record["schema"] == contract.EVIDENCE_FRAME_SCHEMA_ID,
        "frame record schema drift",
    )
    _require(
        record["pilot_label"] == contract.PILOT_LABEL,
        "frame record is missing the mandatory pilot label",
    )
    _require(
        record["pilot_contract_sha256"] == contract.PILOT_CONTRACT_SHA256,
        "frame record was produced under a different pilot binding",
    )
    for name in _STATUS_FIELDS:
        value = record[name]
        _require(
            isinstance(value, str) and value,
            f"{name} must be an explicit status string, never null: {value!r}",
        )
    return dict(record)


# --------------------------------------------------------------------------- #
# Create-only writer
# --------------------------------------------------------------------------- #


class PilotEvidenceWriter:
    """Append-only JSONL frame stream plus create-only session documents.

    The writer owns its directory.  Constructing it creates that directory and
    fails if it already exists, which is what makes "never overwrite a prior
    campaign or failed run" a property of the code rather than of operator
    discipline.
    """

    FRAME_FILENAME = "frames.jsonl"
    SESSION_FILENAME = "session.json"
    TERMINAL_FILENAME = "SESSION_COMPLETE.json"

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)
        try:
            self._directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise EvidenceError(
                f"evidence directory {self._directory} already exists; pilot "
                f"evidence is create-only and never overwrites a prior "
                f"campaign or failed run"
            ) from exc
        self._frames_path = self._directory / self.FRAME_FILENAME
        self._handle = self._exclusive_open(self._frames_path)
        self._frame_count = 0
        self._closed = False

    @staticmethod
    def _exclusive_open(path: Path):
        try:
            descriptor = os.open(
                path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
        except FileExistsError as exc:
            raise EvidenceError(
                f"{path} already exists; refusing to overwrite evidence"
            ) from exc
        return os.fdopen(descriptor, "w", encoding="utf-8")

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def frames_path(self) -> Path:
        return self._frames_path

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def write_session(self, document: Mapping[str, Any]) -> Path:
        """Write the one-time session header.  A second call fails closed."""
        payload = dict(document)
        payload.setdefault("record", contract.EVIDENCE_SESSION_SCHEMA_ID)
        payload["pilot_label"] = contract.PILOT_LABEL
        payload["pilot_contract_sha256"] = contract.PILOT_CONTRACT_SHA256
        payload["pilot_phase"] = contract.PILOT_PHASE
        payload["pilot_scope"] = contract.PILOT_SCOPE
        payload["training_support_limitation"] = contract.TRAINING_SUPPORT_LIMITATION
        return self._write_json(self.SESSION_FILENAME, payload)

    def write_frame(self, record: Mapping[str, Any]) -> None:
        """Append one validated frame record to the JSONL stream."""
        _require(not self._closed, "the evidence writer is already closed")
        validated = validate_frame_record(record)
        self._handle.write(
            contract.canonical_json_bytes(validated).decode("utf-8") + "\n"
        )
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._frame_count += 1

    def close(self, terminal: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
        """Flush, close and optionally write the create-only terminal marker."""
        if not self._closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._closed = True
        if terminal is None:
            return None
        payload = dict(terminal)
        payload["frame_count"] = self._frame_count
        payload.setdefault("record", "splitfusion.live_route_b_pilot_terminal.v1")
        payload["pilot_label"] = contract.PILOT_LABEL
        payload["pilot_contract_sha256"] = contract.PILOT_CONTRACT_SHA256
        return self._write_json(self.TERMINAL_FILENAME, payload)

    def __enter__(self) -> "PilotEvidenceWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if not self._closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._closed = True

    def _write_json(self, filename: str, payload: Mapping[str, Any]) -> Path:
        path = self._directory / filename
        document = dict(payload)
        document["document_sha256"] = contract.canonical_sha256(document)
        handle = self._exclusive_open(path)
        try:
            handle.write(
                contract.canonical_json_bytes(document).decode("utf-8") + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        return path
