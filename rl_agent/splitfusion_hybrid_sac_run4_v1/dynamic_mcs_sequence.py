"""Hash-bound, causal UE UL-MCS traces for Run-4 offline training.

The retained 2026-09-23 qualification contains the UE-decoded UL grant that
the policy can observe.  This module binds that evidence without turning the
profile label, RFsim command, gNB SNR, action, payload, or queue into a policy
feature.  It provides only the latest *strictly prior* UE MCS and its
exogenous successor inside one predeclared contiguous segment.

The source has one measured realization per profile.  MID_VARIABLE and
FADE_RECOVERY are therefore useful measured training traces, not evidence of
cross-realization generalization.  The first 174 causal decision instants of
each profile are frozen as fit and the final 75 as internal validation.  A
transition never crosses that boundary or a profile boundary.

Import is side-effect free.  Loading is explicit, CPU-only, and fails closed
on any source, hash, identity, filter, clock, or verifier discrepancy.
Production remains unavailable until a later composite verifier pins this
module's binding together with every other Run-4 prerequisite.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract

__all__ = [
    "DynamicMcsError",
    "EvidenceHashMismatch",
    "EvidenceSchemaError",
    "EvidenceIdentityError",
    "CausalSelectionError",
    "SegmentBoundaryResetRequired",
    "ProductionDynamicMcsUnavailable",
    "McsValidity",
    "SequencePartition",
    "RawUeGrantV1",
    "CausalMcsObservationV1",
    "DynamicMcsTransitionV1",
    "FrozenSegmentPlanV1",
    "DynamicMcsSegmentV1",
    "McsVerifierStatsV1",
    "DynamicMcsEvidenceV1",
    "SOURCE_RUN_RELATIVE_PATH",
    "EXPECTED_SOURCE_MANIFEST_SHA256",
    "EXPECTED_SOURCE_CONFIG_SHA256",
    "EXPECTED_UE_RNTI",
    "DECISION_PERIOD_NS",
    "FIT_DECISION_COUNT",
    "INTERNAL_VALIDATION_DECISION_COUNT",
    "ONE_REALIZATION_LIMITATION",
    "REGISTERED_COMPOSITE_DYNAMIC_MCS_BINDING_SHA256",
    "predeclare_segment_plans",
    "select_latest_strictly_prior_mcs",
    "load_dynamic_mcs_evidence",
    "require_production_dynamic_mcs_binding",
]


SCHEMA_ID = "splitfusion_run4_dynamic_ue_mcs_sequence_v1"
SCHEMA_VERSION = 1

SOURCE_RUN_RELATIVE_PATH = Path(
    "rl_agent/experiments/ue_snr_bridge_qualification_v1/20260923_220340"
)
SOURCE_CONFIG_RELATIVE_PATH = Path(
    "rl_agent/ue_snr_bridge_qualification_v1/config_v1.json"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "43eb4fdcb478a9df3ae546815586fae95889ca4f5a566bb5996022e182599510"
)
EXPECTED_SOURCE_CONFIG_SHA256 = (
    "6b93aa01fe1561760d8a4724ba995b105b49b7624130fbfeafe76eb2c632ff37"
)
EXPECTED_UE_RNTI = 33457
EXPECTED_IMSI = "001010000000001"
EXPECTED_UE_INTERFACE = "oaitun_ue1"
EXPECTED_UE_IPV4 = "10.0.0.2"

UE_DCI_RELATIVE_PATH = Path("ttracer/ue/csv/NRUE_MAC_DCI_GRANT.csv")
GNB_MCS_RELATIVE_PATH = Path("ttracer/gnb/csv/GNB_MAC_UL_MCS_DECISION.csv")
CLOCK_ANCHORS_RELATIVE_PATH = Path("clock_anchors.json")
UE_IDENTITY_RELATIVE_PATH = Path("ue_network_identity.json")

EXPECTED_REQUIRED_FILE_SHA256 = {
    "clock_anchors.json": (
        "8cddf330141de96556a6e413687af7c943c8f97d2a40d3d5434758bcac4e7be6"
    ),
    "command_log.csv": (
        "41da01a264677140b39325ae2509a7012aeb8d393ffb8e1d23fc2822c261cbdc"
    ),
    "runtime/config_hashes.json": (
        "ad15730211075ee6c4816b532db1b89731e12bb3642b04fe039bbd1d6d7245e0"
    ),
    "runtime/effective_gnb.conf": (
        "a3c2140dceb28aee34865ddd28fa6ffeae69abd8bfd59dfdb97083847d981d22"
    ),
    "runtime/effective_ue.conf": (
        "ab06afcc1cfc2bb777fa35973811aaad655032f6541e1b4565541cac9cc04f97"
    ),
    str(GNB_MCS_RELATIVE_PATH): (
        "aaf7c6a68d597dab9e31c76abe2ad221c078a180bfca2b1ab7a30cc8060017a1"
    ),
    str(UE_DCI_RELATIVE_PATH): (
        "6064f055cf3a3f63e71fa2f68201cbe81f234cc3886082532415df0ced2e72f0"
    ),
    str(UE_IDENTITY_RELATIVE_PATH): (
        "3c848129c0eecc679ecdadbc86c916864c096a1239cc0ebf103561f940e9d768"
    ),
}

EXPECTED_PROFILE_TRACE_IDS = {
    "MID_VARIABLE": "GM_V2_MID_VARIABLE_SEED_2026082102",
    "FADE_RECOVERY": "GM_V2_FADE_RECOVERY_SEED_2026082104",
}
EXPECTED_PROFILE_ORDER = (
    "FAVORABLE_STABLE",
    "MID_VARIABLE",
    "ADVERSE_STABLE",
    "FADE_RECOVERY",
)

DECISION_PERIOD_NS = 100_000_000
# Decision ordinal zero is the measured-window boundary.  Starting at one
# ensures that a selected grant also belongs to the measured window rather
# than silently borrowing a warm-up or previous-profile grant.
FIRST_DECISION_ORDINAL = 1
LAST_DECISION_ORDINAL = 249
FIT_DECISION_COUNT = 174
INTERNAL_VALIDATION_DECISION_COUNT = 75
FIT_LAST_ORDINAL = FIT_DECISION_COUNT
VALIDATION_FIRST_ORDINAL = FIT_LAST_ORDINAL + 1

UE_GNB_JOIN_TOLERANCE_NS = 5_000_000
MINIMUM_UNIQUE_GNB_JOIN_FRACTION = 0.99
ONE_REALIZATION_LIMITATION = (
    "ONE_RETAINED_25_SECOND_REALIZATION_PER_DYNAMIC_PROFILE; "
    "FIT_AND_INTERNAL_VALIDATION_ARE_CONTIGUOUS_SEGMENTS_OF_THE_SAME_RUN; "
    "NO_CROSS_REALIZATION_GENERALIZATION_CLAIM"
)
EVIDENCE_CLASS = "MEASURED_UE_DCI_DYNAMIC_TRACE_OFFLINE_TRAINING_ONLY"

# A later reviewed composite verifier may pin the canonical evidence binding.
# Leaving this unset is deliberate: this module can support offline fitting,
# but it cannot authorize a production state provider by itself.
REGISTERED_COMPOSITE_DYNAMIC_MCS_BINDING_SHA256: Optional[str] = None


UE_DCI_HEADER = (
    "time", "direction", "dci_format", "rnti_type", "rnti", "dci_frame",
    "dci_slot", "sched_frame", "sched_slot", "mcs", "mcs_table",
    "rb_start", "rb_size", "start_symbol", "nr_symbols", "tbs",
    "harq_pid", "ndi", "rv", "round", "qam_mod_order",
    "target_code_rate", "tpc", "n_cce", "N_cce",
)
GNB_MCS_HEADER = (
    "time", "rnti", "frame", "slot", "sched_frame", "sched_slot",
    "avg_snr_x10", "mcs_table", "ul_bler_mcs_before", "selected_mcs",
    "pre_phr_mcs", "post_phr_mcs", "final_mcs", "estimated_ul_buffer",
    "sched_ul_bytes", "B", "min_rb", "available_rb_before",
    "available_rb_after", "ph", "pcmax", "rb_size_final", "tbs_final",
    "force_ul_mcs",
)


class DynamicMcsError(ValueError):
    """Base class for dynamic-MCS binding failures."""


class EvidenceHashMismatch(DynamicMcsError):
    """A retained input no longer matches its frozen digest."""


class EvidenceSchemaError(DynamicMcsError):
    """An input has an unexpected schema or unsupported value."""


class EvidenceIdentityError(DynamicMcsError):
    """UE, RNTI, profile, grant, or verifier identity is contradictory."""


class CausalSelectionError(DynamicMcsError):
    """A causal sample or transition request is invalid."""


class SegmentBoundaryResetRequired(CausalSelectionError):
    """The caller attempted to cross a frozen episode boundary."""


class ProductionDynamicMcsUnavailable(DynamicMcsError):
    """No reviewed composite verifier currently authorizes production use."""


class McsValidity(str, Enum):
    VALID = "VALID"
    MISSING = "MISSING"
    STALE = "STALE"


class SequencePartition(str, Enum):
    FIT = "FIT"
    INTERNAL_VALIDATION = "INTERNAL_VALIDATION"


def _transition_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CausalSelectionError(
            f"{name} must be an exact int >= {minimum}"
        )
    return value


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
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise EvidenceHashMismatch(f"cannot read required evidence {path}") from exc
    return digest.hexdigest()


def _exact_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise EvidenceSchemaError(f"{name} must be an exact int >= {minimum}")
    return value


def _csv_int(value: str, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, str) or value == "":
        raise EvidenceSchemaError(f"{name} is missing")
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise EvidenceSchemaError(f"{name} is not a base-10 integer") from exc
    if str(parsed) != value or parsed < minimum:
        raise EvidenceSchemaError(f"{name} is not a canonical integer >= {minimum}")
    return parsed


def _load_json(path: Path) -> Any:
    def no_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceSchemaError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=no_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceSchemaError(f"cannot parse required JSON {path}") from exc


def _read_exact_csv(path: Path, expected: Sequence[str]) -> Tuple[Dict[str, str], ...]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(expected):
                raise EvidenceSchemaError(f"{path}: unexpected CSV header")
            rows = []
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise EvidenceSchemaError(
                        f"{path}: malformed row {row_number}"
                    )
                row["__row_number__"] = str(row_number)
                rows.append(row)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise EvidenceSchemaError(f"cannot parse required CSV {path}") from exc
    return tuple(rows)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tracer_clock(reference_anchor: Mapping[str, Any]):
    try:
        local = datetime.fromisoformat(reference_anchor["local_iso"])
        wall_ns = _exact_int(reference_anchor["wall_ns"], "anchor.wall_ns")
        offset_s = int(reference_anchor["utc_offset_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceSchemaError("clock anchor is malformed") from exc
    if local.tzinfo is None or int(local.utcoffset().total_seconds()) != offset_s:
        raise EvidenceSchemaError("clock anchor timezone/offset disagree")
    if abs(int(local.timestamp() * 1_000_000_000) - wall_ns) > 1_000:
        raise EvidenceSchemaError("clock anchor local_iso/wall_ns disagree")
    midnight = datetime.combine(local.date(), time(), tzinfo=local.tzinfo)
    midnight_ns = int(midnight.timestamp()) * 1_000_000_000

    def convert(text: str) -> int:
        try:
            parsed = datetime.strptime(text.strip(), "%H:%M:%S.%f").time()
        except (AttributeError, ValueError) as exc:
            raise EvidenceSchemaError(f"invalid tracer timestamp {text!r}") from exc
        within_day = (
            (parsed.hour * 3600 + parsed.minute * 60 + parsed.second)
            * 1_000_000_000
            + parsed.microsecond * 1_000
        )
        base = midnight_ns + within_day
        return min(
            (base - 86_400_000_000_000, base, base + 86_400_000_000_000),
            key=lambda candidate: abs(candidate - wall_ns),
        )

    return convert


@dataclass(frozen=True, slots=True)
class RawUeGrantV1:
    """One eligible UE-decoded, table-0, new-data UL grant."""

    source_timestamp_ns: int
    source_row_number: int
    rnti: int
    dci_frame: int
    dci_slot: int
    sched_frame: int
    sched_slot: int
    mcs_index: int
    harq_pid: int
    source_identity_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "source_timestamp_ns", "source_row_number", "rnti", "dci_frame",
            "dci_slot", "sched_frame", "sched_slot", "mcs_index", "harq_pid",
        ):
            _exact_int(getattr(self, name), name)
        if self.rnti != EXPECTED_UE_RNTI:
            raise EvidenceIdentityError(f"unexpected UE RNTI {self.rnti}")
        if not contract.UL_MCS_INDEX_MIN <= self.mcs_index <= contract.UL_MCS_INDEX_MAX:
            raise EvidenceSchemaError("eligible UE MCS is outside table-0 support")
        if (
            len(self.source_identity_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_identity_sha256)
        ):
            raise EvidenceIdentityError("grant identity must be a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class CausalMcsObservationV1:
    """One decision-time MCS; invalidity is never represented as MCS zero."""

    decision_ordinal: int
    decision_timestamp_ns: int
    validity: McsValidity
    mcs_index: Optional[int]
    source_timestamp_ns: Optional[int]
    age_ns: Optional[int]
    source_identity_sha256: Optional[str]
    invalid_reason: Optional[str]

    def __post_init__(self) -> None:
        _exact_int(self.decision_ordinal, "decision_ordinal")
        _exact_int(self.decision_timestamp_ns, "decision_timestamp_ns")
        if not isinstance(self.validity, McsValidity):
            raise CausalSelectionError("validity must be McsValidity")
        if self.validity is McsValidity.VALID:
            if type(self.mcs_index) is not int or not (
                contract.UL_MCS_INDEX_MIN
                <= self.mcs_index
                <= contract.UL_MCS_INDEX_MAX
            ):
                raise CausalSelectionError("valid MCS must be an in-range exact int")
            if type(self.source_timestamp_ns) is not int:
                raise CausalSelectionError("valid MCS requires source timestamp")
            if not self.source_timestamp_ns < self.decision_timestamp_ns:
                raise CausalSelectionError("MCS source must be strictly prior")
            if type(self.age_ns) is not int or self.age_ns != (
                self.decision_timestamp_ns - self.source_timestamp_ns
            ):
                raise CausalSelectionError("MCS age is inconsistent")
            if not isinstance(self.source_identity_sha256, str):
                raise CausalSelectionError("valid MCS requires source identity")
            if self.invalid_reason is not None:
                raise CausalSelectionError("valid MCS cannot have invalid_reason")
        else:
            if self.mcs_index is not None or self.source_identity_sha256 is not None:
                raise CausalSelectionError(
                    "missing/stale MCS must not retain a numeric value or identity"
                )
            if not isinstance(self.invalid_reason, str) or not self.invalid_reason:
                raise CausalSelectionError("invalid MCS requires an explicit reason")
            if self.validity is McsValidity.MISSING:
                if self.source_timestamp_ns is not None or self.age_ns is not None:
                    raise CausalSelectionError("missing MCS cannot invent source timing")
            elif (
                type(self.source_timestamp_ns) is not int
                or type(self.age_ns) is not int
                or self.age_ns != self.decision_timestamp_ns - self.source_timestamp_ns
            ):
                raise CausalSelectionError("stale MCS requires rejected-source timing")

    def policy_feature_dict(self) -> Dict[str, int]:
        """Return only the deployable scalar; identity labels never enter state."""
        if self.validity is not McsValidity.VALID:
            raise CausalSelectionError(
                "invalid MCS requires external fallback, not a numeric policy feature"
            )
        return {"prior_ul_mcs_index": int(self.mcs_index)}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "age_ns": self.age_ns,
            "decision_ordinal": self.decision_ordinal,
            "decision_timestamp_ns": self.decision_timestamp_ns,
            "invalid_reason": self.invalid_reason,
            "mcs_index": self.mcs_index,
            "source_identity_sha256": self.source_identity_sha256,
            "source_timestamp_ns": self.source_timestamp_ns,
            "validity": self.validity.value,
        }


def select_latest_strictly_prior_mcs(
    grants: Sequence[RawUeGrantV1],
    *,
    decision_ordinal: int,
    decision_timestamp_ns: int,
    maximum_age_ns: int,
) -> CausalMcsObservationV1:
    """Select one causal grant without imputation or forward filling."""
    _exact_int(decision_ordinal, "decision_ordinal")
    _exact_int(decision_timestamp_ns, "decision_timestamp_ns")
    _exact_int(maximum_age_ns, "maximum_age_ns", minimum=1)
    if any(type(item) is not RawUeGrantV1 for item in grants):
        raise CausalSelectionError("grants must contain exact RawUeGrantV1 records")
    times = tuple(item.source_timestamp_ns for item in grants)
    if any(left >= right for left, right in zip(times, times[1:])):
        raise CausalSelectionError("grant timestamps must be strictly increasing")
    index = bisect.bisect_left(times, decision_timestamp_ns) - 1
    if index < 0:
        return CausalMcsObservationV1(
            decision_ordinal=decision_ordinal,
            decision_timestamp_ns=decision_timestamp_ns,
            validity=McsValidity.MISSING,
            mcs_index=None,
            source_timestamp_ns=None,
            age_ns=None,
            source_identity_sha256=None,
            invalid_reason="NO_ELIGIBLE_GRANT_STRICTLY_PRIOR_IN_PROFILE_WINDOW",
        )
    selected = grants[index]
    age = decision_timestamp_ns - selected.source_timestamp_ns
    if age > maximum_age_ns:
        return CausalMcsObservationV1(
            decision_ordinal=decision_ordinal,
            decision_timestamp_ns=decision_timestamp_ns,
            validity=McsValidity.STALE,
            mcs_index=None,
            source_timestamp_ns=selected.source_timestamp_ns,
            age_ns=age,
            source_identity_sha256=None,
            invalid_reason="LATEST_STRICTLY_PRIOR_GRANT_EXCEEDS_CALLER_MAXIMUM_AGE",
        )
    return CausalMcsObservationV1(
        decision_ordinal=decision_ordinal,
        decision_timestamp_ns=decision_timestamp_ns,
        validity=McsValidity.VALID,
        mcs_index=selected.mcs_index,
        source_timestamp_ns=selected.source_timestamp_ns,
        age_ns=age,
        source_identity_sha256=selected.source_identity_sha256,
        invalid_reason=None,
    )


@dataclass(frozen=True, slots=True)
class FrozenSegmentPlanV1:
    """Outcome-independent, contiguous split declared from manifest timing."""

    source_profile_id: str
    source_trace_id: str
    partition: SequencePartition
    first_decision_ordinal: int
    last_decision_ordinal: int
    first_decision_timestamp_ns: int
    last_decision_timestamp_ns: int
    segment_identity_sha256: str

    @property
    def decision_count(self) -> int:
        return self.last_decision_ordinal - self.first_decision_ordinal + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "first_decision_ordinal": self.first_decision_ordinal,
            "first_decision_timestamp_ns": self.first_decision_timestamp_ns,
            "last_decision_ordinal": self.last_decision_ordinal,
            "last_decision_timestamp_ns": self.last_decision_timestamp_ns,
            "partition": self.partition.value,
            "segment_identity_sha256": self.segment_identity_sha256,
            "source_profile_id": self.source_profile_id,
            "source_trace_id": self.source_trace_id,
        }


def _segment_plan(
    profile: Mapping[str, Any],
    partition: SequencePartition,
    first: int,
    last: int,
) -> FrozenSegmentPlanV1:
    profile_id = profile.get("profile_id")
    trace_id = profile.get("trace_id")
    if EXPECTED_PROFILE_TRACE_IDS.get(profile_id) != trace_id:
        raise EvidenceIdentityError("dynamic profile/trace identity mismatch")
    start = _exact_int(
        profile.get("measured_window_start_wall_ns"),
        f"{profile_id}.measured_window_start_wall_ns",
    )
    end = _exact_int(profile.get("profile_end_wall_ns"), f"{profile_id}.end")
    first_ns = start + first * DECISION_PERIOD_NS
    last_ns = start + last * DECISION_PERIOD_NS
    if last_ns > end:
        raise EvidenceSchemaError("predeclared decision grid exceeds profile window")
    payload = {
        "decision_period_ns": DECISION_PERIOD_NS,
        "first_decision_ordinal": first,
        "first_decision_timestamp_ns": first_ns,
        "last_decision_ordinal": last,
        "last_decision_timestamp_ns": last_ns,
        "partition": partition.value,
        "profile_id": profile_id,
        "schema_id": SCHEMA_ID,
        "trace_id": trace_id,
    }
    return FrozenSegmentPlanV1(
        source_profile_id=profile_id,
        source_trace_id=trace_id,
        partition=partition,
        first_decision_ordinal=first,
        last_decision_ordinal=last,
        first_decision_timestamp_ns=first_ns,
        last_decision_timestamp_ns=last_ns,
        segment_identity_sha256=_canonical_sha256(payload),
    )


def predeclare_segment_plans(manifest: Mapping[str, Any]) -> Tuple[FrozenSegmentPlanV1, ...]:
    """Freeze fit/validation windows using metadata before reading MCS values."""
    if not isinstance(manifest, Mapping):
        raise EvidenceSchemaError("manifest must be a mapping")
    replay = manifest.get("replay_design")
    if not isinstance(replay, Mapping):
        raise EvidenceSchemaError("manifest replay_design is missing")
    if tuple(replay.get("profile_order", ())) != EXPECTED_PROFILE_ORDER:
        raise EvidenceIdentityError("profile order differs from retained design")
    if replay.get("sample_period_s") != 0.1:
        raise EvidenceSchemaError("retained profile period must be exactly 0.1 s")
    if replay.get("warmup_samples") != 50 or replay.get("measured_samples") != 250:
        raise EvidenceSchemaError("retained warmup/measured counts differ")
    profiles = manifest.get("profiles")
    if not isinstance(profiles, list):
        raise EvidenceSchemaError("manifest profiles are missing")
    by_id = {row.get("profile_id"): row for row in profiles if isinstance(row, Mapping)}
    if len(by_id) != len(profiles):
        raise EvidenceIdentityError("profile identities are missing or duplicated")
    result = []
    for profile_id in EXPECTED_PROFILE_TRACE_IDS:
        profile = by_id.get(profile_id)
        if profile is None:
            raise EvidenceIdentityError(f"missing dynamic profile {profile_id}")
        for field, expected in (
            ("warmup_samples", 50),
            ("measured_samples", 250),
            ("samples_scheduled", 300),
            ("commands_skipped_obsolete", 0),
            ("targets_clamped_to_mapping", 0),
        ):
            if profile.get(field) != expected:
                raise EvidenceSchemaError(f"{profile_id}.{field} differs")
        result.extend(
            (
                _segment_plan(
                    profile,
                    SequencePartition.FIT,
                    FIRST_DECISION_ORDINAL,
                    FIT_LAST_ORDINAL,
                ),
                _segment_plan(
                    profile,
                    SequencePartition.INTERNAL_VALIDATION,
                    VALIDATION_FIRST_ORDINAL,
                    LAST_DECISION_ORDINAL,
                ),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class DynamicMcsTransitionV1:
    segment_identity_sha256: str
    duration: int
    current: CausalMcsObservationV1
    successor: CausalMcsObservationV1
    successor_action_open_timestamp_ns: int

    def __post_init__(self) -> None:
        duration = _transition_int(
            self.duration,
            "duration",
            minimum=contract.MINIMUM_HOLD_TENSORS,
        )
        successor_open = _transition_int(
            self.successor_action_open_timestamp_ns,
            "successor_action_open_timestamp_ns",
        )
        expected_ordinal = self.current.decision_ordinal + duration
        if self.successor.decision_ordinal != expected_ordinal:
            raise CausalSelectionError(
                "MCS successor ordinal must advance by the exact transmitted-"
                "tensor duration"
            )
        expected_timestamp = (
            self.current.decision_timestamp_ns + duration * DECISION_PERIOD_NS
        )
        if successor_open != expected_timestamp:
            raise CausalSelectionError(
                "successor action-open timestamp is incompatible with duration "
                "and the frozen 10-Hz grid"
            )
        if self.successor.decision_timestamp_ns != successor_open:
            raise CausalSelectionError(
                "MCS successor must be sampled at the real successor action-open "
                "timestamp"
            )
        # Missing or stale MCS is an external-fallback condition. Refuse the
        # learning transition here rather than letting an invalid value reach
        # policy-state construction or be silently encoded as zero.
        self.current.policy_feature_dict()
        self.successor.policy_feature_dict()

    def policy_values(self) -> Tuple[int, int]:
        return (
            self.current.policy_feature_dict()["prior_ul_mcs_index"],
            self.successor.policy_feature_dict()["prior_ul_mcs_index"],
        )


@dataclass(frozen=True, slots=True)
class DynamicMcsSegmentV1:
    plan: FrozenSegmentPlanV1
    observations: Tuple[CausalMcsObservationV1, ...]

    def __post_init__(self) -> None:
        if len(self.observations) != self.plan.decision_count:
            raise CausalSelectionError("segment observation count differs from plan")
        expected = tuple(
            range(self.plan.first_decision_ordinal, self.plan.last_decision_ordinal + 1)
        )
        if tuple(item.decision_ordinal for item in self.observations) != expected:
            raise CausalSelectionError("segment decision ordinals are not contiguous")

    def transition_at(
        self,
        offset: int,
        *,
        duration: int,
        successor_action_open_timestamp_ns: int,
    ) -> DynamicMcsTransitionV1:
        """Bind one feedback-gated successor at its real action-open time.

        ``duration`` is the exact number of 10-Hz tensors transmitted under
        the current action. It is deliberately not treated as one decision:
        a two-tensor hold advances the retained MCS trace by 200 ms. The
        caller must also supply the real successor action-open timestamp so a
        cadence or duration mismatch fails closed instead of slowing the
        exogenous channel trace.
        """
        _transition_int(offset, "offset")
        exact_duration = _transition_int(
            duration,
            "duration",
            minimum=contract.MINIMUM_HOLD_TENSORS,
        )
        successor_open = _transition_int(
            successor_action_open_timestamp_ns,
            "successor_action_open_timestamp_ns",
        )
        successor_offset = offset + exact_duration
        if offset >= len(self.observations) or successor_offset >= len(
            self.observations
        ):
            raise SegmentBoundaryResetRequired(
                "duration-selected successor lies outside this frozen segment; "
                "reset instead of crossing a split or profile boundary"
            )
        return DynamicMcsTransitionV1(
            segment_identity_sha256=self.plan.segment_identity_sha256,
            duration=exact_duration,
            current=self.observations[offset],
            successor=self.observations[successor_offset],
            successor_action_open_timestamp_ns=successor_open,
        )


@dataclass(frozen=True, slots=True)
class McsVerifierStatsV1:
    eligible_ue_rows: int
    uniquely_joined_rows: int
    unmatched_rows: int
    ambiguous_rows: int
    mcs_mismatch_rows: int
    unique_join_fraction: float

    def __post_init__(self) -> None:
        if self.eligible_ue_rows <= 0:
            raise EvidenceIdentityError("UE verifier population is empty")
        if self.ambiguous_rows != 0 or self.mcs_mismatch_rows != 0:
            raise EvidenceIdentityError("UE/gNB MCS verifier has ambiguity or mismatch")
        if self.unique_join_fraction < MINIMUM_UNIQUE_GNB_JOIN_FRACTION:
            raise EvidenceIdentityError("UE/gNB MCS verifier coverage is below 99%")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ambiguous_rows": self.ambiguous_rows,
            "eligible_ue_rows": self.eligible_ue_rows,
            "mcs_mismatch_rows": self.mcs_mismatch_rows,
            "unique_join_fraction": self.unique_join_fraction,
            "uniquely_joined_rows": self.uniquely_joined_rows,
            "unmatched_rows": self.unmatched_rows,
        }


@dataclass(frozen=True, slots=True)
class DynamicMcsEvidenceV1:
    source_manifest_sha256: str
    source_config_sha256: str
    maximum_age_ns: int
    verifier: McsVerifierStatsV1
    segments: Tuple[DynamicMcsSegmentV1, ...]
    source_binding_sha256: str
    evidence_binding_sha256: str
    evidence_class: str = EVIDENCE_CLASS
    limitation: str = ONE_REALIZATION_LIMITATION

    def __post_init__(self) -> None:
        _exact_int(self.maximum_age_ns, "maximum_age_ns", minimum=1)
        if len(self.segments) != 4:
            raise EvidenceIdentityError("exactly four dynamic MCS segments are required")
        ids = tuple(item.plan.segment_identity_sha256 for item in self.segments)
        if len(set(ids)) != len(ids):
            raise EvidenceIdentityError("dynamic MCS segment identities collide")

    def segment(
        self, profile_id: str, partition: SequencePartition
    ) -> DynamicMcsSegmentV1:
        matches = tuple(
            item
            for item in self.segments
            if item.plan.source_profile_id == profile_id
            and item.plan.partition is partition
        )
        if len(matches) != 1:
            raise EvidenceIdentityError("requested segment is absent or ambiguous")
        return matches[0]


def _verify_sources(root: Path) -> Tuple[Mapping[str, Any], Mapping[str, Any], Path]:
    run_dir = root / SOURCE_RUN_RELATIVE_PATH
    manifest_path = run_dir / "manifest.json"
    if _sha256_file(manifest_path) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise EvidenceHashMismatch("source manifest digest differs")
    config_path = root / SOURCE_CONFIG_RELATIVE_PATH
    if _sha256_file(config_path) != EXPECTED_SOURCE_CONFIG_SHA256:
        raise EvidenceHashMismatch("source config digest differs")
    manifest = _load_json(manifest_path)
    config = _load_json(config_path)
    if manifest.get("schema") != "scenesense.ue_snr_bridge_qualification.v1":
        raise EvidenceSchemaError("source manifest schema differs")
    if manifest.get("status") != "UE_SNR_BRIDGE_EVIDENCE_CAPTURED":
        raise EvidenceSchemaError("source capture status is not complete")
    if manifest.get("interrupted") is not False:
        raise EvidenceSchemaError("source capture was interrupted")
    if manifest.get("config_sha256") != EXPECTED_SOURCE_CONFIG_SHA256:
        raise EvidenceHashMismatch("manifest/config digest binding differs")
    if config.get("radio", {}).get("mcs_policy") != "sinr":
        raise EvidenceSchemaError("source did not use the SINR-driven MCS policy")
    if config.get("replay", {}).get("profile_order") != list(EXPECTED_PROFILE_ORDER):
        raise EvidenceIdentityError("config profile order differs")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise EvidenceSchemaError("manifest file inventory is missing")
    inventory: Dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("relative_path"), str):
            raise EvidenceSchemaError("manifest has a malformed file entry")
        relative = entry["relative_path"]
        if relative in inventory:
            raise EvidenceIdentityError(f"duplicate manifest file {relative}")
        inventory[relative] = entry
    for relative, expected_sha in EXPECTED_REQUIRED_FILE_SHA256.items():
        entry = inventory.get(relative)
        if entry is None or entry.get("sha256") != expected_sha:
            raise EvidenceHashMismatch(f"manifest binding differs for {relative}")
        path = run_dir / relative
        if _sha256_file(path) != expected_sha:
            raise EvidenceHashMismatch(f"required artifact digest differs for {relative}")
        if path.stat().st_size != entry.get("size_bytes"):
            raise EvidenceHashMismatch(f"required artifact size differs for {relative}")
    identity = _load_json(run_dir / UE_IDENTITY_RELATIVE_PATH)
    expected_identity = {
        "ue_count": 1,
        "imsi": EXPECTED_IMSI,
        "interface": EXPECTED_UE_INTERFACE,
        "discovered_ipv4": EXPECTED_UE_IPV4,
        "ext_dn_ip": "192.168.70.135",
        "ping_pass": True,
    }
    if identity != expected_identity:
        raise EvidenceIdentityError("retained UE network identity differs")
    return manifest, config, run_dir


def _eligible_ue_grants(
    run_dir: Path, convert_time
) -> Tuple[RawUeGrantV1, ...]:
    rows = _read_exact_csv(run_dir / UE_DCI_RELATIVE_PATH, UE_DCI_HEADER)
    grants = []
    seen_identities: Dict[str, RawUeGrantV1] = {}
    for row in rows:
        if not (
            row["direction"] == "1"
            and row["dci_format"] == "7"
            and row["rnti_type"] == "0"
            and row["mcs_table"] == "0"
            and row["round"] == "0"
            and row["ndi"] == "1"
            and row["rv"] == "0"
        ):
            continue
        payload = {
            "dci_frame": _csv_int(row["dci_frame"], "dci_frame"),
            "dci_slot": _csv_int(row["dci_slot"], "dci_slot"),
            "harq_pid": _csv_int(row["harq_pid"], "harq_pid"),
            "mcs_index": _csv_int(row["mcs"], "mcs"),
            "rnti": _csv_int(row["rnti"], "rnti"),
            "sched_frame": _csv_int(row["sched_frame"], "sched_frame"),
            "sched_slot": _csv_int(row["sched_slot"], "sched_slot"),
            "source_row_number": _csv_int(row["__row_number__"], "row_number"),
            "source_timestamp_ns": convert_time(row["time"]),
        }
        identity = _canonical_sha256(
            {"record": "eligible_ue_ul_dci_grant_v1", **payload}
        )
        grant = RawUeGrantV1(source_identity_sha256=identity, **payload)
        previous = seen_identities.get(identity)
        if previous is not None and previous != grant:
            raise EvidenceIdentityError("one grant identity names conflicting rows")
        seen_identities[identity] = grant
        grants.append(grant)
    if not grants:
        raise EvidenceIdentityError("no eligible UE grants")
    rntis = {item.rnti for item in grants}
    if rntis != {EXPECTED_UE_RNTI}:
        raise EvidenceIdentityError(f"eligible UE RNTIs differ: {sorted(rntis)}")
    grants.sort(key=lambda item: (item.source_timestamp_ns, item.source_row_number))
    times = [item.source_timestamp_ns for item in grants]
    if len(times) != len(set(times)):
        raise EvidenceIdentityError("eligible UE grants share a tracer timestamp")
    return tuple(grants)


def _verify_against_gnb(
    run_dir: Path, convert_time, grants: Sequence[RawUeGrantV1]
) -> McsVerifierStatsV1:
    rows = _read_exact_csv(run_dir / GNB_MCS_RELATIVE_PATH, GNB_MCS_HEADER)
    by_key: Dict[Tuple[int, int, int, int, int], list[Tuple[int, int]]] = defaultdict(list)
    for row in rows:
        if row["mcs_table"] != "0":
            continue
        key = (
            _csv_int(row["rnti"], "gnb.rnti"),
            _csv_int(row["frame"], "gnb.frame"),
            _csv_int(row["slot"], "gnb.slot"),
            _csv_int(row["sched_frame"], "gnb.sched_frame"),
            _csv_int(row["sched_slot"], "gnb.sched_slot"),
        )
        final_mcs = _csv_int(row["final_mcs"], "gnb.final_mcs")
        if not contract.UL_MCS_INDEX_MIN <= final_mcs <= contract.UL_MCS_INDEX_MAX:
            raise EvidenceSchemaError("gNB final MCS is outside table-0 support")
        by_key[key].append((convert_time(row["time"]), final_mcs))
    joined = unmatched = ambiguous = mismatched = 0
    for grant in grants:
        key = (
            grant.rnti,
            grant.dci_frame,
            grant.dci_slot,
            grant.sched_frame,
            grant.sched_slot,
        )
        candidates = sorted(
            (
                (abs(grant.source_timestamp_ns - timestamp), timestamp, mcs)
                for timestamp, mcs in by_key.get(key, ())
                if abs(grant.source_timestamp_ns - timestamp)
                <= UE_GNB_JOIN_TOLERANCE_NS
            )
        )
        if not candidates:
            unmatched += 1
            continue
        best_distance = candidates[0][0]
        nearest = tuple(item for item in candidates if item[0] == best_distance)
        if len(nearest) != 1:
            ambiguous += 1
            continue
        joined += 1
        if nearest[0][2] != grant.mcs_index:
            mismatched += 1
    return McsVerifierStatsV1(
        eligible_ue_rows=len(grants),
        uniquely_joined_rows=joined,
        unmatched_rows=unmatched,
        ambiguous_rows=ambiguous,
        mcs_mismatch_rows=mismatched,
        unique_join_fraction=joined / len(grants),
    )


def load_dynamic_mcs_evidence(
    *,
    maximum_age_ns: int,
    repository_root: Optional[Path] = None,
) -> DynamicMcsEvidenceV1:
    """Verify retained evidence and build four causal offline segments."""
    maximum_age_ns = _exact_int(maximum_age_ns, "maximum_age_ns", minimum=1)
    root = _repo_root() if repository_root is None else Path(repository_root)
    manifest, _config, run_dir = _verify_sources(root)

    # This plan is intentionally constructed before either MCS CSV is opened.
    plans = predeclare_segment_plans(manifest)
    anchors = _load_json(run_dir / CLOCK_ANCHORS_RELATIVE_PATH)
    if not isinstance(anchors, list) or not anchors:
        raise EvidenceSchemaError("clock anchors are missing")
    offsets = {item.get("utc_offset_s") for item in anchors if isinstance(item, Mapping)}
    if len(offsets) != 1:
        raise EvidenceSchemaError("clock anchors cross inconsistent UTC offsets")
    convert_time = _tracer_clock(anchors[0])
    grants = _eligible_ue_grants(run_dir, convert_time)
    verifier = _verify_against_gnb(run_dir, convert_time, grants)

    profile_rows = {
        item["profile_id"]: item
        for item in manifest["profiles"]
        if item["profile_id"] in EXPECTED_PROFILE_TRACE_IDS
    }
    segments = []
    used_by_partition: Dict[Tuple[str, SequencePartition], set[str]] = {}
    for plan in plans:
        profile = profile_rows[plan.source_profile_id]
        start = int(profile["measured_window_start_wall_ns"])
        end = int(profile["profile_end_wall_ns"])
        profile_grants = tuple(
            item for item in grants if start <= item.source_timestamp_ns < end
        )
        observations = tuple(
            select_latest_strictly_prior_mcs(
                profile_grants,
                decision_ordinal=ordinal,
                decision_timestamp_ns=start + ordinal * DECISION_PERIOD_NS,
                maximum_age_ns=maximum_age_ns,
            )
            for ordinal in range(
                plan.first_decision_ordinal, plan.last_decision_ordinal + 1
            )
        )
        used_by_partition[(plan.source_profile_id, plan.partition)] = {
            item.source_identity_sha256
            for item in observations
            if item.source_identity_sha256 is not None
        }
        segments.append(DynamicMcsSegmentV1(plan=plan, observations=observations))
    for profile_id in EXPECTED_PROFILE_TRACE_IDS:
        fit_ids = used_by_partition[(profile_id, SequencePartition.FIT)]
        validation_ids = used_by_partition[
            (profile_id, SequencePartition.INTERNAL_VALIDATION)
        ]
        if fit_ids & validation_ids:
            raise EvidenceIdentityError(
                f"{profile_id}: fit/internal-validation reuse a grant identity"
            )

    source_binding = _canonical_sha256(
        {
            "config_sha256": EXPECTED_SOURCE_CONFIG_SHA256,
            "file_sha256": EXPECTED_REQUIRED_FILE_SHA256,
            "filter": {
                "direction": 1,
                "dci_format": 7,
                "harq_round": 0,
                "mcs_table": 0,
                "new_data_indicator": 1,
                "rnti": EXPECTED_UE_RNTI,
                "rnti_type": 0,
                "rv": 0,
                "scheduler_policy_id": contract.UL_MCS_POLICY_ID,
                "selection_rule_id": contract.UL_MCS_SELECTION_RULE_ID,
            },
            "manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
            "plans": [item.to_dict() for item in plans],
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
        }
    )
    evidence_payload = {
        "evidence_class": EVIDENCE_CLASS,
        "limitation": ONE_REALIZATION_LIMITATION,
        "maximum_age_ns": maximum_age_ns,
        "segments": [
            {
                "observations": [item.to_dict() for item in segment.observations],
                "plan": segment.plan.to_dict(),
            }
            for segment in segments
        ],
        "source_binding_sha256": source_binding,
        "verifier": verifier.to_dict(),
    }
    return DynamicMcsEvidenceV1(
        source_manifest_sha256=EXPECTED_SOURCE_MANIFEST_SHA256,
        source_config_sha256=EXPECTED_SOURCE_CONFIG_SHA256,
        maximum_age_ns=maximum_age_ns,
        verifier=verifier,
        segments=tuple(segments),
        source_binding_sha256=source_binding,
        evidence_binding_sha256=_canonical_sha256(evidence_payload),
    )


def require_production_dynamic_mcs_binding(
    evidence: DynamicMcsEvidenceV1,
    *,
    composite_verifier_manifest_sha256: str,
) -> None:
    """Fail closed until a reviewed composite verifier pins this evidence."""
    if type(evidence) is not DynamicMcsEvidenceV1:
        raise ProductionDynamicMcsUnavailable("evidence must be exact DynamicMcsEvidenceV1")
    if (
        not isinstance(composite_verifier_manifest_sha256, str)
        or len(composite_verifier_manifest_sha256) != 64
        or any(c not in "0123456789abcdef" for c in composite_verifier_manifest_sha256)
    ):
        raise ProductionDynamicMcsUnavailable("invalid composite verifier digest")
    if REGISTERED_COMPOSITE_DYNAMIC_MCS_BINDING_SHA256 is None:
        raise ProductionDynamicMcsUnavailable(
            "no reviewed composite verifier pins dynamic MCS production use"
        )
    candidate = _canonical_sha256(
        {
            "composite_verifier_manifest_sha256": (
                composite_verifier_manifest_sha256
            ),
            "dynamic_mcs_evidence_binding_sha256": (
                evidence.evidence_binding_sha256
            ),
            "record": "splitfusion_run4_composite_dynamic_mcs_binding_v1",
        }
    )
    if candidate != REGISTERED_COMPOSITE_DYNAMIC_MCS_BINDING_SHA256:
        raise ProductionDynamicMcsUnavailable(
            "dynamic MCS binding differs from reviewed composite verifier"
        )
