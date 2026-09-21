"""One-step, fit-only empirical contextual environment (D1).

The environment is deliberately not Gym and contains no training loop.  It
samples one eligible fit scene, one hidden radio/network profile and a
same-profile measured radio row, constructs the existing exact 31-feature
genesis state, and evaluates one exact supported action with deterministic
expected utility.  The held split has no type or method in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from .anchor_store import NETWORK_PROFILE_ORDER
from .action_contract import CATALOG_SHA256
from .corrected_p40_sidecar import (
    CorrectedP40Sidecar,
    load_exact_corrected_p40_sidecar,
)
from .empirical_contextual_contract import (
    DIRECT_QUALITY_COMPONENT,
    FIXED_END_TO_FEEDBACK_STAGES_MS,
    MODELED_SMOKE_SUPPORT,
    MODELED_SMOKE_SUPPORT_SHA256,
    PILOT_UTILITY_SPEC,
    PILOT_UTILITY_SPEC_SHA256,
    PROFILE_ORDER_RNG_CONTRACT_SHA256,
    SURFACE_QUALIFICATION_REPORT_SHA256,
    EmpiricalActionV1,
    EmpiricalPilotBindingV1,
    fixed_stage_latency_ms,
    require_supported_action,
)
from .empirical_quality_surface import (
    DATABASE_FILE_SHA256,
    SELECTION_FILE_SHA256,
    EmpiricalQualitySurface,
    load_empirical_quality_surface,
)
from .empirical_radio_context import (
    CALIBRATION_SHA256,
    MCS_TABLE_ID,
    MCS_TABLE_MAX_INDEX,
    OaiRadioCalibrationStoreV1,
    RadioContextDrawV1,
    RadioContextSamplerV1,
    RadioSamplerStateV1,
)
from .modeled_smoke_support import require_registered_modeled_smoke_support
from .payload_network_surrogate import (
    EVIDENCE_CLASS as NETWORK_EVIDENCE_CLASS,
    ExtrapolationRefusedError,
    PayloadNetworkSurrogate,
    PrevalidatedPredictionSession,
    UDP_PAYLOAD_CAPACITY_BYTES,
    build_payload_network_surrogate,
)
from .reward_ticket_controller import RewardTicketController
from .scene_descriptors import SceneDescriptorSample
from .state_reward_transition_contract import (
    BsrScope,
    CausalStateV1,
    EpisodeStartProofV1,
    POLICY_FEATURE_ORDER,
    POLICY_OBSERVATION_DEPLOYABILITY,
    SceneObservationV1,
    SnrMetric,
    StateFreshnessPolicyV1,
    StateNormalizationSpecV1,
    build_policy_features,
)
from .transaction_identity import canonical_sha256

__all__ = [
    "EmpiricalEnvironmentError",
    "EmpiricalOneStepEnvironmentV1",
    "EmpiricalOutcomeV1",
    "EmpiricalPolicyObservationV1",
    "EmpiricalPreflightReportV1",
    "EmpiricalStepAuditV1",
    "EmpiricalStepResultV1",
    "EmpiricalEnvironmentStateV1",
    "FitContextIndexV1",
    "FitQualityInventoryV1",
    "FitTrainingContextV1",
    "build_registered_freshness_policy",
    "build_registered_normalization_spec",
    "load_registered_d1_environment",
    "run_empirical_d1_preflight",
]


class EmpiricalEnvironmentError(RuntimeError):
    """D1 dependency, lifecycle, or evidence invariant failed closed."""


_PREFLIGHT_ATTESTATION = object()


@dataclass(frozen=True, slots=True)
class FitQualityInventoryV1:
    fit_scene_count: int
    rewardable_scene_count: int
    action_independent_invalid_scene_count: int
    rows_per_scene: int
    invalidity_is_action_independent: bool
    eligible_sample_ids: Tuple[str, ...]
    invalid_sample_ids: Tuple[str, ...]


def _audit_fit_quality_inventory(database_path: Path) -> FitQualityInventoryV1:
    uri = f"file:{Path(database_path).resolve(strict=True)}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=30.0)
    connection.execute("PRAGMA query_only=ON")
    try:
        rows = connection.execute(
            "SELECT sample_id, MIN(CAST(json_extract(row_json, '$.quality_valid') "
            "AS INTEGER)), MAX(CAST(json_extract(row_json, '$.quality_valid') "
            "AS INTEGER)), COUNT(*) FROM quality_rows WHERE grid_split='fit' "
            "GROUP BY sample_id ORDER BY sample_id"
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != 512:
        raise EmpiricalEnvironmentError(f"fit scene inventory is {len(rows)}, not 512")
    if any(count != 132 for _sample, _minimum, _maximum, count in rows):
        raise EmpiricalEnvironmentError("a fit scene does not have 12 x 11 exact rows")
    mixed = [sample for sample, minimum, maximum, _count in rows if minimum != maximum]
    if mixed:
        raise EmpiricalEnvironmentError(
            "quality invalidity depends on action for fit scenes; D1 context "
            f"filter is invalid (first mixed sample {mixed[0]!r})"
        )
    eligible = tuple(sample for sample, minimum, _maximum, _count in rows if minimum == 1)
    invalid = tuple(sample for sample, minimum, _maximum, _count in rows if minimum == 0)
    if (len(eligible), len(invalid)) != (476, 36):
        raise EmpiricalEnvironmentError(
            f"fit quality inventory drift: eligible={len(eligible)}, invalid={len(invalid)}"
        )
    return FitQualityInventoryV1(
        fit_scene_count=512,
        rewardable_scene_count=476,
        action_independent_invalid_scene_count=36,
        rows_per_scene=132,
        invalidity_is_action_independent=True,
        eligible_sample_ids=eligible,
        invalid_sample_ids=invalid,
    )


@dataclass(frozen=True, slots=True)
class FitTrainingContextV1:
    """Environment-hidden fit record.  No split discriminator is accepted."""

    sample_id: str
    episode_id: str
    frame_id: int
    camera_si: float
    radar_p40: float
    sampling_weight: float
    selection_rank_within_fit: int

    def __post_init__(self) -> None:
        if not self.sample_id or not self.episode_id:
            raise EmpiricalEnvironmentError("fit context identities must be non-empty")
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise EmpiricalEnvironmentError("fit context frame_id must be non-negative")
        if type(self.selection_rank_within_fit) is not int or self.selection_rank_within_fit < 0:
            raise EmpiricalEnvironmentError("fit selection rank must be non-negative")
        if not math.isfinite(self.camera_si) or self.camera_si < 0.0:
            raise EmpiricalEnvironmentError("fit camera SI must be finite/non-negative")
        if not math.isfinite(self.radar_p40) or not 0.0 <= self.radar_p40 <= 1.0:
            raise EmpiricalEnvironmentError("fit corrected P40 must be finite in [0,1]")
        if not math.isfinite(self.sampling_weight) or self.sampling_weight <= 0.0:
            raise EmpiricalEnvironmentError("fit sampling weight must be finite/positive")


@dataclass(frozen=True, slots=True)
class FitContextIndexV1:
    contexts: Tuple[FitTrainingContextV1, ...]
    excluded_action_independent_invalid_count: int
    source_selection_sha256: str
    source_database_sha256: str
    all_fit_camera_si_min: float
    all_fit_camera_si_max: float
    canonical_sha256: str

    @classmethod
    def load_registered(
        cls,
        *,
        surface: EmpiricalQualitySurface,
        sidecar: CorrectedP40Sidecar,
        project_root: Optional[Path] = None,
    ) -> "FitContextIndexV1":
        root = (
            Path(__file__).resolve().parents[2]
            if project_root is None
            else Path(project_root).resolve(strict=True)
        )
        selection_path = root / (
            "experiments/splitfusion_hybrid_sac_quality_grid_v1/"
            "20260918_exact_continuous_q_grid_a1b_full/selection_manifest.json"
        )
        raw = selection_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != SELECTION_FILE_SHA256:
            raise EmpiricalEnvironmentError("selection manifest SHA-256 drift")
        document = json.loads(raw.decode("utf-8"))
        inventory = _audit_fit_quality_inventory(surface.database_path)
        eligible = set(inventory.eligible_sample_ids)
        sidecar_fit = {
            record.sample_id: record
            for record in sidecar.records
            if record.grid_split == "fit"
        }
        contexts = []
        all_fit_si = []
        # Held rows are rejected at this boundary and never represented by a
        # training-context object.
        for row in document["selected_frames"]:
            if row["grid_split"] != "fit":
                continue
            fit_si = float(row["camera_si"])
            if not math.isfinite(fit_si) or fit_si < 0.0:
                raise EmpiricalEnvironmentError(
                    f"nonfinite/negative normalization-fit SI: {row.get('sample_id')!r}"
                )
            all_fit_si.append(fit_si)
            sample_id = str(row["sample_id"])
            if sample_id not in eligible:
                continue
            corrected = sidecar_fit.get(sample_id)
            if corrected is None:
                raise EmpiricalEnvironmentError(f"missing fit P40 for {sample_id}")
            camera_si = float(row["camera_si"])
            radar_p40 = float(corrected.corrected_p40)
            sampling_weight = float(row["sampling_weight"])
            if not math.isfinite(camera_si) or camera_si < 0.0:
                raise EmpiricalEnvironmentError(f"nonfinite/negative fit SI: {sample_id}")
            if not math.isfinite(radar_p40) or not 0.0 <= radar_p40 <= 1.0:
                raise EmpiricalEnvironmentError(f"invalid corrected P40: {sample_id}")
            if not math.isfinite(sampling_weight) or sampling_weight <= 0.0:
                raise EmpiricalEnvironmentError(f"invalid sampling weight: {sample_id}")
            contexts.append(
                FitTrainingContextV1(
                    sample_id=sample_id,
                    episode_id=str(row["episode_id"]),
                    frame_id=int(row["frame_id"]),
                    camera_si=camera_si,
                    radar_p40=radar_p40,
                    sampling_weight=sampling_weight,
                    selection_rank_within_fit=int(row["selection_rank_within_split"]),
                )
            )
        contexts.sort(key=lambda item: item.sample_id)
        if len(contexts) != 476 or any(item.sampling_weight <= 0 for item in contexts):
            raise EmpiricalEnvironmentError("eligible fit context inventory drift")
        if len(all_fit_si) != 512:
            raise EmpiricalEnvironmentError("normalization fit inventory is not 512")
        index_document = {
            "contexts": [asdict(item) for item in contexts],
            "excluded_action_independent_invalid_count": 36,
            "record": "fit_context_index_v1",
            "source_database_sha256": DATABASE_FILE_SHA256,
            "source_selection_sha256": SELECTION_FILE_SHA256,
            "all_fit_camera_si_min": min(all_fit_si),
            "all_fit_camera_si_max": max(all_fit_si),
        }
        return cls(
            contexts=tuple(contexts),
            excluded_action_independent_invalid_count=36,
            source_selection_sha256=SELECTION_FILE_SHA256,
            source_database_sha256=DATABASE_FILE_SHA256,
            all_fit_camera_si_min=min(all_fit_si),
            all_fit_camera_si_max=max(all_fit_si),
            canonical_sha256=canonical_sha256(index_document),
        )


def build_registered_normalization_spec() -> StateNormalizationSpecV1:
    fit_config = {
        "camera_si": "MIN_MAX_OVER_ALL_512_FIT_SCENES",
        "camera_si_max": 147.98822021484375,
        "camera_si_min": 84.53893280029297,
        "radio_joint_valid_count": 399,
        "radio_source_sha256": CALIBRATION_SHA256,
        "snr_max_db": 23.5,
        "snr_min_db": 6.0,
    }
    return StateNormalizationSpecV1(
        spec_id="splitfusion_empirical_contextual_d1_fit_normalization_v1",
        spec_version=1,
        train_split_id="QUALITY_GRID_FIT_512_PLUS_RADIO_CALIBRATION_399",
        fit_population_count=512,
        fit_config_sha256=canonical_sha256(fit_config),
        camera_si_clip_min=84.53893280029297,
        camera_si_clip_max=147.98822021484375,
        achieved_snr_db_clip_min=6.0,
        achieved_snr_db_clip_max=23.5,
        bsr_log1p_scale=1.0,
        snr_metric=SnrMetric.SIMULATOR_EFFECTIVE_UL_SNR_DB,
        mcs_table_id=MCS_TABLE_ID,
        mcs_table_max_index=MCS_TABLE_MAX_INDEX,
        bsr_scope=BsrScope.ALL_GROUPS_LATEST,
        provenance={
            "bsr_scale_status": "INERT_FOR_EXACT_GENESIS_ZERO_ONLY",
            "camera_fit_population": "512_FIT_SCENES_HELD_EXCLUDED",
            "radio_calibration_population": "399_JOINT_VALID_ROWS",
            "radio_deployability": "SIMULATOR_TESTBED_ONLY",
            "radio_source_sha256": CALIBRATION_SHA256,
        },
    )


def build_registered_freshness_policy() -> StateFreshnessPolicyV1:
    return StateFreshnessPolicyV1(
        policy_id="splitfusion_empirical_contextual_d1_genesis_freshness_v1",
        max_scene_age_ns=100_000_000,
        max_snr_age_ns=100_000_000,
        max_bsr_age_ns=100_000_000,
        max_mcs_age_ns=100_000_000,
        provenance={
            "environment_invariant": "ALL_D1_GENESIS_MEASUREMENT_AGES_ARE_EXACTLY_ZERO",
            "scope": "ONE_STEP_SIMULATOR_TESTBED_ONLY",
        },
    )


@dataclass(frozen=True, slots=True)
class EmpiricalPolicyObservationV1:
    """The complete policy handoff: 31 floats and harmless static bindings."""

    values: Tuple[float, ...]
    policy_feature_order: Tuple[str, ...]
    environment_binding_sha256: str
    normalization_spec_sha256: str
    freshness_policy_sha256: str
    deployability: str = POLICY_OBSERVATION_DEPLOYABILITY

    def __post_init__(self) -> None:
        if type(self.values) is not tuple or len(self.values) != len(
            POLICY_FEATURE_ORDER
        ):
            raise EmpiricalEnvironmentError("policy observation must contain 31 values")
        if any(type(value) is not float or not math.isfinite(value) for value in self.values):
            raise EmpiricalEnvironmentError("policy observation values must be finite floats")
        if self.policy_feature_order != tuple(POLICY_FEATURE_ORDER):
            raise EmpiricalEnvironmentError("policy feature order drift")
        for name in (
            "environment_binding_sha256",
            "normalization_spec_sha256",
            "freshness_policy_sha256",
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise EmpiricalEnvironmentError(f"{name} must be a lowercase SHA-256")
        if self.deployability != POLICY_OBSERVATION_DEPLOYABILITY:
            raise EmpiricalEnvironmentError("policy observation deployability drift")

    def as_mapping(self) -> Dict[str, float]:
        return dict(zip(self.policy_feature_order, self.values))


@dataclass(frozen=True, slots=True)
class EmpiricalOutcomeV1:
    reward: Optional[float]
    status: str
    terminated: bool
    truncated: bool
    q_perc: Optional[float]
    p_complete_reassembly_given_sent: Optional[float]
    p_edge_admission_given_reassembled: Optional[float]
    p_edge_admission_given_sent: Optional[float]
    conditional_feature_uplink_p50_ms: Optional[float]
    conditional_feature_uplink_p95_ms: Optional[float]
    conditional_feature_uplink_p99_ms: Optional[float]
    fixed_stage_latency_ms: float
    fixed_latency_stages_ms: Tuple[Tuple[str, float], ...]
    latency_proxy_ms: Optional[float]
    latency_proxy_p95_ms: Optional[float]
    latency_proxy_p99_ms: Optional[float]
    deadline_ms: float
    modeled_budget_miss: Optional[bool]
    modeled_budget_miss_p95: Optional[bool]
    modeled_budget_miss_p99: Optional[bool]
    estimator: str
    service_non_admission_semantics: str
    timeout_probability_status: str


@dataclass(frozen=True, slots=True)
class EmpiricalStepAuditV1:
    sample_id: str
    episode_id: str
    frame_id: int
    hidden_network_profile: str
    hidden_radio_csv_row_number: int
    hidden_trace_id: str
    hidden_trace_step_index: int
    hidden_target_snr_db: float
    hidden_radio_row_sha256: str
    surface_evidence_status: str
    total_transmitted_bytes: float
    datagram_count: int
    network_evidence_class: str
    utility_spec_sha256: str
    executed_mode_id: int
    executed_q_e4: int


@dataclass(frozen=True, slots=True)
class EmpiricalStepResultV1:
    policy: EmpiricalOutcomeV1
    audit: EmpiricalStepAuditV1


@dataclass(frozen=True, slots=True)
class EmpiricalEnvironmentStateV1:
    """Checkpoint state valid only between one-step episodes."""

    environment_binding_sha256: str
    master_seed: int
    reset_count: int
    context_rng_state: tuple
    radio_sampler_state: RadioSamplerStateV1


@dataclass(frozen=True, slots=True)
class EmpiricalPreflightReportV1:
    status: str
    fit_scene_count: int
    rewardable_fit_context_count: int
    excluded_action_independent_invalid_count: int
    invalidity_is_action_independent: bool
    minimum_supported_fit_payload_bytes: float
    maximum_supported_fit_payload_bytes: float
    supported_endpoint_query_count: int
    exhaustive_network_endpoint_profile_query_count: int
    network_latency_interior_support_hole_count: int
    radio_source_row_count: int
    radio_joint_valid_row_count: int
    radio_cross_profile_aliased_row_count: int
    radio_profile_counts: Mapping[str, int]
    policy_feature_count: int
    all_genesis_ages_zero: bool
    conditional_uplink_p50_range_ms: Tuple[float, float]
    conditional_uplink_p95_range_ms: Tuple[float, float]
    conditional_uplink_p99_range_ms: Tuple[float, float]
    total_proxy_p50_range_ms: Tuple[float, float]
    total_proxy_p95_range_ms: Tuple[float, float]
    total_proxy_p99_range_ms: Tuple[float, float]
    expected_reward_range: Tuple[float, float]
    expected_reward_nondegenerate: bool
    modeled_budget_miss_counts_p50_p95_p99: Tuple[int, int, int]
    surface_qualification_report_sha256: str
    surface_qualification_overall_status: str
    surface_q_perc_fit_held_qualified: bool
    surface_q_seg_fit_held_qualified: bool
    surface_held_payload_qualified: bool
    surface_full_component_qualified: bool
    environment_binding_sha256: str
    disclosures: Tuple[str, ...]
    report_sha256: str
    _attestation: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._attestation is not _PREFLIGHT_ATTESTATION:
            raise EmpiricalEnvironmentError(
                "preflight report must be issued by run_empirical_d1_preflight"
            )
        object.__setattr__(
            self,
            "radio_profile_counts",
            MappingProxyType(dict(self.radio_profile_counts)),
        )
        if self.report_sha256 != self.canonical_sha256():
            raise EmpiricalEnvironmentError("preflight report canonical digest drift")

    def _canonical_document(self) -> Dict[str, Any]:
        result = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name not in {"report_sha256", "_attestation"}
        }
        result["radio_profile_counts"] = dict(self.radio_profile_counts)
        return result

    def canonical_sha256(self) -> str:
        return canonical_sha256(self._canonical_document())

    def require_registered(self) -> None:
        if self._attestation is not _PREFLIGHT_ATTESTATION:
            raise EmpiricalEnvironmentError("unattested preflight report")
        if self.report_sha256 != self.canonical_sha256():
            raise EmpiricalEnvironmentError("preflight report digest mismatch")

    def to_dict(self) -> Dict[str, Any]:
        result = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "_attestation"
        }
        result["radio_profile_counts"] = dict(self.radio_profile_counts)
        return result


def _surface_binding_sha256(surface: EmpiricalQualitySurface) -> str:
    return canonical_sha256(asdict(surface.binding))


def _implementation_bundle_sha256() -> str:
    module_dir = Path(__file__).resolve().parent
    names = (
        "corrected_p40_sidecar.py",
        "empirical_contextual_contract.py",
        "empirical_contextual_environment.py",
        "empirical_quality_surface.py",
        "empirical_radio_context.py",
        "modeled_smoke_support.py",
        "payload_network_surrogate.py",
        "state_reward_transition_contract.py",
    )
    entries = []
    for name in names:
        raw = (module_dir / name).read_bytes()
        entries.append([name, hashlib.sha256(raw).hexdigest()])
    return canonical_sha256(
        {"record": "d1_implementation_bundle_v1", "files": entries}
    )


def _build_binding(
    *,
    sidecar: CorrectedP40Sidecar,
    surface: EmpiricalQualitySurface,
    network: PayloadNetworkSurrogate,
    radio_store: OaiRadioCalibrationStoreV1,
    contexts: FitContextIndexV1,
    normalization: StateNormalizationSpecV1,
    freshness: StateFreshnessPolicyV1,
) -> EmpiricalPilotBindingV1:
    return EmpiricalPilotBindingV1(
        corrected_p40_binding_sha256=sidecar.binding_sha256,
        quality_surface_binding_sha256=_surface_binding_sha256(surface),
        network_surrogate_sha256=network.canonical_sha256(),
        radio_calibration_sha256=radio_store.source_sha256,
        modeled_smoke_support_sha256=MODELED_SMOKE_SUPPORT_SHA256,
        utility_spec_sha256=PILOT_UTILITY_SPEC_SHA256,
        normalization_spec_sha256=normalization.canonical_sha256(),
        freshness_policy_sha256=freshness.canonical_sha256(),
        eligible_fit_context_index_sha256=contexts.canonical_sha256,
        action_catalog_sha256=CATALOG_SHA256,
        surface_qualification_report_sha256=(
            SURFACE_QUALIFICATION_REPORT_SHA256
        ),
        profile_order_rng_contract_sha256=PROFILE_ORDER_RNG_CONTRACT_SHA256,
        implementation_bundle_sha256=_implementation_bundle_sha256(),
    )


def run_empirical_d1_preflight(
    *,
    sidecar: CorrectedP40Sidecar,
    surface: EmpiricalQualitySurface,
    network: PayloadNetworkSurrogate,
    radio_store: OaiRadioCalibrationStoreV1,
    contexts: FitContextIndexV1,
    normalization: StateNormalizationSpecV1,
    freshness: StateFreshnessPolicyV1,
) -> EmpiricalPreflightReportV1:
    """Exhaustively prove the D1 fit context/action endpoint envelope."""
    support = require_registered_modeled_smoke_support(MODELED_SMOKE_SUPPORT)
    inventory = _audit_fit_quality_inventory(surface.database_path)
    qualification = surface.qualify_leave_one_q_out()
    if qualification.report_sha256 != SURFACE_QUALIFICATION_REPORT_SHA256:
        raise EmpiricalEnvironmentError("surface qualification report hash drift")
    qualification_document = qualification.document()
    gates = qualification_document.get("gate_results", {})
    components = gates.get("quality_component_by_split", {})
    if not (
        qualification_document.get("status") == "FAIL"
        and gates.get("held_payload_p95") is True
        and gates.get("reward_target_q_perc_qualified") is True
        and gates.get("full_component_surface_qualified") is False
        and all(
            components.get(split, {}).get("q_perc") is True
            and components.get(split, {}).get("q_seg") is False
            for split in ("fit", "held_scene")
        )
    ):
        raise EmpiricalEnvironmentError(
            "surface qualification outcome drift: D1 requires q_perc and held "
            "payload PASS while preserving overall/q_seg FAIL"
        )
    if tuple(sorted(item.sample_id for item in contexts.contexts)) != inventory.eligible_sample_ids:
        raise EmpiricalEnvironmentError("fit context index differs from quality audit")
    if (
        contexts.all_fit_camera_si_min != normalization.camera_si_clip_min
        or contexts.all_fit_camera_si_max != normalization.camera_si_clip_max
    ):
        raise EmpiricalEnvironmentError(
            "camera SI normalization was not re-derived from all 512 fit rows"
        )
    if radio_store.achieved_snr_range_db != (
        normalization.achieved_snr_db_clip_min,
        normalization.achieved_snr_db_clip_max,
    ):
        raise EmpiricalEnvironmentError("radio normalization range drift")

    payload_min = math.inf
    payload_max = -math.inf
    endpoint_count = 0
    network_endpoint_count = 0
    p50_values = []
    p95_values = []
    p99_values = []
    reward_values = []
    budget_miss_counts = [0, 0, 0]
    prediction_session = network.prevalidated_prediction_session()
    for context in contexts.contexts:
        for mode_id, (lower, upper) in enumerate(support.mode_q_e4_bounds):
            for q_e4 in (lower, upper):
                result = surface.query_fit_q_e4(context.sample_id, mode_id, q_e4)
                component = result.policy.component(DIRECT_QUALITY_COMPONENT)
                if not component.valid or component.value is None:
                    raise EmpiricalEnvironmentError(
                        "eligible context produced undefined direct q_perc"
                    )
                payload = float(result.policy.payload.total_transmitted_bytes)
                payload_min = min(payload_min, payload)
                payload_max = max(payload_max, payload)
                endpoint_count += 1
                for profile in NETWORK_PROFILE_ORDER:
                    prediction = prediction_session.predict(
                        network_profile=profile,
                        payload_bytes=payload,
                        datagram_count=math.ceil(
                            payload / UDP_PAYLOAD_CAPACITY_BYTES
                        ),
                    )
                    latency = prediction.conditional_retained_survivor_latency_model()
                    p50_values.append(latency.p50_ms)
                    p95_values.append(latency.p95_ms)
                    p99_values.append(latency.p99_ms)
                    proxies = tuple(
                        fixed_stage_latency_ms() + value
                        for value in (latency.p50_ms, latency.p95_ms, latency.p99_ms)
                    )
                    for index, proxy in enumerate(proxies):
                        budget_miss_counts[index] += int(
                            proxy > PILOT_UTILITY_SPEC.deadline_ms
                        )
                    reward_values.append(
                        PILOT_UTILITY_SPEC.expected_utility(
                            p_edge_admission_given_sent=(
                                prediction.p_complete_reassembly_given_sent
                                * prediction.p_edge_admission_given_reassembled
                            ),
                            q_perc=float(component.value),
                            latency_proxy_ms=proxies[0],
                        )
                    )
                    network_endpoint_count += 1
    common_min, common_max = support.all_profile_payload_support
    if not common_min <= payload_min <= payload_max <= common_max:
        raise EmpiricalEnvironmentError(
            f"fit curriculum payload [{payload_min}, {payload_max}] escapes "
            f"common support [{common_min}, {common_max}]"
        )
    x_min, x_max = math.log(payload_min), math.log(payload_max)
    interior_holes = 0
    for profile in NETWORK_PROFILE_ORDER:
        profile_model = network.profile_models[profile]
        if any(
            curve.raw_x_min > x_min or curve.raw_x_max < x_max
            for curve in profile_model.latency_curves.values()
        ):
            interior_holes += 1
            continue
        knots = profile_model.latency_support_knots
        for left, right in zip(knots, knots[1:]):
            interval_overlaps = right[0] >= x_min and left[0] <= x_max
            if interval_overlaps and min(left[1], right[1]) < network.contract.latency_min_support:
                interior_holes += 1
        if not knots[0][0] <= x_min <= x_max <= knots[-1][0]:
            interior_holes += 1
    if interior_holes != 0:
        raise EmpiricalEnvironmentError(
            f"qualified latency support has {interior_holes} interior holes"
        )
    binding = _build_binding(
        sidecar=sidecar,
        surface=surface,
        network=network,
        radio_store=radio_store,
        contexts=contexts,
        normalization=normalization,
        freshness=freshness,
    )
    report_fields = dict(
        status="GO_D1_FIT_ONLY_ONE_STEP_EXPECTED_UTILITY",
        fit_scene_count=inventory.fit_scene_count,
        rewardable_fit_context_count=inventory.rewardable_scene_count,
        excluded_action_independent_invalid_count=(
            inventory.action_independent_invalid_scene_count
        ),
        invalidity_is_action_independent=True,
        minimum_supported_fit_payload_bytes=payload_min,
        maximum_supported_fit_payload_bytes=payload_max,
        supported_endpoint_query_count=endpoint_count,
        exhaustive_network_endpoint_profile_query_count=network_endpoint_count,
        network_latency_interior_support_hole_count=interior_holes,
        radio_source_row_count=radio_store.source_row_count,
        radio_joint_valid_row_count=radio_store.joint_valid_row_count,
        radio_cross_profile_aliased_row_count=(
            radio_store.cross_profile_aliased_row_count
        ),
        radio_profile_counts=dict(radio_store.profile_counts),
        policy_feature_count=len(POLICY_FEATURE_ORDER),
        all_genesis_ages_zero=True,
        conditional_uplink_p50_range_ms=(min(p50_values), max(p50_values)),
        conditional_uplink_p95_range_ms=(min(p95_values), max(p95_values)),
        conditional_uplink_p99_range_ms=(min(p99_values), max(p99_values)),
        total_proxy_p50_range_ms=(
            fixed_stage_latency_ms() + min(p50_values),
            fixed_stage_latency_ms() + max(p50_values),
        ),
        total_proxy_p95_range_ms=(
            fixed_stage_latency_ms() + min(p95_values),
            fixed_stage_latency_ms() + max(p95_values),
        ),
        total_proxy_p99_range_ms=(
            fixed_stage_latency_ms() + min(p99_values),
            fixed_stage_latency_ms() + max(p99_values),
        ),
        expected_reward_range=(min(reward_values), max(reward_values)),
        expected_reward_nondegenerate=(
            max(reward_values) - min(reward_values) > 1e-12
        ),
        modeled_budget_miss_counts_p50_p95_p99=tuple(budget_miss_counts),
        surface_qualification_report_sha256=qualification.report_sha256,
        surface_qualification_overall_status=str(
            qualification_document["status"]
        ),
        surface_q_perc_fit_held_qualified=True,
        surface_q_seg_fit_held_qualified=False,
        surface_held_payload_qualified=True,
        surface_full_component_qualified=False,
        environment_binding_sha256=binding.canonical_sha256(),
        disclosures=(
            "FIT_ONLY_REWARDS; HELD_SCENE_HAS_NO_TRAINING API OR TYPE",
            "HELD_SCENES_WERE_CONSULTED_FOR_OUT_OF_SAMPLE_SURFACE_VALIDATION_"
            "AND_AUDIT; THEY_ARE_NOT_AN_UNTOUCHED_FINAL_TEST_SET",
            "36_SCENES_EXCLUDED_BECAUSE_DIRECT_QPERC_INVALID_FOR_ALL_132_ACTION_ROWS",
            "RADIO_CONTEXT_IS_OAI_CALIBRATED_SIMULATOR_TESTBED_ONLY",
            "NETWORK_PROFILE_TARGET_TRACE_AND_FRAME_IDENTITIES_ARE_POLICY_HIDDEN",
            "NETWORK_PROFILE_PRIOR_IS_SYNTHETIC_UNIFORM_AND_ROWS_ARE_UNIFORM_"
            "WITHIN_PROFILE; OVERLAPPING_SNR_MCS_MAKES_PROFILE_PARTIALLY_"
            "OBSERVABLE_AND_ANY_PROFILE_ORACLE_IS_PRIVILEGED",
            "ONLY_SI_P40_SNR_AND_MCS_VARY_IN_D1_GENESIS; BSR_FRESHNESS_AND_"
            "PREVIOUS_OUTCOME_FEATURES_ARE_CONSTANT",
            "DETERMINISTIC_EXPECTED_UTILITY; NO_BERNOULLI_TERMINAL_SAMPLING",
            "P50_PLUS_113MS_IS_A_PROXY; NO_TIMEOUT_PROBABILITY_IS_INFERRED",
            "P_EDGE_ADMISSION_GIVEN_SENT_IS_NOT_END_TO_FEEDBACK_SUCCESS;_"
            "ADMITTED_BRANCH_ASSUMES_EVALUATION_AND_ACK_COMPLETION_BECAUSE_"
            "THEIR_SUCCESS_PROBABILITIES_ARE_UNAVAILABLE",
            "MODELED_BUDGET_MISS_IS_DIAGNOSTIC_AND_IS_NOT_CENSORED",
            "FIT_ENDPOINT_SUPPORT_IS_EXHAUSTIVE; SURFACE_LOADER_PROVES_"
            "STRICT_PAYLOAD_MONOTONICITY_AND_PREFLIGHT_EXHAUSTIVELY_CHECKS_"
            "EVERY_OVERLAPPING_NETWORK_LATENCY_SUPPORT_KNOT_INTERVAL_HAS_NO_"
            "INTERIOR_HOLES",
        ),
    )
    report_document = dict(report_fields)
    report_document["radio_profile_counts"] = dict(
        report_document["radio_profile_counts"]
    )
    return EmpiricalPreflightReportV1(
        **report_fields,
        report_sha256=canonical_sha256(report_document),
        _attestation=_PREFLIGHT_ATTESTATION,
    )


class EmpiricalOneStepEnvironmentV1:
    """Fit-only one-step environment with local deterministic RNG streams."""

    def __init__(
        self,
        *,
        sidecar: CorrectedP40Sidecar,
        surface: EmpiricalQualitySurface,
        network: PayloadNetworkSurrogate,
        radio_store: OaiRadioCalibrationStoreV1,
        contexts: FitContextIndexV1,
        normalization: StateNormalizationSpecV1,
        freshness: StateFreshnessPolicyV1,
        seed: int,
        preflight: EmpiricalPreflightReportV1,
    ) -> None:
        if type(seed) is not int:
            raise EmpiricalEnvironmentError("seed must be an exact integer")
        self._sidecar = sidecar
        self._surface = surface
        self._network = network
        self._prediction_session: PrevalidatedPredictionSession = (
            network.prevalidated_prediction_session()
        )
        self._radio_store = radio_store
        self._contexts = contexts
        self.normalization = normalization
        self.freshness = freshness
        self.binding = _build_binding(
            sidecar=sidecar,
            surface=surface,
            network=network,
            radio_store=radio_store,
            contexts=contexts,
            normalization=normalization,
            freshness=freshness,
        )
        preflight.require_registered()
        if preflight.environment_binding_sha256 != self.binding.canonical_sha256():
            raise EmpiricalEnvironmentError("preflight/environment binding mismatch")
        self.preflight = preflight
        self._seed = seed
        self._context_rng = random.Random(
            int.from_bytes(
                hashlib.sha256(f"{seed}:D1_CONTEXT_V1".encode("ascii")).digest(),
                "big",
            )
        )
        self._radio_sampler = RadioContextSamplerV1(radio_store, seed=seed)
        self._reset_count = 0
        self._active_context: Optional[FitTrainingContextV1] = None
        self._active_radio: Optional[RadioContextDrawV1] = None
        self._active = False

    @classmethod
    def load_registered(
        cls, *, seed: int, project_root: Optional[Path] = None
    ) -> "EmpiricalOneStepEnvironmentV1":
        root = (
            Path(__file__).resolve().parents[2]
            if project_root is None
            else Path(project_root).resolve(strict=True)
        )
        sidecar = load_exact_corrected_p40_sidecar(root=root)
        surface = load_empirical_quality_surface(sidecar, project_root=root)
        try:
            network = build_payload_network_surrogate()
            radio_store = OaiRadioCalibrationStoreV1.load_registered(project_root=root)
            contexts = FitContextIndexV1.load_registered(
                surface=surface, sidecar=sidecar, project_root=root
            )
            normalization = build_registered_normalization_spec()
            freshness = build_registered_freshness_policy()
            preflight = run_empirical_d1_preflight(
                sidecar=sidecar,
                surface=surface,
                network=network,
                radio_store=radio_store,
                contexts=contexts,
                normalization=normalization,
                freshness=freshness,
            )
            return cls(
                sidecar=sidecar,
                surface=surface,
                network=network,
                radio_store=radio_store,
                contexts=contexts,
                normalization=normalization,
                freshness=freshness,
                seed=seed,
                preflight=preflight,
            )
        except Exception:
            surface.close()
            raise

    def close(self) -> None:
        self._surface.close()

    def state_dict(self) -> EmpiricalEnvironmentStateV1:
        if self._active:
            raise EmpiricalEnvironmentError(
                "checkpoint is allowed only between completed one-step episodes"
            )
        return EmpiricalEnvironmentStateV1(
            environment_binding_sha256=self.binding.canonical_sha256(),
            master_seed=self._seed,
            reset_count=self._reset_count,
            context_rng_state=self._context_rng.getstate(),
            radio_sampler_state=self._radio_sampler.state_dict(),
        )

    def load_state_dict(self, state: EmpiricalEnvironmentStateV1) -> None:
        if self._active:
            raise EmpiricalEnvironmentError(
                "cannot restore over an active one-step episode"
            )
        if type(state) is not EmpiricalEnvironmentStateV1:
            raise EmpiricalEnvironmentError(
                "environment state must be EmpiricalEnvironmentStateV1"
            )
        if state.environment_binding_sha256 != self.binding.canonical_sha256():
            raise EmpiricalEnvironmentError("checkpoint D1 binding mismatch")
        if state.master_seed != self._seed:
            raise EmpiricalEnvironmentError("checkpoint master-seed mismatch")
        if type(state.reset_count) is not int or state.reset_count < 0:
            raise EmpiricalEnvironmentError("invalid checkpoint reset count")
        try:
            context_probe = random.Random()
            context_probe.setstate(state.context_rng_state)
        except (TypeError, ValueError) as exc:
            raise EmpiricalEnvironmentError("invalid context RNG state") from exc
        self._radio_sampler.validate_state_dict(state.radio_sampler_state)
        if state.radio_sampler_state.draw_count != state.reset_count:
            raise EmpiricalEnvironmentError("radio/reset draw-count mismatch")
        # Everything has been validated into disposable RNGs.  Commit only
        # after all failure points, so malformed checkpoints are atomic no-ops.
        self._context_rng.setstate(state.context_rng_state)
        self._radio_sampler.load_state_dict(state.radio_sampler_state)
        self._reset_count = state.reset_count

    def _sample_fit_context(self) -> FitTrainingContextV1:
        total = sum(item.sampling_weight for item in self._contexts.contexts)
        threshold = self._context_rng.random() * total
        cumulative = 0.0
        for item in self._contexts.contexts:
            cumulative += item.sampling_weight
            if threshold < cumulative:
                return item
        return self._contexts.contexts[-1]

    def reset(self) -> EmpiricalPolicyObservationV1:
        if self._active:
            raise EmpiricalEnvironmentError(
                "active one-step episode must be stepped before reset"
            )
        context = self._sample_fit_context()
        ordinal = self._reset_count
        binding_sha = self.binding.canonical_sha256()
        session_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{binding_sha}:{self._seed}:{ordinal}:session"))
        lineage_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{binding_sha}:{self._seed}:{ordinal}:lineage"))
        observed_ns = 1_000_000_000 + ordinal
        radio = self._radio_sampler.sample(
            observed_ns=observed_ns, control_session_id=session_uuid
        )
        controller = RewardTicketController(
            session_uuid, controller_lineage_uuid=lineage_uuid
        )
        genesis = controller.authorize_episode_start(
            first_decision_seq=0,
            first_tensor_seq=0,
            first_carla_frame_id=context.frame_id,
            state_observed_ns=observed_ns,
        )
        episode_start = EpisodeStartProofV1.from_controller_genesis(
            genesis,
            source_id="EMPIRICAL_CONTEXTUAL_D1_ONE_STEP_GENESIS",
            source_sha256=binding_sha,
        )
        scene = SceneObservationV1(
            sample=SceneDescriptorSample(
                camera_si=context.camera_si, radar_p40=context.radar_p40
            ),
            measured_ns=observed_ns,
            carla_frame_id=context.frame_id,
            source_id="HASH_BOUND_CORRECTED_P40_PLUS_FIT_CAMERA_SI",
            source_sha256=canonical_sha256(
                {
                    "corrected_p40_binding_sha256": self._sidecar.binding_sha256,
                    "fit_context_index_sha256": self._contexts.canonical_sha256,
                    "record": "d1_scene_si_p40_source_binding_v1",
                    "selection_sha256": self._contexts.source_selection_sha256,
                }
            ),
        )
        state = CausalStateV1(
            scene=scene,
            radio=radio.observation,
            session_uuid=session_uuid,
            observed_ns=observed_ns,
            tensor_seq=0,
            carla_frame_id=context.frame_id,
            previous=None,
            episode_start=episode_start,
        )
        if any(age != 0 for age in state.measurement_ages_ns.values()):
            raise EmpiricalEnvironmentError("D1 genesis state has non-zero age")
        features = build_policy_features(state, self.normalization, self.freshness)
        observation = EmpiricalPolicyObservationV1(
            values=features.as_tuple(),
            policy_feature_order=tuple(POLICY_FEATURE_ORDER),
            environment_binding_sha256=binding_sha,
            normalization_spec_sha256=self.normalization.canonical_sha256(),
            freshness_policy_sha256=self.freshness.canonical_sha256(),
        )
        self._active_context = context
        self._active_radio = radio
        self._active = True
        self._reset_count += 1
        return observation

    def step(self, action: EmpiricalActionV1) -> EmpiricalStepResultV1:
        if not self._active or self._active_context is None or self._active_radio is None:
            raise EmpiricalEnvironmentError("reset is required before step")
        if type(action) is not EmpiricalActionV1:
            raise EmpiricalEnvironmentError("action must be exact EmpiricalActionV1")
        checked = require_supported_action(action.mode_id, action.q_e4)
        context = self._active_context
        radio = self._active_radio
        self._active = False
        self._active_context = None
        self._active_radio = None

        query = self._surface.query_fit_q_e4(
            context.sample_id, checked.mode_id, checked.q_e4
        )
        q_component = query.policy.component(DIRECT_QUALITY_COMPONENT)
        payload = float(query.policy.payload.total_transmitted_bytes)
        datagrams = math.ceil(payload / UDP_PAYLOAD_CAPACITY_BYTES)
        base_audit = dict(
            sample_id=context.sample_id,
            episode_id=context.episode_id,
            frame_id=context.frame_id,
            hidden_network_profile=radio.hidden_profile,
            hidden_radio_csv_row_number=radio.hidden_csv_row_number,
            hidden_trace_id=radio.hidden_trace_id,
            hidden_trace_step_index=radio.hidden_trace_step_index,
            hidden_target_snr_db=radio.hidden_target_snr_db,
            hidden_radio_row_sha256=radio.hidden_row_sha256,
            surface_evidence_status=query.policy.evidence_status,
            total_transmitted_bytes=payload,
            datagram_count=datagrams,
            network_evidence_class=NETWORK_EVIDENCE_CLASS,
            utility_spec_sha256=PILOT_UTILITY_SPEC_SHA256,
            executed_mode_id=checked.mode_id,
            executed_q_e4=checked.q_e4,
        )
        if not q_component.valid or q_component.value is None:
            outcome = self._unavailable_outcome("QUALITY_UNDEFINED_NO_REWARD")
            return EmpiricalStepResultV1(
                policy=outcome, audit=EmpiricalStepAuditV1(**base_audit)
            )
        try:
            prediction = self._prediction_session.predict(
                network_profile=radio.hidden_profile,
                payload_bytes=payload,
                datagram_count=datagrams,
            )
            latency = prediction.conditional_retained_survivor_latency_model()
        except ExtrapolationRefusedError:
            outcome = self._unavailable_outcome(
                "NETWORK_OR_CONDITIONAL_LATENCY_UNSUPPORTED_NO_REWARD",
                q_perc=float(q_component.value),
            )
            return EmpiricalStepResultV1(
                policy=outcome, audit=EmpiricalStepAuditV1(**base_audit)
            )
        p_edge_admission = (
            prediction.p_complete_reassembly_given_sent
            * prediction.p_edge_admission_given_reassembled
        )
        if not math.isclose(
            p_edge_admission,
            prediction.p_edge_admission_given_sent,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise EmpiricalEnvironmentError("network probability factorization drift")
        latency_proxy = fixed_stage_latency_ms() + latency.p50_ms
        latency_proxy_p95 = fixed_stage_latency_ms() + latency.p95_ms
        latency_proxy_p99 = fixed_stage_latency_ms() + latency.p99_ms
        reward = PILOT_UTILITY_SPEC.expected_utility(
            p_edge_admission_given_sent=p_edge_admission,
            q_perc=float(q_component.value),
            latency_proxy_ms=latency_proxy,
        )
        outcome = EmpiricalOutcomeV1(
            reward=reward,
            status="MODELED_EXPECTED_UTILITY_DEFINED",
            terminated=True,
            truncated=False,
            q_perc=float(q_component.value),
            p_complete_reassembly_given_sent=(
                prediction.p_complete_reassembly_given_sent
            ),
            p_edge_admission_given_reassembled=(
                prediction.p_edge_admission_given_reassembled
            ),
            p_edge_admission_given_sent=p_edge_admission,
            conditional_feature_uplink_p50_ms=latency.p50_ms,
            conditional_feature_uplink_p95_ms=latency.p95_ms,
            conditional_feature_uplink_p99_ms=latency.p99_ms,
            fixed_stage_latency_ms=fixed_stage_latency_ms(),
            fixed_latency_stages_ms=FIXED_END_TO_FEEDBACK_STAGES_MS,
            latency_proxy_ms=latency_proxy,
            latency_proxy_p95_ms=latency_proxy_p95,
            latency_proxy_p99_ms=latency_proxy_p99,
            deadline_ms=PILOT_UTILITY_SPEC.deadline_ms,
            modeled_budget_miss=latency_proxy > PILOT_UTILITY_SPEC.deadline_ms,
            modeled_budget_miss_p95=(
                latency_proxy_p95 > PILOT_UTILITY_SPEC.deadline_ms
            ),
            modeled_budget_miss_p99=(
                latency_proxy_p99 > PILOT_UTILITY_SPEC.deadline_ms
            ),
            estimator=PILOT_UTILITY_SPEC.estimator,
            service_non_admission_semantics=(
                "MODELED_EDGE_NON_ADMISSION_MASS_NOT_AN_AUTHORITATIVE_PER_"
                "FRAME_SYSTEM_FAILURE; CONDITIONAL_ADMITTED_BRANCH_ASSUMES_"
                "EVALUATION_AND_ACK_COMPLETE_BECAUSE_SUCCESS_PROBABILITIES_"
                "ARE_UNAVAILABLE"
            ),
            timeout_probability_status=(
                "NOT_INFERRED_FROM_CONDITIONAL_P50_OR_BUDGET_MISS"
            ),
        )
        return EmpiricalStepResultV1(
            policy=outcome, audit=EmpiricalStepAuditV1(**base_audit)
        )

    @staticmethod
    def _unavailable_outcome(
        status: str, *, q_perc: Optional[float] = None
    ) -> EmpiricalOutcomeV1:
        return EmpiricalOutcomeV1(
            reward=None,
            status=status,
            terminated=True,
            truncated=False,
            q_perc=q_perc,
            p_complete_reassembly_given_sent=None,
            p_edge_admission_given_reassembled=None,
            p_edge_admission_given_sent=None,
            conditional_feature_uplink_p50_ms=None,
            conditional_feature_uplink_p95_ms=None,
            conditional_feature_uplink_p99_ms=None,
            fixed_stage_latency_ms=fixed_stage_latency_ms(),
            fixed_latency_stages_ms=FIXED_END_TO_FEEDBACK_STAGES_MS,
            latency_proxy_ms=None,
            latency_proxy_p95_ms=None,
            latency_proxy_p99_ms=None,
            deadline_ms=PILOT_UTILITY_SPEC.deadline_ms,
            modeled_budget_miss=None,
            modeled_budget_miss_p95=None,
            modeled_budget_miss_p99=None,
            estimator=PILOT_UTILITY_SPEC.estimator,
            service_non_admission_semantics=(
                "UNSUPPORTED_EVIDENCE_IS_NOT_RELABELLED_AS_SYSTEM_FAILURE"
            ),
            timeout_probability_status="NOT_INFERRED",
        )


def load_registered_d1_environment(
    *, seed: int, project_root: Optional[Path] = None
) -> EmpiricalOneStepEnvironmentV1:
    return EmpiricalOneStepEnvironmentV1.load_registered(
        seed=seed, project_root=project_root
    )
