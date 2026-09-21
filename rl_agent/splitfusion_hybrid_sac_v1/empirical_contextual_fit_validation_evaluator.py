"""Post-hoc actor evaluation on the frozen reward-held fit-validation panel.

This module is evaluation-only.  It never constructs replay, an optimizer, or
a trainer, and it never samples a scene or radio row.  Each actor checkpoint is
executed deterministically (argmax joint mode and that mode's conditional-mean
``q``) on the 340 entries of the registered panel.

The panel is *reward held* relative to preliminary training.  It is not an
untouched final/generalization set: D1's unsupervised normalization and quality
surface qualification consulted the complete fit covariate population before
the reward-blind train/fit-validation partition was registered.

The radio calibration stores median MCS, including exact half-integers.  The
training sampler resolves those ties with a local seeded RNG, but the frozen
panel does not retain a sampled outcome.  Evaluation therefore uses a separate
identity-derived SHA-256 bit.  This is deterministic, independent of ambient
RNG state, and explicitly bound in the output.  It is not represented as a
measured MCS realization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

from .empirical_contextual_baseline_runner import (
    BASELINE_PROVENANCE_LABEL,
    PHASE_LABEL as BASELINE_PHASE_LABEL,
    REGISTERED_BASELINE_CONFIG,
)
from .empirical_contextual_contract import (
    DIRECT_QUALITY_COMPONENT,
    MODELED_SMOKE_SUPPORT,
    PILOT_UTILITY_SPEC,
    PILOT_UTILITY_SPEC_SHA256,
    EmpiricalActionV1,
    fixed_stage_latency_ms,
    require_supported_action,
)
from .empirical_contextual_environment import (
    EmpiricalOneStepEnvironmentV1,
    EmpiricalPolicyObservationV1,
    FitTrainingContextV1,
)
from .empirical_contextual_fit_partition import (
    FIT_VALIDATION_SPLIT,
    REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
)
from .empirical_contextual_fit_validation_panel import (
    REGISTERED_FIT_VALIDATION_PANEL_SHA256,
    FitValidationPanelEntryV1,
    FitValidationPanelManifestV1,
    load_registered_fit_validation_panel,
)
from .empirical_contextual_smoke_runner import (
    EmpiricalSmokeCheckpointV1,
    _derive_seed,
    _hash_state,
)
from .empirical_radio_context import (
    GENESIS_BSR_JUSTIFICATION,
    MCS_TABLE_ID,
    MCS_TABLE_MAX_INDEX,
    RadioCalibrationRowV1,
    RadioContextDrawV1,
)
from .hybrid_sac_models import HybridSacModelConfig, build_actor
from .payload_network_surrogate import UDP_PAYLOAD_CAPACITY_BYTES
from .reward_ticket_controller import RewardTicketController
from .scene_descriptors import SceneDescriptorSample
from .state_reward_transition_contract import (
    BsrReportType,
    BsrReportV1,
    BsrScope,
    BsrSource,
    CausalStateV1,
    EpisodeStartProofV1,
    POLICY_FEATURE_ORDER,
    RadioEventProvenanceV1,
    RadioObservationV1,
    RadioSourceWall,
    SceneObservationV1,
    build_policy_features,
)
from .transaction_identity import canonical_json_bytes, canonical_sha256

__all__ = [
    "EVALUATION_CHECKPOINT_UPDATES",
    "EVALUATION_PROVENANCE_LABEL",
    "EVALUATION_SCOPE_DISCLOSURE",
    "MCS_ROUNDING_RULE_SHA256",
    "FitValidationActorEvaluatorV1",
    "FitValidationEvaluationError",
    "FitValidationEvaluationRowV1",
    "LoadedActorIdentityV1",
    "evaluate_registered_campaign",
]


EVALUATION_SCHEMA = "splitfusion.hybrid_sac_fit_validation_actor_evaluation.v1"
EVALUATION_PROVENANCE_LABEL = (
    "POSTHOC_REWARD_HELD_FIT_VALIDATION_DETERMINISTIC_ACTOR_EXECUTION_"
    "NOT_UNTOUCHED_FINAL_OR_GENERALIZATION_EVIDENCE"
)
EVALUATION_SCOPE_DISCLOSURE = (
    "Reward-held fit-validation evaluation of a preliminary train-split "
    "baseline. Normalization and earlier surface qualification consulted all "
    "fit covariates. Results do not establish final-test generalization, "
    "deployment performance, or convergence and must not select or tune a "
    "checkpoint before a separately registered selection rule exists."
)
EVALUATION_CHECKPOINT_UPDATES: Tuple[int, ...] = (
    0,
    500,
    1000,
    1500,
    2000,
    2500,
    3000,
    3500,
    4000,
    4500,
    5000,
)
_EVALUATION_NAMESPACE = uuid.UUID("4efe68cc-ee86-55f2-b9e6-e8d5f50471ed")
_MCS_ROUNDING_RULE = {
    "bit_rule": "SHA256(CANONICAL_JSON(document))[0]_AND_1; 0=FLOOR, 1=CEIL",
    "document_fields": [
        "panel_sha256",
        "panel_index",
        "radio_csv_row_number",
        "radio_row_sha256",
        "mcs_median",
        "rule_schema",
    ],
    "integer_semantics": "UNCHANGED",
    "purpose": "EVALUATION_ONLY_IDENTITY_DERIVED_HALF_INTEGER_MCS",
    "rule_schema": "splitfusion.fit_validation_mcs_rounding.v1",
}
MCS_ROUNDING_RULE_SHA256 = canonical_sha256(_MCS_ROUNDING_RULE)


class FitValidationEvaluationError(RuntimeError):
    """A campaign, panel, checkpoint, or evaluation invariant failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if type(document) is not dict:
        raise FitValidationEvaluationError(f"{path} is not a JSON object")
    return document


def _load_checkpoint(path: Path) -> EmpiricalSmokeCheckpointV1:
    with path.open("rb") as stream:
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
    if type(checkpoint) is not EmpiricalSmokeCheckpointV1:
        raise FitValidationEvaluationError("checkpoint contains a foreign object")
    checkpoint.require_valid()
    return checkpoint


def _round_panel_mcs(
    entry: FitValidationPanelEntryV1, row: RadioCalibrationRowV1
) -> Tuple[int, str]:
    numeric = float(row.mcs_median)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= MCS_TABLE_MAX_INDEX:
        raise FitValidationEvaluationError("panel MCS is outside table 0")
    floor = math.floor(numeric)
    fraction = numeric - floor
    if fraction == 0.0:
        return floor, "EXACT_INTEGER_NO_ROUNDING"
    if fraction != 0.5:
        raise FitValidationEvaluationError("panel MCS is not integer/half-integer")
    document = {
        "mcs_median": numeric,
        "panel_index": entry.panel_index,
        "panel_sha256": REGISTERED_FIT_VALIDATION_PANEL_SHA256,
        "radio_csv_row_number": row.csv_row_number,
        "radio_row_sha256": row.row_sha256,
        "rule_schema": _MCS_ROUNDING_RULE["rule_schema"],
    }
    bit = hashlib.sha256(canonical_json_bytes(document)).digest()[0] & 1
    return (
        floor + bit,
        "IDENTITY_DERIVED_SHA256_FLOOR_CEIL_FOR_HALF_INTEGER_EVALUATION_ONLY",
    )


@dataclass(frozen=True, slots=True)
class LoadedActorIdentityV1:
    seed: int
    update_index: int
    actor_state_sha256: str
    checkpoint_canonical_sha256: str
    checkpoint_file_sha256: Optional[str]
    source: str

    def to_canonical_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FitValidationEvaluationRowV1:
    seed: int
    update_index: int
    actor_state_sha256: str
    checkpoint_canonical_sha256: str
    panel_index: int
    scene_rank: int
    sample_id: str
    episode_id: str
    frame_id: int
    network_profile: str
    radio_csv_row_number: int
    radio_row_sha256: str
    achieved_pusch_snr_median_db: float
    mcs_median: float
    rounded_mcs_index: int
    mcs_rounding_status: str
    camera_si: float
    radar_p40: float
    executed_mode_id: int
    requested_q: float
    executed_q_e4: int
    q_perc: float
    total_transmitted_bytes: float
    datagram_count: int
    p_complete_reassembly_given_sent: float
    p_edge_admission_given_reassembled: float
    p_edge_admission_given_sent: float
    conditional_feature_uplink_p50_ms: float
    conditional_feature_uplink_p95_ms: float
    conditional_feature_uplink_p99_ms: float
    fixed_stage_latency_ms: float
    latency_proxy_ms: float
    latency_proxy_p95_ms: float
    latency_proxy_p99_ms: float
    modeled_budget_miss: bool
    modeled_budget_miss_p95: bool
    modeled_budget_miss_p99: bool
    reward: float
    result_status: str

    def to_canonical_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FitValidationActorEvaluatorV1:
    """Read-only D1 view for exact fixed-panel actor evaluation."""

    def __init__(self, *, project_root: Optional[Path] = None) -> None:
        self._global_python_rng = random.getstate()
        self._global_torch_rng = torch.get_rng_state().clone()
        self._cuda_initialized = torch.cuda.is_initialized()
        self.panel: FitValidationPanelManifestV1 = (
            load_registered_fit_validation_panel()
        )
        self.environment = EmpiricalOneStepEnvironmentV1.load_registered(
            seed=0, project_root=project_root
        )
        self._closed = False
        binding_sha = self.environment.binding.canonical_sha256()
        if self.panel.d1_pilot_binding_sha256 != binding_sha:
            self.environment.close()
            self._closed = True
            raise FitValidationEvaluationError("panel/D1 binding mismatch")
        self._contexts = {
            item.sample_id: item for item in self.environment._contexts.contexts
        }
        self._radios = {
            item.csv_row_number: item for item in self.environment._radio_store.rows
        }
        if len(self._contexts) != 476 or len(self._radios) != 399:
            self.close()
            raise FitValidationEvaluationError("D1 evaluation inventory drift")
        self._assert_no_runtime_side_effects()

    def __enter__(self) -> "FitValidationActorEvaluatorV1":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self.environment.close()
            self._closed = True

    def _assert_open(self) -> None:
        if self._closed:
            raise FitValidationEvaluationError("evaluator is closed")

    def _assert_no_runtime_side_effects(self) -> None:
        if random.getstate() != self._global_python_rng:
            raise FitValidationEvaluationError("evaluation advanced global Python RNG")
        if not torch.equal(torch.get_rng_state(), self._global_torch_rng):
            raise FitValidationEvaluationError("evaluation advanced global Torch RNG")
        if not self._cuda_initialized and torch.cuda.is_initialized():
            raise FitValidationEvaluationError("evaluation initialized CUDA")

    def _join_entry(
        self, entry: FitValidationPanelEntryV1
    ) -> Tuple[FitTrainingContextV1, RadioCalibrationRowV1]:
        if type(entry) is not FitValidationPanelEntryV1:
            raise FitValidationEvaluationError("entry has a foreign type")
        context = self._contexts.get(entry.scene_sample_id)
        radio = self._radios.get(entry.radio_csv_row_number)
        if context is None or radio is None:
            raise FitValidationEvaluationError("panel entry failed D1 identity join")
        if (
            context.episode_id != entry.scene_episode_id
            or context.frame_id != entry.scene_frame_id
            or radio.network_profile != entry.network_profile
            or radio.trace_id != entry.radio_trace_id
            or radio.trace_step_index != entry.radio_trace_step_index
            or radio.row_sha256 != entry.radio_row_sha256
            or entry.scene_split != FIT_VALIDATION_SPLIT
            or entry.radio_split != FIT_VALIDATION_SPLIT
        ):
            raise FitValidationEvaluationError("panel/D1 identity fields drifted")
        return context, radio

    def _make_observation(
        self, entry: FitValidationPanelEntryV1
    ) -> Tuple[EmpiricalPolicyObservationV1, FitTrainingContextV1, RadioContextDrawV1]:
        context, radio_row = self._join_entry(entry)
        binding_sha = self.environment.binding.canonical_sha256()
        identity = f"{REGISTERED_FIT_VALIDATION_PANEL_SHA256}:{entry.panel_index}"
        session_uuid = str(uuid.uuid5(_EVALUATION_NAMESPACE, identity + ":session"))
        lineage_uuid = str(uuid.uuid5(_EVALUATION_NAMESPACE, identity + ":lineage"))
        observed_ns = 1_000_000_000 + entry.panel_index
        mcs_index, rounding_status = _round_panel_mcs(entry, radio_row)
        epoch = "d1-oai-calibrated-simulator-epoch"
        event = RadioEventProvenanceV1(
            source_wall=RadioSourceWall.SIMULATOR_TESTBED,
            source_event_id=f"calibration-csv-row-{radio_row.csv_row_number}",
            source_event_index=radio_row.csv_row_number,
            source_event_timestamp_ns=observed_ns,
            collector_ingest_wall_time_ns=observed_ns,
            collector_ingest_monotonic_ns=observed_ns,
            ran_epoch_id=epoch,
            control_session_id=session_uuid,
            raw_event_sha256=radio_row.row_sha256,
        )
        bsr_sha = canonical_sha256(
            {
                "justification": GENESIS_BSR_JUSTIFICATION,
                "lcg_bytes": [0] * 8,
                "observed_ns": observed_ns,
                "record": "d1_genesis_bsr_v1",
            }
        )
        bsr_event = RadioEventProvenanceV1(
            source_wall=RadioSourceWall.SIMULATOR_TESTBED,
            source_event_id="d1-genesis-empty-bsr",
            source_event_index=0,
            source_event_timestamp_ns=observed_ns,
            collector_ingest_wall_time_ns=observed_ns,
            collector_ingest_monotonic_ns=observed_ns,
            ran_epoch_id=epoch,
            control_session_id=session_uuid,
            raw_event_sha256=bsr_sha,
        )
        bsr = BsrReportV1(
            lcg_bytes=(0,) * 8,
            valid_mask=(True,) * 8,
            missing_reasons=(None,) * 8,
            scope=BsrScope.ALL_GROUPS_LATEST,
            logical_channel_group=0,
            report_type=BsrReportType.SIMULATOR_VECTOR,
            source=BsrSource.SIMULATOR_TESTBED_PRIVILEGED,
            measured_ns=observed_ns,
            event=bsr_event,
        )
        radio_observation = RadioObservationV1.for_simulator_testbed(
            achieved_snr_db=radio_row.achieved_pusch_snr_median_db,
            snr_measured_ns=observed_ns,
            mcs_index=mcs_index,
            mcs_table_id=MCS_TABLE_ID,
            mcs_measured_ns=observed_ns,
            bsr_bytes=0,
            bsr_scope=BsrScope.ALL_GROUPS_LATEST,
            bsr_logical_channel_group=0,
            bsr_measured_ns=observed_ns,
            snr_event=event,
            mcs_event=event,
            bsr_report=bsr,
            source_id="OAI_REPLAY_CALIBRATED_SIMULATOR_TESTBED_CONTEXT",
            source_sha256=self.environment._radio_store.source_sha256,
        )
        radio = RadioContextDrawV1(
            observation=radio_observation,
            hidden_profile=radio_row.network_profile,
            hidden_csv_row_number=radio_row.csv_row_number,
            hidden_trace_id=radio_row.trace_id,
            hidden_trace_step_index=radio_row.trace_step_index,
            hidden_target_snr_db=radio_row.target_snr_db,
            hidden_row_sha256=radio_row.row_sha256,
            mcs_median=radio_row.mcs_median,
            rounded_mcs_index=mcs_index,
            rounding_status=rounding_status,
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
                    "corrected_p40_binding_sha256": (
                        self.environment._sidecar.binding_sha256
                    ),
                    "fit_context_index_sha256": (
                        self.environment._contexts.canonical_sha256
                    ),
                    "record": "d1_scene_si_p40_source_binding_v1",
                    "selection_sha256": (
                        self.environment._contexts.source_selection_sha256
                    ),
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
            raise FitValidationEvaluationError("evaluation genesis age is non-zero")
        features = build_policy_features(
            state, self.environment.normalization, self.environment.freshness
        )
        observation = EmpiricalPolicyObservationV1(
            values=features.as_tuple(),
            policy_feature_order=tuple(POLICY_FEATURE_ORDER),
            environment_binding_sha256=binding_sha,
            normalization_spec_sha256=(
                self.environment.normalization.canonical_sha256()
            ),
            freshness_policy_sha256=self.environment.freshness.canonical_sha256(),
        )
        return observation, context, radio

    def evaluate_entry(
        self,
        *,
        actor: torch.nn.Module,
        actor_identity: LoadedActorIdentityV1,
        entry: FitValidationPanelEntryV1,
    ) -> FitValidationEvaluationRowV1:
        self._assert_open()
        observation, context, radio = self._make_observation(entry)
        state = torch.tensor([observation.values], dtype=torch.float32)
        execution = actor.deterministic_execution(state)
        mode_id = int(execution.mode_index.item())
        q_e4 = int(execution.q_e4.item())
        action: EmpiricalActionV1 = require_supported_action(mode_id, q_e4)
        if self.environment._active:
            raise FitValidationEvaluationError("D1 evaluation view is unexpectedly active")
        self.environment._active_context = context
        self.environment._active_radio = radio
        self.environment._active = True
        try:
            result = self.environment.step(action)
        except Exception:
            self.environment._active_context = None
            self.environment._active_radio = None
            self.environment._active = False
            raise
        outcome = result.policy
        if (
            outcome.reward is None
            or outcome.q_perc is None
            or outcome.p_complete_reassembly_given_sent is None
            or outcome.p_edge_admission_given_reassembled is None
            or outcome.p_edge_admission_given_sent is None
            or outcome.conditional_feature_uplink_p50_ms is None
            or outcome.conditional_feature_uplink_p95_ms is None
            or outcome.conditional_feature_uplink_p99_ms is None
            or outcome.latency_proxy_ms is None
            or outcome.latency_proxy_p95_ms is None
            or outcome.latency_proxy_p99_ms is None
            or outcome.modeled_budget_miss is None
            or outcome.modeled_budget_miss_p95 is None
            or outcome.modeled_budget_miss_p99 is None
        ):
            raise FitValidationEvaluationError(
                f"panel action has unavailable D1 reward: {outcome.status}"
            )
        row = FitValidationEvaluationRowV1(
            seed=actor_identity.seed,
            update_index=actor_identity.update_index,
            actor_state_sha256=actor_identity.actor_state_sha256,
            checkpoint_canonical_sha256=(
                actor_identity.checkpoint_canonical_sha256
            ),
            panel_index=entry.panel_index,
            scene_rank=entry.scene_rank,
            sample_id=context.sample_id,
            episode_id=context.episode_id,
            frame_id=context.frame_id,
            network_profile=radio.hidden_profile,
            radio_csv_row_number=radio.hidden_csv_row_number,
            radio_row_sha256=radio.hidden_row_sha256,
            achieved_pusch_snr_median_db=(
                radio.observation.achieved_snr_db
            ),
            mcs_median=radio.mcs_median,
            rounded_mcs_index=radio.rounded_mcs_index,
            mcs_rounding_status=radio.rounding_status,
            camera_si=context.camera_si,
            radar_p40=context.radar_p40,
            executed_mode_id=mode_id,
            requested_q=float(execution.q.item()),
            executed_q_e4=q_e4,
            q_perc=float(outcome.q_perc),
            total_transmitted_bytes=result.audit.total_transmitted_bytes,
            datagram_count=result.audit.datagram_count,
            p_complete_reassembly_given_sent=float(
                outcome.p_complete_reassembly_given_sent
            ),
            p_edge_admission_given_reassembled=float(
                outcome.p_edge_admission_given_reassembled
            ),
            p_edge_admission_given_sent=float(outcome.p_edge_admission_given_sent),
            conditional_feature_uplink_p50_ms=float(
                outcome.conditional_feature_uplink_p50_ms
            ),
            conditional_feature_uplink_p95_ms=float(
                outcome.conditional_feature_uplink_p95_ms
            ),
            conditional_feature_uplink_p99_ms=float(
                outcome.conditional_feature_uplink_p99_ms
            ),
            fixed_stage_latency_ms=outcome.fixed_stage_latency_ms,
            latency_proxy_ms=float(outcome.latency_proxy_ms),
            latency_proxy_p95_ms=float(outcome.latency_proxy_p95_ms),
            latency_proxy_p99_ms=float(outcome.latency_proxy_p99_ms),
            modeled_budget_miss=bool(outcome.modeled_budget_miss),
            modeled_budget_miss_p95=bool(outcome.modeled_budget_miss_p95),
            modeled_budget_miss_p99=bool(outcome.modeled_budget_miss_p99),
            reward=float(outcome.reward),
            result_status=outcome.status,
        )
        self._assert_no_runtime_side_effects()
        return row

    def evaluate_actor(
        self, *, actor: torch.nn.Module, actor_identity: LoadedActorIdentityV1
    ) -> Tuple[FitValidationEvaluationRowV1, ...]:
        rows = tuple(
            self.evaluate_entry(actor=actor, actor_identity=actor_identity, entry=entry)
            for entry in self.panel.entries
        )
        if len(rows) != 340 or tuple(row.panel_index for row in rows) != tuple(range(340)):
            raise FitValidationEvaluationError("actor evaluation did not cover the panel")
        return rows


def _actor_config() -> HybridSacModelConfig:
    return HybridSacModelConfig(
        dtype=torch.float32, modeled_smoke_support=MODELED_SMOKE_SUPPORT
    )


def _initial_actor(seed: int) -> Tuple[torch.nn.Module, LoadedActorIdentityV1]:
    init_rng = random.Random(_derive_seed(seed, "init"))
    actor_seed = init_rng.randrange(0, 1 << 63)
    # Consume the critic seed as the baseline runner does; retaining it in the
    # identity proves the reconstruction follows the registered seed schedule.
    critic_seed = init_rng.randrange(0, 1 << 63)
    actor = build_actor(_actor_config(), seed=actor_seed)
    state_hash = _hash_state(actor.state_dict())
    reconstruction = {
        "actor_seed": actor_seed,
        "actor_state_sha256": state_hash,
        "baseline_config_sha256": REGISTERED_BASELINE_CONFIG.canonical_sha256(),
        "critic_seed_consumed": critic_seed,
        "master_seed": seed,
        "record": "splitfusion.preliminary_baseline_initial_actor.v1",
        "seed_derivation_stream": "init",
    }
    return actor, LoadedActorIdentityV1(
        seed=seed,
        update_index=0,
        actor_state_sha256=state_hash,
        checkpoint_canonical_sha256=canonical_sha256(reconstruction),
        checkpoint_file_sha256=None,
        source="DETERMINISTIC_REGISTERED_UPDATE_ZERO_RECONSTRUCTION",
    )


def load_actor_for_evaluation(
    *, campaign_directory: Path, seed: int, update_index: int
) -> Tuple[torch.nn.Module, LoadedActorIdentityV1]:
    campaign_directory = Path(campaign_directory).resolve(strict=True)
    if seed not in REGISTERED_BASELINE_CONFIG.seeds:
        raise FitValidationEvaluationError("seed is outside the registered baseline")
    if update_index not in EVALUATION_CHECKPOINT_UPDATES:
        raise FitValidationEvaluationError("update is outside the registered evaluation cadence")
    seed_directory = campaign_directory / f"seed_{seed}"
    config = _read_json(seed_directory / "config.json")
    bindings = _read_json(seed_directory / "bindings.json")
    report = _read_json(seed_directory / "report.json")
    if (
        config.get("seed") != seed
        or config.get("config_sha256")
        != REGISTERED_BASELINE_CONFIG.canonical_sha256()
        or config.get("provenance_label") != BASELINE_PROVENANCE_LABEL
        or bindings.get("sampling_split") != "train"
        or bindings.get("fit_partition_sha256")
        != REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256
        or bindings.get("provenance_label") != BASELINE_PROVENANCE_LABEL
        or report.get("seed") != seed
        or report.get("status") != "COMPLETE"
        or report.get("completed_updates") != REGISTERED_BASELINE_CONFIG.update_count
        or report.get("phase_label") != BASELINE_PHASE_LABEL
    ):
        raise FitValidationEvaluationError("baseline seed artifact identity drift")
    report_content = dict(report)
    stated_report_hash = report_content.pop("report_content_sha256", None)
    if stated_report_hash != canonical_sha256(report_content):
        raise FitValidationEvaluationError("baseline seed report hash drift")
    if update_index == 0:
        return _initial_actor(seed)
    path = seed_directory / "checkpoints" / f"checkpoint_{update_index:06d}.pt"
    checkpoint = _load_checkpoint(path)
    if (
        checkpoint.seed != seed
        or checkpoint.update_count != update_index
        or checkpoint.config != REGISTERED_BASELINE_CONFIG
        or checkpoint.runner_binding_sha256
        != bindings.get("baseline_binding_sha256")
    ):
        raise FitValidationEvaluationError("checkpoint identity drift")
    actor = build_actor(_actor_config(), seed=0)
    actor.load_state_dict(checkpoint.actor_state, strict=True)
    state_hash = _hash_state(actor.state_dict())
    if state_hash != _hash_state(checkpoint.actor_state):
        raise FitValidationEvaluationError("loaded actor state differs from checkpoint")
    return actor, LoadedActorIdentityV1(
        seed=seed,
        update_index=update_index,
        actor_state_sha256=state_hash,
        checkpoint_canonical_sha256=checkpoint.checkpoint_sha256,
        checkpoint_file_sha256=_sha256_file(path),
        source="HASH_VALIDATED_NUMBERED_BASELINE_CHECKPOINT",
    )


def _csv_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    materialized = list(rows)
    if not materialized:
        raise FitValidationEvaluationError("CSV requires at least one row")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(materialized[0]))
    writer.writeheader()
    writer.writerows(materialized)
    return stream.getvalue().encode("utf-8")


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise FitValidationEvaluationError("aggregate population is empty")
    return math.fsum(values) / len(values)


def aggregate_evaluation_rows(
    rows: Sequence[FitValidationEvaluationRowV1],
) -> Tuple[Dict[str, Any], ...]:
    groups: Dict[Tuple[int, int, str], list[FitValidationEvaluationRowV1]] = {}
    identities = sorted({(row.seed, row.update_index) for row in rows})
    profile_order = tuple(load_registered_fit_validation_panel().profile_order)
    for seed, update in identities:
        actor_rows = [
            row for row in rows if row.seed == seed and row.update_index == update
        ]
        for profile in (*profile_order, "ALL_PROFILES"):
            selected = (
                actor_rows
                if profile == "ALL_PROFILES"
                else [row for row in actor_rows if row.network_profile == profile]
            )
            groups[(seed, update, profile)] = selected
    result = []
    for (seed, update, profile), selected in groups.items():
        expected = 340 if profile == "ALL_PROFILES" else 85
        if len(selected) != expected:
            raise FitValidationEvaluationError("aggregate panel coverage drift")
        result.append(
            {
                "seed": seed,
                "update_index": update,
                "network_profile": profile,
                "panel_row_count": len(selected),
                "reward_mean": _mean([row.reward for row in selected]),
                "reward_min": min(row.reward for row in selected),
                "reward_max": max(row.reward for row in selected),
                "q_perc_mean": _mean([row.q_perc for row in selected]),
                "payload_bytes_mean": _mean(
                    [row.total_transmitted_bytes for row in selected]
                ),
                "p_edge_admission_given_sent_mean": _mean(
                    [row.p_edge_admission_given_sent for row in selected]
                ),
                "latency_proxy_ms_mean": _mean(
                    [row.latency_proxy_ms for row in selected]
                ),
                "budget_miss_rate_p50": _mean(
                    [float(row.modeled_budget_miss) for row in selected]
                ),
                "budget_miss_rate_p95": _mean(
                    [float(row.modeled_budget_miss_p95) for row in selected]
                ),
                "budget_miss_rate_p99": _mean(
                    [float(row.modeled_budget_miss_p99) for row in selected]
                ),
                "unique_executed_action_count": len(
                    {(row.executed_mode_id, row.executed_q_e4) for row in selected}
                ),
                "actor_state_sha256": selected[0].actor_state_sha256,
                "checkpoint_canonical_sha256": (
                    selected[0].checkpoint_canonical_sha256
                ),
            }
        )
    return tuple(result)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def evaluate_registered_campaign(
    *, campaign_directory: Path, output_directory: Path, project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Evaluate all 3 x 11 actors and atomically write a new artifact set."""
    campaign_directory = Path(campaign_directory).resolve(strict=True)
    output_directory = Path(output_directory).resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FitValidationEvaluationError("output directory must be new or empty")
    campaign_report_path = campaign_directory / "campaign_report.json"
    campaign_report = _read_json(campaign_report_path)
    campaign_content = dict(campaign_report)
    stated_campaign_hash = campaign_content.pop(
        "campaign_report_content_sha256", None
    )
    if (
        stated_campaign_hash != canonical_sha256(campaign_content)
        or campaign_report.get("config_sha256")
        != REGISTERED_BASELINE_CONFIG.canonical_sha256()
        or campaign_report.get("phase_label") != BASELINE_PHASE_LABEL
        or campaign_report.get("provenance_label") != BASELINE_PROVENANCE_LABEL
    ):
        raise FitValidationEvaluationError("campaign report identity drift")

    all_rows: list[FitValidationEvaluationRowV1] = []
    actor_identities: list[LoadedActorIdentityV1] = []
    with FitValidationActorEvaluatorV1(project_root=project_root) as evaluator:
        for seed in REGISTERED_BASELINE_CONFIG.seeds:
            for update in EVALUATION_CHECKPOINT_UPDATES:
                actor, identity = load_actor_for_evaluation(
                    campaign_directory=campaign_directory,
                    seed=seed,
                    update_index=update,
                )
                actor_identities.append(identity)
                all_rows.extend(
                    evaluator.evaluate_actor(actor=actor, actor_identity=identity)
                )
        evaluator._assert_no_runtime_side_effects()

    expected_count = (
        len(REGISTERED_BASELINE_CONFIG.seeds)
        * len(EVALUATION_CHECKPOINT_UPDATES)
        * 340
    )
    if len(all_rows) != expected_count:
        raise FitValidationEvaluationError("campaign evaluation row-count drift")
    per_context = _csv_bytes(row.to_canonical_dict() for row in all_rows)
    aggregate_rows = aggregate_evaluation_rows(all_rows)
    aggregate = _csv_bytes(aggregate_rows)
    per_context_name = "fit_validation_per_context.csv"
    aggregate_name = "fit_validation_aggregate.csv"
    manifest_name = "manifest.json"
    output_directory.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(output_directory / per_context_name, per_context)
    _atomic_bytes(output_directory / aggregate_name, aggregate)
    manifest = {
        "actor_execution": "ARGMAX_MODE_PLUS_SELECTED_CONDITIONAL_MEAN_Q",
        "actor_identities": [item.to_canonical_dict() for item in actor_identities],
        "aggregate_row_count": len(aggregate_rows),
        "baseline_campaign_report_content_sha256": stated_campaign_hash,
        "baseline_campaign_report_file_sha256": _sha256_file(campaign_report_path),
        "baseline_config_sha256": REGISTERED_BASELINE_CONFIG.canonical_sha256(),
        "checkpoint_updates": list(EVALUATION_CHECKPOINT_UPDATES),
        "fit_partition_sha256": REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
        "fit_validation_panel_sha256": REGISTERED_FIT_VALIDATION_PANEL_SHA256,
        "mcs_rounding_rule": dict(_MCS_ROUNDING_RULE),
        "mcs_rounding_rule_sha256": MCS_ROUNDING_RULE_SHA256,
        "output_files": {
            aggregate_name: hashlib.sha256(aggregate).hexdigest(),
            per_context_name: hashlib.sha256(per_context).hexdigest(),
        },
        "per_context_row_count": len(all_rows),
        "pilot_utility_spec_sha256": PILOT_UTILITY_SPEC_SHA256,
        "provenance_label": EVALUATION_PROVENANCE_LABEL,
        "schema": EVALUATION_SCHEMA,
        "scope_disclosure": EVALUATION_SCOPE_DISCLOSURE,
        "seeds": list(REGISTERED_BASELINE_CONFIG.seeds),
        "status": "COMPLETE",
    }
    manifest["manifest_content_sha256"] = canonical_sha256(manifest)
    _atomic_bytes(
        output_directory / manifest_name,
        (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the preliminary actors on the frozen fit-validation panel."
    )
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = evaluate_registered_campaign(
        campaign_directory=args.campaign, output_directory=args.output
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
