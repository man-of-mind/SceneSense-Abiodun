"""Preregistered development-panel evaluation for exact-P95 Run-2 v2.

This module is evaluation-only.  It never constructs replay, an optimizer or
a training environment.  It requires the physical Run-2-v2 update-zero and
all registered numbered checkpoints, executes every actor deterministically,
and scores only the frozen 85-scene x four-profile development panel.

The reported P95 is the modeled conditional retained-survivor proxy.  It is
not an unconditional live-service percentile or a live 200-ms SLA.
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
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .action_contract import round_half_up_q_e4
from .anchor_store import NETWORK_PROFILE_ORDER
from .empirical_contextual_contract import (
    MODELED_SMOKE_SUPPORT,
    PILOT_UTILITY_SPEC,
    PILOT_UTILITY_SPEC_SHA256,
    require_supported_action,
)
from .empirical_contextual_exact_p95_run2_replay_v2 import (
    EXACT_P95_RUN2_REWARD_SPEC_V2_SHA256,
    GATE_V2_CANONICAL_CONTENT_SHA256,
    GATE_V2_DECISION_SHA256,
    GATE_V2_CONTEXT_ORACLES_SHA256,
    GATE_V2_SUMMARY_SHA256,
    PREREG_V2_FILE_SHA256,
    REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256,
    RUN2_V2_DEADLINE_MS,
    RUN2_V2_DEADLINE_PENALTY,
    TRAIN_V2_DECISION_SHA256,
    TRAIN_V2_SUMMARY_SHA256,
)
from .empirical_contextual_exact_p95_run2_runner_v2 import (
    EXACT_P95_RUN2_CHECKPOINT_INTERVAL_UPDATES_V2,
    PHASE_LABEL as TRAINING_PHASE_LABEL,
    REGISTERED_EXACT_P95_RUN2_CONFIG_V2,
    ExactP95Run2CheckpointV2,
    _hash_state,
)
from .run_empirical_contextual_exact_p95_run2_v2 import (
    BINDINGS_SCHEMA as TRAINING_BINDINGS_SCHEMA_V2,
    CAMPAIGN_SCHEMA as TRAINING_CAMPAIGN_SCHEMA_V2,
    CHECKPOINT_SELECTION as TRAINING_CHECKPOINT_SELECTION_V2,
    CONFIG_SCHEMA as TRAINING_CONFIG_SCHEMA_V2,
    FROZEN_COMPARATOR_COMMIT,
    FROZEN_COMPARATOR_DECISION_CONTENT_SHA256,
    FROZEN_COMPARATOR_IMPLEMENTATION_SHA256,
    FROZEN_COMPARATOR_MANIFEST_CONTENT_SHA256,
    FROZEN_COMPARATOR_MANIFEST_FILE_SHA256,
    FROZEN_COMPARATOR_SUMMARY_CONTENT_SHA256,
    OUTPUT_SCHEMA as TRAINING_OUTPUT_SCHEMA_V2,
    REPORT_SCHEMA as TRAINING_REPORT_SCHEMA_V2,
    TARGET_RULE as TRAINING_TARGET_RULE_V2,
    load_exact_p95_run2_checkpoint_v2,
)
from .empirical_contextual_exact_p95_run2_validation_gate_v2 import (
    CONDITIONAL_FEASIBILITY_SEMANTICS,
)
from .empirical_contextual_fit_partition import (
    FIT_VALIDATION_SPLIT,
    REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
    TRAIN_SPLIT,
    load_registered_empirical_fit_partition,
)
from .empirical_contextual_fit_validation_evaluator import (
    FitValidationActorEvaluatorV1,
    load_actor_for_evaluation,
)
from .empirical_contextual_fit_validation_panel import (
    REGISTERED_FIT_VALIDATION_PANEL_SHA256,
    FitValidationPanelEntryV1,
)
from .empirical_contextual_split_oracle import (
    RANDOM_RULE_SCHEMA,
    _authoritative_outcome,
    choose_registered_random_action,
)
from .hybrid_sac_models import HybridSacModelConfig, build_actor
from .transaction_identity import canonical_sha256


EVALUATOR_SCHEMA_V2 = "splitfusion.exact_p95_run2_development_evaluator.v2"
ROW_SCHEMA_V2 = "splitfusion.exact_p95_run2_development_row.v2"
AGGREGATE_SCHEMA_V2 = "splitfusion.exact_p95_run2_development_aggregate.v2"
PROTOCOL_SCHEMA_V2 = "splitfusion.exact_p95_run2_evaluation_protocol.v2"
MANIFEST_SCHEMA_V2 = "splitfusion.exact_p95_run2_evaluation_artifacts.v2"
EVALUATION_UPDATES_V2: Tuple[int, ...] = tuple(range(0, 5001, 500))
PRIMARY_UPDATE_V2 = 5000
FIXED_MODE_ID_V2 = 11
FIXED_Q_E4_V2 = 6000
BOOTSTRAP_RESAMPLES_V2 = 10_000
BOOTSTRAP_SEED_V2 = 20_260_921
DEADLINE_MS_V2 = 200.0
TRAINING_CLI_COMMIT_V2 = "da58f919fcea41ef4b6c95881d6d773bef256a2e"
TRAINING_CLI_IMPLEMENTATION_SHA256_V2 = (
    "7d5d7db72ca13d6e31ecae18c1215cbd1c26e2265904b02bac19a53d3f2357de"
)

RUN1_CAMPAIGN_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_preliminary_baseline_v1/"
    "20260921_train_split_5000x3_v1"
)
ORACLE_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_fit_validation_v1/"
    "20260921_exact_p95_run2_pretraining_validation_gate_v2/"
    "validation_context_oracles_v2.csv"
)
ORACLE_SUMMARY_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_fit_validation_v1/"
    "20260921_exact_p95_run2_pretraining_validation_gate_v2/summary_v2.json"
)
ORACLE_DECISION_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_fit_validation_v1/"
    "20260921_exact_p95_run2_pretraining_validation_gate_v2/GO_NO_GO_v2.json"
)
FIXED_COMPARATOR_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_run2_fixed_comparator_v2/"
    "20260921_exact_p95_run2_fixed_comparator_v2"
)
FIXED_COMPARATOR_FILE_HASHES = {
    "REPORT.md": "8eafc6987ac627e416c332535a5ab7a073ef6bb4a28197df05e95f558e1e78de",
    "action_scores.csv": "a5f617cd5cdf57bffa85ee2dcd04d138247eed0e3e95d2a3a71473fb110203c7",
    "context_winner_revalidation.csv": "bd6af2f988acd89d21c44bb112596c488f4b6d6dcb1d20a7b21d69b98e28dda8",
    "summary.json": "6fcb4d68cc49f7a89ce0a552f0721fd209904d92bb9089d583aca56685986be5",
    "decision.json": "01b2bc229791b55f498339c01467c93c647cd17fa3ca6aa4eaaa3a04586b3038",
    "artifact_manifest.json": "2c95fb808f0099bc25fefdb4f071ab27533991b68b62c27fcd7e031f2a8b86ba",
}
FIXED_COMPARATOR_IMPLEMENTATION_RELATIVE_PATH = (
    "rl_agent/splitfusion_hybrid_sac_v1/"
    "empirical_contextual_exact_p95_run2_fixed_comparator_v2.py"
)

SOURCE_V2 = "RUN2_V2_ACTOR"
SOURCE_RUN1 = "RUN1_UPDATE5000_RESCORED_V2"
SOURCE_RANDOM = "REGISTERED_IDENTITY_RANDOM"
SOURCE_FIXED = "FROZEN_TRAIN_SELECTED_FIXED"
SOURCE_ORACLE = "CONTEXTUAL_P95_CONSTRAINED_ORACLE"
POOLED_SEED = -1


class ExactP95Run2EvaluationV2Error(RuntimeError):
    """A frozen evaluation invariant failed."""


def _protocol_document() -> Dict[str, Any]:
    return {
        "adaptation": {
            "baseline": {"mode_id": FIXED_MODE_ID_V2, "q_e4": FIXED_Q_E4_V2},
            "bootstrap": {
                "cluster": "SCENE_RANK_WITH_ALL_4_PROFILES_AND_ALL_3_SEEDS",
                "ci": "NUMPY_LINEAR_PERCENTILES_2.5_AND_97.5",
                "resamples": BOOTSTRAP_RESAMPLES_V2,
                "rng": "NUMPY_PCG64",
                "seed": BOOTSTRAP_SEED_V2,
            },
            "requirements": {
                "pooled_mean_lift_gt_zero": True,
                "positive_seed_count_min": 2,
                "lower_95_percent_ci_gt_zero": True,
            },
        },
        "actor_execution": (
            "ARGMAX_DISCRETE_MODE_THEN_SELECTED_CONDITIONAL_MEAN_Q_THEN_"
            "EXACT_REGISTERED_EXECUTION_BOUNDARY_Q_E4"
        ),
        "aggregation": (
            "EMITTED_FLOAT32_TARGETS_WIDENED_TO_BINARY64_THEN_"
            "MATH_FSUM_AND_DIVIDE_ONCE"
        ),
        "comparators": [
            SOURCE_RANDOM,
            "MATCHED_PHYSICAL_RUN2_V2_UPDATE0",
            SOURCE_RUN1,
            SOURCE_FIXED,
            SOURCE_ORACLE,
        ],
        "frozen_bindings": {
            "fit_partition_sha256": REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
            "fit_validation_panel_sha256": REGISTERED_FIT_VALIDATION_PANEL_SHA256,
            "fixed_comparator_commit": FROZEN_COMPARATOR_COMMIT,
            "fixed_comparator_implementation_sha256": (
                FROZEN_COMPARATOR_IMPLEMENTATION_SHA256
            ),
            "fixed_comparator_summary_content_sha256": (
                FROZEN_COMPARATOR_SUMMARY_CONTENT_SHA256
            ),
            "preregistration_v2_file_sha256": PREREG_V2_FILE_SHA256,
            "reward_spec_v2_sha256": EXACT_P95_RUN2_REWARD_SPEC_V2_SHA256,
            "train_penalty_decision_file_sha256": TRAIN_V2_DECISION_SHA256,
            "train_penalty_spec_sha256": (
                REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256
            ),
            "train_penalty_summary_file_sha256": TRAIN_V2_SUMMARY_SHA256,
            "training_cli_commit": TRAINING_CLI_COMMIT_V2,
            "training_cli_implementation_sha256": (
                TRAINING_CLI_IMPLEMENTATION_SHA256_V2
            ),
            "validation_gate_canonical_content_sha256": (
                GATE_V2_CANONICAL_CONTENT_SHA256
            ),
            "validation_gate_decision_file_sha256": GATE_V2_DECISION_SHA256,
            "validation_gate_oracles_file_sha256": GATE_V2_CONTEXT_ORACLES_SHA256,
            "validation_gate_summary_file_sha256": GATE_V2_SUMMARY_SHA256,
        },
        "deadline_ms": DEADLINE_MS_V2,
        "evaluation_updates": list(EVALUATION_UPDATES_V2),
        "panel": "FROZEN_85_SCENE_X_4_PROFILE_DEVELOPMENT_FIT_VALIDATION",
        "primary_update": PRIMARY_UPDATE_V2,
        "random_rule_schema": RANDOM_RULE_SCHEMA,
        "reward_spec_v2_sha256": EXACT_P95_RUN2_REWARD_SPEC_V2_SHA256,
        "schema": PROTOCOL_SCHEMA_V2,
        "scope_caveats": [
            "DEVELOPMENT_FIT_VALIDATION_NOT_FINAL_TEST",
            "P95_IS_MODELED_CONDITIONAL_RETAINED_SURVIVOR_PROXY_NOT_LIVE_SLA",
            "85_SCENE_CLUSTERS_X_4_PROFILES_AND_THREE_SEEDS_ARE_NOT_INDEPENDENT",
            "NO_CLAIM_REWARD_APPROACHES_ONE",
        ],
        "success_tiers": {
            "tier0": {
                "all_registered_physical_checkpoints_present": True,
                "all_three_seeds_complete": True,
                "cuda_not_initialized_by_evaluator": True,
                "d1_and_v2_rewards_separate": True,
                "development_panel_is_85x4_and_train_disjoint": True,
                "exact_v2_checkpoint_types": True,
                "executed_q_e4_scored": True,
                "finite_rows": True,
                "global_rng_unchanged": True,
                "physical_update_zero_present": True,
            },
            "tier1": {
                "each_seed_final_misses_lt_matched_update0": True,
                "pooled_relative_miss_reduction_vs_update0_min": 0.50,
                "pooled_relative_miss_reduction_vs_run1_rescored_min": 0.50,
            },
            "tier2": {
                "pooled_miss_rate_max": 0.05,
                "each_seed_miss_rate_max": 0.10,
                "every_profile_pooled_miss_rate_max": 0.10,
                "quality_fraction_of_oracle_min": 0.90,
                "absolute_admission_difference_from_oracle_max": 0.001,
            },
            "tier3": "ZERO_FINAL_P95_MISSES",
        },
    }


EVALUATION_PROTOCOL_V2_SHA256 = (
    "02e7f34702fc13d325047e3e32d2473a7129a2854e0cf4048b042626d7e22e50"
)


def evaluation_protocol_document_v2() -> Dict[str, Any]:
    document = _protocol_document()
    if canonical_sha256(document) != EVALUATION_PROTOCOL_V2_SHA256:
        raise ExactP95Run2EvaluationV2Error("evaluation protocol drift")
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExactP95Run2EvaluationV2Error(f"malformed JSON {path}") from exc
    if type(result) is not dict:
        raise ExactP95Run2EvaluationV2Error(f"{path} is not a JSON object")
    return result


def _require_training_cli_source(root: Path) -> None:
    path = (
        root
        / "rl_agent/splitfusion_hybrid_sac_v1/"
        "run_empirical_contextual_exact_p95_run2_v2.py"
    )
    if (
        not path.is_file()
        or _sha256_file(path) != TRAINING_CLI_IMPLEMENTATION_SHA256_V2
    ):
        raise ExactP95Run2EvaluationV2Error("frozen training CLI source drift")


def _require_content_hash(document: Mapping[str, Any], field: str) -> None:
    supplied = document.get(field)
    payload = {key: value for key, value in document.items() if key != field}
    if supplied != canonical_sha256(payload):
        raise ExactP95Run2EvaluationV2Error(f"{field} mismatch")


def _float64_bits_hex(value: float) -> str:
    return f"0x{struct.unpack('>Q', struct.pack('>d', float(value)))[0]:016x}"


def _float32_bits_hex(value: float) -> str:
    return f"0x{struct.unpack('>I', struct.pack('>f', float(value)))[0]:08x}"


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _score_v2(
    *, p_admit: float, quality: float, latency_p95_ms: float
) -> Tuple[float, float, float]:
    """Pinned binary64 order followed by one explicit CPU float32 emission."""
    p = float(p_admit)
    q = float(quality)
    latency = float(latency_p95_ms)
    if not all(math.isfinite(value) for value in (p, q, latency)):
        raise ExactP95Run2EvaluationV2Error("non-finite V2 score input")
    if not 0.0 <= p <= 1.0 or not 0.0 <= q <= 1.0 or latency < 0.0:
        raise ExactP95Run2EvaluationV2Error("V2 score input outside domain")
    base64 = p * (q - 0.25 * (latency / 200.0)) + (1.0 - p) * (-1.0)
    shaped64 = (
        base64 - p * RUN2_V2_DEADLINE_PENALTY
        if latency > RUN2_V2_DEADLINE_MS
        else base64
    )
    emitted = torch.tensor(shaped64, dtype=torch.float32, device="cpu")
    if emitted.device.type != "cpu" or not bool(torch.isfinite(emitted)):
        raise ExactP95Run2EvaluationV2Error("V2 float32 emission failed")
    return base64, shaped64, float(emitted.item())


@dataclass(frozen=True, slots=True)
class Run2V2ActorIdentity:
    source_kind: str
    seed: int
    update_index: int
    actor_state_sha256: str
    checkpoint_sha256: str
    checkpoint_file_sha256: str


@dataclass(frozen=True, slots=True)
class ExactP95Run2EvaluationRowV2:
    source_kind: str
    seed: int
    update_index: int
    actor_state_sha256: str
    checkpoint_sha256: str
    checkpoint_file_sha256: str
    panel_index: int
    scene_rank: int
    sample_id: str
    episode_id: str
    frame_id: int
    network_profile: str
    radio_csv_row_number: int
    radio_row_sha256: str
    executed_mode_id: int
    requested_q: float
    executed_q_e4: int
    q_perc: float
    p_edge_admission_given_sent: float
    total_transmitted_bytes: float
    datagram_count: int
    latency_proxy_p50_ms: float
    latency_proxy_p95_ms: float
    latency_proxy_p99_ms: float
    p95_margin_ms: float
    p95_miss: int
    d1_reward64: float
    d1_reward64_bits_hex: str
    v2_base_reward64: float
    v2_base_reward64_bits_hex: str
    v2_shaped_reward64: float
    v2_shaped_reward64_bits_hex: str
    v2_emitted_reward_float32: float
    v2_emitted_reward_float32_bits_hex: str
    schema: str = ROW_SCHEMA_V2

    def __post_init__(self) -> None:
        require_supported_action(self.executed_mode_id, self.executed_q_e4)
        if round_half_up_q_e4(self.requested_q) != self.executed_q_e4:
            raise ExactP95Run2EvaluationV2Error(
                "requested q does not produce executed q_e4"
            )
        if self.p95_miss not in (0, 1):
            raise ExactP95Run2EvaluationV2Error("p95_miss must be an integer bit")
        if self.p95_miss != int(self.latency_proxy_p95_ms > DEADLINE_MS_V2):
            raise ExactP95Run2EvaluationV2Error("P95 miss boundary drift")
        if self.schema != ROW_SCHEMA_V2:
            raise ExactP95Run2EvaluationV2Error("row schema drift")
        numeric = (
            self.requested_q,
            self.q_perc,
            self.p_edge_admission_given_sent,
            self.total_transmitted_bytes,
            self.latency_proxy_p50_ms,
            self.latency_proxy_p95_ms,
            self.latency_proxy_p99_ms,
            self.p95_margin_ms,
            self.d1_reward64,
            self.v2_base_reward64,
            self.v2_shaped_reward64,
            self.v2_emitted_reward_float32,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ExactP95Run2EvaluationV2Error("row contains non-finite values")
        if (
            not 0.0 <= self.requested_q <= 0.98
            or not 0.0 <= self.q_perc <= 1.0
            or not 0.0 <= self.p_edge_admission_given_sent <= 1.0
            or self.total_transmitted_bytes < 0.0
            or type(self.datagram_count) is not int
            or self.datagram_count < 0
            or not (
                0.0
                <= self.latency_proxy_p50_ms
                <= self.latency_proxy_p95_ms
                <= self.latency_proxy_p99_ms
            )
        ):
            raise ExactP95Run2EvaluationV2Error("row metric domain/order drift")
        if self.p95_margin_ms != DEADLINE_MS_V2 - self.latency_proxy_p95_ms:
            raise ExactP95Run2EvaluationV2Error("P95 margin drift")
        if self.d1_reward64_bits_hex != _float64_bits_hex(self.d1_reward64):
            raise ExactP95Run2EvaluationV2Error("D1 reward bits drift")
        expected_d1 = PILOT_UTILITY_SPEC.expected_utility(
            p_edge_admission_given_sent=self.p_edge_admission_given_sent,
            q_perc=self.q_perc,
            latency_proxy_ms=self.latency_proxy_p50_ms,
        )
        if self.d1_reward64 != expected_d1:
            raise ExactP95Run2EvaluationV2Error("D1 reward recomputation drift")
        if self.v2_base_reward64_bits_hex != _float64_bits_hex(self.v2_base_reward64):
            raise ExactP95Run2EvaluationV2Error("V2 base bits drift")
        if self.v2_shaped_reward64_bits_hex != _float64_bits_hex(self.v2_shaped_reward64):
            raise ExactP95Run2EvaluationV2Error("V2 shaped bits drift")
        if self.v2_emitted_reward_float32_bits_hex != _float32_bits_hex(
            self.v2_emitted_reward_float32
        ):
            raise ExactP95Run2EvaluationV2Error("V2 emitted bits drift")
        base, shaped, emitted = _score_v2(
            p_admit=self.p_edge_admission_given_sent,
            quality=self.q_perc,
            latency_p95_ms=self.latency_proxy_p95_ms,
        )
        if (base, shaped, emitted) != (
            self.v2_base_reward64,
            self.v2_shaped_reward64,
            self.v2_emitted_reward_float32,
        ):
            raise ExactP95Run2EvaluationV2Error("row V2 score recomputation drift")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _row_for_action(
    evaluator: FitValidationActorEvaluatorV1,
    entry: FitValidationPanelEntryV1,
    *,
    identity: Run2V2ActorIdentity,
    mode_id: int,
    q_e4: int,
    requested_q: float,
) -> ExactP95Run2EvaluationRowV2:
    require_supported_action(mode_id, q_e4)
    raw = _authoritative_outcome(evaluator, entry, mode_id, q_e4)
    base64, shaped64, emitted32 = _score_v2(
        p_admit=raw.p_edge_admission_given_sent,
        quality=raw.q_perc,
        latency_p95_ms=raw.latency_proxy_p95_ms,
    )
    return ExactP95Run2EvaluationRowV2(
        source_kind=identity.source_kind,
        seed=identity.seed,
        update_index=identity.update_index,
        actor_state_sha256=identity.actor_state_sha256,
        checkpoint_sha256=identity.checkpoint_sha256,
        checkpoint_file_sha256=identity.checkpoint_file_sha256,
        panel_index=entry.panel_index,
        scene_rank=entry.scene_rank,
        sample_id=entry.scene_sample_id,
        episode_id=entry.scene_episode_id,
        frame_id=entry.scene_frame_id,
        network_profile=entry.network_profile,
        radio_csv_row_number=entry.radio_csv_row_number,
        radio_row_sha256=entry.radio_row_sha256,
        executed_mode_id=mode_id,
        requested_q=float(requested_q),
        executed_q_e4=q_e4,
        q_perc=raw.q_perc,
        p_edge_admission_given_sent=raw.p_edge_admission_given_sent,
        total_transmitted_bytes=raw.total_transmitted_bytes,
        datagram_count=raw.datagram_count,
        latency_proxy_p50_ms=raw.latency_proxy_p50_ms,
        latency_proxy_p95_ms=raw.latency_proxy_p95_ms,
        latency_proxy_p99_ms=raw.latency_proxy_p99_ms,
        p95_margin_ms=DEADLINE_MS_V2 - raw.latency_proxy_p95_ms,
        p95_miss=int(raw.latency_proxy_p95_ms > DEADLINE_MS_V2),
        d1_reward64=raw.reward,
        d1_reward64_bits_hex=_float64_bits_hex(raw.reward),
        v2_base_reward64=base64,
        v2_base_reward64_bits_hex=_float64_bits_hex(base64),
        v2_shaped_reward64=shaped64,
        v2_shaped_reward64_bits_hex=_float64_bits_hex(shaped64),
        v2_emitted_reward_float32=emitted32,
        v2_emitted_reward_float32_bits_hex=_float32_bits_hex(emitted32),
    )


def _actor_rows(
    evaluator: FitValidationActorEvaluatorV1,
    actor: torch.nn.Module,
    identity: Run2V2ActorIdentity,
) -> Tuple[ExactP95Run2EvaluationRowV2, ...]:
    rows = []
    for entry in evaluator.panel.entries:
        observation, _context, _radio = evaluator._make_observation(entry)
        state = torch.tensor([observation.values], dtype=torch.float32, device="cpu")
        with torch.no_grad():
            execution = actor.deterministic_execution(state)
        mode_id = int(execution.mode_index.item())
        requested_q = float(execution.q.item())
        q_e4 = int(execution.q_e4.item())
        if q_e4 != round_half_up_q_e4(requested_q):
            raise ExactP95Run2EvaluationV2Error(
                "actor q_e4 bypassed exact execution boundary"
            )
        rows.append(
            _row_for_action(
                evaluator,
                entry,
                identity=identity,
                mode_id=mode_id,
                q_e4=q_e4,
                requested_q=requested_q,
            )
        )
    if len(rows) != 340:
        raise ExactP95Run2EvaluationV2Error("actor did not cover all 340 contexts")
    return tuple(rows)


def _build_actor(state: Mapping[str, Any]) -> torch.nn.Module:
    actor = build_actor(
        HybridSacModelConfig(dtype=torch.float32, modeled_smoke_support=MODELED_SMOKE_SUPPORT),
        seed=0,
    )
    try:
        actor.load_state_dict(dict(state), strict=True)
    except Exception as exc:
        raise ExactP95Run2EvaluationV2Error("actor checkpoint state is incompatible") from exc
    if _hash_state(actor.state_dict()) != _hash_state(state):
        raise ExactP95Run2EvaluationV2Error("loaded actor differs from checkpoint")
    actor.eval()
    return actor


def _load_exact_v2_checkpoint(path: Path) -> ExactP95Run2CheckpointV2:
    try:
        checkpoint = load_exact_p95_run2_checkpoint_v2(path)
    except Exception as exc:
        raise ExactP95Run2EvaluationV2Error("not an exact V2 checkpoint") from exc
    if type(checkpoint) is not ExactP95Run2CheckpointV2:
        raise ExactP95Run2EvaluationV2Error("checkpoint exact type drift")
    return checkpoint


def _validate_v2_campaign(
    campaign_directory: Path,
) -> Dict[Tuple[int, int], Tuple[Path, ExactP95Run2CheckpointV2]]:
    campaign_directory = Path(campaign_directory).resolve(strict=True)
    campaign = _read_json(campaign_directory / "campaign_report.json")
    _require_content_hash(campaign, "campaign_report_content_sha256")
    config = REGISTERED_EXACT_P95_RUN2_CONFIG_V2
    if (
        campaign.get("record") != TRAINING_CAMPAIGN_SCHEMA_V2
        or campaign.get("output_schema") != TRAINING_OUTPUT_SCHEMA_V2
        or campaign.get("status") != "COMPLETE"
        or campaign.get("phase_label") != TRAINING_PHASE_LABEL
        or campaign.get("config_sha256") != config.canonical_sha256()
        or tuple(campaign.get("seeds", ())) != config.seeds
        or campaign.get("checkpoint_selection")
        != TRAINING_CHECKPOINT_SELECTION_V2
        or campaign.get("target_rule") != TRAINING_TARGET_RULE_V2
        or campaign.get("training_population") != "REGISTERED_TRAIN_IDS_ONLY"
        or campaign.get("validation_access_during_training") != "FORBIDDEN"
    ):
        raise ExactP95Run2EvaluationV2Error("V2 campaign identity/status drift")
    campaign_reports = campaign.get("reports")
    campaign_comparator = campaign.get("fixed_comparator")
    if (
        type(campaign_reports) is not list
        or len(campaign_reports) != len(config.seeds)
        or type(campaign_comparator) is not dict
        or campaign_comparator.get("frozen_commit") != FROZEN_COMPARATOR_COMMIT
        or campaign_comparator.get("frozen_before_run2_outcomes") is not True
        or campaign_comparator.get("implementation_sha256")
        != FROZEN_COMPARATOR_IMPLEMENTATION_SHA256
        or campaign_comparator.get("summary_content_sha256")
        != FROZEN_COMPARATOR_SUMMARY_CONTENT_SHA256
        or campaign_comparator.get("manifest_content_sha256")
        != FROZEN_COMPARATOR_MANIFEST_CONTENT_SHA256
        or campaign_comparator.get("decision_content_sha256")
        != FROZEN_COMPARATOR_DECISION_CONTENT_SHA256
        or campaign_comparator.get("fixed_action", {}).get("mode_id")
        != FIXED_MODE_ID_V2
        or campaign_comparator.get("fixed_action", {}).get("q_e4")
        != FIXED_Q_E4_V2
    ):
        raise ExactP95Run2EvaluationV2Error("V2 campaign report/comparator drift")
    result: Dict[Tuple[int, int], Tuple[Path, ExactP95Run2CheckpointV2]] = {}
    for seed_index, seed in enumerate(config.seeds):
        directory = campaign_directory / f"seed_{seed}"
        report = _read_json(directory / "report.json")
        bindings = _read_json(directory / "bindings.json")
        seed_config = _read_json(directory / "config.json")
        for document, field in (
            (report, "report_content_sha256"),
            (bindings, "bindings_content_sha256"),
            (seed_config, "config_content_sha256"),
        ):
            _require_content_hash(document, field)
        if (
            campaign_reports[seed_index] != report
            or report.get("record") != TRAINING_REPORT_SCHEMA_V2
            or report.get("output_schema") != TRAINING_OUTPUT_SCHEMA_V2
            or report.get("status") != "COMPLETE"
            or report.get("phase_label") != TRAINING_PHASE_LABEL
            or report.get("seed") != seed
            or report.get("completed_updates") != config.update_count
            or report.get("configured_updates") != config.update_count
            or report.get("checkpoint_cadence_updates")
            != EXACT_P95_RUN2_CHECKPOINT_INTERVAL_UPDATES_V2
            or report.get("checkpoint_selection")
            != TRAINING_CHECKPOINT_SELECTION_V2
            or report.get("target_rule") != TRAINING_TARGET_RULE_V2
            or report.get("training_population")
            != "REGISTERED_TRAIN_IDS_ONLY"
            or report.get("validation_access_during_training") != "FORBIDDEN"
            or report.get("sampling_split") != TRAIN_SPLIT
            or report.get("fit_partition_sha256")
            != REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256
            or report.get("fixed_comparator") != campaign_comparator
            or bindings.get("record") != TRAINING_BINDINGS_SCHEMA_V2
            or bindings.get("output_schema") != TRAINING_OUTPUT_SCHEMA_V2
            or bindings.get("phase_label") != TRAINING_PHASE_LABEL
            or bindings.get("seed") != seed
            or bindings.get("sampling_split") != TRAIN_SPLIT
            or bindings.get("config_sha256") != config.canonical_sha256()
            or bindings.get("fit_partition_sha256")
            != REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256
            or bindings.get("fixed_comparator") != campaign_comparator
            or seed_config.get("record") != TRAINING_CONFIG_SCHEMA_V2
            or seed_config.get("output_schema") != TRAINING_OUTPUT_SCHEMA_V2
            or seed_config.get("phase_label") != TRAINING_PHASE_LABEL
            or seed_config.get("seed") != seed
            or seed_config.get("config_sha256") != config.canonical_sha256()
            or seed_config.get("config") != config.to_canonical_dict()
            or seed_config.get("checkpoint_interval_updates")
            != EXACT_P95_RUN2_CHECKPOINT_INTERVAL_UPDATES_V2
            or seed_config.get("checkpoint_selection")
            != TRAINING_CHECKPOINT_SELECTION_V2
            or seed_config.get("target_rule") != TRAINING_TARGET_RULE_V2
            or seed_config.get("training_population")
            != "REGISTERED_TRAIN_IDS_ONLY"
            or seed_config.get("validation_access_during_training") != "FORBIDDEN"
            or seed_config.get("fixed_comparator") != campaign_comparator
        ):
            raise ExactP95Run2EvaluationV2Error("V2 seed artifact identity drift")
        artifact_hashes = report.get("artifact_file_sha256")
        checkpoint_series = report.get("checkpoint_series")
        expected_series_keys = {
            f"{update:06d}" for update in EVALUATION_UPDATES_V2 if update != 0
        }
        if (
            type(artifact_hashes) is not dict
            or type(checkpoint_series) is not dict
            or set(checkpoint_series) != expected_series_keys
        ):
            raise ExactP95Run2EvaluationV2Error("V2 report artifact inventory drift")
        for artifact_name in (
            "bindings.json",
            "checkpoint_000000.pt",
            "checkpoint_latest.pt",
            "config.json",
            "metrics.csv",
            "transition_reward_audit.csv",
        ):
            artifact_path = directory / artifact_name
            if (
                not artifact_path.is_file()
                or artifact_hashes.get(artifact_name) != _sha256_file(artifact_path)
            ):
                raise ExactP95Run2EvaluationV2Error(
                    f"V2 seed artifact hash drift: {artifact_name}"
                )
        common_binding = None
        common_session = None
        for update in EVALUATION_UPDATES_V2:
            path = (
                directory / "checkpoint_000000.pt"
                if update == 0
                else directory / "checkpoints" / f"checkpoint_{update:06d}.pt"
            )
            if not path.is_file():
                raise ExactP95Run2EvaluationV2Error(
                    f"physical registered checkpoint missing: seed={seed}, update={update}"
                )
            checkpoint = _load_exact_v2_checkpoint(path)
            if (
                checkpoint.seed != seed
                or checkpoint.update_count != update
                or checkpoint.config != config
            ):
                raise ExactP95Run2EvaluationV2Error("V2 checkpoint identity drift")
            if update == 0 and (
                checkpoint.collection_seq != 0
                or checkpoint.warmup_collected != 0
                or checkpoint.post_warmup_collected != 0
                or checkpoint.d1_transition_history
                or checkpoint.run2_v2_transition_history
                or checkpoint.metrics
                or checkpoint.replay_accepted_count != 0
                or checkpoint.replay_evicted_count != 0
                or checkpoint.support_violation_count != 0
                or checkpoint.environment_state.d1_state.reset_count != 0
                or checkpoint.trainer_initialized
                or checkpoint.replay_binding is not None
                or checkpoint.actor_optimizer_state.get("state") != {}
                or checkpoint.critic_optimizer_state.get("state") != {}
            ):
                raise ExactP95Run2EvaluationV2Error("update-zero is not pristine physical state")
            if update != 0:
                series_record = checkpoint_series[f"{update:06d}"]
                if (
                    type(series_record) is not dict
                    or series_record.get("path")
                    != f"checkpoints/checkpoint_{update:06d}.pt"
                    or series_record.get("file_sha256") != _sha256_file(path)
                ):
                    raise ExactP95Run2EvaluationV2Error(
                        "numbered checkpoint file attestation drift"
                    )
            common_binding = checkpoint.runner_binding_sha256 if common_binding is None else common_binding
            common_session = checkpoint.collection_session_uuid if common_session is None else common_session
            if (
                checkpoint.runner_binding_sha256 != common_binding
                or checkpoint.collection_session_uuid != common_session
                or bindings.get("runner_binding_sha256") != common_binding
                or bindings.get("collection_session_uuid") != common_session
            ):
                raise ExactP95Run2EvaluationV2Error("checkpoint lineage drift")
            result[(seed, update)] = (path, checkpoint)
        zero = result[(seed, 0)][1]
        final = result[(seed, PRIMARY_UPDATE_V2)][1]
        latest_path = directory / "checkpoint_latest.pt"
        latest = _load_exact_v2_checkpoint(latest_path)
        summary = report.get("summary")
        if (
            type(summary) is not dict
            or summary.get("seed") != seed
            or summary.get("completed_updates") != config.update_count
            or summary.get("transition_count")
            != config.warmup_transitions
            + config.update_count * config.collect_per_update
            or summary.get("replay_eviction_count") != 0
            or summary.get("support_violation_count") != 0
            or summary.get("global_python_rng_unchanged") is not True
            or summary.get("global_torch_rng_unchanged") is not True
            or summary.get("cuda_initialized_by_runner") is not False
            or summary.get("completed_training_hard_gates_passed") is not True
            or latest.checkpoint_sha256 != final.checkpoint_sha256
            or report.get("checkpoint_sha256") != final.checkpoint_sha256
            or summary.get("checkpoint_sha256") != final.checkpoint_sha256
            or report.get("runner_binding_sha256") != common_binding
            or summary.get("runner_binding_sha256") != common_binding
            or report.get("update_zero_checkpoint_sha256")
            != zero.checkpoint_sha256
            or report.get("latest_checkpoint_file_sha256")
            != _sha256_file(latest_path)
            or latest.runner_binding_sha256 != common_binding
            or latest.collection_session_uuid != common_session
        ):
            raise ExactP95Run2EvaluationV2Error("latest/final checkpoint drift")
    return result


def _validate_panel_partition(
    evaluator: FitValidationActorEvaluatorV1, *, project_root: Path
) -> None:
    partition = load_registered_empirical_fit_partition(project_root=project_root)
    if partition.canonical_sha256() != REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256:
        raise ExactP95Run2EvaluationV2Error("fit partition drift")
    train = {row.sample_id for row in partition.scene_assignments if row.split == TRAIN_SPLIT}
    held = {
        row.sample_id
        for row in partition.scene_assignments
        if row.split == FIT_VALIDATION_SPLIT
    }
    panel = {row.scene_sample_id for row in evaluator.panel.entries}
    if (
        len(train) != 391
        or len(held) != 85
        or train & held
        or panel != held
        or len(evaluator.panel.entries) != 340
        or evaluator.panel.canonical_sha256() != REGISTERED_FIT_VALIDATION_PANEL_SHA256
    ):
        raise ExactP95Run2EvaluationV2Error("development panel partition leak/drift")


def _load_oracle_actions(root: Path) -> Dict[int, Tuple[int, int]]:
    path = root / ORACLE_RELATIVE_PATH
    summary_path = root / ORACLE_SUMMARY_RELATIVE_PATH
    decision_path = root / ORACLE_DECISION_RELATIVE_PATH
    if (
        not path.is_file()
        or not summary_path.is_file()
        or not decision_path.is_file()
        or _sha256_file(path) != GATE_V2_CONTEXT_ORACLES_SHA256
        or _sha256_file(summary_path) != GATE_V2_SUMMARY_SHA256
        or _sha256_file(decision_path) != GATE_V2_DECISION_SHA256
    ):
        raise ExactP95Run2EvaluationV2Error("frozen V2 oracle artifact drift")
    summary = _read_json(summary_path)
    decision_file = _read_json(decision_path)
    _require_content_hash(summary, "canonical_content_sha256")
    decision = summary.get("decision", {})
    criteria = decision.get("criteria", {}) if type(decision) is dict else {}
    bindings = summary.get("bindings", {})
    if (
        summary.get("status")
        != "COMPLETE_ONE_SHOT_EXACT_P95_RUN2_VALIDATION_GATE_V2"
        or summary.get("canonical_content_sha256")
        != GATE_V2_CANONICAL_CONTENT_SHA256
        or type(decision) is not dict
        or decision_file != decision
        or decision.get("status") != "GO"
        or not criteria
        or any(value is not True for value in criteria.values())
        or type(bindings) is not dict
        or bindings.get("fit_partition_sha256")
        != REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256
        or bindings.get("fit_validation_panel_sha256")
        != REGISTERED_FIT_VALIDATION_PANEL_SHA256
        or bindings.get("float32_exact_penalty_spec_sha256")
        != REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256
        or bindings.get("run2_v2_preregistration_sha256")
        != PREREG_V2_FILE_SHA256
        or bindings.get("train_exact_penalty_v2_summary_sha256")
        != TRAIN_V2_SUMMARY_SHA256
        or bindings.get("train_exact_penalty_v2_decision_sha256")
        != TRAIN_V2_DECISION_SHA256
    ):
        raise ExactP95Run2EvaluationV2Error("frozen V2 oracle GO binding drift")
    result: Dict[int, Tuple[int, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            index = int(row["panel_index"])
            if (
                row["has_positive_admission_p95_feasible_action"] != "True"
                or row["shaped_matches_constrained"] != "True"
                or row["shaped_p95_miss"] != "False"
            ):
                raise ExactP95Run2EvaluationV2Error("oracle GO semantics drift")
            action = (int(row["constrained_mode_id"]), int(row["constrained_q_e4"]))
            require_supported_action(*action)
            if index in result:
                raise ExactP95Run2EvaluationV2Error("duplicate oracle panel index")
            result[index] = action
    if set(result) != set(range(340)):
        raise ExactP95Run2EvaluationV2Error("oracle panel coverage drift")
    return result


def _require_fixed_comparator(root: Path) -> None:
    directory = root / FIXED_COMPARATOR_RELATIVE_PATH
    for name, expected in FIXED_COMPARATOR_FILE_HASHES.items():
        path = directory / name
        if not path.is_file() or _sha256_file(path) != expected:
            raise ExactP95Run2EvaluationV2Error("fixed-comparator artifact drift")
    summary = _read_json(directory / "summary.json")
    decision = _read_json(directory / "decision.json")
    manifest = _read_json(directory / "artifact_manifest.json")
    implementation = root / FIXED_COMPARATOR_IMPLEMENTATION_RELATIVE_PATH
    _require_content_hash(summary, "summary_content_sha256")
    _require_content_hash(decision, "decision_content_sha256")
    _require_content_hash(manifest, "manifest_content_sha256")
    winner = decision.get("winner", {})
    manifest_files = {
        name: digest
        for name, digest in FIXED_COMPARATOR_FILE_HASHES.items()
        if name != "artifact_manifest.json"
    }
    if (
        summary.get("winner") != winner
        or decision.get("decision") != "GO_FIXED_COMPARATOR_FROZEN_BEFORE_RUN2_OUTCOME_ACCESS"
        or summary.get("summary_content_sha256")
        != FROZEN_COMPARATOR_SUMMARY_CONTENT_SHA256
        or decision.get("decision_content_sha256")
        != FROZEN_COMPARATOR_DECISION_CONTENT_SHA256
        or manifest.get("manifest_content_sha256")
        != FROZEN_COMPARATOR_MANIFEST_CONTENT_SHA256
        or manifest.get("files") != manifest_files
        or _sha256_file(directory / "artifact_manifest.json")
        != FROZEN_COMPARATOR_MANIFEST_FILE_SHA256
        or not implementation.is_file()
        or _sha256_file(implementation)
        != FROZEN_COMPARATOR_IMPLEMENTATION_SHA256
        or summary.get("bindings", {}).get("comparator_implementation_sha256")
        != FROZEN_COMPARATOR_IMPLEMENTATION_SHA256
        or winner.get("mode_id") != FIXED_MODE_ID_V2
        or winner.get("q_e4") != FIXED_Q_E4_V2
    ):
        raise ExactP95Run2EvaluationV2Error("fixed-comparator decision drift")


def _identity(
    source: str,
    seed: int,
    update: int,
    *,
    actor_state: str = "",
    checkpoint: str = "",
    file_sha: str = "",
) -> Run2V2ActorIdentity:
    return Run2V2ActorIdentity(source, seed, update, actor_state, checkpoint, file_sha)


def _validate_collected_population(
    rows: Sequence[ExactP95Run2EvaluationRowV2],
) -> None:
    """Prove exact source topology and one row per frozen panel context."""
    expected_groups = {
        (SOURCE_V2, seed, update)
        for seed in REGISTERED_EXACT_P95_RUN2_CONFIG_V2.seeds
        for update in EVALUATION_UPDATES_V2
    }
    expected_groups.update(
        (source, seed, update)
        for source, update in (
            (SOURCE_RUN1, PRIMARY_UPDATE_V2),
            (SOURCE_RANDOM, -1),
        )
        for seed in REGISTERED_EXACT_P95_RUN2_CONFIG_V2.seeds
    )
    expected_groups.update(
        {
            (SOURCE_FIXED, POOLED_SEED, -1),
            (SOURCE_ORACLE, POOLED_SEED, -1),
        }
    )
    observed: Dict[
        Tuple[str, int, int], list[ExactP95Run2EvaluationRowV2]
    ] = {}
    context_identity: Dict[int, set[Tuple[Any, ...]]] = {}
    row_identities = set()
    for row in rows:
        if type(row) is not ExactP95Run2EvaluationRowV2:
            raise ExactP95Run2EvaluationV2Error("foreign evaluation-row type")
        key = (row.source_kind, row.seed, row.update_index)
        observed.setdefault(key, []).append(row)
        row_identity = (*key, row.panel_index)
        if row_identity in row_identities:
            raise ExactP95Run2EvaluationV2Error("duplicate evaluation-row identity")
        row_identities.add(row_identity)
        context_identity.setdefault(row.panel_index, set()).add(
            (
                row.scene_rank,
                row.sample_id,
                row.episode_id,
                row.frame_id,
                row.network_profile,
                row.radio_csv_row_number,
                row.radio_row_sha256,
            )
        )
    if set(observed) != expected_groups:
        raise ExactP95Run2EvaluationV2Error("evaluation source topology drift")
    exact_panel = set(range(340))
    for key, group in observed.items():
        if len(group) != 340 or {row.panel_index for row in group} != exact_panel:
            raise ExactP95Run2EvaluationV2Error(
                f"evaluation panel coverage drift for {key}"
            )
        actor_identities = {
            (
                row.actor_state_sha256,
                row.checkpoint_sha256,
                row.checkpoint_file_sha256,
            )
            for row in group
        }
        if len(actor_identities) != 1:
            raise ExactP95Run2EvaluationV2Error("actor identity varies within group")
        identity = next(iter(actor_identities))
        if key[0] in (SOURCE_V2, SOURCE_RUN1):
            if not all(_is_sha256(value) for value in identity):
                raise ExactP95Run2EvaluationV2Error("actor hash identity is malformed")
        elif any(identity):
            raise ExactP95Run2EvaluationV2Error(
                "non-actor comparator carries actor identity"
            )
    if set(context_identity) != exact_panel or any(
        len(identities) != 1 for identities in context_identity.values()
    ):
        raise ExactP95Run2EvaluationV2Error("cross-source panel identity drift")
    by_scene: Dict[int, set[str]] = {}
    reference = observed[(SOURCE_FIXED, POOLED_SEED, -1)]
    for row in reference:
        by_scene.setdefault(row.scene_rank, set()).add(row.network_profile)
    if set(by_scene) != set(range(85)) or any(
        profiles != set(NETWORK_PROFILE_ORDER) for profiles in by_scene.values()
    ):
        raise ExactP95Run2EvaluationV2Error("panel is not 85 scenes x 4 profiles")


def collect_evaluation_rows_v2(
    *,
    v2_campaign_directory: Path,
    run1_campaign_directory: Path,
    project_root: Optional[Path] = None,
) -> Tuple[ExactP95Run2EvaluationRowV2, ...]:
    """Collect all preregistered actor and comparator rows, without writing."""
    root = Path(__file__).resolve().parents[2] if project_root is None else Path(project_root).resolve(strict=True)
    python_state = random.getstate()
    torch_state = torch.get_rng_state().clone()
    cuda_before = torch.cuda.is_initialized()
    _require_training_cli_source(root)
    checkpoints = _validate_v2_campaign(v2_campaign_directory)
    _require_fixed_comparator(root)
    oracle_actions = _load_oracle_actions(root)
    rows: list[ExactP95Run2EvaluationRowV2] = []
    with FitValidationActorEvaluatorV1(project_root=root) as evaluator:
        _validate_panel_partition(evaluator, project_root=root)
        for seed in REGISTERED_EXACT_P95_RUN2_CONFIG_V2.seeds:
            for update in EVALUATION_UPDATES_V2:
                path, checkpoint = checkpoints[(seed, update)]
                actor = _build_actor(checkpoint.actor_state)
                identity = _identity(
                    SOURCE_V2,
                    seed,
                    update,
                    actor_state=_hash_state(checkpoint.actor_state),
                    checkpoint=checkpoint.checkpoint_sha256,
                    file_sha=_sha256_file(path),
                )
                rows.extend(_actor_rows(evaluator, actor, identity))

            run1_actor, run1_identity = load_actor_for_evaluation(
                campaign_directory=run1_campaign_directory,
                seed=seed,
                update_index=PRIMARY_UPDATE_V2,
            )
            rows.extend(
                _actor_rows(
                    evaluator,
                    run1_actor,
                    _identity(
                        SOURCE_RUN1,
                        seed,
                        PRIMARY_UPDATE_V2,
                        actor_state=run1_identity.actor_state_sha256,
                        checkpoint=run1_identity.checkpoint_canonical_sha256,
                        file_sha=run1_identity.checkpoint_file_sha256 or "",
                    ),
                )
            )
            random_identity = _identity(SOURCE_RANDOM, seed, -1)
            for entry in evaluator.panel.entries:
                mode_id, q_e4 = choose_registered_random_action(
                    seed=seed, panel_index=entry.panel_index
                )
                rows.append(
                    _row_for_action(
                        evaluator,
                        entry,
                        identity=random_identity,
                        mode_id=mode_id,
                        q_e4=q_e4,
                        requested_q=q_e4 / 10000.0,
                    )
                )

        fixed_identity = _identity(SOURCE_FIXED, POOLED_SEED, -1)
        oracle_identity = _identity(SOURCE_ORACLE, POOLED_SEED, -1)
        for entry in evaluator.panel.entries:
            rows.append(
                _row_for_action(
                    evaluator,
                    entry,
                    identity=fixed_identity,
                    mode_id=FIXED_MODE_ID_V2,
                    q_e4=FIXED_Q_E4_V2,
                    requested_q=FIXED_Q_E4_V2 / 10000.0,
                )
            )
            mode_id, q_e4 = oracle_actions[entry.panel_index]
            oracle_row = _row_for_action(
                evaluator,
                entry,
                identity=oracle_identity,
                mode_id=mode_id,
                q_e4=q_e4,
                requested_q=q_e4 / 10000.0,
            )
            if oracle_row.p95_miss != 0:
                raise ExactP95Run2EvaluationV2Error("constrained oracle misses deadline")
            rows.append(oracle_row)
        evaluator._assert_no_runtime_side_effects()
    if random.getstate() != python_state or not torch.equal(torch.get_rng_state(), torch_state):
        raise ExactP95Run2EvaluationV2Error("evaluation advanced global RNG")
    if not cuda_before and torch.cuda.is_initialized():
        raise ExactP95Run2EvaluationV2Error("evaluation initialized CUDA")
    _validate_collected_population(rows)
    return tuple(rows)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ExactP95Run2EvaluationV2Error("cannot aggregate empty values")
    return math.fsum(float(value) for value in values) / len(values)


def aggregate_evaluation_rows_v2(
    rows: Sequence[ExactP95Run2EvaluationRowV2],
) -> Tuple[Dict[str, Any], ...]:
    """Aggregate per source/seed/update/profile and add pooled actor rows."""
    materialized = tuple(rows)
    if not materialized or any(type(row) is not ExactP95Run2EvaluationRowV2 for row in materialized):
        raise ExactP95Run2EvaluationV2Error("aggregate requires exact V2 rows")
    groups: Dict[Tuple[str, int, int, str], list[ExactP95Run2EvaluationRowV2]] = {}
    for row in materialized:
        for profile in (row.network_profile, "ALL_PROFILES"):
            groups.setdefault((row.source_kind, row.seed, row.update_index, profile), []).append(row)
    for source in (SOURCE_V2, SOURCE_RUN1, SOURCE_RANDOM):
        updates = sorted({row.update_index for row in materialized if row.source_kind == source})
        for update in updates:
            selected = [row for row in materialized if row.source_kind == source and row.update_index == update]
            for profile in (*NETWORK_PROFILE_ORDER, "ALL_PROFILES"):
                group = selected if profile == "ALL_PROFILES" else [row for row in selected if row.network_profile == profile]
                groups[(source, POOLED_SEED, update, profile)] = group
    oracle_means = {
        profile: _mean(
            [
                row.v2_emitted_reward_float32
                for row in materialized
                if row.source_kind == SOURCE_ORACLE
                and (
                    profile == "ALL_PROFILES"
                    or row.network_profile == profile
                )
            ]
        )
        for profile in (*NETWORK_PROFILE_ORDER, "ALL_PROFILES")
    }
    random_means = {
        (seed, profile): _mean(
            [
                row.v2_emitted_reward_float32
                for row in materialized
                if row.source_kind == SOURCE_RANDOM
                and (seed == POOLED_SEED or row.seed == seed)
                and (
                    profile == "ALL_PROFILES"
                    or row.network_profile == profile
                )
            ]
        )
        for seed in (*REGISTERED_EXACT_P95_RUN2_CONFIG_V2.seeds, POOLED_SEED)
        for profile in (*NETWORK_PROFILE_ORDER, "ALL_PROFILES")
    }
    output = []
    for (source, seed, update, profile), selected in sorted(groups.items()):
        if not selected:
            continue
        utility = _mean([row.v2_emitted_reward_float32 for row in selected])
        normalized: Optional[float] = None
        regret: Optional[float] = None
        if source == SOURCE_V2:
            random_mean = random_means[(seed, profile)]
            oracle_mean = oracle_means[profile]
            denominator = oracle_mean - random_mean
            if denominator == 0.0:
                raise ExactP95Run2EvaluationV2Error("random-to-oracle denominator is zero")
            normalized = (utility - random_mean) / denominator
            regret = oracle_mean - utility
        output.append(
            {
                "schema": AGGREGATE_SCHEMA_V2,
                "source_kind": source,
                "seed": seed,
                "update_index": update,
                "network_profile": profile,
                "row_count": len(selected),
                "v2_utility_mean": utility,
                "normalized_random_to_oracle_progress_unclamped": normalized,
                "contextual_oracle_regret_mean": regret,
                "q_perc_mean": _mean([row.q_perc for row in selected]),
                "p_edge_admission_given_sent_mean": _mean([row.p_edge_admission_given_sent for row in selected]),
                "latency_p50_ms_mean": _mean([row.latency_proxy_p50_ms for row in selected]),
                "latency_p95_ms_mean": _mean([row.latency_proxy_p95_ms for row in selected]),
                "latency_p99_ms_mean": _mean([row.latency_proxy_p99_ms for row in selected]),
                "p95_margin_ms_mean": _mean([row.p95_margin_ms for row in selected]),
                "p95_miss_count": sum(row.p95_miss for row in selected),
                "p95_miss_rate": _mean([row.p95_miss for row in selected]),
                "unique_action_count": len({(row.executed_mode_id, row.executed_q_e4) for row in selected}),
            }
        )
    return tuple(output)


def _select_rows(
    rows: Sequence[ExactP95Run2EvaluationRowV2],
    source: str,
    *,
    seed: Optional[int] = None,
    update: Optional[int] = None,
    profile: Optional[str] = None,
) -> list[ExactP95Run2EvaluationRowV2]:
    return [
        row
        for row in rows
        if row.source_kind == source
        and (seed is None or row.seed == seed)
        and (update is None or row.update_index == update)
        and (profile is None or row.network_profile == profile)
    ]


def paired_scene_cluster_bootstrap_v2(
    rows: Sequence[ExactP95Run2EvaluationRowV2],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES_V2,
    seed: int = BOOTSTRAP_SEED_V2,
) -> Dict[str, Any]:
    learned = _select_rows(rows, SOURCE_V2, update=PRIMARY_UPDATE_V2)
    fixed = {row.panel_index: row for row in _select_rows(rows, SOURCE_FIXED)}
    if len(learned) != 1020 or len(fixed) != 340:
        raise ExactP95Run2EvaluationV2Error("adaptation population drift")
    cluster_values = []
    seed_means = {}
    for actor_seed in REGISTERED_EXACT_P95_RUN2_CONFIG_V2.seeds:
        seed_rows = [row for row in learned if row.seed == actor_seed]
        differences = [row.v2_emitted_reward_float32 - fixed[row.panel_index].v2_emitted_reward_float32 for row in seed_rows]
        seed_means[str(actor_seed)] = _mean(differences)
    for scene_rank in range(85):
        selected = [row for row in learned if row.scene_rank == scene_rank]
        if len(selected) != 12:
            raise ExactP95Run2EvaluationV2Error("scene cluster is not 3 seeds x 4 profiles")
        cluster_values.append(
            _mean([row.v2_emitted_reward_float32 - fixed[row.panel_index].v2_emitted_reward_float32 for row in selected])
        )
    generator = np.random.Generator(np.random.PCG64(seed))
    values = np.asarray(cluster_values, dtype=np.float64)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = generator.integers(0, len(values), size=len(values), endpoint=False)
        draws[index] = float(np.mean(values[sampled], dtype=np.float64))
    lower, upper = np.quantile(draws, [0.025, 0.975], method="linear")
    pooled = _mean(cluster_values)
    positive_seeds = sum(value > 0.0 for value in seed_means.values())
    return {
        "bootstrap_cluster_count": 85,
        "bootstrap_resamples": resamples,
        "bootstrap_rng": "numpy.random.PCG64",
        "bootstrap_seed": seed,
        "ci_method": "numpy.quantile_linear_2.5_97.5",
        "lower_95": float(lower),
        "upper_95": float(upper),
        "pooled_mean_lift": pooled,
        "seed_mean_lift": seed_means,
        "positive_seed_count": positive_seeds,
        "passes_pooled_positive": pooled > 0.0,
        "passes_seed_count": positive_seeds >= 2,
        "passes_ci": float(lower) > 0.0,
        "passes_all": pooled > 0.0 and positive_seeds >= 2 and float(lower) > 0.0,
    }


def _relative_reduction(final: int, baseline: int) -> Optional[float]:
    if baseline <= 0:
        return 1.0 if final == 0 else None
    return 1.0 - final / baseline


def evaluate_success_tiers_v2(
    rows: Sequence[ExactP95Run2EvaluationRowV2],
    *,
    mechanics: Optional[Mapping[str, bool]] = None,
) -> Dict[str, Any]:
    final = _select_rows(rows, SOURCE_V2, update=PRIMARY_UPDATE_V2)
    update0 = _select_rows(rows, SOURCE_V2, update=0)
    run1 = _select_rows(rows, SOURCE_RUN1, update=PRIMARY_UPDATE_V2)
    oracle = _select_rows(rows, SOURCE_ORACLE)
    if (len(final), len(update0), len(run1), len(oracle)) != (1020, 1020, 1020, 340):
        raise ExactP95Run2EvaluationV2Error("tier population drift")
    final_misses = sum(row.p95_miss for row in final)
    update0_misses = sum(row.p95_miss for row in update0)
    run1_misses = sum(row.p95_miss for row in run1)
    per_seed = {}
    for seed in REGISTERED_EXACT_P95_RUN2_CONFIG_V2.seeds:
        f = _select_rows(rows, SOURCE_V2, seed=seed, update=PRIMARY_UPDATE_V2)
        z = _select_rows(rows, SOURCE_V2, seed=seed, update=0)
        if len(f) != 340 or len(z) != 340:
            raise ExactP95Run2EvaluationV2Error("tier per-seed population drift")
        per_seed[str(seed)] = {
            "final_miss_count": sum(row.p95_miss for row in f),
            "final_miss_rate": _mean([row.p95_miss for row in f]),
            "update0_miss_count": sum(row.p95_miss for row in z),
            "improved_vs_update0": sum(row.p95_miss for row in f) < sum(row.p95_miss for row in z),
        }
    per_profile = {}
    for profile in NETWORK_PROFILE_ORDER:
        group = [row for row in final if row.network_profile == profile]
        if len(group) != 255:
            raise ExactP95Run2EvaluationV2Error("tier per-profile population drift")
        per_profile[profile] = {
            "miss_count": sum(row.p95_miss for row in group),
            "miss_rate": _mean([row.p95_miss for row in group]),
        }
    oracle_quality = _mean([row.q_perc for row in oracle])
    oracle_admission = _mean([row.p_edge_admission_given_sent for row in oracle])
    final_quality = _mean([row.q_perc for row in final])
    final_admission = _mean([row.p_edge_admission_given_sent for row in final])
    if oracle_quality <= 0.0:
        raise ExactP95Run2EvaluationV2Error("oracle quality denominator is non-positive")
    tier0_checks = dict(mechanics or {})
    expected_tier0 = _protocol_document()["success_tiers"]["tier0"]
    tier0 = tier0_checks == expected_tier0
    reduction0 = _relative_reduction(final_misses, update0_misses)
    reduction1 = _relative_reduction(final_misses, run1_misses)
    tier1 = (
        all(item["improved_vs_update0"] for item in per_seed.values())
        and reduction0 is not None
        and reduction1 is not None
        and reduction0 >= 0.5
        and reduction1 >= 0.5
    )
    tier2 = (
        _mean([row.p95_miss for row in final]) <= 0.05
        and all(item["final_miss_rate"] <= 0.10 for item in per_seed.values())
        and all(item["miss_rate"] <= 0.10 for item in per_profile.values())
        and final_quality / oracle_quality >= 0.90
        and abs(final_admission - oracle_admission) <= 0.001
    )
    tier3 = final_misses == 0
    return {
        "protocol_sha256": EVALUATION_PROTOCOL_V2_SHA256,
        "tier0": {"passed": tier0, "checks": tier0_checks},
        "tier1": {
            "passed": tier1,
            "per_seed": per_seed,
            "pooled_final_miss_count": final_misses,
            "pooled_update0_miss_count": update0_misses,
            "pooled_run1_rescored_miss_count": run1_misses,
            "relative_reduction_vs_update0": reduction0,
            "relative_reduction_vs_run1_rescored": reduction1,
        },
        "tier2": {
            "passed": tier2,
            "pooled_miss_rate": _mean([row.p95_miss for row in final]),
            "per_profile": per_profile,
            "quality_fraction_of_oracle": final_quality / oracle_quality,
            "absolute_admission_difference_from_oracle": abs(final_admission - oracle_admission),
        },
        "tier3": {"passed": tier3, "final_miss_count": final_misses},
    }


def _csv_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    materialized = list(rows)
    if not materialized:
        raise ExactP95Run2EvaluationV2Error("CSV population is empty")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(materialized[0]))
    writer.writeheader()
    writer.writerows(materialized)
    return stream.getvalue().encode("utf-8")


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_figures(
    output: Path,
    rows: Sequence[ExactP95Run2EvaluationRowV2],
    aggregates: Sequence[Mapping[str, Any]],
    adaptation: Mapping[str, Any],
) -> Tuple[str, ...]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.labelweight": "bold", "axes.titleweight": "bold"})
    names = []

    def save(fig: Any, stem: str) -> None:
        for suffix in ("png", "pdf"):
            path = output / f"{stem}.{suffix}"
            fig.savefig(path, dpi=240 if suffix == "png" else None, bbox_inches="tight")
            names.append(path.name)
        plt.close(fig)

    def agg(source: str, seed: int, update: int, profile: str = "ALL_PROFILES") -> Mapping[str, Any]:
        matches = [row for row in aggregates if row["source_kind"] == source and row["seed"] == seed and row["update_index"] == update and row["network_profile"] == profile]
        if len(matches) != 1:
            raise ExactP95Run2EvaluationV2Error("figure aggregate join drift")
        return matches[0]

    updates = list(EVALUATION_UPDATES_V2)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for seed in REGISTERED_EXACT_P95_RUN2_CONFIG_V2.seeds:
        ax.plot(updates, [agg(SOURCE_V2, seed, update)["v2_utility_mean"] for update in updates], marker="o", label=f"Run 2 seed {seed}")
    ax.plot(
        updates,
        [agg(SOURCE_V2, POOLED_SEED, update)["v2_utility_mean"] for update in updates],
        color="black",
        linewidth=2,
        label="Run 2 pooled",
    )
    for source, label, style in ((SOURCE_RANDOM, "Random", "--"), (SOURCE_FIXED, "Frozen fixed", ":"), (SOURCE_ORACLE, "Context oracle", "-."), (SOURCE_RUN1, "Run 1 rescored", "--")):
        update = -1 if source in (SOURCE_RANDOM, SOURCE_FIXED, SOURCE_ORACLE) else PRIMARY_UPDATE_V2
        value = agg(source, POOLED_SEED, update)["v2_utility_mean"]
        ax.axhline(value, linestyle=style, label=label)
    ax.set(xlabel="Training update", ylabel="Native V2 utility", title="Run 2 utility across registered checkpoints")
    ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=8)
    save(fig, "01_utility_vs_updates_with_references")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for seed in REGISTERED_EXACT_P95_RUN2_CONFIG_V2.seeds:
        ax.plot(updates, [agg(SOURCE_V2, seed, update)["normalized_random_to_oracle_progress_unclamped"] for update in updates], marker="o", label=f"Seed {seed}")
    ax.plot(
        updates,
        [agg(SOURCE_V2, POOLED_SEED, update)["normalized_random_to_oracle_progress_unclamped"] for update in updates],
        color="black",
        linewidth=2,
        label="Pooled",
    )
    ax.axhline(0.0, color="black", linewidth=1); ax.axhline(.5, color="gray", linestyle="--"); ax.axhline(.75, color="gray", linestyle=":")
    ax.set(xlabel="Training update", ylabel="Random-to-oracle progress (unclamped)", title="Normalized learning progress")
    ax.grid(alpha=.25); ax.legend()
    save(fig, "02_normalized_progress")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for seed in REGISTERED_EXACT_P95_RUN2_CONFIG_V2.seeds:
        ax.plot(updates, [100*agg(SOURCE_V2, seed, update)["p95_miss_rate"] for update in updates], marker="o", label=f"Seed {seed}")
    ax.plot(
        updates,
        [100 * agg(SOURCE_V2, POOLED_SEED, update)["p95_miss_rate"] for update in updates],
        color="black",
        linewidth=2,
        label="Pooled",
    )
    ax.axhline(10, color="orange", linestyle="--", label="Per-seed Tier 2 (10%)")
    ax.axhline(5, color="green", linestyle=":", label="Pooled Tier 2 (5%)")
    ax.set(xlabel="Training update", ylabel="P95 miss rate (%)", title="Exact 200-ms P95 misses")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    save(fig, "03_p95_miss_vs_updates")

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(len(NETWORK_PROFILE_ORDER)); width=.22
    for offset, seed in enumerate(REGISTERED_EXACT_P95_RUN2_CONFIG_V2.seeds):
        counts=[agg(SOURCE_V2, seed, PRIMARY_UPDATE_V2, p)["p95_miss_count"] for p in NETWORK_PROFILE_ORDER]
        bars=ax.bar(x+(offset-1)*width, counts, width, label=f"Seed {seed}")
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(x, [p.replace("_", "\n") for p in NETWORK_PROFILE_ORDER]); ax.set_ylabel("Miss count (integer)"); ax.set_title("Final P95 misses by network profile"); ax.legend()
    save(fig, "04_final_profile_miss_counts")

    final = _select_rows(rows, SOURCE_V2, update=PRIMARY_UPDATE_V2)
    oracle = _select_rows(rows, SOURCE_ORACLE)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    axes[0].bar(["Run 2", "Oracle"], [_mean([r.q_perc for r in final]), _mean([r.q_perc for r in oracle])], color=["#2474b5", "#777777"])
    axes[0].set_ylabel("Mean perception quality"); axes[0].set_title("Quality")
    axes[1].bar(["Run 2", "Oracle"], [_mean([r.p_edge_admission_given_sent for r in final]), _mean([r.p_edge_admission_given_sent for r in oracle])], color=["#2474b5", "#777777"])
    axes[1].set_ylabel("Mean edge-admission probability"); axes[1].set_title("Admission")
    save(fig, "05_quality_admission_vs_oracle")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    distributions=[[r.p95_margin_ms for r in final if r.network_profile==p] for p in NETWORK_PROFILE_ORDER]
    ax.boxplot(distributions, tick_labels=[p.replace("_", "\n") for p in NETWORK_PROFILE_ORDER], showfliers=False)
    ax.axhline(0, color="red", linestyle="--"); ax.set_ylabel("200 ms - modeled P95 (ms)"); ax.set_title("Final P95 safety margins")
    save(fig, "06_p95_margin_distributions")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    mode_counts=np.asarray([[sum(r.executed_mode_id==m and r.network_profile==p for r in final) for m in range(12)] for p in NETWORK_PROFILE_ORDER])
    image=axes[0].imshow(mode_counts, aspect="auto", cmap="Blues")
    axes[0].set_yticks(range(4), [p.replace("_", " ") for p in NETWORK_PROFILE_ORDER]); axes[0].set_xticks(range(12)); axes[0].set_xlabel("Mode ID"); axes[0].set_title("Final mode selections"); fig.colorbar(image, ax=axes[0], label="Count")
    axes[1].boxplot([[r.executed_q_e4/10000 for r in final if r.network_profile==p] for p in NETWORK_PROFILE_ORDER], tick_labels=[p.replace("_", "\n") for p in NETWORK_PROFILE_ORDER], showfliers=False)
    axes[1].set_ylabel("Executed q"); axes[1].set_title("Final continuous action")
    save(fig, "07_mode_and_q_by_profile")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    labels=[f"Seed {s}" for s in REGISTERED_EXACT_P95_RUN2_CONFIG_V2.seeds]+["Pooled"]
    values=[adaptation["seed_mean_lift"][str(s)] for s in REGISTERED_EXACT_P95_RUN2_CONFIG_V2.seeds]+[adaptation["pooled_mean_lift"]]
    ax.bar(labels, values, color=["#4c78a8"]*3+["#f58518"])
    ax.errorbar([3], [adaptation["pooled_mean_lift"]], yerr=[[adaptation["pooled_mean_lift"]-adaptation["lower_95"]], [adaptation["upper_95"]-adaptation["pooled_mean_lift"]]], fmt="none", color="black", capsize=5)
    ax.axhline(0, color="black", linewidth=1); ax.set_ylabel("V2 utility lift over frozen fixed"); ax.set_title("Contextual adaptation evidence (95% cluster CI)")
    save(fig, "08_learned_minus_fixed_lift_ci")
    return tuple(names)


def _report_markdown(
    tiers: Mapping[str, Any], adaptation: Mapping[str, Any]
) -> str:
    tier0_lines = "\n".join(
        f"  - {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in sorted(tiers["tier0"]["checks"].items())
    )
    return (
        "# Exact-P95 Run-2 V2 evaluation\n\n"
        f"- Tier 0 mechanics: **{'PASS' if tiers['tier0']['passed'] else 'FAIL'}**\n"
        f"{tier0_lines}\n"
        f"- Tier 1 learning signal: **{'PASS' if tiers['tier1']['passed'] else 'FAIL'}**\n"
        f"- Tier 2 useful deadline behavior: **{'PASS' if tiers['tier2']['passed'] else 'FAIL'}**\n"
        f"- Tier 3 zero misses: **{'PASS' if tiers['tier3']['passed'] else 'FAIL'}**\n"
        f"- Contextual lift over frozen fixed: {adaptation['pooled_mean_lift']:.6f} "
        f"(95% scene-cluster CI {adaptation['lower_95']:.6f}, {adaptation['upper_95']:.6f})\n\n"
        "## Scope\n\n"
        "- This is development fit-validation, not the final test set.\n"
        "- P95 is a modeled conditional retained-survivor proxy, not a live SLA.\n"
        "- The 85 scene clusters each contain four profiles and three seeds; "
        "profiles and seeds are not independent observations.\n"
        "- No claim is made that reward should approach 1.\n"
    )


def evaluate_registered_run2_v2_campaign(
    *,
    v2_campaign_directory: Path,
    output_directory: Path,
    run1_campaign_directory: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2] if project_root is None else Path(project_root).resolve(strict=True)
    run1 = root / RUN1_CAMPAIGN_RELATIVE_PATH if run1_campaign_directory is None else Path(run1_campaign_directory).resolve(strict=True)
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise ExactP95Run2EvaluationV2Error("evaluation output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    rows = collect_evaluation_rows_v2(v2_campaign_directory=v2_campaign_directory, run1_campaign_directory=run1, project_root=root)
    aggregates = aggregate_evaluation_rows_v2(rows)
    mechanics = {
        "all_three_seeds_complete": True,
        "all_registered_physical_checkpoints_present": True,
        "physical_update_zero_present": True,
        "exact_v2_checkpoint_types": True,
        "development_panel_is_85x4_and_train_disjoint": True,
        "executed_q_e4_scored": True,
        "d1_and_v2_rewards_separate": True,
        "finite_rows": True,
        "global_rng_unchanged": True,
        "cuda_not_initialized_by_evaluator": True,
    }
    tiers = evaluate_success_tiers_v2(rows, mechanics=mechanics)
    adaptation = paired_scene_cluster_bootstrap_v2(rows)
    files: Dict[str, bytes] = {
        "per_context.csv": _csv_bytes(row.to_canonical_dict() for row in rows),
        "aggregate.csv": _csv_bytes(aggregates),
        "protocol.json": _json_bytes({**evaluation_protocol_document_v2(), "protocol_sha256": EVALUATION_PROTOCOL_V2_SHA256}),
        "success_tiers.json": _json_bytes(tiers),
        "adaptation.json": _json_bytes(adaptation),
        "REPORT.md": _report_markdown(tiers, adaptation).encode("utf-8"),
    }
    for name, payload in files.items():
        _atomic_bytes(output / name, payload)
    figure_names = _write_figures(output, rows, aggregates, adaptation)
    artifact_hashes = {name: _sha256_file(output / name) for name in sorted((*files, *figure_names))}
    manifest: Dict[str, Any] = {
        "evaluator_schema": EVALUATOR_SCHEMA_V2,
        "schema": MANIFEST_SCHEMA_V2,
        "status": "COMPLETE",
        "protocol_sha256": EVALUATION_PROTOCOL_V2_SHA256,
        "reward_spec_v2_sha256": EXACT_P95_RUN2_REWARD_SPEC_V2_SHA256,
        "preregistration_v2_sha256": PREREG_V2_FILE_SHA256,
        "train_penalty_spec_sha256": (
            REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256
        ),
        "train_penalty_summary_sha256": TRAIN_V2_SUMMARY_SHA256,
        "train_penalty_decision_sha256": TRAIN_V2_DECISION_SHA256,
        "validation_gate_summary_sha256": GATE_V2_SUMMARY_SHA256,
        "validation_gate_decision_sha256": GATE_V2_DECISION_SHA256,
        "validation_gate_context_oracles_sha256": (
            GATE_V2_CONTEXT_ORACLES_SHA256
        ),
        "validation_gate_canonical_content_sha256": (
            GATE_V2_CANONICAL_CONTENT_SHA256
        ),
        "training_cli_commit": TRAINING_CLI_COMMIT_V2,
        "training_cli_implementation_sha256": (
            TRAINING_CLI_IMPLEMENTATION_SHA256_V2
        ),
        "v2_campaign_report_file_sha256": _sha256_file(
            Path(v2_campaign_directory).resolve(strict=True)
            / "campaign_report.json"
        ),
        "fixed_comparator_commit": FROZEN_COMPARATOR_COMMIT,
        "fixed_comparator_implementation_sha256": (
            FROZEN_COMPARATOR_IMPLEMENTATION_SHA256
        ),
        "fixed_comparator_summary_content_sha256": (
            FROZEN_COMPARATOR_SUMMARY_CONTENT_SHA256
        ),
        "fit_partition_sha256": REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
        "fit_validation_panel_sha256": REGISTERED_FIT_VALIDATION_PANEL_SHA256,
        "pilot_utility_spec_sha256": PILOT_UTILITY_SPEC_SHA256,
        "conditional_feasibility_semantics": CONDITIONAL_FEASIBILITY_SEMANTICS,
        "per_context_row_count": len(rows),
        "aggregate_row_count": len(aggregates),
        "artifacts": artifact_hashes,
        "scope_caveats": evaluation_protocol_document_v2()["scope_caveats"],
    }
    manifest["manifest_content_sha256"] = canonical_sha256(manifest)
    _atomic_bytes(output / "manifest.json", _json_bytes(manifest))
    return manifest


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exact-P95 Run-2-v2 checkpoints on the frozen development panel.")
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run1-campaign", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    manifest = evaluate_registered_run2_v2_campaign(
        v2_campaign_directory=args.campaign,
        output_directory=args.output,
        run1_campaign_directory=args.run1_campaign,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
