"""Hash-closed terminal replay for emitted-float32 exact-P95 Run 2 v2.

The historical v1 replay remains immutable.  This module accepts an exact,
fully revalidated D1 terminal transition, derives the v2 reward in the pinned
Python-binary64 order, and emits the learning target through one explicit CPU
float32 conversion.  Binary64 diagnostics and their IEEE-754 bit patterns are
preserved separately from the float32 target.

The replay is terminal: a future trainer must call ``batch.revalidate()`` and
use ``batch.terminal_target()`` verbatim.  Reward or discount recomputation in
that trainer would violate this contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Deque, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor

from .action_contract import EXPECTED_MODE_COUNT
from .empirical_contextual_contract import (
    PILOT_UTILITY_SPEC_SHA256,
    require_supported_action,
)
from .empirical_contextual_terminal_replay import (
    EmpiricalTerminalBindingV1,
    EmpiricalTerminalTransitionV1,
)
from .state_reward_transition_contract import (
    POLICY_FEATURE_COUNT,
    POLICY_FEATURE_ORDER,
)
from .transaction_identity import canonical_sha256

__all__ = [
    "EXACT_P95_RUN2_REWARD_SPEC_V2_SHA256",
    "ExactP95Run2BatchV2",
    "ExactP95Run2BindingV2Error",
    "ExactP95Run2ReplayBindingV2",
    "ExactP95Run2ReplayV2",
    "ExactP95Run2ReplayV2Error",
    "ExactP95Run2RewardBindingV2",
    "ExactP95Run2ShapedTransitionV2",
    "RUN2_V2_DEADLINE_MS",
    "RUN2_V2_DEADLINE_PENALTY",
    "RUN2_V2_DEADLINE_PENALTY_HEX",
    "RUN2_V2_DEADLINE_PENALTY_UINT64_HEX",
    "RUN2_V2_TERMINAL_DISCOUNT",
    "exact_p95_run2_reward_spec_document_v2",
]


RUN2_V2_DEADLINE_MS = 200.0
RUN2_V2_DEADLINE_PENALTY = 0.5742957622788527
RUN2_V2_DEADLINE_PENALTY_HEX = "0x1.260a181a70290p-1"
RUN2_V2_DEADLINE_PENALTY_UINT64_HEX = "0x3fe260a181a70290"
RUN2_V2_TERMINAL_DISCOUNT = 0.0

RUN2_V2_REPLAY_SCHEMA = "splitfusion.exact_p95_run2_terminal_replay.v2"
RUN2_V2_REWARD_SCHEMA = "splitfusion.exact_p95_run2_reward.v2"
RUN2_V2_TRANSITION_SCHEMA = "splitfusion.exact_p95_run2_shaped_transition.v2"
RUN2_V2_BATCH_AUDIT_SCHEMA = "splitfusion.exact_p95_run2_batch_audit_row.v2"
RUN2_V2_RUNTIME_ARITHMETIC = (
    "PYTHON_BINARY64_BASE_THEN_CONDITIONAL_BINARY64_SUBTRACTION_THEN_"
    "ONE_EXPLICIT_CPU_TORCH_FLOAT32_SCALAR_EMISSION"
)

# Keep the execution-time replay boundary independent of the exhaustive
# train/development analysis modules.  Importing the derivation module would
# transitively import the split-oracle evaluator and validation-panel modules,
# even though replay needs only this already-frozen digest and the two scalar
# operations below.
REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256 = (
    "9ea4e4a3d2ffa791ae189b1ff478871f48ca6d84572b2830a0a6795f2cd254e3"
)

TRAIN_V2_RELATIVE_DIRECTORY = (
    "experiments/splitfusion_hybrid_sac_fit_validation_v1/"
    "20260921_train_exact_p95_deadline_penalty_v2"
)
TRAIN_V2_SUMMARY_SHA256 = (
    "4e102b3309c1c2bf8c6e7868957285f440f091a194ff06756d1c5f49d33b8dcd"
)
TRAIN_V2_DECISION_SHA256 = (
    "eeef505e81be93a01cc6d289315738191f9139ed13c387c7bbb54231d219fda7"
)
TRAIN_V2_REPORT_SHA256 = (
    "e34e605da83159c7e458aa51c6e991dd120b1f5b4001f40d37e0324de667ee25"
)
TRAIN_V2_ORACLES_SHA256 = (
    "4ed4a5bc92643781462710fcb70d7792307f9ac723d98422c0ca12550629c4f2"
)
TRAIN_V2_IMPLEMENTATION_SHA256 = (
    "40975469634225c651c78a8bedcdebd12399f875dd55680c1f83c91573bd20ff"
)
TRAIN_V2_CANONICAL_CONTENT_SHA256 = (
    "a5ca38cc5415bd1aaf612907b379717a1fa5c0374a61ae44358c5c7551829953"
)

PREREG_V2_RELATIVE_DIRECTORY = (
    "experiments/splitfusion_hybrid_sac_fit_validation_v1/"
    "20260921_exact_p95_run2_preregistration_v2"
)
PREREG_V2_FILE_SHA256 = (
    "b1b42a622ae514f469d7a4c49214abe2002739e4bedf5f1c94ef52c3d298b8ac"
)
PREREG_V2_REPORT_SHA256 = (
    "7bdf6c1217119baee299a92cc706cd6f475a3cb86fe0b8d9f63c9af7f86ea2b0"
)
PREREG_V2_HASH_MANIFEST_SHA256 = (
    "7557b3be3040c4aa7e88e2e3c80328d306610cbd15e1eb53dfa93858ba2aa650"
)

GATE_V2_RELATIVE_DIRECTORY = (
    "experiments/splitfusion_hybrid_sac_fit_validation_v1/"
    "20260921_exact_p95_run2_pretraining_validation_gate_v2"
)
GATE_V2_SUMMARY_SHA256 = (
    "c65b50a5c43568292a0033ba4ceb88223a375aafa200815aff6a74fa718762c0"
)
GATE_V2_DECISION_SHA256 = (
    "16ed4b1616616170344d77383bf45d7daef6e623144b11cdcf71949f4a541af7"
)
GATE_V2_REPORT_SHA256 = (
    "053685b48a36c709aafa90676a7f60e3b57b2762590a75c2d2b302e3b4f029cc"
)
GATE_V2_PROFILE_SUMMARY_SHA256 = (
    "19ab8493a249d0f40d7b69f67dee2118ba5b0589636132e161c12a979becfcf5"
)
GATE_V2_CONTEXT_ORACLES_SHA256 = (
    "0e1775d2d070967f3e906982d2968f0a9ab9e2682399671b5b8570bdaa90c988"
)
GATE_V2_IMPLEMENTATION_SHA256 = (
    "f3363b9f267f5bd7339e8416a91b8db9e96c194b902e37dd566a5736dc3fabbd"
)
GATE_V2_TEST_SHA256 = (
    "12bc42686f74b7320852a6786b7b7d493f25e328ebe8cd78b32c902ef5123c4e"
)
GATE_V2_CANONICAL_CONTENT_SHA256 = (
    "a15cb5c5ca202d2cc06b1493a6561562786bc1fc4641f6cc7590472b71ec82d5"
)


class ExactP95Run2ReplayV2Error(ValueError):
    """A v2 transition, replay, or batch failed closed."""


class ExactP95Run2BindingV2Error(ExactP95Run2ReplayV2Error):
    """A v2 evidence or replay binding differs from the frozen contract."""


def _float64_bits_hex(value: float) -> str:
    return f"0x{struct.unpack('>Q', struct.pack('>d', float(value)))[0]:016x}"


def _float32_bits_hex(value: float) -> str:
    return f"0x{struct.unpack('>I', struct.pack('>f', float(value)))[0]:08x}"


def _float32_roundtrip(value: float) -> float:
    return float(struct.unpack(">f", struct.pack(">f", float(value)))[0])


def _require_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ExactP95Run2ReplayV2Error(f"{name} must be an exact finite float")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _emit_cpu_float32(value64: float) -> float:
    """Perform the sole binary64-to-learning-target emission on CPU."""
    value = _require_float(value64, "binary64 reward")
    emitted = torch.tensor(value, dtype=torch.float32, device="cpu")
    if emitted.device.type != "cpu" or not bool(torch.isfinite(emitted)):
        raise ExactP95Run2ReplayV2Error("CPU float32 reward emission failed")
    return float(emitted.item())


def _base_p95_expected_utility64_v2(
    *, p_admit: float, q_perc: float, latency_p95_ms: float
) -> float:
    """Execute the frozen binary64 base formula in its pinned order."""

    p = _require_float(float(p_admit), "p_admit")
    quality = _require_float(float(q_perc), "q_perc")
    latency = _require_float(float(latency_p95_ms), "latency_p95_ms")
    if not 0.0 <= p <= 1.0:
        raise ExactP95Run2ReplayV2Error("p_admit must lie in [0,1]")
    if not 0.0 <= quality <= 1.0:
        raise ExactP95Run2ReplayV2Error("q_perc must lie in [0,1]")
    if latency < 0.0:
        raise ExactP95Run2ReplayV2Error(
            "latency_p95_ms must be non-negative"
        )
    return p * (quality - 0.25 * (latency / 200.0)) + (1.0 - p) * (-1.0)


def _shaped_p95_expected_utility64_v2(
    *,
    p_admit: float,
    q_perc: float,
    latency_p95_ms: float,
    deadline_penalty: float,
) -> float:
    """Apply the frozen conditional subtraction without analysis imports."""

    penalty = _require_float(float(deadline_penalty), "deadline_penalty")
    if penalty < 0.0:
        raise ExactP95Run2ReplayV2Error(
            "deadline_penalty must be non-negative"
        )
    base64 = _base_p95_expected_utility64_v2(
        p_admit=p_admit,
        q_perc=q_perc,
        latency_p95_ms=latency_p95_ms,
    )
    if float(latency_p95_ms) > RUN2_V2_DEADLINE_MS:
        return base64 - float(p_admit) * penalty
    return base64


def exact_p95_run2_reward_spec_document_v2() -> Dict[str, Any]:
    return {
        "base64_formula_in_pinned_order": (
            "p*(Q-0.25*(L95/200.0))+(1.0-p)*(-1.0)"
        ),
        "deadline_ms": RUN2_V2_DEADLINE_MS,
        "deadline_penalty": RUN2_V2_DEADLINE_PENALTY,
        "deadline_penalty_float_hex": RUN2_V2_DEADLINE_PENALTY_HEX,
        "deadline_penalty_uint64_hex": RUN2_V2_DEADLINE_PENALTY_UINT64_HEX,
        "emitted_target": (
            "torch.tensor(shaped64,dtype=torch.float32,device='cpu').item()"
        ),
        "infeasible_shaped64_formula_in_pinned_order": "base64-p*lambda",
        "intermediate_dtype": "PYTHON_BINARY64",
        "p_zero_semantics": (
            "EMITTED_TARGET_EXACTLY_MINUS_ONE_AND_EXCLUDED_FROM_"
            "CONDITIONAL_SURVIVOR_COMPETITION"
        ),
        "penalty_placement": "INSIDE_ADMITTED_BRANCH",
        "record": RUN2_V2_REWARD_SCHEMA,
        "runtime_arithmetic": RUN2_V2_RUNTIME_ARITHMETIC,
        "source_d1_reward": "PRESERVED_BINARY64_SEPARATELY_NOT_RELABELED",
        "terminal_discount": RUN2_V2_TERMINAL_DISCOUNT,
        "terminal_target": "BIT_IDENTICAL_TO_EMITTED_CPU_FLOAT32_REWARD",
        "train_penalty_spec_sha256": (
            REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256
        ),
        "train_summary_sha256": TRAIN_V2_SUMMARY_SHA256,
        "preregistration_sha256": PREREG_V2_FILE_SHA256,
        "validation_gate_summary_sha256": GATE_V2_SUMMARY_SHA256,
    }


EXACT_P95_RUN2_REWARD_SPEC_V2_SHA256 = (
    "58fcfd34f4acdeb2b087a09239735c42eb11ad70450aec54bbd4149245964a38"
)


def _read_json(path: Path) -> Dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if type(document) is not dict:
        raise ExactP95Run2BindingV2Error(f"{path.name} is not a JSON object")
    return document


def _require_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or _sha256_file(path) != expected:
        raise ExactP95Run2BindingV2Error(f"{label} artifact drift")


def _require_registered_v2_evidence(project_root: Optional[Path]) -> None:
    root = _project_root() if project_root is None else Path(project_root).resolve(strict=True)
    train = root / TRAIN_V2_RELATIVE_DIRECTORY
    prereg = root / PREREG_V2_RELATIVE_DIRECTORY
    gate = root / GATE_V2_RELATIVE_DIRECTORY
    files = (
        (train / "summary_v2.json", TRAIN_V2_SUMMARY_SHA256, "train summary"),
        (train / "selection_decision_v2.json", TRAIN_V2_DECISION_SHA256, "train decision"),
        (train / "REPORT_v2.md", TRAIN_V2_REPORT_SHA256, "train report"),
        (train / "train_context_oracles_v2.csv", TRAIN_V2_ORACLES_SHA256, "train oracles"),
        (prereg / "preregistration_v2.json", PREREG_V2_FILE_SHA256, "preregistration"),
        (prereg / "REPORT_v2.md", PREREG_V2_REPORT_SHA256, "preregistration report"),
        (prereg / "sha256_v2.txt", PREREG_V2_HASH_MANIFEST_SHA256, "preregistration hash manifest"),
        (gate / "summary_v2.json", GATE_V2_SUMMARY_SHA256, "validation summary"),
        (gate / "GO_NO_GO_v2.json", GATE_V2_DECISION_SHA256, "validation decision"),
        (gate / "REPORT_v2.md", GATE_V2_REPORT_SHA256, "validation report"),
        (gate / "profile_summary_v2.csv", GATE_V2_PROFILE_SUMMARY_SHA256, "validation profile summary"),
        (gate / "validation_context_oracles_v2.csv", GATE_V2_CONTEXT_ORACLES_SHA256, "validation oracles"),
    )
    for path, digest, label in files:
        _require_file(path, digest, label)

    train_summary = _read_json(train / "summary_v2.json")
    train_decision = _read_json(train / "selection_decision_v2.json")
    preregistration = _read_json(prereg / "preregistration_v2.json")
    gate_summary = _read_json(gate / "summary_v2.json")
    gate_decision = _read_json(gate / "GO_NO_GO_v2.json")

    train_criteria = train_summary.get("decision", {}).get("criteria", {})
    if (
        train_summary.get("schema")
        != "splitfusion.train_exact_p95_deadline_penalty.v2"
        or train_summary.get("status")
        != "COMPLETE_TRAIN_ONLY_EMITTED_FLOAT32_EXACT_P95_PENALTY_V2"
        or train_summary.get("canonical_content_sha256")
        != TRAIN_V2_CANONICAL_CONTENT_SHA256
        or train_summary.get("decision", {}).get("status") != "GO"
        or not train_criteria
        or not all(value is True for value in train_criteria.values())
        or train_summary.get("decision", {}).get("deadline_penalty")
        != RUN2_V2_DEADLINE_PENALTY
        or train_summary.get("decision", {}).get("deadline_penalty_float_hex")
        != RUN2_V2_DEADLINE_PENALTY_HEX
        or train_summary.get("decision", {}).get("deadline_penalty_uint64_hex")
        != RUN2_V2_DEADLINE_PENALTY_UINT64_HEX
        or train_summary.get("decision", {}).get("strict_ordering_violation_count") != 0
        or train_summary.get("decision", {}).get("winner_identity_match_count") != 1564
        or train_summary.get("emitted_float32_constrained_oracle", {}).get("p95_miss_count") != 0
        or train_summary.get("bindings", {}).get("implementation_sha256")
        != TRAIN_V2_IMPLEMENTATION_SHA256
        or train_summary.get("bindings", {}).get("float32_exact_penalty_spec_sha256")
        != REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256
        or train_summary.get("files", {}).get("selection_decision_v2.json")
        != TRAIN_V2_DECISION_SHA256
        or train_summary.get("files", {}).get("REPORT_v2.md")
        != TRAIN_V2_REPORT_SHA256
        or train_summary.get("files", {}).get("train_context_oracles_v2.csv")
        != TRAIN_V2_ORACLES_SHA256
        or train_decision.get("status") != "GO"
        or train_decision.get("deadline_penalty") != RUN2_V2_DEADLINE_PENALTY
        or not all(
            value is True
            for value in train_decision.get("criteria", {}).values()
        )
    ):
        raise ExactP95Run2BindingV2Error("train v2 semantic binding drift")

    reward = preregistration.get("reward", {})
    prereg_bindings = preregistration.get("bindings", {})
    if (
        preregistration.get("schema")
        != "splitfusion.hybrid_sac.exact_p95_run2_preregistration.v2"
        or preregistration.get("status")
        != "FROZEN_AFTER_TRAIN_ONLY_DERIVATION_BEFORE_ANY_V2_VALIDATION_ACCESS"
        or reward.get("deadline_penalty") != RUN2_V2_DEADLINE_PENALTY
        or reward.get("deadline_penalty_float_hex")
        != RUN2_V2_DEADLINE_PENALTY_HEX
        or reward.get("deadline_penalty_uint64_hex")
        != RUN2_V2_DEADLINE_PENALTY_UINT64_HEX
        or reward.get("emitted_target_formula")
        != "torch.tensor(shaped64,dtype=torch.float32).item()"
        or preregistration.get("train_only_derivation", {}).get("decision") != "GO"
        or preregistration.get("train_only_derivation", {}).get("strict_ordering_violation_count") != 0
        or preregistration.get("train_only_derivation", {}).get("p95_miss_count") != 0
        or prereg_bindings.get("float32_exact_penalty_spec_sha256")
        != REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256
        or prereg_bindings.get("exact_penalty_v2_summary_sha256")
        != TRAIN_V2_SUMMARY_SHA256
        or prereg_bindings.get("exact_penalty_v2_selection_decision_sha256")
        != TRAIN_V2_DECISION_SHA256
        or prereg_bindings.get("exact_penalty_v2_train_context_oracles_sha256")
        != TRAIN_V2_ORACLES_SHA256
    ):
        raise ExactP95Run2BindingV2Error("preregistration v2 semantic binding drift")

    gate_criteria = gate_summary.get("decision", {}).get("criteria", {})
    gate_bindings = gate_summary.get("bindings", {})
    gate_metrics = gate_summary.get("validation_gate", {})
    if (
        gate_summary.get("schema")
        != "splitfusion.exact_p95_run2_validation_gate.v2"
        or gate_summary.get("status")
        != "COMPLETE_ONE_SHOT_EXACT_P95_RUN2_VALIDATION_GATE_V2"
        or gate_summary.get("canonical_content_sha256")
        != GATE_V2_CANONICAL_CONTENT_SHA256
        or gate_summary.get("decision", {}).get("status") != "GO"
        or gate_decision != gate_summary.get("decision")
        or not gate_criteria
        or not all(value is True for value in gate_criteria.values())
        or gate_summary.get("reward", {}).get("deadline_penalty")
        != RUN2_V2_DEADLINE_PENALTY
        or gate_summary.get("reward", {}).get("deadline_penalty_float_hex")
        != RUN2_V2_DEADLINE_PENALTY_HEX
        or gate_summary.get("reward", {}).get("arithmetic_contract")
        != (
            "PYTHON_BINARY64_BASE_THEN_CONDITIONAL_BINARY64_SUBTRACTION_THEN_"
            "ONE_TORCH_FLOAT32_SCALAR_EMISSION"
        )
        or gate_metrics.get("context_count") != 340
        or gate_metrics.get("shaped_constrained_identity_match_count") != 340
        or gate_metrics.get("shaped_constrained_identity_mismatch_count") != 0
        or gate_metrics.get("shaped_oracle_p95_miss_count") != 0
        or gate_metrics.get("strict_infeasible_ordering_violation_count") != 0
        or gate_bindings.get("run2_v2_preregistration_sha256")
        != PREREG_V2_FILE_SHA256
        or gate_bindings.get("train_exact_penalty_v2_summary_sha256")
        != TRAIN_V2_SUMMARY_SHA256
        or gate_bindings.get("train_exact_penalty_v2_decision_sha256")
        != TRAIN_V2_DECISION_SHA256
        or gate_bindings.get("float32_exact_penalty_spec_sha256")
        != REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256
        or gate_bindings.get("validation_implementation_sha256")
        != GATE_V2_IMPLEMENTATION_SHA256
        or gate_bindings.get("validation_test_sha256") != GATE_V2_TEST_SHA256
        or gate_summary.get("files", {}).get("GO_NO_GO_v2.json")
        != GATE_V2_DECISION_SHA256
        or gate_summary.get("files", {}).get("REPORT_v2.md")
        != GATE_V2_REPORT_SHA256
        or gate_summary.get("files", {}).get("profile_summary_v2.csv")
        != GATE_V2_PROFILE_SUMMARY_SHA256
        or gate_summary.get("files", {}).get("validation_context_oracles_v2.csv")
        != GATE_V2_CONTEXT_ORACLES_SHA256
    ):
        raise ExactP95Run2BindingV2Error("validation gate v2 semantic binding drift")


@dataclass(frozen=True, slots=True)
class ExactP95Run2RewardBindingV2:
    source_d1_binding_sha256: str
    source_d1_utility_spec_sha256: str
    shaped_reward_spec_sha256: str
    float32_exact_penalty_spec_sha256: str
    deadline_ms: float
    deadline_penalty: float
    deadline_penalty_float_hex: str
    deadline_penalty_uint64_hex: str
    terminal_discount: float
    train_summary_file_sha256: str
    train_decision_file_sha256: str
    train_report_file_sha256: str
    train_context_oracles_file_sha256: str
    train_implementation_sha256: str
    train_canonical_content_sha256: str
    preregistration_file_sha256: str
    preregistration_report_sha256: str
    preregistration_hash_manifest_sha256: str
    validation_summary_file_sha256: str
    validation_decision_file_sha256: str
    validation_report_file_sha256: str
    validation_profile_summary_file_sha256: str
    validation_context_oracles_file_sha256: str
    validation_implementation_sha256: str
    validation_test_sha256: str
    validation_canonical_content_sha256: str
    schema: str = RUN2_V2_REWARD_SCHEMA

    def __post_init__(self) -> None:
        self.require_valid()

    @classmethod
    def from_validated_d1(
        cls,
        transition: EmpiricalTerminalTransitionV1,
        *,
        project_root: Optional[Path] = None,
    ) -> "ExactP95Run2RewardBindingV2":
        if type(transition) is not EmpiricalTerminalTransitionV1:
            raise ExactP95Run2BindingV2Error(
                "v2 reward binding requires an exact D1 transition"
            )
        transition.revalidate()
        _require_registered_v2_evidence(project_root)
        result = cls(
            source_d1_binding_sha256=transition.d1_binding.canonical_sha256(),
            source_d1_utility_spec_sha256=transition.d1_binding.utility_spec_sha256,
            shaped_reward_spec_sha256=EXACT_P95_RUN2_REWARD_SPEC_V2_SHA256,
            float32_exact_penalty_spec_sha256=(
                REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256
            ),
            deadline_ms=RUN2_V2_DEADLINE_MS,
            deadline_penalty=RUN2_V2_DEADLINE_PENALTY,
            deadline_penalty_float_hex=RUN2_V2_DEADLINE_PENALTY_HEX,
            deadline_penalty_uint64_hex=RUN2_V2_DEADLINE_PENALTY_UINT64_HEX,
            terminal_discount=RUN2_V2_TERMINAL_DISCOUNT,
            train_summary_file_sha256=TRAIN_V2_SUMMARY_SHA256,
            train_decision_file_sha256=TRAIN_V2_DECISION_SHA256,
            train_report_file_sha256=TRAIN_V2_REPORT_SHA256,
            train_context_oracles_file_sha256=TRAIN_V2_ORACLES_SHA256,
            train_implementation_sha256=TRAIN_V2_IMPLEMENTATION_SHA256,
            train_canonical_content_sha256=TRAIN_V2_CANONICAL_CONTENT_SHA256,
            preregistration_file_sha256=PREREG_V2_FILE_SHA256,
            preregistration_report_sha256=PREREG_V2_REPORT_SHA256,
            preregistration_hash_manifest_sha256=PREREG_V2_HASH_MANIFEST_SHA256,
            validation_summary_file_sha256=GATE_V2_SUMMARY_SHA256,
            validation_decision_file_sha256=GATE_V2_DECISION_SHA256,
            validation_report_file_sha256=GATE_V2_REPORT_SHA256,
            validation_profile_summary_file_sha256=(
                GATE_V2_PROFILE_SUMMARY_SHA256
            ),
            validation_context_oracles_file_sha256=(
                GATE_V2_CONTEXT_ORACLES_SHA256
            ),
            validation_implementation_sha256=GATE_V2_IMPLEMENTATION_SHA256,
            validation_test_sha256=GATE_V2_TEST_SHA256,
            validation_canonical_content_sha256=(
                GATE_V2_CANONICAL_CONTENT_SHA256
            ),
        )
        result.require_valid()
        return result

    def require_valid(self) -> None:
        expected = {
            "source_d1_utility_spec_sha256": PILOT_UTILITY_SPEC_SHA256,
            "shaped_reward_spec_sha256": EXACT_P95_RUN2_REWARD_SPEC_V2_SHA256,
            "float32_exact_penalty_spec_sha256": REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256,
            "train_summary_file_sha256": TRAIN_V2_SUMMARY_SHA256,
            "train_decision_file_sha256": TRAIN_V2_DECISION_SHA256,
            "train_report_file_sha256": TRAIN_V2_REPORT_SHA256,
            "train_context_oracles_file_sha256": TRAIN_V2_ORACLES_SHA256,
            "train_implementation_sha256": TRAIN_V2_IMPLEMENTATION_SHA256,
            "train_canonical_content_sha256": TRAIN_V2_CANONICAL_CONTENT_SHA256,
            "preregistration_file_sha256": PREREG_V2_FILE_SHA256,
            "preregistration_report_sha256": PREREG_V2_REPORT_SHA256,
            "preregistration_hash_manifest_sha256": PREREG_V2_HASH_MANIFEST_SHA256,
            "validation_summary_file_sha256": GATE_V2_SUMMARY_SHA256,
            "validation_decision_file_sha256": GATE_V2_DECISION_SHA256,
            "validation_report_file_sha256": GATE_V2_REPORT_SHA256,
            "validation_profile_summary_file_sha256": GATE_V2_PROFILE_SUMMARY_SHA256,
            "validation_context_oracles_file_sha256": GATE_V2_CONTEXT_ORACLES_SHA256,
            "validation_implementation_sha256": GATE_V2_IMPLEMENTATION_SHA256,
            "validation_test_sha256": GATE_V2_TEST_SHA256,
            "validation_canonical_content_sha256": GATE_V2_CANONICAL_CONTENT_SHA256,
        }
        if any(getattr(self, key) != value for key, value in expected.items()):
            raise ExactP95Run2BindingV2Error("v2 reward evidence binding drift")
        if (
            canonical_sha256(exact_p95_run2_reward_spec_document_v2())
            != EXACT_P95_RUN2_REWARD_SPEC_V2_SHA256
            or self.shaped_reward_spec_sha256
            == self.source_d1_utility_spec_sha256
            or type(self.deadline_ms) is not float
            or self.deadline_ms != RUN2_V2_DEADLINE_MS
            or type(self.deadline_penalty) is not float
            or self.deadline_penalty != RUN2_V2_DEADLINE_PENALTY
            or self.deadline_penalty.hex() != RUN2_V2_DEADLINE_PENALTY_HEX
            or _float64_bits_hex(self.deadline_penalty)
            != RUN2_V2_DEADLINE_PENALTY_UINT64_HEX
            or self.deadline_penalty_float_hex != RUN2_V2_DEADLINE_PENALTY_HEX
            or self.deadline_penalty_uint64_hex
            != RUN2_V2_DEADLINE_PENALTY_UINT64_HEX
            or type(self.terminal_discount) is not float
            or self.terminal_discount != RUN2_V2_TERMINAL_DISCOUNT
            or self.schema != RUN2_V2_REWARD_SCHEMA
        ):
            raise ExactP95Run2BindingV2Error("v2 reward scalar binding drift")
        hashes = (
            self.source_d1_binding_sha256,
            self.source_d1_utility_spec_sha256,
            self.shaped_reward_spec_sha256,
            self.float32_exact_penalty_spec_sha256,
            self.train_summary_file_sha256,
            self.train_decision_file_sha256,
            self.train_report_file_sha256,
            self.train_context_oracles_file_sha256,
            self.train_implementation_sha256,
            self.train_canonical_content_sha256,
            self.preregistration_file_sha256,
            self.preregistration_report_sha256,
            self.preregistration_hash_manifest_sha256,
            self.validation_summary_file_sha256,
            self.validation_decision_file_sha256,
            self.validation_report_file_sha256,
            self.validation_profile_summary_file_sha256,
            self.validation_context_oracles_file_sha256,
            self.validation_implementation_sha256,
            self.validation_test_sha256,
            self.validation_canonical_content_sha256,
        )
        if any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ExactP95Run2BindingV2Error("v2 reward binding hash malformed")

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.require_valid()
        return asdict(self)

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


def _derive_reward_components(
    source: EmpiricalTerminalTransitionV1,
) -> Tuple[float, float, float, float]:
    source.revalidate()
    policy = source.result.policy
    source_reward64 = _require_float(source.reward, "source D1 reward")
    base64 = _base_p95_expected_utility64_v2(
        p_admit=float(policy.p_edge_admission_given_sent),
        q_perc=float(policy.q_perc),
        latency_p95_ms=float(policy.latency_proxy_p95_ms),
    )
    shaped64 = _shaped_p95_expected_utility64_v2(
        p_admit=float(policy.p_edge_admission_given_sent),
        q_perc=float(policy.q_perc),
        latency_p95_ms=float(policy.latency_proxy_p95_ms),
        deadline_penalty=RUN2_V2_DEADLINE_PENALTY,
    )
    emitted32 = _emit_cpu_float32(shaped64)
    return source_reward64, base64, shaped64, emitted32


@dataclass(frozen=True, slots=True)
class ExactP95Run2ShapedTransitionV2:
    source_d1_transition: EmpiricalTerminalTransitionV1
    reward_binding: ExactP95Run2RewardBindingV2
    source_d1_reward64: float
    p95_base_reward64: float
    shaped_reward64: float
    emitted_reward_float32: float
    terminal_discount: float
    _attestation_sha256: str = field(repr=False)

    def __post_init__(self) -> None:
        self.revalidate()

    @classmethod
    def from_validated_d1(
        cls,
        transition: EmpiricalTerminalTransitionV1,
        *,
        project_root: Optional[Path] = None,
        reward_binding: Optional[ExactP95Run2RewardBindingV2] = None,
    ) -> "ExactP95Run2ShapedTransitionV2":
        if type(transition) is not EmpiricalTerminalTransitionV1:
            raise ExactP95Run2ReplayV2Error("v2 adapter requires an exact D1 transition")
        transition.revalidate()
        if reward_binding is None:
            binding = ExactP95Run2RewardBindingV2.from_validated_d1(
                transition, project_root=project_root
            )
        else:
            if type(reward_binding) is not ExactP95Run2RewardBindingV2:
                raise ExactP95Run2BindingV2Error("reused v2 reward binding has a foreign type")
            reward_binding.require_valid()
            if reward_binding.source_d1_binding_sha256 != transition.d1_binding.canonical_sha256():
                raise ExactP95Run2BindingV2Error("reused v2 reward binding differs from D1")
            binding = reward_binding
        source_reward64, base64, shaped64, emitted32 = _derive_reward_components(
            transition
        )
        document = _transition_document_v2(
            source=transition,
            reward_binding=binding,
            source_reward64=source_reward64,
            base64=base64,
            shaped64=shaped64,
            emitted32=emitted32,
            terminal_discount=RUN2_V2_TERMINAL_DISCOUNT,
        )
        result = cls(
            source_d1_transition=transition,
            reward_binding=binding,
            source_d1_reward64=source_reward64,
            p95_base_reward64=base64,
            shaped_reward64=shaped64,
            emitted_reward_float32=emitted32,
            terminal_discount=RUN2_V2_TERMINAL_DISCOUNT,
            _attestation_sha256=canonical_sha256(document),
        )
        result.revalidate()
        return result

    @property
    def logical_key(self) -> Tuple[str, int]:
        return self.source_d1_transition.logical_key

    @property
    def reward(self) -> float:
        return self.emitted_reward_float32

    def _document(self) -> Dict[str, Any]:
        return _transition_document_v2(
            source=self.source_d1_transition,
            reward_binding=self.reward_binding,
            source_reward64=self.source_d1_reward64,
            base64=self.p95_base_reward64,
            shaped64=self.shaped_reward64,
            emitted32=self.emitted_reward_float32,
            terminal_discount=self.terminal_discount,
        )

    def canonical_sha256(self) -> str:
        return canonical_sha256(self._document())

    def revalidate(self) -> None:
        if type(self) is not ExactP95Run2ShapedTransitionV2:
            raise ExactP95Run2ReplayV2Error("v2 shaped transition has a foreign type")
        if type(self.source_d1_transition) is not EmpiricalTerminalTransitionV1:
            raise ExactP95Run2ReplayV2Error("v2 source transition has a foreign type")
        if type(self.reward_binding) is not ExactP95Run2RewardBindingV2:
            raise ExactP95Run2BindingV2Error("v2 reward binding has a foreign type")
        self.source_d1_transition.revalidate()
        self.reward_binding.require_valid()
        values = (
            self.source_d1_reward64,
            self.p95_base_reward64,
            self.shaped_reward64,
            self.emitted_reward_float32,
            self.terminal_discount,
        )
        if any(type(value) is not float or not math.isfinite(value) for value in values):
            raise ExactP95Run2ReplayV2Error("v2 transition scalar drift")
        if self.reward_binding.source_d1_binding_sha256 != self.source_d1_transition.d1_binding.canonical_sha256():
            raise ExactP95Run2BindingV2Error("v2 reward/source D1 binding mismatch")
        expected = _derive_reward_components(self.source_d1_transition)
        if (
            values[:4] != expected
            or self.terminal_discount != RUN2_V2_TERMINAL_DISCOUNT
            or _float32_roundtrip(self.emitted_reward_float32)
            != self.emitted_reward_float32
            or self._attestation_sha256 != self.canonical_sha256()
        ):
            raise ExactP95Run2ReplayV2Error("v2 shaped transition attestation drift")


def _transition_document_v2(
    *,
    source: EmpiricalTerminalTransitionV1,
    reward_binding: ExactP95Run2RewardBindingV2,
    source_reward64: float,
    base64: float,
    shaped64: float,
    emitted32: float,
    terminal_discount: float,
) -> Dict[str, Any]:
    return {
        "emitted_reward_float32": emitted32,
        "emitted_reward_float32_bits_hex": _float32_bits_hex(emitted32),
        "p95_base_reward64": base64,
        "p95_base_reward64_bits_hex": _float64_bits_hex(base64),
        "record": RUN2_V2_TRANSITION_SCHEMA,
        "reward_binding_sha256": reward_binding.canonical_sha256(),
        "shaped_reward64": shaped64,
        "shaped_reward64_bits_hex": _float64_bits_hex(shaped64),
        "source_d1_reward64": source_reward64,
        "source_d1_reward64_bits_hex": _float64_bits_hex(source_reward64),
        "source_d1_transition_sha256": source.canonical_sha256(),
        "terminal_discount": terminal_discount,
    }


@dataclass(frozen=True, slots=True)
class ExactP95Run2ReplayBindingV2:
    source_terminal_binding: EmpiricalTerminalBindingV1
    reward_binding: ExactP95Run2RewardBindingV2
    policy_feature_order: Tuple[str, ...]
    policy_feature_count: int
    float_dtype: str
    diagnostic_float_dtype: str
    terminal_discount: float
    schema: str = RUN2_V2_REPLAY_SCHEMA

    def __post_init__(self) -> None:
        self.require_valid()

    @classmethod
    def from_transition(
        cls, transition: ExactP95Run2ShapedTransitionV2
    ) -> "ExactP95Run2ReplayBindingV2":
        if type(transition) is not ExactP95Run2ShapedTransitionV2:
            raise ExactP95Run2BindingV2Error("v2 replay binding requires exact transition")
        transition.revalidate()
        result = cls(
            source_terminal_binding=EmpiricalTerminalBindingV1.from_transition(
                transition.source_d1_transition
            ),
            reward_binding=transition.reward_binding,
            policy_feature_order=tuple(POLICY_FEATURE_ORDER),
            policy_feature_count=POLICY_FEATURE_COUNT,
            float_dtype=str(torch.float32),
            diagnostic_float_dtype=str(torch.float64),
            terminal_discount=RUN2_V2_TERMINAL_DISCOUNT,
        )
        result.require_valid()
        return result

    def require_valid(self) -> None:
        if type(self.source_terminal_binding) is not EmpiricalTerminalBindingV1:
            raise ExactP95Run2BindingV2Error("v2 source replay binding has foreign type")
        if type(self.reward_binding) is not ExactP95Run2RewardBindingV2:
            raise ExactP95Run2BindingV2Error("v2 reward binding has foreign type")
        self.source_terminal_binding.__post_init__()
        self.reward_binding.require_valid()
        if (
            self.source_terminal_binding.d1_binding.canonical_sha256()
            != self.reward_binding.source_d1_binding_sha256
            or self.policy_feature_order != tuple(POLICY_FEATURE_ORDER)
            or self.policy_feature_count != POLICY_FEATURE_COUNT
            or self.float_dtype != str(torch.float32)
            or self.diagnostic_float_dtype != str(torch.float64)
            or self.terminal_discount != RUN2_V2_TERMINAL_DISCOUNT
            or self.schema != RUN2_V2_REPLAY_SCHEMA
        ):
            raise ExactP95Run2BindingV2Error("v2 replay binding drift")

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.require_valid()
        return {
            "diagnostic_float_dtype": self.diagnostic_float_dtype,
            "float_dtype": self.float_dtype,
            "policy_feature_count": self.policy_feature_count,
            "policy_feature_order": list(self.policy_feature_order),
            "reward_binding": self.reward_binding.to_canonical_dict(),
            "schema": self.schema,
            "source_terminal_binding": self.source_terminal_binding.to_canonical_dict(),
            "terminal_discount": self.terminal_discount,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())

    def assert_matches(self, other: object) -> None:
        if type(other) is not ExactP95Run2ReplayBindingV2 or other != self:
            raise ExactP95Run2BindingV2Error("v2 replay binding mismatch")


_BATCH_SOURCE_KEY = "_authenticated_run2_v2_transition"
_BATCH_ATTESTATION_KEY = "row_attestation_sha256"


def _batch_audit_payload(
    transition: ExactP95Run2ShapedTransitionV2,
) -> Dict[str, Any]:
    if type(transition) is not ExactP95Run2ShapedTransitionV2:
        raise ExactP95Run2ReplayV2Error("v2 batch audit source has foreign type")
    transition.revalidate()
    source = transition.source_d1_transition
    return {
        "collection_seq": source.collection_seq,
        "collection_session_uuid": source.collection_session_uuid,
        "emitted_reward_float32": transition.emitted_reward_float32,
        "emitted_reward_float32_bits_hex": _float32_bits_hex(
            transition.emitted_reward_float32
        ),
        "mode_id": source.action.mode_id,
        "p95_base_reward64": transition.p95_base_reward64,
        "p95_base_reward64_bits_hex": _float64_bits_hex(
            transition.p95_base_reward64
        ),
        "q_e4": source.action.q_e4,
        "record": RUN2_V2_BATCH_AUDIT_SCHEMA,
        "reward_binding_sha256": transition.reward_binding.canonical_sha256(),
        "run2_v2_transition_sha256": transition.canonical_sha256(),
        "shaped_reward64": transition.shaped_reward64,
        "shaped_reward64_bits_hex": _float64_bits_hex(
            transition.shaped_reward64
        ),
        "source_d1_reward64": transition.source_d1_reward64,
        "source_d1_reward64_bits_hex": _float64_bits_hex(
            transition.source_d1_reward64
        ),
        "source_d1_transition_sha256": source.canonical_sha256(),
        "state_float32": [
            float(value)
            for value in torch.tensor(
                source.observation.values, dtype=torch.float32, device="cpu"
            ).tolist()
        ],
        "terminal_discount": transition.terminal_discount,
        "terminal_discount_float32_bits_hex": _float32_bits_hex(
            transition.terminal_discount
        ),
    }


def _batch_audit_row(
    transition: ExactP95Run2ShapedTransitionV2,
) -> Dict[str, Any]:
    payload = _batch_audit_payload(transition)
    return {
        **payload,
        _BATCH_ATTESTATION_KEY: canonical_sha256(payload),
        _BATCH_SOURCE_KEY: transition,
    }


@dataclass(frozen=True, slots=True, eq=False)
class ExactP95Run2BatchV2:
    _state: Tensor
    _mode_id: Tensor
    _q_e4: Tensor
    _source_d1_reward64: Tensor
    _p95_base_reward64: Tensor
    _shaped_reward64: Tensor
    _reward: Tensor
    _discount: Tensor
    binding: ExactP95Run2ReplayBindingV2
    audit: Tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if type(self.binding) is not ExactP95Run2ReplayBindingV2:
            raise ExactP95Run2BindingV2Error("v2 batch binding has foreign type")
        self.binding.require_valid()
        tensor_names = (
            "_state", "_mode_id", "_q_e4", "_source_d1_reward64",
            "_p95_base_reward64", "_shaped_reward64", "_reward", "_discount",
        )
        for name in tensor_names:
            value = getattr(self, name)
            if type(value) is not Tensor:
                raise ExactP95Run2ReplayV2Error(f"{name} must be an exact tensor")
            object.__setattr__(self, name, value.detach().clone())
        size = int(self._state.shape[0]) if self._state.ndim else 0
        if size < 1 or self._state.shape != (size, POLICY_FEATURE_COUNT):
            raise ExactP95Run2ReplayV2Error("v2 state tensor shape drift")
        if self._state.dtype is not torch.float32 or self._state.device.type != "cpu" or not bool(torch.isfinite(self._state).all()):
            raise ExactP95Run2ReplayV2Error("v2 state must be finite CPU float32")
        for name in ("_mode_id", "_q_e4"):
            value = getattr(self, name)
            if value.shape != (size,) or value.dtype is not torch.int64 or value.device.type != "cpu":
                raise ExactP95Run2ReplayV2Error(f"{name} must be [B] CPU int64")
        for name in ("_source_d1_reward64", "_p95_base_reward64", "_shaped_reward64"):
            value = getattr(self, name)
            if value.shape != (size,) or value.dtype is not torch.float64 or value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
                raise ExactP95Run2ReplayV2Error(f"{name} must be finite [B] CPU float64")
        for name in ("_reward", "_discount"):
            value = getattr(self, name)
            if value.shape != (size,) or value.dtype is not torch.float32 or value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
                raise ExactP95Run2ReplayV2Error(f"{name} must be finite [B] CPU float32")
        if not torch.equal(self._discount, torch.zeros_like(self._discount)):
            raise ExactP95Run2ReplayV2Error("v2 discount differs from terminal contract")
        if bool((self._mode_id < 0).any()) or bool((self._mode_id >= EXPECTED_MODE_COUNT).any()):
            raise ExactP95Run2ReplayV2Error("v2 mode IDs escaped the catalog")
        for mode_id, q_e4 in zip(self._mode_id.tolist(), self._q_e4.tolist()):
            try:
                require_supported_action(mode_id, q_e4)
            except ValueError as exc:
                raise ExactP95Run2ReplayV2Error("v2 batch action escaped support") from exc
        if type(self.audit) is not tuple or len(self.audit) != size:
            raise ExactP95Run2ReplayV2Error("v2 batch audit cardinality drift")
        checked = []
        for index, supplied in enumerate(self.audit):
            if not isinstance(supplied, Mapping):
                raise ExactP95Run2ReplayV2Error("v2 batch audit row is not a mapping")
            row = dict(supplied)
            source = row.pop(_BATCH_SOURCE_KEY, None)
            attestation = row.pop(_BATCH_ATTESTATION_KEY, None)
            expected = _batch_audit_payload(source)
            if (
                set(row) != set(expected)
                or row != expected
                or canonical_sha256(row) != attestation
                or canonical_sha256(expected) != attestation
            ):
                raise ExactP95Run2ReplayV2Error("v2 batch audit attestation drift")
            if (
                source.reward_binding != self.binding.reward_binding
                or EmpiricalTerminalBindingV1.from_transition(
                    source.source_d1_transition
                ) != self.binding.source_terminal_binding
            ):
                raise ExactP95Run2BindingV2Error("v2 batch source differs from binding")
            expected_state = torch.tensor(expected["state_float32"], dtype=torch.float32, device="cpu")
            expected64 = (
                (self._source_d1_reward64, "source_d1_reward64"),
                (self._p95_base_reward64, "p95_base_reward64"),
                (self._shaped_reward64, "shaped_reward64"),
            )
            if (
                not torch.equal(self._state[index], expected_state)
                or int(self._mode_id[index]) != expected["mode_id"]
                or int(self._q_e4[index]) != expected["q_e4"]
                or any(
                    not torch.equal(
                        tensor[index:index + 1],
                        torch.tensor([expected[name]], dtype=torch.float64, device="cpu"),
                    )
                    for tensor, name in expected64
                )
                or not torch.equal(
                    self._reward[index:index + 1],
                    torch.tensor([expected["emitted_reward_float32"]], dtype=torch.float32, device="cpu"),
                )
                or _float32_bits_hex(float(self._reward[index]))
                != expected["emitted_reward_float32_bits_hex"]
                or not torch.equal(
                    self._discount[index:index + 1],
                    torch.tensor([expected["terminal_discount"]], dtype=torch.float32, device="cpu"),
                )
            ):
                raise ExactP95Run2ReplayV2Error("v2 batch tensor/source mismatch")
            checked.append(
                MappingProxyType(
                    {
                        **expected,
                        _BATCH_ATTESTATION_KEY: attestation,
                        _BATCH_SOURCE_KEY: source,
                    }
                )
            )
        object.__setattr__(self, "audit", tuple(checked))

    def revalidate(self) -> None:
        if type(self) is not ExactP95Run2BatchV2:
            raise ExactP95Run2ReplayV2Error("v2 batch has a foreign type")
        self.__post_init__()

    @property
    def batch_size(self) -> int:
        return int(self._state.shape[0])

    @property
    def state(self) -> Tensor:
        return self._state.clone()

    @property
    def mode_id(self) -> Tensor:
        return self._mode_id.clone()

    @property
    def q_e4(self) -> Tensor:
        return self._q_e4.clone()

    @property
    def source_d1_reward64(self) -> Tensor:
        return self._source_d1_reward64.clone()

    @property
    def p95_base_reward64(self) -> Tensor:
        return self._p95_base_reward64.clone()

    @property
    def shaped_reward64(self) -> Tensor:
        return self._shaped_reward64.clone()

    @property
    def reward(self) -> Tensor:
        return self._reward.clone()

    def discount(self) -> Tensor:
        return self._discount.clone()

    def terminal_target(self) -> Tensor:
        return self._reward.clone()


class ExactP95Run2ReplayV2:
    """FIFO replay that accepts only exact, attested Run-2-v2 transitions."""

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ExactP95Run2ReplayV2Error("capacity must be an exact positive integer")
        self._capacity = capacity
        self._rows: Deque[ExactP95Run2ShapedTransitionV2] = deque()
        self._binding: Optional[ExactP95Run2ReplayBindingV2] = None
        self._seen_digests: set[str] = set()
        self._seen_keys: Dict[Tuple[str, int], str] = {}
        self._accepted_count = 0
        self._evicted_count = 0

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def binding(self) -> Optional[ExactP95Run2ReplayBindingV2]:
        return self._binding

    @property
    def accepted_count(self) -> int:
        return self._accepted_count

    @property
    def evicted_count(self) -> int:
        return self._evicted_count

    def resident_sources(self) -> Tuple[EmpiricalTerminalTransitionV1, ...]:
        return tuple(row.source_d1_transition for row in self._rows)

    def insert(self, transition: object) -> None:
        if type(transition) is not ExactP95Run2ShapedTransitionV2:
            raise ExactP95Run2ReplayV2Error("v2 replay accepts only exact v2 transitions")
        transition.revalidate()
        candidate_binding = ExactP95Run2ReplayBindingV2.from_transition(transition)
        if self._binding is not None:
            self._binding.assert_matches(candidate_binding)
        digest = transition.canonical_sha256()
        key = transition.logical_key
        known = self._seen_keys.get(key)
        if known is not None and known != digest:
            raise ExactP95Run2ReplayV2Error("v2 collection identity conflict")
        if digest in self._seen_digests:
            raise ExactP95Run2ReplayV2Error("duplicate v2 transition")
        # Prove all execution/diagnostic conversions before any replay mutation.
        state = torch.tensor(
            transition.source_d1_transition.observation.values,
            dtype=torch.float32,
            device="cpu",
        )
        diagnostic = torch.tensor(
            [
                transition.source_d1_reward64,
                transition.p95_base_reward64,
                transition.shaped_reward64,
            ],
            dtype=torch.float64,
            device="cpu",
        )
        execution = torch.tensor(
            [transition.emitted_reward_float32, transition.terminal_discount],
            dtype=torch.float32,
            device="cpu",
        )
        if not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(diagnostic).all()) or not bool(torch.isfinite(execution).all()):
            raise ExactP95Run2ReplayV2Error("v2 row is non-finite after conversion")
        self._binding = candidate_binding if self._binding is None else self._binding
        self._rows.append(transition)
        self._seen_digests.add(digest)
        self._seen_keys[key] = digest
        self._accepted_count += 1
        while len(self._rows) > self._capacity:
            self._rows.popleft()
            self._evicted_count += 1

    @staticmethod
    def _require_generator(generator: object) -> torch.Generator:
        if (
            type(generator) is not torch.Generator
            or generator is torch.default_generator
            or generator.device.type != "cpu"
        ):
            raise ExactP95Run2ReplayV2Error("sample requires a local CPU generator")
        return generator

    def sample(
        self, batch_size: int, generator: torch.Generator
    ) -> ExactP95Run2BatchV2:
        if type(batch_size) is not int or batch_size < 1:
            raise ExactP95Run2ReplayV2Error("batch size must be an exact positive integer")
        self._require_generator(generator)
        if batch_size > len(self._rows) or self._binding is None:
            raise ExactP95Run2ReplayV2Error("cannot sample requested v2 batch")
        indices = torch.randperm(
            len(self._rows), generator=generator, device="cpu"
        )[:batch_size]
        resident = tuple(self._rows)
        rows = tuple(resident[int(index)] for index in indices)
        return ExactP95Run2BatchV2(
            _state=torch.tensor(
                [row.source_d1_transition.observation.values for row in rows],
                dtype=torch.float32,
                device="cpu",
            ),
            _mode_id=torch.tensor(
                [row.source_d1_transition.action.mode_id for row in rows],
                dtype=torch.int64,
                device="cpu",
            ),
            _q_e4=torch.tensor(
                [row.source_d1_transition.action.q_e4 for row in rows],
                dtype=torch.int64,
                device="cpu",
            ),
            _source_d1_reward64=torch.tensor(
                [row.source_d1_reward64 for row in rows],
                dtype=torch.float64,
                device="cpu",
            ),
            _p95_base_reward64=torch.tensor(
                [row.p95_base_reward64 for row in rows],
                dtype=torch.float64,
                device="cpu",
            ),
            _shaped_reward64=torch.tensor(
                [row.shaped_reward64 for row in rows],
                dtype=torch.float64,
                device="cpu",
            ),
            _reward=torch.tensor(
                [row.emitted_reward_float32 for row in rows],
                dtype=torch.float32,
                device="cpu",
            ),
            _discount=torch.tensor(
                [row.terminal_discount for row in rows],
                dtype=torch.float32,
                device="cpu",
            ),
            binding=self._binding,
            audit=tuple(_batch_audit_row(row) for row in rows),
        )
