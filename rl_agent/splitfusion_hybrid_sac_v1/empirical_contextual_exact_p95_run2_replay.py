"""Isolated Phase-1 provenance and replay for the exact-P95 Run 2.

The registered D1 transition remains the immutable source record.  This
module validates that record through its original contract, then attaches a
separate reward binding and a separately attested shaped reward.  It never
relables the D1 P50 reward or its utility-spec hash.

Run 2 is still a terminal contextual problem.  Its emitted discount is the
explicit terminal contract value zero, and a later terminal trainer must use
the shaped reward unchanged as its target.
"""

from __future__ import annotations

import hashlib
import json
import math
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
from .empirical_contextual_exact_p95_deadline_penalty import (
    REGISTERED_EXACT_PENALTY_SPEC_SHA256,
    base_p95_expected_utility,
    shaped_p95_expected_utility,
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
    "EXACT_P95_RUN2_REWARD_SPEC_SHA256",
    "ExactP95Run2BatchV1",
    "ExactP95Run2BindingError",
    "ExactP95Run2ReplayBindingV1",
    "ExactP95Run2ReplayError",
    "ExactP95Run2ReplayV1",
    "ExactP95Run2RewardBindingV1",
    "ExactP95Run2ShapedTransitionV1",
    "RUN2_DEADLINE_MS",
    "RUN2_DEADLINE_PENALTY",
    "RUN2_DEADLINE_PENALTY_HEX",
    "RUN2_PREREGISTRATION_FILE_SHA256",
    "RUN2_TERMINAL_DISCOUNT",
    "exact_p95_run2_reward_spec_document",
]


RUN2_DEADLINE_MS = 200.0
RUN2_DEADLINE_PENALTY = 0.5742957604173842
RUN2_DEADLINE_PENALTY_HEX = "0x1.260a180a72bf6p-1"
RUN2_TERMINAL_DISCOUNT = 0.0
RUN2_SCHEMA = "splitfusion.exact_p95_run2_terminal_replay.v1"
RUN2_REWARD_SCHEMA = "splitfusion.exact_p95_run2_reward.v1"
RUN2_TRANSITION_SCHEMA = "splitfusion.exact_p95_run2_shaped_transition.v1"
PREFLIGHT_RELATIVE_DIRECTORY = (
    "experiments/splitfusion_hybrid_sac_fit_validation_v1/"
    "20260921_train_exact_p95_deadline_penalty_v1"
)
PREFLIGHT_SUMMARY_FILE_SHA256 = (
    "29e9677caad5c2e558da572c784a00ae8a2ef068f5c55191bf30eb8af934cd63"
)
PREFLIGHT_DECISION_FILE_SHA256 = (
    "5caea12b04bf9be7c7c9a05bbfe934910fc71c3ef189587e8a19edae493f5e45"
)
PREFLIGHT_REPORT_FILE_SHA256 = (
    "c2c84e2a05fb15188ac238d75d094fe546954a946f37861b9c4cee5c90cd24e2"
)
PREFLIGHT_CONTEXT_ORACLES_FILE_SHA256 = (
    "2597b4050552cfdcef15396bc811170d458d3a5a9b5d50fae559e4987b654c1f"
)
PREFLIGHT_IMPLEMENTATION_SHA256 = (
    "ed0f53a1b82d877c7ccc48e4388895cf24f3e2ff5e1a6ca18f7e7a04974aad85"
)
PREFLIGHT_CANONICAL_CONTENT_SHA256 = (
    "b1abbf2b38d895222eeba845c18022ee45abaa499f4e704104eb16318513307a"
)
RUN2_PREREGISTRATION_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_fit_validation_v1/"
    "20260921_exact_p95_run2_preregistration_v1/preregistration.json"
)
RUN2_PREREGISTRATION_FILE_SHA256 = (
    "77e6e6b98c4fbf7e03994d73621bdec12ba9853872e1dfdcce17ce15905c1171"
)


class ExactP95Run2ReplayError(ValueError):
    """A Run-2 transition, replay, or batch failed closed."""


class ExactP95Run2BindingError(ExactP95Run2ReplayError):
    """A Run-2 reward or replay identity differs."""


def exact_p95_run2_reward_spec_document() -> Dict[str, Any]:
    return {
        "base_formula": "b=p*(Q-0.25*L95/200)+(1-p)*(-1)",
        "deadline_ms": RUN2_DEADLINE_MS,
        "deadline_penalty": RUN2_DEADLINE_PENALTY,
        "deadline_penalty_float_hex": RUN2_DEADLINE_PENALTY_HEX,
        "penalty_formula": "R=b-p*lambda*I[L95>200]",
        "penalty_placement": "INSIDE_ADMITTED_BRANCH",
        "p_zero_semantics": "REWARD_IS_MINUS_ONE;CONDITIONAL_LATENCY_NOT_USED",
        "preflight_exact_penalty_spec_sha256": (
            REGISTERED_EXACT_PENALTY_SPEC_SHA256
        ),
        "run2_preregistration_file_sha256": RUN2_PREREGISTRATION_FILE_SHA256,
        "record": RUN2_REWARD_SCHEMA,
        "source_d1_reward": "PRESERVED_SEPARATELY_NOT_RELABELLED",
        "terminal_discount": RUN2_TERMINAL_DISCOUNT,
        "terminal_target": "BIT_IDENTICAL_TO_SHAPED_REWARD_AFTER_DTYPE_CONVERSION",
    }


# This is deliberately distinct from the frozen D1 P50 utility-spec hash.
EXACT_P95_RUN2_REWARD_SPEC_SHA256 = (
    "92aecdca353fbdb759121e6401557c5a99ba3ddcffd081216c1dca22214b0b92"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_registered_preflight(project_root: Optional[Path]) -> None:
    root = _project_root() if project_root is None else Path(project_root).resolve(strict=True)
    directory = root / PREFLIGHT_RELATIVE_DIRECTORY
    expected = {
        "summary.json": PREFLIGHT_SUMMARY_FILE_SHA256,
        "selection_decision.json": PREFLIGHT_DECISION_FILE_SHA256,
        "REPORT.md": PREFLIGHT_REPORT_FILE_SHA256,
        "train_context_oracles.csv": PREFLIGHT_CONTEXT_ORACLES_FILE_SHA256,
    }
    for name, digest in expected.items():
        path = directory / name
        if not path.is_file() or _sha256_file(path) != digest:
            raise ExactP95Run2BindingError(f"exact-P95 preflight artifact drift: {name}")
    preregistration_path = root / RUN2_PREREGISTRATION_RELATIVE_PATH
    if (
        not preregistration_path.is_file()
        or _sha256_file(preregistration_path)
        != RUN2_PREREGISTRATION_FILE_SHA256
    ):
        raise ExactP95Run2BindingError("Run-2 preregistration artifact drift")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    decision = json.loads(
        (directory / "selection_decision.json").read_text(encoding="utf-8")
    )
    if (
        summary.get("canonical_content_sha256")
        != PREFLIGHT_CANONICAL_CONTENT_SHA256
        or summary.get("bindings", {}).get("implementation_sha256")
        != PREFLIGHT_IMPLEMENTATION_SHA256
        or summary.get("bindings", {}).get("exact_penalty_spec_sha256")
        != REGISTERED_EXACT_PENALTY_SPEC_SHA256
        or summary.get("decision", {}).get("status") != "GO"
        or summary.get("decision", {}).get("deadline_penalty")
        != RUN2_DEADLINE_PENALTY
        or not all(summary.get("decision", {}).get("criteria", {}).values())
        or decision.get("status") != "GO"
        or decision.get("deadline_penalty") != RUN2_DEADLINE_PENALTY
    ):
        raise ExactP95Run2BindingError("exact-P95 preflight semantic binding drift")


def _require_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ExactP95Run2ReplayError(f"{name} must be an exact finite float")
    return value


@dataclass(frozen=True, slots=True)
class ExactP95Run2RewardBindingV1:
    """Distinct shaped-reward identity anchored to one validated D1 binding."""

    source_d1_binding_sha256: str
    source_d1_utility_spec_sha256: str
    shaped_reward_spec_sha256: str
    deadline_ms: float
    deadline_penalty: float
    deadline_penalty_float_hex: str
    terminal_discount: float
    preflight_summary_file_sha256: str
    preflight_decision_file_sha256: str
    preflight_report_file_sha256: str
    preflight_context_oracles_file_sha256: str
    preflight_implementation_sha256: str
    preflight_canonical_content_sha256: str
    run2_preregistration_file_sha256: str
    schema: str = RUN2_REWARD_SCHEMA

    def __post_init__(self) -> None:
        self.require_valid()

    @classmethod
    def from_validated_d1(
        cls,
        transition: EmpiricalTerminalTransitionV1,
        *,
        project_root: Optional[Path] = None,
    ) -> "ExactP95Run2RewardBindingV1":
        if type(transition) is not EmpiricalTerminalTransitionV1:
            raise ExactP95Run2BindingError("reward binding requires an exact D1 transition")
        transition.revalidate()
        _require_registered_preflight(project_root)
        binding = cls(
            source_d1_binding_sha256=transition.d1_binding.canonical_sha256(),
            source_d1_utility_spec_sha256=transition.d1_binding.utility_spec_sha256,
            shaped_reward_spec_sha256=EXACT_P95_RUN2_REWARD_SPEC_SHA256,
            deadline_ms=RUN2_DEADLINE_MS,
            deadline_penalty=RUN2_DEADLINE_PENALTY,
            deadline_penalty_float_hex=RUN2_DEADLINE_PENALTY_HEX,
            terminal_discount=RUN2_TERMINAL_DISCOUNT,
            preflight_summary_file_sha256=PREFLIGHT_SUMMARY_FILE_SHA256,
            preflight_decision_file_sha256=PREFLIGHT_DECISION_FILE_SHA256,
            preflight_report_file_sha256=PREFLIGHT_REPORT_FILE_SHA256,
            preflight_context_oracles_file_sha256=(
                PREFLIGHT_CONTEXT_ORACLES_FILE_SHA256
            ),
            preflight_implementation_sha256=PREFLIGHT_IMPLEMENTATION_SHA256,
            preflight_canonical_content_sha256=(
                PREFLIGHT_CANONICAL_CONTENT_SHA256
            ),
            run2_preregistration_file_sha256=(
                RUN2_PREREGISTRATION_FILE_SHA256
            ),
        )
        binding.require_valid()
        return binding

    def require_valid(self) -> None:
        expected = (
            canonical_sha256(exact_p95_run2_reward_spec_document())
            == EXACT_P95_RUN2_REWARD_SPEC_SHA256
            and self.source_d1_utility_spec_sha256 == PILOT_UTILITY_SPEC_SHA256
            and self.shaped_reward_spec_sha256
            == EXACT_P95_RUN2_REWARD_SPEC_SHA256
            and self.shaped_reward_spec_sha256 != self.source_d1_utility_spec_sha256
            and type(self.deadline_ms) is float
            and self.deadline_ms == RUN2_DEADLINE_MS
            and type(self.deadline_penalty) is float
            and self.deadline_penalty == RUN2_DEADLINE_PENALTY
            and self.deadline_penalty.hex() == RUN2_DEADLINE_PENALTY_HEX
            and self.deadline_penalty_float_hex == RUN2_DEADLINE_PENALTY_HEX
            and type(self.terminal_discount) is float
            and self.terminal_discount == RUN2_TERMINAL_DISCOUNT
            and self.preflight_summary_file_sha256
            == PREFLIGHT_SUMMARY_FILE_SHA256
            and self.preflight_decision_file_sha256
            == PREFLIGHT_DECISION_FILE_SHA256
            and self.preflight_report_file_sha256 == PREFLIGHT_REPORT_FILE_SHA256
            and self.preflight_context_oracles_file_sha256
            == PREFLIGHT_CONTEXT_ORACLES_FILE_SHA256
            and self.preflight_implementation_sha256
            == PREFLIGHT_IMPLEMENTATION_SHA256
            and self.preflight_canonical_content_sha256
            == PREFLIGHT_CANONICAL_CONTENT_SHA256
            and self.run2_preregistration_file_sha256
            == RUN2_PREREGISTRATION_FILE_SHA256
            and self.schema == RUN2_REWARD_SCHEMA
        )
        hashes = (
            self.source_d1_binding_sha256,
            self.source_d1_utility_spec_sha256,
            self.shaped_reward_spec_sha256,
            self.preflight_summary_file_sha256,
            self.preflight_decision_file_sha256,
            self.preflight_report_file_sha256,
            self.preflight_context_oracles_file_sha256,
            self.preflight_implementation_sha256,
            self.preflight_canonical_content_sha256,
            self.run2_preregistration_file_sha256,
        )
        if not expected or any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ExactP95Run2BindingError("exact-P95 reward binding drift")

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.require_valid()
        return asdict(self)

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class ExactP95Run2ShapedTransitionV1:
    """An original validated D1 transition plus a separately bound reward."""

    source_d1_transition: EmpiricalTerminalTransitionV1
    reward_binding: ExactP95Run2RewardBindingV1
    source_d1_reward: float
    p95_base_reward: float
    shaped_reward: float
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
        reward_binding: Optional[ExactP95Run2RewardBindingV1] = None,
    ) -> "ExactP95Run2ShapedTransitionV1":
        if type(transition) is not EmpiricalTerminalTransitionV1:
            raise ExactP95Run2ReplayError("adapter requires an exact D1 transition")
        transition.revalidate()
        if reward_binding is None:
            binding = ExactP95Run2RewardBindingV1.from_validated_d1(
                transition, project_root=project_root
            )
        else:
            if type(reward_binding) is not ExactP95Run2RewardBindingV1:
                raise ExactP95Run2BindingError(
                    "reused reward binding has a foreign type"
                )
            reward_binding.require_valid()
            if (
                reward_binding.source_d1_binding_sha256
                != transition.d1_binding.canonical_sha256()
            ):
                raise ExactP95Run2BindingError(
                    "reused reward binding differs from D1"
                )
            binding = reward_binding
        policy = transition.result.policy
        p = _require_float(policy.p_edge_admission_given_sent, "p_admit")
        quality = _require_float(policy.q_perc, "q_perc")
        latency = _require_float(policy.latency_proxy_p95_ms, "latency_proxy_p95_ms")
        base = base_p95_expected_utility(
            p_admit=p, q_perc=quality, latency_p95_ms=latency
        )
        shaped = shaped_p95_expected_utility(
            p_admit=p,
            q_perc=quality,
            latency_p95_ms=latency,
            deadline_penalty=RUN2_DEADLINE_PENALTY,
        )
        document = {
            "p95_base_reward": base,
            "record": RUN2_TRANSITION_SCHEMA,
            "reward_binding_sha256": binding.canonical_sha256(),
            "shaped_reward": shaped,
            "source_d1_reward": transition.reward,
            "source_d1_transition_sha256": transition.canonical_sha256(),
            "terminal_discount": RUN2_TERMINAL_DISCOUNT,
        }
        result = cls(
            source_d1_transition=transition,
            reward_binding=binding,
            source_d1_reward=transition.reward,
            p95_base_reward=base,
            shaped_reward=shaped,
            terminal_discount=RUN2_TERMINAL_DISCOUNT,
            _attestation_sha256=canonical_sha256(document),
        )
        result.revalidate()
        return result

    @property
    def logical_key(self) -> Tuple[str, int]:
        return self.source_d1_transition.logical_key

    def _document(self) -> Dict[str, Any]:
        return {
            "p95_base_reward": self.p95_base_reward,
            "record": RUN2_TRANSITION_SCHEMA,
            "reward_binding_sha256": self.reward_binding.canonical_sha256(),
            "shaped_reward": self.shaped_reward,
            "source_d1_reward": self.source_d1_reward,
            "source_d1_transition_sha256": (
                self.source_d1_transition.canonical_sha256()
            ),
            "terminal_discount": self.terminal_discount,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self._document())

    def revalidate(self) -> None:
        if type(self.source_d1_transition) is not EmpiricalTerminalTransitionV1:
            raise ExactP95Run2ReplayError("source transition has a foreign type")
        if type(self.reward_binding) is not ExactP95Run2RewardBindingV1:
            raise ExactP95Run2BindingError("reward binding has a foreign type")
        self.source_d1_transition.revalidate()
        self.reward_binding.require_valid()
        for name in (
            "source_d1_reward",
            "p95_base_reward",
            "shaped_reward",
            "terminal_discount",
        ):
            _require_float(getattr(self, name), name)
        if (
            self.reward_binding.source_d1_binding_sha256
            != self.source_d1_transition.d1_binding.canonical_sha256()
        ):
            raise ExactP95Run2BindingError("reward/source D1 binding mismatch")
        policy = self.source_d1_transition.result.policy
        expected_base = base_p95_expected_utility(
            p_admit=float(policy.p_edge_admission_given_sent),
            q_perc=float(policy.q_perc),
            latency_p95_ms=float(policy.latency_proxy_p95_ms),
        )
        expected_shaped = shaped_p95_expected_utility(
            p_admit=float(policy.p_edge_admission_given_sent),
            q_perc=float(policy.q_perc),
            latency_p95_ms=float(policy.latency_proxy_p95_ms),
            deadline_penalty=RUN2_DEADLINE_PENALTY,
        )
        if (
            self.source_d1_reward != self.source_d1_transition.reward
            or self.p95_base_reward != expected_base
            or self.shaped_reward != expected_shaped
            or self.terminal_discount != RUN2_TERMINAL_DISCOUNT
            or self._attestation_sha256 != self.canonical_sha256()
        ):
            raise ExactP95Run2ReplayError("shaped transition attestation drift")


@dataclass(frozen=True, slots=True)
class ExactP95Run2ReplayBindingV1:
    source_terminal_binding: EmpiricalTerminalBindingV1
    reward_binding: ExactP95Run2RewardBindingV1
    policy_feature_order: Tuple[str, ...]
    policy_feature_count: int
    float_dtype: str
    terminal_discount: float
    schema: str = RUN2_SCHEMA

    def __post_init__(self) -> None:
        self.require_valid()

    @classmethod
    def from_transition(
        cls, transition: ExactP95Run2ShapedTransitionV1
    ) -> "ExactP95Run2ReplayBindingV1":
        if type(transition) is not ExactP95Run2ShapedTransitionV1:
            raise ExactP95Run2BindingError("replay binding requires exact Run-2 type")
        transition.revalidate()
        result = cls(
            source_terminal_binding=EmpiricalTerminalBindingV1.from_transition(
                transition.source_d1_transition
            ),
            reward_binding=transition.reward_binding,
            policy_feature_order=tuple(POLICY_FEATURE_ORDER),
            policy_feature_count=POLICY_FEATURE_COUNT,
            float_dtype=str(torch.float32),
            terminal_discount=RUN2_TERMINAL_DISCOUNT,
        )
        result.require_valid()
        return result

    def require_valid(self) -> None:
        if type(self.source_terminal_binding) is not EmpiricalTerminalBindingV1:
            raise ExactP95Run2BindingError("source replay binding has a foreign type")
        if type(self.reward_binding) is not ExactP95Run2RewardBindingV1:
            raise ExactP95Run2BindingError("reward binding has a foreign type")
        self.source_terminal_binding.__post_init__()
        self.reward_binding.require_valid()
        if (
            self.source_terminal_binding.d1_binding.canonical_sha256()
            != self.reward_binding.source_d1_binding_sha256
            or self.policy_feature_order != tuple(POLICY_FEATURE_ORDER)
            or self.policy_feature_count != POLICY_FEATURE_COUNT
            or self.float_dtype != str(torch.float32)
            or self.terminal_discount != RUN2_TERMINAL_DISCOUNT
            or self.schema != RUN2_SCHEMA
        ):
            raise ExactP95Run2BindingError("exact-P95 replay binding drift")

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.require_valid()
        return {
            "float_dtype": self.float_dtype,
            "policy_feature_count": self.policy_feature_count,
            "policy_feature_order": list(self.policy_feature_order),
            "reward_binding": self.reward_binding.to_canonical_dict(),
            "schema": self.schema,
            "source_terminal_binding": (
                self.source_terminal_binding.to_canonical_dict()
            ),
            "terminal_discount": self.terminal_discount,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())

    def assert_matches(self, other: object) -> None:
        if type(other) is not ExactP95Run2ReplayBindingV1 or other != self:
            raise ExactP95Run2BindingError("Run-2 replay binding mismatch")


_BATCH_AUDIT_SOURCE_KEY = "_authenticated_run2_transition"
_BATCH_AUDIT_ATTESTATION_KEY = "row_attestation_sha256"


def _float32_scalar(value: float) -> float:
    """Return the exact scalar value stored by a CPU float32 tensor."""
    return float(torch.tensor(value, dtype=torch.float32).item())


def _batch_audit_payload(
    transition: ExactP95Run2ShapedTransitionV1,
) -> Dict[str, Any]:
    """Derive one canonical execution-dtype row from its authenticated source."""
    if type(transition) is not ExactP95Run2ShapedTransitionV1:
        raise ExactP95Run2ReplayError("batch audit source has a foreign type")
    transition.revalidate()
    source = transition.source_d1_transition
    return {
        "collection_seq": source.collection_seq,
        "mode_id": source.action.mode_id,
        "p95_base_reward": transition.p95_base_reward,
        "p95_base_reward_float32": _float32_scalar(transition.p95_base_reward),
        "q_e4": source.action.q_e4,
        "record": "splitfusion.exact_p95_run2_batch_audit_row.v1",
        "reward_binding_sha256": transition.reward_binding.canonical_sha256(),
        "run2_transition_sha256": transition.canonical_sha256(),
        "shaped_reward": transition.shaped_reward,
        "shaped_reward_float32": _float32_scalar(transition.shaped_reward),
        "source_d1_reward": transition.source_d1_reward,
        "source_d1_reward_float32": _float32_scalar(transition.source_d1_reward),
        "source_d1_transition_sha256": source.canonical_sha256(),
        "state_float32": [
            float(value)
            for value in torch.tensor(
                source.observation.values, dtype=torch.float32
            ).tolist()
        ],
        "terminal_discount": transition.terminal_discount,
        "terminal_discount_float32": _float32_scalar(
            transition.terminal_discount
        ),
    }


def _batch_audit_row(
    transition: ExactP95Run2ShapedTransitionV1,
) -> Dict[str, Any]:
    payload = _batch_audit_payload(transition)
    return {
        **payload,
        _BATCH_AUDIT_ATTESTATION_KEY: canonical_sha256(payload),
        _BATCH_AUDIT_SOURCE_KEY: transition,
    }


@dataclass(frozen=True, slots=True, eq=False)
class ExactP95Run2BatchV1:
    _state: Tensor
    _mode_id: Tensor
    _q_e4: Tensor
    _source_d1_reward: Tensor
    _p95_base_reward: Tensor
    _reward: Tensor
    _discount: Tensor
    binding: ExactP95Run2ReplayBindingV1
    audit: Tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if type(self.binding) is not ExactP95Run2ReplayBindingV1:
            raise ExactP95Run2BindingError("batch binding has a foreign type")
        self.binding.require_valid()
        tensors = (
            "_state", "_mode_id", "_q_e4", "_source_d1_reward",
            "_p95_base_reward", "_reward", "_discount",
        )
        for name in tensors:
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                raise ExactP95Run2ReplayError(f"{name} must be a tensor")
            object.__setattr__(self, name, value.detach().clone())
        size = int(self._state.shape[0]) if self._state.ndim else 0
        if self._state.shape != (size, POLICY_FEATURE_COUNT) or size < 1:
            raise ExactP95Run2ReplayError("state tensor shape drift")
        if self._state.dtype is not torch.float32 or self._state.device.type != "cpu":
            raise ExactP95Run2ReplayError("state must be CPU float32")
        for name in ("_mode_id", "_q_e4"):
            value = getattr(self, name)
            if (
                value.shape != (size,)
                or value.dtype is not torch.int64
                or value.device.type != "cpu"
            ):
                raise ExactP95Run2ReplayError(f"{name} must be [B] int64")
        for name in (
            "_source_d1_reward", "_p95_base_reward", "_reward", "_discount"
        ):
            value = getattr(self, name)
            if (
                value.shape != (size,)
                or value.dtype is not torch.float32
                or value.device.type != "cpu"
                or not bool(torch.isfinite(value).all())
            ):
                raise ExactP95Run2ReplayError(f"{name} must be finite CPU float32")
        if not bool(torch.isfinite(self._state).all()):
            raise ExactP95Run2ReplayError("state contains non-finite values")
        if not torch.equal(self._discount, torch.zeros_like(self._discount)):
            raise ExactP95Run2ReplayError("batch discount differs from terminal contract")
        if bool((self._mode_id < 0).any()) or bool(
            (self._mode_id >= EXPECTED_MODE_COUNT).any()
        ):
            raise ExactP95Run2ReplayError("mode IDs escaped the catalog")
        for mode_id, q_e4 in zip(self._mode_id.tolist(), self._q_e4.tolist()):
            try:
                require_supported_action(mode_id, q_e4)
            except ValueError as exc:
                raise ExactP95Run2ReplayError(
                    "batch action escaped modeled support"
                ) from exc
        if type(self.audit) is not tuple or len(self.audit) != size:
            raise ExactP95Run2ReplayError("batch audit cardinality drift")
        checked_audit = []
        for index, supplied in enumerate(self.audit):
            if not isinstance(supplied, Mapping):
                raise ExactP95Run2ReplayError("batch audit row is not a mapping")
            row = dict(supplied)
            source = row.pop(_BATCH_AUDIT_SOURCE_KEY, None)
            stated_attestation = row.pop(_BATCH_AUDIT_ATTESTATION_KEY, None)
            expected = _batch_audit_payload(source)
            if set(row) != set(expected):
                raise ExactP95Run2ReplayError("batch audit row schema drift")
            if (
                canonical_sha256(row) != stated_attestation
                or row != expected
                or canonical_sha256(expected) != stated_attestation
            ):
                raise ExactP95Run2ReplayError("batch audit row attestation drift")
            if (
                source.reward_binding != self.binding.reward_binding
                or EmpiricalTerminalBindingV1.from_transition(
                    source.source_d1_transition
                )
                != self.binding.source_terminal_binding
            ):
                raise ExactP95Run2BindingError(
                    "batch audit source differs from batch binding"
                )
            expected_state = torch.tensor(
                expected["state_float32"], dtype=torch.float32
            )
            scalar_checks = (
                (self._source_d1_reward, "source_d1_reward_float32"),
                (self._p95_base_reward, "p95_base_reward_float32"),
                (self._reward, "shaped_reward_float32"),
                (self._discount, "terminal_discount_float32"),
            )
            if (
                not torch.equal(self._state[index], expected_state)
                or int(self._mode_id[index]) != expected["mode_id"]
                or int(self._q_e4[index]) != expected["q_e4"]
                or any(
                    not torch.equal(
                        tensor[index : index + 1],
                        torch.tensor([expected[name]], dtype=torch.float32),
                    )
                    for tensor, name in scalar_checks
                )
            ):
                raise ExactP95Run2ReplayError(
                    "batch tensor row differs from authenticated audit source"
                )
            checked_audit.append(
                MappingProxyType(
                    {
                        **expected,
                        _BATCH_AUDIT_ATTESTATION_KEY: stated_attestation,
                        _BATCH_AUDIT_SOURCE_KEY: source,
                    }
                )
            )
        object.__setattr__(
            self,
            "audit",
            tuple(checked_audit),
        )

    def revalidate(self) -> None:
        """Re-prove tensor/source identity after construction.

        A later trainer must call this before any optimizer mutation.  Calling
        ``__post_init__`` deliberately repeats every exact source, binding,
        audit and tensor comparison; it also refreshes the private tensor
        clones so no caller-owned storage can become authoritative.
        """

        if type(self) is not ExactP95Run2BatchV1:
            raise ExactP95Run2ReplayError("Run-2 batch has a foreign type")
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
    def source_d1_reward(self) -> Tensor:
        return self._source_d1_reward.clone()

    @property
    def p95_base_reward(self) -> Tensor:
        return self._p95_base_reward.clone()

    @property
    def reward(self) -> Tensor:
        return self._reward.clone()

    def discount(self) -> Tensor:
        return self._discount.clone()

    def terminal_target(self) -> Tensor:
        """Return the later trainer target without recomputation."""
        return self._reward.clone()

class ExactP95Run2ReplayV1:
    """FIFO replay accepting only separately attested Run-2 transitions."""

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ExactP95Run2ReplayError("capacity must be an exact positive integer")
        self._capacity = capacity
        self._rows: Deque[ExactP95Run2ShapedTransitionV1] = deque()
        self._binding: Optional[ExactP95Run2ReplayBindingV1] = None
        self._seen_digests: set[str] = set()
        self._seen_keys: Dict[Tuple[str, int], str] = {}
        self._accepted_count = 0
        self._evicted_count = 0

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def binding(self) -> Optional[ExactP95Run2ReplayBindingV1]:
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
        if type(transition) is not ExactP95Run2ShapedTransitionV1:
            raise ExactP95Run2ReplayError("replay accepts only exact Run-2 transitions")
        transition.revalidate()
        candidate_binding = ExactP95Run2ReplayBindingV1.from_transition(transition)
        if self._binding is not None:
            self._binding.assert_matches(candidate_binding)
        digest = transition.canonical_sha256()
        key = transition.logical_key
        known = self._seen_keys.get(key)
        if known is not None and known != digest:
            raise ExactP95Run2ReplayError("collection identity conflicts with prior row")
        if digest in self._seen_digests:
            raise ExactP95Run2ReplayError("duplicate Run-2 transition")
        # Prove conversion before committing any replay mutation.
        state = torch.tensor(
            transition.source_d1_transition.observation.values,
            dtype=torch.float32,
        )
        scalars = torch.tensor(
            [
                transition.source_d1_reward,
                transition.p95_base_reward,
                transition.shaped_reward,
                transition.terminal_discount,
            ],
            dtype=torch.float32,
        )
        if not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(scalars).all()):
            raise ExactP95Run2ReplayError("Run-2 row is non-finite in replay dtype")
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
            not isinstance(generator, torch.Generator)
            or generator is torch.default_generator
            or generator.device.type != "cpu"
        ):
            raise ExactP95Run2ReplayError("sample requires a local CPU generator")
        return generator

    def sample(
        self, batch_size: int, generator: torch.Generator
    ) -> ExactP95Run2BatchV1:
        if type(batch_size) is not int or batch_size < 1:
            raise ExactP95Run2ReplayError("batch size must be an exact positive integer")
        self._require_generator(generator)
        if batch_size > len(self._rows) or self._binding is None:
            raise ExactP95Run2ReplayError("cannot sample requested Run-2 batch")
        indices = torch.randperm(len(self._rows), generator=generator)[:batch_size]
        resident = tuple(self._rows)
        rows = tuple(resident[int(index)] for index in indices)
        audits = tuple(_batch_audit_row(row) for row in rows)
        return ExactP95Run2BatchV1(
            _state=torch.tensor(
                [row.source_d1_transition.observation.values for row in rows],
                dtype=torch.float32,
            ),
            _mode_id=torch.tensor(
                [row.source_d1_transition.action.mode_id for row in rows],
                dtype=torch.int64,
            ),
            _q_e4=torch.tensor(
                [row.source_d1_transition.action.q_e4 for row in rows],
                dtype=torch.int64,
            ),
            _source_d1_reward=torch.tensor(
                [row.source_d1_reward for row in rows], dtype=torch.float32
            ),
            _p95_base_reward=torch.tensor(
                [row.p95_base_reward for row in rows], dtype=torch.float32
            ),
            _reward=torch.tensor(
                [row.shaped_reward for row in rows], dtype=torch.float32
            ),
            _discount=torch.tensor(
                [row.terminal_discount for row in rows], dtype=torch.float32
            ),
            binding=self._binding,
            audit=audits,
        )
