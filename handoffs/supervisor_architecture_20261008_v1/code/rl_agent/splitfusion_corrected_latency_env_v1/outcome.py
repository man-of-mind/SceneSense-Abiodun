"""Live-parity outcome classification for one reward-requested decision.

Mirrors ``reward_hold_controller_v2`` (poll/on_feedback) and ``run4_live_wire_v2.quality_feedback``:

* the controller times out only when ``now > open + 170 ms`` (strict), so a receipt at exactly
  ``open + 170 000 000 ns`` is accepted and resolved at its receipt time;
* an accepted reply with defined Q_perc is ``DELIVERED_SUCCESS`` (reward Q - 0.25 L/170 via the unchanged
  ``run4_contract.resolve_reward``); an accepted reply with undefined Q_perc is ``EVALUATOR_FAULT``: excluded
  from learning, and the session breaks so the next decision is a fresh-session genesis;
* any later or missing reply resolves ``TIMEOUT`` at ``open + 170 ms + 1 ns``; the late reply is a LATE_ORPHAN
  whose Q and latency never reach the state, whether or not its quality was defined;
* holds carry no ticket and fallbacks open none: neither creates a state or changes the previous outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_run4_v1 import sequential_kernel

DEADLINE_NS = contract.REWARD_DEADLINE_NS                        # 170_000_000, unchanged
TIMEOUT_RESOLUTION_ELAPSED_NS = sequential_kernel.TIMEOUT_RESOLUTION_ELAPSED_NS
assert DEADLINE_NS == 170_000_000 and TIMEOUT_RESOLUTION_ELAPSED_NS == DEADLINE_NS + 1


@dataclass(frozen=True)
class OutcomeV1:
    kind: contract.RewardEventKind
    resolution_elapsed_ns: int
    q_perc: Optional[float]
    feedback_class: str          # ACCEPTED | LATE_ORPHAN | MISSING

    @property
    def breaks_session(self) -> bool:
        return self.kind in (contract.RewardEventKind.EVALUATOR_FAULT, contract.RewardEventKind.INFRASTRUCTURE_FAULT)


def receipt_elapsed_ns(receipt_ms: Optional[float]) -> Optional[int]:
    """Composed receipt (ms after action-open) on the integer-ns controller clock."""
    if receipt_ms is None:
        return None
    return int(round(receipt_ms * 1_000_000))


def classify(receipt_ns: Optional[int], q_perc: Optional[float]) -> OutcomeV1:
    """Resolve one decision exactly as the live controller does, from its receipt time and quality."""
    if receipt_ns is None:
        return OutcomeV1(contract.RewardEventKind.TIMEOUT, TIMEOUT_RESOLUTION_ELAPSED_NS, None, "MISSING")
    if type(receipt_ns) is not int or receipt_ns < 0:
        raise ValueError("receipt must be a non-negative integer ns offset from action-open")
    if receipt_ns > DEADLINE_NS:
        return OutcomeV1(contract.RewardEventKind.TIMEOUT, TIMEOUT_RESOLUTION_ELAPSED_NS, None, "LATE_ORPHAN")
    if q_perc is None:
        return OutcomeV1(contract.RewardEventKind.EVALUATOR_FAULT, receipt_ns, None, "ACCEPTED")
    return OutcomeV1(contract.RewardEventKind.DELIVERED_SUCCESS, receipt_ns, float(q_perc), "ACCEPTED")


CONTROLLER_TERMINAL = {
    contract.RewardEventKind.DELIVERED_SUCCESS: "SUCCESS",
    contract.RewardEventKind.TIMEOUT: "TIMEOUT",
    contract.RewardEventKind.EVALUATOR_FAULT: "EVALUATOR_FAULT",
}


def previous_outcome_features(kind: Optional[contract.RewardEventKind], *, mode_id: Optional[int],
                              q_e4: Optional[int], q_perc: Optional[float],
                              latency_ms: Optional[float]) -> tuple[float, ...]:
    """Positions 4-20 of the unchanged Run-4/Run-5 state for the decision after ``kind``.

    ``kind=None`` is genesis (first decision of a session, including after a session break). This helper is for
    parity checks only; the environment itself always uses ``run4_contract.build_policy_features``.
    """
    one_hot = [0.0] * 12
    if kind is None:
        return tuple(one_hot + [0.0, 0.0, 0.0, 0.0, 0.0])
    one_hot[int(mode_id)] = 1.0
    success = kind is contract.RewardEventKind.DELIVERED_SUCCESS
    return tuple(one_hot + [int(q_e4) / float(contract.Q_E4_MAX), float(q_perc) if success else 0.0,
                            float(latency_ms) / contract.REWARD_DEADLINE_MS if success else 0.0, 1.0, 1.0 if success else 0.0])
