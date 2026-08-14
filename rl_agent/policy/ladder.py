"""Common shield-gated execution path for deployable controller comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .controllers import DeployableController
from .env import SurrogateEnv


@dataclass
class ControllerRun:
    controller: str
    training: bool
    rows: List[Dict[str, object]]


def run_deployable_controller(
    env: SurrogateEnv,
    controller: DeployableController,
    *,
    training: bool = False,
    feedback_source: str = "matched_true_expected_reward",
) -> ControllerRun:
    """Run one episode without exposing latent state to ``controller.select``.

    Hidden matched truth is consulted only after action selection, for training
    feedback and evaluation fields.  Evaluation runs never update a fitted
    controller.
    """

    if feedback_source != "matched_true_expected_reward":
        raise ValueError("only matched_true_expected_reward feedback is currently contracted")
    controller.reset(env.frame.episode_id)
    rows: List[Dict[str, object]] = []
    epsilon = float(env.config["safety"]["epsilon_m"])
    while not env.done:
        observation = env.observation(truth=False)
        decision = env.shielded_decision()
        selection = controller.select(observation, decision)
        if selection.action_id not in decision.candidate_action_ids:
            raise RuntimeError(
                f"{controller.name} bypassed the shared shield with {selection.action_id}"
            )
        selected = next(
            item for item in decision.evaluations if item.action.action_id == selection.action_id
        )

        # Evaluation-only counterfactuals are computed after the controller has
        # committed to an action and are never passed to select().
        true_decision = env.clairvoyant_decision()
        matched_true_decision = env.matched_truth_decision()
        selected_true = next(
            item for item in true_decision.evaluations if item.action.action_id == selection.action_id
        )
        selected_matched_true = next(
            item
            for item in matched_true_decision.evaluations
            if item.action.action_id == selection.action_id
        )
        selected_true_safe = selected_true.risk_p95_m <= epsilon
        selected_matched_true_safe = selected_matched_true.risk_p95_m <= epsilon
        selected_in_true_candidates = selection.action_id in true_decision.candidate_action_ids
        selected_raw_safe = selection.action_id in decision.raw_safe_action_ids
        selected_admitted_split = selected_raw_safe and selected.action.mode == "SPLIT"
        true_safe_overlap = bool(decision.raw_safe_action_ids & true_decision.raw_safe_action_ids)
        matched_true_safe_overlap = bool(
            decision.raw_safe_action_ids & matched_true_decision.raw_safe_action_ids
        )
        safe_only_gap = (
            max(0.0, true_decision.selected.expected_reward - selected_true.expected_reward)
            if selected_in_true_candidates
            else None
        )
        unobserved_gt_object_count = max(
            0, len(env.frame.truth_objects) - len(env.frame.observed_objects)
        )

        row = env.step(selected.action)
        feedback_reward = selected_matched_true.expected_reward
        if training:
            controller.update(observation, selection.action_id, feedback_reward)
        row.update(
            {
                "controller": controller.name,
                "controller_training": training,
                "controller_feedback_source": feedback_source if training else "none",
                "controller_feedback_reward": feedback_reward if training else None,
                "selection_changed_from_greedy": (
                    selection.action_id != decision.selected.action.action_id
                ),
                "shield_feasible": decision.feasible,
                "shield_over_budget": decision.over_budget,
                "shield_ood": decision.shield_ood,
                "degraded_tier_used": (
                    selected.action.mode == "SPLIT" and not selected.action.core_tier
                ),
                "shield_bound_m": selected.bound_m,
                "shield_expected_g_m": selected.expected_g_m,
                "shield_risk_sigma_m": selected.risk_sigma_m,
                "shield_expected_task_utility": selected.expected_task_utility,
                "shield_prb_cost": selected.prb_cost,
                "shield_expected_reward": selected.expected_reward,
                "matched_true_expected_reward": selected_matched_true.expected_reward,
                "selected_true_safe": selected_true_safe,
                "matched_true_risk_p95_m": selected_matched_true.risk_p95_m,
                "selected_matched_true_safe": selected_matched_true_safe,
                "matched_true_unobserved_sentinel": (
                    selected_matched_true.risk_p95_m >= 500_000.0
                ),
                "selected_raw_safe": selected_raw_safe,
                "selected_admitted_split": selected_admitted_split,
                "raw_safe_action_count": len(decision.raw_safe_action_ids),
                "candidate_action_count": len(decision.candidate_action_ids),
                "raw_safe_action_ids": "|".join(sorted(decision.raw_safe_action_ids)),
                "candidate_action_ids": "|".join(sorted(decision.candidate_action_ids)),
                "observed_vulnerable_count": decision.observed_vulnerable_count,
                "observed_low_confidence_vulnerable_count": (
                    decision.observed_low_confidence_vulnerable_count
                ),
                "vulnerable_guardrail_applied": decision.vulnerable_guardrail_applied,
                "vulnerable_guardrail_unachievable": (
                    decision.vulnerable_guardrail_unachievable
                ),
                "vulnerable_guardrail_removed_action_count": len(
                    decision.vulnerable_guardrail_removed_action_ids
                ),
                "vulnerable_guardrail_removed_action_ids": "|".join(
                    sorted(decision.vulnerable_guardrail_removed_action_ids)
                ),
                "false_admit_selected": selected_raw_safe and not selected_true_safe,
                "false_admit_selected_matched": (
                    selected_raw_safe and not selected_matched_true_safe
                ),
                "true_feasible_frame": true_decision.feasible,
                "matched_true_feasible_frame": matched_true_decision.feasible,
                "false_reject_frame": bool(true_decision.feasible and not true_safe_overlap),
                "false_reject_frame_matched": bool(
                    matched_true_decision.feasible and not matched_true_safe_overlap
                ),
                "clairvoyant_best_action_id": true_decision.selected.action.action_id,
                "clairvoyant_best_mode": true_decision.selected.action.mode,
                "shield_skip_clairvoyant_split": (
                    selected.action.mode == "SKIP"
                    and true_decision.selected.action.mode == "SPLIT"
                ),
                "clairvoyant_best_reward": true_decision.selected.expected_reward,
                "selected_in_true_candidate_set": selected_in_true_candidates,
                "oracle_reward_gap_safe_only": safe_only_gap,
                "oracle_action_set_mismatch": not selected_in_true_candidates,
                "unobserved_gt_object_count": unobserved_gt_object_count,
                **dict(selection.diagnostics),
            }
        )
        rows.append(row)
    return ControllerRun(controller=controller.name, training=training, rows=rows)
