"""Deployable shielded and non-deployable clairvoyant Track A oracles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .env import SurrogateEnv


@dataclass
class OracleRun:
    controller: str
    rows: List[Dict[str, object]]


def run_oracle(env: SurrogateEnv, controller: str) -> OracleRun:
    if controller not in {"shielded", "clairvoyant"}:
        raise ValueError("controller must be shielded or clairvoyant")
    rows: List[Dict[str, object]] = []
    while not env.done:
        decision = env.shielded_decision() if controller == "shielded" else env.clairvoyant_decision()
        selected_true = env.true_evaluation(decision.selected.action)
        selected_matched_true = env.matched_truth_evaluation(decision.selected.action)
        true_decision = env.clairvoyant_decision()
        selected_is_true_safe = selected_true.risk_p95_m <= float(env.config["safety"]["epsilon_m"])
        selected_is_matched_true_safe = selected_matched_true.risk_p95_m <= float(
            env.config["safety"]["epsilon_m"]
        )
        selected_in_true_candidate_set = decision.selected.action.action_id in true_decision.safe_action_ids
        safe_only_gap = (
            max(0.0, true_decision.selected.expected_reward - selected_true.expected_reward)
            if selected_in_true_candidate_set
            else None
        )
        unobserved_gt_object_count = max(
            0, len(env.frame.truth_objects) - len(env.frame.observed_objects)
        )
        row = env.step(decision.selected.action)
        row.update(
            {
                "controller": controller,
                "shield_feasible": decision.feasible,
                "shield_over_budget": decision.over_budget,
                "shield_ood": decision.shield_ood,
                "degraded_tier_used": decision.degraded_tier_used,
                "shield_bound_m": decision.selected.bound_m,
                "shield_expected_g_m": decision.selected.expected_g_m,
                "shield_expected_task_utility": decision.selected.expected_task_utility,
                "shield_prb_cost": decision.selected.prb_cost,
                "shield_expected_reward": decision.selected.expected_reward,
                "selected_true_safe": selected_is_true_safe,
                "matched_true_risk_p95_m": selected_matched_true.risk_p95_m,
                "selected_matched_true_safe": selected_is_matched_true_safe,
                "false_admit_selected": decision.selected.bound_m <= float(env.config["safety"]["epsilon_m"])
                and not selected_is_true_safe,
                "false_admit_selected_matched": decision.selected.bound_m
                <= float(env.config["safety"]["epsilon_m"])
                and not selected_is_matched_true_safe,
                "false_reject_frame": bool(
                    true_decision.feasible
                    and not any(
                        evaluation.action.action_id in decision.safe_action_ids
                        for evaluation in true_decision.evaluations
                        if evaluation.hard_admitted
                        and evaluation.risk_p95_m <= float(env.config["safety"]["epsilon_m"])
                    )
                ),
                "clairvoyant_best_action_id": true_decision.selected.action.action_id,
                "clairvoyant_best_reward": true_decision.selected.expected_reward,
                "selected_in_true_candidate_set": selected_in_true_candidate_set,
                "oracle_reward_gap_safe_only": safe_only_gap,
                "oracle_action_set_mismatch": not selected_in_true_candidate_set,
                "unobserved_gt_object_count": unobserved_gt_object_count,
            }
        )
        rows.append(row)
    return OracleRun(controller=controller, rows=rows)
