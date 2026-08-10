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
        true_decision = env.clairvoyant_decision()
        matched_true_decision = env.matched_truth_decision()
        selected_action_id = decision.selected.action.action_id
        selected_true = next(
            item for item in true_decision.evaluations if item.action.action_id == selected_action_id
        )
        selected_matched_true = next(
            item for item in matched_true_decision.evaluations if item.action.action_id == selected_action_id
        )
        selected_is_true_safe = selected_true.risk_p95_m <= float(env.config["safety"]["epsilon_m"])
        selected_is_matched_true_safe = selected_matched_true.risk_p95_m <= float(
            env.config["safety"]["epsilon_m"]
        )
        selected_in_true_candidate_set = selected_action_id in true_decision.candidate_action_ids
        selected_raw_safe = selected_action_id in decision.raw_safe_action_ids
        selected_admitted_split = selected_raw_safe and decision.selected.action.mode == "SPLIT"
        true_safe_overlap = bool(decision.raw_safe_action_ids & true_decision.raw_safe_action_ids)
        matched_true_safe_overlap = bool(
            decision.raw_safe_action_ids & matched_true_decision.raw_safe_action_ids
        )
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
                "shield_risk_sigma_m": decision.selected.risk_sigma_m,
                "shield_expected_task_utility": decision.selected.expected_task_utility,
                "shield_prb_cost": decision.selected.prb_cost,
                "shield_expected_reward": decision.selected.expected_reward,
                "matched_true_expected_reward": selected_matched_true.expected_reward,
                "selected_true_safe": selected_is_true_safe,
                "matched_true_risk_p95_m": selected_matched_true.risk_p95_m,
                "selected_matched_true_safe": selected_is_matched_true_safe,
                "matched_true_unobserved_sentinel": selected_matched_true.risk_p95_m >= 500_000.0,
                "selected_raw_safe": selected_raw_safe,
                "selected_admitted_split": selected_admitted_split,
                "raw_safe_action_count": len(decision.raw_safe_action_ids),
                "candidate_action_count": len(decision.candidate_action_ids),
                "raw_safe_action_ids": "|".join(sorted(decision.raw_safe_action_ids)),
                "candidate_action_ids": "|".join(sorted(decision.candidate_action_ids)),
                "false_admit_selected": selected_raw_safe and not selected_is_true_safe,
                "false_admit_selected_matched": selected_raw_safe and not selected_is_matched_true_safe,
                "true_feasible_frame": true_decision.feasible,
                "matched_true_feasible_frame": matched_true_decision.feasible,
                "false_reject_frame": bool(true_decision.feasible and not true_safe_overlap),
                "false_reject_frame_matched": bool(
                    matched_true_decision.feasible and not matched_true_safe_overlap
                ),
                "clairvoyant_best_action_id": true_decision.selected.action.action_id,
                "clairvoyant_best_mode": true_decision.selected.action.mode,
                "shield_skip_clairvoyant_split": (
                    decision.selected.action.mode == "SKIP"
                    and true_decision.selected.action.mode == "SPLIT"
                ),
                "clairvoyant_best_reward": true_decision.selected.expected_reward,
                "selected_in_true_candidate_set": selected_in_true_candidate_set,
                "oracle_reward_gap_safe_only": safe_only_gap,
                "oracle_action_set_mismatch": not selected_in_true_candidate_set,
                "unobserved_gt_object_count": unobserved_gt_object_count,
            }
        )
        rows.append(row)
    return OracleRun(controller=controller, rows=rows)
