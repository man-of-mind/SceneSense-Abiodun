"""Focused tests for the exact Run-2 pre-training validation gate."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_hybrid_sac_v1.empirical_contextual_exact_p95_deadline_penalty import (
    BUDGET_MS,
    base_p95_expected_utility,
    shaped_p95_expected_utility,
)
from rl_agent.splitfusion_hybrid_sac_v1.empirical_contextual_exact_p95_run2_validation_gate import (
    EXPECTED_VALIDATION_CONTEXT_COUNT,
    RUN2_DEADLINE_PENALTY,
    RUN2_PREREGISTRATION_SHA256,
    Run2ValidationGateError,
    _decision,
    _profile_rows,
    _project_root,
    _require_frozen_preregistration,
    select_fixed_action_from_train_sums,
    validate_frozen_fixed_selection,
)
from rl_agent.splitfusion_hybrid_sac_v1.empirical_contextual_split_oracle import (
    EXACT_ACTION_COUNT_PER_SCENE,
    enumerate_supported_actions,
)
from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import (
    MODELED_SMOKE_MODE_Q_E4_BOUNDS,
)


def _zero_sums() -> list[np.ndarray]:
    return [
        np.zeros(upper - lower + 1, dtype=np.float64)
        for lower, upper in MODELED_SMOKE_MODE_Q_E4_BOUNDS
    ]


class ExactP95Run2ValidationGateTest(unittest.TestCase):
    def test_frozen_preregistration_hash_and_contract(self) -> None:
        path, document = _require_frozen_preregistration(_project_root())
        self.assertTrue(path.is_file())
        self.assertEqual(
            document["reward"]["deadline_penalty"], RUN2_DEADLINE_PENALTY
        )
        self.assertEqual(document["reward"]["deadline_ms"], BUDGET_MS)
        self.assertEqual(len(RUN2_PREREGISTRATION_SHA256), 64)

    def test_complete_executable_action_quotient(self) -> None:
        actions = enumerate_supported_actions()
        self.assertEqual(len(actions), EXACT_ACTION_COUNT_PER_SCENE)
        self.assertEqual(len(actions), 52240)
        self.assertEqual(len(actions), len(set(actions)))

    def test_fixed_selector_uses_minimum_mode_and_q_for_global_tie(self) -> None:
        sums = _zero_sums()
        self.assertEqual(
            select_fixed_action_from_train_sums(sums, 4),
            (0, MODELED_SMOKE_MODE_Q_E4_BOUNDS[0][0], 0.0),
        )

    def test_fixed_selector_finds_unique_maximum(self) -> None:
        sums = _zero_sums()
        lower, _upper = MODELED_SMOKE_MODE_Q_E4_BOUNDS[7]
        sums[7][11] = 8.0
        self.assertEqual(
            select_fixed_action_from_train_sums(sums, 4),
            (7, lower + 11, 2.0),
        )

    def test_fixed_selector_rejects_wrong_shape_and_nonfinite(self) -> None:
        sums = _zero_sums()
        sums[0] = sums[0][:-1]
        with self.assertRaises(ValueError):
            select_fixed_action_from_train_sums(sums, 1)
        sums = _zero_sums()
        sums[1][0] = math.nan
        with self.assertRaises(ValueError):
            select_fixed_action_from_train_sums(sums, 1)

    def test_frozen_fixed_selection_digest_detects_tampering(self) -> None:
        from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
            canonical_sha256,
        )

        selection = {"action": {"mode_id": 11, "q_e4": 6000}}
        selection["frozen_selection_sha256"] = canonical_sha256(selection)
        self.assertEqual(
            validate_frozen_fixed_selection(selection),
            selection["frozen_selection_sha256"],
        )
        selection["action"] = {"mode_id": 11, "q_e4": 6001}
        with self.assertRaises(Run2ValidationGateError):
            validate_frozen_fixed_selection(selection)

    def test_deadline_boundary_is_unpenalized(self) -> None:
        arguments = dict(p_admit=0.93, q_perc=0.71)
        base = base_p95_expected_utility(
            **arguments, latency_p95_ms=BUDGET_MS
        )
        shaped = shaped_p95_expected_utility(
            **arguments,
            latency_p95_ms=BUDGET_MS,
            deadline_penalty=RUN2_DEADLINE_PENALTY,
        )
        self.assertEqual(shaped, base)

    def test_smooth_latency_order_remains_above_deadline(self) -> None:
        arguments = dict(
            p_admit=0.93,
            q_perc=0.71,
            deadline_penalty=RUN2_DEADLINE_PENALTY,
        )
        at_201 = shaped_p95_expected_utility(
            **arguments, latency_p95_ms=201.0
        )
        at_300 = shaped_p95_expected_utility(
            **arguments, latency_p95_ms=300.0
        )
        self.assertGreater(at_201, at_300)
        self.assertAlmostEqual(
            at_201 - at_300,
            0.93 * 0.25 * (300.0 - 201.0) / 200.0,
            places=15,
        )

    def test_decision_requires_every_preregistered_gate(self) -> None:
        complete = {
            "feasible_context_count": EXPECTED_VALIDATION_CONTEXT_COUNT,
            "shaped_constrained_identity_match_count": (
                EXPECTED_VALIDATION_CONTEXT_COUNT
            ),
            "shaped_oracle_p95_miss_count": 0,
            "constrained_scalar_revalidation_count": (
                EXPECTED_VALIDATION_CONTEXT_COUNT
            ),
            "shaped_scalar_revalidation_count": EXPECTED_VALIDATION_CONTEXT_COUNT,
        }
        self.assertEqual(_decision(complete)["status"], "GO")
        for key in (
            "feasible_context_count",
            "shaped_constrained_identity_match_count",
            "constrained_scalar_revalidation_count",
            "shaped_scalar_revalidation_count",
        ):
            changed = dict(complete)
            changed[key] -= 1
            self.assertEqual(_decision(changed)["status"], "NO_GO", key)
        changed = dict(complete)
        changed["shaped_oracle_p95_miss_count"] = 1
        self.assertEqual(_decision(changed)["status"], "NO_GO")

    def test_profile_summary_does_not_hide_fixed_or_shaped_misses(self) -> None:
        rows = []
        for index, profile in enumerate(
            ("FAVORABLE_STABLE", "MID_VARIABLE")
        ):
            rows.append(
                {
                    "network_profile": profile,
                    "has_p95_feasible_action": index == 0,
                    "shaped_p95_miss": index == 1,
                    "shaped_matches_constrained": index == 0,
                    "constrained_q_perc": 0.7 if index == 0 else "",
                    "constrained_p_admit": 0.99 if index == 0 else "",
                    "shaped_q_perc": 0.7,
                    "shaped_p_admit": 0.99,
                    "fixed_p95_miss": index == 1,
                    "fixed_q_perc": 0.6,
                    "fixed_p_admit": 0.98,
                    "fixed_shaped_reward": 0.2,
                }
            )
        summary = _profile_rows(rows)
        overall = summary[-1]
        self.assertEqual(overall["context_count"], 2)
        self.assertEqual(overall["feasible_context_count"], 1)
        self.assertEqual(overall["shaped_p95_miss_count"], 1)
        self.assertEqual(overall["fixed_p95_miss_count"], 1)

    def test_project_root_contains_preregistered_sources(self) -> None:
        root = _project_root()
        self.assertIsInstance(root, Path)
        self.assertTrue(
            (
                root
                / "rl_agent/splitfusion_hybrid_sac_v1/"
                "empirical_contextual_exact_p95_deadline_penalty.py"
            ).is_file()
        )


if __name__ == "__main__":
    unittest.main()
