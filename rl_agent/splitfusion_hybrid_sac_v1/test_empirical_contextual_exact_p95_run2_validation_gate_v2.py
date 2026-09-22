"""Focused adversarial tests for the one-shot emitted-float32 v2 gate."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from .empirical_contextual_exact_p95_deadline_penalty_v2 import (
    emitted_float32_target_v2,
)
from .empirical_contextual_exact_p95_run2_validation_gate_v2 import (
    EXPECTED_VALIDATION_ACTION_CONTEXT_EVALUATIONS,
    EXPECTED_VALIDATION_CONTEXT_COUNT,
    PREREGISTRATION_RELATIVE_PATH,
    RUNTIME_ARITHMETIC,
    RUN2_V2_DEADLINE_PENALTY,
    RUN2_V2_PREDECESSOR,
    RUN2_V2_PREREGISTRATION_SHA256,
    RUN2_V2_TRAIN_SUMMARY_SHA256,
    TRAIN_SUMMARY_RELATIVE_PATH,
    Run2V2ValidationGateError,
    _decision_verbatim,
    _open_validation_evaluator,
    _project_root,
    _require_disjoint_scene_ids,
    _require_file_sha256,
    _require_frozen_contracts,
    _require_runtime_reward_contract,
)
from .transaction_identity import canonical_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    PROJECT_ROOT
    / "experiments/splitfusion_hybrid_sac_fit_validation_v1"
    / "20260921_exact_p95_run2_pretraining_validation_gate_v2"
)


class ExactP95Run2ValidationGateV2UnitTest(unittest.TestCase):
    def test_frozen_preregistration_and_train_summary_hashes(self) -> None:
        root = _project_root()
        frozen = _require_frozen_contracts(root)
        self.assertEqual(
            hashlib.sha256(
                (root / PREREGISTRATION_RELATIVE_PATH).read_bytes()
            ).hexdigest(),
            RUN2_V2_PREREGISTRATION_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(
                (root / TRAIN_SUMMARY_RELATIVE_PATH).read_bytes()
            ).hexdigest(),
            RUN2_V2_TRAIN_SUMMARY_SHA256,
        )
        frozen.require_current()

    def test_wrong_lambda_is_rejected(self) -> None:
        with self.assertRaises(Run2V2ValidationGateError):
            _require_runtime_reward_contract(
                deadline_penalty=math.nextafter(
                    RUN2_V2_DEADLINE_PENALTY, math.inf
                ),
                arithmetic=RUNTIME_ARITHMETIC,
            )

    def test_float32_first_arithmetic_is_detectably_different_and_rejected(self) -> None:
        p = 0.9919169403402207
        quality = 0.7106802277300279
        latency = 214.58358472906113
        base64 = p * (quality - 0.25 * (latency / 200.0)) + (1.0 - p) * (-1.0)
        exact = emitted_float32_target_v2(
            base64 - p * RUN2_V2_DEADLINE_PENALTY
        )
        p32 = torch.tensor(p, dtype=torch.float32)
        q32 = torch.tensor(quality, dtype=torch.float32)
        latency32 = torch.tensor(latency, dtype=torch.float32)
        lambda32 = torch.tensor(RUN2_V2_DEADLINE_PENALTY, dtype=torch.float32)
        float32_first_base = p32 * (
            q32
            - torch.tensor(0.25, dtype=torch.float32)
            * (latency32 / torch.tensor(200.0, dtype=torch.float32))
        ) + (torch.tensor(1.0, dtype=torch.float32) - p32) * torch.tensor(
            -1.0, dtype=torch.float32
        )
        float32_first = float((float32_first_base - p32 * lambda32).item())
        self.assertNotEqual(exact, float32_first)
        with self.assertRaises(Run2V2ValidationGateError):
            _require_runtime_reward_contract(
                deadline_penalty=RUN2_V2_DEADLINE_PENALTY,
                arithmetic="FLOAT32_FIRST",
            )

    def test_train_validation_overlap_is_rejected(self) -> None:
        with self.assertRaises(Run2V2ValidationGateError):
            _require_disjoint_scene_ids(
                frozenset({"train", "collision"}),
                frozenset({"validation", "collision"}),
            )
        _require_disjoint_scene_ids(
            frozenset({"train"}), frozenset({"validation"})
        )

    def test_hash_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            path.write_text("{}\n", encoding="utf-8")
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(_require_file_sha256(path, observed, "test"), observed)
            path.write_text('{"drift":true}\n', encoding="utf-8")
            with self.assertRaises(Run2V2ValidationGateError):
                _require_file_sha256(path, observed, "test")

    def test_predecessor_failure_and_selected_strict_ordering(self) -> None:
        target = emitted_float32_target_v2(-0.13886236214769926)
        p = 0.9919169403402207
        infeasible_base64 = 0.4307913313758727
        predecessor = emitted_float32_target_v2(
            infeasible_base64 - p * RUN2_V2_PREDECESSOR
        )
        selected = emitted_float32_target_v2(
            infeasible_base64 - p * RUN2_V2_DEADLINE_PENALTY
        )
        self.assertGreaterEqual(predecessor, target)
        self.assertLess(selected, target)
        self.assertEqual(
            RUN2_V2_PREDECESSOR,
            math.nextafter(RUN2_V2_DEADLINE_PENALTY, -math.inf),
        )

    def test_validation_evaluator_cannot_open_before_freeze_check(self) -> None:
        module = (
            "rl_agent.splitfusion_hybrid_sac_v1."
            "empirical_contextual_exact_p95_run2_validation_gate_v2."
            "FitValidationActorEvaluatorV1"
        )
        with mock.patch(module) as evaluator:
            with self.assertRaises(Run2V2ValidationGateError):
                with _open_validation_evaluator(
                    root=_project_root(), frozen=None
                ):
                    pass
            evaluator.assert_not_called()

    def test_decision_uses_preregistered_gates_verbatim(self) -> None:
        prereg = _require_frozen_contracts(_project_root()).preregistration
        complete = {
            "positive_admission_feasible_context_count": 340,
            "shaped_constrained_identity_match_count": 340,
            "shaped_oracle_p95_miss_count": 0,
            "unconstrained_scalar_revalidation_count": 340,
            "constrained_scalar_revalidation_count": 340,
            "shaped_scalar_revalidation_count": 340,
        }
        decision = _decision_verbatim(complete, prereg)
        self.assertEqual(decision["status"], "GO")
        self.assertEqual(
            tuple(decision["criteria"]),
            (
                "every_fit_validation_context_has_at_least_one_p95_feasible_action",
                "frozen_lambda_shaped_oracle_must_equal_emitted_float32_p95_constrained_oracle",
                "frozen_lambda_shaped_oracle_p95_miss_count",
                "actual_emitted_float32_targets_must_be_used",
            ),
        )
        for key in (
            "positive_admission_feasible_context_count",
            "shaped_constrained_identity_match_count",
            "unconstrained_scalar_revalidation_count",
            "constrained_scalar_revalidation_count",
            "shaped_scalar_revalidation_count",
        ):
            changed = dict(complete)
            changed[key] -= 1
            self.assertEqual(_decision_verbatim(changed, prereg)["status"], "NO_GO")
        changed = dict(complete)
        changed["shaped_oracle_p95_miss_count"] = 1
        self.assertEqual(_decision_verbatim(changed, prereg)["status"], "NO_GO")


@unittest.skipUnless(EVIDENCE.exists(), "v2 one-shot validation not yet run")
class ExactP95Run2ValidationGateV2EvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((EVIDENCE / "summary_v2.json").read_text())
        cls.decision = json.loads((EVIDENCE / "GO_NO_GO_v2.json").read_text())

    def test_exact_population_and_exhaustive_coverage(self) -> None:
        metrics = self.summary["validation_gate"]
        self.assertEqual(metrics["context_count"], EXPECTED_VALIDATION_CONTEXT_COUNT)
        self.assertEqual(metrics["validation_scene_count"], 85)
        self.assertEqual(
            metrics["exhaustive_action_context_evaluations"],
            EXPECTED_VALIDATION_ACTION_CONTEXT_EVALUATIONS,
        )
        self.assertEqual(metrics["train_validation_scene_id_intersection_count"], 0)

    def test_gate_is_fail_closed_and_verbatim(self) -> None:
        expected = (
            "GO"
            if all(self.summary["decision"]["criteria"].values())
            else "NO_GO"
        )
        self.assertEqual(self.summary["decision"]["status"], expected)
        self.assertEqual(self.decision["status"], expected)
        prereg = _require_frozen_contracts(_project_root()).preregistration
        self.assertEqual(
            self.summary["decision"]["failure_action"],
            prereg["pretraining_go_no_go"]["failure_action"],
        )

    def test_winners_are_scalar_revalidated_at_emitted_dtype(self) -> None:
        metrics = self.summary["validation_gate"]
        self.assertEqual(metrics["unconstrained_scalar_revalidation_count"], 340)
        self.assertEqual(metrics["constrained_scalar_revalidation_count"], 340)
        self.assertEqual(metrics["shaped_scalar_revalidation_count"], 340)
        self.assertEqual(
            self.summary["reward"]["emitted_target"],
            "torch.tensor(shaped64,dtype=torch.float32).item()",
        )

    def test_quality_and_admission_comparison_is_reported(self) -> None:
        comparison = self.summary["quality_admission_vs_unconstrained"]
        self.assertIsNotNone(
            comparison["constrained"]["quality_retention_vs_unconstrained"]
        )
        self.assertIsNotNone(
            comparison["constrained"]["admission_change_vs_unconstrained"]
        )

    def test_frozen_source_hashes_and_scope_reconcile(self) -> None:
        bindings = self.summary["bindings"]
        self.assertEqual(
            bindings["run2_v2_preregistration_sha256"],
            RUN2_V2_PREREGISTRATION_SHA256,
        )
        self.assertEqual(
            bindings["train_exact_penalty_v2_summary_sha256"],
            RUN2_V2_TRAIN_SUMMARY_SHA256,
        )
        scope = self.summary["scope"]
        self.assertTrue(scope["preregistration_verified_before_validation_access"])
        self.assertEqual(scope["validation_retry_count"], 0)
        self.assertEqual(scope["policy_training_count"], 0)
        self.assertEqual(scope["replay_change_count"], 0)

    def test_artifact_hashes_reconcile(self) -> None:
        for name, expected in self.summary["files"].items():
            self.assertEqual(
                hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest(),
                expected,
            )
        document = dict(self.summary)
        observed = document.pop("canonical_content_sha256")
        self.assertEqual(canonical_sha256(document), observed)


if __name__ == "__main__":
    unittest.main()
