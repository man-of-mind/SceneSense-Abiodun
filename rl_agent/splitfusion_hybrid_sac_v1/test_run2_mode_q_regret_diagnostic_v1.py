"""Focused tests for the read-only Run-2 mode/q/regret diagnostic.

These use :mod:`unittest` rather than pytest because this environment has no
pytest installed, and an unrunnable test proves nothing.  Run with::

    python3 -m unittest rl_agent.splitfusion_hybrid_sac_v1.\
test_run2_mode_q_regret_diagnostic_v1 -v

Every test is intended to fail if the thing it names actually breaks.  The
identity tests deliberately perturb copies of the frozen evidence and require a
hard refusal, so a silently-rebound or silently-rescored diagnostic cannot pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from . import run2_mode_q_regret_diagnostic_v1 as diagnostic
from .run2_mode_q_regret_diagnostic_v1 import (
    MODE_COUNT,
    Run2ModeQDiagnosticError,
    average_ranks,
    decompose_context_regret,
    descending_rank,
    require_file_sha256,
    shannon_entropy_nats,
    spearman_rho,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PureHelperTests(unittest.TestCase):
    def test_entropy_of_uniform_twelve_is_log_twelve(self) -> None:
        self.assertAlmostEqual(
            shannon_entropy_nats([1.0 / 12.0] * 12), math.log(12.0), places=12
        )

    def test_entropy_of_point_mass_is_zero(self) -> None:
        self.assertEqual(shannon_entropy_nats([0.0] * 11 + [1.0]), 0.0)

    def test_entropy_rejects_unnormalized_and_negative_mass(self) -> None:
        with self.assertRaises(Run2ModeQDiagnosticError):
            shannon_entropy_nats([0.5, 0.2])
        with self.assertRaises(Run2ModeQDiagnosticError):
            shannon_entropy_nats([1.5, -0.5])

    def test_descending_rank_is_a_permutation_with_index_tie_break(self) -> None:
        self.assertEqual(descending_rank([0.1, 0.9, 0.5]), (2, 0, 1))
        self.assertEqual(descending_rank([1.0, 1.0, 0.0]), (0, 1, 2))
        self.assertEqual(sorted(descending_rank([3.0, 3.0, 3.0, 1.0])), [0, 1, 2, 3])

    def test_average_ranks_averages_ties(self) -> None:
        self.assertEqual(average_ranks([10.0, 20.0, 30.0]), (1.0, 2.0, 3.0))
        self.assertEqual(average_ranks([5.0, 5.0, 9.0]), (1.5, 1.5, 3.0))

    def test_spearman_rho_known_values(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(spearman_rho(values, values), 1.0, places=12)
        self.assertAlmostEqual(
            spearman_rho(values, list(reversed(values))), -1.0, places=12
        )
        self.assertTrue(math.isnan(spearman_rho(values, [7.0] * 4)))

    def test_spearman_rho_rejects_bad_shapes(self) -> None:
        with self.assertRaises(Run2ModeQDiagnosticError):
            spearman_rho([1.0, 2.0], [1.0])
        with self.assertRaises(Run2ModeQDiagnosticError):
            spearman_rho([1.0], [1.0])


class RegretDecompositionTests(unittest.TestCase):
    def test_decomposition_is_exactly_additive(self) -> None:
        result = decompose_context_regret(
            utility_a=0.30,
            utility_b=0.35,
            utility_c=0.33,
            utility_d=0.42,
            utility_e=0.32,
        )
        self.assertAlmostEqual(result.continuous_q_regret, 0.05, places=12)
        self.assertAlmostEqual(result.mode_given_best_q_regret, 0.07, places=12)
        self.assertAlmostEqual(result.total_oracle_regret, 0.12, places=12)
        self.assertAlmostEqual(
            result.continuous_q_regret + result.mode_given_best_q_regret,
            result.total_oracle_regret,
            places=12,
        )
        # The overlapping term is reported but never summed with the two above.
        self.assertAlmostEqual(result.mode_at_actor_q_regret, 0.03, places=12)
        self.assertAlmostEqual(result.learned_minus_fixed, -0.02, places=12)
        result.require_additive()

    def test_decomposition_refuses_dominance_violations(self) -> None:
        with self.assertRaises(Run2ModeQDiagnosticError):
            decompose_context_regret(
                utility_a=0.4, utility_b=0.3, utility_c=0.3,
                utility_d=0.5, utility_e=0.1,
            )
        with self.assertRaises(Run2ModeQDiagnosticError):
            decompose_context_regret(
                utility_a=0.1, utility_b=0.4, utility_c=0.2,
                utility_d=0.3, utility_e=0.1,
            )
        with self.assertRaises(Run2ModeQDiagnosticError):
            decompose_context_regret(
                utility_a=0.1, utility_b=0.2, utility_c=0.9,
                utility_d=0.3, utility_e=0.1,
            )

    def test_require_additive_detects_a_tampered_decomposition(self) -> None:
        tampered = diagnostic.RegretDecomposition(
            utility_a=0.1,
            utility_b=0.2,
            utility_c=0.2,
            utility_d=0.5,
            utility_e=0.1,
            continuous_q_regret=0.1,
            mode_given_best_q_regret=0.1,   # should be 0.3
            total_oracle_regret=0.4,
            mode_at_actor_q_regret=0.1,
            learned_minus_fixed=0.0,
        )
        with self.assertRaises(Run2ModeQDiagnosticError):
            tampered.require_additive()


class HashAndIdentityRefusalTests(unittest.TestCase):
    def test_require_file_sha256_accepts_the_true_hash(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "payload.bin"
            path.write_bytes(b"split-fusion")
            expected = hashlib.sha256(b"split-fusion").hexdigest()
            self.assertEqual(require_file_sha256(path, expected, "payload"), expected)

    def test_require_file_sha256_refuses_drift_and_absence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "payload.bin"
            path.write_bytes(b"split-fusion")
            with self.assertRaisesRegex(Run2ModeQDiagnosticError, "drift"):
                require_file_sha256(path, "0" * 64, "payload")
            with self.assertRaisesRegex(Run2ModeQDiagnosticError, "missing"):
                require_file_sha256(Path(raw) / "absent.bin", "0" * 64, "absent")

    def test_revalidate_refuses_a_root_without_the_frozen_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(Exception):
                diagnostic.revalidate_frozen_bindings(Path(raw))

    def test_manifest_content_hash_drift_is_refused(self) -> None:
        source = PROJECT_ROOT / diagnostic.EVALUATION_RELATIVE_PATH / "manifest.json"
        document = json.loads(source.read_text(encoding="utf-8"))
        document["fit_partition_sha256"] = "1" * 64  # body changed, self-hash stale
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / diagnostic.EVALUATION_RELATIVE_PATH
            target.mkdir(parents=True)
            (target / "manifest.json").write_text(
                json.dumps(document), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                Run2ModeQDiagnosticError, "manifest content hash mismatch"
            ):
                diagnostic.revalidate_frozen_bindings(Path(raw))

    def test_artifact_hash_drift_inside_a_valid_manifest_is_refused(self) -> None:
        """A self-consistent manifest must still catch a mutated artifact."""
        evaluation = PROJECT_ROOT / diagnostic.EVALUATION_RELATIVE_PATH
        document = json.loads((evaluation / "manifest.json").read_text("utf-8"))
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / diagnostic.EVALUATION_RELATIVE_PATH
            target.mkdir(parents=True)
            (target / "manifest.json").write_text(
                json.dumps(document), encoding="utf-8"
            )
            # Copy every artifact faithfully except one, which is corrupted.
            for name in document["artifacts"]:
                payload = (evaluation / name).read_bytes()
                if name == "protocol.json":
                    payload = payload + b"\n"
                (target / name).write_bytes(payload)
            with self.assertRaisesRegex(Run2ModeQDiagnosticError, "drift"):
                diagnostic.revalidate_frozen_bindings(Path(raw))

    def test_emitters_refuse_empty_and_non_finite_payloads(self) -> None:
        with self.assertRaises(Run2ModeQDiagnosticError):
            diagnostic._csv_bytes([])
        with self.assertRaises(Run2ModeQDiagnosticError):
            diagnostic._finite(float("nan"))
        with self.assertRaises(Run2ModeQDiagnosticError):
            diagnostic._finite(float("inf"))
        with self.assertRaises(ValueError):
            diagnostic._json_bytes({"value": float("nan")})


class FrozenEvidenceReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = diagnostic.load_frozen_per_context(PROJECT_ROOT)

    def test_frozen_rows_reproduce_every_registered_anchor(self) -> None:
        anchors = diagnostic.reproduce_anchors(self.rows)
        for name, expected in diagnostic.EXPECTED_ANCHORS.items():
            self.assertAlmostEqual(
                anchors[name], expected, delta=diagnostic.ANCHOR_TOLERANCE
            )
        self.assertEqual(anchors["run2_final_p95_miss_count"], 23.0)
        self.assertAlmostEqual(
            anchors["run2_final_p95_miss_rate"], 23.0 / 1020.0, places=12
        )

    def test_a_single_perturbed_reward_breaks_anchor_reproduction(self) -> None:
        mutated = [dict(row) for row in self.rows]
        target = next(
            row
            for row in mutated
            if row["source_kind"] == diagnostic.SOURCE_V2
            and int(row["update_index"]) == diagnostic.PRIMARY_UPDATE
        )
        target["v2_emitted_reward_float32"] = str(
            float(target["v2_emitted_reward_float32"]) + 1.0
        )
        with self.assertRaisesRegex(Run2ModeQDiagnosticError, "did not reproduce"):
            diagnostic.reproduce_anchors(mutated)

    def test_a_flipped_final_mode_breaks_the_mode_population_check(self) -> None:
        mutated = [dict(row) for row in self.rows]
        target = next(
            row
            for row in mutated
            if row["source_kind"] == diagnostic.SOURCE_V2
            and int(row["update_index"]) == diagnostic.PRIMARY_UPDATE
        )
        target["executed_mode_id"] = "8"
        with self.assertRaisesRegex(
            Run2ModeQDiagnosticError, "mode population drift"
        ):
            diagnostic.reproduce_anchors(mutated)

    def test_row_count_drift_is_refused(self) -> None:
        source = PROJECT_ROOT / diagnostic.EVALUATION_RELATIVE_PATH / "per_context.csv"
        text = source.read_text(encoding="utf-8").splitlines(keepends=True)
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / diagnostic.EVALUATION_RELATIVE_PATH
            target.mkdir(parents=True)
            (target / "per_context.csv").write_text(
                "".join(text[:-1]), encoding="utf-8"
            )
            with self.assertRaisesRegex(Run2ModeQDiagnosticError, "row count drift"):
                diagnostic.load_frozen_per_context(Path(raw))


class TieRuleTests(unittest.TestCase):
    def test_outcome_better_implements_the_registered_tie_rule(self) -> None:
        better = {"utility": 0.5, "p95_ms": 180.0, "mode_id": 3, "q_e4": 100}
        worse = {"utility": 0.4, "p95_ms": 100.0, "mode_id": 1, "q_e4": 50}
        self.assertTrue(diagnostic._outcome_better(better, worse))
        tie_low_latency = {"utility": 0.5, "p95_ms": 150.0, "mode_id": 9, "q_e4": 900}
        self.assertTrue(diagnostic._outcome_better(tie_low_latency, better))
        self.assertFalse(diagnostic._outcome_better(better, tie_low_latency))
        same = {"utility": 0.5, "p95_ms": 180.0, "mode_id": 2, "q_e4": 100}
        self.assertTrue(diagnostic._outcome_better(same, better))


class ExactActionTableTests(unittest.TestCase):
    """Live D1 checks: the vector tables must equal the frozen scalar path."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluator = diagnostic.FitValidationActorEvaluatorV1(
            project_root=PROJECT_ROOT
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.evaluator.close()

    def test_vector_table_matches_the_frozen_scalar_scorer(self) -> None:
        table = diagnostic.build_context_table(
            self.evaluator, self.evaluator.panel.entries[0]
        )
        for mode_id in range(MODE_COUNT):
            lower, upper = diagnostic.MODELED_SMOKE_MODE_Q_E4_BOUNDS[mode_id]
            for q_e4 in (lower, (lower + upper) // 2, upper):
                diagnostic.require_table_matches_scalar_scorer(table, mode_id, q_e4)

    def test_table_reproduces_the_frozen_constrained_oracle(self) -> None:
        oracle_actions = diagnostic._load_oracle_actions(PROJECT_ROOT)
        for entry in self.evaluator.panel.entries[:8]:
            table = diagnostic.build_context_table(self.evaluator, entry)
            expected = oracle_actions[int(entry.panel_index)]
            best = table.best_across_modes(feasible_only=True)
            self.assertEqual((int(best["mode_id"]), int(best["q_e4"])), expected)

    def test_best_within_mode_never_beats_best_across_modes(self) -> None:
        table = diagnostic.build_context_table(
            self.evaluator, self.evaluator.panel.entries[3]
        )
        overall = table.best_across_modes()
        for mode_id in range(MODE_COUNT):
            candidate = table.best_within_mode(mode_id)
            if candidate is not None:
                self.assertLessEqual(candidate["utility"], overall["utility"])

    def test_table_rejects_an_out_of_support_q(self) -> None:
        table = diagnostic.build_context_table(
            self.evaluator, self.evaluator.panel.entries[0]
        )
        lower, _upper = diagnostic.MODELED_SMOKE_MODE_Q_E4_BOUNDS[0]
        with self.assertRaisesRegex(Run2ModeQDiagnosticError, "outside mode"):
            table.index_of(0, lower - 1)

    def test_diagnostic_does_not_initialize_cuda(self) -> None:
        diagnostic.build_context_table(
            self.evaluator, self.evaluator.panel.entries[1]
        )
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
