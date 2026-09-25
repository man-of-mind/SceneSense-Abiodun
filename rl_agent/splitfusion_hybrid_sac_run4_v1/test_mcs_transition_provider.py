"""Adversarial tests for the FIT-only arbitrary-length MCS provider."""

from __future__ import annotations

import math
import random
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path

from . import dynamic_mcs_273prb_evidence as evidence_module
from . import mcs_transition_provider as subject


ROOT = Path(__file__).resolve().parents[2]


class FitMcsMarkovProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = evidence_module.load_dynamic_mcs_273prb_evidence(
            repository_root=ROOT
        )
        cls.model = subject.fit_mcs_markov_model(cls.evidence)

    def test_fit_statistics_use_every_and_only_fit_transition(self) -> None:
        self.assertEqual(
            sum(sum(row) for row in self.model.transition_counts),
            len(self.evidence.fit_transitions),
        )
        self.assertEqual(sum(self.model.delta_counts), 416)
        self.assertEqual(sum(self.model.initial_counts), 416)
        swapped_validation = replace(
            self.evidence,
            internal_validation_sequences=tuple(
                reversed(self.evidence.internal_validation_sequences)
            ),
        )
        self.assertEqual(
            subject.fit_mcs_markov_model(swapped_validation), self.model
        )

    def test_every_row_is_a_probability_distribution(self) -> None:
        for current in range(subject.MCS_MIN, subject.MCS_MAX + 1):
            probabilities = self.model.probabilities(current)
            self.assertEqual(len(probabilities), subject.STATE_COUNT)
            self.assertTrue(all(
                math.isfinite(value) and value >= 0.0
                for value in probabilities
            ))
            self.assertAlmostEqual(sum(probabilities), 1.0, places=14)

    def test_validation_only_current_state_has_backoff_support(self) -> None:
        # MCS 12 is absent as a FIT current state but occurs in validation.
        self.assertEqual(
            sum(self.model.transition_counts[12 - subject.MCS_MIN]), 0
        )
        probabilities = self.model.probabilities(12)
        validation_successors = {
            transition.policy_values()[1]
            for transition in self.evidence.internal_validation_transitions
            if transition.policy_values()[0] == 12
        }
        self.assertTrue(validation_successors)
        self.assertTrue(all(
            probabilities[value - subject.MCS_MIN] > 0.0
            for value in validation_successors
        ))

    def test_validation_diagnostics_are_finite_and_held_out(self) -> None:
        report = subject.evaluate_internal_validation(self.model, self.evidence)
        self.assertEqual(report["fit_transitions"], 416)
        self.assertEqual(report["validation_transitions"], 176)
        self.assertTrue(math.isfinite(report["mean_negative_log_likelihood"]))
        self.assertTrue(math.isfinite(report["brier_mean"]))
        self.assertGreaterEqual(report["top1_accuracy"], 0.0)
        self.assertLessEqual(report["top1_accuracy"], 1.0)
        self.assertIn(12, report["validation_current_states"])

    def test_arbitrary_length_stays_inside_measured_support(self) -> None:
        provider = subject.FitMcsMarkovProviderV1(self.model, seed=17)
        first = provider.reset()
        self.assertGreaterEqual(first, subject.MCS_MIN)
        observed = {first}
        for _ in range(10_000):
            step = provider.step()
            self.assertEqual(step.duration_tensors, 2)
            self.assertEqual(step.evidence_class, subject.EVIDENCE_CLASS)
            self.assertGreaterEqual(step.successor_mcs, subject.MCS_MIN)
            self.assertLessEqual(step.successor_mcs, subject.MCS_MAX)
            observed.add(step.successor_mcs)
        self.assertGreater(len(observed), 1)

    def test_checkpoint_resume_is_bit_identical(self) -> None:
        source = subject.FitMcsMarkovProviderV1(self.model, seed=29)
        source.reset()
        for _ in range(137):
            source.step()
        checkpoint = source.checkpoint()
        expected = [source.step() for _ in range(500)]

        resumed = subject.FitMcsMarkovProviderV1(self.model, seed=999)
        resumed.restore(checkpoint)
        self.assertEqual([resumed.step() for _ in range(500)], expected)

    def test_foreign_checkpoint_is_refused_without_mutation(self) -> None:
        provider = subject.FitMcsMarkovProviderV1(self.model, seed=43)
        provider.reset()
        before = provider.checkpoint()
        foreign = replace(before, model_binding_sha256="0" * 64)
        with self.assertRaisesRegex(
            subject.McsTransitionProviderError, "binding"
        ):
            provider.restore(foreign)
        self.assertEqual(provider.checkpoint(), before)

    def test_step_requires_explicit_reset(self) -> None:
        provider = subject.FitMcsMarkovProviderV1(self.model, seed=17)
        with self.assertRaisesRegex(
            subject.McsTransitionProviderError, "reset"
        ):
            provider.step()

    def test_local_rng_does_not_touch_global_rng(self) -> None:
        random.seed(123456)
        before = random.getstate()
        provider = subject.FitMcsMarkovProviderV1(self.model, seed=17)
        provider.reset()
        for _ in range(100):
            provider.step()
        self.assertEqual(random.getstate(), before)

    def test_equal_seed_is_deterministic_and_different_seed_diverges(self) -> None:
        def draw(seed: int) -> list[int]:
            provider = subject.FitMcsMarkovProviderV1(self.model, seed=seed)
            values = [provider.reset()]
            values.extend(provider.step().successor_mcs for _ in range(100))
            return values

        self.assertEqual(draw(17), draw(17))
        self.assertNotEqual(draw(17), draw(29))

    def test_import_is_io_and_runtime_pure(self) -> None:
        code = r'''\
import builtins, os, socket, subprocess
def stop(*args, **kwargs): raise AssertionError("side effect")
builtins.open = stop
os.open = stop
socket.socket = stop
subprocess.Popen = stop
subprocess.run = stop
import rl_agent.splitfusion_hybrid_sac_run4_v1.mcs_transition_provider
print("PURE")
'''
        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "PURE")


if __name__ == "__main__":
    unittest.main()
