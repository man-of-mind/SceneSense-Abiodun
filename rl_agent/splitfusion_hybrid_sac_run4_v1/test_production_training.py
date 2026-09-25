"""Adversarial tests for the fail-closed Run-4 production schedule."""

from __future__ import annotations

import importlib
import random
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import canonical_sha256

from . import persistent_runner, production_training as src


def _d(label: str) -> str:
    return canonical_sha256({"production-training-test": label})


def _prerequisites() -> persistent_runner.CompositeRunnerPrerequisitesV1:
    return persistent_runner.CompositeRunnerPrerequisitesV1(
        fit_scene_provider_binding_sha256=_d("fit"),
        state_provider_binding_sha256=_d("state"),
        kernel_prerequisites_sha256=_d("kernel-prerequisites"),
        kernel_support_sha256=_d("kernel-support"),
        state_stager_binding_sha256=_d("stager"),
        prediction_provider_binding_sha256=_d("prediction"),
        replay_binding_sha256=_d("replay"),
        replay_capacity=65_536,
        model_binding_sha256=_d("models"),
        trainer_config_sha256=_d("trainer"),
        warmup_schedule_id=_d("warmup"),
        exploration_gate_config_sha256=_d("exploration"),
        verifier_manifest_sha256=_d("manifest"),
        near_capacity_kernel_validated=True,
        scaling_and_freshness_validated=True,
        prior_outcome_chain_validated=True,
    )


class _FakeRunner:
    """Schedule-only harness; never accepted by a public production API."""

    def __init__(self, *, updates: int = 0, decisions: int = 0, started=False):
        self.trainer = SimpleNamespace(update_count=updates)
        self._decision_count = decisions
        self._started = started
        self.start_calls = []
        self.step_calls = 0
        self.train_batch_sizes = []
        self.checkpoint_calls = 0
        self.gradient_gate_calls = 0

    @property
    def started(self):
        return self._started

    @property
    def decision_count(self):
        return self._decision_count

    def start(self, *, session_uuid: str, ue_id: str) -> None:
        if self._started:
            raise AssertionError("duplicate start")
        self._started = True
        self.start_calls.append((session_uuid, ue_id))

    def step(self):
        if not self._started:
            raise AssertionError("step before start")
        self.step_calls += 1
        self._decision_count += 1
        return ("transition", self._decision_count)

    def train_once(self, batch_size: int):
        schedule = src.FROZEN_PRODUCTION_SCHEDULE
        expected_decisions = schedule.expected_decisions(
            self.trainer.update_count + 1
        )
        if self._decision_count != expected_decisions:
            raise AssertionError("gradient before four new transitions")
        self.train_batch_sizes.append(batch_size)
        self.trainer.update_count += 1
        return ("metric", self.trainer.update_count)

    def require_gradient_start(self):
        schedule = src.FROZEN_PRODUCTION_SCHEDULE
        if self._decision_count < schedule.warmup_decisions:
            raise AssertionError("gradient gate checked before warm-up")
        self.gradient_gate_calls += 1
        return ("coverage", self._decision_count)

    def checkpoint(self):
        self.checkpoint_calls += 1
        return (self.trainer.update_count, self._decision_count)


class ProductionTrainingTest(unittest.TestCase):
    def test_schedule_is_exact_projection_of_preregistration(self) -> None:
        schedule = src.FROZEN_PRODUCTION_SCHEDULE
        self.assertEqual(schedule.seed, 17)
        self.assertEqual(schedule.warmup_decisions, 288)
        self.assertEqual(schedule.transitions_per_update, 4)
        self.assertEqual(schedule.batch_size, 256)
        self.assertEqual(schedule.checkpoint_updates, (0, 100, 250, 500))
        self.assertEqual(schedule.hard_stop_update, 500)
        self.assertEqual(schedule.expected_decisions(500), 2288)

    def test_new_schedule_is_exact_four_to_one_and_emits_all_checkpoints(self) -> None:
        runner = _FakeRunner(started=True)
        events = []
        boundaries = []
        summary = src._drive_schedule(
            runner,
            schedule=src.FROZEN_PRODUCTION_SCHEDULE,
            emit_checkpoint=lambda update, current, metric: events.append(
                (update, current.decision_count, metric)
            ),
            before_transition=lambda index: boundaries.append(index),
            emit_initial_checkpoint=True,
        )
        self.assertEqual(runner.step_calls, 288 + 4 * 500)
        self.assertEqual(boundaries, list(range(2288)))
        self.assertEqual(len(runner.train_batch_sizes), 500)
        self.assertEqual(runner.gradient_gate_calls, 1)
        self.assertEqual(set(runner.train_batch_sizes), {256})
        self.assertEqual([row[0] for row in events], [0, 100, 250, 500])
        self.assertEqual(
            [row[1] for row in events],
            [288, 688, 1288, 2288],
        )
        self.assertIsNone(events[0][2])
        self.assertEqual(events[-1][2], ("metric", 500))
        self.assertEqual(summary.starting_update, 0)
        self.assertEqual(summary.final_update, 500)
        self.assertEqual(summary.final_decision_count, 2288)
        self.assertEqual(summary.emitted_checkpoint_updates, (0, 100, 250, 500))

    def test_resume_continues_after_checkpoint_without_reemitting_it(self) -> None:
        runner = _FakeRunner(updates=250, decisions=1288, started=True)
        events = []
        boundaries = []
        summary = src._drive_schedule(
            runner,
            schedule=src.FROZEN_PRODUCTION_SCHEDULE,
            emit_checkpoint=lambda update, current, metric: events.append(
                (update, current.decision_count)
            ),
            before_transition=lambda index: boundaries.append(index),
            emit_initial_checkpoint=False,
        )
        self.assertEqual(runner.step_calls, 4 * 250)
        self.assertEqual(boundaries, list(range(1288, 2288)))
        self.assertEqual(len(runner.train_batch_sizes), 250)
        self.assertEqual(runner.gradient_gate_calls, 1)
        self.assertEqual(events, [(500, 2288)])
        self.assertEqual(summary.starting_update, 250)
        self.assertEqual(summary.emitted_checkpoint_updates, (500,))

    def test_partial_collection_cannot_be_mislabeled_as_resume_boundary(self) -> None:
        runner = _FakeRunner(updates=100, decisions=689, started=True)
        with self.assertRaisesRegex(
            src.ProductionScheduleError, "exact update/checkpoint boundary"
        ):
            src._drive_schedule(
                runner,
                schedule=src.FROZEN_PRODUCTION_SCHEDULE,
                emit_checkpoint=lambda *_: None,
                before_transition=lambda *_: None,
                emit_initial_checkpoint=False,
            )
        self.assertEqual(runner.step_calls, 0)
        self.assertEqual(runner.train_batch_sizes, [])

    def test_hard_stop_is_idempotent_and_performs_no_extra_work(self) -> None:
        runner = _FakeRunner(updates=500, decisions=2288, started=True)
        events = []
        summary = src._drive_schedule(
            runner,
            schedule=src.FROZEN_PRODUCTION_SCHEDULE,
            emit_checkpoint=lambda *args: events.append(args),
            before_transition=lambda *_: None,
            emit_initial_checkpoint=False,
        )
        self.assertEqual(summary.final_update, 500)
        self.assertEqual(events, [])
        self.assertEqual(runner.step_calls, 0)
        self.assertEqual(runner.train_batch_sizes, [])

    def test_schedule_driver_does_not_touch_global_rng_or_cuda(self) -> None:
        runner = _FakeRunner(started=True)
        torch_before = torch.random.get_rng_state().clone()
        python_before = random.getstate()
        cuda_before = torch.cuda.is_initialized()
        src._drive_schedule(
            runner,
            schedule=src.FROZEN_PRODUCTION_SCHEDULE,
            emit_checkpoint=lambda *_: None,
            before_transition=lambda *_: None,
            emit_initial_checkpoint=True,
        )
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_before))
        self.assertEqual(random.getstate(), python_before)
        self.assertEqual(torch.cuda.is_initialized(), cuda_before)

    def test_factory_refuses_wrong_seed_before_build(self) -> None:
        built = []
        with self.assertRaisesRegex(src.ProductionFactoryError, "seed 17"):
            src.ProductionRunnerFactoryV1(
                training_seed=29,
                prerequisites_sha256=_d("prerequisites"),
                verifier_manifest_sha256=_d("manifest"),
                build=lambda: built.append(True),
            )
        self.assertEqual(built, [])

    def test_factory_never_accepts_synthetic_or_foreign_runner(self) -> None:
        foreign = _FakeRunner()
        prerequisites = _prerequisites()
        factory = src.ProductionRunnerFactoryV1(
            training_seed=17,
            prerequisites_sha256=prerequisites.canonical_sha256,
            verifier_manifest_sha256=prerequisites.verifier_manifest_sha256,
            build=lambda: foreign,
        )
        with self.assertRaisesRegex(
            src.ProductionFactoryError, "no synthetic or test runner fallback"
        ):
            factory.build_runner()
        self.assertFalse(foreign.started)
        self.assertEqual(foreign.decision_count, 0)

    def test_composite_verifier_none_blocks_production_authorization(self) -> None:
        prerequisites = _prerequisites()
        with mock.patch.object(
            persistent_runner,
            "REGISTERED_COMPOSITE_PREREQUISITES_SHA256",
            None,
        ):
            with self.assertRaisesRegex(
                persistent_runner.RunnerAuthorizationError,
                "no reviewed composite",
            ):
                persistent_runner.verify_composite_prerequisites(
                    prerequisites
                )

    def test_even_private_training_token_cannot_fallback_to_synthetic(self) -> None:
        prerequisites = _prerequisites()
        private_token = persistent_runner._issue_authorization(
            prerequisites,
            persistent_runner.RunnerAuthorizationClass.VERIFIED_TRAINING,
        )
        with self.assertRaisesRegex(
            persistent_runner.RunnerAuthorizationError,
            "calibrated production environment factory",
        ):
            persistent_runner.Run4PersistentTrainingRunnerV1(
                authorization=private_token
            )

    def test_test_runner_refuses_injected_environment_factory(self) -> None:
        kwargs = {
            "environment_factory": lambda **_: object(),
        }
        with self.assertRaisesRegex(
            persistent_runner.RunnerAuthorizationError,
            "retain the synthetic environment path",
        ):
            persistent_runner._TestOnlyPersistentRunnerV1(**kwargs)

    def test_import_is_rng_cuda_and_io_neutral(self) -> None:
        torch_before = torch.random.get_rng_state().clone()
        python_before = random.getstate()
        cuda_before = torch.cuda.is_initialized()
        with mock.patch("builtins.open", side_effect=AssertionError("I/O")):
            importlib.reload(src)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_before))
        self.assertEqual(random.getstate(), python_before)
        self.assertEqual(torch.cuda.is_initialized(), cuda_before)


if __name__ == "__main__":
    unittest.main()
