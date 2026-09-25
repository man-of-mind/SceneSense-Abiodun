"""Fail-closed production orchestration for the bounded Run-4 SAC smoke.

This module owns only the deterministic collection/update schedule.  It does
not construct a radio model, state provider, empirical predictor or composite
verifier, and it never substitutes the synthetic mechanics runner.  A caller
must supply a factory whose product is the exact production runner; that
runner can exist only after the independent registered evidence gates in
``persistent_runner``, ``environment``, ``production_state_provider``,
``sequential_kernel`` and ``replay`` have all opened.

The initial smoke is deliberately bounded: seed 17, a balanced 288-decision
warm-up, four environment transitions before every gradient, batch size 256,
diagnostic/checkpoint hooks at updates 0/100/250/500, and a hard stop at 500.
Checkpoint resume starts strictly *after* the restored checkpoint and cannot
re-emit or skip a scheduled update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple

import torch

from . import persistent_runner, smoke_preregistration, trainer


SCHEMA_ID = "splitfusion.run4.production_training.v1"
SCHEMA_VERSION = 1


class ProductionTrainingError(RuntimeError):
    """Base class for orchestration refusal."""


class ProductionFactoryError(ProductionTrainingError):
    """A supplied factory is foreign or contradicts its verified binding."""


class ProductionScheduleError(ProductionTrainingError):
    """Runner progress does not lie on the preregistered schedule."""


class ProductionDeviceError(ProductionTrainingError):
    """The bounded CPU smoke would touch a non-CPU model or RNG."""


class _SchedulableRunner(Protocol):
    @property
    def started(self) -> bool: ...

    @property
    def decision_count(self) -> int: ...

    @property
    def trainer(self) -> object: ...

    def start(self, *, session_uuid: str, ue_id: str) -> None: ...

    def step(self) -> object: ...

    def train_once(self, batch_size: int) -> trainer.UpdateMetricsV1: ...

    def require_gradient_start(self) -> object: ...

    def checkpoint(self) -> persistent_runner.PersistentRunnerCheckpointV1: ...


@dataclass(frozen=True, slots=True)
class ProductionSmokeScheduleV1:
    """Exact executable projection of the reviewed smoke preregistration."""

    seed: int
    warmup_decisions: int
    transitions_per_update: int
    batch_size: int
    checkpoint_updates: Tuple[int, ...]
    hard_stop_update: int
    torch_intraop_threads: int
    preregistration_sha256: str

    def __post_init__(self) -> None:
        expected = smoke_preregistration.FROZEN_CONFIG
        observed = (
            self.seed,
            self.warmup_decisions,
            self.transitions_per_update,
            self.batch_size,
            self.checkpoint_updates,
            self.hard_stop_update,
            self.torch_intraop_threads,
            self.preregistration_sha256,
        )
        wanted = (
            expected.initial_smoke_seed,
            expected.warmup_decision_count,
            expected.environment_transitions_per_update,
            expected.batch_size,
            expected.smoke_checkpoint_updates,
            expected.smoke_stop_update,
            expected.torch_intraop_threads,
            smoke_preregistration.PREREGISTRATION_SHA256,
        )
        if observed != wanted:
            raise ProductionScheduleError(
                "production schedule differs from the reviewed preregistration"
            )
        if self.checkpoint_updates[0] != 0 or self.checkpoint_updates[-1] != (
            self.hard_stop_update
        ):
            raise ProductionScheduleError(
                "checkpoint schedule must include both genesis and hard stop"
            )

    def expected_decisions(self, update: int) -> int:
        if type(update) is not int or not 0 <= update <= self.hard_stop_update:
            raise ProductionScheduleError("update lies outside the bounded smoke")
        return self.warmup_decisions + self.transitions_per_update * update


FROZEN_PRODUCTION_SCHEDULE = ProductionSmokeScheduleV1(
    seed=smoke_preregistration.FROZEN_CONFIG.initial_smoke_seed,
    warmup_decisions=smoke_preregistration.FROZEN_CONFIG.warmup_decision_count,
    transitions_per_update=(
        smoke_preregistration.FROZEN_CONFIG.environment_transitions_per_update
    ),
    batch_size=smoke_preregistration.FROZEN_CONFIG.batch_size,
    checkpoint_updates=(
        smoke_preregistration.FROZEN_CONFIG.smoke_checkpoint_updates
    ),
    hard_stop_update=smoke_preregistration.FROZEN_CONFIG.smoke_stop_update,
    torch_intraop_threads=(
        smoke_preregistration.FROZEN_CONFIG.torch_intraop_threads
    ),
    preregistration_sha256=smoke_preregistration.PREREGISTRATION_SHA256,
)


@dataclass(frozen=True, slots=True)
class ProductionRunnerFactoryV1:
    """Bound constructor for a fresh exact production runner."""

    training_seed: int
    prerequisites_sha256: str
    verifier_manifest_sha256: str
    build: Callable[[], persistent_runner.Run4PersistentTrainingRunnerV1]

    def __post_init__(self) -> None:
        if type(self.training_seed) is not int or self.training_seed != (
            FROZEN_PRODUCTION_SCHEDULE.seed
        ):
            raise ProductionFactoryError("factory must bind initial smoke seed 17")
        for value, name in (
            (self.prerequisites_sha256, "prerequisites_sha256"),
            (self.verifier_manifest_sha256, "verifier_manifest_sha256"),
        ):
            try:
                persistent_runner._digest(value, name)
            except persistent_runner.RunnerBindingError as exc:
                raise ProductionFactoryError(str(exc)) from exc
        if not callable(self.build):
            raise ProductionFactoryError("build must be callable")

    def build_runner(self) -> persistent_runner.Run4PersistentTrainingRunnerV1:
        candidate = self.build()
        if type(candidate) is not persistent_runner.Run4PersistentTrainingRunnerV1:
            raise ProductionFactoryError(
                "factory must return the exact production runner; no synthetic "
                "or test runner fallback is permitted"
            )
        candidate.require_calibrated_cycle_export()
        candidate.authorization.require_training_eligible()
        if candidate.runner_binding_sha256 != self.prerequisites_sha256:
            raise ProductionFactoryError("factory/runner prerequisite digest differs")
        if candidate.prerequisites.verifier_manifest_sha256 != (
            self.verifier_manifest_sha256
        ):
            raise ProductionFactoryError("factory/runner verifier manifest differs")
        _require_cpu_runner(candidate)
        return candidate


@dataclass(frozen=True, slots=True)
class ProductionCheckpointEventV1:
    """One checkpoint plus the diagnostic input at the same update boundary."""

    update: int
    decision_count: int
    checkpoint: persistent_runner.PersistentRunnerCheckpointV1
    latest_metrics: Optional[trainer.UpdateMetricsV1]

    def __post_init__(self) -> None:
        schedule = FROZEN_PRODUCTION_SCHEDULE
        if self.update not in schedule.checkpoint_updates:
            raise ProductionScheduleError("event update is not registered")
        if self.decision_count != schedule.expected_decisions(self.update):
            raise ProductionScheduleError("event decision count differs")
        if type(self.checkpoint) is not (
            persistent_runner.PersistentRunnerCheckpointV1
        ):
            raise ProductionScheduleError("event contains a foreign checkpoint")
        if self.checkpoint.trainer_update_count != self.update:
            raise ProductionScheduleError("checkpoint update count differs")
        if self.update == 0:
            if self.latest_metrics is not None:
                raise ProductionScheduleError(
                    "update-zero checkpoint cannot carry gradient metrics"
                )
        elif type(self.latest_metrics) is not trainer.UpdateMetricsV1:
            raise ProductionScheduleError(
                "post-gradient checkpoint requires exact update metrics"
            )
        elif self.latest_metrics.update_index != self.update:
            raise ProductionScheduleError("metric/checkpoint update differs")


@dataclass(frozen=True, slots=True)
class ProductionTrainingSummaryV1:
    seed: int
    starting_update: int
    final_update: int
    final_decision_count: int
    emitted_checkpoint_updates: Tuple[int, ...]

    def __post_init__(self) -> None:
        schedule = FROZEN_PRODUCTION_SCHEDULE
        if self.seed != schedule.seed:
            raise ProductionScheduleError("summary seed differs")
        if not 0 <= self.starting_update <= self.final_update <= (
            schedule.hard_stop_update
        ):
            raise ProductionScheduleError("summary update range is invalid")
        if self.final_decision_count != schedule.expected_decisions(
            self.final_update
        ):
            raise ProductionScheduleError("summary decision count differs")


CheckpointCallback = Callable[[ProductionCheckpointEventV1], None]
DiagnosticCallback = Callable[[ProductionCheckpointEventV1], None]
BeforeTransitionHook = Callable[[int], None]


def _require_cpu_runner(
    runner: persistent_runner.Run4PersistentTrainingRunnerV1,
) -> None:
    schedule = FROZEN_PRODUCTION_SCHEDULE
    warmup = runner.warmup_schedule.config
    if (
        len(runner.warmup_schedule) != schedule.warmup_decisions
        or warmup.master_seed != schedule.seed
        or warmup.q_bin_count
        != smoke_preregistration.FROZEN_CONFIG.warmup_q_bin_count
        or warmup.samples_per_q_bin
        != smoke_preregistration.FROZEN_CONFIG.warmup_samples_per_mode_q_bin
    ):
        raise ProductionScheduleError(
            "runner warm-up differs from the balanced seed-17 schedule"
        )
    config = runner.trainer.config
    expected_config = smoke_preregistration.FROZEN_CONFIG
    if (
        config.alpha_d != expected_config.alpha_d
        or config.alpha_c != expected_config.alpha_c
        or config.tau != expected_config.polyak_tau
        or config.actor_lr != expected_config.actor_learning_rate
        or config.critic_lr != expected_config.critic_learning_rate
        or config.nominal_batch_size != expected_config.batch_size
        or config.float_dtype is not torch.float32
        or runner.replay_buffer.capacity != expected_config.replay_capacity
        or runner.replay_buffer.binding.gamma
        != expected_config.gamma_per_tensor
    ):
        raise ProductionScheduleError(
            "runner trainer/replay configuration differs from preregistration"
        )
    tensors = (
        *runner.model_bundle.actor.parameters(),
        *runner.model_bundle.actor.buffers(),
        *runner.model_bundle.critics.parameters(),
        *runner.model_bundle.critics.buffers(),
    )
    if any(value.device.type != "cpu" for value in tensors):
        raise ProductionDeviceError("production smoke is CPU-only")
    generators = (
        runner._decision_q_generator,
        runner._decision_mode_generator,
        runner._replay_generator,
        runner.trainer._target_generator,
        runner.trainer._actor_generator,
    )
    if any(
        not isinstance(value, torch.Generator)
        or value is torch.default_generator
        or value.device.type != "cpu"
        for value in generators
    ):
        raise ProductionDeviceError(
            "production smoke requires five private CPU RNG streams"
        )
    if len({id(value) for value in generators}) != len(generators):
        raise ProductionDeviceError("production RNG streams must be distinct")


def _trainer_update_count(runner: _SchedulableRunner) -> int:
    value = getattr(runner.trainer, "update_count", None)
    if type(value) is not int or value < 0:
        raise ProductionScheduleError("runner has an invalid trainer update count")
    return value


def _drive_schedule(
    runner: _SchedulableRunner,
    *,
    schedule: ProductionSmokeScheduleV1,
    emit_checkpoint: Callable[[int, _SchedulableRunner, Optional[object]], None],
    before_transition: Callable[[int], None],
    emit_initial_checkpoint: bool,
) -> ProductionTrainingSummaryV1:
    """Deterministic mechanics shared by new and resumed production runs.

    This private function is deliberately runner-agnostic so its schedule can
    be tested without fabricating production attestations.  Public entry points
    admit only the exact production runner before reaching this function.
    """

    start_update = _trainer_update_count(runner)
    if not callable(before_transition):
        raise ProductionScheduleError("before-transition reset hook is required")
    if start_update > schedule.hard_stop_update:
        raise ProductionScheduleError("runner is beyond the bounded smoke stop")

    if start_update == 0 and runner.decision_count < schedule.warmup_decisions:
        if not runner.started:
            raise ProductionScheduleError("runner must be started before collection")
        while runner.decision_count < schedule.warmup_decisions:
            before_transition(runner.decision_count)
            runner.step()
    expected = schedule.expected_decisions(start_update)
    if runner.decision_count != expected:
        raise ProductionScheduleError(
            "runner is not at an exact update/checkpoint boundary"
        )
    runner.require_gradient_start()

    emitted: list[int] = []
    if start_update == 0 and emit_initial_checkpoint:
        emit_checkpoint(0, runner, None)
        emitted.append(0)

    latest: Optional[object] = None
    while _trainer_update_count(runner) < schedule.hard_stop_update:
        before_update = _trainer_update_count(runner)
        before_decisions = runner.decision_count
        for _ in range(schedule.transitions_per_update):
            before_transition(runner.decision_count)
            runner.step()
        if runner.decision_count != before_decisions + (
            schedule.transitions_per_update
        ):
            raise ProductionScheduleError(
                "environment did not emit exactly four transitions"
            )
        latest = runner.train_once(schedule.batch_size)
        after_update = _trainer_update_count(runner)
        if after_update != before_update + 1:
            raise ProductionScheduleError(
                "one schedule iteration must perform exactly one gradient"
            )
        if after_update in schedule.checkpoint_updates:
            emit_checkpoint(after_update, runner, latest)
            emitted.append(after_update)

    final_update = _trainer_update_count(runner)
    if final_update != schedule.hard_stop_update:
        raise ProductionScheduleError("bounded smoke did not stop exactly at 500")
    if runner.decision_count != schedule.expected_decisions(final_update):
        raise ProductionScheduleError("final collection/update ratio differs")
    return ProductionTrainingSummaryV1(
        seed=schedule.seed,
        starting_update=start_update,
        final_update=final_update,
        final_decision_count=runner.decision_count,
        emitted_checkpoint_updates=tuple(emitted),
    )


def _callbacks(
    *,
    checkpoint_callback: CheckpointCallback,
    diagnostic_callback: DiagnosticCallback,
) -> Callable[[int, _SchedulableRunner, Optional[object]], None]:
    if not callable(checkpoint_callback) or not callable(diagnostic_callback):
        raise ProductionScheduleError(
            "checkpoint and diagnostic callbacks are both required"
        )

    def emit(
        update: int,
        raw_runner: _SchedulableRunner,
        latest: Optional[object],
    ) -> None:
        if type(raw_runner) is not (
            persistent_runner.Run4PersistentTrainingRunnerV1
        ):
            raise ProductionFactoryError("callback source is not production runner")
        checkpoint = raw_runner.checkpoint()
        event = ProductionCheckpointEventV1(
            update=update,
            decision_count=raw_runner.decision_count,
            checkpoint=checkpoint,
            latest_metrics=latest,
        )
        checkpoint_callback(event)
        diagnostic_callback(event)

    return emit


def run_new_production_smoke(
    *,
    factory: ProductionRunnerFactoryV1,
    session_uuid: str,
    ue_id: str,
    checkpoint_callback: CheckpointCallback,
    diagnostic_callback: DiagnosticCallback,
    before_transition_hook: BeforeTransitionHook,
) -> ProductionTrainingSummaryV1:
    """Build, start and execute the preregistered bounded smoke."""

    if type(factory) is not ProductionRunnerFactoryV1:
        raise ProductionFactoryError("factory must be ProductionRunnerFactoryV1")
    if not callable(before_transition_hook):
        raise ProductionScheduleError("before-transition reset hook is required")
    emitter = _callbacks(
        checkpoint_callback=checkpoint_callback,
        diagnostic_callback=diagnostic_callback,
    )
    runner = factory.build_runner()
    if runner.started or runner.decision_count != 0 or runner.trainer.update_count != 0:
        raise ProductionScheduleError("new-run factory did not return a pristine runner")
    if torch.get_num_threads() != FROZEN_PRODUCTION_SCHEDULE.torch_intraop_threads:
        raise ProductionDeviceError(
            "torch intra-op threads differ from the preregistered value 4"
        )
    runner.start(session_uuid=session_uuid, ue_id=ue_id)
    return _drive_schedule(
        runner,
        schedule=FROZEN_PRODUCTION_SCHEDULE,
        emit_checkpoint=emitter,
        before_transition=before_transition_hook,
        emit_initial_checkpoint=True,
    )


def resume_production_smoke(
    *,
    factory: ProductionRunnerFactoryV1,
    checkpoint: persistent_runner.PersistentRunnerCheckpointV1,
    checkpoint_callback: CheckpointCallback,
    diagnostic_callback: DiagnosticCallback,
    before_transition_hook: BeforeTransitionHook,
) -> ProductionTrainingSummaryV1:
    """Restore one exact checkpoint and continue without re-emitting it."""

    if type(factory) is not ProductionRunnerFactoryV1:
        raise ProductionFactoryError("factory must be ProductionRunnerFactoryV1")
    if type(checkpoint) is not persistent_runner.PersistentRunnerCheckpointV1:
        raise ProductionFactoryError("checkpoint must be exact Run-4 checkpoint")
    if not callable(before_transition_hook):
        raise ProductionScheduleError("before-transition reset hook is required")
    emitter = _callbacks(
        checkpoint_callback=checkpoint_callback,
        diagnostic_callback=diagnostic_callback,
    )
    runner = persistent_runner.Run4PersistentTrainingRunnerV1.restore(
        checkpoint, fresh_factory=factory.build_runner
    )
    _require_cpu_runner(runner)
    if torch.get_num_threads() != FROZEN_PRODUCTION_SCHEDULE.torch_intraop_threads:
        raise ProductionDeviceError(
            "torch intra-op threads differ from the preregistered value 4"
        )
    return _drive_schedule(
        runner,
        schedule=FROZEN_PRODUCTION_SCHEDULE,
        emit_checkpoint=emitter,
        before_transition=before_transition_hook,
        emit_initial_checkpoint=False,
    )


__all__ = [
    "BeforeTransitionHook",
    "FROZEN_PRODUCTION_SCHEDULE",
    "ProductionCheckpointEventV1",
    "ProductionDeviceError",
    "ProductionFactoryError",
    "ProductionRunnerFactoryV1",
    "ProductionScheduleError",
    "ProductionSmokeScheduleV1",
    "ProductionTrainingError",
    "ProductionTrainingSummaryV1",
    "resume_production_smoke",
    "run_new_production_smoke",
]
