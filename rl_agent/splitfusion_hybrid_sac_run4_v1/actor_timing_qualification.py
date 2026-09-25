"""Create-only CPU timing qualification for the Run-4 actor decision path.

The direct action-50 feedback probe used a fixed action, so it contains no
policy inference cost.  Run 4 defines ``action_open`` before actor inference;
this utility therefore measures the exact batch-1 stochastic path used by the
persistent runner: actor forward/sample, categorical mode draw, q quantization,
catalog resolution and executed-action identity construction.

This is engineering timing evidence, not a live end-to-end measurement.  The
registered modeled reserve is deliberately the larger of 1 ms and the measured
P99.  A future live qualification may replace it, but no caller may silently
use zero actor latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from rl_agent.splitfusion_hybrid_sac_v1 import action_contract
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
)

from . import models
from .run4_contract import POLICY_FEATURE_COUNT, POLICY_FEATURE_ORDER


SCHEMA = "splitfusion.run4.actor_timing_qualification.v1"
EVIDENCE_CLASS = "MEASURED_CPU_ENGINEERING_TIMING_NOT_LIVE_E2E"
DEFAULT_WARMUP = 100
DEFAULT_ITERATIONS = 5_000
DEFAULT_THREADS = 4
DEFAULT_ACTOR_SEED = 17
DEFAULT_Q_RNG_SEED = 1_701
DEFAULT_MODE_RNG_SEED = 1_702
MINIMUM_MODELED_RESERVE_NS = 1_000_000


class ActorTimingError(RuntimeError):
    """The timing qualification is malformed or would overwrite evidence."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Sequence[int], q: float) -> int:
    if not values:
        raise ActorTimingError("cannot summarize an empty timing population")
    if not 0.0 <= q <= 1.0:
        raise ActorTimingError("percentile must lie in [0,1]")
    ordered = sorted(values)
    return int(ordered[round((len(ordered) - 1) * q)])


def representative_states() -> tuple[tuple[float, ...], ...]:
    """Return three valid-shape causal-state patterns for timing coverage."""

    genesis = (0.5, 0.5, 9.0 / 28.0, 0.0, *((0.0,) * 12), 0, 0, 0, 0, 0)
    success_modes = [0.0] * 12
    success_modes[6] = 1.0
    success = (0.8, 0.3, 25.0 / 28.0, 0.1, *success_modes, 0.7, 0.6, 0.5, 1, 1)
    failure_modes = [0.0] * 12
    failure_modes[11] = 1.0
    failure = (0.2, 0.9, 9.0 / 28.0, 0.9, *failure_modes, 0.9, 0, 0, 1, 0)
    result = tuple(tuple(float(value) for value in row) for row in (genesis, success, failure))
    if any(len(row) != POLICY_FEATURE_COUNT for row in result):
        raise ActorTimingError("representative actor state width drifted")
    return result


@dataclass(frozen=True, slots=True)
class ActorTimingConfig:
    warmup: int = DEFAULT_WARMUP
    iterations: int = DEFAULT_ITERATIONS
    threads: int = DEFAULT_THREADS
    actor_seed: int = DEFAULT_ACTOR_SEED
    q_rng_seed: int = DEFAULT_Q_RNG_SEED
    mode_rng_seed: int = DEFAULT_MODE_RNG_SEED

    def __post_init__(self) -> None:
        for name in ("warmup", "iterations", "threads", "actor_seed", "q_rng_seed", "mode_rng_seed"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name in {"iterations", "threads"} else 0):
                raise ActorTimingError(f"{name} has an invalid value")


def measure(config: ActorTimingConfig = ActorTimingConfig()) -> dict[str, Any]:
    """Measure the exact CPU actor path without initializing CUDA."""

    if torch.cuda.is_initialized():
        raise ActorTimingError("CUDA was already initialized; CPU-only evidence refused")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        actor = models.build_run4_models(
            actor_seed=config.actor_seed, critic_seed=config.actor_seed
        ).actor
        catalog = action_contract.load_contract()
        states = representative_states()
        q_generator = torch.Generator(device="cpu")
        q_generator.manual_seed(config.q_rng_seed)
        mode_generator = torch.Generator(device="cpu")
        mode_generator.manual_seed(config.mode_rng_seed)

        def select(index: int) -> ExecutedActionIdentity:
            # This conversion is inside ``_actor_action`` in the production
            # runner and occurs after action-open, so it belongs in the clock.
            state = torch.tensor((states[index % len(states)],), dtype=torch.float32)
            sample = actor.sample_all_modes(
                state, generator=q_generator
            )
            mode_id = int(
                torch.multinomial(
                    sample.probs[0], 1, generator=mode_generator
                )[0]
            )
            q_e4 = int(sample.q_e4[0, mode_id])
            executable = catalog.resolve(
                mode_id, q_e4 / float(action_contract.Q_E4_SCALE)
            )
            result = ExecutedActionIdentity.from_executable_action(
                executable, catalog
            )
            if (result.mode_id, result.q_e4) != (mode_id, q_e4):
                raise ActorTimingError("catalog resolution changed actor output")
            return result

        for index in range(config.warmup):
            select(index)
        elapsed: list[int] = []
        mode_counts = [0] * action_contract.EXPECTED_MODE_COUNT
        for index in range(config.iterations):
            start = time.perf_counter_ns()
            action = select(index)
            elapsed.append(time.perf_counter_ns() - start)
            mode_counts[action.mode_id] += 1
        if any(value <= 0 or not math.isfinite(float(value)) for value in elapsed):
            raise ActorTimingError("non-positive or non-finite elapsed timing")
        p99 = _percentile(elapsed, 0.99)
        result = {
            "schema": SCHEMA,
            "evidence_class": EVIDENCE_CLASS,
            "clock": "time.perf_counter_ns",
            "actor_path": [
                "GUARDED_21D_FEATURE_TUPLE_READY",
                "FLOAT32_BATCH1_TENSOR_CONSTRUCTION",
                "ACTOR_SAMPLE_ALL_12_MODES",
                "CATEGORICAL_MODE_DRAW",
                "EXACT_Q_E4_QUANTIZATION",
                "CATALOG_RESOLUTION",
                "EXECUTED_ACTION_IDENTITY",
            ],
            "config": {
                "warmup": config.warmup,
                "iterations": config.iterations,
                "threads": config.threads,
                "actor_seed": config.actor_seed,
                "q_rng_seed": config.q_rng_seed,
                "mode_rng_seed": config.mode_rng_seed,
                "batch_size": 1,
            },
            "feature_order": list(POLICY_FEATURE_ORDER),
            "representative_state_count": len(states),
            "mode_counts": mode_counts,
            "timing_ns": {
                "p50": _percentile(elapsed, 0.50),
                "p95": _percentile(elapsed, 0.95),
                "p99": p99,
                "max": max(elapsed),
                "mean": sum(elapsed) / len(elapsed),
            },
            "modeled_actor_reserve_ns": max(MINIMUM_MODELED_RESERVE_NS, p99),
            "reserve_rule": "MAX_OF_1MS_AND_MEASURED_P99",
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
                "cuda_initialized_after": torch.cuda.is_initialized(),
            },
            "limitations": [
                "CPU engineering microbenchmark, not a live end-to-end frame",
                "fixed architecture and initialization; later trained weights use the same operations",
                "dispatch stops at executed action identity",
            ],
        }
        if result["runtime"]["cuda_initialized_after"]:
            raise ActorTimingError("qualification initialized CUDA")
        return result
    finally:
        torch.set_num_threads(old_threads)


def write_create_only(output_dir: Path, result: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "ACTOR_TIMING_QUALIFICATION.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "splitfusion.run4.actor_timing_manifest.v1",
        "result": {"path": result_path.name, "sha256": _sha256(result_path)},
        "source": {
            "path": str(Path(__file__).relative_to(Path(__file__).parents[2])),
            "sha256": _sha256(Path(__file__)),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result_path, manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    args = parser.parse_args(argv)
    result = measure(ActorTimingConfig(iterations=args.iterations))
    paths = write_create_only(args.output_dir, result)
    print(json.dumps({"result": str(paths[0]), "manifest": str(paths[1])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
