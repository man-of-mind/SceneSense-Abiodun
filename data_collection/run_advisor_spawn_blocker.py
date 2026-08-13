#!/usr/bin/env python3
"""Run the read-only advisor blocker with reliable passive frame polling.

CARLA 0.10 on L10319 intermittently aborts secondary-client
``World.wait_for_tick`` calls even while the sole synchronous owner advances
frames.  The advisor state machine is otherwise used unchanged.  This derived
launcher substitutes a read-only ``get_snapshot`` poll and never calls
``world.tick`` or mutates world settings.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
ADVISOR_CODES = REPO_ROOT / "rl_agent" / "advisor_helper_scripts" / "codes"
if str(ADVISOR_CODES) not in sys.path:
    sys.path.insert(0, str(ADVISOR_CODES))

import spawn_blocker_v4 as advisor_blocker  # noqa: E402


def _poll_for_tick(world: object, timeout_s: Optional[float] = None) -> object:
    """Return the next snapshot without taking ownership of the world clock."""

    timeout = 10.0 if timeout_s is None else float(timeout_s)
    deadline = time.monotonic() + timeout
    initial = world.get_snapshot()
    initial_frame = int(initial.frame)
    while time.monotonic() < deadline:
        snapshot = world.get_snapshot()
        if int(snapshot.frame) != initial_frame:
            return snapshot
        time.sleep(0.005)
    raise RuntimeError(
        f"timed out waiting {timeout:.3f}s for the CARLA clock master "
        f"after frame {initial_frame}"
    )


def _install_one_shot_pedestrian_lifecycle() -> None:
    """Retire a completed crossing without starting another generation."""

    original_request_respawn = advisor_blocker.request_respawn

    def request_respawn_one_shot(
        state: object,
        simulation_time: float,
        reason: str,
        registry: object,
        args: argparse.Namespace,
        delay_override: Optional[float] = None,
    ) -> None:
        if str(reason).startswith("post-event hold completed after"):
            advisor_blocker.LOG.info(
                "Pedestrian #%d generation=%d id=%s completed one-shot "
                "crossing; retiring without respawn",
                state.index,
                state.generation,
                getattr(state.actor, "id", None),
            )
            advisor_blocker.retire_state_actors(state, registry)
            advisor_blocker.reset_activation_fields(state)
            state.state = advisor_blocker.STATE_RESPAWN_PENDING
            state.respawn_due = float("inf")
            return
        original_request_respawn(
            state,
            simulation_time,
            reason,
            registry,
            args,
            delay_override=delay_override,
        )

    advisor_blocker.request_respawn = request_respawn_one_shot


def main(argv: Optional[Sequence[str]] = None) -> int:
    wrapper_parser = argparse.ArgumentParser(add_help=False)
    wrapper_parser.add_argument("--one-shot-pedestrians", action="store_true")
    wrapper_args, advisor_args = wrapper_parser.parse_known_args(
        list(sys.argv[1:] if argv is None else argv)
    )
    advisor_blocker.carla.World.wait_for_tick = _poll_for_tick
    if wrapper_args.one_shot_pedestrians:
        _install_one_shot_pedestrian_lifecycle()
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0], *advisor_args]
    try:
        return int(advisor_blocker.main())
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
