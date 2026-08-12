#!/usr/bin/env python3
"""Run the read-only advisor blocker with reliable passive frame polling.

CARLA 0.10 on L10319 intermittently aborts secondary-client
``World.wait_for_tick`` calls even while the sole synchronous owner advances
frames.  The advisor state machine is otherwise used unchanged.  This derived
launcher substitutes a read-only ``get_snapshot`` poll and never calls
``world.tick`` or mutates world settings.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional


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


def main() -> int:
    advisor_blocker.carla.World.wait_for_tick = _poll_for_tick
    return int(advisor_blocker.main())


if __name__ == "__main__":
    raise SystemExit(main())
