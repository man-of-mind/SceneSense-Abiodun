"""Small, dependency-free controlled signal-unwind helper."""

from __future__ import annotations

import signal
from typing import Any, Callable, TypeVar


T = TypeVar("T")


def controlled_shutdown(_signum: int, _frame: Any) -> None:
    """Raise on the Python main thread so surrounding ``finally`` blocks run."""

    raise SystemExit(0)


def run_with_shutdown_handlers(run: Callable[[], T]) -> T:
    """Run one owner while TERM/INT cause a controlled Python unwind."""

    previous = {
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        signal.SIGINT: signal.getsignal(signal.SIGINT),
    }
    for signum in previous:
        signal.signal(signum, controlled_shutdown)
    try:
        return run()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
