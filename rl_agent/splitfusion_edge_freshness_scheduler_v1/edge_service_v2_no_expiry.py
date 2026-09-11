"""Live latest-only edge using the v2 synchronization-light tail."""

from rl_agent.splitfusion_edge_optimization_v1.detached_edge_preload_v2 import (
    preload_detached_optimized_edge_v2,
)

from . import live_edge_service as service
from .pipeline import CandidatePolicy


def main(argv: list[str] | None = None) -> int:
    # Additive dependency injection keeps the qualified v1 service untouched.
    service.preload_detached_optimized_edge = preload_detached_optimized_edge_v2
    return service.main(
        argv,
        forced_policy=CandidatePolicy.LATEST_ONLY_NO_EXPIRY,
    )


if __name__ == "__main__":
    raise SystemExit(main())
