"""Live v3 edge with causal predicted-install admission."""

from __future__ import annotations

from rl_agent.splitfusion_edge_optimization_v1.detached_edge_preload_v3 import (
    preload_detached_optimized_edge_v3,
)

from . import edge_service_v2_predicted as v2
from . import live_edge_service as service
from .pipeline import CandidatePolicy


def main(argv: list[str] | None = None) -> int:
    service.preload_detached_optimized_edge = preload_detached_optimized_edge_v3
    return service.main(
        argv,
        forced_policy=CandidatePolicy.PREDICTED_INSTALL_HORIZON,
        pipeline_config_factory=v2._pipeline_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
