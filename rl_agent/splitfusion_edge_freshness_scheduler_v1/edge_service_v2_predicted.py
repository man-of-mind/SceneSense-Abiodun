"""Live v2 edge with causal predicted-install admission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rl_agent.splitfusion_edge_optimization_v1.detached_edge_preload_v2 import (
    preload_detached_optimized_edge_v2,
)
from rl_agent.splitfusion_live_dispatch_v1.registry import SplitActionRegistry

from . import live_edge_service as service
from .pipeline import CandidatePolicy, PipelineConfig


ROOT = Path(__file__).resolve().parents[2]
PREDICTION_EVIDENCE = ROOT / (
    "experiments/splitfusion_edge_freshness_scheduler_v1/"
    "20260910_predicted_horizon_288_v2/comparison.json"
)
PREDICTION_EVIDENCE_SHA256 = (
    "1e07d146ed39497625be62cf42e1a0569c9393c9297df5b9274700223739c990"
)
NETWORK_PROFILE = "FAVORABLE_STABLE"
EWMA_ALPHA = 0.2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pipeline_config(
    args: argparse.Namespace,
    policy: CandidatePolicy,
    processing_horizon_ns: int,
) -> PipelineConfig:
    if policy is not CandidatePolicy.PREDICTED_INSTALL_HORIZON:
        raise RuntimeError("predicted edge received a different policy")
    if _sha256(PREDICTION_EVIDENCE) != PREDICTION_EVIDENCE_SHA256:
        raise RuntimeError("predicted-horizon evidence hash drift")
    document = json.loads(PREDICTION_EVIDENCE.read_text(encoding="utf-8"))
    profile = SplitActionRegistry.from_runtime_binding().resolve(
        int(args.action_id)
    )
    estimate = document["prediction_inputs"][
        f"{profile.family}/{NETWORK_PROFILE}"
    ]
    return PipelineConfig(
        policy=policy,
        processing_horizon_ns=processing_horizon_ns,
        initial_predicted_compute_ns=int(
            round(float(estimate["compute_ms"]) * 1e6)
        ),
        initial_predicted_publication_ns=int(
            round(float(estimate["publication_ms"]) * 1e6)
        ),
        predicted_post_publication_install_ns=int(
            round(float(estimate["post_publication_install_ms"]) * 1e6)
        ),
        prediction_ewma_alpha=EWMA_ALPHA,
    )


def main(argv: list[str] | None = None) -> int:
    service.preload_detached_optimized_edge = preload_detached_optimized_edge_v2
    return service.main(
        argv,
        forced_policy=CandidatePolicy.PREDICTED_INSTALL_HORIZON,
        pipeline_config_factory=_pipeline_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
