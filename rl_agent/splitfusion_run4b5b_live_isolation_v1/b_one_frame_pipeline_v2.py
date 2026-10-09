"""Authoritative-lineage one-frame B pipeline factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from . import b_one_frame_pipeline_v1 as V1
from . import b_one_frame_processor_v2 as P
from . import b_opportunity_processor_v1 as BASE
from . import b_production_dependencies_v1 as D
from . import b_route_bridge_v4 as V4
from . import b_ue_process_v1 as U


def build_one_frame_pipeline_v2(
        *, request: U.BUEProcessRequestV1, evidence_root: Path,
        controller_lineage_sha256: str,
        dependencies: D.ProductionDependenciesV1, cell_id: str,
        route_kwargs: Mapping[str, Any], raw_spool_root: Path,
        postrun_materializer: Optional[Any] = None,
        route_driver: Optional[Callable[[V1.EngineeringOneFrameBridgeV1], Any]] = None,
        ) -> V1.EngineeringOneFrameBridgeV1:
    """Bind the verified actor while keeping actor and controller identities distinct."""
    if dependencies.variant is not request.variant:
        raise V4.BRouteBridgeError("dependency/request variant mismatch")
    actor = BASE.load_final_actor_v1(
        variant=request.variant, evidence_root=Path(evidence_root))
    processor = P.OneFrameOpportunityProcessorV2(
        request=request, actor=actor,
        controller_lineage_sha256=controller_lineage_sha256,
        telemetry=dependencies.telemetry,
        dynamic_contract=dependencies.dynamic_contract,
        continuous_ue=dependencies.continuous_ue,
        sender=dependencies.sender, remote=dependencies.remote,
        input_builder=dependencies.input_builder, cell_id=cell_id,
        chunk_bytes=dependencies.chunk_bytes,
        snr_provider=dependencies.snr_reader)
    driver = route_driver or V4.pinned_route_driver(route_kwargs)
    return V1.EngineeringOneFrameBridgeV1(
        variant=request.variant,
        feature_schema_sha256=request.feature_schema_sha256,
        actor_boundary_sha256=request.actor_boundary_sha256,
        processor=processor, route_driver=driver,
        raw_spool_root=Path(raw_spool_root),
        postrun_materializer=postrun_materializer)


__all__ = ["build_one_frame_pipeline_v2"]
