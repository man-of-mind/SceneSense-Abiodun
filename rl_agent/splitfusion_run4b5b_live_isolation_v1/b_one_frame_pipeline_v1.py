"""One-frame engineering bridge without weakening the 300-frame contract."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from . import b_opportunity_processor_v1 as P
from . import b_production_dependencies_v1 as D
from . import b_route_bridge_v3 as V3
from . import b_route_bridge_v4 as V4
from . import b_ue_process_v1 as U


class EngineeringOneFrameBridgeV1(V4.BRouteBridgeV4):
    """The v4 mechanics with an immutable engineering budget of exactly one."""

    def __init__(self, *, variant: Any, feature_schema_sha256: str,
                 actor_boundary_sha256: str, processor: Any,
                 route_driver: Callable[[Any], Any], raw_spool_root: Path,
                 postrun_materializer: Optional[Any] = None) -> None:
        self.variant, self.feature_schema_sha256 = variant, feature_schema_sha256
        self.actor_boundary_sha256 = actor_boundary_sha256
        self.processor, self.route_driver = processor, route_driver
        self.transmitted_budget = 1
        self.spool = V3.RawGroundTruthSpoolV3(raw_spool_root)
        self.postrun_materializer = postrun_materializer
        self.slot = V3.LatestOnlyOpportunitySlotV3()
        self.stop_requested = threading.Event(); self._lock = threading.Lock()
        self._route_started = False; self._route_thread = None
        self._route_error = None; self._sent = 0; self._sealed = None


def build_one_frame_pipeline_v1(
        *, request: U.BUEProcessRequestV1, evidence_root: Path,
        dependencies: D.ProductionDependenciesV1, cell_id: str,
        route_kwargs: Mapping[str, Any], raw_spool_root: Path,
        postrun_materializer: Optional[Any] = None,
        route_driver: Optional[Callable[[EngineeringOneFrameBridgeV1], Any]] = None,
        ) -> EngineeringOneFrameBridgeV1:
    """Bind verified actor + proven dependencies for one engineering frame."""
    if dependencies.variant is not request.variant:
        raise V3.BRouteBridgeError("dependency/request variant mismatch")
    actor = P.load_final_actor_v1(variant=request.variant,
                                  evidence_root=Path(evidence_root))
    processor = P.BOpportunityProcessorV1(
        request=request, actor=actor, telemetry=dependencies.telemetry,
        dynamic_contract=dependencies.dynamic_contract,
        continuous_ue=dependencies.continuous_ue,
        sender=dependencies.sender, remote=dependencies.remote,
        input_builder=dependencies.input_builder, cell_id=cell_id,
        chunk_bytes=dependencies.chunk_bytes,
        snr_provider=dependencies.snr_reader)
    driver = route_driver or V4.pinned_route_driver(route_kwargs)
    return EngineeringOneFrameBridgeV1(
        variant=request.variant,
        feature_schema_sha256=request.feature_schema_sha256,
        actor_boundary_sha256=request.actor_boundary_sha256,
        processor=processor, route_driver=driver,
        raw_spool_root=Path(raw_spool_root),
        postrun_materializer=postrun_materializer)


__all__ = ["EngineeringOneFrameBridgeV1", "build_one_frame_pipeline_v1"]
