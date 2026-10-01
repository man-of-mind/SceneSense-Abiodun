#!/usr/bin/env python3
"""Map-server entry point: the unchanged Phase-6 direct map, profiled.

Applies ``phase6_map_server_v2.install`` exactly as the Phase-6 entry point
does, then wraps (pass-through) the ingest worker's per-update handler, the
locked map install and the install-feedback response, and runs the same
``spatial_map_direct_server_v1.main``.  Records go to
``$SCENESENSE_PROFILER_DIR/map.profile.jsonl``; without that variable the map
runs exactly as the Phase-6 entry point.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (  # noqa: E402
    phase6_map_server_v2 as M6,
)
from rl_agent.splitfusion_single_gpu_profiler_v1 import recorder_v1 as RV  # noqa: E402
from rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1 import (  # noqa: E402
    document_identity,
)

PROFILER_DIR_ENV = "SCENESENSE_PROFILER_DIR"


def install_map_seams(server: Any, rec: Any) -> None:
    original_install = server.install_under_state_lock

    def install(document: Mapping[str, Any], ingest_at: float) -> dict[str, Any]:
        with rec.span("map.install", **document_identity(document)):
            return original_install(document, ingest_at)

    server.install_under_state_lock = install
    service_base = server.DirectMapIngestService

    class ProfiledMapIngestServiceV1(service_base):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            rec.register_gauge("map.ingest_queue_depth", self._ingest_queue.qsize)

        def _handle_complete(self, complete: Any, stamps: Mapping[str, Any]) -> None:
            with rec.span("map.ingest_handle", frame_id=int(complete.message_id)):
                return super()._handle_complete(complete, stamps)

        def _respond(self, document: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
            with rec.span("map.respond", **document_identity(document)):
                return super()._respond(document, **kwargs)

    server.DirectMapIngestService = ProfiledMapIngestServiceV1


def main() -> int:  # pragma: no cover - live process
    from rl_agent.splitfusion_live_dispatch_v1 import dynamic_execution_contract as dec

    server = M6.install(dec.load_dynamic_execution_contract())
    directory = os.environ.get(PROFILER_DIR_ENV)
    if not directory:
        return int(server.main())
    rec = RV.ProfileRecorderV1(Path(directory) / "map.profile.jsonl", role="map").start()
    RV.activate(rec)
    try:
        install_map_seams(server, rec)
        rec.set_phase("live")
        return int(server.main())
    finally:
        rec.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
