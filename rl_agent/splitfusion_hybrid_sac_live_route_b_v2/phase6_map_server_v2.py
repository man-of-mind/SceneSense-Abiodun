#!/usr/bin/env python3
"""Phase 6 map-server entrypoint: the unchanged direct map, Run-4 aware.

Runs ``splitfusion_direct_edge_map_v1.spatial_map_direct_server_v1.main``
unchanged (same baseline map state, locked install, feedback socket, Flask
API).  Inside this process only:

* ``map_ingest.protocol`` is a :class:`MapProtocolProxyV2`, which validates
  and acknowledges the Run-4 update schema against the dynamic execution
  contract and forwards every legacy call unchanged;
* the ingest service ignores the legacy integer action allowlist for Run-4
  documents (their identity is checked by the proxy instead) and writes a
  sidecar CSV reconciling every map outcome to the exact SFD3 identity;
* the frozen baseline install sees the real anchor id or ``""`` as its
  free-form ``action_id`` label, never a fabricated catalog id.

The map remains a host process bound on the CN5G bridge; nothing here goes
over the radio.
"""

from __future__ import annotations

import csv
import json
import sys
import threading
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent.splitfusion_direct_edge_map_v1 import map_ingest as MI  # noqa: E402
from rl_agent.splitfusion_direct_edge_map_v1 import protocol as DP  # noqa: E402

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (  # noqa: E402
    run4_map_protocol_v2 as MP,
)

IDENTITY_FIELDS = ("frame_id", "outcome", "run4_identity_json", "feedback_emit_at")


class Run4DirectMapIngestServiceV2(MI.DirectMapIngestService):
    def __init__(self, **kwargs: Any) -> None:
        self.legacy_allowed_action_ids = tuple(kwargs.pop("allowed_action_ids", ()))
        super().__init__(allowed_action_ids=(), **kwargs)
        sidecar = (None if self.ingest_csv is None
                   else self.ingest_csv.with_name("run4_map_identity.csv"))
        self._identity_lock = threading.Lock()
        self._identity_handle = None
        if sidecar is not None:
            self._identity_handle = sidecar.open("x", newline="", encoding="utf-8")
            self._identity_writer = csv.DictWriter(self._identity_handle,
                                                   fieldnames=list(IDENTITY_FIELDS))
            self._identity_writer.writeheader()
            self._identity_handle.flush()

    def _respond(self, document: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        result = super()._respond(document, **kwargs)
        message = result.get("feedback") or {}
        if self._identity_handle is not None and message:
            with self._identity_lock:
                self._identity_writer.writerow({
                    "frame_id": message.get("frame_id", ""),
                    "outcome": message.get("outcome", ""),
                    "run4_identity_json": json.dumps(message.get("run4_identity"),
                                                     sort_keys=True),
                    "feedback_emit_at": message.get("feedback_emit_at", ""),
                })
                self._identity_handle.flush()
        return result

    def close(self) -> dict[str, Any]:
        report = super().close()
        if self._identity_handle is not None:
            self._identity_handle.close()
            self._identity_handle = None
        report["run4_updates_validated"] = int(
            getattr(MI.protocol, "run4_updates_validated", 0))
        return report


def install(contract: Any) -> Any:
    """Process-local seams; returns the imported map-server module."""
    from rl_agent.splitfusion_direct_edge_map_v1 import spatial_map_direct_server_v1 as MS

    MI.protocol = MP.MapProtocolProxyV2(DP, contract=contract)
    MS.DirectMapIngestService = Run4DirectMapIngestServiceV2
    original = MS.install_under_state_lock

    def install_run4(document: Mapping[str, Any], ingest_at: float) -> dict[str, Any]:
        return original(MP.run4_install_document(document), ingest_at)

    MS.install_under_state_lock = install_run4
    return MS


def main() -> int:  # pragma: no cover - live process
    from rl_agent.splitfusion_live_dispatch_v1 import dynamic_execution_contract as dec

    server = install(dec.load_dynamic_execution_contract())
    return int(server.main())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
