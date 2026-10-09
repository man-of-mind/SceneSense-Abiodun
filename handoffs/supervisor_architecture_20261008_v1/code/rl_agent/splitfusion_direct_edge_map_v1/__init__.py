"""Direct edge-to-map SplitFusion deployment (versioned, additive).

The completed 288-cell campaign routed compact object results
edge -> UE (10.0.0.2, over the radio) -> UE-forwarded map publication -> install.
The spatial map is an edge application, so that detour is an architectural
defect: it puts the map update on the downlink and makes installation depend on
UE reachability.

This package implements the corrected path

    UE feature -> OAI uplink -> edge inference
      -> direct edge-to-map publication and installation
      -> compact feedback/agent-credit message to the UE

entirely additively. Every SHA-256-pinned module of the historical campaign --
``live_pilot_runtime.py``, ``ue_route_b_split_cell_adapter_v1.py``,
``spatial_map_server_moving_ego_uplink_only_baseline.py``,
``ue_map_install_feedback_v1.py``, ``edge_runtime.py`` and ``context_tail.py`` --
is imported unchanged, so the frozen models, split points, ranker/AE/quantizer
settings, SFD1 payload semantics and the immutable 288-cell evidence all keep
their identity.
"""

from __future__ import annotations

ARCHITECTURE = "DIRECT_EDGE_TO_MAP_V1"
PROTOCOL_VERSION = 1

__all__ = ["ARCHITECTURE", "PROTOCOL_VERSION"]
