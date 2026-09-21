"""Versioned, transport-independent cooperative-map input seam.

This package is pure: it opens no socket, starts no service, imports no CARLA,
CUDA or Docker dependency, and touches no production module. It normalizes two
*documents* into one type so a viewer can present both without either transport
borrowing the other's claims.

See ``PRODUCTION_COMPATIBILITY_V1.md`` for the contract-by-contract comparison
and for what still requires a master-based bridge worktree.
"""

from .normalized import (
    CLOCK_CARLA_SIM_NS,
    CLOCK_UNIX_WALL_NS,
    CLOCK_UNIX_WALL_S,
    NORMALIZED_SCHEMA,
    POSE_AVAILABLE,
    POSE_STATES,
    POSE_UNAVAILABLE_NO_SIDECAR,
    POSE_UNAVAILABLE_NOT_IN_PROTOCOL,
    TRANSPORT_DIRECT_UDP,
    TRANSPORT_HTTP_DEMO,
    ClockStamp,
    CoopInputError,
    EdgeTimingProvenance,
    MapTerminal,
    NormalizedCoopBatch,
    NormalizedPose,
    clocks_compatible,
)
from .identity_config import (
    BINDING_SOURCE,
    IDENTITY_CONFIG_SCHEMA,
    CoopIdentityConfig,
    IdentityBinding,
    SourceIdentity,
)
from .adapter_direct_v1 import from_direct_object_map_update
from .adapter_http_demo_v1 import (
    LIVE_VIEW_UPDATE_SCHEMA,
    from_live_view_update,
)
from .pose_sidecar_v1 import (
    POSE_SIDECAR_SCHEMA,
    POSE_UNAVAILABLE_LABEL,
    PoseSidecarIndex,
    PoseSidecarRecord,
    attach_pose,
    follow_view_contract,
    parse_pose_sidecar,
)
from .route_contract_v1 import (
    CANONICAL_ROUTE_B_SHA256,
    ROUTE_MODE_ROUTE_B_PASSIVE,
    ROUTE_MODE_STAGE2,
    ROUTE_MODES,
    CanonicalRoute,
    RouteContract,
    build_route_contract,
    load_canonical_route_b,
)
from .live_metrics_v1 import (
    FORBIDDEN_LABELS,
    INCOMPATIBLE_CLOCKS,
    NOT_AVAILABLE,
    DirectTiming,
    MetricsRegistry,
    RollingWindow,
    StreamRateMeter,
    association_reduction,
    cross_clock_age_ms,
    demo_transport_labels,
    direct_timing_for,
)
from .reference_pins import (
    DIRECT_OBJECT_MAP_UPDATE_SCHEMA,
    INTEGRATION_STATUS,
    REFERENCE_PINS,
    reference_pin_report,
    verify_reference_pins,
)

__all__ = (
    "BINDING_SOURCE",
    "CANONICAL_ROUTE_B_SHA256",
    "CLOCK_CARLA_SIM_NS",
    "CLOCK_UNIX_WALL_NS",
    "CLOCK_UNIX_WALL_S",
    "CanonicalRoute",
    "ClockStamp",
    "CoopIdentityConfig",
    "CoopInputError",
    "DIRECT_OBJECT_MAP_UPDATE_SCHEMA",
    "DirectTiming",
    "EdgeTimingProvenance",
    "FORBIDDEN_LABELS",
    "IDENTITY_CONFIG_SCHEMA",
    "INCOMPATIBLE_CLOCKS",
    "INTEGRATION_STATUS",
    "IdentityBinding",
    "LIVE_VIEW_UPDATE_SCHEMA",
    "MapTerminal",
    "MetricsRegistry",
    "NORMALIZED_SCHEMA",
    "NOT_AVAILABLE",
    "NormalizedCoopBatch",
    "NormalizedPose",
    "POSE_AVAILABLE",
    "POSE_SIDECAR_SCHEMA",
    "POSE_STATES",
    "POSE_UNAVAILABLE_LABEL",
    "POSE_UNAVAILABLE_NOT_IN_PROTOCOL",
    "POSE_UNAVAILABLE_NO_SIDECAR",
    "PoseSidecarIndex",
    "PoseSidecarRecord",
    "REFERENCE_PINS",
    "ROUTE_MODES",
    "ROUTE_MODE_ROUTE_B_PASSIVE",
    "ROUTE_MODE_STAGE2",
    "RollingWindow",
    "RouteContract",
    "SourceIdentity",
    "StreamRateMeter",
    "TRANSPORT_DIRECT_UDP",
    "TRANSPORT_HTTP_DEMO",
    "association_reduction",
    "attach_pose",
    "build_route_contract",
    "clocks_compatible",
    "cross_clock_age_ms",
    "demo_transport_labels",
    "direct_timing_for",
    "follow_view_contract",
    "from_direct_object_map_update",
    "from_live_view_update",
    "load_canonical_route_b",
    "parse_pose_sidecar",
    "reference_pin_report",
    "verify_reference_pins",
)
