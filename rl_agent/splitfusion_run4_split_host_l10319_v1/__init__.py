"""Offline-only contract for the prospective W10275/L10319 split host."""

from .contract import (  # noqa: F401
    ARTIFACTS,
    EDGE_IMAGE_CANONICAL_INSPECT_SHA256,
    EDGE_IMAGE_CONFIG_DIGEST,
    EDGE_IMAGE_MANIFEST_DIGEST,
    REMOTE_CONTAINER_IMAGE_ID,
    REMOTE_IMAGE_ID,
    LOCAL_ACTOR_ARTIFACT,
    SplitHostContractError,
    default_topology,
    readiness_report,
)
