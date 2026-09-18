"""SplitFusion conditional Hybrid-SAC foundation (Phase 1: action contract only).

This package currently exposes exactly one thing: a strict, read-only adapter
from the frozen 72-profile SplitFusion action catalog to the Hybrid-SAC action
representation ``(joint mode, continuous q)``.

Nothing here implements reward, radar/camera state, payload/accuracy/latency
prediction, q-anchor interpolation, actor/critic or SAC updates, replay buffers,
transition schemas, timeout/ticket logic, or any CARLA/OAI/Docker/CUDA/network
integration.  ``SPLIT`` is the only execution mode represented; ``LOCAL_*`` and
``SKIP`` are separate future top-level modes.

Importing this package has no filesystem or other runtime side effect: the
locked catalog is read only when :func:`load_contract` or
:func:`default_contract` is called explicitly.
"""

from .action_contract import (  # noqa: F401
    ActionContractError,
    AnchorAction,
    CATALOG_RELATIVE_PATH,
    CATALOG_SCHEMA,
    CATALOG_SHA256,
    CatalogIntegrityError,
    EXECUTION_MODE,
    EXPECTED_FAMILY_COUNT,
    EXPECTED_MODE_COUNT,
    EXPECTED_PROFILE_COUNT,
    EXPECTED_QUANTIZER_COUNT,
    EXPECTED_Q_ANCHOR_COUNT,
    ExecutableAction,
    InvalidQualityError,
    JointMode,
    Q_E4_MAX,
    Q_E4_MIN,
    Q_E4_SCALE,
    Q_MAX,
    Q_MIN,
    QualityWireValue,
    SPATIAL_CELLS,
    SplitActionContract,
    UnknownJointModeError,
    default_catalog_path,
    default_contract,
    keep_drop_counts,
    load_contract,
    round_half_up_q_e4,
)

__all__ = [
    "ActionContractError",
    "AnchorAction",
    "CATALOG_RELATIVE_PATH",
    "CATALOG_SCHEMA",
    "CATALOG_SHA256",
    "CatalogIntegrityError",
    "EXECUTION_MODE",
    "EXPECTED_FAMILY_COUNT",
    "EXPECTED_MODE_COUNT",
    "EXPECTED_PROFILE_COUNT",
    "EXPECTED_QUANTIZER_COUNT",
    "EXPECTED_Q_ANCHOR_COUNT",
    "ExecutableAction",
    "InvalidQualityError",
    "JointMode",
    "Q_E4_MAX",
    "Q_E4_MIN",
    "Q_E4_SCALE",
    "Q_MAX",
    "Q_MIN",
    "QualityWireValue",
    "SPATIAL_CELLS",
    "SplitActionContract",
    "UnknownJointModeError",
    "default_catalog_path",
    "default_contract",
    "keep_drop_counts",
    "load_contract",
    "round_half_up_q_e4",
]
