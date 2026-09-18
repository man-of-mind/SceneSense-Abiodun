"""SplitFusion conditional Hybrid-SAC foundation (Phases 1-2: contracts only).

This package exposes two things:

* **Phase 1** (:mod:`.action_contract`) -- a strict, read-only adapter from the
  frozen 72-profile SplitFusion action catalog to the Hybrid-SAC action
  representation ``(joint mode, continuous q)``.
* **Phase 2** (:mod:`.transaction_identity`) -- versioned immutable
  transaction / action / feedback identity records with canonical
  serialization, following DESIGN.md sections 2, 3 and 9.  ``tensor_seq`` is
  the frozen sender chronology within a decision; a completed action hold has
  at least ``k_min = 2`` tensors and requests its reward on the earliest
  ``tensor_seq``; and every serializable record requires an executed action
  identity already reconciled against the frozen catalog.

Nothing here implements reward, radar/camera state, payload/accuracy/latency
prediction, q-anchor interpolation, actor/critic or SAC updates, replay buffers
or storage, feedback joining/acceptance/deduplication, timeout/ticket logic, or
any CARLA/OAI/Docker/CUDA/network integration.  ``SPLIT`` is the only execution
mode represented; ``LOCAL_*`` and ``SKIP`` are separate future top-level modes.

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
from .transaction_identity import (  # noqa: F401
    ACTION_IDENTITY_DESCRIPTOR,
    ACTION_IDENTITY_SCHEMA_ID,
    ACTION_IDENTITY_SCHEMA_SHA256,
    ActionHoldError,
    ActionHoldManifest,
    ActionIdentityError,
    ExecutedActionIdentity,
    IdentityFieldError,
    MINIMUM_HOLD_TENSORS,
    RewardFeedbackIdentity,
    SCHEMA_DESCRIPTOR,
    SCHEMA_ID,
    SCHEMA_SHA256,
    SCHEMA_VERSION,
    TensorTransactionId,
    TensorTransmissionEnvelope,
    TransactionIdentityError,
    UnreconciledActionIdentityError,
    canonical_json_bytes,
    canonical_sha256,
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
    # Phase 2: transaction / action / feedback identity
    "ACTION_IDENTITY_DESCRIPTOR",
    "ACTION_IDENTITY_SCHEMA_ID",
    "ACTION_IDENTITY_SCHEMA_SHA256",
    "ActionHoldError",
    "ActionHoldManifest",
    "ActionIdentityError",
    "ExecutedActionIdentity",
    "IdentityFieldError",
    "MINIMUM_HOLD_TENSORS",
    "RewardFeedbackIdentity",
    "SCHEMA_DESCRIPTOR",
    "SCHEMA_ID",
    "SCHEMA_SHA256",
    "SCHEMA_VERSION",
    "TensorTransactionId",
    "TensorTransmissionEnvelope",
    "TransactionIdentityError",
    "UnreconciledActionIdentityError",
    "canonical_json_bytes",
    "canonical_sha256",
]
