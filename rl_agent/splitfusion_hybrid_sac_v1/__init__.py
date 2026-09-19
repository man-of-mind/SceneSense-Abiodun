"""SplitFusion conditional Hybrid-SAC research implementation.

The package contains the fail-closed contracts and CPU-side learning mechanics
for the feed-forward parameterized-action controller: continuous-q action and
transaction identity, SI/P40 state descriptors, the one-ticket/two-tensor hold
controller, perception-quality/reward/replay schemas, measured-anchor binding,
production replay storage, conditional actor/twin critics, one-update trainer,
and a bounded analytic algorithm-qualification runner.

Important claim boundaries remain.  The 288 measured cells are aggregate
anchors, not sequential policy transitions, and their leave-one-anchor-out
proxy is not qualified for continuous-q interpolation.  Protocol v2 can bind
exact CARLA-only quality evidence but is not integrated into the live path.
The analytic qualification checks optimizer mechanics only; it is not
SplitFusion performance or deployable-policy evidence.  Exact off-anchor
quality and payload/network outcome models are separate, source-bound stages.

``SPLIT`` is the only execution mode represented; ``LOCAL_*`` and ``SKIP`` are
future top-level modes.  This package does not launch CARLA, OAI, Docker, CUDA
or network services on import.

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
