"""Versioned immutable transaction / action / feedback identity records.

Phase 2 of the SplitFusion conditional Hybrid-SAC foundation.  This module
implements *data contracts only*, following DESIGN.md section 9 ("Transaction
identity and replay record") and section 3 ("Runtime control contract").

The identifier separation is the one registered in DESIGN.md section 9::

    session_uuid    identifies one run/UE session
    decision_seq    identifies one policy invocation
    tensor_seq      identifies one transmitted tensor
    carla_frame_id  validates the simulator frame

Several ``tensor_seq`` values may share one ``decision_seq`` (the action hold).
``tensor_seq`` is the **frozen sender chronology within a decision**: neither
``carla_frame_id`` nor input-list position is ever used to infer order.  Per
DESIGN.md section 3, the actor opens the reward ticket on the first frame of the
hold, so the *earliest* ``tensor_seq`` carries ``reward_requested=true`` and
every subsequent tensor reuses the action with ``reward_requested=false``.
DESIGN.md section 2 freezes ``k_min = 2`` frames, so a *completed* hold contains
at least two tensors.

Every serializable record fails closed unless its
:class:`ExecutedActionIdentity` has been reconciled against the frozen catalog,
so a manually fabricated anchor identity cannot reach canonical serialization.

Records
-------
* :class:`TensorTransactionId` -- the four-part tensor identity.
* :class:`ExecutedActionIdentity` -- the authoritative executed action value,
  built from a Phase-1 :class:`~.action_contract.ExecutableAction`.
* :class:`TensorTransmissionEnvelope` -- one transmitted tensor: identity +
  ``reward_requested`` + executed action.
* :class:`ActionHoldManifest` -- a completed hold over one or more tensors.
* :class:`RewardFeedbackIdentity` -- the identity of the single reward
  feedback derived from a hold's registered reward tensor.

Scope
-----
Identity and canonical serialization only.  This module does **not** implement
camera/radar state features, rewards or weights, feedback joining, acceptance,
deduplication, timeout/ticket scheduling, stale-session or late-orphan
handling, replay storage, actor/critic/SAC/training, or any
CARLA/OAI/Docker/CUDA/network integration.  ``q`` conversion is never
reimplemented here: ``q_e4``, ``keep_count`` and ``drop_count`` are carried
through from the Phase-1 action contract.

``q_e4`` is the authoritative continuous action value.  ``action_id`` and
``profile_id`` are carried only when the executed action is exactly one of the
72 registered anchors, and serialize as ``null`` otherwise; no nearest anchor
is ever substituted.

Importing this module performs no filesystem access.  The schema hashes below
are computed from in-module literals at import time.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable as _AbcIterable
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .action_contract import (
    ActionContractError,
    CATALOG_SCHEMA,
    CATALOG_SHA256,
    EXECUTION_MODE,
    EXPECTED_MODE_COUNT,
    ExecutableAction,
    Q_E4_MAX,
    Q_E4_MIN,
    SPATIAL_CELLS,
    SplitActionContract,
    keep_drop_counts,
)

__all__ = [
    "TransactionIdentityError",
    "IdentityFieldError",
    "ActionIdentityError",
    "UnreconciledActionIdentityError",
    "ActionHoldError",
    "MINIMUM_HOLD_TENSORS",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "SCHEMA_SHA256",
    "SCHEMA_DESCRIPTOR",
    "ACTION_IDENTITY_SCHEMA_ID",
    "ACTION_IDENTITY_SCHEMA_SHA256",
    "ACTION_IDENTITY_DESCRIPTOR",
    "canonical_json_bytes",
    "canonical_sha256",
    "TensorTransactionId",
    "ExecutedActionIdentity",
    "TensorTransmissionEnvelope",
    "ActionHoldManifest",
    "RewardFeedbackIdentity",
]


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class TransactionIdentityError(ActionContractError):
    """Base class for every transaction-identity contract violation."""


class IdentityFieldError(TransactionIdentityError):
    """An identity field is malformed, mistyped, or out of range."""


class ActionIdentityError(TransactionIdentityError):
    """An executed-action identity is inconsistent with its declared bindings."""


class UnreconciledActionIdentityError(ActionIdentityError):
    """An executed-action identity was used before it was reconciled.

    Raised when a record that must be serializable, or canonical serialization
    itself, is handed an :class:`ExecutedActionIdentity` that never passed
    :meth:`ExecutedActionIdentity.verify_against_catalog`.  Build identities
    with :meth:`ExecutedActionIdentity.from_executable_action` or
    :meth:`ExecutedActionIdentity.reconciled_against`.
    """


class ActionHoldError(TransactionIdentityError):
    """A set of tensors does not form one valid completed action hold."""


#: Minimum number of tensors in a *completed* action hold.  DESIGN.md section 2
#: freezes the minimum hold at ``k_min = 2`` frames; section 3 holds the
#: selected action for at least two frames.  No maximum is imposed: the hold is
#: a variable-duration relationship.
MINIMUM_HOLD_TENSORS = 2


# --------------------------------------------------------------------------- #
# Canonical serialization
# --------------------------------------------------------------------------- #


def _deep_freeze(value: Any) -> Any:
    """Recursively freeze a literal into read-only mappings and tuples.

    Mappings become :class:`types.MappingProxyType` and lists become tuples, at
    every depth, so an exported schema descriptor cannot be mutated into
    disagreeing with its published SHA-256.
    """
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Recursively convert frozen containers back to plain JSON containers.

    Only mappings and list/tuple sequences are converted; anything else is
    passed through unchanged so that :func:`json.dumps` still rejects
    unsupported types rather than silently coercing them.
    """
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` to canonical bytes.

    Canonical form is sorted keys, compact separators, ASCII escaping and
    ``allow_nan=False``, encoded UTF-8.  ``allow_nan=False`` means a NaN or
    infinity anywhere in a record is a serialization failure rather than
    non-standard JSON.  Deep-frozen containers are accepted and serialize
    identically to their plain equivalents.
    """
    text = json.dumps(
        _thaw(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return text.encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    """Return the SHA-256 hex digest of ``payload``'s canonical bytes."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _make_reconciliation_gate():
    """Build an attestation issuer/checker pair over a closure-held sentinel.

    An attestation is ``(sentinel, binding)``, where ``binding`` is derived from
    **every serialized field** of the identity it was issued for (see
    :meth:`ExecutedActionIdentity._identity_binding`).  It is therefore not
    transferable: copying it onto a record whose serialized fields differ --
    whether by :func:`dataclasses.replace` or by assigning another identity's
    attestation -- leaves the recomputed binding mismatched, and the attestation
    is refused.  An attestation does authenticate any record with *identical*
    serialized fields, because such a record is the same reconciled fact and is
    byte-indistinguishable once serialized.

    The sentinel never becomes a module attribute, so an attestation cannot be
    produced by ordinary construction or by importing a private name.  This is a
    guard against accidental or mistaken bypass, not a security boundary: Python
    offers no true privacy, and a caller determined to reach into ``__closure__``
    can still forge one.
    """
    sentinel = object()

    def issue(binding: Any) -> Tuple[Any, Any]:
        return (sentinel, binding)

    def is_valid(token: Any, binding: Any) -> bool:
        return (
            type(token) is tuple
            and len(token) == 2
            and token[0] is sentinel
            and token[1] == binding
        )

    return issue, is_valid


_issue_reconciliation, _is_valid_reconciliation = _make_reconciliation_gate()


#: Semantic descriptor of the executed-action identity record.  This is a pure
#: literal: it describes field *semantics*, not any measured value, so its hash
#: changes only when the action-identity contract itself changes.
ACTION_IDENTITY_DESCRIPTOR: Mapping[str, Any] = _deep_freeze(
    {
        "schema_id": "splitfusion_hybrid_sac_executed_action_identity_v1",
        "version": 1,
        "catalog_reconciliation_required": (
            "canonical serialization requires an identity produced by "
            "from_executable_action() or reconciled_against(); an unreconciled "
            "or fabricated identity cannot be serialized"
        ),
        "execution_mode": "the single SPLIT execution mode of the frozen catalog",
        "authoritative_continuous_value": "q_e4",
        "anchor_id_rule": (
            "action_id and profile_id are present only for an exact registered "
            "anchor and are null otherwise; no nearest anchor is substituted"
        ),
        "keep_drop_rule": "drop=floor(q*N+0.5); keep=N-drop",
        "spatial_cells": SPATIAL_CELLS,
        "q_e4_bounds": [Q_E4_MIN, Q_E4_MAX],
        "joint_mode_count": EXPECTED_MODE_COUNT,
        "fields": {
            "action_id": "optional int: exact catalog action_id, else null",
            "action_identity_schema": "str: this record's schema id",
            "action_identity_sha256": "str: this record's schema descriptor hash",
            "catalog_schema": "str: frozen action catalog schema id",
            "catalog_sha256": "str: exact SHA-256 of the frozen action catalog",
            "drop_count": "int: spatial cells dropped at q_e4",
            "execution_mode": "str: SPLIT",
            "family": "str: executed feature family",
            "keep_count": "int: spatial cells retained at q_e4",
            "mode_id": "int: stable joint-mode id in catalog-declared order",
            "profile_id": "optional str: exact catalog profile_id, else null",
            "q_e4": "int: executed wire quality in units of 1e-4",
            "quantizer": "str: executed quantizer",
        },
    }
)

#: Versioned action-identity schema id and hash, carried in every action record.
ACTION_IDENTITY_SCHEMA_ID: str = str(ACTION_IDENTITY_DESCRIPTOR["schema_id"])
ACTION_IDENTITY_SCHEMA_SHA256: str = canonical_sha256(ACTION_IDENTITY_DESCRIPTOR)


#: Semantic descriptor of the whole transaction-identity contract.  The
#: action-identity descriptor is embedded so there is exactly one source of
#: truth for it.
SCHEMA_DESCRIPTOR: Mapping[str, Any] = _deep_freeze(
    {
        "schema_id": "splitfusion_hybrid_sac_transaction_identity_v1",
        "version": 2,
        "phase": "identity records only; no reward, state, scheduling or storage",
        "design_reference": (
            "DESIGN.md section 2 frozen minimum hold, section 3 runtime control "
            "contract, section 9 transaction identity"
        ),
        "catalog_reconciliation": (
            "every serializable envelope, hold manifest and reward-feedback "
            "record requires an executed action identity already reconciled "
            "against the frozen catalog"
        ),
        "tensor_seq_semantics": (
            "tensor_seq defines sender chronology within one decision_seq; "
            "carla_frame_id and input-list position are never used to infer "
            "order"
        ),
        "catalog_binding": {
            "schema": CATALOG_SCHEMA,
            "sha256": CATALOG_SHA256,
        },
        "canonical_json": {
            "allow_nan": False,
            "ensure_ascii": True,
            "separators": [",", ":"],
            "sort_keys": True,
            "encoding": "utf-8",
        },
        "executed_action_identity": ACTION_IDENTITY_DESCRIPTOR,
        "records": {
            "action_hold_manifest": {
                "canonicalization": (
                    "tensors are ordered by ascending tensor_seq before any "
                    "ordering rule is applied, so input permutation cannot "
                    "change the serialized bytes"
                ),
                "chronology": (
                    "tensor_seq is the sender chronology within the decision; "
                    "carla_frame_id and input order are not chronology"
                ),
                "minimum_tensors": MINIMUM_HOLD_TENSORS,
                "maximum_tensors": None,
                "fields": {
                    "decision_seq": "int: the held policy invocation",
                    "executed_action": "executed_action_identity: shared by all tensors",
                    "reward_tensor_seq": (
                        "int: tensor_seq of the registered reward tensor, "
                        "always the minimum tensor_seq of the hold"
                    ),
                    "session_uuid": "str: canonical lowercase hyphenated uuid",
                    "tensor_count": "int: number of tensors in the hold",
                    "tensors": "list: per-tensor transaction + reward_requested",
                },
                "invariants": [
                    "a completed hold has at least two tensors (k_min=2)",
                    "all tensors share session_uuid and decision_seq",
                    "all tensors carry an identical reconciled executed action",
                    "tensor_seq values are unique",
                    "the earliest tensor_seq has reward_requested true",
                    "every subsequent tensor has reward_requested false",
                    "tensor_seq and carla_frame_id need not be consecutive",
                    "no maximum hold length is imposed",
                ],
            },
            "reward_feedback_identity": {
                "fields": {
                    "carla_frame_id": "int: frame of the registered reward tensor",
                    "decision_seq": "int: the decision this feedback closes",
                    "executed_action": "executed_action_identity",
                    "reward_tensor_seq": "int: tensor_seq of the reward tensor",
                    "session_uuid": "str: canonical lowercase hyphenated uuid",
                },
                "excluded": [
                    "joining",
                    "acceptance",
                    "deduplication",
                    "timeout",
                    "stale-session handling",
                    "late-orphan handling",
                ],
            },
            "tensor_transaction_id": {
                "fields": {
                    "carla_frame_id": "int >= 0: simulator frame validator",
                    "decision_seq": "int >= 0: policy invocation index",
                    "session_uuid": "str: canonical lowercase hyphenated uuid",
                    "tensor_seq": "int >= 0: transmitted tensor index",
                },
            },
            "tensor_transmission_envelope": {
                "fields": {
                    "executed_action": "executed_action_identity",
                    "reward_requested": "bool: true only for the reward tensor",
                    "transaction": "tensor_transaction_id",
                },
            },
        },
    }
)

#: Versioned transaction-identity schema id, version and hash.
SCHEMA_ID: str = str(SCHEMA_DESCRIPTOR["schema_id"])
SCHEMA_VERSION: int = int(SCHEMA_DESCRIPTOR["version"])
SCHEMA_SHA256: str = canonical_sha256(SCHEMA_DESCRIPTOR)


# --------------------------------------------------------------------------- #
# Field validators: fail closed, never normalize
# --------------------------------------------------------------------------- #


def _require_non_negative_int(value: Any, field_name: str) -> int:
    """Validate a non-negative integer identifier.

    Booleans are rejected even though ``bool`` is an ``int`` subclass, and so
    are non-``int`` numerics (including NumPy integers): an identity field must
    be an exact Python ``int`` so that canonical JSON is unambiguous.
    """
    if isinstance(value, bool):
        raise IdentityFieldError(
            f"{field_name} must be a non-negative int, not a bool: {value!r}"
        )
    if type(value) is not int:
        raise IdentityFieldError(
            f"{field_name} must be an exact int, got "
            f"{type(value).__name__}: {value!r}"
        )
    if value < 0:
        raise IdentityFieldError(f"{field_name} must be >= 0, got {value}")
    return value


def _require_canonical_uuid(value: Any, field_name: str = "session_uuid") -> str:
    """Validate a canonical lowercase hyphenated UUID string.

    Accepts only the 36-character ``8-4-4-4-12`` lowercase hyphenated form.
    Uppercase, braced, ``urn:uuid:`` and unhyphenated spellings are rejected
    rather than normalized, so that identity text compares byte-for-byte.
    """
    if not isinstance(value, str):
        raise IdentityFieldError(
            f"{field_name} must be a str, got {type(value).__name__}: {value!r}"
        )
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise IdentityFieldError(
            f"{field_name} is not a parsable UUID: {value!r}"
        ) from exc
    if str(parsed) != value:
        raise IdentityFieldError(
            f"{field_name} must be the canonical lowercase hyphenated form "
            f"{str(parsed)!r}, got {value!r}"
        )
    return value


def _require_exact_bool(value: Any, field_name: str) -> bool:
    """Validate a strict boolean; ``0``/``1`` and truthy objects are rejected."""
    if type(value) is not bool:
        raise IdentityFieldError(
            f"{field_name} must be a bool, got {type(value).__name__}: {value!r}"
        )
    return value


def _require_non_empty_str(value: Any, field_name: str) -> str:
    """Validate a non-empty string field."""
    if not isinstance(value, str) or value == "":
        raise IdentityFieldError(
            f"{field_name} must be a non-empty str, got "
            f"{type(value).__name__}: {value!r}"
        )
    return value


# --------------------------------------------------------------------------- #
# Tensor transaction identity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TensorTransactionId:
    """The four-part identity of one transmitted tensor (DESIGN.md section 9)."""

    session_uuid: str
    decision_seq: int
    tensor_seq: int
    carla_frame_id: int

    def __post_init__(self) -> None:
        _require_canonical_uuid(self.session_uuid)
        _require_non_negative_int(self.decision_seq, "decision_seq")
        _require_non_negative_int(self.tensor_seq, "tensor_seq")
        _require_non_negative_int(self.carla_frame_id, "carla_frame_id")

    @property
    def decision_key(self) -> Tuple[str, int]:
        """The ``(session_uuid, decision_seq)`` hold key."""
        return self.session_uuid, self.decision_seq

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Return the canonical mapping for this component record."""
        return {
            "carla_frame_id": self.carla_frame_id,
            "decision_seq": self.decision_seq,
            "session_uuid": self.session_uuid,
            "tensor_seq": self.tensor_seq,
        }


# --------------------------------------------------------------------------- #
# Executed action identity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ExecutedActionIdentity:
    """The authoritative identity of one executed SplitFusion action.

    Build this with :meth:`from_executable_action` so that ``q_e4``,
    ``keep_count`` and ``drop_count`` are carried through from the Phase-1
    action contract rather than recomputed here.

    ``__post_init__`` enforces every invariant that is checkable without the
    catalog (types, ranges, the registered keep/drop rule, frozen bindings, and
    that ``action_id``/``profile_id`` are both present or both null).  The
    remaining claim -- that a present anchor identity really is *this*
    ``(family, quantizer, q_e4)``'s registered anchor -- needs the catalog and
    is checked by :meth:`verify_against_catalog`.

    A directly constructed record is therefore **unreconciled**: it carries no
    catalog attestation, so :meth:`to_canonical_dict` and every record that
    embeds it fail closed.  :meth:`reconciled_against` runs the catalog check
    and returns an attested copy; :meth:`from_executable_action` does both in
    one step.  This is how a fabricated anchor identity is prevented from
    reaching canonical serialization.

    The attestation is bound to :meth:`_identity_binding`, a value derived from
    every serialized field, and that binding is recomputed from the record's own
    current fields every time the attestation is accepted.  An attestation
    therefore cannot be carried onto a mutated copy: passing a reconciled record
    through :func:`dataclasses.replace` with a changed ``q_e4``, ``keep_count``,
    ``drop_count``, mode, family, quantizer, ``action_id`` or ``profile_id``
    fails at construction, and so does copying another identity's attestation.
    """

    execution_mode: str
    mode_id: int
    family: str
    quantizer: str
    q_e4: int
    keep_count: int
    drop_count: int
    action_id: Optional[int] = None
    profile_id: Optional[str] = None
    catalog_schema: str = CATALOG_SCHEMA
    catalog_sha256: str = CATALOG_SHA256
    action_identity_schema: str = ACTION_IDENTITY_SCHEMA_ID
    action_identity_sha256: str = ACTION_IDENTITY_SCHEMA_SHA256
    #: Private catalog-reconciliation attestation, bound to every serialized
    #: field of this record.  Never serialized, excluded from equality and
    #: repr, only obtainable via reconciled_against(), and not transferable to
    #: a record whose serialized fields differ.
    _reconciliation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.execution_mode != EXECUTION_MODE:
            raise ActionIdentityError(
                f"execution_mode must be {EXECUTION_MODE!r} (the only mode this "
                f"catalog represents), got {self.execution_mode!r}"
            )
        _require_non_negative_int(self.mode_id, "mode_id")
        if self.mode_id >= EXPECTED_MODE_COUNT:
            raise ActionIdentityError(
                f"mode_id {self.mode_id} is outside "
                f"[0, {EXPECTED_MODE_COUNT - 1}]"
            )
        _require_non_empty_str(self.family, "family")
        _require_non_empty_str(self.quantizer, "quantizer")

        _require_non_negative_int(self.q_e4, "q_e4")
        if not Q_E4_MIN <= self.q_e4 <= Q_E4_MAX:
            raise ActionIdentityError(
                f"q_e4 {self.q_e4} is outside the mechanical range "
                f"[{Q_E4_MIN}, {Q_E4_MAX}]"
            )
        _require_non_negative_int(self.keep_count, "keep_count")
        _require_non_negative_int(self.drop_count, "drop_count")
        expected_keep, expected_drop = keep_drop_counts(self.q_e4)
        if (self.keep_count, self.drop_count) != (expected_keep, expected_drop):
            raise ActionIdentityError(
                f"keep/drop ({self.keep_count}, {self.drop_count}) contradict "
                f"the registered rule for q_e4={self.q_e4} "
                f"({expected_keep}, {expected_drop})"
            )

        if (self.action_id is None) != (self.profile_id is None):
            raise ActionIdentityError(
                "action_id and profile_id must both be present or both be null; "
                f"got action_id={self.action_id!r}, profile_id={self.profile_id!r}"
            )
        if self.action_id is not None:
            _require_non_negative_int(self.action_id, "action_id")
            _require_non_empty_str(self.profile_id, "profile_id")

        if self.catalog_schema != CATALOG_SCHEMA:
            raise ActionIdentityError(
                f"catalog_schema must be {CATALOG_SCHEMA!r}, "
                f"got {self.catalog_schema!r}"
            )
        if self.catalog_sha256 != CATALOG_SHA256:
            raise ActionIdentityError(
                f"catalog_sha256 must be the frozen {CATALOG_SHA256}, "
                f"got {self.catalog_sha256!r}"
            )
        if self.action_identity_schema != ACTION_IDENTITY_SCHEMA_ID:
            raise ActionIdentityError(
                f"action_identity_schema must be {ACTION_IDENTITY_SCHEMA_ID!r}, "
                f"got {self.action_identity_schema!r}"
            )
        if self.action_identity_sha256 != ACTION_IDENTITY_SCHEMA_SHA256:
            raise ActionIdentityError(
                f"action_identity_sha256 must be "
                f"{ACTION_IDENTITY_SCHEMA_SHA256}, "
                f"got {self.action_identity_sha256!r}"
            )

        if self._reconciliation is not None and not _is_valid_reconciliation(
            self._reconciliation, self._identity_binding()
        ):
            raise UnreconciledActionIdentityError(
                f"the catalog-reconciliation attestation does not bind to this "
                f"record's own serialized fields ({self.canonical_mode} "
                f"q_e4={self.q_e4} action_id={self.action_id!r}).  An "
                f"attestation is issued for one exact set of serialized fields "
                f"and is not transferable: it cannot be carried onto a copy "
                f"mutated by dataclasses.replace(), nor taken from another "
                f"identity.  Re-reconcile the mutated record with "
                f"reconciled_against(contract), or build it with "
                f"from_executable_action(action, contract)"
            )

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_executable_action(
        cls,
        action: ExecutableAction,
        contract: SplitActionContract,
    ) -> "ExecutedActionIdentity":
        """Build an identity from a Phase-1 executable action.

        ``q_e4``, ``keep_count`` and ``drop_count`` are taken verbatim from
        ``action``; no quality conversion happens here.  The anchor identity is
        carried only when ``action`` is exactly a registered anchor.
        """
        if not isinstance(action, ExecutableAction):
            raise ActionIdentityError(
                f"expected a Phase-1 ExecutableAction, got "
                f"{type(action).__name__}"
            )
        if not isinstance(contract, SplitActionContract):
            raise ActionIdentityError(
                f"expected a Phase-1 SplitActionContract, got "
                f"{type(contract).__name__}"
            )
        identity = cls(
            execution_mode=action.execution_mode,
            mode_id=action.mode.mode_id,
            family=action.mode.family,
            quantizer=action.mode.quantizer,
            q_e4=action.q_e4,
            keep_count=action.keep_count,
            drop_count=action.drop_count,
            action_id=action.action_id,
            profile_id=action.profile_id,
            catalog_schema=contract.schema,
            catalog_sha256=contract.catalog_sha256,
        )
        return identity.reconciled_against(contract)

    # -- verification ------------------------------------------------------ #

    def verify_against_catalog(self, contract: SplitActionContract) -> None:
        """Reconcile this identity against the frozen catalog.

        Raises:
            ActionIdentityError: if the joint mode, keep/drop counts or anchor
                identity disagree with the catalog -- including a fabricated
                anchor identity attached to an unmeasured ``q_e4``, or a missing
                anchor identity on an exactly registered one.
        """
        if contract.catalog_sha256 != self.catalog_sha256:
            raise ActionIdentityError(
                f"catalog SHA mismatch: identity carries {self.catalog_sha256}, "
                f"contract is bound to {contract.catalog_sha256}"
            )
        if contract.schema != self.catalog_schema:
            raise ActionIdentityError(
                f"catalog schema mismatch: identity carries "
                f"{self.catalog_schema!r}, contract is {contract.schema!r}"
            )
        mode = contract.mode(self.mode_id)
        if (mode.family, mode.quantizer) != (self.family, self.quantizer):
            raise ActionIdentityError(
                f"mode_id {self.mode_id} is {mode.canonical}, which contradicts "
                f"the declared ({self.family!r}, {self.quantizer!r})"
            )
        anchor = contract.find_anchor(self.family, self.quantizer, self.q_e4)
        if anchor is None:
            if self.action_id is not None or self.profile_id is not None:
                raise ActionIdentityError(
                    f"q_e4={self.q_e4} is not a registered anchor of "
                    f"{mode.canonical}, so action_id/profile_id must be null; "
                    f"got action_id={self.action_id!r}, "
                    f"profile_id={self.profile_id!r}"
                )
        else:
            if self.action_id != anchor.action_id or (
                self.profile_id != anchor.profile_id
            ):
                raise ActionIdentityError(
                    f"q_e4={self.q_e4} is the registered anchor "
                    f"{anchor.profile_id!r} (action_id {anchor.action_id}) of "
                    f"{mode.canonical}, but this identity carries "
                    f"action_id={self.action_id!r}, "
                    f"profile_id={self.profile_id!r}"
                )
            if (self.keep_count, self.drop_count) != (
                anchor.keep_count,
                anchor.drop_count,
            ):
                raise ActionIdentityError(
                    f"keep/drop ({self.keep_count}, {self.drop_count}) "
                    f"contradict the measured anchor {anchor.profile_id!r} "
                    f"({anchor.keep_count}, {anchor.drop_count})"
                )

    def reconciled_against(
        self,
        contract: SplitActionContract,
    ) -> "ExecutedActionIdentity":
        """Verify against the catalog and return an attested copy.

        The returned record is byte-identical in every serialized field; it
        differs only by carrying the private attestation that the catalog check
        actually ran, which is what makes it serializable.
        """
        self.verify_against_catalog(contract)
        return replace(
            self,
            _reconciliation=_issue_reconciliation(self._identity_binding()),
        )

    def _serialized_fields(self) -> Dict[str, Any]:
        """The canonical payload of this identity, without any reconciliation check.

        This is the single definition of *which* fields are serialized, used
        both by :meth:`to_canonical_dict` and by :meth:`_identity_binding`, so
        the attestation binding can never drift from the serialized content.
        """
        return {
            "action_id": self.action_id,
            "action_identity_schema": self.action_identity_schema,
            "action_identity_sha256": self.action_identity_sha256,
            "catalog_schema": self.catalog_schema,
            "catalog_sha256": self.catalog_sha256,
            "drop_count": self.drop_count,
            "execution_mode": self.execution_mode,
            "family": self.family,
            "keep_count": self.keep_count,
            "mode_id": self.mode_id,
            "profile_id": self.profile_id,
            "q_e4": self.q_e4,
            "quantizer": self.quantizer,
        }

    def _identity_binding(self) -> Tuple[Tuple[str, Any], ...]:
        """The value an attestation is bound to: every serialized field.

        A sorted key/value tuple over :meth:`_serialized_fields`.  A tuple
        rather than a hash keeps the comparison exact and cheap, and deriving it
        from the serialized payload means a future serialized field is covered
        automatically.
        """
        return tuple(sorted(self._serialized_fields().items()))

    def require_reconciled(self) -> None:
        """Fail closed unless this identity carries a catalog attestation.

        Raises:
            UnreconciledActionIdentityError: if the record was never reconciled.
        """
        if not self.is_catalog_reconciled:
            raise UnreconciledActionIdentityError(
                f"executed action {self.canonical_mode} q_e4={self.q_e4} "
                f"carries no attestation bound to its own serialized fields, "
                f"so it has not been reconciled against catalog "
                f"{self.catalog_sha256}; call reconciled_against(contract) or "
                f"build it with from_executable_action(action, contract)"
            )

    # -- properties / serialization ---------------------------------------- #

    @property
    def is_catalog_reconciled(self) -> bool:
        """True when this identity carries an attestation bound to its own fields.

        The binding is recomputed from the record's current fields on every
        call, so this stays correct even for a record whose fields were mutated
        after construction through :func:`object.__setattr__`.
        """
        return _is_valid_reconciliation(
            self._reconciliation, self._identity_binding()
        )

    @property
    def is_registered_anchor(self) -> bool:
        """True when this executed action is exactly one of the 72 anchors."""
        return self.action_id is not None

    @property
    def canonical_mode(self) -> str:
        """Canonical joint-mode string, e.g. ``'SPLIT/AE32/UINT4'``."""
        return f"{self.execution_mode}/{self.family}/{self.quantizer}"

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Return the canonical mapping, with null anchor IDs for non-anchors.

        Raises:
            UnreconciledActionIdentityError: if this identity was never
                reconciled against the frozen catalog.  A fabricated anchor
                identity therefore cannot be serialized.
        """
        self.require_reconciled()
        return self._serialized_fields()

    def canonical_bytes(self) -> bytes:
        """Canonical serialization of this action identity."""
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        """SHA-256 of :meth:`canonical_bytes`."""
        return canonical_sha256(self.to_canonical_dict())


# --------------------------------------------------------------------------- #
# Tensor transmission envelope
# --------------------------------------------------------------------------- #


def _schema_binding() -> Dict[str, Any]:
    """The schema/catalog binding block carried by every top-level record."""
    return {
        "catalog_schema": CATALOG_SCHEMA,
        "catalog_sha256": CATALOG_SHA256,
        "schema_id": SCHEMA_ID,
        "schema_sha256": SCHEMA_SHA256,
        "schema_version": SCHEMA_VERSION,
    }


@dataclass(frozen=True, slots=True)
class TensorTransmissionEnvelope:
    """One transmitted tensor: exact identity, reward flag, executed action."""

    transaction: TensorTransactionId
    reward_requested: bool
    action: ExecutedActionIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.transaction, TensorTransactionId):
            raise IdentityFieldError(
                f"transaction must be a TensorTransactionId, got "
                f"{type(self.transaction).__name__}"
            )
        _require_exact_bool(self.reward_requested, "reward_requested")
        if not isinstance(self.action, ExecutedActionIdentity):
            raise ActionIdentityError(
                f"action must be an ExecutedActionIdentity, got "
                f"{type(self.action).__name__}"
            )
        self.action.require_reconciled()

    @property
    def tensor_seq(self) -> int:
        """Convenience accessor for the tensor sequence number."""
        return self.transaction.tensor_seq

    def to_manifest_member_dict(self) -> Dict[str, Any]:
        """Per-tensor block as embedded in a hold manifest.

        The executed action is stated once at manifest level (all tensors in a
        hold are required to carry an identical action), so it is not repeated
        per tensor.
        """
        return {
            "reward_requested": self.reward_requested,
            "transaction": self.transaction.to_canonical_dict(),
        }

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Canonical mapping for the envelope as a top-level record."""
        payload = _schema_binding()
        payload.update(
            {
                "executed_action": self.action.to_canonical_dict(),
                "record": "tensor_transmission_envelope",
                "reward_requested": self.reward_requested,
                "transaction": self.transaction.to_canonical_dict(),
            }
        )
        return payload

    def canonical_bytes(self) -> bytes:
        """Canonical serialization of this envelope."""
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        """SHA-256 of :meth:`canonical_bytes`."""
        return canonical_sha256(self.to_canonical_dict())


# --------------------------------------------------------------------------- #
# Completed action-hold manifest
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ActionHoldManifest:
    """A completed action hold over two or more transmitted tensors.

    Invariants (DESIGN.md sections 2, 3 and 9):

    * at least :data:`MINIMUM_HOLD_TENSORS` tensors -- section 2 freezes the
      minimum hold at ``k_min = 2`` frames, so a one-tensor hold is not a
      completed hold;
    * all tensors share ``session_uuid`` and ``decision_seq``;
    * all tensors carry an identical, catalog-reconciled executed action;
    * ``tensor_seq`` values are unique;
    * the **earliest** ``tensor_seq`` has ``reward_requested=True`` and is
      exposed as :attr:`reward_tensor`;
    * every subsequent tensor has ``reward_requested=False``.

    ``tensor_seq`` is the frozen sender chronology within a decision.  Section 3
    opens the reward ticket on the first frame of the hold and reuses the action
    on later frames, so "first" means the minimum ``tensor_seq`` -- **not** the
    smallest ``carla_frame_id`` and **not** the first element of the input
    iterable.  Neither is ever consulted to infer order.

    ``tensor_seq`` and ``carla_frame_id`` are **not** assumed consecutive, and
    no maximum hold length is imposed: section 3 makes the hold a
    variable-duration relationship.  Tensors are canonicalized into ascending
    ``tensor_seq`` order *before* the ordering rules are applied, so input
    permutation can neither change :meth:`canonical_bytes` nor change which
    tensor is accepted as the reward tensor.
    """

    tensors: Tuple[TensorTransmissionEnvelope, ...]

    def __post_init__(self) -> None:
        if isinstance(self.tensors, (str, bytes)) or not isinstance(
            self.tensors, _AbcIterable
        ):
            raise ActionHoldError(
                f"tensors must be an iterable of TensorTransmissionEnvelope, "
                f"got {type(self.tensors).__name__}"
            )
        members = tuple(self.tensors)
        if not members:
            raise ActionHoldError(
                "a completed action hold must contain at least "
                f"{MINIMUM_HOLD_TENSORS} tensors; got none"
            )
        for position, member in enumerate(members):
            if not isinstance(member, TensorTransmissionEnvelope):
                raise ActionHoldError(
                    f"tensors[{position}] must be a TensorTransmissionEnvelope, "
                    f"got {type(member).__name__}"
                )
        if len(members) < MINIMUM_HOLD_TENSORS:
            raise ActionHoldError(
                f"a completed action hold must contain at least "
                f"{MINIMUM_HOLD_TENSORS} tensors (DESIGN.md section 2 freezes "
                f"the minimum hold at k_min={MINIMUM_HOLD_TENSORS} frames); "
                f"got {len(members)}"
            )

        reference = members[0]
        for member in members[1:]:
            if member.transaction.decision_key != reference.transaction.decision_key:
                raise ActionHoldError(
                    f"all tensors in a hold must share (session_uuid, "
                    f"decision_seq); {member.transaction.decision_key!r} != "
                    f"{reference.transaction.decision_key!r}"
                )
            if member.action != reference.action:
                raise ActionHoldError(
                    "all tensors in a hold must carry the same executed "
                    f"action; tensor_seq {member.tensor_seq} carries "
                    f"{member.action.canonical_mode} q_e4="
                    f"{member.action.q_e4} but tensor_seq "
                    f"{reference.tensor_seq} carries "
                    f"{reference.action.canonical_mode} q_e4="
                    f"{reference.action.q_e4}"
                )

        seqs = [member.tensor_seq for member in members]
        if len(set(seqs)) != len(seqs):
            duplicates = sorted({s for s in seqs if seqs.count(s) > 1})
            raise ActionHoldError(
                f"tensor_seq values must be unique within a hold; duplicates: "
                f"{duplicates}"
            )

        # Canonicalize by tensor_seq *first*: tensor_seq is the frozen sender
        # chronology within a decision, so every ordering rule below is applied
        # to this order and never to the input order or to carla_frame_id.
        # Permutation of the input therefore cannot change the serialized bytes
        # or which tensor is accepted as the reward tensor.
        ordered = tuple(sorted(members, key=lambda m: m.tensor_seq))

        earliest = ordered[0]
        if not earliest.reward_requested:
            requested = [m.tensor_seq for m in ordered if m.reward_requested]
            raise ActionHoldError(
                f"reward_requested must be True on the earliest tensor_seq "
                f"{earliest.tensor_seq} of the hold (DESIGN.md section 3 opens "
                f"the reward ticket on the first frame of the hold); it is "
                + (
                    f"set on tensor_seq {requested} instead"
                    if requested
                    else "set on no tensor at all"
                )
            )
        later_requests = [m.tensor_seq for m in ordered[1:] if m.reward_requested]
        if later_requests:
            raise ActionHoldError(
                f"every tensor after the earliest tensor_seq "
                f"{earliest.tensor_seq} must have reward_requested=False "
                f"(DESIGN.md section 3: later frames reuse the held action); "
                f"tensor_seq {later_requests} also requested a reward"
            )

        object.__setattr__(self, "tensors", ordered)

    # -- construction ------------------------------------------------------ #

    @classmethod
    def build(
        cls,
        tensors: Iterable[TensorTransmissionEnvelope],
    ) -> "ActionHoldManifest":
        """Build a manifest from any iterable of envelopes, in any order.

        Input order is irrelevant: the tensors are canonicalized by
        ``tensor_seq`` before the hold rules are applied.
        """
        return cls(tuple(tensors))

    # -- derived identity -------------------------------------------------- #

    @property
    def session_uuid(self) -> str:
        """The session shared by every tensor in the hold."""
        return self.tensors[0].transaction.session_uuid

    @property
    def decision_seq(self) -> int:
        """The single policy invocation this hold covers."""
        return self.tensors[0].transaction.decision_seq

    @property
    def action(self) -> ExecutedActionIdentity:
        """The executed action shared by every tensor in the hold."""
        return self.tensors[0].action

    @property
    def tensor_count(self) -> int:
        """Number of tensors transmitted under this hold."""
        return len(self.tensors)

    @property
    def tensor_seqs(self) -> Tuple[int, ...]:
        """Canonical ascending ``tensor_seq`` values; not necessarily contiguous.

        This is the frozen sender chronology of the hold.
        """
        return tuple(m.tensor_seq for m in self.tensors)

    @property
    def reward_tensor(self) -> TensorTransmissionEnvelope:
        """The registered reward tensor: the earliest ``tensor_seq`` of the hold."""
        earliest = self.tensors[0]
        if not earliest.reward_requested:
            raise ActionHoldError(  # pragma: no cover - construction guarantees it
                "internal invariant violated: the earliest tensor_seq is not "
                "the registered reward tensor"
            )
        return earliest

    @property
    def reward_tensor_seq(self) -> int:
        """``tensor_seq`` of the registered reward tensor (the hold minimum)."""
        return self.reward_tensor.tensor_seq

    def reward_feedback_identity(self) -> "RewardFeedbackIdentity":
        """Derive this hold's reward-feedback identity record."""
        return RewardFeedbackIdentity.from_manifest(self)

    # -- serialization ----------------------------------------------------- #

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Canonical mapping, with tensors already in ``tensor_seq`` order."""
        payload = _schema_binding()
        payload.update(
            {
                "decision_seq": self.decision_seq,
                "executed_action": self.action.to_canonical_dict(),
                "record": "action_hold_manifest",
                "reward_tensor_seq": self.reward_tensor_seq,
                "session_uuid": self.session_uuid,
                "tensor_count": self.tensor_count,
                "tensors": [m.to_manifest_member_dict() for m in self.tensors],
            }
        )
        return payload

    def canonical_bytes(self) -> bytes:
        """Canonical serialization; invariant to input tensor order."""
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        """SHA-256 of :meth:`canonical_bytes`."""
        return canonical_sha256(self.to_canonical_dict())


# --------------------------------------------------------------------------- #
# Reward feedback identity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RewardFeedbackIdentity:
    """Identity of the single reward feedback for one completed hold.

    Derived from the hold's registered reward tensor.  This is an identity
    record only: it implements no feedback joining, acceptance, deduplication,
    timeout, stale-session or late-orphan logic.  Those belong to the later
    ticket/environment phase.
    """

    session_uuid: str
    decision_seq: int
    reward_tensor_seq: int
    carla_frame_id: int
    action: ExecutedActionIdentity

    def __post_init__(self) -> None:
        _require_canonical_uuid(self.session_uuid)
        _require_non_negative_int(self.decision_seq, "decision_seq")
        _require_non_negative_int(self.reward_tensor_seq, "reward_tensor_seq")
        _require_non_negative_int(self.carla_frame_id, "carla_frame_id")
        if not isinstance(self.action, ExecutedActionIdentity):
            raise ActionIdentityError(
                f"action must be an ExecutedActionIdentity, got "
                f"{type(self.action).__name__}"
            )
        self.action.require_reconciled()

    @classmethod
    def from_manifest(cls, manifest: ActionHoldManifest) -> "RewardFeedbackIdentity":
        """Derive the feedback identity from a hold's registered reward tensor."""
        if not isinstance(manifest, ActionHoldManifest):
            raise ActionHoldError(
                f"expected an ActionHoldManifest, got {type(manifest).__name__}"
            )
        reward = manifest.reward_tensor
        return cls(
            session_uuid=reward.transaction.session_uuid,
            decision_seq=reward.transaction.decision_seq,
            reward_tensor_seq=reward.transaction.tensor_seq,
            carla_frame_id=reward.transaction.carla_frame_id,
            action=reward.action,
        )

    @property
    def decision_key(self) -> Tuple[str, int]:
        """The ``(session_uuid, decision_seq)`` key this feedback belongs to."""
        return self.session_uuid, self.decision_seq

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Canonical mapping for this top-level record."""
        payload = _schema_binding()
        payload.update(
            {
                "carla_frame_id": self.carla_frame_id,
                "decision_seq": self.decision_seq,
                "executed_action": self.action.to_canonical_dict(),
                "record": "reward_feedback_identity",
                "reward_tensor_seq": self.reward_tensor_seq,
                "session_uuid": self.session_uuid,
            }
        )
        return payload

    def canonical_bytes(self) -> bytes:
        """Canonical serialization of this feedback identity."""
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        """SHA-256 of :meth:`canonical_bytes`."""
        return canonical_sha256(self.to_canonical_dict())
