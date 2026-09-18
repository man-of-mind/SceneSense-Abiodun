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
Exactly one of them is marked ``reward_requested=true`` and is the only tensor
that may generate that decision's reward.

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
from dataclasses import dataclass
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
    "ActionHoldError",
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


class ActionHoldError(TransactionIdentityError):
    """A set of tensors does not form one valid completed action hold."""


# --------------------------------------------------------------------------- #
# Canonical serialization
# --------------------------------------------------------------------------- #


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` to canonical bytes.

    Canonical form is sorted keys, compact separators, ASCII escaping and
    ``allow_nan=False``, encoded UTF-8.  ``allow_nan=False`` means a NaN or
    infinity anywhere in a record is a serialization failure rather than
    non-standard JSON.
    """
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return text.encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    """Return the SHA-256 hex digest of ``payload``'s canonical bytes."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


#: Semantic descriptor of the executed-action identity record.  This is a pure
#: literal: it describes field *semantics*, not any measured value, so its hash
#: changes only when the action-identity contract itself changes.
ACTION_IDENTITY_DESCRIPTOR: Mapping[str, Any] = MappingProxyType(
    {
        "schema_id": "splitfusion_hybrid_sac_executed_action_identity_v1",
        "version": 1,
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
ACTION_IDENTITY_SCHEMA_SHA256: str = canonical_sha256(dict(ACTION_IDENTITY_DESCRIPTOR))


#: Semantic descriptor of the whole transaction-identity contract.  The
#: action-identity descriptor is embedded so there is exactly one source of
#: truth for it.
SCHEMA_DESCRIPTOR: Mapping[str, Any] = MappingProxyType(
    {
        "schema_id": "splitfusion_hybrid_sac_transaction_identity_v1",
        "version": 1,
        "phase": "identity records only; no reward, state, scheduling or storage",
        "design_reference": "DESIGN.md section 9 transaction identity",
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
        "executed_action_identity": dict(ACTION_IDENTITY_DESCRIPTOR),
        "records": {
            "action_hold_manifest": {
                "canonicalization": "tensors are ordered by tensor_seq",
                "fields": {
                    "decision_seq": "int: the held policy invocation",
                    "executed_action": "executed_action_identity: shared by all tensors",
                    "reward_tensor_seq": "int: tensor_seq of the registered reward tensor",
                    "session_uuid": "str: canonical lowercase hyphenated uuid",
                    "tensor_count": "int: number of tensors in the hold",
                    "tensors": "list: per-tensor transaction + reward_requested",
                },
                "invariants": [
                    "at least one tensor",
                    "all tensors share session_uuid and decision_seq",
                    "all tensors carry an identical executed action",
                    "tensor_seq values are unique",
                    "exactly one tensor has reward_requested true",
                    "frame and sequence numbers need not be consecutive",
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
SCHEMA_SHA256: str = canonical_sha256(dict(SCHEMA_DESCRIPTOR))


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
        identity.verify_against_catalog(contract)
        return identity

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

    # -- properties / serialization ---------------------------------------- #

    @property
    def is_registered_anchor(self) -> bool:
        """True when this executed action is exactly one of the 72 anchors."""
        return self.action_id is not None

    @property
    def canonical_mode(self) -> str:
        """Canonical joint-mode string, e.g. ``'SPLIT/AE32/UINT4'``."""
        return f"{self.execution_mode}/{self.family}/{self.quantizer}"

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Return the canonical mapping, with null anchor IDs for non-anchors."""
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
    """A completed action hold over one or more transmitted tensors.

    Invariants (DESIGN.md section 3 and section 9):

    * at least one tensor;
    * all tensors share ``session_uuid`` and ``decision_seq``;
    * all tensors carry an identical executed action;
    * ``tensor_seq`` values are unique;
    * exactly one tensor has ``reward_requested=True``, exposed as
      :attr:`reward_tensor`.

    Frame and sequence numbers are **not** assumed consecutive and no maximum
    hold length is imposed: DESIGN.md section 3 makes the hold a
    variable-duration relationship.  Tensors are canonicalized into
    ``tensor_seq`` order at construction, so input permutation cannot change
    :meth:`canonical_bytes`.
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
                "an action hold must contain at least one tensor"
            )
        for position, member in enumerate(members):
            if not isinstance(member, TensorTransmissionEnvelope):
                raise ActionHoldError(
                    f"tensors[{position}] must be a TensorTransmissionEnvelope, "
                    f"got {type(member).__name__}"
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

        reward_seqs = [m.tensor_seq for m in members if m.reward_requested]
        if len(reward_seqs) != 1:
            raise ActionHoldError(
                f"exactly one tensor must have reward_requested=True, found "
                f"{len(reward_seqs)}"
                + (f" at tensor_seq {sorted(reward_seqs)}" if reward_seqs else "")
            )

        # Canonicalize by tensor_seq: permutation of the input must not change
        # the serialized bytes.
        object.__setattr__(
            self, "tensors", tuple(sorted(members, key=lambda m: m.tensor_seq))
        )

    # -- construction ------------------------------------------------------ #

    @classmethod
    def build(
        cls,
        tensors: Iterable[TensorTransmissionEnvelope],
    ) -> "ActionHoldManifest":
        """Build a manifest from any iterable of envelopes, in any order."""
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
        """Canonical ascending ``tensor_seq`` values; not necessarily contiguous."""
        return tuple(m.tensor_seq for m in self.tensors)

    @property
    def reward_tensor(self) -> TensorTransmissionEnvelope:
        """The single registered reward-requested tensor of this hold."""
        for member in self.tensors:
            if member.reward_requested:
                return member
        raise ActionHoldError(  # pragma: no cover - construction guarantees one
            "internal invariant violated: no registered reward tensor"
        )

    @property
    def reward_tensor_seq(self) -> int:
        """``tensor_seq`` of the registered reward tensor."""
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
