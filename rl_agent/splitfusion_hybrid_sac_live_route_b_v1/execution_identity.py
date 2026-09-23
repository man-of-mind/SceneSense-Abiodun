"""Exact executed-action identity for a continuous-``q`` pilot decision.

The policy is hybrid: a discrete ``(family, quantizer)`` mode and a continuous
``q`` that is almost never one of the six measured anchors of that mode.  The
identity produced here therefore has to be honest about two different things at
once -- the action is fully *executable*, and it is not a *measured* catalog
row.

This module builds on the repository's existing identity records rather than
inventing a parallel one:

* ``SplitActionContract.resolve(mode_id, q)`` converts the proposal into an
  ``ExecutableAction`` with the registered half-up quantization and the
  registered ``drop = floor(q*N + 0.5)``, ``keep = N - drop`` rule;
* ``ExecutedActionIdentity.from_executable_action`` reconciles that action
  against the frozen catalog and returns an *attested* record.  An identity
  that fabricates an ``action_id`` for an off-anchor ``q_e4`` cannot be
  reconciled, and an unreconciled identity cannot be serialized at all.

``q_e4`` is authoritative.  The float ``q`` is quantized exactly once, here, at
the single boundary; nothing downstream re-quantizes it and nothing snaps it to
a nearby anchor.

Phase 1 binds the catalog layer.  The execution *bundle* -- the artifact-bound
descriptor produced by
``splitfusion_live_dispatch_v1.dynamic_execution_contract`` with its
checkpoint, codec and behavioral-source pins -- is a Phase-3 deliverable,
because loading it verifies runtime artifacts that Phase 1 must not touch.
Until then :attr:`PilotExecutionIdentityV1.execution_bundle_sha256` is ``None``
behind an explicit :attr:`PilotExecutionIdentityV1.bundle_binding_status`, and
:func:`bind_execution_bundle` is the seam Phase 3 will fill.  A null bundle is
a declared gap, not a silent one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    ExecutableAction,
    Q_E4_MAX,
    Q_E4_MIN,
    Q_E4_SCALE,
    SPATIAL_CELLS,
    SplitActionContract,
    keep_drop_counts,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
)

from . import pilot_contract as contract
from .frozen_actor import PolicyProposalV1

__all__ = [
    "BUNDLE_BINDING_PENDING",
    "ExecutionIdentityError",
    "MEASURED_ANCHOR",
    "PilotExecutionIdentityV1",
    "UNMEASURED_OFF_ANCHOR",
    "bind_execution_bundle",
    "resolve_pilot_execution_identity",
]

#: Labels mirroring ``dynamic_execution_contract``'s own measurement statuses,
#: so a Phase-3 bundle and a Phase-1 identity cannot disagree on the word.
MEASURED_ANCHOR = "MEASURED_ANCHOR"
UNMEASURED_OFF_ANCHOR = "UNMEASURED_OFF_ANCHOR"

BUNDLE_BINDING_PENDING = (
    "PHASE1_CATALOG_IDENTITY_ONLY;"
    "EXECUTION_BUNDLE_NOT_BOUND_UNTIL_PHASE3_DYNAMIC_EXECUTION_CONTRACT_LOAD"
)
_BUNDLE_BINDING_BOUND = "PHASE3_DYNAMIC_EXECUTION_CONTRACT_BOUND"


class ExecutionIdentityError(RuntimeError):
    """A proposal could not be turned into a reconciled executable identity."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExecutionIdentityError(message)


@dataclass(frozen=True, slots=True)
class PilotExecutionIdentityV1:
    """The authoritative identity of one executed pilot action.

    ``action_id`` and ``profile_id`` are populated only when the exact
    ``(family, quantizer, q_e4)`` triple is a registered anchor.  Off anchor
    they are ``None`` and :attr:`measurement_status` says
    ``UNMEASURED_OFF_ANCHOR``.  There is no third possibility and no nearest
    anchor.
    """

    mode_id: int
    family: str
    quantizer: str
    q_e4: int
    q_exec: float
    keep_count: int
    drop_count: int
    spatial_cells: int
    action_id: Optional[int]
    profile_id: Optional[str]
    measurement_status: str
    requested_q: float
    clipped_below: bool
    clipped_above: bool
    support_q_e4_low: int
    support_q_e4_high: int
    execution_bundle_sha256: Optional[str]
    bundle_binding_status: str
    identity: ExecutedActionIdentity
    catalog_sha256: str
    pilot_label: str
    pilot_contract_sha256: str

    def __post_init__(self) -> None:
        _require(
            (self.action_id is None) == (self.profile_id is None),
            "action_id and profile_id must both be present or both be null",
        )
        _require(
            self.measurement_status
            == (MEASURED_ANCHOR if self.action_id is not None else UNMEASURED_OFF_ANCHOR),
            "measurement status contradicts the anchor identity",
        )
        expected_keep, expected_drop = keep_drop_counts(self.q_e4)
        _require(
            (self.keep_count, self.drop_count) == (expected_keep, expected_drop),
            f"keep/drop counts contradict the registered rule for "
            f"q_e4={self.q_e4}",
        )
        _require(
            self.keep_count + self.drop_count == self.spatial_cells,
            "keep and drop counts do not partition the spatial grid",
        )
        _require(
            self.support_q_e4_low <= self.q_e4 <= self.support_q_e4_high,
            f"q_e4 {self.q_e4} is outside mode {self.mode_id}'s registered "
            f"executable interval",
        )
        _require(
            self.identity.mode_id == self.mode_id
            and self.identity.q_e4 == self.q_e4
            and self.identity.action_id == self.action_id
            and self.identity.profile_id == self.profile_id,
            "the reconciled catalog identity disagrees with this record",
        )
        # A reconciled identity can serialize; an unreconciled one cannot.
        self.identity.to_canonical_dict()

    @property
    def is_registered_anchor(self) -> bool:
        return self.action_id is not None

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "bundle_binding_status": self.bundle_binding_status,
            "catalog_sha256": self.catalog_sha256,
            "clipped_above": self.clipped_above,
            "clipped_below": self.clipped_below,
            "drop_count": self.drop_count,
            "execution_bundle_sha256": self.execution_bundle_sha256,
            "executed_action_identity": self.identity.to_canonical_dict(),
            "family": self.family,
            "keep_count": self.keep_count,
            "measurement_status": self.measurement_status,
            "mode_id": self.mode_id,
            "pilot_contract_sha256": self.pilot_contract_sha256,
            "pilot_label": self.pilot_label,
            "profile_id": self.profile_id,
            "q_e4": self.q_e4,
            "q_exec": self.q_exec,
            "quantizer": self.quantizer,
            "record": "splitfusion.live_route_b_pilot_execution_identity.v1",
            "requested_q": self.requested_q,
            "spatial_cells": self.spatial_cells,
            "support_q_e4_high": self.support_q_e4_high,
            "support_q_e4_low": self.support_q_e4_low,
        }

    def canonical_sha256(self) -> str:
        return contract.canonical_sha256(self.to_canonical_dict())


def resolve_pilot_execution_identity(
    proposal: PolicyProposalV1,
    action_contract: SplitActionContract,
) -> PilotExecutionIdentityV1:
    """Turn a frozen-actor proposal into a reconciled, executable identity.

    The proposal's ``q_e4`` is treated as authoritative.  The float ``q`` is
    still passed through the contract so that the contract's own half-up
    quantization is *recomputed* and compared: if the actor's wire integer and
    the contract's disagree by even one unit, the decision fails closed rather
    than transmitting an action the catalog would describe differently.
    """
    if type(proposal) is not PolicyProposalV1:
        raise ExecutionIdentityError(
            f"proposal must be a PolicyProposalV1, got {type(proposal).__name__}"
        )
    if not isinstance(action_contract, SplitActionContract):
        raise ExecutionIdentityError(
            f"action_contract must be the frozen SplitActionContract, got "
            f"{type(action_contract).__name__}"
        )

    action: ExecutableAction = action_contract.resolve(proposal.mode_id, proposal.q)
    _require(
        action.q_e4 == proposal.q_e4,
        f"the action contract quantizes q={proposal.q!r} to "
        f"{action.q_e4}, but the actor emitted q_e4={proposal.q_e4}; the wire "
        f"quantization boundary must be exact",
    )
    _require(
        Q_E4_MIN <= action.q_e4 <= Q_E4_MAX,
        f"resolved q_e4 {action.q_e4} is outside the mechanical wire range",
    )
    _require(
        action.mode.mode_id == proposal.mode_id,
        "the resolved mode differs from the proposed mode",
    )

    identity = ExecutedActionIdentity.from_executable_action(action, action_contract)
    return PilotExecutionIdentityV1(
        mode_id=action.mode.mode_id,
        family=action.mode.family,
        quantizer=action.mode.quantizer,
        q_e4=action.q_e4,
        q_exec=action.q_e4 / float(Q_E4_SCALE),
        keep_count=action.keep_count,
        drop_count=action.drop_count,
        spatial_cells=SPATIAL_CELLS,
        action_id=action.action_id,
        profile_id=action.profile_id,
        measurement_status=(
            MEASURED_ANCHOR if action.is_registered_anchor else UNMEASURED_OFF_ANCHOR
        ),
        requested_q=float(action.quality.requested_q),
        clipped_below=bool(action.quality.clipped_below),
        clipped_above=bool(action.quality.clipped_above),
        support_q_e4_low=proposal.support_q_e4_low,
        support_q_e4_high=proposal.support_q_e4_high,
        execution_bundle_sha256=None,
        bundle_binding_status=BUNDLE_BINDING_PENDING,
        identity=identity,
        catalog_sha256=action_contract.catalog_sha256,
        pilot_label=contract.PILOT_LABEL,
        pilot_contract_sha256=contract.PILOT_CONTRACT_SHA256,
    )


def bind_execution_bundle(
    resolved: PilotExecutionIdentityV1, profile: Any
) -> PilotExecutionIdentityV1:
    """Phase-3 seam: attach a verified ``ExecutableDispatchProfile``'s bundle.

    Phase 1 deliberately does not import
    ``splitfusion_live_dispatch_v1.dynamic_execution_contract``: loading it
    verifies runtime checkpoints, codec sources and startup artifacts, which is
    outside this phase's no-live-artifact boundary.  The duck-typed checks below
    are therefore structural; Phase 3 replaces them with
    ``DynamicExecutionContract.verify_profile``.
    """
    if type(resolved) is not PilotExecutionIdentityV1:
        raise ExecutionIdentityError(
            f"resolved must be a PilotExecutionIdentityV1, got "
            f"{type(resolved).__name__}"
        )
    for attribute in (
        "mode_id",
        "q_e4",
        "keep_count",
        "drop_count",
        "action_id",
        "profile_id",
        "measurement_status",
        "execution_bundle_sha256",
    ):
        _require(
            hasattr(profile, attribute),
            f"dispatch profile lacks the required field {attribute!r}",
        )
    for attribute in (
        "mode_id",
        "q_e4",
        "keep_count",
        "drop_count",
        "action_id",
        "profile_id",
        "measurement_status",
    ):
        _require(
            getattr(profile, attribute) == getattr(resolved, attribute),
            f"dispatch profile {attribute}={getattr(profile, attribute)!r} "
            f"disagrees with the reconciled identity "
            f"{getattr(resolved, attribute)!r}",
        )
    digest = getattr(profile, "execution_bundle_sha256")
    _require(
        isinstance(digest, str) and len(digest) == 64,
        "dispatch profile does not carry an execution-bundle SHA-256",
    )
    fields = {
        name: getattr(resolved, name)
        for name in resolved.__slots__
        if name not in ("execution_bundle_sha256", "bundle_binding_status")
    }
    return PilotExecutionIdentityV1(
        **fields,
        execution_bundle_sha256=digest,
        bundle_binding_status=_BUNDLE_BINDING_BOUND,
    )
