"""Phase 3: continuous ``(mode_id, q_e4)`` execution through UE and edge.

One authoritative seam converts an
:class:`~splitfusion_live_dispatch_v1.dynamic_execution_contract.ExecutableDispatchProfile`
into the attested Run-4 :class:`ExecutedActionIdentity`.  The seam reads only
the profile's exact integer fields and reconciles them with the frozen
catalog; it never re-quantizes a float and never fabricates an anchor
``action_id`` for an off-anchor action.

The SFD1 envelope and the anchor-only ``Preloaded*Runtime`` classes resolve
an integer ``action_id`` through the 72-row registry, so they cannot carry an
off-anchor action.  They are left untouched.  This module adds:

* :class:`CodecProfileViewV2` -- the attribute view the existing
  ``ProductionSplitCodec`` expects, with ``q`` derived from exact ``q_e4``;
* the **SFD3** envelope: exact mode/q/keep/bundle/nullable-anchor identity,
  session/lineage/decision/ticket/frame/tensor/capture identity,
  ``reward_requested``, and SHA-256 over the inner payload and the header;
* UE and edge runtimes over ``ExecutableDispatchProfile``.  The edge
  re-resolves the profile from the envelope and refuses any mode, q, keep,
  codec, bundle or anchor mismatch.

This is CPU protocol/identity parity only.  It makes no CUDA, model-output or
live-performance claim.  Importing it starts nothing.
"""

from __future__ import annotations

import hashlib
import struct
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    continuous_q,
)
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract as ac
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
)
from rl_agent.splitfusion_live_dispatch_v1 import dynamic_execution_contract as dec
from rl_agent.splitfusion_live_dispatch_v1.timing import (
    EDGE_STAGES,
    UE_STAGES,
    StageRecorder,
    TimingTrace,
)
from rl_agent.splitfusion_live_dispatch_v1.transport import (
    InnerIdentity,
    require_inner_agreement,
)

__all__ = [
    "ContinuousExecutionError",
    "executed_identity_from_profile",
    "CodecProfileViewV2",
    "codec_view",
    "ExecutionEnvelopeV3",
    "pack_envelope_v3",
    "unpack_envelope_v3",
    "ContinuousUERuntimeV2",
    "ContinuousEdgeRuntimeV2",
]


class ContinuousExecutionError(RuntimeError):
    """A continuous-execution identity, envelope or codec agreement failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContinuousExecutionError(message)


# ---------------------------------------------------------------------------
# The authoritative identity seam
# ---------------------------------------------------------------------------


def executed_identity_from_profile(
    profile: dec.ExecutableDispatchProfile,
    contract: dec.DynamicExecutionContract,
) -> ExecutedActionIdentity:
    """ExecutableDispatchProfile -> attested ExecutedActionIdentity.

    The profile must equal the contract's own resolution (``verify_profile``).
    Every field is copied from its exact integer/string value and then
    reconciled against the frozen catalog by ``reconciled_against``.
    """
    contract.verify_profile(profile)
    identity = ExecutedActionIdentity(
        execution_mode=ac.EXECUTION_MODE,
        mode_id=profile.mode_id,
        family=profile.family,
        quantizer=profile.quantizer,
        q_e4=profile.q_e4,
        keep_count=profile.keep_count,
        drop_count=profile.drop_count,
        action_id=profile.action_id,
        profile_id=profile.profile_id,
        catalog_schema=profile.catalog_schema,
        catalog_sha256=profile.catalog_sha256,
    ).reconciled_against(contract.action_contract)
    anchored = profile.measurement_status == dec.MEASURED_ANCHOR
    _require(anchored == (identity.action_id is not None),
             "measurement status disagrees with the anchor identity")
    return identity


@dataclass(frozen=True, slots=True)
class CodecProfileViewV2:
    """Exactly the attributes ``ProductionSplitCodec``/``InnerIdentity`` use."""

    family: str
    family_id: int
    quantizer: str
    bit_width: int
    q_e4: int
    keep_count: int
    drop_count: int
    routing_tag: int
    transported_channels: int
    latent_width: Optional[int]
    wire: dec.WireExecutionIdentity
    action_id: Optional[int]
    profile_id: Optional[str]
    execution_bundle_sha256: str

    @property
    def q(self) -> float:
        """Execution value derived from exact ``q_e4``; never a policy float."""
        return self.q_e4 / ac.Q_E4_SCALE


def codec_view(profile: dec.ExecutableDispatchProfile) -> CodecProfileViewV2:
    view = CodecProfileViewV2(
        family=profile.family, family_id=profile.family_id,
        quantizer=profile.quantizer, bit_width=profile.bit_width,
        q_e4=profile.q_e4, keep_count=profile.keep_count,
        drop_count=profile.drop_count, routing_tag=profile.routing_tag,
        transported_channels=profile.transported_channels,
        latent_width=profile.latent_width, wire=profile.wire,
        action_id=profile.action_id, profile_id=profile.profile_id,
        execution_bundle_sha256=profile.execution_bundle_sha256)
    plan = continuous_q.quantize_q(view.q)
    _require(plan.q_e4 == profile.q_e4 and plan.keep_count == profile.keep_count
             and plan.drop_count == profile.drop_count and not plan.snapped,
             "codec q plan disagrees with the exact profile q_e4/keep/drop")
    return view


def expected_inner(view: CodecProfileViewV2) -> InnerIdentity:
    return InnerIdentity(
        family=view.family, family_id=view.family_id, quantizer=view.quantizer,
        bit_width=view.bit_width, q_e4=view.q_e4, keep_count=view.keep_count,
        routing_tag=view.routing_tag,
        transported_channels=view.transported_channels,
        latent_width=view.latent_width, wire_magic_ascii=view.wire.magic_ascii,
        wire_codec_id=view.wire.codec_id, wire_version=view.wire.version)


# ---------------------------------------------------------------------------
# SFD3 envelope
# ---------------------------------------------------------------------------

MAGIC_V3 = b"SFD3"
VERSION_V3 = 3
FLAG_ANCHOR = 0x01
FLAG_REWARD_REQUESTED = 0x02
_HEADER_V3 = struct.Struct("<4sHHBBHIIQQQQQ16s32s32s32sQ")
_DIGEST_BYTES = 32
HEADER_V3_BYTES = _HEADER_V3.size + _DIGEST_BYTES
_NO_ANCHOR = 0xFFFFFFFF


@dataclass(frozen=True, slots=True)
class ExecutionEnvelopeV3:
    mode_id: int
    q_e4: int
    keep_count: int
    anchor_action_id: Optional[int]
    reward_requested: bool
    session_uuid: str
    controller_lineage_sha256: str
    decision_seq: int
    ticket_seq: int
    frame_id: int
    tensor_seq: int
    capture_timestamp_ns: int
    execution_bundle_sha256: str
    inner_payload_sha256: str
    inner_payload: bytes

    def identity_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__
                if name != "inner_payload"}


def _hex32(value: str, name: str) -> bytes:
    _require(isinstance(value, str) and len(value) == 64, f"{name} must be 64 hex")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ContinuousExecutionError(f"{name} is not hex") from exc
    _require(raw.hex() == value, f"{name} must be lowercase hex")
    return raw


def pack_envelope_v3(envelope: ExecutionEnvelopeV3) -> bytes:
    e = envelope
    _require(0 <= e.mode_id < ac.EXPECTED_MODE_COUNT, "mode_id out of range")
    _require(ac.Q_E4_MIN <= e.q_e4 <= ac.Q_E4_MAX, "q_e4 out of range")
    _require(isinstance(e.inner_payload, bytes) and e.inner_payload,
             "inner payload must be non-empty bytes")
    _require(hashlib.sha256(e.inner_payload).hexdigest() == e.inner_payload_sha256,
             "inner payload digest disagrees")
    _require(type(e.reward_requested) is bool, "reward_requested must be bool")
    if e.anchor_action_id is not None:
        _require(0 <= e.anchor_action_id < _NO_ANCHOR, "anchor action_id range")
    session = uuid.UUID(e.session_uuid)
    _require(str(session) == e.session_uuid, "session UUID must be canonical")
    flags = (FLAG_ANCHOR if e.anchor_action_id is not None else 0) | (
        FLAG_REWARD_REQUESTED if e.reward_requested else 0)
    header = _HEADER_V3.pack(
        MAGIC_V3, VERSION_V3, HEADER_V3_BYTES, e.mode_id, flags, e.q_e4,
        e.keep_count,
        _NO_ANCHOR if e.anchor_action_id is None else e.anchor_action_id,
        e.decision_seq, e.ticket_seq, e.frame_id, e.tensor_seq,
        e.capture_timestamp_ns, session.bytes,
        _hex32(e.controller_lineage_sha256, "controller_lineage_sha256"),
        _hex32(e.execution_bundle_sha256, "execution_bundle_sha256"),
        _hex32(e.inner_payload_sha256, "inner_payload_sha256"),
        len(e.inner_payload))
    return header + hashlib.sha256(header).digest() + e.inner_payload


def unpack_envelope_v3(frame: bytes) -> ExecutionEnvelopeV3:
    """Fail closed on any version, length, flag or digest inconsistency."""
    _require(isinstance(frame, (bytes, bytearray, memoryview)), "frame must be bytes")
    data = bytes(frame)
    _require(len(data) >= HEADER_V3_BYTES, "frame shorter than the SFD3 header")
    magic = data[:4]
    _require(magic == MAGIC_V3,
             f"not an SFD3 envelope (magic {magic!r}); SFD1 frames are handled "
             "only by the unchanged anchor-only runtime")
    (magic, version, header_bytes, mode_id, flags, q_e4, keep_count, anchor,
     decision_seq, ticket_seq, frame_id, tensor_seq, capture_ns, session,
     lineage, bundle, payload_sha, payload_length) = _HEADER_V3.unpack_from(data)
    _require(version == VERSION_V3 and header_bytes == HEADER_V3_BYTES,
             "SFD3 version/header length mismatch")
    header = data[:_HEADER_V3.size]
    digest = data[_HEADER_V3.size:HEADER_V3_BYTES]
    _require(hashlib.sha256(header).digest() == digest,
             "SFD3 header digest mismatch (corrupt header)")
    _require(flags & ~(FLAG_ANCHOR | FLAG_REWARD_REQUESTED) == 0, "unknown SFD3 flags")
    _require(bool(flags & FLAG_ANCHOR) == (anchor != _NO_ANCHOR),
             "anchor flag disagrees with the anchor field")
    payload = data[HEADER_V3_BYTES:]
    _require(len(payload) == payload_length and payload_length > 0,
             "SFD3 inner length mismatch")
    _require(hashlib.sha256(payload).digest() == payload_sha,
             "SFD3 inner payload digest mismatch (corrupt payload)")
    return ExecutionEnvelopeV3(
        mode_id=mode_id, q_e4=q_e4, keep_count=keep_count,
        anchor_action_id=None if anchor == _NO_ANCHOR else anchor,
        reward_requested=bool(flags & FLAG_REWARD_REQUESTED),
        session_uuid=str(uuid.UUID(bytes=session)),
        controller_lineage_sha256=lineage.hex(), decision_seq=decision_seq,
        ticket_seq=ticket_seq, frame_id=frame_id, tensor_seq=tensor_seq,
        capture_timestamp_ns=capture_ns, execution_bundle_sha256=bundle.hex(),
        inner_payload_sha256=payload_sha.hex(), inner_payload=payload)


# ---------------------------------------------------------------------------
# UE and edge runtimes
# ---------------------------------------------------------------------------

AE_FAMILIES = ("AE128", "AE64", "AE32")


@dataclass(frozen=True, slots=True)
class FrameIdentityV2:
    session_uuid: str
    controller_lineage_sha256: str
    decision_seq: int
    ticket_seq: int
    frame_id: int
    tensor_seq: int
    capture_timestamp_ns: int
    reward_requested: bool


@dataclass(frozen=True, slots=True)
class PreparedFrameV2:
    wire_bytes: bytes
    envelope: ExecutionEnvelopeV3
    action: ExecutedActionIdentity
    profile: dec.ExecutableDispatchProfile
    timing: TimingTrace


class ContinuousUERuntimeV2:
    """UE side: exact profile -> selective saliency keep/drop -> SFD3."""

    def __init__(self, contract: dec.DynamicExecutionContract, *, front: Callable,
                 ranker: Any, ae_encoders: Mapping[str, Any], codec: Any) -> None:
        _require(set(ae_encoders) == set(AE_FAMILIES), "exactly three AE encoders")
        _require(hasattr(ranker, "score_cells"), "ranker must expose score_cells")
        self._contract = contract
        self._front = front
        self._ranker = ranker
        self._encoders = dict(ae_encoders)
        self._codec = codec

    def prepare(self, profile: dec.ExecutableDispatchProfile, input_7ch: Any,
                frame: FrameIdentityV2) -> PreparedFrameV2:
        action = executed_identity_from_profile(profile, self._contract)
        view = codec_view(profile)
        timing = StageRecorder(UE_STAGES)
        with timing.stage("total_ue_preparation"):
            with timing.stage("front_backbone"):
                with torch.inference_mode():
                    c2 = self._front(input_7ch)
            ranker = None if profile.ranker_bypassed else self._ranker
            encoder = None if profile.family == "noAE" else self._encoders[profile.family]
            with torch.inference_mode():
                inner = self._codec.encode(view, c2, ranker=ranker,
                                           ae_encoder=encoder, timing=timing)
            _require(isinstance(inner, bytes) and inner, "codec returned no payload")
        envelope = ExecutionEnvelopeV3(
            mode_id=profile.mode_id, q_e4=profile.q_e4,
            keep_count=profile.keep_count, anchor_action_id=profile.action_id,
            reward_requested=frame.reward_requested,
            session_uuid=frame.session_uuid,
            controller_lineage_sha256=frame.controller_lineage_sha256,
            decision_seq=frame.decision_seq, ticket_seq=frame.ticket_seq,
            frame_id=frame.frame_id, tensor_seq=frame.tensor_seq,
            capture_timestamp_ns=frame.capture_timestamp_ns,
            execution_bundle_sha256=profile.execution_bundle_sha256,
            inner_payload_sha256=hashlib.sha256(inner).hexdigest(),
            inner_payload=inner)
        return PreparedFrameV2(wire_bytes=pack_envelope_v3(envelope),
                               envelope=envelope, action=action, profile=profile,
                               timing=timing.snapshot())


@dataclass(frozen=True, slots=True)
class EdgeResultV2:
    envelope: ExecutionEnvelopeV3
    action: ExecutedActionIdentity
    profile: dec.ExecutableDispatchProfile
    perception: Any
    timing: TimingTrace


class ContinuousEdgeRuntimeV2:
    """Edge side: SFD3 -> re-resolved profile -> refuse any disagreement."""

    def __init__(self, contract: dec.DynamicExecutionContract, *,
                 tail: Callable[[Any, EdgeResultV2 | None], Any],
                 ae_decoders: Mapping[str, Any], codec: Any,
                 tail_device: torch.device = torch.device("cpu")) -> None:
        _require(set(ae_decoders) == set(AE_FAMILIES), "exactly three AE decoders")
        self._contract = contract
        self._tail = tail
        self._decoders = dict(ae_decoders)
        self._codec = codec
        self._tail_device = tail_device

    def process(self, frame_bytes: bytes) -> EdgeResultV2:
        timing = StageRecorder(EDGE_STAGES)
        with timing.stage("total_edge_processing"):
            envelope = unpack_envelope_v3(frame_bytes)
            try:
                profile = self._contract.resolve_q_e4(envelope.mode_id, envelope.q_e4)
            except dec.DynamicExecutionContractError as exc:
                raise ContinuousExecutionError(f"unresolvable action: {exc}") from exc
            _require(envelope.execution_bundle_sha256 == profile.execution_bundle_sha256,
                     "execution-bundle digest mismatch")
            _require(envelope.keep_count == profile.keep_count, "keep_count mismatch")
            _require(envelope.anchor_action_id == profile.action_id,
                     "anchor identity mismatch (fabricated or dropped anchor)")
            action = executed_identity_from_profile(profile, self._contract)
            view = codec_view(profile)
            inspected = self._codec.inspect(envelope.inner_payload, timing=timing)
            try:
                require_inner_agreement(view, inspected.identity)
            except Exception as exc:  # DispatchContractError from the SFD1 layer
                raise ContinuousExecutionError(f"inner/profile mismatch: {exc}") from exc
            decoder = None if profile.family == "noAE" else self._decoders[profile.family]
            with torch.inference_mode():
                decoded = self._codec.decode(inspected, decoder=decoder,
                                             tail_device=self._tail_device,
                                             timing=timing)
            _require(decoded.finite, "reconstructed C2 is non-finite")
            with timing.stage("frozen_tail"):
                with torch.inference_mode():
                    perception = self._tail(decoded.c2, None)
        return EdgeResultV2(envelope=envelope, action=action, profile=profile,
                            perception=perception, timing=timing.snapshot())
