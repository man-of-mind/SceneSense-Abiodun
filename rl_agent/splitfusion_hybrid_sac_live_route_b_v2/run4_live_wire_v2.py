"""Phase 6 wire adapters: clock domains, SFD4 context envelope, feedback codec.

Everything here is additive and versioned; SFD1, SFD3, the direct-map and the
quality-probe wire formats are imported unchanged and never re-encoded.

* :class:`Stamp` makes the two clock domains explicit.  Physical capture wall
  time (CARLA identity and map AoI) and ``CLOCK_MONOTONIC_RAW`` (state commit,
  action open, feedback receipt and the inclusive 170-ms deadline) cannot be
  compared or subtracted across domains.
* **SFD4** = SFD3 (exact continuous action identity, SHA-protected) plus the
  exact ``FrameContextV1`` extension.  The extension bytes are produced and
  parsed by the unchanged SFD1-v2 context code, so the edge reconstructs the
  qualified context exactly; a SHA-256 covers the whole frame.
* **R4FB** is a compact, SHA-protected encoding of the Phase-4
  ``RewardFeedbackV2`` plus an evaluator reason code.
* :func:`quality_feedback` turns an evaluator measurement into Q_perc through
  the authoritative ``offline_quality_grid.quality.evaluate_exact_quality``
  bound to the Run-4 ``scientific_basis`` reward-spec file.

Importing this module starts nothing and reads no file.
"""

from __future__ import annotations

import enum
import hashlib
import math
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from rl_agent.splitfusion_live_dispatch_v1 import envelope as sfd1
from rl_agent.splitfusion_live_dispatch_v1.frame_context import (
    FrameContextV1,
    validate_frame_context,
)

from . import continuous_execution_v2 as X
from . import reward_hold_controller_v2 as R

__all__ = [
    "WireError",
    "ClockDomain",
    "Stamp",
    "wall",
    "raw",
    "pack_sfd4",
    "unpack_sfd4",
    "encode_feedback",
    "decode_feedback",
    "EvaluatorReason",
    "load_run4_quality_spec",
    "LIVE_MEASUREMENT_SEMANTICS",
    "live_measurement",
    "quality_feedback",
]


class WireError(RuntimeError):
    """A Phase-6 wire frame or clock operation failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WireError(message)


# ---------------------------------------------------------------------------
# Clock domains
# ---------------------------------------------------------------------------


class ClockDomain(str, enum.Enum):
    PHYSICAL_CAPTURE_WALL = "PHYSICAL_CAPTURE_WALL_CLOCK_REALTIME"
    MONOTONIC_RAW = "CLOCK_MONOTONIC_RAW"


@dataclass(frozen=True, slots=True)
class Stamp:
    domain: ClockDomain
    ns: int

    def __post_init__(self) -> None:
        _require(isinstance(self.domain, ClockDomain), "unknown clock domain")
        _require(type(self.ns) is int and self.ns >= 0, "stamp must be int ns >= 0")

    def _same(self, other: object) -> "Stamp":
        _require(isinstance(other, Stamp), "stamps compare only with stamps")
        _require(other.domain is self.domain,
                 f"mixed-clock operation refused: {self.domain.value} vs "
                 f"{other.domain.value}")
        return other

    def __sub__(self, other: object) -> int:
        return self.ns - self._same(other).ns

    def __lt__(self, other: object) -> bool:
        return self.ns < self._same(other).ns

    def __le__(self, other: object) -> bool:
        return self.ns <= self._same(other).ns

    def __gt__(self, other: object) -> bool:
        return self.ns > self._same(other).ns

    def __ge__(self, other: object) -> bool:
        return self.ns >= self._same(other).ns

    def to_dict(self) -> dict[str, Any]:
        return {"domain": self.domain.value, "ns": self.ns}


def wall(ns: int) -> Stamp:
    return Stamp(ClockDomain.PHYSICAL_CAPTURE_WALL, ns)


def raw(ns: int) -> Stamp:
    return Stamp(ClockDomain.MONOTONIC_RAW, ns)


# ---------------------------------------------------------------------------
# SFD4 = SFD3 + exact FrameContextV1 (SFD1-v2 context bytes), digest-protected
# ---------------------------------------------------------------------------

MAGIC_V4 = b"SFD4"
VERSION_V4 = 4
_HEAD_V4 = struct.Struct("<4sHHII")      # magic, version, reserved, ctx, sfd3
_DIGEST = 32


def _context_extension(context: FrameContextV1) -> bytes:
    """The exact SFD1-v2 context extension, produced by the unchanged packer."""
    validate_frame_context(context)
    frame = sfd1.pack_envelope(
        b"\x00", action_id=0, sequence_id=context.sequence_id,
        capture_timestamp_ns=context.capture_timestamp_ns, frame_context=context)
    header_bytes = struct.unpack_from("<4sHH", frame)[2]
    return frame[sfd1.HEADER_BYTES:header_bytes]


def pack_sfd4(envelope: X.ExecutionEnvelopeV3, context: FrameContextV1) -> bytes:
    """Bind one continuous action envelope to its exact frame context."""
    _require(isinstance(context, FrameContextV1), "context must be FrameContextV1")
    _require(context.frame_id == envelope.frame_id,
             "context frame_id differs from the action envelope")
    _require(context.sequence_id == envelope.tensor_seq,
             "context sequence_id must equal the envelope tensor_seq")
    _require(context.capture_timestamp_ns == envelope.capture_timestamp_ns,
             "context capture wall time differs from the action envelope")
    extension = _context_extension(context)
    inner = X.pack_envelope_v3(envelope)
    head = _HEAD_V4.pack(MAGIC_V4, VERSION_V4, 0, len(extension), len(inner))
    body = head + extension + inner
    return body + hashlib.sha256(body).digest()


def unpack_sfd4(frame: bytes) -> Tuple[X.ExecutionEnvelopeV3, FrameContextV1]:
    _require(isinstance(frame, (bytes, bytearray, memoryview)), "frame must be bytes")
    data = bytes(frame)
    _require(len(data) >= _HEAD_V4.size + _DIGEST, "frame shorter than SFD4 header")
    _require(data[:4] == MAGIC_V4, f"not an SFD4 frame (magic {data[:4]!r})")
    body, digest = data[:-_DIGEST], data[-_DIGEST:]
    _require(hashlib.sha256(body).digest() == digest, "SFD4 digest mismatch")
    magic, version, reserved, ctx_len, inner_len = _HEAD_V4.unpack_from(body)
    _require(version == VERSION_V4 and reserved == 0, "SFD4 version/reserved drift")
    _require(len(body) == _HEAD_V4.size + ctx_len + inner_len, "SFD4 length mismatch")
    extension = body[_HEAD_V4.size:_HEAD_V4.size + ctx_len]
    try:
        envelope = X.unpack_envelope_v3(body[_HEAD_V4.size + ctx_len:])
    except X.ContinuousExecutionError as exc:
        raise WireError(f"SFD4 action envelope refused: {exc}") from exc
    try:
        context = sfd1._unpack_context(  # noqa: SLF001 - exact reviewed parser
            extension, sequence_id=envelope.tensor_seq,
            capture_timestamp_ns=envelope.capture_timestamp_ns)
    except sfd1.EnvelopeError as exc:
        raise WireError(f"SFD4 frame context refused: {exc}") from exc
    _require(context.frame_id == envelope.frame_id,
             "SFD4 context frame_id differs from the action envelope")
    _require(_context_extension(context) == extension,
             "SFD4 context is not in canonical SFD1-v2 form")
    return envelope, context


# ---------------------------------------------------------------------------
# R4FB: compact RewardFeedbackV2
# ---------------------------------------------------------------------------

MAGIC_FB = b"R4FB"
VERSION_FB = 1
_KIND_CODES = {"DELIVERED_SUCCESS": 1, "REGISTERED_DELIVERY_FAILURE": 2,
               "REGISTERED_SERVICE_FAILURE": 3, "INFRASTRUCTURE_FAULT": 4,
               "EVALUATOR_FAULT": 5}
_KIND_NAMES = {code: name for name, code in _KIND_CODES.items()}
_FB = struct.Struct("<4sHBBBBHIQQQQQ16s32s32sd")
_NO_ANCHOR = 0xFFFFFFFF
_FLAG_ANCHOR, _FLAG_REWARD, _FLAG_QPERC = 0x01, 0x02, 0x04


class EvaluatorReason(enum.IntEnum):
    NONE = 0
    QUALITY_UNDEFINED_NO_ELIGIBLE_GT = 1
    EVALUATOR_EXCEPTION = 2
    GROUND_TRUTH_UNAVAILABLE = 3


def encode_feedback(fb: R.RewardFeedbackV2,
                    reason: EvaluatorReason = EvaluatorReason.NONE) -> bytes:
    _require(isinstance(fb, R.RewardFeedbackV2), "feedback must be RewardFeedbackV2")
    _require(fb.reward_requested is True, "only reward-requested frames get feedback")
    session = uuid.UUID(fb.session_uuid)
    _require(str(session) == fb.session_uuid, "session UUID must be canonical")
    flags = ((_FLAG_ANCHOR if fb.anchor_action_id is not None else 0)
             | _FLAG_REWARD
             | (_FLAG_QPERC if fb.q_perc is not None else 0))
    q = 0.0 if fb.q_perc is None else float(fb.q_perc)
    _require(math.isfinite(q) and 0.0 <= q <= 1.0, "q_perc must lie in [0, 1]")
    body = _FB.pack(
        MAGIC_FB, VERSION_FB, _KIND_CODES[fb.kind], flags, int(reason), fb.mode_id,
        fb.q_e4, _NO_ANCHOR if fb.anchor_action_id is None else fb.anchor_action_id,
        fb.decision_seq, fb.ticket_seq, fb.frame_id, fb.tensor_seq,
        fb.capture_timestamp_ns, session.bytes,
        bytes.fromhex(fb.controller_lineage_sha256),
        bytes.fromhex(fb.execution_bundle_sha256), q)
    return body + hashlib.sha256(body).digest()


def decode_feedback(payload: bytes) -> Tuple[R.RewardFeedbackV2, EvaluatorReason]:
    _require(isinstance(payload, (bytes, bytearray)), "feedback must be bytes")
    data = bytes(payload)
    _require(len(data) == _FB.size + _DIGEST, "R4FB length mismatch")
    body, digest = data[:_FB.size], data[_FB.size:]
    _require(hashlib.sha256(body).digest() == digest, "R4FB digest mismatch")
    (magic, version, kind, flags, reason, mode_id, q_e4, anchor, decision, ticket,
     frame_id, tensor, capture, session, lineage, bundle, q) = _FB.unpack(body)
    _require(magic == MAGIC_FB and version == VERSION_FB, "not an R4FB v1 frame")
    _require(kind in _KIND_NAMES, "unknown R4FB kind")
    _require(flags & ~(_FLAG_ANCHOR | _FLAG_REWARD | _FLAG_QPERC) == 0, "unknown flags")
    _require(bool(flags & _FLAG_ANCHOR) == (anchor != _NO_ANCHOR), "anchor flag drift")
    _require(bool(flags & _FLAG_REWARD), "feedback must name a reward request")
    try:
        reason_code = EvaluatorReason(reason)
    except ValueError as exc:
        raise WireError("unknown evaluator reason") from exc
    fb = R.RewardFeedbackV2(
        session_uuid=str(uuid.UUID(bytes=session)),
        controller_lineage_sha256=lineage.hex(), decision_seq=decision,
        ticket_seq=ticket, frame_id=frame_id, tensor_seq=tensor,
        capture_timestamp_ns=capture, mode_id=mode_id, q_e4=q_e4,
        execution_bundle_sha256=bundle.hex(),
        anchor_action_id=None if anchor == _NO_ANCHOR else anchor,
        reward_requested=True, kind=_KIND_NAMES[kind],
        q_perc=q if flags & _FLAG_QPERC else None)
    return fb, reason_code


# ---------------------------------------------------------------------------
# Authoritative Q_perc bridge
# ---------------------------------------------------------------------------


def load_run4_quality_spec(repo_root: Path):
    """The approved Run-4 Q_perc spec (scientific_basis-verified file)."""
    from rl_agent.splitfusion_hybrid_sac_run4_v1 import scientific_basis as SB
    from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality as Q

    digest = SB.verify_quality_source(Path(repo_root).resolve())
    return Q.load_reward_spec(Path(repo_root) / SB.QUALITY_SOURCE_RELATIVE_PATH, digest)


LIVE_MEASUREMENT_SEMANTICS = (
    "LIVE_CARLA_GT_QUALITY_EVALUATOR__GREEDY_MATCH_WITHIN_MATCH_DISTANCE__ALL_"
    "LIVE_GT_ELIGIBLE__NOT_THE_OFFLINE_AVO_ELIGIBILITY_OF_THE_TRAINING_GRID"
)


def live_measurement(*, frame_id: int, predicted_mask: Any, ground_truth_mask: Any,
                     predictions: Any, ground_truth_objects: Any,
                     match_distance_m: float) -> dict[str, Any]:
    """Live evaluator output -> the sufficient statistics Q_perc consumes.

    Uses the live quality evaluator's own class ids, nearest-neighbour GT
    resize and ``greedy_match_predictions`` matcher (the functions behind
    ``scoring.score_segmentation`` / ``score_localization``), and returns
    exactly the fields ``evaluate_exact_quality`` reads.  Person eligibility is
    every live person GT object: the offline AVO table used by the training
    grid has no live counterpart (see ``LIVE_MEASUREMENT_SEMANTICS``).
    """
    import cv2
    import numpy as np

    from rl_agent.splitfusion_quality_feedback_probe_v1 import scoring

    predicted = scoring.immutable_mask(predicted_mask)
    truth = scoring.immutable_mask(ground_truth_mask)
    if predicted.shape != truth.shape:
        truth = cv2.resize(truth, (predicted.shape[1], predicted.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    output: dict[str, Any] = {"frame_id": int(frame_id)}
    for name, class_id in (("vehicle", scoring.CLASS_ID_VEHICLE),
                           ("person", scoring.CLASS_ID_PERSON)):
        gt = truth == class_id
        pred = predicted == class_id
        intersection = int(np.logical_and(gt, pred).sum())
        gt_pixels, pred_pixels = int(gt.sum()), int(pred.sum())
        output.update({
            f"seg_{name}_gt_pixels": gt_pixels,
            f"seg_{name}_pred_pixels": pred_pixels,
            f"seg_{name}_intersection_pixels": intersection,
            f"seg_{name}_union_pixels": gt_pixels + pred_pixels - intersection,
        })
    rows = scoring.immutable_rows(predictions)
    objects = scoring.immutable_rows(ground_truth_objects)
    for name in ("vehicle", "person"):
        preds = [row for row in rows if row.get("class_name") == name]
        targets = [row for row in objects if row.get("class_name") == name]
        matches = scoring.greedy_match_predictions(
            preds, targets, max_distance_m=float(match_distance_m), class_aware=True)
        output.update({
            f"loc_{name}_eligible_gt": len(targets),
            f"loc_{name}_tp": len(matches),
            f"loc_{name}_fp": len(preds) - len(matches),
            f"loc_{name}_fn": len(targets) - len(matches),
            f"loc_{name}_matched_xy_errors_m": [float(d) for _p, _t, d in matches],
        })
    return output


def quality_feedback(spec: Any, measurement: Mapping[str, Any],
                     envelope: X.ExecutionEnvelopeV3
                     ) -> Tuple[R.RewardFeedbackV2, EvaluatorReason]:
    """Evaluator measurement for one reward-requested frame -> feedback.

    Undefined Q_perc (no eligible GT) is not a zero-quality frame and not a
    policy failure: it is an excluded ``EVALUATOR_FAULT`` with an explicit
    reason.  Any evaluator exception is also excluded, never a timeout.
    """
    from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality as Q

    _require(envelope.reward_requested, "held/fallback frames are never evaluated")
    _require(int(measurement["frame_id"]) == envelope.frame_id,
             "measurement frame differs from the reward-requested frame")
    try:
        exact = Q.evaluate_exact_quality(spec, measurement)
        reason = (EvaluatorReason.NONE if exact is not None
                  else EvaluatorReason.QUALITY_UNDEFINED_NO_ELIGIBLE_GT)
    except Exception:  # noqa: BLE001 - classified, never silently rewarded
        exact, reason = None, EvaluatorReason.EVALUATOR_EXCEPTION
    kind = "DELIVERED_SUCCESS" if exact is not None else "EVALUATOR_FAULT"
    fb = R.RewardFeedbackV2(
        session_uuid=envelope.session_uuid,
        controller_lineage_sha256=envelope.controller_lineage_sha256,
        decision_seq=envelope.decision_seq, ticket_seq=envelope.ticket_seq,
        frame_id=envelope.frame_id, tensor_seq=envelope.tensor_seq,
        capture_timestamp_ns=envelope.capture_timestamp_ns,
        mode_id=envelope.mode_id, q_e4=envelope.q_e4,
        execution_bundle_sha256=envelope.execution_bundle_sha256,
        anchor_action_id=envelope.anchor_action_id, reward_requested=True,
        kind=kind, q_perc=None if exact is None else float(exact.q_perc))
    return fb, reason
