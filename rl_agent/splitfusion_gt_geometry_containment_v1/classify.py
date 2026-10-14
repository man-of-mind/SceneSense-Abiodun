"""Typed recognition of exactly the two tilt-aware AVO ground-rule-envelope refusals, and nothing else.

``splitfusion_avo_tilt_v1.quality._envelope`` refuses a tilted pedestrian whose footprint tilt exceeds
``core.GROUND_REJECT_MARGIN_M`` (0.03 m, unchanged) when it is a scored TARGET, or a structural actor that is not
proven disjoint from every target (POSSIBLE_COMPETITOR).  It raises ``LiveGtContractError`` with a structured
``geometry_diagnostics`` attribute.  An error qualifies for per-frame containment only if ALL of these hold:

* exact type ``LiveGtContractError`` (no subclass), no ``__cause__``/``__context__`` (no chained cleanup error);
* provenance: the innermost four traceback frames are exactly ``composition.assemble`` -> ``quality.numeric_avo`` ->
  ``quality.score_live_avo_people`` -> ``quality._envelope`` (code-object identity, not text);
* ``geometry_diagnostics`` = ``{"status", "offender"}`` with the exact offender schema, finite numbers, a footprint
  tilt strictly above the unchanged envelope, and role/structural reasons consistent with the status;
* the message and ``args`` equal the exact text ``_envelope`` formats from those diagnostics (integrity check).

Anything else -- other ``LiveGtContractError`` messages (transforms, timestamps, calibration, bbox rotation, corrupt
inputs), missing/malformed diagnostics, foreign raise sites, chained errors -- is NOT recognised and stays fatal.
"""
from __future__ import annotations

import math

from data_collection.route_b_publication_actor_volume_visibility_v1 import core
from rl_agent.splitfusion_avo_tilt_v1 import quality as Q
from rl_agent.splitfusion_numeric_gt_integration_v2 import composition as COM
from rl_agent.splitfusion_run5_live_avo_v1.quality_adapter import LiveGtContractError

VERSION = "splitfusion.gt_geometry_containment.v1"
STATUS_TARGET = "TILT_OUTSIDE_GROUND_RULE_ENVELOPE_TARGET"
STATUS_COMPETITOR = "TILT_OUTSIDE_GROUND_RULE_ENVELOPE_POSSIBLE_COMPETITOR"
REASONS = {STATUS_TARGET: "GT_GEOMETRY_ENVELOPE_TARGET",
           STATUS_COMPETITOR: "GT_GEOMETRY_ENVELOPE_POSSIBLE_COMPETITOR"}
OFFENDER_KEYS = frozenset(("gt_actor_id", "pitch_deg", "roll_deg", "footprint_tilt_m", "role", "structural_reasons"))
RAISE_PATH = (COM.assemble.__code__, Q.numeric_avo.__code__, Q.score_live_avo_people.__code__,
              Q._envelope.__code__)


def message_for(status, offender):
    """The exact text ``quality._envelope`` formats (kept in one place; checked against the live source in tests)."""
    return (f"tilt-aware AVO v1 does not support actor {offender['gt_actor_id']}: {status}; "
            f"pitch={offender['pitch_deg']!r}, roll={offender['roll_deg']!r}, "
            f"footprint_tilt_m={offender['footprint_tilt_m']!r}")


def _codes(traceback):
    out = []
    while traceback is not None:
        out.append(traceback.tb_frame.f_code)
        traceback = traceback.tb_next
    return out


def _finite_float(value):
    return type(value) is float and math.isfinite(value)


def recognise(error):
    """``(record, None)`` for a recognised envelope refusal, else ``(None, why)``.  Never raises."""
    try:
        return _recognise(error)
    except Exception as unexpected:                    # a classifier defect must never enable recovery
        return None, f"CLASSIFIER_ERROR:{type(unexpected).__name__}"


def _recognise(error):
    if type(error) is not LiveGtContractError:
        return None, "TYPE"
    if error.__cause__ is not None or error.__context__ is not None:
        return None, "CHAINED"
    codes = _codes(error.__traceback__)
    if len(codes) < len(RAISE_PATH) or tuple(codes[-len(RAISE_PATH):]) != RAISE_PATH:
        return None, "RAISE_SITE"
    return content(error)


def content(error):
    """The diagnostics/message checks alone (no type/chain/raise-site); used by ``recognise`` and the AVO-step
    replay of a saved failure, whose re-raised error necessarily lacks the producer composition frame."""
    diagnostics = getattr(error, "geometry_diagnostics", None)
    if type(diagnostics) is not dict or set(diagnostics) != {"status", "offender"}:
        return None, "DIAGNOSTICS_SHAPE"
    status, offender = diagnostics["status"], diagnostics["offender"]
    if status not in REASONS:
        return None, "STATUS"
    if type(offender) is not dict or set(offender) != OFFENDER_KEYS:
        return None, "OFFENDER_SHAPE"
    actor = offender["gt_actor_id"]
    if type(actor) is not str or not actor:
        return None, "OFFENDER_ID"
    if not all(_finite_float(offender[k]) for k in ("pitch_deg", "roll_deg", "footprint_tilt_m")):
        return None, "OFFENDER_NUMBERS"
    if not offender["footprint_tilt_m"] > core.GROUND_REJECT_MARGIN_M:
        return None, "INSIDE_ENVELOPE"
    reasons = offender["structural_reasons"]
    if type(reasons) is not list or not all(type(r) is str and r for r in reasons):
        return None, "STRUCTURAL_REASONS"
    if status == STATUS_TARGET and not (offender["role"] == "TARGET" and reasons == []):
        return None, "ROLE_STATUS_MISMATCH"
    if status == STATUS_COMPETITOR and not (offender["role"] == "STRUCTURAL_IGNORE" and reasons):
        return None, "ROLE_STATUS_MISMATCH"
    expected = message_for(status, offender)
    if error.args != (expected,) or str(error) != expected:
        return None, "MESSAGE_MISMATCH"
    record = dict(schema=VERSION + ".refusal", status=status, reason=REASONS[status],
                  offender=dict(offender, structural_reasons=list(reasons)), message=expected,
                  ground_rule_envelope_m=core.GROUND_REJECT_MARGIN_M)
    return record, None
