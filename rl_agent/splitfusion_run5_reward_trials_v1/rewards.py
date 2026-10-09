"""The three reward arms of this experiment (fixed before training; no sweep, no post-hoc adjustment).

Per exact quality-grid row (one scene, one measured (mode, q) anchor):

    Rv = vehicle hits within 1.0 m / eligible vehicles        (misses count; defined iff eligible vehicles > 0)
    Rp = person  hits within 1.2 m / eligible persons         (defined iff eligible persons > 0)
    Y  = (0.4 Rv [if defined] + 0.6 Rp [if defined]) / (sum of the weights of the defined classes)
    M  = min of the defined class recalls                     (one defined class -> M = Y)
    S  = the row's registered Q_seg; its multiplier is neutral (1.0) when Q_seg is undefined

    C0  value = Y * (0.7 + 0.3 S)                              (matched current reward)
    C1  value = Y * (0.9 + 0.1 S)                              (localisation-heavy)
    C2  value = (0.75 Y + 0.25 M) * (0.9 + 0.1 S)               (localisation-heavy + weaker-class safeguard)

Neither class eligible -> value undefined (exactly when Q_perc is undefined); min() is never taken over an empty set.
Off-grid q: the scalar value is linear between the two bracketing grid anchors, undefined if either is (the
existing rule). Outcome: r = value on a timely (<= 170 ms inclusive) defined reply, 0 on timeout; timely undefined is
excluded with a session reset, late undefined is a timeout (unchanged collector semantics).
No per-frame division by uncompressed quality. ``C0`` is bit-identical to the historical agreed reward
(``splitfusion_run5_restricted_pilot_v1.quality.candidate_from_row``), which is imported only for the test, unchanged.
"""
from __future__ import annotations

import hashlib
import json
from typing import Mapping, Optional

TOL = {"vehicle": 1.0, "person": 1.2}
WEIGHT = {"person": 0.6, "vehicle": 0.4}
ARMS = ("C0", "C1", "C2")
SEG_BETA = {"C0": 0.3, "C1": 0.1, "C2": 0.1}
MIN_WEIGHT = {"C0": 0.0, "C1": 0.0, "C2": 0.25}
EQUATIONS = {
    "C0": "r = D * Y * (0.7 + 0.3*S)",
    "C1": "r = D * Y * (0.9 + 0.1*S)",
    "C2": "r = D * (0.75*Y + 0.25*M) * (0.9 + 0.1*S)",
}


class RewardError(ValueError):
    pass


def definition(arm: str) -> dict:
    if arm not in ARMS:
        raise RewardError(f"unknown arm {arm!r}")
    return dict(schema="splitfusion.run5_reward_trials.reward.v1", arm=arm, equation=EQUATIONS[arm],
                tolerances_m=dict(TOL), class_weights=dict(WEIGHT), seg_beta=SEG_BETA[arm],
                min_class_weight=MIN_WEIGHT[arm], deadline_ms_inclusive=170, timeout=0.0,
                timely_undefined="EXCLUDED_EVALUATOR_FAULT_SESSION_RESET", late_undefined="TIMEOUT",
                absent_class="RENORMALIZED_OVER_ELIGIBLE_CLASSES", seg_undefined="NEUTRAL_MULTIPLIER_1",
                no_eligible_class="UNDEFINED", off_grid_q="LINEAR_BETWEEN_GRID_ANCHORS_UNDEFINED_IF_EITHER",
                state_previous_quality="Q_perc (unchanged Run-5 feature)")


def reward_sha256(arm: str) -> str:
    return hashlib.sha256(json.dumps(definition(arm), sort_keys=True).encode()).hexdigest()


def class_hits(row: Mapping) -> dict[str, tuple[int, int]]:
    """(within-tolerance hits, eligible GT) per class with eligible GT (misses stay in the denominator)."""
    out = {}
    for cls in TOL:
        eligible = int(row[f"loc_{cls}_eligible_gt"])
        if eligible:
            out[cls] = (sum(1 for e in row[f"loc_{cls}_matched_xy_errors_m"] if float(e) <= TOL[cls]), eligible)
    return out


def value_from_row(row: Mapping, arm: str) -> Optional[float]:
    if arm not in ARMS:
        raise RewardError(f"unknown arm {arm!r}")
    num = den = 0.0
    hits = class_hits(row)
    for cls in TOL:                       # same classes, order and float expression as the historical helper
        if cls in hits:
            num += WEIGHT[cls] * hits[cls][0] / hits[cls][1]
            den += WEIGHT[cls]
    if den == 0:
        if row["q_perc"] is not None:
            raise RewardError("Y undefined but Q_perc defined")
        return None
    if row["q_perc"] is None:
        raise RewardError("Y defined but Q_perc undefined")
    y = num / den
    if MIN_WEIGHT[arm] and len(hits) > 1:    # one eligible class: M = Y, so C2 = C1 exactly (no float drift)
        y = (1.0 - MIN_WEIGHT[arm]) * y + MIN_WEIGHT[arm] * min(h / e for h, e in hits.values())
    q_seg = row["q_seg"]
    beta = SEG_BETA[arm]
    modulation = 1.0 if q_seg is None else (1.0 - beta) + beta * float(q_seg)
    return y * modulation
