"""Reference decision taxonomy for analyses of captures that can contain GT-unavailable outcomes (descriptive).

Input: the existing per-decision ``DECISION_TIMELINE.csv`` (one row per POLICY decision; ``service_outcome``,
``quality_defined``, ``feedback_receipt_ms``, ``timely_170``).  Nothing is excluded: every policy decision stays in
every operational denominator.  The original screen (any feedback <=170 ms / all policy decisions) is unchanged
and is reported first; the other quantities are reported BESIDE it, never instead of it.

  any_feedback_le170         feedback (of ANY kind) received at the UE <=170 ms      / all policy decisions  [screen]
  scored_feedback_le170      service SCORED and feedback received <=170 ms           / all policy decisions
  scored                     service outcome SCORED (GT supported, scorer ran)       / all policy decisions
  scored_eligible            SCORED with >=1 eligible object (quality defined)
  scored_zero_eligible       SCORED with zero eligible objects (a legitimate result; NOT missing GT)
  gt_unavailable_geometry    NO_GT_DEFERRAL_REFUSED_GT_GEOMETRY_ENVELOPE_* (this candidate's contained refusals)
  gt_unavailable_other       any other NO_GT_* / UNSCORED_* / CANCELLED_* outcome (e.g. NO_GT_FREEZE_QUEUE_FULL)
  late                       feedback received >170 ms (any kind)
  missing_feedback           no feedback received at the UE

GT-unavailable decisions have UNKNOWN ground truth: they are neither zero-object frames nor quality zero, and they
are absent from every quality numerator AND denominator; the quality coverage (scored / all) must be reported
with any quality figure.  A fast GT-unavailable reply that arrives <=170 ms counts in ``any_feedback_le170`` (the
unchanged screen) but NOT in ``scored_feedback_le170``; it is not useful feedback.
"""
import csv
from collections import Counter

GEOMETRY_PREFIX = "NO_GT_DEFERRAL_REFUSED_GT_GEOMETRY_ENVELOPE_"


def category(row):
    outcome = row["service_outcome"] or "MISSING_SERVICE_OUTCOME"
    if outcome == "SCORED":
        return "scored_eligible" if row["quality_defined"] == "True" else "scored_zero_eligible"
    if outcome.startswith(GEOMETRY_PREFIX):
        return "gt_unavailable_geometry"
    return "gt_unavailable_other"


def summarise(rows):
    n = len(rows)
    received = [r for r in rows if r["feedback_receipt_ms"] not in ("", None)]
    timely = [r for r in rows if r["timely_170"] == "True"]
    cats = Counter(category(r) for r in rows)
    scored = cats["scored_eligible"] + cats["scored_zero_eligible"]
    count = dict(policy_decisions=n, any_feedback_le170=len(timely),
                 scored_feedback_le170=sum(r["service_outcome"] == "SCORED" for r in timely),
                 scored=scored, scored_eligible=cats["scored_eligible"], scored_zero_eligible=cats["scored_zero_eligible"],
                 gt_unavailable_geometry=cats["gt_unavailable_geometry"], gt_unavailable_other=cats["gt_unavailable_other"],
                 gt_unavailable_feedback_le170=sum(category(r).startswith("gt_unavailable") for r in timely),
                 late=len(received) - len(timely), missing_feedback=n - len(received))
    count["fractions_of_all_policy_decisions"] = {k: (v / n if n else None) for k, v in count.items()
                                                  if k != "policy_decisions"}
    count["gt_unavailable_outcomes"] = dict(Counter(r["service_outcome"] for r in rows
                                                    if category(r).startswith("gt_unavailable")))
    return count


def load(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))
