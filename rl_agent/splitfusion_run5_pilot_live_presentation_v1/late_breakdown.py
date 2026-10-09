"""Late-feedback breakdown of a pilot live run from its per-decision rows (per-decision chains; no added medians).

Classes for LATE decisions (UE receipt > 170 ms), on two input-readiness boundaries:
  prediction at the service (``service_prediction_received``) and GT publication (``gt_published``) -- GT
  publication is the producer handing the ready target to the service; scorer ADMISSION of that target is the
  separate scorer TARGET request (``target_worker_begin`` .. ``target_worker_end``, which includes the target's
  validation and, when the prediction is already waiting, its scoring -- a request duration, not pure scoring time).
Usage: python -m rl_agent.splitfusion_run5_pilot_live_presentation_v1.late_breakdown <DECISIONS.jsonl> <out.json>
"""
import collections
import json
import sys

import numpy as np

D = 170.0


def dist(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    return dict(n=int(len(v))) if not len(v) else dict(n=int(len(v)), p50=round(float(np.median(v)), 2),
        p90=round(float(np.percentile(v, 90)), 2), max=round(float(v.max()), 2))


def analyze(rows):
    late = [r for r in rows if r["ue_receipt"] is not None and r["ue_receipt"] > D]
    cls = collections.Counter()
    for r in late:
        cls[("PREDICTION_LATE" if r["service_prediction_received"] > D else "PREDICTION_ON_TIME") + "__" +
            ("GT_PUBLISHED_LATE" if r["gt_published"] > D else "GT_PUBLISHED_ON_TIME")] += 1
    both = [r for r in late if r["service_prediction_received"] <= D and r["gt_published"] <= D]
    out = dict(decisions=len(rows), late=len(late), late_within_10ms=sum(r["ue_receipt"] <= D + 10 for r in late),
               classes=dict(cls), both_inputs_on_time=dict(n=len(both),
               final_scorer_request=dict(collections.Counter(r["scoring_request_op"] for r in both))))
    for op in ("PREDICTION", "TARGET"):
        s = [r for r in both if r["scoring_request_op"] == op]
        chain = dict(n=len(s), last_input_ready=dist([max(r["service_prediction_received"], r["gt_published"]) for r in s]),
                     scoring_end_to_emitted=dist([r["feedback_emitted"] - r["scoring_worker_end"] for r in s]),
                     emitted_to_ue_receipt=dist([r["ue_receipt"] - r["feedback_emitted"] for r in s]),
                     overshoot=dist([r["ue_receipt"] - D for r in s]))
        if op == "PREDICTION":
            chain.update(queue_wait=dist([r["prediction_worker_begin"] - r["service_prediction_received"] for r in s]),
                         request_duration=dist([r["prediction_worker_end"] - r["prediction_worker_begin"] for r in s]),
                         target_admission_still_running_at_prediction_arrival=sum(
                             r["target_worker_end"] > r["service_prediction_received"] for r in s))
        else:
            chain.update(gt_publication_to_admission_start=dist([r["target_worker_begin"] - r["gt_published"] for r in s]),
                         admission_request_duration=dist([r["target_worker_end"] - r["target_worker_begin"] for r in s]))
        out["both_inputs_on_time"][op] = chain
    out["all_decisions"] = dict(
        gt_publication_to_admission_start=dist([r["target_worker_begin"] - r["gt_published"] for r in rows]),
        target_admission_request_duration=dist([r["target_worker_end"] - r["target_worker_begin"] for r in rows]),
        prediction_request_duration=dist([r["prediction_worker_end"] - r["prediction_worker_begin"] for r in rows
                                          if r["prediction_worker_begin"] is not None]))
    on_path = [r for r in both if r["scoring_request_op"] == "TARGET" or r["target_worker_end"] > r["service_prediction_received"]]
    out["target_admission_on_or_blocking_critical_path"] = len(on_path)
    out["upper_bound_flips"] = {f"{s}ms": dict(on_path=sum(r["ue_receipt"] - D <= s for r in on_path),
                                               any_late=sum(r["ue_receipt"] - D <= s for r in late)) for s in (1.0, 1.6, 2.6)}
    return out


if __name__ == "__main__":
    rows = [json.loads(line) for line in open(sys.argv[1])]
    result = analyze(rows)
    open(sys.argv[2], "x").write(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps({k: result[k] for k in ("late", "late_within_10ms", "classes", "target_admission_on_or_blocking_critical_path")}))
