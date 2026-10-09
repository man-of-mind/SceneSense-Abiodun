"""Edge-only process-local quality seam; unchanged quality/reward arithmetic."""
from __future__ import annotations

import time
from pathlib import Path

from . import live_gt
from .quality_adapter import person_avo_measurement

_INSTALLED = False


def measurement(*, frame_id, predicted_mask, ground_truth_mask, predictions,
                ground_truth_objects, match_distance_m, avo):
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    value = W.live_measurement(frame_id=frame_id, predicted_mask=predicted_mask,
        ground_truth_mask=ground_truth_mask, predictions=predictions,
        ground_truth_objects=ground_truth_objects, match_distance_m=match_distance_m)
    value.update(person_avo_measurement(predictions, avo))
    value["live_gt_contract"] = live_gt.MEASUREMENT_SEMANTICS
    return value


def install_edge_quality_seams() -> dict:
    global _INSTALLED
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as G

    if not _INSTALLED:
        original_read = G.read_ground_truth

        def read(directory, *, expected_identity, timeout_s):
            start = time.monotonic()
            # AVO is the last published object-GT component. Wait for it first.
            extra = live_gt.read_avo_bundle(Path(directory), expected_identity=expected_identity, timeout_s=timeout_s)
            gt = original_read(directory, expected_identity=expected_identity,
                               timeout_s=max(.000001, timeout_s - (time.monotonic() - start)))
            gt.update(extra)
            gt["gt_ready_wall_ns"] = max(gt["gt_ready_wall_ns"], extra["avo_ready_wall_ns"])
            return gt

        def evaluate(self, ticket, gt, timing):
            timing["evaluator_start_wall_ns"] = time.time_ns()
            value = measurement(frame_id=ticket.context.frame_id, predicted_mask=ticket.predicted_mask,
                ground_truth_mask=gt["semantic"], predictions=list(ticket.records),
                ground_truth_objects=list(gt["objects"]), match_distance_m=self._match, avo=gt["avo"])
            value["avo_sidecar_sha256"] = gt["avo_sidecar_sha256"]
            feedback, reason = W.quality_feedback(self._spec, value, ticket.envelope)
            timing.update(evaluator_end_wall_ns=time.time_ns(), gt_ready_wall_ns=gt.get("gt_ready_wall_ns"),
                          avo_sidecar_sha256=gt["avo_sidecar_sha256"])
            self._emit(ticket, feedback, reason, measurement=value, timing=timing)

        G.read_ground_truth = read
        E.Run4EvaluatorV2._evaluate_with = evaluate
        E.Run4EvaluatorV2.GT_COMPONENTS = (*E.Run4EvaluatorV2.GT_COMPONENTS, "avo.json")
        W.LIVE_MEASUREMENT_SEMANTICS = live_gt.MEASUREMENT_SEMANTICS
        _INSTALLED = True
    return {"measurement_semantics": live_gt.MEASUREMENT_SEMANTICS,
            "reward_arithmetic": "UNCHANGED", "evaluation_inside_170_ms": True}
