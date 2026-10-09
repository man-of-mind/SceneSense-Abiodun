"""Read-only sealed fixtures with the ACTUAL archived UE/edge FrameContext.

This does not recreate a live sensor source. Numeric producer fixture limits
(notably some archived actor-pose components) remain those of the sealed loader.
"""
import json

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_producer_v1 import cases as K
from rl_agent.splitfusion_live_dispatch_v1.frame_context import FrameContextV1, Pose6D, validate_frame_context
from rl_agent.splitfusion_live_parity_runtime_v1 import pose


def context_from_document(document):
    C.require(set(document) == {"stream_id", "frame_id", "sequence_id", "capture_timestamp_ns",
        "ego_world", "camera_model_sha256", "camera_mount_sha256"}, "complete archived context required")
    value = FrameContextV1(**{k: v for k, v in document.items() if k != "ego_world"},
                           ego_world=Pose6D(*document["ego_world"]))
    validate_frame_context(value)
    C.require(pose.context_document(value) == document, "context archive conversion changed values")
    return value


def load_pose_contexts(cases, predictions):
    wanted = {case.frame_id for case in cases}
    contexts = {}
    for side, relative, kind in (("ue", "pose_archive/pose_evidence.jsonl", "UE_WIRE"),
            ("edge", "profiler/edge/pose_archive/pose_evidence.jsonl", "EDGE_WIRE")):
        rows = [json.loads(line) for line in (K.CELL / relative).read_text().splitlines()]
        selected = {}
        for row in rows:
            if row.get("kind") != kind or row["gt_identity"]["frame_id"] not in wanted:
                continue
            frame = row["gt_identity"]["frame_id"]
            C.require(frame not in selected, "duplicate archived wire identity")
            pred = predictions[frame]["identity"]
            C.require(row["gt_identity"] == pred["gt_identity"]
                and row["action_identity"] == pred["action_identity"], "archived prediction/wire differs")
            selected[frame] = row
        C.require(set(selected) == wanted, "incomplete actual archived wire contexts")
        contexts[side] = selected
    out = {}
    for case in cases:
        frame = case.frame_id
        ue, edge = contexts["ue"][frame], contexts["edge"][frame]
        C.require(ue["context"] == edge["context"], "UE/edge FrameContext differs")
        context = context_from_document(edge["context"])
        delta = pose._agree(pose.context_camera(context), case.kwargs["camera_matrix"],
                            "archived actual FrameContext/source")
        out[frame] = dict(context=context, camera_delta=delta, ue=ue, edge=edge)
    return out


def plan_for(case, prediction, context):
    # Replay plan, not a claim that this exact planning document existed live.
    return C.canonical(dict(scope="OFFLINE_V2_REPLAY_PLAN_WITH_ACTUAL_RETAINED_WIRE_CONTEXT",
        capture=case.kwargs["identity"], action=prediction["identity"]["action_identity"],
        frame_context=pose.context_document(context)))


def admission_arguments(case, prediction, context, *, run_binding, producer_binding):
    return dict(expected_capture_identity=case.kwargs["identity"],
        expected_action_identity=prediction["identity"]["action_identity"], frame_context=context,
        plan_sha256=C.sha256(plan_for(case, prediction, context)),
        run_binding=run_binding, producer_binding=producer_binding,
        prediction_contract=K.read(K.CELL / "profiler/edge/prediction_archive/MANIFEST.json")["prediction_contract"])
