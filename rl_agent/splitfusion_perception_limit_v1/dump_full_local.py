"""Re-run the sealed FULL_LOCAL branch on the 49 retained v11 inputs and keep per-object evidence.

Diagnostic only. Identical joins/adapter/evaluator as rl_agent/splitfusion_paired_baselines_v1/run.py, but
only the dense branch is scored and the complete evaluator evidence (prediction records, vehicle targets, AVO
person rows, matched pairs), the native labels, the GT target grids and the pre-p025 post-NMS candidates
(captured read-only from the v3 tail's call into apply_p025_service_policy_v3) are persisted.
Reproduction gate: dense_records_sha256 / dense_labels_array_sha256 and the full-local measurement must equal
the sealed REPLAY.json row exactly; any difference marks the frame NOT_REPRODUCED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rl_agent.splitfusion_paired_quality_replay_v2 import analyze as A
from rl_agent.splitfusion_paired_quality_replay_v2.model_adapter import load_adapter
from rl_agent.splitfusion_paired_quality_replay_v1 import capture as C
from rl_agent.splitfusion_paired_quality_replay_v1 import replay as RP
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_edge_optimization_v1 import optimized_tail_v3 as T3

PAIRED = Path("experiments/splitfusion_run5_paired_baselines_v1/20261006_125848_v11_treatment")
ARCHIVE = Path("experiments/splitfusion_gt_main_thread_live_trial_v1/20261005_v11/treatment/capture/cells/a71__favorable_stable")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    out = a.output.absolute()
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir()
    sealed = json.loads((PAIRED / "replay/REPLAY.json").read_text())
    sealed_rows = {r["frame_id"]: r for r in sealed["rows"]}
    manifest = C.verify_archive(ARCHIVE / "paired_input_archive", expected_manifest_sha256=sealed["input_manifest_sha256"])
    E.verify_sources()
    pins = json.loads((PAIRED / "binding/CELL_PINS.json").read_text())
    cell = A.load_cell(ARCHIVE, pins)
    join_cell = dict(cell, evaluations={})
    adapter = load_adapter(**json.loads((PAIRED / "binding/MODEL_OPTIONS.json").read_text()))

    captured = []
    original_policy = T3.apply_p025_service_policy_v3

    def observing_policy(detections, components):
        captured.append({k: v.detach().cpu().clone() for k, v in detections.items()})
        return original_policy(detections, components)

    T3.apply_p025_service_policy_v3 = observing_policy
    summary = []
    try:
        for frame in manifest["frames"]:
            ident = frame["identity"]
            fid = ident["frame_id"]
            joined = A.join_frame(frame, join_cell)
            wire = (ARCHIVE / "paired_input_archive" / frame["wire_file"]).read_bytes()
            A.verify_wire_source_pose(wire, joined)
            array = np.load(ARCHIVE / "paired_input_archive" / frame["input_file"], allow_pickle=False)
            captured.clear()
            pair = RP.replay_pair(prepared_input=torch.from_numpy(array), selected_wire=wire,
                                  expected_identity=ident, archived_selected=joined["archived_selected"],
                                  **adapter.replay_callbacks())
            assert len(captured) == 2, len(captured)  # selected tail, then dense tail
            pre = captured[1]
            dense = pair.dense
            ev = E.evaluate_prepared(identity=frame["identity"], prepared=joined["prepared"],
                                     predicted_labels=dense.labels, predictions=dense.records, avo=joined["avo"])
            s = sealed_rows[fid]
            reproduced = (pair.evidence["dense_records_sha256"] == s["parity"]["dense_records_sha256"]
                          and pair.evidence["dense_labels_array_sha256"] == s["parity"]["dense_labels_array_sha256"]
                          and dict(ev.measurement) == s["full_local"]["measurement"])
            pre_rows = []
            n = pre["scores"].numel()
            for i in range(n):
                row = {"index": i, "label_internal": int(pre["labels_internal"][i]), "score": float(pre["scores"][i]),
                       "box": [float(x) for x in pre["boxes"][i].tolist()]}
                if "world_xyz" in pre:
                    row["world_xyz"] = [float(x) for x in pre["world_xyz"][i].tolist()]
                pre_rows.append(row)
            evidence = json.loads(E.targets.canonical(ev.evidence))
            evidence["pre_p025_post_nms_candidates"] = pre_rows
            evidence["pre_p025_fields"] = sorted(pre)
            evidence["reproduced_sealed_full_local"] = reproduced
            evidence["source_input_file"] = frame["input_file"]
            (out / "frames" / f"frame_{fid}.json").write_text(json.dumps(evidence, sort_keys=True))
            tgt = joined["prepared"].target
            np.savez_compressed(out / "frames" / f"frame_{fid}.npz", pred_labels=dense.labels,
                                gt_segmentation=tgt["segmentation"], gt_object_ignore=tgt["object_ignore"])
            summary.append({"frame_id": fid, "reproduced": reproduced})
            print(fid, "REPRODUCED" if reproduced else "NOT_REPRODUCED", flush=True)
    finally:
        T3.apply_p025_service_policy_v3 = original_policy
    (out / "DUMP_SUMMARY.json").write_text(json.dumps({
        "frames": summary, "reproduced": sum(r["reproduced"] for r in summary), "total": len(summary),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sealed_replay_sha256": hashlib.sha256((PAIRED / "replay/REPLAY.json").read_bytes()).hexdigest(),
        "input_manifest_sha256": sealed["input_manifest_sha256"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
