"""Profile retained v1 GT functions on all 45 saved v4 frames, CPU only.

The function-profile pass is distinct from ordinary timings. No policy, world,
radio, CUDA, process-affinity change or live clock modification occurs. The
saved observation stamp is supplied only for exact source-evidence replay.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pstats
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch

from rl_agent.splitfusion_live_gt_optimization_v1 import replay_source as R
from rl_agent.splitfusion_live_gt_optimization_v1 import runtime as V1
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_parity_repair_v1 import gt_bridge as G
from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG

ROOT = Path(__file__).resolve().parents[2]
CELL = ROOT / "experiments/splitfusion_live_parity_runtime_v4/20261003_optimization_v4/qualification_100/capture/cells/a71__favorable_stable"
STAGES = ("avo", "source", "target", "publish", "archive")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def load_cases():
    reference = R.load_reference()
    # This is a new private module instance, not a change to the sealed helper.
    reference.CELL = CELL
    records = json.loads((CELL / "run5_live_avo_gt_summary.json").read_text())["records"]
    assert len(records) == 45
    cases = []
    for record in records:
        bundle, evidence, kwargs = reference.owned(record)
        avo_path = CELL / "run5_live_avo_inputs" / Path(record["path"]).name
        if sha(avo_path) != record["sha256"]:
            raise ValueError("retained AVO hash changed")
        avo = json.loads(avo_path.read_text())
        # The legacy replay helper only needed yaw for complete target geometry.
        # AVO additionally uses the retained pitch/roll fields of pedestrians.
        people = {int(row["gt_actor_id"]): row for row in avo["raw_pedestrian_rows"]}
        for actor in kwargs["frozen_world"].get_actors():
            if actor.id in people:
                row = people[actor.id]
                rotation = actor.get_transform().rotation
                rotation.yaw = row["object_yaw_deg"]
                rotation.pitch = row["object_pitch_deg"]
                rotation.roll = row["object_roll_deg"]
        cases.append((bundle, evidence, kwargs, record, avo))
    return reference, cases


def timed(fn, profile=None):
    wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
    if profile is not None:
        profile.enable()
    try:
        result = fn()
    finally:
        if profile is not None:
            profile.disable()
    return result, {"wall_ms": (time.perf_counter_ns() - wall) / 1e6,
                    "thread_cpu_ms": (time.thread_time_ns() - cpu) / 1e6}


def run_pass(reference, cases, scratch, *, profiled=False):
    profiles = {stage: cProfile.Profile() for stage in STAGES} if profiled else {}
    rows = []
    with V1.install_cpu_gt_optimizations(expected_sources=V1.source_inventory()):
        producer = reference.producer(*cases[0][:2])
        for bundle, evidence, kwargs, record, saved_avo in cases:
            population = evidence["dynamic_snapshot_evidence"]
            frame = kwargs["identity"]["frame_id"]
            if producer.population != frozenset(population["registered_population_ids"]):
                producer.register_population(population["registered_population_ids"],
                    population["population_sha256"], before_frame_id=frame)
            avo_kwargs = {key: kwargs[key] for key in ("identity", "rgb", "semantic_image",
                "depth_image", "frozen_world", "camera_matrix", "camera_inverse")}
            avo_record, avo_cost = timed(lambda: LG.write_avo_bundle(scratch,
                **avo_kwargs, policy_timestamp_s=kwargs["rgb"]["timestamp"]), profiles.get("avo"))
            replayed_avo = json.loads(Path(avo_record["path"]).read_text())
            assert replayed_avo["avo"] == saved_avo["avo"], f"AVO changed frame {frame}"
            assert replayed_avo["raw_pedestrian_rows"] == saved_avo["raw_pedestrian_rows"]
            with patch("rl_agent.splitfusion_run5_live_avo_v1.gt_source_sync.time.clock_gettime_ns",
                       return_value=evidence["source_synchronization"]["observed_raw_ns"]):
                inputs, source_cost = timed(lambda: producer.build(**kwargs), profiles.get("source"))
            assert inputs.provenance == bundle.inputs.provenance, f"source provenance changed frame {frame}"
            assert G.canonical(producer.last_evidence) == G.canonical(evidence)
            prepared, target_cost = timed(lambda: E.prepare_target(inputs), profiles.get("target"))
            assert prepared.content_sha256 == bundle.prepared.content_sha256
            for key in ("segmentation_png", "object_ignore_png"):
                assert prepared.target[key] == bundle.prepared.target[key]
            desc, publish_cost = timed(lambda: T.write_bundle(scratch, identity=kwargs["identity"],
                inputs=inputs, prepared=prepared), profiles.get("publish"))
            for name, detail in bundle.manifest["payloads"].items():
                assert sha(Path(desc["directory"]) / name) == detail["sha256"]
            aux, archive_cost = timed(lambda: G.write_source_archive(
                Path(desc["directory"]).with_suffix(".sources"), inputs=inputs,
                evidence=producer.last_evidence, rgb=kwargs["rgb"], radar=kwargs["radar_world_xyz"]), profiles.get("archive"))
            original = CELL / "parity_complete_targets" / Path(record["complete_sources"]["directory"]).name
            assert R.tree(original) == R.tree(Path(aux["directory"]))
            timings = dict(zip(STAGES, (avo_cost, source_cost, target_cost, publish_cost, archive_cost)))
            rows.append({"frame_id": frame, "timings": timings, "exact": True,
                         "through_ready_ms": sum(timings[k]["wall_ms"] for k in STAGES[:-1]),
                         "through_archive_ms": sum(t["wall_ms"] for t in timings.values())})
    return rows, profiles


def stats(profile):
    result = []
    for (file, line, name), (primitive, calls, own, cumulative, _callers) in pstats.Stats(profile).stats.items():
        result.append({"file": file, "line": line, "name": name, "calls": calls,
                       "primitive_calls": primitive, "own_seconds": own, "cumulative_seconds": cumulative})
    return sorted(result, key=lambda row: row["cumulative_seconds"], reverse=True)


def quantiles(values):
    return {name: float(np.quantile(values, q)) for name, q in (("min", 0), ("p50", .5), ("p95", .95), ("max", 1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only replay required")
    before = R.tree(CELL / "parity_complete_targets")
    sources = V1.source_inventory()
    reference, cases = load_cases()
    for label, profiled in (("unprofiled", False), ("function_profile", True)):
        with tempfile.TemporaryDirectory(prefix="gt-v2-cost-") as temporary:
            rows, profiles = run_pass(reference, cases, Path(temporary), profiled=profiled)
        write(args.output / (label + ".json"), rows)
        for stage, profile in profiles.items():
            write(args.output / (stage + "_functions.json"), stats(profile))
            stream = io.StringIO()
            pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(60)
            with (args.output / (stage + "_profile.txt")).open("x") as target:
                target.write(stream.getvalue())
        print(json.dumps({"pass": label, "frames": len(rows), "exact": True}), flush=True)
    assert not torch.cuda.is_initialized()
    assert before == R.tree(CELL / "parity_complete_targets")
    V1.verify_sources(sources)
    rows = json.loads((args.output / "unprofiled.json").read_text())
    result = {"status": "ALL45_SAVED_GT_REPLAY_EXACT__FUNCTION_PROFILE_COMPLETE",
        "unique_frames": 45, "passes": 2, "live_started": False, "cuda_initialized": False,
        "input_sources_unchanged": True, "source_inventory_v1": sources,
        "script_sha256": sha(__file__), "settings": V1.settings(),
        "stage_cost_ms": {stage: {key: quantiles([r["timings"][stage][key] for r in rows])
            for key in ("wall_ms", "thread_cpu_ms")} for stage in STAGES},
        "through_ready_ms": quantiles([r["through_ready_ms"] for r in rows]),
        "through_archive_ms": quantiles([r["through_archive_ms"] for r in rows]),
        "limits": ["Offline timing is not live deadline performance.",
            "The full existing GT worker base-object builder, queue, network and edge are outside replay.",
            "cProfile alters timing: function shares are diagnostic, unprofiled timings are separate.",
            "AVO result and target/source payloads are exact; fresh READY/AVO diagnostic timestamps are not historical byte replays."]}
    write(args.output / "RESULT.json", result)
    write(args.output / "INPUT_SHA256.json", before)
    write(args.output / "OUTPUT_SHA256.json", R.tree(args.output))
    print(json.dumps(result["stage_cost_ms"], indent=2))


if __name__ == "__main__":
    main()
