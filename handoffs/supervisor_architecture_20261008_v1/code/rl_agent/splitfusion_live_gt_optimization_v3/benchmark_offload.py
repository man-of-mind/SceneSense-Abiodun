"""Bounded, CPU-only saved-fixture comparison of exact V2 GT ownership.

Two counterordered passes compare Engine.process to a synchronous Client.run.
Both call the unchanged write body and genuine LiveSyncEvidence. This is not
a live collector benchmark: historical archives lack some unused native pose
components, explicitly described in FIXTURE_LIMITS below. All consumed target,
source and AVO values must nevertheless match the sealed capture exactly.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import resource
import tempfile
import time

import numpy as np
import torch

from . import profile_saved as P
from . import worker as W
from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T

ORDERS = (("inprocess", "process"), ("process", "inprocess"))
STAGES = ("avo_scoring_and_sidecar", "complete_source_build", "prepare_complete_target",
          "publish_complete_target", "archive_auxiliary_sources")
AVO_CLOCKS = ("start_raw_ns", "write_raw_ns", "prescore_input_ready_raw_ns",
              "scoring_start_raw_ns", "end_raw_ns", "ready_wall_ns")
FIXTURE_LIMITS = [
    "Both arms use the same sealed saved-fixture reconstruction, not a claim of native live snapshot capture.",
    "The sealed helper reconstructs actor inverse matrices with numpy.linalg.inv from archived original matrices.",
    "Vehicle pitch/roll were not archived and are zero placeholders; off-view yaw retains the sealed helper's zero placeholder. These fields do not affect the verified consumed outputs.",
    "Pedestrian yaw/pitch/roll are restored exactly from archived AVO rows; bounding-box rotations come from registered-population evidence.",
    "Image inverses use the archived camera inverse only after exact matrix equality with that camera is established.",
    "Semantic BGRA's unused channels are reconstructed as zero by the sealed helper; raw semantic IDs remain exact.",
]


def require(value, message):
    P.require(value, message)


def registered_bbox_rotation(values):
    """gt_bridge registration stores roll,pitch,yaw, not constructor order."""
    import carla
    require(len(values) == 3, "registered bbox rotation length differs")
    roll, pitch, yaw = values
    return carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)


def rotation_mapping_proof():
    # Distinct, nonzero, exactly representable values detect axis permutations.
    value = registered_bbox_rotation([11.25, -23.5, 71.75])
    actual = [value.roll, value.pitch, value.yaw]
    require(actual == [11.25, -23.5, 71.75], "native registered bbox rotation mapping differs")
    return dict(registered_axis_order=["roll", "pitch", "yaw"], native_keyword_mapping_exact=True,
                synthetic_nonzero_roundtrip=actual)


def fixture(case):
    """Complete the private replay proxy; never mutate a sealed module/file."""
    _bundle, evidence, original, _record, _avo = case
    kwargs = {key: original[key] for key in ("identity", "rgb", "semantic_image", "depth_image",
        "frozen_world", "camera_matrix", "camera_inverse")}
    kwargs["policy_timestamp_s"] = original["rgb"]["timestamp"]
    world = kwargs["frozen_world"]
    world.population_evidence = evidence["dynamic_snapshot_evidence"]
    registered = {row["actor_id"]: row for row in world.population_evidence["population"]["registered_actors"]}
    for actor in world.get_actors():
        rotation = actor.get_transform().rotation
        if not actor.type_id.startswith("walker.pedestrian."):
            require(not hasattr(rotation, "pitch") or rotation.pitch == 0., "unexpected fixture pitch")
            require(not hasattr(rotation, "roll") or rotation.roll == 0., "unexpected fixture roll")
            rotation.pitch = rotation.roll = 0.
        require(all(hasattr(rotation, key) for key in ("pitch", "yaw", "roll")), "missing retained person pose")
        native = registered[actor.id]
        require(native["type_id"] == actor.type_id, "registered actor type differs")
        require(native["bbox_extent"] == [float(getattr(actor.bounding_box.extent, k)) for k in ("x", "y", "z")],
                "registered bbox extent differs")
        require(native["bbox_location"] == [float(getattr(actor.bounding_box.location, k)) for k in ("x", "y", "z")],
                "registered bbox location differs")
        actor.bounding_box.rotation = registered_bbox_rotation(native["bbox_rotation"])
    for image in (kwargs["rgb"]["_rgb_image"], kwargs["semantic_image"], kwargs["depth_image"]):
        require(np.array_equal(np.asarray(image.transform.get_matrix()), kwargs["camera_matrix"]),
                "image camera differs; archived inverse cannot be assigned")
        image.transform.get_inverse_matrix = lambda inverse=kwargs["camera_inverse"]: inverse.tolist()
    return kwargs, original["radar_world_xyz"]


def without_avo_clocks(document):
    value = copy.deepcopy(document)
    for key in AVO_CLOCKS:
        value.pop(key, None)
    value["source_synchronization"].pop("observed_raw_ns")
    if "prescore_input" in value:
        # This derived digest is separately checked against the actual prescore
        # bytes; those bytes contain the explicitly excluded genuine clocks.
        value["prescore_input"].pop("sha256")
    return value


def sync_binding(cell, qualification, outer_sha, cases):
    outer = P.read(qualification / "RUN5_POSTFLIGHT_INPUTS.json", outer_sha)["files"]
    result = {}
    for case in cases:
        frame = case[2]["identity"]["frame_id"]
        for suffix in ("observed", "accepted"):
            path = cell / "run5_live_sync_evidence" / f"frame_{frame:010d}.{suffix}.json"
            name = str(path.relative_to(qualification))
            require(name in outer, "missing postflight-bound synchronization evidence")
            require(P.file_sha(path) == outer[name], "retained synchronization hash differs")
            result[name] = outer[name]
    return result


def verify_frame(*, case, result, target_root, sync_root, cell, tree, diagnostic_dir):
    bundle, _evidence, kwargs, saved_record, saved_avo = case
    frame = kwargs["identity"]["frame_id"]
    require(result["status"] == "OK" and result["identity"] == kwargs["identity"], "GT request failed/foreign")
    record = result["record"]
    require(record["complete_identity"] == saved_record["complete_identity"], "complete identity differs")
    replay = T.verify_full_inputs(target_root, expected_identity=kwargs["identity"],
        expected_manifest_sha256=record["complete_target"]["manifest_sha256"])
    if replay.inputs.provenance != bundle.inputs.provenance:
        actual, expected = T._input_metadata(replay.inputs), T._input_metadata(bundle.inputs)
        differing = [name for name in expected if actual[name] != expected[name]]
        P.write(diagnostic_dir / f"MISMATCH_frame_{frame}.json", dict(frame_id=frame,
            check="complete native inputs/provenance", differing_fields=differing,
            actual={key: actual[key] for key in differing}, expected={key: expected[key] for key in differing},
            actual_manifest=replay.manifest, expected_manifest=bundle.manifest,
            actual_source_evidence=P.read(Path(record["complete_sources"]["directory"]) / "SOURCE_EVIDENCE.json"),
            expected_source_evidence=case[1], worker_result=result))
    require(replay.native_inputs_verified and replay.inputs.provenance == bundle.inputs.provenance,
            "complete native inputs/provenance differ")
    require(replay.prepared.content_sha256 == bundle.prepared.content_sha256, "prepared target content differs")
    strip_ready = lambda m: {key: value for key, value in m.items() if key not in ("ready_raw_ns", "ready_wall_ns")}
    require(strip_ready(replay.manifest) == strip_ready(bundle.manifest), "target manifest/payload bytes differ")
    new_source = Path(record["complete_sources"]["directory"])
    saved_source = cell / "parity_complete_targets" / Path(saved_record["complete_sources"]["directory"]).name
    require(tree(new_source) == tree(saved_source), "complete source archive byte tree differs")
    require(record["complete_sources"]["manifest_sha256"] == saved_record["complete_sources"]["manifest_sha256"],
            "source manifest digest differs")
    new_path = Path(record["path"])
    replay_avo = P.read(new_path, record["sha256"])
    require(without_avo_clocks(replay_avo) == without_avo_clocks(saved_avo), "AVO values or non-clock fields differ")
    for role, payload in replay_avo["payloads"].items():
        require(P.file_sha(new_path.parent / payload["filename"]) == payload["sha256"], "AVO native payload hash differs: " + role)
    replay_pre = P.read(new_path.parent / replay_avo["prescore_input"]["filename"], replay_avo["prescore_input"]["sha256"])
    saved_pre = P.read(cell / "run5_live_avo_inputs" / saved_avo["prescore_input"]["filename"], saved_avo["prescore_input"]["sha256"])
    require(without_avo_clocks(replay_pre) == without_avo_clocks(saved_pre), "AVO prescore inputs differ")
    observed_name = f"frame_{frame:010d}.observed.json"
    accepted_name = f"frame_{frame:010d}.accepted.json"
    observed = P.read(sync_root / observed_name)
    accepted = P.read(sync_root / accepted_name)
    require(accepted["observed_sha256"] == P.file_sha(sync_root / observed_name), "sync acceptance unbound")
    require(observed["observed_raw_ns"] == replay_avo["source_synchronization"]["observed_raw_ns"], "sync/AVO clocks differ")
    saved_observed = P.read(cell / "run5_live_sync_evidence" / observed_name)
    saved_accepted = P.read(cell / "run5_live_sync_evidence" / accepted_name)
    require(saved_accepted["observed_sha256"] == P.file_sha(cell / "run5_live_sync_evidence" / observed_name), "saved sync acceptance unbound")
    require({k: v for k, v in observed.items() if k != "observed_raw_ns"} ==
            {k: v for k, v in saved_observed.items() if k != "observed_raw_ns"}, "genuine synchronization content differs")
    require({k: v for k, v in accepted.items() if k != "observed_sha256"} ==
            {k: v for k, v in saved_accepted.items() if k != "observed_sha256"}, "genuine synchronization acceptance differs")
    require(result["sync"]["rejected"] == 0 and result["sync"]["raw_bundles"] == 0, "unexpected sync rejection")
    require([row["stage"] for row in result["rows"]] == list(STAGES), "stage population/order differs")
    require(all(row["completed"] and row["frame_id"] == frame for row in result["rows"]), "incomplete stage")
    return dict(all_exact=True, input_content_sha256=replay.manifest["input_content_sha256"],
        prepared_content_sha256=replay.prepared.content_sha256, target_payloads=replay.manifest["payloads"],
        source_manifest_sha256=record["complete_sources"]["manifest_sha256"],
        target_ready_manifest_sha256=record["complete_target"]["manifest_sha256"],
        avo_sha256=record["sha256"], avo_prescore_sha256=replay_avo["prescore_input"]["sha256"],
        sync_observed_sha256=accepted["observed_sha256"],
        historical_clock_override="SOURCE_EVIDENCE.observed_raw_ns ONLY")


def quantiles(values):
    return {name: float(np.quantile(values, q)) for name, q in (("min", 0), ("p50", .5), ("p95", .95), ("max", 1))}


def child_cpu_ns():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return int((usage.ru_utime + usage.ru_stime) * 1e9)


def run_arm(*, mode, repeat, cases, fixtures, cell, output, sources, tree):
    arm_dir = output / f"order_{repeat}_{mode}"
    arm_dir.mkdir(exist_ok=False)
    sync_root = arm_dir / "genuine_source_sync"
    sync = dict(directory=str(sync_root), raw_bundle_limit=8, raw_byte_limit=134217728)
    factory = W.Engine if mode == "inprocess" else W.Client
    before_child_cpu = child_cpu_ns()
    started, cpu = W.raw(), time.thread_time_ns()
    engine = factory(attempt_dir=arm_dir / "owner", catalogue_dir=cell / "parity_static_catalogue",
        max_frames=100, expected_sources=sources, allow_replay_clock=True, source_attempt_name=cell.name)
    startup = dict(wall_ms=(W.raw()-started)/1e6, parent_thread_cpu_ms=(time.thread_time_ns()-cpu)/1e6)
    rows = []
    error = None
    close_result = None
    try:
        with tempfile.TemporaryDirectory(prefix="exact-gt-v3-offload-") as temporary:
            target_root = Path(temporary) / "targets"
            for index, (case, (kwargs, radar)) in enumerate(zip(cases, fixtures)):
                request = dict(directory=target_root, kwargs=kwargs, radar=radar, sent_raw_ns=W.raw()-1,
                    sync_config=sync, replay_observed_raw_ns=case[1]["source_synchronization"]["observed_raw_ns"])
                start, start_cpu = W.raw(), time.thread_time_ns()
                result = (engine.process if mode == "inprocess" else engine.run)(**request)
                end, end_cpu = W.raw(), time.thread_time_ns()
                # Measurement ends before the deliberately expensive verifier.
                timing = dict(request_wall_ms=(end-start)/1e6,
                    caller_thread_cpu_ms=(end_cpu-start_cpu)/1e6,
                    gt_body_wall_ms=(result["worker"]["end_raw_ns"]-result["worker"]["start_raw_ns"])/1e6,
                    gt_body_thread_cpu_ms=result["worker"]["thread_cpu_ns"]/1e6)
                publish = next((r for r in result["rows"] if r["stage"] == "publish_complete_target"), None)
                if publish is not None:
                    timing["request_to_ready_ms"] = (publish["end_raw_ns"]-start)/1e6
                if "ipc" in result:
                    ipc = result["ipc"]
                    timing.update(pack_wall_ms=(ipc["packed_raw_ns"]-ipc["start_raw_ns"])/1e6,
                        request_minus_gt_body_wall_ms=timing["request_wall_ms"]-timing["gt_body_wall_ms"],
                        combined_caller_and_gt_body_cpu_ms=timing["caller_thread_cpu_ms"]+timing["gt_body_thread_cpu_ms"])
                else:
                    timing["combined_caller_and_gt_body_cpu_ms"] = timing["caller_thread_cpu_ms"]
                evidence = verify_frame(case=case, result=result, target_root=target_root, sync_root=sync_root,
                    cell=cell, tree=tree, diagnostic_dir=arm_dir)
                require(result["processed"] == index+1 and result["sync"]["accepted"] == index+1, "sequential count differs")
                rows.append(dict(frame_id=kwargs["identity"]["frame_id"], timing=timing, proof=evidence,
                    worker=result["worker"], ipc=result.get("ipc"), stages=result["rows"]))
                P.write(arm_dir / f"frame_{index:03d}.json", rows[-1])
            final_sync = result["sync"]
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        close_start, close_cpu = W.raw(), time.thread_time_ns()
        try:
            close_result = engine.close()
        finally:
            P.write(arm_dir / "OWNER_CLOSE.json", dict(result=close_result, error=error,
                completed_verified_frames=len(rows), close_wall_ms=(W.raw()-close_start)/1e6,
                close_parent_thread_cpu_ms=(time.thread_time_ns()-close_cpu)/1e6,
                full_child_lifetime_process_cpu_ms=(child_cpu_ns()-before_child_cpu)/1e6))
    summary = dict(mode=mode, order_index=repeat, frames=len(rows), all_exact=True,
        startup=startup, final_sync=final_sync,
        timing_ms={key: quantiles([r["timing"][key] for r in rows]) for key in rows[0]["timing"]},
        steady_after_first_timing_ms={key: quantiles([r["timing"][key] for r in rows[1:]]) for key in rows[0]["timing"]},
        stages_ms={stage: {metric: quantiles([next(r for r in row["stages"] if r["stage"] == stage)[metric]/1e6
            for row in rows]) for metric in ("wall_ns", "thread_cpu_ns")} for stage in STAGES})
    P.write(arm_dir / "SUMMARY.json", summary)
    print(json.dumps(dict(order=repeat, mode=mode, frames=len(rows), all_exact=True,
        request_wall_ms=summary["timing_ms"]["request_wall_ms"])), flush=True)
    return summary, rows


def run(*, cell, output, frames):
    cell, output = P.safe(cell), P.safe(output)
    require(frames in (2, 48), "only a two-frame smoke or the entire 48-frame cohort is allowed")
    require(not output.exists(), "benchmark output is create-only")
    settings = P.verify_environment()
    rotation_proof = rotation_mapping_proof()
    qualification, v2_sources, hashes, outer_sha = P.input_binding(cell)
    profile = P.private_profile(cell, v2_sources)
    _reference, all_cases = P.load_cases(profile, cell)
    cases = all_cases[:frames]
    hashes.update(sync_binding(cell, qualification, outer_sha, cases))
    fixtures = [fixture(case) for case in cases]
    sources = W.sources()
    output.mkdir(parents=True, exist_ok=False)
    design = dict(schema="splitfusion.gt_v3.offline_offload_design.v1", cell=str(cell), frames_per_arm=frames,
        orders=ORDERS, arms=4, total_expected_replays=4*frames, mode="SMOKE_NOT_PERFORMANCE" if frames == 2 else "ALL48_COUNTERORDERED",
        settings=settings, fixture_rotation_mapping_unit_proof=rotation_proof,
        worker_source_hashes=sources, retained_v2_source_hashes=v2_sources,
        parent_postflight_input_manifest_sha256=outer_sha, fixture_limits=FIXTURE_LIMITS,
        genuine_sync_in_both_arms=True, clock_override="ONLY original source evidence observation timestamp; no live/global clock override",
        startup_scope="Engine/Client constructor separate; first request retains producer initialization",
        timing_scope="Entire process() or run() call before verification; Client wall includes packing, IPC, child decode, GT and reply",
        cpu_scope="Caller thread plus GT body thread; combined excludes child decode/IPC thread CPU, reported separately as full child lifetime process CPU",
        exclusions=["No live deadline, GIL-contention, UE inference, base-object collector or callback cost claim.",
            "No concurrency/overlap or queue throughput test; one request in flight.",
            "Startup is cold subprocess versus warm parent imports, not a steady-state per-frame comparison.",
            "Scratch target/source/native AVO payloads are exact-verified then removed; proof digests and genuine synchronization files are retained."])
    P.write(output / "DESIGN.json", design)
    P.write(output / "INPUT_SHA256.json", hashes)
    arms, rows_by_arm = [], {}
    try:
        for repeat, order in enumerate(ORDERS):
            for mode in order:
                summary, rows = run_arm(mode=mode, repeat=repeat, cases=cases, fixtures=fixtures, cell=cell,
                    output=output, sources=sources, tree=profile.R.tree)
                arms.append(summary)
                rows_by_arm[(repeat, mode)] = rows
        comparisons = []
        for repeat in range(2):
            baseline, candidate = rows_by_arm[(repeat, "inprocess")], rows_by_arm[(repeat, "process")]
            require([r["frame_id"] for r in baseline] == [r["frame_id"] for r in candidate], "paired frame denominator differs")
            delta_keys = ("request_wall_ms", "request_to_ready_ms", "caller_thread_cpu_ms", "gt_body_thread_cpu_ms",
                          "combined_caller_and_gt_body_cpu_ms")
            comparisons.append(dict(order_index=repeat, frames=frames,
                paired_process_minus_inprocess_ms={key: quantiles([b["timing"][key]-a["timing"][key]
                    for a, b in zip(baseline, candidate)]) for key in delta_keys}))
        require(W.sources() == sources, "candidate worker sources changed during proof")
        for name, digest in hashes.items():
            require(P.file_sha(qualification / name) == digest, "retained input changed during proof")
        require(P.verify_environment() == settings and not torch.cuda.is_initialized(), "CPU numerical context changed")
        result = dict(schema="splitfusion.gt_v3.offline_offload_result.v1",
            status="SMOKE_2_EXACT_NOT_PERFORMANCE" if frames == 2 else "ALL48_FOUR_ARMS_EXACT_CPU_OFFLOAD_PROOF",
            unique_frames=frames, total_replays=4*frames, all_exact=True, live_started=False,
            cuda_initialized=False, inputs_and_sources_unchanged=True, arms=arms,
            paired_differences=comparisons, fixture_limits=FIXTURE_LIMITS,
            cpu_scope=design["cpu_scope"], exclusions=design["exclusions"])
        P.write(output / "RESULT.json", result)
        P.write(output / "OUTPUT_SHA256.json", profile.R.tree(output))
        return result
    except BaseException as exc:
        P.write(output / "FAILURE.json", dict(error=f"{type(exc).__name__}: {exc}", completed_arms=len(arms),
            planned_unique_frames=frames, cuda_initialized=torch.cuda.is_initialized()))
        P.write(output / "OUTPUT_SHA256.json", profile.R.tree(output))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, default=P.DEFAULT_CELL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, choices=(2, 48), default=48)
    args = parser.parse_args(argv)
    run(cell=args.cell, output=args.output, frames=args.frames)


if __name__ == "__main__":
    main()
