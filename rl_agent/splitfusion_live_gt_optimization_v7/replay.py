"""Create-only, counterordered offline source-build proof on sealed v8 frames.

Each arm retains the complete chronological tracker/conflict state. Timers
include source candidate work, not downstream verification or profiling.
Full input/source/target hashes, PNGs and original saved quality are required.
This is not a live worker, readiness benchmark, or deadline qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

from rl_agent.splitfusion_live_gt_optimization_v1 import replay_source as R
from rl_agent.splitfusion_live_gt_optimization_v2 import runtime as V2
from rl_agent.splitfusion_live_gt_optimization_v6.depth_cache import DepthReuse
from rl_agent.splitfusion_live_gt_optimization_v6.engine import source_producer as depth_producer
from rl_agent.splitfusion_live_gt_handoff_v1 import contract as H
from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as S, evaluation as E, gt_bridge as G
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as Wire
from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality as Q
from .source_geometry import source_producer

ROOT = Path(__file__).resolve().parents[2]
CELL = ROOT / "experiments/splitfusion_live_parity_runtime_v8/20261004_handoff_live_v8/qualification_100/capture/cells/a71__favorable_stable"
OUT = ROOT / "experiments/splitfusion_live_gt_optimization_v7/20261004_v7"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def read(path):
    return json.loads(Path(path).read_bytes())


def quantiles(values):
    return {name: float(np.quantile(values, p)) for name, p in
            (("min", 0), ("p50", .5), ("p95", .95), ("max", 1))}


def saved_predictions():
    directory = CELL / "profiler/edge/prediction_archive"
    result = {}
    for path in sorted(directory.glob("*.identity.json")):
        identity = read(path)
        stem = path.name.removesuffix(".identity.json")
        result[identity["gt_identity"]["frame_id"]] = dict(identity=identity,
            labels=np.load(directory / (stem+".labels.npy"), allow_pickle=False),
            records=read(directory / (stem+".records.json")),
            evidence=read(directory / (stem+".evaluation.json")))
    feedback = read(CELL / "run4_phase6/PHASE6_UE_EVIDENCE.json")["feedback_rows"]
    for row in feedback:
        result[row["frame_id"]]["q_perc"] = row["q_perc"]
    require(len(result) == len(feedback) == 48, "saved prediction/feedback cohort differs")
    return result


def run(output, frames):
    require(sys.executable == "/usr/bin/python3" and np.__version__ == "1.26.4", "registered interpreter required")
    require(all(os.environ.get(key) == "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "CPU caps differ")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized()
            and torch.get_num_threads() == 1, "CPU-only environment required")
    require(frames in (2, 48), "only two-frame proof or complete cohort permitted")
    require(not output.exists() and output.resolve().is_relative_to(OUT), "new output root required")
    seal_root = CELL.parents[3]
    seal = read(seal_root / "OUTPUT_SHA256.json")
    # Bind the already-sealed input inventory; full outer verification is also
    # run before/after this experiment by the root verification workflow.
    source_hashes = H.sources()
    candidate = {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("source_geometry.py"))}
    records = read(CELL / "run5_live_avo_gt_summary.json")["records"]
    require(len(records) == 48, "native full48 cohort required")
    records = records[:frames]
    reference = R.load_reference(); reference.CELL = CELL
    predictions = saved_predictions()
    spec = Wire.load_run4_quality_spec(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    design = dict(frames=frames, passes=2, builds=frames*4, arms=["control", "candidate"],
        ordering="CT/TC alternating by frame and reversed on pass2", source_cell=str(CELL),
        candidate_sources=candidate, sealed_handoff_sources=source_hashes,
        input_seal_sha256=sha(seal_root/"OUTPUT_SHA256.json"), saved_seal_files=len(seal.get("files", seal)),
        executable=sys.executable, numpy=np.__version__, torch_threads=torch.get_num_threads(),
        intervention="immutable request-local radar homogeneous reuse and static semantic membership reuse",
        unchanged="all numerical geometry, validation, eligibility, history, target bytes and quality",
        timer="source builder only; depth cache prefilled as in preceding AVO; excludes verification",
        clock_override="recorded source observation stamp only, offline parity", live_qualified=False)
    write(output/"DESIGN.json", design)
    write(output/"DESIGN_SEAL.json", dict(sha256=sha(output/"DESIGN.json"), before_any_build=True))
    rows = []
    try:
        for repeat in range(2):
            with V2.install_cpu_gt_optimizations(expected_sources=V2.source_inventory()):
                depths = {name: DepthReuse(LG.decode_depth, S._original()[3].decode_depth_bgra)
                          for name in ("control", "candidate")}
                types = {name: depth_producer(S.LiveCompleteTargetProducer, depth) for name, depth in depths.items()}
                types["candidate"] = source_producer(types["candidate"])
                producers = {}
                for index, record in enumerate(records):
                    bundle, evidence, kwargs = reference.owned(record)
                    # Match the worker's immutable bytes-backed native C array.
                    # Both arms see the very same object/layout. No floats change.
                    radar = kwargs["radar_world_xyz"]
                    kwargs["radar_world_xyz"] = np.frombuffer(radar.tobytes(), dtype=radar.dtype).reshape(radar.shape)
                    population = evidence["dynamic_snapshot_evidence"]
                    order = ("control", "candidate") if (index+repeat)%2 == 0 else ("candidate", "control")
                    timings = {}; reuse = None
                    for name in order:
                        if name not in producers:
                            producers[name] = types[name](static_catalogue_dir=CELL/"parity_static_catalogue",
                                experiment_id=bundle.inputs.frame["experiment_id"], ego_id=population["population"]["ego_id"],
                                dynamic_population_ids=population["registered_population_ids"],
                                population_evidence_sha256=evidence["population_evidence_sha256"], max_frames=100)
                        producer, depth = producers[name], depths[name]
                        if producer.population != frozenset(population["registered_population_ids"]):
                            producer.register_population(population["registered_population_ids"], population["population_sha256"],
                                before_frame_id=record["frame_id"])
                        with depth.request_scope(identity=kwargs["identity"], depth_image=kwargs["depth_image"]):
                            depth.decode_image(kwargs["depth_image"])
                            with patch("rl_agent.splitfusion_run5_live_avo_v1.gt_source_sync.time.clock_gettime_ns",
                                       return_value=evidence["source_synchronization"]["observed_raw_ns"]):
                                start, cpu = time.perf_counter_ns(), time.thread_time_ns()
                                inputs = producer.build(**kwargs)
                                timings[name] = dict(wall_ms=(time.perf_counter_ns()-start)/1e6,
                                    thread_cpu_ms=(time.thread_time_ns()-cpu)/1e6)
                        require(not depth.active and not depth.retained, "depth request retained")
                        require(inputs.provenance == bundle.inputs.provenance, name+" native input hash differs")
                        require(G.canonical(producer.last_evidence) == G.canonical(evidence), name+" source evidence differs")
                        prepared = E.prepare_target(inputs)
                        require(prepared.content_sha256 == bundle.prepared.content_sha256, name+" target content differs")
                        require(all(prepared.target[k] == bundle.prepared.target[k] for k in ("segmentation_png", "object_ignore_png")), name+" PNG bytes differ")
                        prediction = predictions[record["frame_id"]]
                        avo = read(CELL/"run5_live_avo_inputs"/Path(record["path"]).name)["avo"]
                        evaluated = E.evaluate_prepared(identity=prediction["identity"]["action_identity"], prepared=prepared,
                            predicted_labels=prediction["labels"], predictions=prediction["records"], avo=avo)
                        require(G.canonical(evaluated.evidence) == G.canonical(prediction["evidence"]), name+" evaluation differs")
                        quality = Q.evaluate_exact_quality(spec, evaluated.measurement)
                        require((None if quality is None else float(quality.q_perc)) == prediction["q_perc"], name+" actual Q_perc differs")
                        if name == "candidate":
                            reuse = dict(producer.last_array_reuse)
                            require(reuse["completed"] and not reuse["retained_after_request"], "candidate retained request")
                            require(reuse["radar_builds"] == 1 and reuse["radar_reuses"] > 0
                                    and reuse["radar_fallbacks"] == 0, "radar reuse was not exercised")
                    rows.append(dict(repeat=repeat, frame_id=record["frame_id"], order=order, timing=timings, reuse=reuse,
                        input_sha256=inputs.provenance["input_content_sha256"], target_sha256=prepared.content_sha256,
                        quality=prediction["q_perc"], all_exact=True))
                    print(json.dumps(dict(repeat=repeat, frame_id=record["frame_id"], exact=True, timing=timings)), flush=True)
        require(source_hashes == H.sources(), "sealed source changed")
        require(candidate == {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("source_geometry.py"))}, "candidate source changed")
        require(not torch.cuda.is_initialized(), "CUDA initialized")
        result = dict(status="OFFLINE_EXACT_SOURCE_REUSE_PASS", frames=frames, builds=frames*4, rows=rows,
            timing={name: {metric: quantiles([row["timing"][name][metric] for row in rows])
                           for metric in ("wall_ms", "thread_cpu_ms")} for name in ("control", "candidate")},
            paired_savings_ms={metric: quantiles([row["timing"]["control"][metric]-row["timing"]["candidate"][metric] for row in rows])
                               for metric in ("wall_ms", "thread_cpu_ms")},
            all_source_input_target_png_quality_exact=True, sealed_sources_unchanged=True,
            cuda_initialized=False, live_qualified=False, deployed=False,
            caveats=["Source stage only; not full child service, handoff or action-to-feedback timing.",
                     "Saved current48frames; timings offline and not an inclusive170ms pass.",
                     "Unchanged actor and saved predictions; no retraining or optimization of quality values."])
        write(output/"RESULT.json", result)
        print(json.dumps({k: result[k] for k in ("status", "frames", "builds", "timing", "paired_savings_ms")}), flush=True)
    except BaseException as exc:
        write(output/"FAILURE.json", dict(error=f"{type(exc).__name__}: {exc}", rows=rows))
        raise
    finally:
        write(output/"OUTPUT_SHA256.json", {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, choices=(2, 48), default=48)
    args = parser.parse_args(); run(args.output, args.frames)
