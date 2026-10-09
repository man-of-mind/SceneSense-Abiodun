"""Bounded serial CPU timing of the numeric producer versus the current composition.

Run ONLY when the coordinator schedules it (no concurrent performance work):

    cd <abiodun>; CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      /usr/bin/python3 -m rl_agent.splitfusion_numeric_gt_producer_v1.benchmark \
      --output experiments/splitfusion_numeric_gt_producer_v1/<stamp>_serial_benchmark --frames 49 --passes 2

Baseline ``live_composition`` is the v1+v2 exact CPU optimizations with the v6
request-scoped depth reuse (the composition the v11 treatment ran), driven in
process without the worker IPC, broker or archive. Baseline ``pinned`` is the
unmodified original modules. Both baselines use the engine's in-memory AVO
evidence sink. Timings are offline CPU costs, never a live deadline claim;
``--frames 2`` is a harness smoke, explicitly not performance.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import tempfile
import time
from types import FunctionType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E, source_geometry as S
from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as GE
from rl_agent.splitfusion_run5_live_avo_v1 import gt_source_sync as SY, live_gt as LG

from . import artifacts as AR, avo as NA, cases as K, originals as O, payload as NP, replay_v11 as RP, source as NS, target as NT

KEYS = RP.KEYS
STAGES_BASELINE = ("avo_scoring_and_sidecar", "complete_source_build", "prepare_complete_target")
STAGES_NUMERIC = ("numeric_source_compute", "numeric_target", "numeric_avo", "payload")
STAGES_DEFERRED = ("deferred_legacy_target", "deferred_prepared_restore", "deferred_avo_documents", "deferred_binding")


def clock():
    return time.perf_counter_ns(), time.thread_time_ns()


def elapsed(start):
    wall, cpu = clock()
    return dict(wall_ms=(wall - start[0]) / 1e6, thread_cpu_ms=(cpu - start[1]) / 1e6)


def quantiles(values):
    values = sorted(values)
    def q(p):
        return float(np.quantile(values, p)) if values else None
    return dict(n=len(values), min=q(0), p50=q(.5), p95=q(.95), max=q(1))


def live_composition(args):
    """v1+v2 private functions with v6 depth reuse; no global installation."""
    from rl_agent.splitfusion_live_gt_optimization_v2 import candidate as V2C
    from rl_agent.splitfusion_live_gt_optimization_v6 import depth_cache, engine as V6E
    functions = V2C.private_functions(fast_labels=True, fast_metadata=True)
    depth = depth_cache.DepthReuse(LG.decode_depth, S._original()[3].decode_depth_bgra)
    producer_class = V6E.source_producer(functions["producer"], depth)
    prepare = V6E.prepare_function(functions["prepare"], depth)
    avo = FunctionType(LG.write_avo_bundle.__code__, dict(LG.write_avo_bundle.__globals__, decode_depth=depth.decode_image),
                       LG.write_avo_bundle.__name__, LG.write_avo_bundle.__defaults__, LG.write_avo_bundle.__closure__)
    avo.__kwdefaults__ = LG.write_avo_bundle.__kwdefaults__
    return dict(producer=producer_class(**args), prepare=prepare, avo=avo, depth=depth, label="v1_v2_v6_live_composition")


def pinned_composition(args):
    return dict(producer=S.LiveCompleteTargetProducer(**args), prepare=E.prepare_target, avo=LG.write_avo_bundle,
                depth=None, label="pinned_original_modules")


def run_baseline(base, case, kw, sink_dir, timings):
    sink = {}
    def add(path, payload):
        sink[str(path)] = payload
    scope = base["depth"].request_scope(identity=kw["identity"], depth_image=kw["depth_image"]) if base["depth"] else None
    if scope is not None:
        scope.__enter__()
    try:
        with patch.object(GE, "_create_identical_or_fail", add):
            start = clock()
            record = base["avo"](Path(sink_dir), identity=kw["identity"], rgb=kw["rgb"], semantic_image=kw["semantic_image"],
                depth_image=kw["depth_image"], frozen_world=kw["frozen_world"], camera_matrix=kw["camera_matrix"],
                camera_inverse=kw["camera_inverse"], policy_timestamp_s=case.kwargs["policy_timestamp_s"])
            timings["avo_scoring_and_sidecar"] = elapsed(start)
        start = clock()
        inputs = base["producer"].build(**kw)
        timings["complete_source_build"] = elapsed(start)
        start = clock()
        prepared = base["prepare"](inputs)
        timings["prepare_complete_target"] = elapsed(start)
    finally:
        if scope is not None:
            scope.__exit__(None, None, None)
    return inputs, prepared, json.loads(sink[record["path"]])


def run_numeric(numeric, case, kw, timings):
    start = clock()
    staged = numeric.compute(**kw)
    timings["numeric_source_compute"] = elapsed(start)
    start = clock()
    target = NT.numeric_target(frame=staged.frame, split="val", depth=staged.depth, semantic=staged.semantic,
        original_mask=staged.mask, boxes=staged.boxes, visibility=staged.visibility, static_rows=staged.static,
        quarantine=staged.conflict["quarantine"], vehicle_mask=staged.vehicle_mask)
    timings["numeric_target"] = elapsed(start)
    start = clock()
    avo = NA.numeric_avo(identity=kw["identity"], rgb=kw["rgb"], semantic_image=kw["semantic_image"],
        depth_image=kw["depth_image"], frozen_world=kw["frozen_world"], camera_matrix=staged.camera_matrix,
        camera_inverse=staged.camera_inverse, depth=staged.depth, sync_validated=True)
    timings["numeric_avo"] = elapsed(start)
    start = clock()
    payload = NP.build_payload(staged=staged, target=target, avo=avo)
    timings["payload"] = elapsed(start)
    clocks = {k: case.avo_document[k] for k in AR.AVO_CLOCKS if k != "sync_observed_raw_ns"}
    clocks["sync_observed_raw_ns"] = case.avo_prescore["source_synchronization"]["observed_raw_ns"]
    start = clock()
    legacy = AR.legacy_target(staged, target)
    timings["deferred_legacy_target"] = elapsed(start)
    start = clock()
    prepared = AR.prepared_target(staged, legacy)
    timings["deferred_prepared_restore"] = elapsed(start)
    start = clock()
    documents = AR.avo_documents(staged, avo, identity=kw["identity"], clocks=clocks,
        payload_filenames={k: v["filename"] for k, v in case.avo_document["payloads"].items()},
        prescore_filename=case.avo_document["prescore_input"]["filename"])
    timings["deferred_avo_documents"] = elapsed(start)
    start = clock()
    AR.binding(payload, legacy, prepared, avo_document_sha256=documents["document_sha256"])
    timings["deferred_binding"] = elapsed(start)
    numeric.commit(staged)
    return payload, prepared, documents


def run(output: Path, frames: int, passes: int, baseline: str) -> dict:
    environment = RP.verify_environment()
    RP.require(frames in (2, K.COHORT) and 1 <= passes <= 3, "two-frame smoke or 49 frames; 1..3 passes")
    output = Path(output).absolute()
    RP.require(output.resolve().is_relative_to(RP.EVIDENCE_ROOT) and not output.exists(), "create-only output below the evidence root")
    load_before = open("/proc/loadavg").read().split()[:3]
    input_hashes = K.verify_inputs()
    cases = K.load_cases()[:frames]
    binding = O.startup_binding()
    output.mkdir(parents=True, exist_ok=False)
    RP.write(output / "INPUT_SHA256.json", input_hashes)
    plan = dict(schema="splitfusion.numeric_gt_producer.serial_benchmark_plan.v1", frames=frames, passes=passes,
        baseline=baseline, frame_ids=[c.frame_id for c in cases], environment=environment, load_average_before=load_before,
        startup_binding_sha256=binding["binding_sha256"], order="per frame alternate baseline-first / numeric-first per pass",
        scope="in-process stage CPU/wall; excludes worker IPC, broker, archive, network and evaluator",
        performance_claim=frames == K.COHORT, smoke=frames == 2)
    RP.write(output / "PLAN.json", plan)
    rows = []
    try:
        with tempfile.TemporaryDirectory(prefix="numeric-gt-producer-bench-") as sink_dir:
            for pass_index in range(passes):
                args = K.producer_arguments(cases[0])
                base = live_composition(args) if baseline == "live_composition" else pinned_composition(args)
                numeric = NS.NumericSourceProducer(**args)
                for index, case in enumerate(cases):
                    kw = {key: case.kwargs[key] for key in KEYS}
                    observed = case.evidence["source_synchronization"]["observed_raw_ns"]
                    original_alignment = SY.source_alignment_record
                    def alignment(*a, _o=observed, **k):
                        value = original_alignment(*a, **k)
                        value["observed_raw_ns"] = _o
                        return value
                    population = case.evidence["dynamic_snapshot_evidence"]
                    for producer in (base["producer"], numeric):
                        if producer.population != frozenset(population["registered_population_ids"]):
                            producer.register_population(population["registered_population_ids"], population["population_sha256"],
                                                         before_frame_id=case.frame_id)
                    timings = {}
                    order = ("baseline", "numeric") if (index + pass_index) % 2 == 0 else ("numeric", "baseline")
                    with patch.object(SY, "source_alignment_record", alignment):
                        for arm in order:
                            if arm == "baseline":
                                inputs, prepared0, avo0 = run_baseline(base, case, kw, sink_dir, timings)
                            else:
                                payload, prepared1, documents = run_numeric(numeric, case, kw, timings)
                    exact = (prepared0.content_sha256 == prepared1.content_sha256 == case.ready["prepared_content_sha256"]
                             and inputs.provenance["input_content_sha256"] == payload.input_content_sha256
                             and documents["document_sha256"] == case.avo_sha256
                             and RP.canon(avo0["avo"]) == RP.canon(payload.avo))
                    rows.append(dict(pass_index=pass_index, frame_id=case.frame_id, order=order, exact=exact, timings=timings))
                    with (output / "PER_FRAME.jsonl").open("a") as stream:
                        stream.write(json.dumps(rows[-1], sort_keys=True) + "\n")
        RP.require(all(r["exact"] for r in rows), "benchmark arms disagree with retained digests")
        RP.require(K.verify_inputs() == input_hashes, "sealed inputs changed during benchmark")
        def stage_summary(stages, steady):
            return {stage: {metric: quantiles([r["timings"][stage][metric] for r in rows if not steady or r["frame_id"] != rows[0]["frame_id"]])
                            for metric in ("wall_ms", "thread_cpu_ms")} for stage in stages}
        totals = {}
        for label, stages in (("baseline_total", STAGES_BASELINE), ("numeric_readiness_total", STAGES_NUMERIC),
                              ("deferred_total", STAGES_DEFERRED)):
            totals[label] = {metric: quantiles([sum(r["timings"][s][metric] for s in stages) for r in rows])
                             for metric in ("wall_ms", "thread_cpu_ms")}
        paired = {metric: quantiles([sum(r["timings"][s][metric] for s in STAGES_BASELINE)
                                     - sum(r["timings"][s][metric] for s in STAGES_NUMERIC) for r in rows])
                  for metric in ("wall_ms", "thread_cpu_ms")}
        result = dict(schema="splitfusion.numeric_gt_producer.serial_benchmark_result.v1",
            status=("SERIAL_CPU_TIMING_NOT_LIVE_DEADLINE" if frames == K.COHORT else "SMOKE_HARNESS_CHECK_NOT_PERFORMANCE"),
            baseline=base["label"], frames=frames, passes=passes, replays=len(rows), all_exact=True,
            stages=stage_summary(STAGES_BASELINE + STAGES_NUMERIC + STAGES_DEFERRED, steady=False),
            stages_after_first_frame=stage_summary(STAGES_BASELINE + STAGES_NUMERIC + STAGES_DEFERRED, steady=True),
            totals=totals, paired_baseline_minus_numeric_readiness=paired,
            load_average_before=load_before, load_average_after=open("/proc/loadavg").read().split()[:3],
            caveats=["In-process stage timings; worker IPC, broker admission, archive, network and evaluator are excluded.",
                     "Offline CPU replay is not live deadline performance; the inclusive 170 ms clock is not addressed here.",
                     "Only valid if run serially under coordinator scheduling; concurrent runs are non-comparable.",
                     "The deferred total is post-readiness archive work, reported separately and not subtracted."])
        RP.write(output / "RESULT.json", result)
        print(json.dumps(dict(status=result["status"], baseline=result["baseline"], totals={k: v["wall_ms"]["p50"] for k, v in totals.items()},
                              paired_p50_wall=paired["wall_ms"]["p50"])), flush=True)
        return result
    except BaseException as error:
        RP.write(output / "FAILURE.json", dict(error=f"{type(error).__name__}: {error}", completed=len(rows)))
        raise
    finally:
        RP.write(output / "OUTPUT_SHA256.json", RP.tree(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, choices=(2, K.COHORT), default=K.COHORT)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--baseline", choices=("live_composition", "pinned"), default="live_composition")
    arguments = parser.parse_args()
    run(arguments.output, arguments.frames, arguments.passes, arguments.baseline)
