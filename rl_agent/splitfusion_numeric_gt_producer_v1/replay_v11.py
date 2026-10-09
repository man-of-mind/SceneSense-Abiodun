"""Create-only exact parity of the numeric producer on the sealed 49 v11 frames.

For every frame: the pinned original path (producer.build -> prepare_target ->
write_avo_bundle with an in-memory sink) and the numeric path (compute ->
numeric_target -> numeric_avo -> payload -> deferred artifacts) are both run and
compared bit-exactly with the retained files: native inputs/provenance, source
evidence, legacy target.json, PNG bytes, READY digests, AVO documents (with the
recorded clocks), and the saved native prediction evaluation plus received
Q_perc (including the eight undefined-quality frames). A shadow producer
computes, cancels, recomputes and commits each frame to prove that cancellation
restores history exactly. Deterministic work counters are recorded per frame.
No live service, GPU, retained-evidence write or performance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np

from rl_agent.splitfusion_live_gt_optimization_v1.source_geometry import json_value
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E, gt_bridge as G, source_geometry as S
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as GE
from rl_agent.splitfusion_run5_live_avo_v1 import gt_source_sync as SY, live_gt as LG

from . import accounting as AC, artifacts as AR, avo as NA, cases as K, originals as O, payload as NP, source as NS, target as NT

HERE = Path(__file__).resolve().parent
EVIDENCE_ROOT = K.ROOT / "experiments/splitfusion_numeric_gt_producer_v1"
KEYS = ("identity", "rgb", "semantic_image", "depth_image", "frozen_world", "camera_matrix", "camera_inverse",
        "radar_world_xyz", "dynamic_snapshot_evidence")
RESULT_SCHEMA = "splitfusion.numeric_gt_producer.v11_parity_result.v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def canon(value) -> bytes:
    return A.canonical(json_value(value))


def write(path: Path, value) -> None:
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def tree(path: Path) -> dict:
    return {str(p.relative_to(path)): K.sha(p) for p in sorted(Path(path).rglob("*")) if p.is_file()}


def verify_environment() -> dict:
    require(sys.executable == "/usr/bin/python3" and not sys.flags.optimize, "registered interpreter with assertions required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(k) == "1" for k in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "CPU-only pre-import environment required")
    require(np.__version__ == "1.26.4", "NumPy 1.26.4 required for bit-exact replay")
    import torch
    require(not torch.cuda.is_initialized(), "CUDA initialized")
    return dict(python=sys.executable, numpy=np.__version__, torch=torch.__version__,
                threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})


def quality_bridge():
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as QualityWire
    from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality as Q
    return QualityWire.load_run4_quality_spec(K.ROOT), Q


def original_avo(case, kw, sink_dir):
    """Original AVO writer with the engine's in-memory evidence sink (no fsync)."""
    sink = {}
    def add(path, payload):
        sink[str(path)] = payload
    with patch.object(GE, "_create_identical_or_fail", add):
        record = LG.write_avo_bundle(Path(sink_dir), identity=kw["identity"], rgb=kw["rgb"],
            semantic_image=kw["semantic_image"], depth_image=kw["depth_image"], frozen_world=kw["frozen_world"],
            camera_matrix=kw["camera_matrix"], camera_inverse=kw["camera_inverse"],
            policy_timestamp_s=case.kwargs["policy_timestamp_s"])
    return json.loads(sink[record["path"]])


def quality_proof(prepared, avo, prediction, spec, Q):
    result = E.evaluate_prepared(identity=prediction["identity"]["action_identity"], prepared=prepared,
        predicted_labels=prediction["labels"], predictions=prediction["records"], avo=avo)
    exact_evidence = G.canonical(result.evidence) == G.canonical(prediction["evidence"])
    value = Q.evaluate_exact_quality(spec, result.measurement)
    q_perc = None if value is None else float(value.q_perc)
    return dict(evidence_exact=exact_evidence, q_perc=q_perc, q_perc_exact=(q_perc == prediction["q_perc"]),
                measurement_sha256=hashlib.sha256(G.canonical(result.measurement)).hexdigest())


def run_frame(case, *, prediction, spec, Q, original_producer, numeric, shadow, sink_dir):
    kw = {key: case.kwargs[key] for key in KEYS}
    observed = case.evidence["source_synchronization"]["observed_raw_ns"]
    original_alignment = SY.source_alignment_record
    def alignment(*args, **kwargs):
        value = original_alignment(*args, **kwargs)
        value["observed_raw_ns"] = observed
        return value
    row = dict(frame_id=case.frame_id, checks={}, counters={})
    population = case.evidence["dynamic_snapshot_evidence"]
    for producer in (original_producer, numeric, shadow):
        if producer.population != frozenset(population["registered_population_ids"]):
            producer.register_population(population["registered_population_ids"], population["population_sha256"],
                                         before_frame_id=case.frame_id)
    # ---- original pinned path (cross-check + accounting) ----
    with patch.object(SY, "source_alignment_record", alignment), AC.measure() as counters:
        inputs0 = original_producer.build(**kw)
        prepared0 = E.prepare_target(inputs0)
        avo0 = original_avo(case, kw, sink_dir)
    row["counters"]["original_path"] = counters.table()
    row["checks"]["original_inputs_provenance_exact"] = canon(inputs0.provenance) == canon(case.bundle.inputs.provenance)
    row["checks"]["original_evidence_exact"] = canon(original_producer.last_evidence) == canon(case.evidence)
    row["checks"]["original_prepared_sha_exact"] = prepared0.content_sha256 == case.ready["prepared_content_sha256"]
    row["checks"]["original_avo_exact"] = canon(avo0["avo"]) == canon(case.avo_document["avo"])
    # ---- numeric readiness path ----
    with patch.object(SY, "source_alignment_record", alignment), AC.measure() as counters:
        staged = numeric.compute(**kw)
        target = NT.numeric_target(frame=staged.frame, split="val", depth=staged.depth, semantic=staged.semantic,
            original_mask=staged.mask, boxes=staged.boxes, visibility=staged.visibility, static_rows=staged.static,
            quarantine=staged.conflict["quarantine"], vehicle_mask=staged.vehicle_mask)
        avo = NA.numeric_avo(identity=kw["identity"], rgb=kw["rgb"], semantic_image=kw["semantic_image"],
            depth_image=kw["depth_image"], frozen_world=kw["frozen_world"], camera_matrix=staged.camera_matrix,
            camera_inverse=staged.camera_inverse, depth=staged.depth, sync_validated=True)
        payload = NP.build_payload(staged=staged, target=target, avo=avo)
    row["counters"]["numeric_readiness_path"] = counters.table()
    # ---- deferred artifacts ----
    clocks = {k: case.avo_document[k] for k in AR.AVO_CLOCKS if k != "sync_observed_raw_ns"}
    clocks["sync_observed_raw_ns"] = case.avo_prescore["source_synchronization"]["observed_raw_ns"]
    with AC.measure() as counters:
        legacy = AR.legacy_target(staged, target)
        prepared = AR.prepared_target(staged, legacy)
        documents = AR.avo_documents(staged, avo, identity=kw["identity"], clocks=clocks,
            payload_filenames={k: v["filename"] for k, v in case.avo_document["payloads"].items()},
            prescore_filename=case.avo_document["prescore_input"]["filename"])
        binding = AR.binding(payload, legacy, prepared, avo_document_sha256=documents["document_sha256"])
    row["counters"]["deferred_artifact_path"] = counters.table()
    tj = case.target_json
    legacy_meta = {k: v for k, v in legacy.items() if k not in ("segmentation", "object_ignore", "segmentation_png",
                                                                 "object_ignore_png", "_csv_bytes")}
    retained_seg = case.bundle.prepared.target["segmentation"]
    retained_obj = case.bundle.prepared.target["object_ignore"]
    checks = row["checks"]
    checks.update(
        native_inputs_exact=all(canon(getattr(staged.inputs, name)) == canon(getattr(case.bundle.inputs, name))
                                for name in ("frame", "split", "actor_boxes", "actor_visibility", "static_candidates", "quarantine", "provenance")),
        native_arrays_exact=all(np.array_equal(getattr(staged.inputs, name), getattr(case.bundle.inputs, name))
                                for name in ("depth_bgra", "raw_semantic", "original_v3_mask")),
        input_content_sha_exact=payload.input_content_sha256 == case.ready["input_content_sha256"],
        source_evidence_exact=canon(staged.evidence) == canon(case.evidence),
        segmentation_mask_exact=np.array_equal(payload.segmentation, retained_seg),
        object_ignore_mask_exact=np.array_equal(payload.object_ignore, retained_obj),
        object_boxes_exact=list(map(dict, payload.object_boxes)) == tj["tables"]["object_boxes.csv"],
        object_ignore_regions_exact=list(map(dict, payload.object_ignore_regions)) == tj["tables"]["object_ignore_regions.csv"],
        avo_exact=canon(avo.avo) == canon(case.avo_document["avo"]),
        avo_rows_exact=canon(list(avo.rows)) == canon(case.avo_document["raw_pedestrian_rows"]),
        legacy_target_json_exact=canon(legacy_meta) == canon(tj),
        segmentation_png_exact=legacy["segmentation_png"] == case.segmentation_png,
        object_ignore_png_exact=legacy["object_ignore_png"] == case.object_ignore_png,
        prepared_content_sha_exact=prepared.content_sha256 == case.ready["prepared_content_sha256"] == case.bundle.prepared.content_sha256,
        avo_document_sha_exact=documents["document_sha256"] == case.avo_sha256,
        avo_prescore_sha_exact=documents["prescore_sha256"] == case.prescore_sha256,
        payload_verified=NP.verify_payload(payload) == payload.semantic_sha256,
        semantic_digest_distinct_from_prepared=payload.semantic_sha256 != prepared.content_sha256,
    )
    # ---- scorer-facing parity: saved native predictions through the unchanged evaluator ----
    quality = quality_proof(prepared, json_value(payload.avo), prediction, spec, Q)
    checks["evaluation_evidence_exact"] = quality["evidence_exact"]
    checks["q_perc_exact"] = quality["q_perc_exact"]
    row.update(q_perc=quality["q_perc"], q_perc_undefined=quality["q_perc"] is None,
               measurement_sha256=quality["measurement_sha256"])
    # ---- history: commit primary, exercise cancel/recompute on the shadow ----
    before = shadow.history_sha256()
    with patch.object(SY, "source_alignment_record", alignment):
        trial = shadow.compute(**kw)
        cancelled = shadow.cancel(trial)
        checks["shadow_cancel_restores_history"] = cancelled["history_sha256"] == before == trial.history_before_sha256
        again = shadow.compute(**kw)
    shadow_target = NT.numeric_target(frame=again.frame, split="val", depth=again.depth, semantic=again.semantic,
        original_mask=again.mask, boxes=again.boxes, visibility=again.visibility, static_rows=again.static,
        quarantine=again.conflict["quarantine"], vehicle_mask=again.vehicle_mask)
    shadow_payload = NP.build_payload(staged=again, target=shadow_target, avo=avo)
    checks["shadow_recompute_payload_exact"] = (shadow_payload.semantic_sha256 == payload.semantic_sha256
                                                and again.inputs.provenance == staged.inputs.provenance)
    commit = numeric.commit(staged)
    shadow_commit = shadow.commit(again)
    checks["history_digests_consistent"] = (commit["history_after_sha256"] == payload.history_after_sha256
                                            == shadow_commit["history_after_sha256"]
                                            and commit["history_before_sha256"] == payload.history_before_sha256)
    row.update(semantic_sha256=payload.semantic_sha256, prepared_content_sha256=prepared.content_sha256,
               input_content_sha256=payload.input_content_sha256, binding=binding,
               history_before_sha256=payload.history_before_sha256, history_after_sha256=payload.history_after_sha256,
               eligible_people=sum(r["avo"] >= .65 for r in avo.avo["qualified"]),
               positive_vehicle_targets=sum(r["label"] == "vehicle" for r in payload.object_boxes),
               actor_boxes=len(staged.boxes), static_candidates=len(staged.static),
               pedestrians=len(avo.rows), all_exact=all(checks.values()))
    return row


def aggregate_counters(rows):
    out = {}
    for path in ("original_path", "numeric_readiness_path", "deferred_artifact_path"):
        out[path] = {}
        for name in AC.CATEGORIES:
            values = [r["counters"][path][name]["count"] for r in rows]
            out[path][name] = dict(min=min(values), p50=statistics.median(values), max=max(values),
                                   class_=AC.CATEGORIES[name])
    return out


def run(output: Path, frames: int) -> dict:
    environment = verify_environment()
    require(frames in (2, K.COHORT), "two-frame smoke or complete chronological 49 only")
    output = Path(output).absolute()
    require(not any(p.is_symlink() for p in (output, *output.parents)) and output.resolve().is_relative_to(EVIDENCE_ROOT)
            and not output.exists(), "fresh create-only output below the assigned evidence root required")
    binding = O.startup_binding()
    input_hashes = K.verify_inputs()
    cases = K.load_cases()
    predictions = K.load_predictions(cases)
    spec, Q = quality_bridge()
    cases = cases[:frames]
    frame_ids = [c.frame_id for c in cases]
    output.mkdir(parents=True, exist_ok=False)
    write(output / "INPUT_SHA256.json", input_hashes)
    plan = dict(schema="splitfusion.numeric_gt_producer.v11_parity_plan.v1", frames=frames, full_cohort=K.COHORT,
        frame_ids=frame_ids, cell=str(K.CELL), input_seal_sha256=K.INPUT_SEAL_SHA, startup_binding=binding,
        environment=environment, clock_override="SOURCE_EVIDENCE observed_raw_ns and the seven retained AVO clocks only, offline",
        fixture_limits=K.B3.FIXTURE_LIMITS, no_live=True, no_gpu=True, performance_claim=False)
    write(output / "PLAN.json", plan)
    args = K.producer_arguments(cases[0])
    original_producer = S.LiveCompleteTargetProducer(**args)
    numeric = NS.NumericSourceProducer(**args)
    shadow = NS.NumericSourceProducer(**args)
    rows = []
    status = "FAILED"
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="numeric-gt-producer-avo-sink-") as sink_dir:
            for case in cases:
                rows.append(run_frame(case, prediction=predictions[case.frame_id], spec=spec, Q=Q,
                    original_producer=original_producer, numeric=numeric, shadow=shadow, sink_dir=sink_dir))
                with (output / "PER_FRAME.jsonl").open("a") as stream:
                    stream.write(json.dumps(rows[-1], sort_keys=True, allow_nan=False) + "\n")
                print(json.dumps(dict(frame_id=case.frame_id, all_exact=rows[-1]["all_exact"],
                    q_perc=rows[-1]["q_perc"], failed=[k for k, v in rows[-1]["checks"].items() if not v])), flush=True)
        require(K.verify_inputs() == input_hashes, "sealed inputs changed during replay")
        require(O.startup_binding()["binding_sha256"] == binding["binding_sha256"], "sources changed during replay")
        all_exact = all(r["all_exact"] for r in rows)
        undefined = sum(r["q_perc_undefined"] for r in rows)
        if frames == K.COHORT:
            require(undefined == K.UNDEFINED_QUALITY, "undefined-quality population differs")
        status = ("ALL49_NUMERIC_PRODUCER_EXACT_WITH_8_UNDEFINED_QUALITY" if frames == K.COHORT and all_exact
                  else "SMOKE2_EXACT_NOT_PERFORMANCE" if all_exact else "PARITY_FAILED")
        result = dict(schema=RESULT_SCHEMA, status=status, unique_frames=frames, frame_ids=frame_ids,
            all_exact=all_exact, undefined_quality_frames=[r["frame_id"] for r in rows if r["q_perc_undefined"]],
            undefined_quality_count=undefined, failed_checks={r["frame_id"]: [k for k, v in r["checks"].items() if not v]
                                                               for r in rows if not r["all_exact"]},
            check_names=sorted(rows[0]["checks"]), accounting=aggregate_counters(rows),
            history=dict(primary_final_sha256=numeric.history_sha256(), shadow_final_sha256=shadow.history_sha256(),
                         primary_processed=numeric.processed_frames, shadow_processed=shadow.processed_frames,
                         shadow_cancelled_frames=frames),
            per_frame_summary=[{k: r[k] for k in ("frame_id", "all_exact", "q_perc", "eligible_people",
                "positive_vehicle_targets", "actor_boxes", "static_candidates", "semantic_sha256", "prepared_content_sha256")}
                for r in rows],
            startup_binding_sha256=binding["binding_sha256"], input_seal_sha256=K.INPUT_SEAL_SHA,
            elapsed_seconds=time.monotonic() - started, live_started=False, live_qualified=False, cuda_initialized=False,
            caveats=["Offline exact replay on reconstructed saved frames; not live deadline performance.",
                     "Deterministic call/byte counters are not timings; serial timing is a separate coordinator-run command.",
                     "The producer's semantic digest is a candidate; the coordinator owns the admitted DTO digest and wire format.",
                     "AVO documents are reproduced byte-exactly only when the seven original clocks are supplied; live clocks are genuine.",
                     "Sealed fixture reconstruction limits (vehicle pitch/roll placeholders, archived inverses) apply as in retained harnesses."])
        write(output / "RESULT.json", result)
        print(json.dumps(dict(status=status, frames=frames, undefined=undefined, all_exact=all_exact)), flush=True)
        return result
    except BaseException as error:
        write(output / "FAILURE.json", dict(error=f"{type(error).__name__}: {error}", completed_frames=len(rows), no_live=True))
        raise
    finally:
        write(output / "OUTPUT_SHA256.json", tree(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, choices=(2, K.COHORT), default=K.COHORT)
    arguments = parser.parse_args()
    run(arguments.output, arguments.frames)
