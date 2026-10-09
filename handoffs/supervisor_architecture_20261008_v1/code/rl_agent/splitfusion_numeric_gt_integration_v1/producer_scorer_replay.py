"""Offline producer/scorer composition, NOT the proposed fast runtime path.

The current v1 DTO requires full archive-document hashes. This reference
composition honestly builds those artifacts before admission; it must never
be used as proof of archive-free readiness or of live performance. Numerical
targets come from the actual new producer, not from retained target restoration.
All 49 chronological fixtures, including undefined-quality frames, are required.
"""
from contextlib import ExitStack
import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from . import contract as C, evidence as V

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "experiments/splitfusion_numeric_gt_integration_v1"
PRODUCER_ROOT = ROOT / "experiments/splitfusion_numeric_gt_producer_v1/20261005_v1"
SCORER_ROOT = ROOT / "experiments/splitfusion_minimal_quality_scorer_v1"


def sha(path):
    path = Path(path)
    C.require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
              "regular nonsymbolic evidence/source file required")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for raw in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(raw)
    return h.hexdigest()


def verify_manifest(root):
    manifest_path = root / "OUTPUT_SHA256.json"
    doc = json.loads(manifest_path.read_bytes())
    files = doc.get("files", doc)
    base = ROOT if doc.get("paths_relative_to") == "repository_root" else root
    checked = {str(manifest_path.relative_to(ROOT)): sha(manifest_path)}
    for name, expected in files.items():
        relative = Path(name)
        C.require(not relative.is_absolute() and ".." not in relative.parts, "unsafe manifest path")
        path = base / relative
        C.require(sha(path) == expected, "component output changed: " + str(path))
        checked[str(path.relative_to(ROOT))] = expected
    return checked


def source_inventory(binding):
    paths = {ROOT / item["path"] for item in binding["pins"].values()}
    paths.update(ROOT / name for name in binding["avo_pins"])
    # Include imported optimization/conflict implementations, not just the
    # producer's declared original-source pins and its own package digest.
    packages = ("splitfusion_numeric_gt_integration_v1", "splitfusion_numeric_gt_producer_v1",
        "splitfusion_minimal_quality_scorer_v1", "splitfusion_live_gt_optimization_v1",
        "splitfusion_live_gt_optimization_v2", "splitfusion_live_gt_optimization_v3",
        "splitfusion_live_parity_repair_v1",
        "splitfusion_live_parity_repair_v2", "splitfusion_live_target_parity_v1",
        "splitfusion_run5_live_avo_v1", "splitfusion_live_gt_handoff_v1",
        "splitfusion_hybrid_sac_live_route_b_v2", "splitfusion_quality_feedback_probe_v1",
        "splitfusion_hybrid_sac_v1/offline_quality_grid")
    for package in packages:
        paths.update((ROOT / "rl_agent" / package).glob("*.py"))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def admit_reference_payload(payload, source, *, policy, plan, binding, producer_binding):
    """Reference-only adapter: full source archive already exists at admission."""
    from rl_agent.splitfusion_numeric_gt_producer_v1 import payload as P
    P.verify_payload(payload)
    C.require(payload.binding_sha256 == producer_binding, "foreign producer binding")
    V.require_source(source)
    metadata = dict(schema=C.SCHEMA, target_contract=C.TARGET_CONTRACT, binding=binding,
        capture_identity=payload.capture_identity, policy_identity=policy, plan_sha256=plan,
        sample_id=payload.sample_id, input_content_sha256=payload.input_content_sha256,
        history_before_sha256=payload.history_before_sha256, history_after_sha256=payload.history_after_sha256,
        provenance=payload.provenance, camera_inverse=payload.camera_inverse,
        object_boxes=payload.object_boxes, object_ignore_regions=payload.object_ignore_regions, avo=payload.avo,
        artifact_source_sha256=source.content_sha256, avo_document_sha256=source.avo_document_sha256)
    # Original artifact serialization normalizes NumPy float64 scalars to JSON
    # floats. Preserve its actual bytes/values, then apply strict DTO validation.
    metadata = json.loads(P.A.canonical(P.json_value(metadata)))
    target = C.admit_target(metadata=metadata, segmentation=payload.segmentation_bytes,
        object_ignore=payload.object_ignore_bytes, expected_binding=binding,
        expected_capture_identity=payload.capture_identity, expected_policy_identity=policy,
        expected_plan_sha256=plan)
    V.reconcile_source(target, source)
    return target


def run_frame(case, prediction, *, producer, scorer, binding, producer_binding, prediction_contract, spec, quality):
    from rl_agent.splitfusion_numeric_gt_producer_v1 import artifacts as AR, avo as NA, payload as P, target as NT
    from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import KEYS
    from rl_agent.splitfusion_minimal_quality_scorer_v1 import scorer as S
    from rl_agent.splitfusion_minimal_quality_scorer_v1.shared_admission import admit_prediction, join_admitted_inputs
    from rl_agent.splitfusion_minimal_quality_scorer_v1.validate import forbidden, guarded_score
    from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E, target_transport as T
    from rl_agent.splitfusion_run5_live_avo_v1 import gt_source_sync as SY

    population = case.evidence["dynamic_snapshot_evidence"]
    if producer.population != frozenset(population["registered_population_ids"]):
        producer.register_population(population["registered_population_ids"], population["population_sha256"],
                                     before_frame_id=case.frame_id)
    original_alignment = SY.source_alignment_record
    def alignment(*args, **kwargs):
        value = original_alignment(*args, **kwargs)
        value["observed_raw_ns"] = case.evidence["source_synchronization"]["observed_raw_ns"]
        return value
    kw = {key: case.kwargs[key] for key in KEYS}
    # Fixture clock substitution only; never installed into any live module.
    with patch.object(SY, "source_alignment_record", alignment):
        staged = producer.compute(**kw)
    numeric = NT.numeric_target(frame=staged.frame, split="val", depth=staged.depth, semantic=staged.semantic,
        original_mask=staged.mask, boxes=staged.boxes, visibility=staged.visibility, static_rows=staged.static,
        quarantine=staged.conflict["quarantine"], vehicle_mask=staged.vehicle_mask)
    avo = NA.numeric_avo(identity=kw["identity"], rgb=kw["rgb"], semantic_image=kw["semantic_image"],
        depth_image=kw["depth_image"], frozen_world=kw["frozen_world"], camera_matrix=staged.camera_matrix,
        camera_inverse=staged.camera_inverse, depth=staged.depth, sync_validated=True)
    payload = P.build_payload(staged=staged, target=numeric, avo=avo)

    # Explicitly archive-before-score reference composition. No timing claim.
    clocks = {k: case.avo_document[k] for k in AR.AVO_CLOCKS if k != "sync_observed_raw_ns"}
    clocks["sync_observed_raw_ns"] = case.avo_prescore["source_synchronization"]["observed_raw_ns"]
    documents = AR.avo_documents(staged, avo, identity=kw["identity"], clocks=clocks,
        payload_filenames={k: v["filename"] for k, v in case.avo_document["payloads"].items()},
        prescore_filename=case.avo_document["prescore_input"]["filename"])
    input_metadata = json.loads(P.A.canonical(T._input_metadata(staged.inputs)))
    source = V.admit_source_evidence(input_metadata=input_metadata,
        avo_document=json.loads(documents["document_bytes"]), expected_capture_identity=kw["identity"])
    action = prediction["identity"]["action_identity"]
    policy = {key: action[key] for key in C.POLICY_FIELDS}
    plan = C.sha256(C.canonical(dict(scope="OFFLINE_COMPOSITION_NOT_LIVE_PLAN", action=action,
                                   capture=kw["identity"])))
    target = admit_reference_payload(payload, source, policy=policy, plan=plan, binding=binding,
                                     producer_binding=producer_binding)
    admitted_prediction = admit_prediction(scorer, identity=prediction["identity"], labels=prediction["labels"],
        records=prediction["records"], prediction_contract=prediction_contract)
    with ExitStack() as stack:
        for name in ("builtins.open", "io.open", "pathlib.Path.read_bytes", "pathlib.Path.read_text",
                     "hashlib.sha256", "json.dumps", "json.loads", "cv2.imread", "cv2.imdecode", "cv2.imencode"):
            stack.enter_context(patch(name, forbidden))
        stack.enter_context(patch.object(E, "restore_prepared_target", forbidden))
        frame = join_admitted_inputs(scorer, target=target, prediction=admitted_prediction)
    result, counts = guarded_score(scorer, frame)
    qvalue = quality.evaluate_exact_quality(spec, result.measurement)
    qperc = None if qvalue is None else float(qvalue.q_perc)

    legacy = AR.legacy_target(staged, numeric)
    prepared = AR.prepared_target(staged, legacy)
    reference = V.own_legacy_reference(target, source, target_tables=legacy["tables"],
                                       old_prepared_sha256=prepared.content_sha256)
    with patch.dict(scorer._detection, match_frame=forbidden, load_gt=forbidden, load_predictions=forbidden), \
         patch.dict(scorer._person, greedy_match=forbidden):
        evidence = S.reconstruct_evidence(result, legacy_reference=reference)
        archived = S.archive_payloads(result, legacy_reference=reference)
    oracle = E.evaluate_prepared(identity=action, prepared=case.bundle.prepared,
        predicted_labels=prediction["labels"], predictions=prediction["records"], avo=case.avo_document["avo"])
    commit = producer.commit(staged)
    checks = dict(measurement_exact=result.measurement == oracle.measurement,
        saved_evidence_exact=evidence == prediction["evidence"],
        saved_evidence_hash_exact=C.sha256(S.canonical(evidence)) == prediction["evidence_sha256"],
        quality_exact=qperc == prediction["q_perc"], archive_bytes_exact=archived == E.archive_payloads(oracle),
        prepared_digest_exact=prepared.content_sha256 == case.ready["prepared_content_sha256"],
        segmentation_png_exact=legacy["segmentation_png"] == case.segmentation_png,
        object_ignore_png_exact=legacy["object_ignore_png"] == case.object_ignore_png,
        avo_document_exact=documents["document_sha256"] == case.avo_sha256,
        avo_prescore_exact=documents["prescore_sha256"] == case.prescore_sha256,
        history_commit_exact=commit["history_after_sha256"] == payload.history_after_sha256,
        distinct_semantic_domains=len({payload.semantic_sha256, target.semantic_sha256, prepared.content_sha256}) == 3)
    C.require(all(checks.values()), "composed parity failed: " + str(case.frame_id) + " " + str(checks))
    return dict(frame_id=case.frame_id, checks=checks, q_perc=qperc, matching_calls=counts,
        producer_semantic_sha256=payload.semantic_sha256, coordinator_semantic_sha256=target.semantic_sha256,
        history_before_sha256=payload.history_before_sha256, history_after_sha256=payload.history_after_sha256)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def run(output):
    from rl_agent.splitfusion_numeric_gt_producer_v1 import cases as K, originals as O, source as NS
    from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import verify_environment, quality_bridge
    from rl_agent.splitfusion_minimal_quality_scorer_v1.scorer import NumericalScorer
    output = Path(output).absolute()
    C.require(not output.exists() and output.resolve().is_relative_to(OUTPUT_ROOT)
        and not any(p.is_symlink() for p in (output, *output.parents)), "fresh owned output required")
    environment = verify_environment()
    seals = {**verify_manifest(PRODUCER_ROOT), **verify_manifest(SCORER_ROOT)}
    registration = json.loads((PRODUCER_ROOT / "full49_parity/PLAN.json").read_bytes())
    startup = O.startup_binding()
    C.require(startup == registration["startup_binding"], "producer code/source binding changed")
    sources = source_inventory(startup)
    inputs = K.verify_inputs()
    cases = K.load_cases()
    predictions = K.load_predictions(cases)
    prediction_contract = K.read(K.CELL / "profiler/edge/prediction_archive/MANIFEST.json")["prediction_contract"]
    binding = C.sha256(C.canonical(dict(scope="OFFLINE_REFERENCE_COMPOSITION_V1", sources=sources,
                                      retained_seal=K.INPUT_SEAL_SHA)))
    output.mkdir(parents=True, exist_ok=False)
    write(output / "PLAN.json", dict(schema="splitfusion.numeric_gt.reference_composition.v1",
        frames=49, environment=environment, source_sha256=sources, component_evidence_sha256=seals,
        input_sha256=inputs, binding=binding, order="ARCHIVE_AVO_BEFORE_SCORE_REFERENCE_ONLY",
        performance_claim=False, live_authorization=False,
        fixture_limits=registration["fixture_limits"], clocks="RECORDED_OFFLINE_FIXTURE_CLOCKS_ONLY"))
    producer, scorer = NS.NumericSourceProducer(**K.producer_arguments(cases[0])), NumericalScorer()
    spec, quality = quality_bridge()
    rows = []
    for case in cases:
        rows.append(run_frame(case, predictions[case.frame_id], producer=producer, scorer=scorer,
            binding=binding, producer_binding=startup["binding_sha256"], prediction_contract=prediction_contract,
            spec=spec, quality=quality))
        if len(rows) % 10 == 0:
            print(json.dumps(dict(composed=len(rows), total=49)), flush=True)
    undefined = [r["frame_id"] for r in rows if r["q_perc"] is None]
    C.require(len(rows) == 49 and len(undefined) == 8, "composed population differs")
    C.require(source_inventory(startup) == sources, "source changed during replay")
    C.require({**verify_manifest(PRODUCER_ROOT), **verify_manifest(SCORER_ROOT)} == seals,
              "component proof changed during replay")
    C.require(K.verify_inputs() == inputs, "retained input changed during replay")
    result = dict(status="ALL49_PRODUCER_SCORER_REFERENCE_COMPOSITION_EXACT_NOT_FAST_PATH",
        unique_frames=len(rows), undefined_quality_frames=undefined, all_exact=True, rows=rows,
        live_qualified=False, performance_measured=False, archive_free_readiness_proved=False,
        actual_live_FrameContext_pose_reconciled=False, live_send_history_transaction_proved=False,
        numerical_scorer_unchanged=True, producer_unchanged=True, inputs_and_components_unchanged=True,
        remaining_required_interface_change="Versioned ready DTO and scorer bridge with archive-only document bindings")
    write(output / "RESULT.json", result)
    write(output / "OUTPUT_SHA256.json", {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps({k: result[k] for k in ("status", "unique_frames", "all_exact", "undefined_quality_frames")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
