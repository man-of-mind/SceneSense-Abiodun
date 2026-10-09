"""Create-only postrun joins and paired quality; never a live launcher.

The denominator is the sealed input archive's complete prepared-policy cohort.
Every row survives missing transmission, GT, prediction, or replay evidence.
Model loading is lazy and explicit. Dense inference is permitted only by the
sealed v1 replay core after selected payload and prediction parity succeeds.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_parity_repair_v1 import readiness as R
from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
from rl_agent.splitfusion_live_parity_repair_v2 import prediction_archive as PA
from rl_agent.splitfusion_paired_quality_replay_v1 import capture as C
from rl_agent.splitfusion_paired_quality_replay_v1 import replay as RP
from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality as Q

SCHEMA = "splitfusion.paired_quality_analysis.v2"
UE = "run4_phase6/PHASE6_UE_EVIDENCE.json"
EDGE = "run4_phase6/edge_report.json"
CLOSE = "profiler/edge/evaluator_close_result.json"
PREDICTIONS = "profiler/edge/prediction_archive"
TARGETS = "parity_complete_targets"
AVO = "run5_live_avo_inputs"
REQUIRED_CELL_PINS = (UE, EDGE, CLOSE, PREDICTIONS + "/MANIFEST.json",
                      TARGETS + "/MANIFEST.json", AVO + "/MANIFEST.json")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _safe(path):
    path = Path(path).absolute()
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "symbolic evidence path")
    return path


def _json(path, digest=None):
    return R._json(R._read(_safe(path), digest))


def _key(identity):
    return PA.canonical(dict(identity))


def _index(rows, get_key):
    result = {}
    for row in rows:
        key = get_key(row)
        _require(key not in result, "duplicate evidence identity")
        result[key] = row
    return result


def _manifest(directory, expected, schema):
    directory = _safe(directory)
    document = _json(directory / "MANIFEST.json", expected)
    _require(document.get("schema") == schema, "foreign evidence archive schema")
    hashes = document.get("sha256")
    _require(isinstance(hashes, dict), "missing archive SHA inventory")
    _require({p.name for p in directory.iterdir()} == set(hashes) | {"MANIFEST.json"},
             "archive inventory mismatch")
    for name, digest in hashes.items():
        _require(isinstance(name, str) and Path(name).name == name, "unsafe archive member")
        R._read(_safe(directory / name), digest)
    return document


def load_cell(cell_directory: Path, evidence_sha256: Mapping[str, str]) -> dict:
    """Read/hash-bound retained outputs; no inference or writes.

    Callers must obtain the six outer pins from their trusted postrun evidence,
    rather than treating hashes calculated from arbitrary files as authority.
    Inner payload pins and all archive inventories are verified transitively.
    """
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as G
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
    root = _safe(cell_directory)
    _require(set(evidence_sha256) == set(REQUIRED_CELL_PINS), "exact six cell evidence pins required")
    for digest in evidence_sha256.values():
        R._digest(digest)
    ue, edge, close = (_json(root / name, evidence_sha256[name]) for name in (UE, EDGE, CLOSE))
    predictions = _manifest(root / PREDICTIONS, evidence_sha256[PREDICTIONS + "/MANIFEST.json"], PA.SCHEMA)
    _require(predictions.get("prediction_contract") == PA.native_prediction_contract().document(),
             "actual native calibrated prediction archive required")
    for flag in ("includes_gt_unavailable_tickets", "flush_after_feedback", "quality_and_terminals_unchanged"):
        _require(predictions.get(flag) is True, "prediction archive contract drift")
    _require(close.get("worker_alive") is False and close.get("parity_archive_complete") is True
             and close.get("parity_archive_errors") == [] and close.get("parity_archive") == predictions,
             "prediction archive not bound to drained evaluator close")
    targets = _json(root / TARGETS / "MANIFEST.json", evidence_sha256[TARGETS + "/MANIFEST.json"])
    _require(targets.get("schema") == "splitfusion.complete_target_preservation.v1", "foreign target preservation")
    avo = _manifest(root / AVO, evidence_sha256[AVO + "/MANIFEST.json"], G.SCHEMA)
    sent = _index(ue["transmitted_identities"], lambda r: _key(r["run4_identity"]))
    for row in sent.values():
        MP.validate_run4_identity(row["run4_identity"])
        _require(row["frame_id"] == row["run4_identity"]["frame_id"], "transmitted frame identity differs")
    decisions = _index(ue["decisions"], lambda r: (r["decision_session_uuid"], r["tensor_seq"], r["frame_id"]))
    target_index = _index(targets["frames"], lambda r: _key(T.validate_identity(r["identity"])))
    names = {"MANIFEST.json"}
    for target in targets["frames"]:
        directory = T.bundle_path(root / TARGETS, target["identity"])
        names.update((directory.name, directory.with_suffix(".sources").name))
    _require({p.name for p in (root / TARGETS).iterdir()} == names, "unbound/missing target bundle directory")
    archived = {}
    used_files = set()
    for name, digest in predictions["sha256"].items():
        if not name.endswith(".identity.json"):
            continue
        document = _json(root / PREDICTIONS / name, digest)
        identity = R._identity({k: document[k] for k in R._IDENTITY_KEYS})
        _require(set(document) == R._ARCHIVE_KEYS and document["schema"] == PA.SCHEMA,
                 "prediction identity schema drift")
        key = PA.PredictionArchive.key(identity)
        _require(name == key + ".identity.json", "prediction filename/full identity differs")
        _require(_key(identity["action_identity"]) not in archived, "duplicate archived action")
        _require(type(document["snapshot_raw_ns"]) is int and document["snapshot_raw_ns"] > 0,
                 "invalid prediction snapshot timestamp")
        for suffix in ("identity.json", "labels.npy", "records.json"):
            member = key + "." + suffix
            _require(member in predictions["sha256"], "missing prediction payload")
            used_files.add(member)
        if key + ".evaluation.json" in predictions["sha256"]:
            used_files.add(key + ".evaluation.json")
        archived[_key(identity["action_identity"])] = dict(document, archive_key=key)
    _require(used_files == set(predictions["sha256"]), "foreign prediction archive payload")
    _require(type(predictions["frames"]) is int and predictions["frames"] == len(archived),
             "prediction frame count mismatch")
    _require(predictions["evaluated_frames"] == sum(n.endswith(".evaluation.json") for n in used_files),
             "evaluated prediction count mismatch")
    return dict(root=root, ue=ue, edge=edge, close=close, predictions=predictions, avo=avo,
                sent=sent, decisions=decisions, targets=target_index, archived=archived,
                evaluations=_index(edge["evaluations"], lambda r: _key(r["run4_identity"])))


def join_frame(frame: Mapping, cell: Mapping) -> dict:
    """Resolve one prepared policy input to transmitted/native/complete GT.

    A missing original evaluation is allowed: it does not mean missing native
    predictions or unusable eventual complete GT. Its reason stays descriptive.
    """
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as G
    from rl_agent.splitfusion_run5_live_avo_v1 import verify_live_gt_capture as V
    from rl_agent.splitfusion_run5_live_avo_v1 import quality_adapter as people
    from rl_agent.splitfusion_run5_live_avo_v1.gt_source_sync import validate_source_alignment
    from types import SimpleNamespace
    identity, gt = frame["identity"], T.validate_identity(frame["gt_identity"])
    key, root = _key(identity), cell["root"]
    _require(key in cell["sent"], "MISSING_TRANSMITTED_IDENTITY")
    dkey = (identity["session_uuid"], identity["tensor_seq"], identity["frame_id"])
    _require(dkey in cell["decisions"], "MISSING_POLICY_DECISION")
    decision = cell["decisions"][dkey]
    _require(decision["kind"] == "POLICY_DECISION" and decision["mode_id"] == identity["mode_id"]
             and decision["q_e4"] == identity["q_e4"]
             and decision["anchor_action_id"] == identity["anchor_action_id"]
             and decision["capture"]["ns"] == gt["capture_timestamp_ns"]
             and decision["carla_timestamp"] == frame["carla_timestamp"]
             and decision["radar_window_sha256"] == frame["radar_window_sha256"],
             "policy/input capture, action, timestamp or radar binding differs")
    last = decision.get("stages", {}).get("last_packet_send_raw_ns")
    ledger = cell["ue"].get("gt_last_datagram", {})
    _require(type(last) is int and last > 0 and ledger.get("marks", {}).get(str(gt["frame_id"])) == last
             and str(gt["frame_id"]) not in ledger.get("duplicates", {}), "MISSING_OR_CONFLICTING_LASTSEND")
    _require(key in cell["archived"], "MISSING_ARCHIVED_PREDICTION")
    saved = cell["archived"][key]
    _require(saved["gt_identity"] == gt and saved["capture_timestamp_ns"] == gt["capture_timestamp_ns"],
             "prediction/input full GT identity differs")
    stem = saved["archive_key"]
    directory = root / PREDICTIONS
    files = cell["predictions"]["sha256"]
    records_raw = R._read(_safe(directory / (stem + ".records.json")), files[stem + ".records.json"])
    records = R._json(records_raw)
    _require(isinstance(records, list) and len(records) <= E.MAX_RECORDS, "invalid complete prediction count")
    labels = np.load(io.BytesIO(R._read(_safe(directory / (stem + ".labels.npy")), files[stem + ".labels.npy"])), allow_pickle=False)
    PA.validate_labels(labels)
    _require(hashlib.sha256(labels.tobytes() + records_raw).hexdigest() == saved["prediction_digest"],
             "prediction content digest mismatch")
    for record in records:
        _require(record.get("frame_id") == gt["frame_id"] and record.get("stream_id") == gt["stream_id"]
                 and record.get("capture_timestamp_ns") == gt["capture_timestamp_ns"],
                 "prediction frame/stream/timestamp differs")
    _require(_key(gt) in cell["targets"], "MISSING_COMPLETE_TARGET")
    descriptor = cell["targets"][_key(gt)]
    bundle = T.verify_full_inputs(root / TARGETS, expected_identity=gt,
                                 expected_manifest_sha256=descriptor["manifest_sha256"])
    source_dir = _safe(bundle.directory.with_suffix(".sources"))
    source_binding = R._verify_source_archive(dict(identity=gt, directory=str(source_dir),
        manifest_sha256=descriptor["source_manifest_sha256"]), dict(frame_id=gt["frame_id"],
        provenance=bundle.prepared.target["provenance"], native_shape=list(bundle.inputs.raw_semantic.shape)))
    source = _json(source_dir / "SOURCE_EVIDENCE.json", source_binding["sha256"]["SOURCE_EVIDENCE.json"])
    avo_path = G.sidecar_path(root / AVO, gt)
    _require(avo_path.name in cell["avo"]["sha256"], "MISSING_AVO")
    avo_sha = cell["avo"]["sha256"][avo_path.name]
    sidecar = _json(avo_path, avo_sha)
    _require(sidecar.get("schema") == G.SCHEMA and sidecar.get("mask_contract") == G.MASK_CONTRACT
             and sidecar.get("record_kind") == "AVO_SCORED_FRAME" and sidecar.get("identity") == gt,
             "foreign/incomplete AVO sidecar")
    _require(sidecar["source_frame_id"] == gt["frame_id"]
             and sidecar["sensor_timestamp"] == frame["carla_timestamp"]
             and sidecar["start_raw_ns"] >= last, "AVO source/timestamp/lastsend differs")
    V._verify_prescore_inputs(root / AVO, cell["avo"]["sha256"], sidecar)
    validate_source_alignment(sidecar["source_synchronization"])
    # The two GT workers observe the same owned sources at different times.
    # Observation time is not a sensor/frame identity or a pose measurement.
    sync_content = lambda value: {k: v for k, v in value.items() if k != "observed_raw_ns"}
    _require(sync_content(sidecar["source_synchronization"]) == sync_content(source["source_synchronization"]),
             "AVO/complete-target synchronized source differs")
    arrays = {}
    _require(set(sidecar["payloads"]) == {"depth_bgra", "rawsemantic"}, "incomplete AVO native inputs")
    for role, payload in sidecar["payloads"].items():
        name = payload["filename"]
        _require(Path(name).name == name and payload["sha256"] == cell["avo"]["sha256"].get(name),
                 "AVO native input binding differs")
        arrays[role] = np.load(io.BytesIO(R._read(_safe(root / AVO / name), payload["sha256"])), allow_pickle=False)
    _require(np.array_equal(arrays["depth_bgra"], bundle.inputs.depth_bgra)
             and np.array_equal(arrays["rawsemantic"], bundle.inputs.raw_semantic),
             "AVO/complete-target native inputs differ")
    height, width = sidecar["native_shape"]
    replayed_avo = people.score_live_avo_people(frame_id=gt["frame_id"], timestamp=sidecar["sensor_timestamp"],
        depth_frame_id=gt["frame_id"], depth_timestamp=sidecar["sensor_timestamp"],
        depth_m=G.decode_depth(SimpleNamespace(raw_data=arrays["depth_bgra"].tobytes(), width=width, height=height)),
        camera_matrix=np.asarray(sidecar["camera_matrix"]), camera_inverse=np.asarray(sidecar["camera_inverse"]),
        intrinsics=np.asarray(sidecar["intrinsics"]), raw_actor_rows=sidecar["raw_pedestrian_rows"],
        width=width, height=height)
    _require(_key(replayed_avo) == _key(sidecar["avo"]), "original AVO numerical replay differs")
    original = None
    if stem + ".evaluation.json" in files:
        original = _json(directory / (stem + ".evaluation.json"), files[stem + ".evaluation.json"])
        _require(original.get("schema") == E.SCHEMA and original.get("identity") == identity
                 and original.get("source_pins") == E.verify_sources()
                 and original.get("target_content_sha256") == bundle.prepared.content_sha256
                 and original.get("target_provenance") == bundle.prepared.target["provenance"]
                 and original.get("person_avo") == sidecar["avo"]
                 and original.get("prediction_records") == records
                 and original.get("prediction_records_sha256") == hashlib.sha256(PA.canonical(records)).hexdigest()
                 and original.get("predicted_labels") == {"shape": list(labels.shape), "dtype": str(labels.dtype),
                     "array_sha256": hashlib.sha256(labels.tobytes()).hexdigest()},
                 "original evaluation/prediction/target/AVO source binding differs")
    edge_row = cell["evaluations"].get(key)
    if edge_row is not None:
        _require(edge_row["enqueued_wall_ns"] == saved["enqueued_wall_ns"], "edge/prediction enqueue differs")
        if original is not None:
            _require(edge_row.get("measurement") == original["measurement"]
                     and edge_row.get("timing", {}).get("target_bundle_sha256") == bundle.manifest_sha256
                     and edge_row.get("timing", {}).get("avo_sidecar_sha256") == avo_sha,
                     "emitted/archived evaluation differs")
    return dict(archived_selected=RP.Prediction(labels, records), prepared=bundle.prepared,
        avo=sidecar["avo"], original_evaluation=original, source_camera_matrix=sidecar["camera_matrix"],
        evidence=dict(last_packet_send_raw_ns=last, prediction_key=stem,
            target_manifest_sha256=bundle.manifest_sha256, source_manifest_sha256=descriptor["source_manifest_sha256"],
            avo_sha256=avo_sha, original_evaluation_present=original is not None,
            original_evaluator_reason=None if edge_row is None else edge_row.get("reason")))


def verify_wire_source_pose(selected_wire: bytes, joined: Mapping) -> float:
    """Bind the actual replay camera to the same-frame complete target pose."""
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    from rl_agent.splitfusion_live_dispatch_v1.frame_context import StaticCameraRegistry, camera_world_matrix
    _, context = W.unpack_sfd4(selected_wire)
    calibration = StaticCameraRegistry.audited().resolve(context.camera_model_sha256, context.camera_mount_sha256)
    camera = camera_world_matrix(context, calibration)
    source = np.asarray(joined["source_camera_matrix"], dtype=np.float64)
    _require(source.shape == (4, 4) and bool(np.isfinite(source).all())
             and np.allclose(camera, source, rtol=0, atol=1e-5), "SFD4/complete target camera pose differs")
    return float(np.max(np.abs(camera - source)))


def summarize_measurement(measurement: Mapping, reward_spec) -> dict:
    """Report exact per-class XY and segmentation statistics, plus frozen Q."""
    exact = Q.evaluate_exact_quality(reward_spec, measurement)
    classes = {}
    for name in ("vehicle", "person"):
        prefix = "loc_" + name + "_"
        counts = {key: int(measurement[prefix + key]) for key in ("eligible_gt", "tp", "fp", "fn")}
        errors = [float(v) for v in measurement[prefix + "matched_xy_errors_m"]]
        _require(counts["tp"] + counts["fn"] == counts["eligible_gt"] and len(errors) == counts["tp"]
                 and all(math.isfinite(v) and 0 <= v <= 3 for v in errors), "invalid exact XY statistics")
        seg = {key: int(measurement["seg_" + name + "_" + key]) for key in (
            "gt_pixels", "pred_pixels", "intersection_pixels", "union_pixels")}
        classes[name] = dict(counts, recall=counts["tp"] / counts["eligible_gt"] if counts["eligible_gt"] else None,
            precision=counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else None,
            ignored_predictions=int(measurement.get(prefix + "ignored_predictions", 0)),
            matched_xy_errors_m=errors, matched_xy_error_mean_m=float(np.mean(errors)) if errors else None,
            matched_xy_error_median_m=float(np.median(errors)) if errors else None,
            segmentation_surrogate=dict(seg, iou=seg["intersection_pixels"] / seg["union_pixels"] if seg["union_pixels"] else None))
    fields = ("q_perc", "q_loc", "q_seg", "seg_vehicle_iou", "seg_person_iou", "vehicle_recall",
              "person_recall", "vehicle_xy_error_m", "person_xy_error_m")
    return dict(classes=classes, quality=dict(defined=exact is not None,
        undefined_reason="NO_ELIGIBLE_LOCALIZATION_GT" if exact is None else None,
        **{name: None if exact is None else getattr(exact, name) for name in fields}))


def summarize_cohort(rows: list[dict], global_faults=()) -> dict:
    """Full-cohort claims are suppressed on any missing/faulted prepared row."""
    paired = [r for r in rows if r["status"] == "PAIRED"]
    both_defined = [r for r in paired if r["selected"]["quality"]["defined"] and r["dense"]["quality"]["defined"]]
    complete = len(paired) == len(rows) and bool(rows) and not global_faults
    result = dict(prepared_policy_inputs=len(rows), paired_frames=len(paired), unpaired_frames=len(rows) - len(paired),
        status_counts=dict(Counter(r["status"] for r in rows)), full_cohort_complete=complete,
        q_both_defined_frames=len(both_defined), q_both_undefined_frames=sum(
            not r["selected"]["quality"]["defined"] and not r["dense"]["quality"]["defined"] for r in paired),
        q_definition_mismatch_frames=sum(r["selected"]["quality"]["defined"] != r["dense"]["quality"]["defined"] for r in paired),
        full_cohort_quality=None)
    if complete:
        result["full_cohort_quality"] = dict(q_defined_denominator=len(both_defined),
            q_undefined_denominator=len(rows) - len(both_defined),
            selected_mean_q_perc=float(np.mean([r["selected"]["quality"]["q_perc"] for r in both_defined])) if both_defined else None,
            dense_mean_q_perc=float(np.mean([r["dense"]["quality"]["q_perc"] for r in both_defined])) if both_defined else None,
            mean_dense_minus_selected_q_perc=float(np.mean([r["delta_q_perc"] for r in both_defined])) if both_defined else None)
        for branch in ("selected", "dense"):
            totals = {}
            for name in ("vehicle", "person"):
                classes = [r[branch]["classes"][name] for r in paired]
                counts = {k: sum(c[k] for c in classes) for k in ("eligible_gt", "tp", "fp", "fn", "ignored_predictions")}
                errors = [e for c in classes for e in c["matched_xy_errors_m"]]
                pixels = {k: sum(c["segmentation_surrogate"][k] for c in classes) for k in (
                    "gt_pixels", "pred_pixels", "intersection_pixels", "union_pixels")}
                totals[name] = dict(counts, recall=counts["tp"] / counts["eligible_gt"] if counts["eligible_gt"] else None,
                    precision=counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else None,
                    matched_xy_errors_m=errors, matched_xy_error_mean_m=float(np.mean(errors)) if errors else None,
                    matched_xy_error_median_m=float(np.median(errors)) if errors else None,
                    segmentation_surrogate=dict(pixels,
                        iou=pixels["intersection_pixels"] / pixels["union_pixels"] if pixels["union_pixels"] else None))
            result["full_cohort_quality"][branch + "_classes"] = totals
    return result


def analyze(input_archive: Path, cell_directory: Path, output_directory: Path, *,
            expected_input_manifest_sha256: str, evidence_sha256: Mapping[str, str],
            reward_spec_path: Path, expected_reward_spec_sha256: str,
            model_factory: Callable | None = None, model_options: Mapping | None = None,
            audit_only: bool = False) -> dict:
    """Write all cohort rows, sufficient statistics, exact quality and hashes.

    ``audit_only`` performs archive joins without creating a model. A factory
    receives ``model_options`` as keyword arguments and returns an object whose
    replay_callbacks() implements the sealed replay_pair callback contract.
    A missing join/fault remains a row and prevents a full-cohort comparison.
    """
    output = _safe(output_directory)
    _require(not output.exists(), "paired analysis output must be create-only")
    manifest = C.verify_archive(_safe(input_archive), expected_manifest_sha256=expected_input_manifest_sha256)
    spec = Q.load_reward_spec(_safe(reward_spec_path), expected_reward_spec_sha256)
    evaluator_sources = E.verify_sources()
    global_faults, rows = [], []
    try:
        cell = load_cell(cell_directory, evidence_sha256)
        cohort_keys = {_key(f["identity"]) for f in manifest["frames"]}
        foreign = [r["run4_identity"] for k, r in cell["sent"].items()
                   if r["run4_identity"]["reward_requested"] and k not in cohort_keys]
        _require(not foreign, "transmitted policy inputs absent from retained cohort")
        _require(not (set(cell["archived"]) - cohort_keys), "archived policy inputs absent from retained cohort")
    except Exception as exc:
        cell = None
        global_faults.append(f"{type(exc).__name__}: {exc}")
    adapter, adapter_error, adapter_bindings = None, None, None
    options = dict(model_options or {})
    model_evidence = dict(injected_factory=model_factory is not None,
        device=None if options.get("device") is None else str(options["device"]),
        allow_cold_load=options.get("allow_cold_load", False),
        expected_binding_hashes=dict(options.get("expected_binding_hashes", {})),
        repository_root=None if options.get("repository_root") is None else str(options["repository_root"]),
        load_attempted=False, loaded=False, postrun_sources_verified=False)
    for frame in manifest["frames"]:
        row = dict(index=frame["index"], identity=frame["identity"], gt_identity=frame["gt_identity"], status="JOIN_FAULT")
        rows.append(row)
        if cell is None:
            row["fault"] = global_faults[0]
            continue
        try:
            joined = join_frame(frame, cell)
            row["join"] = joined["evidence"]
            selected_wire = (Path(input_archive) / frame["wire_file"]).read_bytes()
            row["join"]["wire_target_camera_max_abs_delta"] = verify_wire_source_pose(selected_wire, joined)
        except Exception as exc:
            row["fault"] = f"{type(exc).__name__}: {exc}"
            continue
        if audit_only:
            row["status"] = "JOINED_NOT_REPLAYED"
            continue
        if adapter is None and adapter_error is None:
            model_evidence["load_attempted"] = True
            try:
                if model_factory is None:
                    from .model_adapter import load_adapter
                    model_factory = load_adapter
                adapter = model_factory(**options)
                adapter_bindings = dict(adapter.binding_hashes) if hasattr(adapter, "binding_hashes") else None
                model_evidence.update(loaded=True, actual_binding_hashes=adapter_bindings)
            except Exception as exc:
                adapter_error = f"MODEL_UNAVAILABLE {type(exc).__name__}: {exc}"
                global_faults.append(adapter_error)
        if adapter_error is not None:
            row.update(status="MODEL_UNAVAILABLE", fault=adapter_error)
            continue
        try:
            import torch
            array = np.load(Path(input_archive) / frame["input_file"], allow_pickle=False)
            pair = RP.replay_pair(prepared_input=torch.from_numpy(array),
                selected_wire=selected_wire,
                expected_identity=frame["identity"], archived_selected=joined["archived_selected"],
                **adapter.replay_callbacks())
            row["parity"] = dict(pair.evidence)
            for branch in ("selected", "dense"):
                prediction = getattr(pair, branch)
                evaluated = E.evaluate_prepared(identity=frame["identity"], prepared=joined["prepared"],
                    predicted_labels=prediction.labels, predictions=prediction.records, avo=joined["avo"])
                if branch == "selected" and joined["original_evaluation"] is not None:
                    _require(dict(evaluated.measurement) == joined["original_evaluation"]["measurement"],
                             "selected replay/original evaluator measurement mismatch")
                row[branch] = summarize_measurement(evaluated.measurement, spec)
                row[branch]["measurement"] = dict(evaluated.measurement)
                row[branch]["matched_pairs"] = evaluated.evidence["matched_pairs"]
                row[branch]["segmentation_original_score"] = evaluated.evidence["scoring"]["segmentation"]
            sq, dq = row["selected"]["quality"], row["dense"]["quality"]
            _require(sq["defined"] == dq["defined"], "same GT produced different Q support")
            row["delta_q_perc"] = dq["q_perc"] - sq["q_perc"] if sq["defined"] else None
            row["status"] = "PAIRED"
        except RP.ReplayParityError as exc:
            row.update(status="PARITY_FAULT", fault=f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            row.update(status="REPLAY_OR_SCORE_FAULT", fault=f"{type(exc).__name__}: {exc}")
    # Evidence files are read-only; detect drift during inference before publishing.
    for name, digest in evidence_sha256.items():
        try:
            _json(_safe(Path(cell_directory) / name), digest)
        except Exception as exc:
            global_faults.append(f"POSTRUN_EVIDENCE_DRIFT {name}: {exc}")
    try:
        C.verify_archive(input_archive, expected_manifest_sha256=expected_input_manifest_sha256)
        E.verify_sources()
    except Exception as exc:
        global_faults.append(f"POSTRUN_INPUT_OR_SOURCE_DRIFT: {exc}")
    if adapter_bindings is not None:
        try:
            from .model_adapter import _verify_sources
            _verify_sources(adapter_bindings, options.get("repository_root"))
            model_evidence["postrun_sources_verified"] = True
        except Exception as exc:
            global_faults.append(f"POSTRUN_MODEL_SOURCE_DRIFT: {exc}")
    report = dict(schema=SCHEMA, mode="ARCHIVE_JOIN_AUDIT" if audit_only else "PAIRED_QUALITY_REPLAY",
        dense_definition=RP.DENSE_DEFINITION, cohort_definition="ALL_PREPARED_POLICY_INPUTS",
        reward_formula="UNCHANGED_REGISTERED_EXACT_QUALITY", quality_calibration=Q.calibration_status(spec),
        live_latency_claim=False, policy_performance_qualification=False, training_authorized=False,
        input_manifest_sha256=expected_input_manifest_sha256, cell_evidence_sha256=dict(evidence_sha256),
        reward_spec_sha256=expected_reward_spec_sha256, verified_evaluator_sources=evaluator_sources,
        model=model_evidence,
        retained_delivery_diagnostics=None if cell is None else dict(
            ue_faulted=cell["ue"].get("faulted"), gt_missing_high_outputs=cell["ue"].get("gt_missing_high_outputs"),
            edge_failures=cell["edge"].get("failures"), edge_final=cell["edge"].get("final"),
            unresolved_tickets_at_close=cell["ue"].get("unresolved_tickets_at_close")),
        global_faults=global_faults, summary=summarize_cohort(rows, global_faults), rows=rows)
    output.mkdir(parents=False, exist_ok=False)
    C.write_json(output / "ANALYSIS.json", report)
    with (output / "FRAMES.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("index", "frame_id", "session_uuid", "tensor_seq", "status",
            "selected_q_perc", "dense_q_perc", "delta_q_perc", "fault"))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(index=row["index"], frame_id=row["identity"]["frame_id"],
                session_uuid=row["identity"]["session_uuid"], tensor_seq=row["identity"]["tensor_seq"],
                status=row["status"], selected_q_perc=row.get("selected", {}).get("quality", {}).get("q_perc"),
                dense_q_perc=row.get("dense", {}).get("quality", {}).get("q_perc"),
                delta_q_perc=row.get("delta_q_perc"), fault=row.get("fault")))
        stream.flush()
        os.fsync(stream.fileno())
    C.write_json(output / "MANIFEST.json", dict(schema=SCHEMA + ".output", complete=True,
        sha256={p.name: C.sha(p) for p in sorted(output.iterdir())},
        analysis_code_sha256=C.sha(Path(__file__))))
    T._fsync_directory(output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-archive", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--cell-pins", type=Path, required=True, help="JSON mapping of the six independently bound relative cell files")
    parser.add_argument("--reward-spec", type=Path, required=True)
    parser.add_argument("--reward-spec-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--model-options", type=Path, help="Explicit adapter binding/device/cold-load options JSON")
    args = parser.parse_args(argv)
    report = analyze(args.input_archive, args.cell, args.output,
        expected_input_manifest_sha256=args.input_manifest_sha256, evidence_sha256=_json(args.cell_pins),
        reward_spec_path=args.reward_spec, expected_reward_spec_sha256=args.reward_spec_sha256,
        model_options={} if args.model_options is None else _json(args.model_options), audit_only=args.audit_only)
    print(json.dumps(dict(output=str(args.output), summary=report["summary"], global_faults=report["global_faults"]), sort_keys=True))
    return 0 if (not report["global_faults"] and all(r["status"] in ("PAIRED", "JOINED_NOT_REPLAYED") for r in report["rows"])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
