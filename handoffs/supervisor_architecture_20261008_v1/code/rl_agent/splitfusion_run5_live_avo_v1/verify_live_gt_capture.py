"""Read-only qualification of synchronized live GT/AVO capture evidence.

This is an integration check, not policy-performance qualification. It replays
AVO for every reward frame from native encoded depth and frozen actor geometry;
the report cannot turn late feedback into success or relabel an old capture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import live_gt as G
from .quality_adapter import corrected_raw_semantic_3class, score_live_avo_people
from .gt_source_sync import SYNC_SCHEMA, validate_source_alignment


class VerificationError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise VerificationError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def _contained(root, name):
    _require(isinstance(name, str) and Path(name).name == name, "manifest filename escapes evidence directory")
    path = root / name
    _require(path.is_file() and not path.is_symlink(), f"missing/non-regular evidence: {name}")
    _require(path.resolve().parent == root.resolve(), "evidence path escapes directory")
    return path


def _verify_source_record(record):
    _require(record.get("schema") == SYNC_SCHEMA, "source synchronization schema mismatch")
    try:
        validate_source_alignment(record)
    except Exception as exc:
        raise VerificationError(f"source synchronization rejected: {exc}") from exc
    for source in record["sources"].values():
        stamp = float(source["timestamp"])
        _require(np.isfinite(stamp) and source.get("timestamp_repr") == repr(stamp)
                 and source.get("timestamp_hex") == stamp.hex(), "source timestamp representations disagree")
    _require(record.get("timestamp_tolerance_s") == 0. and record.get("pose_matrix_atol") == 1e-5,
             "source synchronization tolerance changed")


def _verify_sync_evidence(cell, summary, recorded):
    sync = summary.get("source_synchronization")
    _require(isinstance(sync, dict) and sync.get("schema") == SYNC_SCHEMA, "source synchronization summary missing")
    _require(sync.get("rejected") == 0 and sync.get("raw_bundles") == 0, "source synchronization failures recorded")
    _require(sync.get("accepted") == len(recorded), "source synchronization accepted count mismatch")
    directory = cell / "run5_live_sync_evidence"
    _require(directory.is_dir() and not directory.is_symlink(), "source synchronization directory missing")
    seen, sources, files = set(), {}, set()
    for accepted in sync.get("records", []):
        frame = int(accepted["frame_id"])
        _require(frame in recorded and frame not in seen, "source synchronization reward identities mismatch")
        seen.add(frame)
        stem = f"frame_{frame:010d}"
        observed_name, accepted_name = stem + ".observed.json", stem + ".accepted.json"
        observed_path = _contained(directory, observed_name)
        accepted_path = _contained(directory, accepted_name)
        _require(_sha(observed_path) == accepted.get("observed_sha256"), "observed source metadata digest mismatch")
        _require(_json(accepted_path) == accepted and accepted.get("status") == "EXACT_SOURCE_SYNCHRONIZATION_ACCEPTED",
                 "source synchronization accepted record mismatch")
        observed = _json(observed_path)
        _require(observed.pop("status", None) == "OBSERVED_BEFORE_VALIDATION", "source metadata not preserved before validation")
        _verify_source_record(observed)
        sources[frame] = observed
        files.update((observed_name, accepted_name))
    _require(seen == set(recorded), "source synchronization reward-frame coverage incomplete")
    _require({path.name for path in directory.iterdir()} == files, "unaccounted source synchronization evidence")
    barrier = summary.get("all_tick_sensor_barrier")
    _require(isinstance(barrier, dict) and barrier.get("outside_action_open") is True,
             "all-tick barrier timing evidence missing")
    _require(barrier.get("rejections", {}).get("rejected") == 0, "all-tick barrier rejected a frame")
    ticks, previous = {}, None
    for record in barrier.get("records", []):
        frame = int(record["identity"]["frame_id"])
        _require(frame not in ticks and (previous is None or frame == previous + 1), "all-tick barrier coverage has gaps or duplicates")
        _require(record.get("status") == "EXACT_ALL_SENSOR_TICK_ACCEPTED"
                 and record.get("outside_action_open") is True, "unaccepted all-tick barrier record")
        _verify_source_record(record)
        owned = np.asarray(record["owned_camera_matrix"], dtype=np.float64)
        _require(owned.shape == (4, 4) and np.all(np.isfinite(owned)), "barrier owned camera pose invalid")
        for role in ("rgb", "semantic", "depth"):
            _require(np.allclose(record["sources"][role]["camera_matrix"], owned, rtol=0, atol=1e-5),
                     "barrier camera differs from owned tick pose")
        _require(record.get("completed_roles") == ["depth", "radar", "rgb", "semantic"], "barrier parent callbacks incomplete")
        _require(record["end_raw_ns"] >= record["start_raw_ns"], "negative barrier interval")
        previous, ticks[frame] = frame, record
    _require(set(recorded).issubset(ticks), "reward frame not admitted by all-tick barrier")
    return sources, ticks


def _verify_prescore_inputs(inputs, hashes, sidecar):
    """Bind the raw inputs that existed before scoring, including failed cases.

    Clean-capture qualification refuses any rejected attempt separately. This
    function verifies successful records against the same pre-score evidence,
    never substituting a later reconstructed scene or silently excluding actors.
    """
    binding = sidecar.get("prescore_input")
    _require(isinstance(binding, dict) and set(binding) == {"filename", "sha256"},
             "pre-scoring raw input binding missing")
    name = binding["filename"]
    _require(name in hashes and binding["sha256"] == hashes[name], "pre-scoring input digest mismatch")
    document = _json(_contained(inputs, name))
    _require(document.get("schema") == G.SCHEMA and document.get("record_kind") == "AVO_PRE_SCORE_INPUTS",
             "pre-scoring input schema mismatch")
    for key in ("identity", "mask_contract", "source_frame_id", "sensor_timestamp", "native_shape",
                "camera_matrix", "camera_inverse", "intrinsics", "source_synchronization",
                "raw_pedestrian_rows", "payloads", "start_raw_ns", "write_raw_ns"):
        _require(key in document and document[key] == sidecar.get(key),
                 f"pre-scoring/final input mismatch: {key}")
    _require(isinstance(document.get("write_raw_ns"), int)
             and isinstance(sidecar.get("prescore_input_ready_raw_ns"), int)
             and isinstance(sidecar.get("scoring_start_raw_ns"), int),
             "pre-scoring timing evidence missing")
    _require(document["start_raw_ns"] <= document["write_raw_ns"] <= sidecar["prescore_input_ready_raw_ns"]
             <= sidecar["scoring_start_raw_ns"]
             <= sidecar["end_raw_ns"], "raw inputs were not preserved before scoring")
    geometry = sidecar.get("avo", {}).get("geometry_relevance")
    tilted = any(abs(float(row[key])) > 1e-6 for row in document["raw_pedestrian_rows"]
                 if row.get("label") == "person" for key in ("object_pitch_deg", "object_roll_deg"))
    if tilted:
        _require(isinstance(geometry, dict)
                 and isinstance(geometry.get("omitted_tilted_structural_actors"), list),
                 "geometry-relevance proof missing")
    # The entire proof, including every separation margin, is subsequently
    # recomputed by the unchanged-input scorer and compared canonically.
    return name


def verify_cell(cell: Path) -> dict:
    cell = Path(cell)
    inputs = cell / "run5_live_avo_inputs"
    summary_path = cell / "run5_live_avo_gt_summary.json"
    _require(inputs.is_dir() and summary_path.is_file(), "live AVO producer evidence missing")
    manifest = _json(inputs / "MANIFEST.json")
    _require(manifest.get("schema") == G.SCHEMA, "AVO manifest schema mismatch")
    hashes = manifest.get("sha256")
    _require(isinstance(hashes, dict) and bool(hashes), "empty AVO manifest")
    for name, digest in hashes.items():
        _require(_sha(_contained(inputs, name)) == digest, f"manifest digest mismatch: {name}")
    actual = {p.name for p in inputs.iterdir() if p.name != "MANIFEST.json"}
    _require(actual == set(hashes), "AVO directory has unmanifested or missing files")
    _require(not any(name.endswith(".avo_rejected.json") for name in hashes),
             "rejected AVO scoring evidence cannot qualify as a clean capture")
    summary = _json(summary_path)
    _require(summary.get("schema") == G.SCHEMA and summary.get("measurement_semantics") == G.MEASUREMENT_SEMANTICS,
             "GT producer summary contract mismatch")
    _require(summary.get("new_gt_helper_threads") == 0, "unexpected action-path GT helper")
    _require(summary.get("rejected_records") == [], "rejected or unaudited AVO scoring attempts")
    records = summary.get("records", [])
    _require(bool(records), "no reward-frame AVO records")
    recorded = {}
    for record in records:
        frame = int(record["frame_id"])
        _require(frame not in recorded, "duplicate frame in AVO summary")
        name = Path(record["path"]).name
        _require(name in hashes and record["sha256"] == hashes[name], "summary/manifest SHA mismatch")
        recorded[frame] = (name, record)
    synchronized, barrier_ticks = _verify_sync_evidence(cell, summary, recorded)

    ue_path = cell / "run4_phase6" / "PHASE6_UE_EVIDENCE.json"
    edge_path = cell / "run4_phase6" / "edge_report.json"
    _require(ue_path.is_file() and edge_path.is_file(), "UE/edge outcome evidence missing")
    ue, edge = _json(ue_path), _json(edge_path)
    sent = {}
    for row in ue["transmitted_identities"]:
        ident = row["run4_identity"]
        if ident.get("reward_requested") is True:
            frame = int(row["frame_id"])
            _require(frame not in sent and frame == int(ident["frame_id"]), "duplicate/mismatched sent reward identity")
            sent[frame] = ident
    _require(set(sent) == set(recorded), "sent reward frames differ from preserved AVO frame set")
    decisions = {int(r["frame_id"]): r for r in ue.get("decisions", []) if r.get("kind") == "POLICY_DECISION"}
    _require(set(decisions) == set(sent), "policy decision/actual transmission mismatch")
    _require(edge.get("final") is True and not edge.get("failures"), "edge report not clean and final")
    counters = edge.get("evaluator", {})
    for key in ("gt_expired", "queue_overflow"):
        _require(counters.get(key) == 0, f"evaluator {key} is not zero")

    replayed = {}
    ready_ms = []
    prescore_files = set()
    for frame, (name, record) in sorted(recorded.items()):
        sidecar = _json(inputs / name)
        _require(sidecar.get("schema") == G.SCHEMA and sidecar.get("mask_contract") == G.MASK_CONTRACT
                 and sidecar.get("record_kind") == "AVO_SCORED_FRAME",
                 "AVO sidecar schema/mask mismatch")
        prescore_file = _verify_prescore_inputs(inputs, hashes, sidecar)
        _require(prescore_file not in prescore_files, "pre-scoring input reused for different frames")
        prescore_files.add(prescore_file)
        identity = sidecar["identity"]
        _require(sidecar.get("source_synchronization") == synchronized[frame], "AVO/source synchronization linkage mismatch")
        _require(synchronized[frame]["identity"] == identity, "AVO/source synchronization identity mismatch")
        _require(int(identity["frame_id"]) == frame == int(sidecar["source_frame_id"]), "AVO frame mismatch")
        _require(identity["run_id"] == edge["run_id"] and identity["cell_id"] == edge["cell_id"], "AVO run/cell mismatch")
        _require(name == G.sidecar_path(inputs, identity).name, "AVO filename/stream identity mismatch")
        capture = decisions[frame].get("capture", {}).get("ns")
        _require(capture is not None and int(capture) == int(identity["capture_timestamp_ns"]), "capture timestamp mismatch")
        _require(float(sidecar["sensor_timestamp"]) == float(decisions[frame]["carla_timestamp"]), "CARLA timestamp mismatch")
        _require(float(sidecar["sensor_timestamp"]) == synchronized[frame]["sources"]["policy_radar"]["timestamp"],
                 "AVO/source policy timestamp mismatch")
        _require(sidecar["camera_matrix"] == synchronized[frame]["sources"]["rgb"]["camera_matrix"],
                 "AVO/source camera pose mismatch")
        _require(int(barrier_ticks[frame]["end_raw_ns"]) <= int(decisions[frame]["action_open"]["ns"]),
                 "sensor barrier overlaps action-open")
        stage = decisions[frame].get("stages", {})
        last_send = stage.get("last_packet_send_raw_ns")
        _require(last_send is not None and int(sidecar["start_raw_ns"]) >= int(last_send), "AVO started before final datagram")
        _require(int(sidecar["end_raw_ns"]) >= int(sidecar["start_raw_ns"]), "AVO negative interval")
        ready_ms.append((int(sidecar["end_raw_ns"]) - int(decisions[frame]["action_open"]["ns"])) / 1e6)
        height, width = map(int, sidecar["native_shape"])
        _require(set(sidecar["payloads"]) == {"depth_bgra", "rawsemantic"}, "missing/foreign raw sensor payload")
        arrays = {}
        for role, payload in sidecar["payloads"].items():
            filename = payload["filename"]
            _require(filename in hashes and payload["sha256"] == hashes[filename], "raw sensor digest mismatch")
            arrays[role] = np.load(_contained(inputs, filename), allow_pickle=False)
        bgra, tags = arrays["depth_bgra"], arrays["rawsemantic"]
        _require(bgra.dtype == np.uint8 and bgra.shape == (height, width, 4), "depth evidence layout mismatch")
        _require(tags.dtype == np.uint8 and tags.shape == (height, width), "semantic evidence layout mismatch")
        decoded = G.decode_depth(SimpleNamespace(raw_data=bgra.tobytes(), width=width, height=height))
        replay = score_live_avo_people(frame_id=frame, timestamp=float(sidecar["sensor_timestamp"]),
            depth_frame_id=frame, depth_timestamp=float(sidecar["sensor_timestamp"]), depth_m=decoded,
            camera_matrix=np.asarray(sidecar["camera_matrix"]), camera_inverse=np.asarray(sidecar["camera_inverse"]),
            intrinsics=np.asarray(sidecar["intrinsics"]), raw_actor_rows=sidecar["raw_pedestrian_rows"], width=width, height=height)
        _require(G._canonical(replay) == G._canonical(sidecar["avo"]), f"original AVO replay mismatch: {frame}")
        stem = name.removesuffix("avo.json")
        semantic_file = stem + "semantic.npy"
        _require(semantic_file in hashes, "corrected semantic mask not preserved")
        mask = np.load(_contained(inputs, semantic_file), allow_pickle=False)
        _require(np.array_equal(mask, corrected_raw_semantic_3class(tags)), f"corrected semantic LUT mismatch: {frame}")
        for suffix in ("objects.json", "semantic.json"):
            _require(stem + suffix in hashes, "object/semantic identity sidecar missing")
            component = _json(inputs / (stem + suffix))
            _require(all(component.get(key) == value for key, value in identity.items()), "GT component identity mismatch")
            _require(component.get("frozen_carla_frame_id") == frame, "GT component snapshot mismatch")
        sem = _json(inputs / (stem + "semantic.json"))
        _require(sem.get("sha256") == hashlib.sha256(mask.tobytes()).hexdigest(), "semantic content SHA mismatch")
        replayed[frame] = (hashes[name], replay)

    evaluations = edge.get("evaluations", [])
    _require(counters.get("emitted") == len(evaluations), "evaluator emitted accounting mismatch")
    evaluated = set()
    undefined = 0
    for row in evaluations:
        frame = int(row["frame_id"])
        _require(frame in replayed and frame not in evaluated, "foreign or duplicate evaluation")
        evaluated.add(frame)
        _require(row.get("run4_identity") == sent[frame], "edge/UE exact reward identity mismatch")
        _require(row.get("measurement_semantics") == G.MEASUREMENT_SEMANTICS, "wrong live evaluator semantics")
        measurement = row.get("measurement")
        _require(isinstance(measurement, dict), "missing quality measurement")
        _require(measurement.get("avo_sidecar_sha256") == replayed[frame][0], "edge/AVO SHA linkage mismatch")
        _require(measurement.get("live_gt_contract") == G.MEASUREMENT_SEMANTICS, "measurement GT contract mismatch")
        _require(measurement.get("frame_id") == frame, "measurement frame mismatch")
        eligible = sum(item["avo"] >= .65 for item in replayed[frame][1]["qualified"])
        _require(measurement.get("loc_person_eligible_gt") == eligible, "person eligibility denominator mismatch")
        _require(measurement.get("loc_person_tp", -1) + measurement.get("loc_person_fn", -1) == eligible,
                 "person TP/FN accounting mismatch")
        _require(row.get("reason") in ("NONE", "QUALITY_UNDEFINED_NO_ELIGIBLE_GT"), "evaluator fault/exception")
        undefined += row.get("reason") == "QUALITY_UNDEFINED_NO_ELIGIBLE_GT"
    _require(bool(evaluations), "no completed live quality evaluations")
    return {"cell": str(cell), "reward_frames": len(sent), "avo_replayed_exact": len(replayed),
            "corrected_semantic_masks_verified": len(replayed), "edge_evaluations_linked": len(evaluated),
            "prescore_input_bundles_verified": len(prescore_files),
            "geometry_relevance_proofs_replayed": sum("geometry_relevance" in replay for _, replay in replayed.values()),
            "exact_source_synchronization_verified": len(synchronized), "all_world_ticks_barrier_verified": len(barrier_ticks),
            "reward_frames_without_evaluation": sorted(set(sent) - evaluated),
            "undefined_quality_outcomes": undefined,
            "avo_ready_after_action_open_ms": {"p50": float(np.percentile(ready_ms, 50)),
                                                "p95": float(np.percentile(ready_ms, 95)), "max": max(ready_ms)},
            "input_manifest_sha256": _sha(inputs / "MANIFEST.json"),
            "producer_summary_sha256": _sha(summary_path), "ue_sha256": _sha(ue_path), "edge_sha256": _sha(edge_path)}


def verify_capture(root: Path) -> dict:
    root = Path(root).resolve()
    summaries = sorted(root.glob("cells/*/run5_live_avo_gt_summary.json"))
    if (root / "run5_live_avo_gt_summary.json").is_file():
        summaries = [root / "run5_live_avo_gt_summary.json"]
    _require(bool(summaries), "no live AVO capture cells found")
    return {"schema": "splitfusion.run5.live_gt_capture_verification.v1",
            "status": "LIVE_GT_AVO_INTEGRATION_VERIFIED", "policy_performance_qualification": False,
            "full_offline_v010_equivalence_claim": False, "cells": [verify_cell(p.parent) for p in summaries]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_capture(args.capture_root)
    except Exception as exc:
        print(json.dumps({"status": "LIVE_GT_AVO_NOT_VERIFIED", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
