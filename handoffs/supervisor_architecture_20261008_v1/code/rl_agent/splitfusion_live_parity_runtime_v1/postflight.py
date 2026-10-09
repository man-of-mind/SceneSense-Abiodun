"""Create-only composed postflight for the repaired live GT contract.

Ticket populations come from UE transmissions and the independent edge report,
never from whichever prediction files survived. No services or inference run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rl_agent.splitfusion_live_parity_repair_v1 import readiness as R
from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
from rl_agent.splitfusion_live_parity_repair_v1.prediction_archive import PredictionArchive, canonical

SCHEMA = "splitfusion.live_parity_runtime.postflight.v1"
SEMANTICS = "V010_NUMERICAL_RULES__PROSPECTIVE_LIVE_CONSERVATIVE_CONFLICTS__PERSON_AVO_GE065"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return R._json(R._read(Path(path)))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value))


def _index(rows, frame_key="frame_id"):
    result = {}
    for row in rows:
        frame = row[frame_key]
        require(type(frame) is int and frame not in result, "duplicate or invalid frame identity")
        result[frame] = row
    return result


def _population(cell):
    """Independent sent/evaluated population and exact enqueue timestamps."""
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
    cell = Path(cell)
    ue = read(cell / "run4_phase6/PHASE6_UE_EVIDENCE.json")
    edge = read(cell / "run4_phase6/edge_report.json")
    summary = read(cell / "run5_live_avo_gt_summary.json")
    sent = _index([r for r in ue["transmitted_identities"] if r["run4_identity"]["reward_requested"]])
    decisions = _index([r for r in ue["decisions"] if r["kind"] == "POLICY_DECISION"])
    records = _index(summary["records"])
    require(bool(sent) and set(sent) == set(decisions) == set(records), "UE/decision/AVO population mismatch")
    require(edge.get("final") is True and edge.get("failures") == [], "edge is not clean and final")
    require(ue.get("faulted") is None and ue.get("gt_missing_high_outputs") == [], "UE/GT producer fault")
    require(summary.get("rejected_records") == [], "rejected GT sources")
    sidecars, gt = {}, {}
    inputs = cell / "run5_live_avo_inputs"
    manifest = read(inputs / "MANIFEST.json")
    for frame, record in records.items():
        name = Path(record["path"]).name
        require(name in manifest["sha256"] and record["sha256"] == manifest["sha256"][name]
                == sha(inputs / name), "AVO summary/input hash mismatch")
        sidecar = read(inputs / name)
        identity = T.validate_identity(sidecar["identity"])
        from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as G
        require(name == G.sidecar_path(inputs, identity).name, "AVO source filename/identity mismatch")
        action = sent[frame]["run4_identity"]
        MP.validate_run4_identity(action)
        require(identity["frame_id"] == frame == action["frame_id"], "sent/GT frame mismatch")
        require(identity["run_id"] == edge["run_id"] and identity["cell_id"] == edge["cell_id"],
                "GT run/cell mismatch")
        require(identity["action_id"] == action["anchor_action_id"]
                and identity["capture_timestamp_ns"] == decisions[frame]["capture"]["ns"],
                "GT anchor/capture mismatch")
        sidecars[frame], gt[frame] = sidecar, identity
    evaluations = _index(edge["evaluations"])
    require(bool(evaluations) and set(evaluations) <= set(sent), "foreign/empty edge evaluation population")
    require(set(evaluations) == set(sent), "reward frames without independent edge evaluation: "
            + repr(sorted(set(sent) - set(evaluations))))
    counters = edge["evaluator"]
    require(counters.get("submitted") == counters.get("emitted") == len(evaluations)
            == edge["counters"].get("evaluations_submitted"), "edge submission/emission population mismatch")
    require(counters.get("gt_expired") == counters.get("queue_overflow") == 0, "GT expiry or evaluator overflow")
    expected = []
    for frame, row in sorted(evaluations.items()):
        require(row["run4_identity"] == sent[frame]["run4_identity"], "edge/UE exact action identity mismatch")
        require(row["reason"] in ("NONE", "QUALITY_UNDEFINED_NO_ELIGIBLE_GT"), "evaluator exception/fault")
        require(row.get("measurement_semantics") == SEMANTICS, "old or foreign evaluator semantics")
        require(row.get("timing", {}).get("avo_sidecar_sha256") == records[frame]["sha256"],
                "edge/AVO actual read linkage mismatch")
        expected.append({"gt_identity": gt[frame], "action_identity": row["run4_identity"],
            "capture_timestamp_ns": gt[frame]["capture_timestamp_ns"], "enqueued_wall_ns": row["enqueued_wall_ns"]})
    return dict(ue=ue, edge=edge, summary=summary, sent=sent, decisions=decisions, records=records,
                sidecars=sidecars, gt=gt, evaluations=evaluations, expected=expected)


def verify_avo_cell(cell):
    """Retain exact legacy AVO/source tests; require the NEW evaluator contract.

The old verifier's raw-mask *evaluator* semantic assertion is intentionally not
reused. Its source synchronization/prescore checks and original AVO scorer are.
The auxiliary raw three-class mask is still preserved and checked, but cannot
stand in for the complete v010 target checked by verify_cell below.
"""
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as G
    from rl_agent.splitfusion_run5_live_avo_v1 import verify_live_gt_capture as V
    from rl_agent.splitfusion_run5_live_avo_v1.quality_adapter import corrected_raw_semantic_3class, score_live_avo_people
    cell = Path(cell)
    data = _population(cell)
    inputs = cell / "run5_live_avo_inputs"
    manifest = read(inputs / "MANIFEST.json")
    hashes = manifest["sha256"]
    require(manifest.get("schema") == G.SCHEMA and bool(hashes), "AVO manifest contract mismatch")
    require(not any(name.endswith(".avo_rejected.json") for name in hashes),
            "rejected AVO scoring evidence cannot qualify as a clean capture")
    require({p.name for p in inputs.iterdir()} == set(hashes) | {"MANIFEST.json"}, "AVO source inventory mismatch")
    for name, digest in hashes.items():
        require(sha(V._contained(inputs, name)) == digest, "AVO source content hash mismatch")
    summary = data["summary"]
    require(summary.get("schema") == G.SCHEMA and summary.get("measurement_semantics") == G.MEASUREMENT_SEMANTICS,
            "AVO producer contract mismatch")
    require(summary.get("new_gt_helper_threads") == 0, "unexpected GT worker")
    synchronized, ticks = V._verify_sync_evidence(cell, summary, data["records"])
    for frame, sidecar in data["sidecars"].items():
        require(sidecar.get("schema") == G.SCHEMA and sidecar.get("mask_contract") == G.MASK_CONTRACT
                and sidecar.get("record_kind") == "AVO_SCORED_FRAME", "AVO sidecar contract mismatch")
        V._verify_prescore_inputs(inputs, hashes, sidecar)
        require(sidecar["source_synchronization"] == synchronized[frame]
                and synchronized[frame]["identity"] == sidecar["identity"], "AVO/source linkage mismatch")
        require(sidecar["source_frame_id"] == frame and sidecar["camera_matrix"]
                == synchronized[frame]["sources"]["rgb"]["camera_matrix"], "AVO frame/camera mismatch")
        decision = data["decisions"][frame]
        require(ticks[frame]["end_raw_ns"] <= decision["action_open"]["ns"], "sensor barrier overlaps action-open")
        require(sidecar["start_raw_ns"] >= decision["stages"]["last_packet_send_raw_ns"], "GT work started before last send")
        require(sidecar["sensor_timestamp"] == decision["carla_timestamp"], "AVO policy timestamp mismatch")
        height, width = sidecar["native_shape"]
        arrays = {}
        require(set(sidecar["payloads"]) == {"depth_bgra", "rawsemantic"}, "native AVO source population mismatch")
        for role, payload in sidecar["payloads"].items():
            require(payload["sha256"] == hashes.get(payload["filename"]), "native AVO source digest mismatch")
            arrays[role] = np.load(V._contained(inputs, payload["filename"]), allow_pickle=False)
        depth, tags = arrays["depth_bgra"], arrays["rawsemantic"]
        require(depth.dtype == np.uint8 and depth.shape == (height, width, 4)
                and tags.dtype == np.uint8 and tags.shape == (height, width), "native AVO layout mismatch")
        replay = score_live_avo_people(frame_id=frame, timestamp=sidecar["sensor_timestamp"],
            depth_frame_id=frame, depth_timestamp=sidecar["sensor_timestamp"],
            depth_m=G.decode_depth(SimpleNamespace(raw_data=depth.tobytes(), width=width, height=height)),
            camera_matrix=np.asarray(sidecar["camera_matrix"]), camera_inverse=np.asarray(sidecar["camera_inverse"]),
            intrinsics=np.asarray(sidecar["intrinsics"]), raw_actor_rows=sidecar["raw_pedestrian_rows"], width=width, height=height)
        require(canonical(replay) == canonical(sidecar["avo"]), "original AVO numerical replay mismatch")
        stem = Path(data["records"][frame]["path"]).name.removesuffix("avo.json")
        mask = np.load(V._contained(inputs, stem + "semantic.npy"), allow_pickle=False)
        require(np.array_equal(mask, corrected_raw_semantic_3class(tags)), "auxiliary raw semantic source mismatch")
        for suffix in ("objects.json", "semantic.json"):
            component = read(V._contained(inputs, stem + suffix))
            require(all(component.get(k) == v for k, v in sidecar["identity"].items())
                    and component.get("frozen_carla_frame_id") == frame, "auxiliary GT identity mismatch")
        require(read(V._contained(inputs, stem + "semantic.json"))["sha256"]
                == hashlib.sha256(mask.tobytes()).hexdigest(), "auxiliary semantic content digest mismatch")
        if frame in data["evaluations"]:
            measurement = data["evaluations"][frame]["measurement"]
            count = sum(item["avo"] >= .65 for item in replay["qualified"])
            require(measurement["loc_person_eligible_gt"] == count
                    == measurement["loc_person_tp"] + measurement["loc_person_fn"], "AVO eligibility denominator mismatch")
    return {"cell": str(cell), "reward_frames": len(data["sent"]), "avo_replayed_exact": len(data["sidecars"]),
            "semantics": SEMANTICS, "all_world_ticks_barrier_verified": len(ticks),
            "old_raw_mask_used_for_scoring": False, "policy_performance_qualification": False}


def verify_avo_capture(capture):
    cells = sorted(Path(capture).glob("cells/*/run5_live_avo_gt_summary.json"))
    require(bool(cells), "no live GT cells")
    return {"schema": "splitfusion.live_parity_runtime.avo_verification.v1",
            "status": "LIVE_GT_AVO_INTEGRATION_VERIFIED", "cells": [verify_avo_cell(p.parent) for p in cells],
            "full_offline_v010_equivalence_claim": False, "policy_performance_qualification": False}


def verify_cell(cell):
    from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as S, conflicts
    from . import pose
    cell = Path(cell)
    data = _population(cell)
    target_root = cell / "parity_complete_targets"
    require(not (cell / "run5_live_avo_inputs/complete_targets").exists(), "unexpected nested complete target archive")
    target_manifest = read(target_root / "MANIFEST.json")
    require(target_manifest.get("schema") == "splitfusion.complete_target_preservation.v1", "wrong complete-target manifest")
    descriptors, sources, target_frames, target_dirs = [], [], {}, set()
    for item in target_manifest["frames"]:
        identity = T.validate_identity(item["identity"])
        frame = identity["frame_id"]
        require(frame not in target_frames and identity == data["gt"].get(frame), "duplicate/foreign target identity")
        target_frames[frame] = item
        directory = T.bundle_path(target_root, identity)
        source = directory.with_suffix(".sources")
        target_dirs.update((directory.name, source.name))
        descriptors.append({"identity": identity, "manifest_sha256": item["manifest_sha256"]})
        sources.append({"identity": identity, "directory": str(source), "manifest_sha256": item["source_manifest_sha256"]})
    require(set(target_frames) == set(data["sent"]), "complete target coverage missing sent reward frames")
    require({p.name for p in target_root.iterdir()} == target_dirs | {"MANIFEST.json"}, "unbound complete-target evidence")
    catalogue = cell / "parity_static_catalogue"
    require(catalogue.is_dir() and not catalogue.is_symlink()
            and not any(p.is_symlink() for p in catalogue.rglob("*")), "unsafe static catalogue archive")
    rows, _snapshot, catalogue_evidence = S._load_catalogue(catalogue)
    catalogue_digest = conflicts.catalogue_content_sha256(rows)
    def catalogue_content(value):
        # An archive may be copied to a durable attempt directory. Only its
        # location changes; all source hashes and actual verification results
        # (including map, frame, enabled state and counts) remain mandatory.
        verification = dict(value["verification"])
        require(isinstance(verification.pop("output_dir"), str), "catalogue archive location missing")
        return {"sha256": value["sha256"], "verification": verification}
    for descriptor in sources:
        source = read(Path(descriptor["directory"]) / "SOURCE_EVIDENCE.json")
        require(catalogue_content(source["static_catalogue"]) == catalogue_content(catalogue_evidence),
                "source catalogue archive binding mismatch")
        bundle = T.read_bundle(target_root, expected_identity=descriptor["identity"], timeout_s=.001)
        require(bundle.prepared.target["provenance"]["evidence_sha256"]["catalogue"] == catalogue_digest,
                "target catalogue content binding mismatch")
    edge_dir = cell / "profiler/edge"
    close_result = read(edge_dir / "evaluator_close_result.json")
    require(close_result.get("submitted") == close_result.get("emitted") == close_result.get("records")
            == len(data["expected"]), "actual close/edge report accounting mismatch")
    archive = edge_dir / "prediction_archive"
    verified = R.verify_offline_evidence(target_directory=target_root, target_descriptors=descriptors,
        archive_directory=archive, expected_archive_manifest_sha256=sha(archive / "MANIFEST.json"),
        evaluator_result=close_result, expected_ticket_identities=data["expected"], source_descriptors=sources)
    require(verified["source_archives_complete"] and verified["unscored_predictions"] == 0
            and verified["missing_complete_targets_for_archived_predictions"] == 0,
            "incomplete source/target/prediction evaluation")
    for identity in data["expected"]:
        frame = identity["gt_identity"]["frame_id"]
        evidence = read(archive / (PredictionArchive.key(identity) + ".evaluation.json"))
        row = data["evaluations"][frame]
        require(canonical(evidence["measurement"]) == canonical(row["measurement"]), "archived/emitted measurement mismatch")
        require(row["timing"].get("target_bundle_sha256") == target_frames[frame]["manifest_sha256"],
                "edge actual target read binding mismatch")
        require(canonical(evidence["person_avo"]) == canonical(data["sidecars"][frame]["avo"]),
                "scored AVO differs from independently retained source")
    pose_result = pose.verify_archives(cell / "pose_archive", edge_dir / "pose_archive",
        expected_transmitted=data["ue"]["transmitted_identities"], expected_evaluated=list(data["evaluations"].values()))
    require(pose_result.get("status") == "PASS", "UE/edge/GT pose reconciliation failed")
    for event in pose.read_archive(edge_dir / "pose_archive")["events"]:
        if event["kind"] != "EDGE_TARGET_RECONCILED":
            continue
        frame = event["gt_identity"]["frame_id"]
        require(event["gt_identity"] == data["gt"][frame]
                and event["target_manifest_sha256"] == target_frames[frame]["manifest_sha256"],
                "pose/complete-target actual read identity mismatch")
        require(np.allclose(event["target_camera_matrix"], data["sidecars"][frame]["camera_matrix"],
                            rtol=0, atol=1e-5), "pose/AVO camera matrix mismatch")
    require(not (cell / "parity_rejected_targets").exists(), "rejected complete GT sources")
    return {"cell": str(cell), "passed": True, "evidence": verified, "pose": pose_result,
        "catalogue": catalogue_evidence, "reward_frames": len(data["sent"]),
        "evaluated_reward_frames": len(data["expected"]),
        "sent_reward_frames_without_predictions": sorted(set(data["sent"]) - set(data["evaluations"])),
        "timeliness_is_descriptive": True, "policy_performance_claim": False}


def postflight(output):
    """Call AFTER retained host postflight, using verify_avo_capture above."""
    from rl_agent.splitfusion_run5_live_avo_v1.host_capture import POSTFLIGHT_CHECKS
    output = Path(output).resolve()
    report_path, inputs_path = output / "PARITY_POSTFLIGHT.json", output / "PARITY_POSTFLIGHT_INPUTS.json"
    require(not report_path.exists() and not inputs_path.exists(), "parity postflight is create-only")
    checks, errors, cells = {}, [], []
    try:
        retained = read(output / "RUN5_POSTFLIGHT.json")
        checks["retained_all_checks"] = (retained.get("passed") is True
            and set(retained.get("checks", {})) == set(POSTFLIGHT_CHECKS)
            and all(v is True for v in retained["checks"].values()))
        require(checks["retained_all_checks"], "retained G1-G9/E1-E5/actor/SNR/GT/partition checks failed")
        manifest_path = output / "RUN5_POSTFLIGHT_INPUTS.json"
        require(sha(manifest_path) == retained["input_manifest_sha256"], "retained postflight input manifest changed")
        for name, digest in read(manifest_path)["files"].items():
            path = output / name
            require(path.resolve().is_relative_to(output) and not path.is_symlink() and sha(path) == digest,
                    "retained postflight input changed/escaped")
        cell_paths = sorted((output / "capture").glob("cells/*/run5_live_avo_gt_summary.json"))
        require(bool(cell_paths), "no live repaired cells")
        for path in cell_paths:
            cells.append(verify_cell(path.parent))
        checks["composed_repair_evidence"] = bool(cells) and all(c["passed"] for c in cells)
    except Exception as exc:
        checks["composed_repair_evidence"] = False
        errors.append(f"{type(exc).__name__}: {exc}")
    files = {}
    for path in sorted(output.rglob("*")):
        if path == report_path or path == inputs_path:
            continue
        if path.is_symlink():
            checks["no_symbolic_evidence"] = False
            errors.append("symbolic evidence: " + str(path.relative_to(output)))
        elif path.is_file():
            files[str(path.relative_to(output))] = sha(path)
    checks.setdefault("no_symbolic_evidence", True)
    write(inputs_path, {"schema": SCHEMA + ".inputs", "sha256": files})
    passed = bool(checks) and all(v is True for v in checks.values()) and not errors
    result = {"schema": SCHEMA, "passed": passed,
        "status": "PARITY_RUNTIME_INTEGRATION_VERIFIED" if passed else "STOP_PARITY_RUNTIME_INTEGRATION_FAILED",
        "checks": checks, "errors": errors, "cells": cells, "input_manifest_sha256": sha(inputs_path),
        "policy_performance_claim": False, "training_authorized": False, "timeliness_is_descriptive": True}
    write(report_path, result)
    return result
