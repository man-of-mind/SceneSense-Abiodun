"""Synthetic exact-depth captures only; no live qualification claim."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import live_gt as G, verify_live_gt_capture as V
from .test_live_gt import arguments, identity, Transform, Actors


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def capture(root, args=None):
    from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as GE
    from .test_live_gt import AllTickBarrierTests
    cell = root / "cells" / "cell"
    scratch = root / "scratch"
    args = arguments() if args is None else args
    helper = AllTickBarrierTests()
    collector = helper.collector(attempt_dir=cell)
    helper.callbacks(collector)
    collector.wait_for_sensor_frame(10, snapshot=helper.snapshot())
    recorder = G.LiveSyncEvidence(cell / "run5_live_sync_evidence")
    record = G.write_avo_bundle(scratch, **args, sync_evidence=recorder)
    GE.write_object_ground_truth(scratch, identity=identity(), frozen_carla_frame_id=10, rows=[])
    GE.write_semantic_ground_truth(scratch, identity=identity(), frozen_carla_frame_id=10,
                                   mask=G.corrected_semantic_image(args["semantic_image"]))
    preservation = G.preserve_avo_evidence([record], cell / "run5_live_avo_inputs")
    sidecar = json.loads(Path(record["path"]).read_text())
    write_json(cell / "run5_live_avo_gt_summary.json",
               dict(schema=G.SCHEMA, measurement_semantics=G.MEASUREMENT_SEMANTICS,
                    records=[record], new_gt_helper_threads=0, preservation=preservation,
                    rejected_records=[],
                    source_synchronization=recorder.summary(),
                    all_tick_sensor_barrier=dict(outside_action_open=True,
                        records=collector._avo_tick_barrier_records,
                        rejections=collector._avo_barrier_rejections.summary())))
    sent = dict(frame_id=10, reward_requested=True, session_uuid="synthetic", mode_id=9, q_e4=1234)
    write_json(cell / "run4_phase6" / "PHASE6_UE_EVIDENCE.json",
               dict(transmitted_identities=[dict(frame_id=10, run4_identity=sent)],
                    decisions=[dict(frame_id=10, kind="POLICY_DECISION", carla_timestamp=1.,
                                    capture=dict(ns=20), action_open=dict(ns=sidecar["start_raw_ns"] - 1000),
                                    stages=dict(last_packet_send_raw_ns=sidecar["start_raw_ns"] - 1))]))
    measurement = dict(frame_id=10, avo_sidecar_sha256=record["sha256"],
                       live_gt_contract=G.MEASUREMENT_SEMANTICS,
                       loc_person_eligible_gt=record["eligible"], loc_person_tp=0,
                       loc_person_fn=record["eligible"])
    write_json(cell / "run4_phase6" / "edge_report.json",
               dict(final=True, failures=[], run_id="r", cell_id="c",
                    evaluator=dict(gt_expired=0, queue_overflow=0, emitted=1),
                    evaluations=[dict(frame_id=10, run4_identity=sent, measurement=measurement,
                                      measurement_semantics=G.MEASUREMENT_SEMANTICS, reason="NONE")]))
    return cell


class VerificationTests(unittest.TestCase):
    @staticmethod
    def reseal_sidecar(cell, sidecar_path, sidecar):
        """Test-only adversarial edit, preserving ordinary hash checks."""
        write_json(sidecar_path, sidecar)
        digest = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
        inputs = cell / "run5_live_avo_inputs"
        manifest_path = inputs / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["sha256"][sidecar_path.name] = digest
        write_json(manifest_path, manifest)
        summary_path = cell / "run5_live_avo_gt_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["records"][0]["sha256"] = digest
        write_json(summary_path, summary)

    def test_exact_synthetic_capture_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture(root)
            result = V.verify_capture(root)
            self.assertEqual(result["status"], "LIVE_GT_AVO_INTEGRATION_VERIFIED")
            self.assertFalse(result["policy_performance_qualification"])
            self.assertEqual(result["cells"][0]["avo_replayed_exact"], 1)
            self.assertEqual(result["cells"][0]["prescore_input_bundles_verified"], 1)
            self.assertEqual(result["cells"][0]["geometry_relevance_proofs_replayed"], 0)

    def test_safe_noncompeting_tilted_actor_proof_replays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = arguments()
            actors = args["frozen_world"].get_actors()
            transform = Transform(-200.)
            transform.rotation.pitch = 15.
            actors.append(SimpleNamespace(id=103, type_id="walker.pedestrian.remote",
                bounding_box=actors[0].bounding_box, get_transform=lambda: transform))
            args["frozen_world"].get_actors = lambda: Actors(actors)
            cell = capture(root, args)
            result = V.verify_capture(root)
            self.assertEqual(result["cells"][0]["geometry_relevance_proofs_replayed"], 1)
            sidecar = json.loads(next((cell / "run5_live_avo_inputs").glob("*.avo.json")).read_text())
            self.assertEqual(sidecar["avo"]["geometry_relevance"]["omitted_tilted_structural_actors"][0]["gt_actor_id"], "103")

    def test_relevant_tilt_failure_preserves_exact_inputs_and_offender(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = arguments()
            args["frozen_world"].get_actors()[0].get_transform().rotation.pitch = 10.
            with self.assertRaises(G.LiveGtContractError) as caught:
                G.write_avo_bundle(root / "scratch", **args)
            record = caught.exception.avo_evidence_record
            G.preserve_avo_evidence([], root / "retained", rejected_records=[record])
            rejected = json.loads((root / "retained" / Path(record["path"]).name).read_text())
            prescore_path = root / "retained" / rejected["prescore_input"]["filename"]
            self.assertEqual(hashlib.sha256(prescore_path.read_bytes()).hexdigest(),
                             rejected["prescore_input"]["sha256"])
            prescore = json.loads(prescore_path.read_text())
            offender = rejected["geometry_diagnostics"]["offender"]
            self.assertEqual((offender["gt_actor_id"], offender["pitch_deg"], offender["roll_deg"]), ("42", 10., 0.))
            self.assertEqual(prescore["raw_pedestrian_rows"][0]["object_pitch_deg"], 10.)
            self.assertLessEqual(prescore["write_raw_ns"], rejected["prescore_input_ready_raw_ns"])
            self.assertLessEqual(rejected["prescore_input_ready_raw_ns"], rejected["scoring_start_raw_ns"])
            self.assertLessEqual(rejected["scoring_start_raw_ns"], rejected["rejected_raw_ns"])
            self.assertFalse(list((root / "retained").glob("*.avo.json")))
            manifest = json.loads((root / "retained" / "MANIFEST.json").read_text())
            for name, digest in manifest["sha256"].items():
                self.assertEqual(hashlib.sha256((root / "retained" / name).read_bytes()).hexdigest(), digest)

    def test_old_capture_without_prescore_proof_cannot_be_relabelled_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = capture(root)
            path = next((cell / "run5_live_avo_inputs").glob("*.avo.json"))
            sidecar = json.loads(path.read_text())
            sidecar.pop("prescore_input")
            self.reseal_sidecar(cell, path, sidecar)
            with self.assertRaisesRegex(V.VerificationError, "raw input binding missing"):
                V.verify_capture(root)

    def test_missing_prescore_file_and_rejected_attempt_refused(self):
        for change in ("missing", "rejected", "not_audited"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cell = capture(root)
                if change == "missing":
                    inputs = cell / "run5_live_avo_inputs"
                    sidecar = json.loads(next(inputs.glob("*.avo.json")).read_text())
                    (inputs / sidecar["prescore_input"]["filename"]).unlink()
                else:
                    path = cell / "run5_live_avo_gt_summary.json"
                    summary = json.loads(path.read_text())
                    if change == "rejected":
                        summary["rejected_records"] = [{"frame_id": 11, "error": "relevant tilted actor"}]
                    else:
                        summary.pop("rejected_records")
                    write_json(path, summary)
                with self.assertRaises(V.VerificationError):
                    V.verify_capture(root)

    def test_resealed_prescore_input_disagreement_refused(self):
        for change in ("rows", "payload", "identity", "clock"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cell = capture(root)
                inputs = cell / "run5_live_avo_inputs"
                sidecar_path = next(inputs.glob("*.avo.json"))
                sidecar = json.loads(sidecar_path.read_text())
                path = inputs / sidecar["prescore_input"]["filename"]
                prescore = json.loads(path.read_text())
                if change == "rows":
                    prescore["raw_pedestrian_rows"][0]["object_pitch_deg"] = 1.
                elif change == "payload":
                    prescore["payloads"]["depth_bgra"]["sha256"] = "f" * 64
                elif change == "identity":
                    prescore["identity"]["frame_id"] = 11
                else:
                    prescore["write_raw_ns"] = sidecar["scoring_start_raw_ns"] + 1
                write_json(path, prescore)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                manifest_path = inputs / "MANIFEST.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["sha256"][path.name] = digest
                write_json(manifest_path, manifest)
                sidecar["prescore_input"]["sha256"] = digest
                self.reseal_sidecar(cell, sidecar_path, sidecar)
                with self.assertRaisesRegex(V.VerificationError, "pre-scoring/final|before scoring"):
                    V.verify_capture(root)

    def test_resealed_geometry_proof_drift_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = capture(root)
            path = next((cell / "run5_live_avo_inputs").glob("*.avo.json"))
            sidecar = json.loads(path.read_text())
            sidecar["avo"]["geometry_relevance"] = {"omitted_tilted_structural_actors": [
                {"gt_actor_id": "invented", "object_pitch_deg": 3., "object_roll_deg": 2.}]}
            self.reseal_sidecar(cell, path, sidecar)
            with self.assertRaisesRegex(V.VerificationError, "AVO replay mismatch"):
                V.verify_capture(root)

    def test_empty_and_missing_raw_input_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(V.VerificationError):
                V.verify_capture(root)
            cell = capture(root)
            next((cell / "run5_live_avo_inputs").glob("*.depth_bgra.npy")).unlink()
            with self.assertRaises(V.VerificationError):
                V.verify_capture(root)

    def test_manifest_corruption_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = capture(root)
            path = next((cell / "run5_live_avo_inputs").glob("*.avo.json"))
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(V.VerificationError, "digest mismatch"):
                V.verify_capture(root)

    def test_gt_expiry_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = capture(root)
            path = cell / "run4_phase6" / "edge_report.json"
            edge = json.loads(path.read_text())
            edge["evaluator"]["gt_expired"] = 1
            write_json(path, edge)
            with self.assertRaisesRegex(V.VerificationError, "gt_expired"):
                V.verify_capture(root)

    def test_wrong_evaluator_link_semantics_and_identity_refused(self):
        for change in ("digest", "semantics", "identity", "exception"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cell = capture(root)
                path = cell / "run4_phase6" / "edge_report.json"
                edge = json.loads(path.read_text())
                row = edge["evaluations"][0]
                if change == "digest":
                    row["measurement"]["avo_sidecar_sha256"] = "b" * 64
                elif change == "semantics":
                    row["measurement_semantics"] = "OLD_UNCORRECTED_GT"
                elif change == "identity":
                    row["run4_identity"]["session_uuid"] = "foreign"
                else:
                    row["reason"] = "EVALUATOR_EXCEPTION"
                write_json(path, edge)
                with self.assertRaises(V.VerificationError):
                    V.verify_capture(root)

    def test_resealed_wrong_avo_fails_exact_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = capture(root)
            inputs = cell / "run5_live_avo_inputs"
            path = next(inputs.glob("*.avo.json"))
            sidecar = json.loads(path.read_text())
            sidecar["avo"]["qualified"][0]["avo"] = .1
            write_json(path, sidecar)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = json.loads((inputs / "MANIFEST.json").read_text())
            manifest["sha256"][path.name] = digest
            write_json(inputs / "MANIFEST.json", manifest)
            summary_path = cell / "run5_live_avo_gt_summary.json"
            summary = json.loads(summary_path.read_text())
            summary["records"][0]["sha256"] = digest
            write_json(summary_path, summary)
            with self.assertRaisesRegex(V.VerificationError, "AVO replay mismatch"):
                V.verify_capture(root)

    def test_gt_started_before_last_send_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = capture(root)
            path = cell / "run4_phase6" / "PHASE6_UE_EVIDENCE.json"
            ue = json.loads(path.read_text())
            ue["decisions"][0]["stages"]["last_packet_send_raw_ns"] += 100000000000
            write_json(path, ue)
            with self.assertRaisesRegex(V.VerificationError, "before final datagram"):
                V.verify_capture(root)

    def test_missing_or_rejected_sync_evidence_refused(self):
        for change in ("missing", "rejected", "uncovered", "missing_observed"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cell = capture(root)
                path = cell / "run5_live_avo_gt_summary.json"
                summary = json.loads(path.read_text())
                if change == "missing":
                    summary.pop("source_synchronization")
                elif change == "rejected":
                    summary["source_synchronization"]["rejected"] = 1
                elif change == "uncovered":
                    summary["source_synchronization"]["records"] = []
                else:
                    next((cell / "run5_live_sync_evidence").glob("*.observed.json")).unlink()
                write_json(path, summary)
                with self.assertRaises(V.VerificationError):
                    V.verify_capture(root)

    def test_barrier_drift_or_inside_deadline_refused(self):
        for change in ("world", "camera", "callback", "clock", "reject"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cell = capture(root)
                path = cell / "run5_live_avo_gt_summary.json"
                summary = json.loads(path.read_text())
                barrier = summary["all_tick_sensor_barrier"]
                row = barrier["records"][0]
                if change == "world":
                    row["sources"]["frozen_world"]["timestamp"] = 1.05
                elif change == "camera":
                    row["owned_camera_matrix"][0][3] = 1.
                elif change == "callback":
                    row["completed_roles"].remove("radar")
                elif change == "clock":
                    row["end_raw_ns"] += 100000000000
                else:
                    barrier["rejections"]["rejected"] = 1
                write_json(path, summary)
                with self.assertRaises(V.VerificationError):
                    V.verify_capture(root)


if __name__ == "__main__":
    unittest.main()
