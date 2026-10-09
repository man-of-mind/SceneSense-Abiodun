"""CPU-only composed evidence tests, including real target/source archives."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import postflight as P
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E, target_transport as T
from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as S, gt_bridge
from rl_agent.splitfusion_live_parity_repair_v1.prediction_archive import PredictionArchive, canonical
from rl_agent.splitfusion_live_parity_repair_v1.test_prediction_archive import ticket
from rl_agent.splitfusion_live_parity_repair_v1 import test_source_geometry as source_fixtures


class PostflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="runtime-postflight-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cell = self.root / "capture/cells/test_cell"
        self.cell.mkdir(parents=True)
        self.pose_result = {"status": "PASS", "transmitted_frames": 1, "evaluated_frames": 1}
        self.pose_events = []
        self.addCleanup(patch.stopall)
        patch("rl_agent.splitfusion_live_parity_runtime_v1.pose.verify_archives", return_value=self.pose_result).start()
        patch("rl_agent.splitfusion_live_parity_runtime_v1.pose.read_archive",
              side_effect=lambda _: {"events": self.pose_events}).start()
        self.build()

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical(value))

    def change(self, relative, mutate):
        path = self.cell / relative
        value = json.loads(path.read_bytes())
        mutate(value)
        path.write_bytes(canonical(value))

    def build(self):
        value = ticket()
        gt = dict(value.gt_identity, frame_id=100, capture_timestamp_ns=5000000000)
        value = replace(value, envelope=replace(value.envelope, frame_id=100, capture_timestamp_ns=5000000000),
            context=replace(value.context, frame_id=100, capture_timestamp_ns=5000000000), gt_identity=gt,
            records=())
        identity = PredictionArchive.identity(value)
        self.identity = identity
        with source_fixtures.SourceGeometryTests().fixture() as (directory, kwargs):
            population = {"actors": [1, 2]}
            population_sha = hashlib.sha256(canonical(population)).hexdigest()
            producer = S.LiveCompleteTargetProducer(static_catalogue_dir=directory / "catalogue",
                experiment_id="runtime_fixture", ego_id=999, dynamic_population_ids=[1, 2],
                population_evidence_sha256=population_sha, max_frames=1)
            kwargs["identity"] = gt
            kwargs["dynamic_snapshot_evidence"].update(population=population, population_sha256=population_sha)
            inputs = producer.build(**kwargs)
            prepared = E.prepare_target(inputs)
            target_root = self.cell / "parity_complete_targets"
            target = T.write_bundle(target_root, identity=gt, inputs=inputs, prepared=prepared)
            source_dir = T.bundle_path(target_root, gt).with_suffix(".sources")
            kwargs["rgb"]["_rgb_image"] = kwargs["semantic_image"]
            source = gt_bridge.write_source_archive(source_dir, inputs=inputs, evidence=producer.last_evidence,
                rgb=kwargs["rgb"], radar=kwargs["radar_world_xyz"])
            shutil.copytree(directory / "catalogue", self.cell / "parity_static_catalogue")
        self.target = target
        self.put(target_root / "MANIFEST.json", {"schema": "splitfusion.complete_target_preservation.v1",
            "frames": [{"identity": gt, "manifest_sha256": target["manifest_sha256"],
                        "source_manifest_sha256": source["manifest_sha256"]}]})
        avo = {"frame_id": 100, "qualified": [], "structural_ignored": []}
        sidecar = {"identity": gt, "avo": avo, "camera_matrix": np.eye(4).tolist()}
        from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as G
        avo_path = G.sidecar_path(self.cell / "run5_live_avo_inputs", gt)
        self.put(avo_path, sidecar)
        avo_hash = P.sha(avo_path)
        self.put(avo_path.parent / "MANIFEST.json", {"sha256": {avo_path.name: avo_hash}})
        self.put(self.cell / "run5_live_avo_gt_summary.json", {"records": [
            {"frame_id": 100, "path": str(avo_path), "sha256": avo_hash}], "rejected_records": []})
        self.put(self.cell / "run4_phase6/PHASE6_UE_EVIDENCE.json", {
            "transmitted_identities": [{"frame_id": 100, "run4_identity": identity["action_identity"]}],
            "decisions": [{"frame_id": 100, "kind": "POLICY_DECISION", "capture": {"ns": 5000000000}}],
            "faulted": None, "gt_missing_high_outputs": []})
        scored = E.evaluate_prepared(identity=identity["action_identity"], prepared=prepared,
            predicted_labels=value.predicted_mask, predictions=[], avo=avo)
        row = {"frame_id": 100, "run4_identity": identity["action_identity"], "reason": "NONE",
            "enqueued_wall_ns": identity["enqueued_wall_ns"], "measurement_semantics": P.SEMANTICS,
            "measurement": scored.measurement,
            "timing": {"avo_sidecar_sha256": avo_hash, "target_bundle_sha256": target["manifest_sha256"]}}
        self.put(self.cell / "run4_phase6/edge_report.json", {"final": True, "failures": [],
            "run_id": gt["run_id"], "cell_id": gt["cell_id"], "evaluations": [row],
            "counters": {"evaluations_submitted": 1},
            "evaluator": {"submitted": 1, "emitted": 1, "gt_expired": 0, "queue_overflow": 0}})
        archive = PredictionArchive(max_frames=1)
        archive.add(value)
        archive.evaluated(value, scored.evidence)
        manifest = archive.flush(self.cell / "profiler/edge/prediction_archive")
        self.put(self.cell / "profiler/edge/evaluator_close_result.json", {"worker_alive": False,
            "parity_archive_complete": True, "parity_archive_errors": [], "parity_archive": manifest,
            "submitted": 1, "emitted": 1, "records": 1})
        self.pose_events.append({"kind": "EDGE_TARGET_RECONCILED", "gt_identity": gt,
            "target_manifest_sha256": target["manifest_sha256"], "target_camera_matrix": np.eye(4).tolist()})

    def test_complete_real_targets_native_sources_archive_and_catalogue_pass(self):
        result = P.verify_cell(self.cell)
        self.assertTrue(result["passed"])
        self.assertEqual(result["evidence"]["scored_predictions"], 1)
        self.assertTrue(result["evidence"]["source_archives_complete"])
        self.assertFalse(result["policy_performance_claim"])

    def test_expected_population_not_derived_from_archive(self):
        data = P._population(self.cell)
        shutil.rmtree(self.cell / "profiler/edge/prediction_archive")
        self.assertEqual(P._population(self.cell)["expected"], data["expected"])
        with self.assertRaises((ValueError, OSError)):
            P.verify_cell(self.cell)

    def test_foreign_edge_session_refused(self):
        self.change("run4_phase6/edge_report.json", lambda d: d["evaluations"][0]["run4_identity"].update(session_uuid="foreign"))
        with self.assertRaisesRegex(ValueError, "action identity"):
            P.verify_cell(self.cell)

    def test_missing_evaluation_cannot_shrink_expected_population_silently(self):
        self.change("run4_phase6/edge_report.json", lambda d: d.update(evaluations=[]))
        with self.assertRaisesRegex(ValueError, "empty edge"):
            P.verify_cell(self.cell)

    def test_partial_sent_population_missing_evaluation_blocks_qualification(self):
        from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as G
        identity = dict(self.identity["gt_identity"], frame_id=101)
        path = G.sidecar_path(self.cell / "run5_live_avo_inputs", identity)
        self.put(path, {"identity": identity})
        self.change("run5_live_avo_inputs/MANIFEST.json", lambda d: d["sha256"].update({path.name: P.sha(path)}))
        self.change("run5_live_avo_gt_summary.json", lambda d: d["records"].append({"frame_id": 101,
            "path": str(path), "sha256": P.sha(path)}))
        action = dict(self.identity["action_identity"], frame_id=101, decision_seq=1, ticket_seq=1, tensor_seq=1)
        def more(ue):
            ue["transmitted_identities"].append({"frame_id": 101, "run4_identity": action})
            ue["decisions"].append({"frame_id": 101, "kind": "POLICY_DECISION", "capture": {"ns": 5000000000}})
        self.change("run4_phase6/PHASE6_UE_EVIDENCE.json", more)
        with self.assertRaisesRegex(ValueError, r"without independent edge evaluation: \[101\]"):
            P.verify_cell(self.cell)

    def test_submitted_emitted_accounting_checked(self):
        self.change("run4_phase6/edge_report.json", lambda d: d["evaluator"].update(submitted=2))
        with self.assertRaisesRegex(ValueError, "population mismatch"):
            P.verify_cell(self.cell)

    def test_gt_expiry_and_old_raw_semantic_contract_refused(self):
        self.change("run4_phase6/edge_report.json", lambda d: d["evaluator"].update(gt_expired=1))
        with self.assertRaisesRegex(ValueError, "GT expiry"):
            P.verify_cell(self.cell)
        self.change("run4_phase6/edge_report.json", lambda d: d["evaluator"].update(gt_expired=0))
        self.change("run4_phase6/edge_report.json", lambda d: d["evaluations"][0].update(measurement_semantics="old"))
        with self.assertRaisesRegex(ValueError, "foreign evaluator"):
            P.verify_cell(self.cell)

    def test_real_close_result_mandatory_not_synthesized_from_archive(self):
        (self.cell / "profiler/edge/evaluator_close_result.json").unlink()
        with self.assertRaisesRegex(ValueError, "evaluator_close_result"):
            P.verify_cell(self.cell)

    def test_archival_errors_fail(self):
        self.change("profiler/edge/evaluator_close_result.json", lambda d: d.update(parity_archive_errors=["injected"]))
        with self.assertRaisesRegex(ValueError, "archive reports errors"):
            P.verify_cell(self.cell)

    def test_catalogue_tamper_fails(self):
        path = next((self.cell / "parity_static_catalogue").glob("*.csv"))
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaises(Exception):
            P.verify_cell(self.cell)

    def test_target_native_tamper_fails(self):
        path = T.bundle_path(self.cell / "parity_complete_targets", self.identity["gt_identity"]) / "raw_semantic.npy"
        body = bytearray(path.read_bytes()); body[-1] ^= 1; path.write_bytes(body)
        with self.assertRaises(ValueError):
            P.verify_cell(self.cell)

    def test_pose_target_actual_hash_join(self):
        self.pose_events[0]["target_manifest_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "actual read identity"):
            P.verify_cell(self.cell)

    def test_pose_camera_join(self):
        self.pose_events[0]["target_camera_matrix"][0][3] = .3
        with self.assertRaisesRegex(ValueError, "camera matrix mismatch"):
            P.verify_cell(self.cell)

    def test_emitted_measurement_must_equal_evaluated_owned_snapshot(self):
        self.change("run4_phase6/edge_report.json", lambda d: d["evaluations"][0]["measurement"].update(loc_vehicle_tp=999))
        with self.assertRaisesRegex(ValueError, "emitted measurement"):
            P.verify_cell(self.cell)

    def test_rejected_target_never_hidden(self):
        (self.cell / "parity_rejected_targets").mkdir()
        with self.assertRaisesRegex(ValueError, "rejected complete"):
            P.verify_cell(self.cell)

    def test_retained_failed_gate_stays_failed_create_only(self):
        from rl_agent.splitfusion_run5_live_avo_v1.host_capture import POSTFLIGHT_CHECKS
        self.put(self.root / "RUN5_POSTFLIGHT.json", {"passed": True,
            "checks": dict.fromkeys(POSTFLIGHT_CHECKS, True) | {"causal_snr": False}})
        result = P.postflight(self.root)
        self.assertFalse(result["passed"])
        self.assertIn("retained", result["errors"][0])
        with self.assertRaisesRegex(ValueError, "create-only"):
            P.postflight(self.root)

    def test_full_composed_postflight_retains_checks_hashes_and_no_performance_claim(self):
        from rl_agent.splitfusion_run5_live_avo_v1.host_capture import POSTFLIGHT_CHECKS
        files = {str(p.relative_to(self.root)): P.sha(p) for p in self.root.rglob("*") if p.is_file()}
        manifest = self.root / "RUN5_POSTFLIGHT_INPUTS.json"
        self.put(manifest, {"files": files})
        self.put(self.root / "RUN5_POSTFLIGHT.json", {"passed": True,
            "checks": dict.fromkeys(POSTFLIGHT_CHECKS, True), "input_manifest_sha256": P.sha(manifest)})
        result = P.postflight(self.root)
        self.assertTrue(result["passed"], result)
        self.assertFalse(result["policy_performance_claim"])
        self.assertFalse(result["training_authorized"])
        self.assertTrue(result["timeliness_is_descriptive"])
        for name, digest in P.read(self.root / "PARITY_POSTFLIGHT_INPUTS.json")["sha256"].items():
            self.assertEqual(P.sha(self.root / name), digest)


class AvoPostflightTests(unittest.TestCase):
    def fixture(self, root):
        from rl_agent.splitfusion_run5_live_avo_v1 import test_verify_live_gt_capture as F
        cell = F.capture(root)
        ue_path, edge_path = cell / "run4_phase6/PHASE6_UE_EVIDENCE.json", cell / "run4_phase6/edge_report.json"
        ue, edge = P.read(ue_path), P.read(edge_path)
        action = dict(PredictionArchive.identity(ticket())["action_identity"], frame_id=10, anchor_action_id=71)
        ue["transmitted_identities"][0]["run4_identity"] = action
        ue.update(faulted=None, gt_missing_high_outputs=[])
        edge["evaluations"][0].update(run4_identity=action, measurement_semantics=P.SEMANTICS,
            enqueued_wall_ns=100, timing={"avo_sidecar_sha256": edge["evaluations"][0]["measurement"]["avo_sidecar_sha256"]})
        edge["evaluator"]["submitted"] = 1
        edge["counters"] = {"evaluations_submitted": 1}
        ue_path.write_bytes(canonical(ue)); edge_path.write_bytes(canonical(edge))
        return cell

    def test_new_semantics_with_original_exact_avo_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            result = P.verify_avo_capture(root)
            self.assertEqual(result["status"], "LIVE_GT_AVO_INTEGRATION_VERIFIED")
            self.assertEqual(result["cells"][0]["avo_replayed_exact"], 1)
            self.assertFalse(result["cells"][0]["old_raw_mask_used_for_scoring"])

    def test_missing_raw_source_not_hidden_by_prepared_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            cell = self.fixture(Path(tmp))
            next((cell / "run5_live_avo_inputs").glob("*.depth_bgra.npy")).unlink()
            with self.assertRaises(ValueError):
                P.verify_avo_cell(cell)

    def test_resealed_avo_result_still_numerically_replayed(self):
        from rl_agent.splitfusion_run5_live_avo_v1.test_verify_live_gt_capture import VerificationTests
        with tempfile.TemporaryDirectory() as tmp:
            cell = self.fixture(Path(tmp))
            sidecar_path = next((cell / "run5_live_avo_inputs").glob("*.avo.json"))
            sidecar = P.read(sidecar_path)
            sidecar["avo"]["geometry_relevance"] = {"invented": 1}
            VerificationTests.reseal_sidecar(cell, sidecar_path, sidecar)
            edge_path = cell / "run4_phase6/edge_report.json"
            edge = P.read(edge_path)
            edge["evaluations"][0]["timing"]["avo_sidecar_sha256"] = P.sha(sidecar_path)
            edge_path.write_bytes(canonical(edge))
            with self.assertRaisesRegex(ValueError, "numerical replay"):
                P.verify_avo_cell(cell)

    def test_manifested_rejection_cannot_hide_behind_clean_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            cell = self.fixture(Path(tmp))
            inputs = cell / "run5_live_avo_inputs"
            rejected = inputs / "hidden.avo_rejected.json"
            rejected.write_bytes(canonical({"error": "injected prescore failure"}))
            manifest = P.read(inputs / "MANIFEST.json")
            manifest["sha256"][rejected.name] = P.sha(rejected)
            (inputs / "MANIFEST.json").write_bytes(canonical(manifest))
            self.assertEqual(P.read(cell / "run5_live_avo_gt_summary.json")["rejected_records"], [])
            with self.assertRaisesRegex(ValueError, "rejected AVO scoring"):
                P.verify_avo_cell(cell)

    def test_relabelled_source_filename_rejected_even_with_updated_hash_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            cell = self.fixture(Path(tmp))
            inputs = cell / "run5_live_avo_inputs"
            path = next(inputs.glob("*.avo.json"))
            foreign = inputs / "foreign.avo.json"
            path.rename(foreign)
            manifest = P.read(inputs / "MANIFEST.json")
            manifest["sha256"][foreign.name] = manifest["sha256"].pop(path.name)
            (inputs / "MANIFEST.json").write_bytes(canonical(manifest))
            summary_path = cell / "run5_live_avo_gt_summary.json"
            summary = P.read(summary_path)
            summary["records"][0]["path"] = str(foreign)
            summary_path.write_bytes(canonical(summary))
            with self.assertRaisesRegex(ValueError, "filename/identity mismatch"):
                P.verify_avo_cell(cell)


if __name__ == "__main__":
    unittest.main()
