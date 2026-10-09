"""Saved-evidence checks only: no CARLA client, CUDA, actors or live services."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from . import evaluation as E, readiness as R, target_transport as T
from .prediction_archive import PredictionArchive, canonical
from .test_prediction_archive import ticket
from .test_target_transport import fixture
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="parity_offline_readiness_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target_root, self.archive_root = self.root / "targets", self.root / "predictions"
        self.ticket = replace(ticket(), records=())
        self.identity = PredictionArchive.identity(self.ticket)
        self.gt = self.identity["gt_identity"]
        _, inputs, prepared = fixture()
        frame = dict(inputs.frame, frame_id=self.gt["frame_id"], sample_id="prospective_test_frame1001")
        provenance = dict(inputs.provenance, sample_id=frame["sample_id"], capture_identity=self.gt)
        inputs = replace(inputs, frame=frame, provenance=provenance)
        inputs = replace(inputs, provenance=dict(provenance, input_content_sha256=A.input_content_sha256(inputs)))
        target = dict(prepared.target, provenance=inputs.provenance, tables={
            "object_boxes.csv": [], "object_ignore_regions.csv": [],
            "target_manifest.csv": [{"sample_id": frame["sample_id"]}]})
        prepared = E.restore_prepared_target(frame_id=self.gt["frame_id"], sample_id=frame["sample_id"],
            target=target, content_sha256=E._target_digest(self.gt["frame_id"], frame["sample_id"], target))
        self.inputs, self.prepared = inputs, prepared

    def build(self, *, scored=True, target=True):
        descriptors = []
        if target:
            result = T.write_bundle(self.target_root, identity=self.gt, inputs=self.inputs, prepared=self.prepared)
            descriptors.append({"identity": self.gt, "manifest_sha256": result["manifest_sha256"]})
        archive = PredictionArchive(max_frames=1)
        self.key = archive.add(self.ticket)
        if scored:
            result = E.evaluate_prepared(identity=self.identity["action_identity"], prepared=self.prepared,
                predicted_labels=self.ticket.predicted_mask, predictions=[],
                avo={"frame_id": self.gt["frame_id"], "qualified": [], "structural_ignored": []})
            archive.evaluated(self.ticket, result.evidence)
        manifest = archive.flush(self.archive_root)
        self.args = dict(target_directory=self.target_root, target_descriptors=descriptors,
            archive_directory=self.archive_root, expected_archive_manifest_sha256=self.manifest_digest(),
            evaluator_result={"worker_alive": False, "parity_archive_complete": True,
                "parity_archive_errors": [], "parity_archive": manifest},
            expected_ticket_identities=[self.identity])
        return self.args

    def manifest_digest(self):
        return hashlib.sha256((self.archive_root / "MANIFEST.json").read_bytes()).hexdigest()

    def reseal(self, filename=None):
        manifest = json.loads((self.archive_root / "MANIFEST.json").read_text())
        if filename:
            manifest["sha256"][filename] = hashlib.sha256((self.archive_root / filename).read_bytes()).hexdigest()
        (self.archive_root / "MANIFEST.json").write_bytes(canonical(manifest))
        self.args["evaluator_result"]["parity_archive"] = manifest
        self.args["expected_archive_manifest_sha256"] = self.manifest_digest()

    def mutate_json(self, suffix, change):
        name = f"{self.key}.{suffix}"
        path = self.archive_root / name
        value = json.loads(path.read_text())
        change(value)
        path.write_bytes(canonical(value))
        self.reseal(name)

    def with_sources(self):
        from . import source_geometry, gt_bridge
        from rl_agent.splitfusion_run5_live_avo_v1.gt_source_sync import sensor_metadata, source_alignment_record
        height, width = self.inputs.raw_semantic.shape
        image = SimpleNamespace(frame=self.gt["frame_id"], timestamp=10., width=width, height=height,
            raw_data=np.zeros((height, width, 4), np.uint8).tobytes(),
            transform=SimpleNamespace(get_matrix=lambda: np.eye(4).tolist()))
        rgb = sensor_metadata(image)
        rgb["_rgb_image"] = image
        rgb["radar_callback"] = dict(sensor_metadata(image), width=None, height=None)
        frozen = SimpleNamespace(frozen_frame_id=self.gt["frame_id"], frozen_timestamp_s=10.,
                                 frozen_observed_raw_ns=1)
        sync = source_alignment_record(identity=self.gt, rgb=rgb, semantic_image=image,
            depth_image=image, frozen_world=frozen, policy_timestamp_s=10.)
        population = {"schema": "synthetic_not_live", "registered_actors": []}
        population_sha = hashlib.sha256(canonical(population)).hexdigest()
        evidence = {"identity": self.gt, "source_synchronization": sync,
            "sources": source_geometry.verify_sources(), "population_evidence_sha256": population_sha,
            "dynamic_snapshot_evidence": {"frame_id": self.gt["frame_id"], "population": population,
                                          "population_sha256": population_sha}, "conflicts": {}}
        provenance = dict(self.inputs.provenance, evidence_sha256={
            "owned_source": hashlib.sha256(canonical(evidence)).hexdigest(),
            "population": population_sha, "conflicts": hashlib.sha256(canonical({})).hexdigest(),
            "catalogue": "a" * 64})
        self.inputs = replace(self.inputs, provenance=provenance)
        provenance = dict(provenance, input_content_sha256=A.input_content_sha256(self.inputs))
        self.inputs = replace(self.inputs, provenance=provenance)
        target = dict(self.prepared.target, provenance=provenance)
        self.prepared = E.restore_prepared_target(frame_id=self.prepared.frame_id,
            sample_id=self.prepared.sample_id, target=target,
            content_sha256=E._target_digest(self.prepared.frame_id, self.prepared.sample_id, target))
        result = gt_bridge.write_source_archive(self.root / "sources", inputs=self.inputs, evidence=evidence,
            rgb=rgb, radar=np.zeros((0, 3), np.float32))
        self.source_descriptor = dict(identity=self.gt, **result)
        args = self.build()
        args["source_descriptors"] = [self.source_descriptor]
        return args

    def test_actual_archive_and_complete_target_verified_never_live_qualified(self):
        args = self.build()
        before = {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.root.rglob("*") if p.is_file()}
        result = R.verify_offline_evidence(**args)
        self.assertEqual(result["scored_predictions"], 1)
        self.assertEqual(result["verified_complete_targets"], 1)
        self.assertEqual(result["unscored_predictions"], 0)
        self.assertEqual(result["missing_source_archives"], 1)
        self.assertFalse(result["source_archives_complete"])
        self.assertEqual(result["status"], "OFFLINE_EVIDENCE_VERIFIED__LIVE_NOT_QUALIFIED")
        for key in ("live_qualified", "cuda_qualified", "training_authorized",
                    "inherited_latency_model_approved", "raw_gt_fallback_allowed"):
            self.assertIs(result[key], False)
        after = {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_unavailable_target_prediction_retained_with_explicit_counts(self):
        result = R.verify_offline_evidence(**self.build(scored=False, target=False))
        self.assertEqual(result["unscored_predictions"], 1)
        self.assertEqual(result["missing_complete_targets_for_archived_predictions"], 1)
        self.assertEqual(result["unscored_tickets"][0]["identity"], self.identity)
        self.assertEqual(result["scored_predictions"], 0)

    def test_unscored_with_target_does_not_invent_missing_gt_reason(self):
        result = R.verify_offline_evidence(**self.build(scored=False))
        self.assertEqual(result["missing_complete_targets_for_archived_predictions"], 0)
        self.assertTrue(result["unscored_tickets"][0]["complete_target_present"])
        self.assertEqual(result["unscored_tickets"][0]["status"], "NOT_SCORED_REASON_NOT_IN_ARCHIVE")

    def test_scored_without_complete_target_refused_no_old_raw_gt_substitute(self):
        with self.assertRaisesRegex(R.ReadinessError, "fallback forbidden"):
            R.verify_offline_evidence(**self.build(target=False))

    def test_worker_archive_status_and_errors_are_mandatory(self):
        args = self.build()
        for replacement in ({"worker_alive": True}, {"parity_archive_complete": False},
                            {"parity_archive_errors": ["injected"]}, {"parity_archive_errors": 0}):
            with self.subTest(replacement=replacement), self.assertRaises(R.ReadinessError):
                R.verify_offline_evidence(**dict(args, evaluator_result=dict(args["evaluator_result"], **replacement)))

    def test_native_source_tamper_refused_even_when_prepared_target_untouched(self):
        args = self.build()
        path = T.bundle_path(self.target_root, self.gt) / "raw_semantic.npy"
        payload = bytearray(path.read_bytes()); payload[-1] ^= 1
        path.write_bytes(payload)
        with self.assertRaisesRegex(T.TargetTransportError, "hash/size"):
            R.verify_offline_evidence(**args)

    def test_archive_missing_hash_tamper_and_unbound_extra_refused(self):
        args = self.build()
        path = self.archive_root / f"{self.key}.records.json"
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaisesRegex(R.ReadinessError, "hash mismatch"):
            R.verify_offline_evidence(**args)
        path.unlink()
        with self.assertRaisesRegex(R.ReadinessError, "missing archive"):
            R.verify_offline_evidence(**args)
        path.write_bytes(original)
        (self.archive_root / "unbound.json").write_text("{}")
        with self.assertRaisesRegex(R.ReadinessError, "unbound"):
            R.verify_offline_evidence(**args)

    def test_duplicate_ticket_or_frame_population_refused(self):
        args = self.build()
        for population in ([], [self.identity, self.identity], [self.identity,
            dict(self.identity, action_identity=dict(self.identity["action_identity"], ticket_seq=1))]):
            with self.subTest(population=population), self.assertRaises(R.ReadinessError):
                R.verify_offline_evidence(**dict(args, expected_ticket_identities=population))
        with self.assertRaisesRegex(R.ReadinessError, "duplicate complete"):
            R.verify_offline_evidence(**dict(args, target_descriptors=args["target_descriptors"] * 2))

    def test_full_action_identity_foreign_session_refused_after_rehash(self):
        args = self.build()
        self.mutate_json("identity.json", lambda value: value["action_identity"].update(session_uuid="foreign"))
        with self.assertRaisesRegex(R.ReadinessError, "foreign action/session/frame"):
            R.verify_offline_evidence(**args)

    def test_evaluation_target_and_prediction_identity_bound_after_rehash(self):
        args = self.build()
        name = f"{self.key}.evaluation.json"
        original = (self.archive_root / name).read_bytes()
        changes = [lambda d: d.update(target_content_sha256="a"*64),
                   lambda d: d["identity"].update(ticket_seq=22),
                   lambda d: d["predicted_labels"].update(array_sha256="b"*64),
                   lambda d: d.update(prediction_records=[{"madeup": True}]),
                   lambda d: d.update(source_pins={})]
        for change in changes:
            (self.archive_root / name).write_bytes(original); self.reseal(name)
            self.mutate_json("evaluation.json", change)
            with self.subTest(change=change), self.assertRaises(R.ReadinessError):
                R.verify_offline_evidence(**args)

    def test_archive_manifest_must_match_actual_evaluator_result(self):
        args = self.build()
        args["evaluator_result"]["parity_archive"] = dict(args["evaluator_result"]["parity_archive"], evaluated_frames=0)
        with self.assertRaisesRegex(R.ReadinessError, "differs from evaluator"):
            R.verify_offline_evidence(**args)

    def test_archive_symlink_and_path_traversal_refused(self):
        args = self.build()
        path = self.archive_root / f"{self.key}.records.json"
        external = self.root / "outside.json"
        path.rename(external); path.symlink_to(external)
        with self.assertRaisesRegex(R.ReadinessError, "symlink"):
            R.verify_offline_evidence(**args)
        path.unlink(); external.rename(path)
        manifest = json.loads((self.archive_root / "MANIFEST.json").read_text())
        digest = manifest["sha256"].pop(path.name)
        manifest["sha256"]["../outside.json"] = digest
        (self.archive_root / "MANIFEST.json").write_bytes(canonical(manifest))
        self.reseal()
        with self.assertRaises(R.ReadinessError):
            R.verify_offline_evidence(**args)

    def test_source_archives_native_arrays_and_provenance_join(self):
        result = R.verify_offline_evidence(**self.with_sources())
        self.assertTrue(result["source_archives_complete"])
        self.assertFalse(result["live_qualified"])
        self.assertEqual(result["missing_source_archives"], 0)
        self.assertEqual(result["verified_source_archives"][0]["rgb_shape"], [4, 6, 4])

    def test_source_archive_native_tamper_or_foreign_identity_refused(self):
        args = self.with_sources()
        source = self.root / "sources"
        path = source / "rgb_bgra.npy"
        original = path.read_bytes()
        changed = bytearray(original); changed[-1] ^= 1
        path.write_bytes(changed)
        with self.assertRaisesRegex(R.ReadinessError, "hash mismatch"):
            R.verify_offline_evidence(**args)
        path.write_bytes(original)
        value = json.loads((source / "MANIFEST.json").read_text())
        value["identity"]["frame_id"] += 1
        (source / "MANIFEST.json").write_bytes(canonical(value))
        args["source_descriptors"][0]["manifest_sha256"] = hashlib.sha256((source / "MANIFEST.json").read_bytes()).hexdigest()
        with self.assertRaisesRegex(R.ReadinessError, "full identity"):
            R.verify_offline_evidence(**args)

    def test_native_radar_nonfinite_refused_even_after_source_manifest_rehash(self):
        import io
        args = self.with_sources()
        source = self.root / "sources"
        buffer = io.BytesIO()
        np.save(buffer, np.array([[np.nan, 0, 0]], np.float32), allow_pickle=False)
        (source / "radar_world_xyz.npy").write_bytes(buffer.getvalue())
        value = json.loads((source / "MANIFEST.json").read_text())
        value["sha256"]["radar_world_xyz.npy"] = hashlib.sha256(buffer.getvalue()).hexdigest()
        (source / "MANIFEST.json").write_bytes(canonical(value))
        args["source_descriptors"][0]["manifest_sha256"] = hashlib.sha256((source / "MANIFEST.json").read_bytes()).hexdigest()
        with self.assertRaisesRegex(R.ReadinessError, "radar"):
            R.verify_offline_evidence(**args)


if __name__ == "__main__":
    unittest.main()
