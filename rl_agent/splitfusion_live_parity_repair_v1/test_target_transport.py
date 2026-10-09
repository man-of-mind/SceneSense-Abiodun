"""CPU-only integrity/identity tests; synthetic data is never live-qualified."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from . import evaluation as E
from . import target_transport as T


def fixture():
    identity = dict(run_id="live_test", cell_id="cell", stream_id="stream", frame_id=10,
                    action_id=None, profile_id=None, capture_timestamp_ns=12345)
    frame = dict(sample_id="live_test_frame10", experiment_id="live_test", frame_id=10,
                 camera_width=6, camera_height=4, mask_path="mask.png", instance_raw_path="raw.png")
    quarantine = {name: [] for name in A.QUARANTINE_KEYS}
    provenance = dict(schema=A.SCHEMA, scope=T.LIVE_SCOPE, sample_id=frame["sample_id"],
                      complete=dict.fromkeys(A.COMPLETENESS_KEYS, True),
                      evidence_sha256={"synthetic_fixture_not_live_proof": "a" * 64},
                      capture_identity=identity)
    inputs = A.CompleteTargetInputs(frame, "val", np.zeros((4, 6, 4), np.uint8),
        np.zeros((4, 6), np.uint8), np.zeros((4, 6), np.uint8), [], [], [], quarantine, provenance)
    inputs = replace(inputs, provenance=dict(provenance, input_content_sha256=A.input_content_sha256(inputs)))
    blank = np.zeros((432, 768), np.uint8)
    ok, encoded = cv2.imencode(".png", blank)
    assert ok
    target = dict(contract=A.CONTRACT, provenance=inputs.provenance, live_qualified=False,
        segmentation=blank.copy(), object_ignore=blank.copy(), segmentation_png=encoded.tobytes(),
        object_ignore_png=encoded.tobytes(), tables={"object_boxes.csv": [], "object_ignore_regions.csv": [],
        "target_manifest.csv": [{"sample_id": frame["sample_id"], "frame_id": "10"}]})
    prepared = E.restore_prepared_target(frame_id=10, sample_id=frame["sample_id"], target=target,
                                         content_sha256=E._target_digest(10, frame["sample_id"], target))
    return identity, inputs, prepared


class TargetTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="complete_v010_transport_test_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.identity, self.inputs, self.prepared = fixture()

    def write(self):
        result = T.write_bundle(self.root, identity=self.identity, inputs=self.inputs, prepared=self.prepared)
        self.bundle = Path(result["directory"])
        return result

    def mutate_manifest(self, mutate):
        path = self.bundle / T.READY_NAME
        doc = json.loads(path.read_text())
        mutate(doc)
        path.write_bytes(T._canonical(doc))

    def test_roundtrip_default_and_full_native_verification(self):
        original_hash = A.input_content_sha256(self.inputs)
        result = self.write()
        light = T.read_bundle(self.root, expected_identity=self.identity, timeout_s=0,
                              expected_manifest_sha256=result["manifest_sha256"])
        self.assertIsNone(light.inputs)
        self.assertFalse(light.native_inputs_verified)
        self.assertEqual(light.prepared.content_sha256, self.prepared.content_sha256)
        full = T.verify_full_inputs(self.root, expected_identity=self.identity,
                                    expected_manifest_sha256=result["manifest_sha256"])
        self.assertTrue(full.native_inputs_verified)
        self.assertEqual(A.input_content_sha256(full.inputs), original_hash)
        self.assertEqual(A.input_content_sha256(self.inputs), original_hash)
        self.assertFalse(full.inputs.depth_bgra.flags.writeable)
        self.assertFalse(full.prepared.target["segmentation"].flags.writeable)
        self.assertGreater(full.manifest["ready_wall_ns"], full.manifest["ready_raw_ns"])

    def test_create_only_and_input_sources_untouched(self):
        result = self.write()
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.bundle.iterdir()}
        with self.assertRaises(FileExistsError):
            self.write()
        self.assertEqual(hashes, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.bundle.iterdir()})
        self.assertEqual(hashes[T.READY_NAME], result["manifest_sha256"])

    def test_all_identity_keys_exact_and_no_path_interpolation(self):
        with self.assertRaises(T.TargetTransportError):
            T.bundle_path(self.root, dict(self.identity, extra="not allowed"))
        with self.assertRaises(T.TargetTransportError):
            T.bundle_path(self.root, {k: v for k, v in self.identity.items() if k != "profile_id"})
        path = T.bundle_path(self.root, dict(self.identity, stream_id="../../escape"))
        self.assertEqual(path.parent, self.root)

    def test_incomplete_avo_and_historical_scope_refused(self):
        with self.assertRaises(T.TargetTransportError):
            T.write_bundle(self.root, identity=self.identity, inputs={"avo": {}}, prepared=self.prepared)
        for provenance in (dict(self.inputs.provenance, scope="HISTORICAL_REGISTERED_V010"),
                           dict(self.inputs.provenance, complete={}),
                           dict(self.inputs.provenance, capture_identity=dict(self.identity, frame_id=11))):
            with self.assertRaises(ValueError):
                T.write_bundle(self.root, identity=self.identity,
                               inputs=replace(self.inputs, provenance=provenance), prepared=self.prepared)

    def test_historical_name_rejected_even_under_new_scope(self):
        frame = dict(self.inputs.frame, experiment_id="canonical_v3_06_val_50_50_s602_tm1602")
        inputs = replace(self.inputs, frame=frame)
        inputs = replace(inputs, provenance=dict(inputs.provenance, input_content_sha256=A.input_content_sha256(inputs)))
        with self.assertRaisesRegex(T.TargetTransportError, "historical"):
            T.write_bundle(self.root, identity=self.identity, inputs=inputs, prepared=self.prepared)

    def test_native_input_tamper_detected_offline_not_reloaded_live(self):
        self.write()
        path = self.bundle / "depth_bgra.npy"
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
        light = T.read_bundle(self.root, expected_identity=self.identity, timeout_s=0)
        self.assertFalse(light.native_inputs_verified)
        with self.assertRaisesRegex(T.TargetTransportError, "hash/size"):
            T.verify_full_inputs(self.root, expected_identity=self.identity)

    def test_predicate_no_npy_reads_on_light_path(self):
        self.write()
        original = T._read
        def guarded(path, **kwargs):
            self.assertNotEqual(path.suffix, ".npy")
            return original(path, **kwargs)
        with patch.object(T, "_read", side_effect=guarded):
            T.read_bundle(self.root, expected_identity=self.identity, timeout_s=0)

    def test_target_tamper_or_missing_refused_immediately(self):
        self.write()
        path = self.bundle / "segmentation.png"
        path.write_bytes(path.read_bytes() + b"tamper")
        with self.assertRaises(T.TargetTransportError):
            T.read_bundle(self.root, expected_identity=self.identity, timeout_s=.5)
        path.unlink()
        with self.assertRaises(T.TargetTransportError):
            T.read_bundle(self.root, expected_identity=self.identity, timeout_s=.5)

    def test_foreign_frame_and_clock_refused(self):
        self.write()
        self.mutate_manifest(lambda d: d.update(frame_id=11))
        with self.assertRaisesRegex(T.TargetTransportError, "foreign"):
            T.read_bundle(self.root, expected_identity=self.identity, timeout_s=0)
        self.mutate_manifest(lambda d: d.update(frame_id=10, clock_domain="CARLA_SIM_SECONDS"))
        with self.assertRaisesRegex(T.TargetTransportError, "clock"):
            T.read_bundle(self.root, expected_identity=self.identity, timeout_s=0)

    def test_path_traversal_and_symlinks_refused(self):
        self.write()
        self.mutate_manifest(lambda d: d["payloads"]["target.json"].update(filename="../outside.json"))
        with self.assertRaisesRegex(T.TargetTransportError, "filename"):
            T.read_bundle(self.root, expected_identity=self.identity, timeout_s=0)
        self.mutate_manifest(lambda d: d["payloads"]["target.json"].update(filename="target.json"))
        path = self.bundle / "target.json"
        backup = self.root / "outside.json"
        path.rename(backup)
        path.symlink_to(backup)
        with self.assertRaisesRegex(T.TargetTransportError, "symbolic"):
            T.read_bundle(self.root, expected_identity=self.identity, timeout_s=0)

    def test_missing_ready_has_bounded_timeout_and_limits(self):
        started = time.monotonic()
        with self.assertRaises(T.TargetNotReadyError):
            T.read_bundle(self.root, expected_identity=self.identity, timeout_s=.01)
        self.assertLess(time.monotonic() - started, .2)
        for timeout in (-1, 4, float("nan")):
            with self.assertRaises(T.TargetTransportError):
                T.read_bundle(self.root, expected_identity=self.identity, timeout_s=timeout)

    def test_write_failure_never_publishes_ready(self):
        with patch.object(T.os, "fsync", side_effect=OSError("injected disk error")):
            with self.assertRaises(OSError):
                self.write()
        self.assertFalse((T.bundle_path(self.root, self.identity) / T.READY_NAME).exists())
        with self.assertRaises(T.TargetNotReadyError):
            T.read_bundle(self.root, expected_identity=self.identity, timeout_s=0)


if __name__ == "__main__":
    unittest.main()
