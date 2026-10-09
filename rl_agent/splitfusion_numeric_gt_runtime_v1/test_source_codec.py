"""Exact source adapter tests using native local CARLA values; no services."""
import copy
import gc
import os
import unittest
from unittest.mock import patch

import carla
import numpy as np

from rl_agent.splitfusion_live_gt_optimization_v4 import snapshot as S, memfd_transport as M
from rl_agent.splitfusion_live_gt_optimization_v4.test_snapshot import case as snapshot_case
from . import source_codec as C


def fixture():
    values, radar = snapshot_case()
    values.pop("sync_evidence")
    stamp = values.pop("policy_timestamp_s")
    values["identity"] = dict(run_id="codec", cell_id="cpu", stream_id="example", frame_id=51,
                              action_id=None, profile_id=None, capture_timestamp_ns=123125000000)
    values["radar_world_xyz"] = radar
    values["dynamic_snapshot_evidence"] = copy.deepcopy(values["frozen_world"].population_evidence)
    return values, stamp


class SourceCodecTests(unittest.TestCase):
    def test_unchanged_snapshot_bytes_and_exact_roundtrip(self):
        source, stamp = fixture()
        chunks, extras = C.encode_source(source, stamp)
        original = {key: source[key] for key in C.SNAPSHOT_KEYS - {"policy_timestamp_s"}}
        original["policy_timestamp_s"] = stamp
        self.assertEqual(b"".join(chunks), S.pack_snapshot(original, source["radar_world_xyz"]))
        with patch.object(carla, "Client", side_effect=AssertionError("no simulator RPC")):
            actual = C.decode_source(b"".join(chunks), extras)
        self.assertEqual(set(actual), C.SOURCE_KEYS)
        self.assertEqual(actual["identity"], source["identity"])
        self.assertEqual(actual["dynamic_snapshot_evidence"], source["dynamic_snapshot_evidence"])
        for key in ("camera_matrix", "camera_inverse", "radar_world_xyz"):
            self.assertEqual(actual[key].tobytes(), source[key].tobytes())
            self.assertEqual(actual[key].dtype, source[key].dtype)
            self.assertEqual(actual[key].ctypes.data % 16, 0)
            with self.assertRaises(ValueError): actual[key].setflags(write=True)
        for name in ("semantic_image", "depth_image"):
            self.assertEqual(bytes(actual[name].raw_data), bytes(source[name].raw_data))
            self.assertEqual(actual[name].transform.get_inverse_matrix(), source[name].transform.get_inverse_matrix())
            self.assertTrue(actual[name].raw_data.readonly)
        camera = carla.Location(x=1.125, y=-3.25, z=2.5)
        for old, new in zip(source["frozen_world"].get_actors(), actual["frozen_world"].get_actors()):
            self.assertEqual(old.get_transform().get_matrix(), new.get_transform().get_matrix())
            self.assertEqual(old.get_transform().get_inverse_matrix(), new.get_transform().get_inverse_matrix())
            self.assertEqual(old.get_location().distance(camera).hex(), new.get_location().distance(camera).hex())

    def test_sealed_mapping_roundtrip_closes_after_releasing_views(self):
        source, stamp = fixture(); chunks, extras = C.encode_source(source, stamp)
        fd, size = M.create(chunks)
        try:
            with M.mapped(fd, size) as mapping:
                actual = C.decode_source(mapping, extras)
                self.assertEqual(actual["radar_world_xyz"].tobytes(), source["radar_world_xyz"].tobytes())
                self.assertEqual(bytes(actual["depth_image"].raw_data), source["depth_image"].raw_data)
                del actual
                gc.collect()
        finally:
            os.close(fd)

    def test_extras_owned_not_aliased_and_cross_snapshot_refused(self):
        source, stamp = fixture(); chunks, extras = C.encode_source(source, stamp)
        source["dynamic_snapshot_evidence"]["active_population_ids"].clear()
        actual = C.decode_source(b"".join(chunks), extras)
        self.assertEqual(actual["dynamic_snapshot_evidence"]["active_population_ids"], [32, 7])
        extras["dynamic_snapshot_evidence"]["active_population_ids"].clear()
        self.assertEqual(actual["dynamic_snapshot_evidence"]["active_population_ids"], [32, 7])
        with self.assertRaises(ValueError): C.decode_source(b"".join(chunks), extras)

    def test_extra_binding_identity_and_population_mutations_refused(self):
        source, stamp = fixture(); chunks, extras = C.encode_source(source, stamp); raw = b"".join(chunks)
        changes = (lambda e: e.update(snapshot_bytes=e["snapshot_bytes"] + 1),
            lambda e: e.update(snapshot_body_sha256="0" * 64),
            lambda e: e.update(schema="other"),
            lambda e: e["capture_identity"].update(stream_id="foreign"),
            lambda e: e["dynamic_snapshot_evidence"].update(population_sha256="b" * 64),
            lambda e: e["dynamic_snapshot_evidence"].update(registry_complete=False),
            lambda e: e["dynamic_snapshot_evidence"].update(frame_id=52),
            lambda e: e["dynamic_snapshot_evidence"].update(active_population_ids=[32, 32]))
        for change in changes:
            mutated = copy.deepcopy(extras); change(mutated)
            with self.subTest(change=change), self.assertRaises(ValueError): C.decode_source(raw, mutated)

    def test_source_inventory_and_timestamp_refused(self):
        source, stamp = fixture()
        for changed in (dict(source, unexpected=True), dict(source, sync_evidence=object()),
                        dict(source, policy_timestamp_s=stamp + 1),
                        {key: value for key, value in source.items() if key != "radar_world_xyz"}):
            with self.assertRaises(ValueError): C.encode_source(changed, stamp)
        for value in (True, float("nan"), float("inf"), stamp + 1):
            with self.assertRaises(ValueError): C.encode_source(source, value)
        C.encode_source(dict(source, policy_timestamp_s=stamp), stamp)

    def test_world_evidence_cannot_differ_from_extra(self):
        source, stamp = fixture()
        source["dynamic_snapshot_evidence"]["population_sha256"] = "b" * 64
        with self.assertRaises(ValueError): C.encode_source(source, stamp)

    def test_mutable_views_corruption_and_truncation_refused(self):
        source, stamp = fixture(); chunks, extras = C.encode_source(source, stamp); raw = b"".join(chunks)
        cases = (bytearray(raw), memoryview(bytearray(raw)).toreadonly(),
                 raw[:-1], raw + b"0", raw[:-1] + bytes([raw[-1] ^ 1]))
        for value in cases:
            with self.assertRaises(ValueError): C.decode_source(value, extras)

    def test_does_not_fabricate_missing_native_geometry(self):
        source, stamp = fixture()
        transform = source["frozen_world"].get_actors()[0].get_transform()
        source["frozen_world"].get_actors()[0].get_transform = lambda: type("MissingPose", (), {
            "get_matrix": transform.get_matrix, "get_inverse_matrix": transform.get_inverse_matrix})()
        with self.assertRaisesRegex(ValueError, "geometry vector"):
            C.encode_source(source, stamp)


if __name__ == "__main__":
    unittest.main()
