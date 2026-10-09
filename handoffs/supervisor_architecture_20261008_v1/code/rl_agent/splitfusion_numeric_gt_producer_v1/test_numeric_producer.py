"""CPU-only oracle parity, immutability, staging and fail-closed tests.

No simulator, service, GPU or retained-evidence write. Synthetic fixtures come
from the retained repair tests; historical fixtures from the target parity
package; the pinned original producer/builder are the oracles.
"""
from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as S
from rl_agent.splitfusion_live_parity_repair_v1 import test_source_geometry as repair_fixtures
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from rl_agent.splitfusion_live_target_parity_v1.fixtures import FIXTURES, load_retained_fixture
from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as GE
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG

from . import accounting as AC, artifacts as AR, avo as NA, originals as O, payload as NP, source as NS, target as NT

KEYS = ("identity", "rgb", "semantic_image", "depth_image", "frozen_world", "camera_matrix", "camera_inverse",
        "radar_world_xyz", "dynamic_snapshot_evidence")
ARGS = dict(experiment_id="numeric_cpu_test", ego_id=999, dynamic_population_ids=[1, 2],
            population_evidence_sha256="d" * 64, max_frames=1000)


def canon(value):
    return A.canonical(A._json_value(value))


def producers(directory):
    return (S.LiveCompleteTargetProducer(static_catalogue_dir=directory / "catalogue", **ARGS),
            NS.NumericSourceProducer(static_catalogue_dir=directory / "catalogue", **ARGS))


def upright(kwargs):
    """The retained synthetic pedestrian rolls 8 degrees; the frozen yaw-only AVO
    scorer refuses tilted scoreable targets (original and numeric alike). Make the
    pedestrian upright so the AVO path can be exercised; geometry otherwise same."""
    for actor in kwargs["frozen_world"].get_actors():
        if actor.type_id.startswith("walker.pedestrian."):
            rotation = actor.get_transform().rotation
            rotation.pitch = rotation.roll = 0.
    return kwargs


def frame_products(staged, kwargs):
    target = NT.numeric_target(frame=staged.frame, split="val", depth=staged.depth, semantic=staged.semantic,
        original_mask=staged.mask, boxes=staged.boxes, visibility=staged.visibility, static_rows=staged.static,
        quarantine=staged.conflict["quarantine"], vehicle_mask=staged.vehicle_mask)
    avo = NA.numeric_avo(identity=kwargs["identity"], rgb=kwargs["rgb"], semantic_image=kwargs["semantic_image"],
        depth_image=kwargs["depth_image"], frozen_world=kwargs["frozen_world"], camera_matrix=staged.camera_matrix,
        camera_inverse=staged.camera_inverse, depth=staged.depth, sync_validated=True)
    return target, avo, NP.build_payload(staged=staged, target=target, avo=avo)


class NumericProducerSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.fixture = repair_fixtures.SourceGeometryTests()

    def test_original_and_numeric_agree_on_synthetic_fixture(self):
        with self.fixture.fixture() as (directory, kwargs), patch("time.clock_gettime_ns", return_value=123456789):
            kwargs = upright(kwargs)
            original, numeric = producers(directory)
            expected = original.build(**kwargs)
            staged = numeric.compute(**kwargs)
            self.assertEqual(canon(staged.inputs.provenance), canon(expected.provenance))
            self.assertEqual(canon(staged.evidence), canon(original.last_evidence))
            for name in ("depth_bgra", "raw_semantic", "original_v3_mask"):
                np.testing.assert_array_equal(getattr(staged.inputs, name), getattr(expected, name))
            target, avo, payload = frame_products(staged, kwargs)
            reference = E.prepare_target(expected)
            legacy = AR.legacy_target(staged, target)
            prepared = AR.prepared_target(staged, legacy)
            self.assertEqual(prepared.content_sha256, reference.content_sha256)
            self.assertEqual(legacy["segmentation_png"], reference.target["segmentation_png"])
            self.assertEqual(legacy["object_ignore_png"], reference.target["object_ignore_png"])
            self.assertEqual(legacy["tables"], reference.target["tables"])
            self.assertEqual(list(map(dict, payload.object_boxes)), reference.target["tables"]["object_boxes.csv"])
            self.assertEqual(list(map(dict, payload.object_ignore_regions)), reference.target["tables"]["object_ignore_regions.csv"])
            np.testing.assert_array_equal(payload.segmentation, reference.target["segmentation"])
            np.testing.assert_array_equal(payload.object_ignore, reference.target["object_ignore"])
            sink = {}
            with patch.object(GE, "_create_identical_or_fail", lambda path, data: sink.__setitem__(str(path), data)):
                record = LG.write_avo_bundle(directory, identity=kwargs["identity"], rgb=kwargs["rgb"],
                    semantic_image=kwargs["semantic_image"], depth_image=kwargs["depth_image"],
                    frozen_world=kwargs["frozen_world"], camera_matrix=kwargs["camera_matrix"],
                    camera_inverse=kwargs["camera_inverse"], policy_timestamp_s=kwargs["rgb"]["timestamp"])
            document = json.loads(sink[record["path"]])
            self.assertEqual(canon(avo.avo), canon(document["avo"]))
            self.assertEqual(canon(list(avo.rows)), canon(document["raw_pedestrian_rows"]))
            clocks = {k: document[k] for k in AR.AVO_CLOCKS if k != "sync_observed_raw_ns"}
            clocks["sync_observed_raw_ns"] = document["source_synchronization"]["observed_raw_ns"]
            rebuilt = AR.avo_documents(staged, avo, identity=kwargs["identity"], clocks=clocks,
                payload_filenames={k: v["filename"] for k, v in document["payloads"].items()},
                prescore_filename=document["prescore_input"]["filename"])
            self.assertEqual(rebuilt["document_sha256"], record["sha256"])
            binding = AR.binding(payload, legacy, prepared, avo_document_sha256=rebuilt["document_sha256"])
            self.assertEqual(binding["prepared_content_sha256"], reference.content_sha256)
            self.assertNotEqual(binding["semantic_sha256"], reference.content_sha256)
            numeric.commit(staged)

    def test_tilted_pedestrian_refused_identically_by_original_and_numeric_avo(self):
        with self.fixture.fixture() as (directory, kwargs), patch("time.clock_gettime_ns", return_value=5):
            _, numeric = producers(directory)
            staged = numeric.compute(**kwargs)
            with self.assertRaisesRegex(LG.LiveGtContractError, "tilted actor"):
                LG.write_avo_bundle(directory, identity=kwargs["identity"], rgb=kwargs["rgb"],
                    semantic_image=kwargs["semantic_image"], depth_image=kwargs["depth_image"],
                    frozen_world=kwargs["frozen_world"], camera_matrix=kwargs["camera_matrix"],
                    camera_inverse=kwargs["camera_inverse"], policy_timestamp_s=kwargs["rgb"]["timestamp"])
            with self.assertRaisesRegex(LG.LiveGtContractError, "tilted actor"):
                NA.numeric_avo(identity=kwargs["identity"], rgb=kwargs["rgb"], semantic_image=kwargs["semantic_image"],
                    depth_image=kwargs["depth_image"], frozen_world=kwargs["frozen_world"], camera_matrix=staged.camera_matrix,
                    camera_inverse=staged.camera_inverse, depth=staged.depth, sync_validated=True)
            numeric.cancel(staged)

    def test_payload_is_immutable_bytes_backed_and_digest_verified(self):
        with self.fixture.fixture() as (directory, kwargs), patch("time.clock_gettime_ns", return_value=5):
            kwargs = upright(kwargs)
            _, numeric = producers(directory)
            staged = numeric.compute(**kwargs)
            _, _, payload = frame_products(staged, kwargs)
            self.assertIs(type(payload.segmentation_bytes), bytes)
            self.assertFalse(payload.segmentation.flags.writeable)
            self.assertFalse(payload.object_ignore.flags.writeable)
            self.assertEqual(payload.segmentation.shape, (432, 768))
            with self.assertRaises(ValueError):
                payload.segmentation[0, 0] = 7
            self.assertIsInstance(payload.object_boxes, tuple)
            self.assertIsInstance(payload.avo, MappingProxyType)
            self.assertIsInstance(payload.provenance["evidence_sha256"], MappingProxyType)
            with self.assertRaises(TypeError):
                payload.avo["qualified"] = []
            with self.assertRaises(FrozenInstanceError):
                payload.sample_id = "x"
            self.assertEqual(NP.verify_payload(payload), payload.semantic_sha256)
            tampered = NP.NumericGtPayload(**{**payload.__dict__, "object_ignore_bytes": bytes(len(payload.object_ignore_bytes))})
            with self.assertRaisesRegex(NP.PayloadError, "digest"):
                NP.verify_payload(tampered)
            thawed = payload.metadata()
            thawed["object_boxes"].append({"sample_id": "injected"})
            self.assertEqual(NP.verify_payload(payload), payload.semantic_sha256)
            numeric.cancel(staged)

    def test_staging_protocol_exactly_once(self):
        with self.fixture.fixture() as (directory, kwargs), patch("time.clock_gettime_ns", return_value=5):
            _, numeric = producers(directory)
            before = numeric.history_sha256()
            staged = numeric.compute(**kwargs)
            self.assertEqual(staged.history_before_sha256, before)
            with self.assertRaisesRegex(NS.StagingError, "pending"):
                numeric.compute(**kwargs)
            with self.assertRaisesRegex(NS.StagingError, "pending"):
                numeric.register_population([1, 2, 3], "e" * 64, 101)
            foreign = copy.copy(staged)
            with self.assertRaisesRegex(NS.StagingError, "not the pending frame"):
                numeric.commit(foreign)
            cancelled = numeric.cancel(staged)
            self.assertEqual(cancelled["history_sha256"], before)
            self.assertEqual(numeric.processed_frames, 0)
            self.assertEqual(numeric.conflict_state.frame_count, 0)
            self.assertEqual(numeric.tracker._ages, {})
            with self.assertRaises(NS.StagingError):
                numeric.cancel(staged)
            again = numeric.compute(**kwargs)
            committed = numeric.commit(again)
            self.assertEqual(committed["history_after_sha256"], again.history_after_sha256)
            self.assertEqual(numeric.history_sha256(), again.history_after_sha256)
            self.assertNotEqual(numeric.history_sha256(), before)
            with self.assertRaises(NS.StagingError):
                numeric.commit(again)
            with self.assertRaisesRegex(S.SourceGeometryError, "strictly increasing"):
                numeric.compute(**kwargs)

    def test_cancel_discards_frame_bound_registration(self):
        with self.fixture.fixture() as (directory, kwargs), patch("time.clock_gettime_ns", return_value=5):
            _, numeric = producers(directory)
            numeric.register_population([1, 2, 3], "e" * 64, 100)
            staged = numeric.compute(**kwargs)
            self.assertEqual(staged.population, frozenset((1, 2, 3)))
            numeric.cancel(staged)
            self.assertIsNone(numeric.pending_registration)
            self.assertEqual(numeric.population, frozenset((1, 2)))
            numeric.register_population([1, 2, 3], "e" * 64, 100)
            staged = numeric.compute(**kwargs)
            numeric.commit(staged)
            self.assertEqual(numeric.population, frozenset((1, 2, 3)))
            self.assertEqual(numeric.population_evidence_sha256, "e" * 64)

    def test_validators_fail_closed_identically_and_leave_history_untouched(self):
        with self.fixture.fixture() as (directory, kwargs), patch("time.clock_gettime_ns", return_value=5):
            original, numeric = producers(directory)
            before = numeric.history_sha256()
            broken = []
            wrong_frame = copy.copy(kwargs); wrong_frame["dynamic_snapshot_evidence"] = dict(kwargs["dynamic_snapshot_evidence"], frame_id=99)
            broken.append(("wrong frame", wrong_frame))
            unknown = copy.copy(kwargs); unknown["dynamic_snapshot_evidence"] = dict(kwargs["dynamic_snapshot_evidence"], active_population_ids=[1, 2, 7])
            broken.append(("unknown actor", unknown))
            pose = copy.copy(kwargs); pose["camera_matrix"] = np.eye(4) * 2; pose["camera_inverse"] = np.eye(4) / 2
            broken.append(("camera pose", pose))
            radar = copy.copy(kwargs); radar["radar_world_xyz"] = np.array([[np.nan, 0, 0]])
            broken.append(("nonfinite radar", radar))
            small = copy.copy(kwargs); small["depth_image"] = SimpleNamespace(**{**vars(kwargs["depth_image"]), "width": 640})
            broken.append(("image size", small))
            for label, value in broken:
                with self.subTest(label):
                    with self.assertRaises((S.SourceGeometryError, LG.LiveGtContractError)) as expected:
                        original.build(**value)
                    with self.assertRaises((S.SourceGeometryError, LG.LiveGtContractError)) as actual:
                        numeric.compute(**value)
                    self.assertIs(type(actual.exception), type(expected.exception))
                    self.assertEqual(str(actual.exception), str(expected.exception))
                    self.assertIsNone(numeric.staged)
                    self.assertEqual(numeric.history_sha256(), before)
            staged = numeric.compute(**kwargs)
            numeric.commit(staged)

    def test_hoisted_radar_transform_exact_and_guarded(self):
        objects = O.bound().objects
        rng = np.random.default_rng(7)
        radar = rng.normal(size=(5000, 3)).astype(np.float32)
        import carla
        transform = carla.Transform(carla.Location(3., -2., .5), carla.Rotation(pitch=2., yaw=37., roll=-1.))
        actor = SimpleNamespace(get_transform=lambda: transform)
        expected = NS._ORIGINAL_TRANSFORM(radar, actor)
        hoisted = NS.hoisted_transform(radar)
        np.testing.assert_array_equal(hoisted(radar, actor), expected)
        np.testing.assert_array_equal(hoisted(radar, actor), expected)
        with self.assertRaisesRegex(S.SourceGeometryError, "foreign point array"):
            hoisted(radar.copy(), actor)
        self.assertIs(objects["transform_world_to_actor_local"], NS._ORIGINAL_TRANSFORM)

    def test_frame_masks_exact(self):
        rng = np.random.default_rng(11)
        semantic = rng.integers(0, 23, (720, 1280), dtype=np.uint8)
        vehicle, same = NS.frame_masks(semantic, O.bound().static)
        np.testing.assert_array_equal(vehicle, np.isin(semantic, [14, 15, 16]))
        np.testing.assert_array_equal(same(15), semantic == 15)
        self.assertIs(same(15), same(15))

    def test_legacy_rows_match_original_csv_file_roundtrip(self):
        builder = O.bound().builder
        rows = [{"sample_id": "s", "frame_id": 3, "gt_bbox_x": 1.5, "stationary_label": 1, "radar_support_radius_m": "",
                 "object_world_x": 0.1 + 0.2, "label": "vehicle", "extra": "ignored"}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "object_boxes.csv"
            builder.write_csv_x(path, builder.OBJECT_FIELDS, rows)
            self.assertEqual(NP.legacy_rows(rows, builder.OBJECT_FIELDS), builder.read_csv(path))
            self.assertEqual(NP.legacy_csv_bytes(rows, builder.OBJECT_FIELDS), path.read_bytes())

    def test_avo_documents_require_all_clocks(self):
        staged = SimpleNamespace(sync={}, depth_raw=np.zeros((2, 2, 4), np.uint8), semantic=np.zeros((2, 2), np.uint8),
                                 frame_id=1, stamp=1., camera_matrix=np.eye(4), camera_inverse=np.eye(4))
        avo = SimpleNamespace(intrinsics=np.eye(3), rows=(), avo={"frame_id": 1})
        with self.assertRaisesRegex(AR.ArtifactError, "seven AVO clocks"):
            AR.avo_documents(staged, avo, identity={}, clocks={"start_raw_ns": 1}, payload_filenames={}, prescore_filename="p")

    def test_accounting_wrappers_restore(self):
        import hashlib as H
        original = H.sha256
        with AC.measure() as counters:
            H.sha256(b"abc").hexdigest()
            np.isin(np.arange(3), [1])
        self.assertIs(H.sha256, original)
        self.assertEqual(counters.sha256_calls, 1)
        self.assertEqual(counters.isin_calls, 1)
        self.assertIs(O.bound().objects["transform_world_to_actor_local"], NS._ORIGINAL_TRANSFORM)

    def test_startup_binding_pins_only_retained_nine_for_evidence(self):
        bound = O.bound()
        self.assertEqual(set(bound.source_pins), set(S.verify_sources()))
        self.assertEqual(dict(bound.source_pins), S.verify_sources())
        self.assertIn("person_avo", bound.pins)
        self.assertFalse(O.startup_binding()["readiness_path_reverifies_source_files"])


class RetainedHistoricalFixtureTests(unittest.TestCase):
    """Older corpus coverage: canonical v3 validation frames through the numeric target."""
    def test_numeric_target_matches_retained_v010_and_full_original_builder(self):
        for sample in FIXTURES:
            with self.subTest(sample):
                inputs, targets = load_retained_fixture(sample)
                numeric = NT.numeric_target_from_inputs(inputs)
                segmentation = cv2.imread(str(targets["segmentation"]), cv2.IMREAD_UNCHANGED)
                object_ignore = cv2.imread(str(targets["object_ignore"]), cv2.IMREAD_UNCHANGED)
                np.testing.assert_array_equal(numeric.segmentation, segmentation)
                np.testing.assert_array_equal(numeric.object_ignore, object_ignore)
                builder = O.bound().builder
                self.assertEqual(NP.legacy_rows(numeric.positive_rows, builder.OBJECT_FIELDS), targets["object_boxes.csv"])
                self.assertEqual(NP.legacy_rows(numeric.ignore_rows, builder.IGNORE_FIELDS), targets["object_ignore_regions.csv"])
                original = A.build_complete_v010(inputs)
                staged = SimpleNamespace(inputs=inputs, frame=dict(inputs.frame))
                legacy = AR.legacy_target(staged, numeric, sources=original["sources"])
                for key in ("summary", "tables", "provenance", "contract", "live_qualified"):
                    self.assertEqual(canon(legacy[key]), canon(original[key]), key)
                self.assertEqual(legacy["segmentation_png"], original["segmentation_png"])
                self.assertEqual(legacy["object_ignore_png"], original["object_ignore_png"])
                prepared = AR.prepared_target(staged, legacy)
                self.assertEqual(prepared.content_sha256, E.prepare_target(inputs).content_sha256)


if __name__ == "__main__":
    unittest.main()
