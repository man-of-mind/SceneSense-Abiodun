"""Startup, population changes, and real existing-GT-worker composition, CPU."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

from . import gt_bridge as B, source_geometry as S, target_transport as T
from . import test_source_geometry as TS


class Snapshot(list):
    frame = 100
    timestamp = SimpleNamespace(elapsed_seconds=5.)
    def find(self, ident):
        return next((item for item in self if item.id == ident), None)


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        B.bind_legacy_namespace()
    def test_population_replenishment_despawn_and_exact_pose(self):
        from rl_agent.splitfusion_quality_feedback_probe_v1 import adapter_quality_v1 as Q
        a, b = TS.actor(1, "vehicle.one", 20., 0.), TS.actor(2, "walker.pedestrian.two", 20., 4.)
        current = [a]
        snapshot = Snapshot(current)
        world = SimpleNamespace(get_actors=lambda: tuple(current), get_snapshot=lambda: snapshot)
        source = B.make_complete_scene_source(Q.QualitySceneSnapshotSource)(world, ego_id=99)
        source.refresh_static(force=True)
        first = source.capture(snapshot)
        self.assertEqual(first.population_evidence["active_population_ids"], [1])
        self.assertFalse(first.population_evidence["registry_refreshed"])
        current[:] = [a, b]; snapshot[:] = current; snapshot.frame += 1
        second = source.capture(snapshot)
        self.assertEqual(second.population_evidence["registered_population_ids"], [1, 2])
        self.assertTrue(second.population_evidence["registry_refreshed"])
        current[:] = [b]; snapshot[:] = current; snapshot.frame += 1
        third = source.capture(snapshot)
        self.assertEqual(third.population_evidence["active_population_ids"], [2])
        self.assertEqual(third.population_evidence["registered_population_ids"], [1, 2])
        self.assertEqual(list(first.get_actors())[0].get_transform().get_matrix(), a.get_transform().get_matrix())
        self.assertEqual(first.population_evidence["active_population_ids"], [1])

    def test_population_unknown_or_world_advance_refuses(self):
        from rl_agent.splitfusion_quality_feedback_probe_v1 import adapter_quality_v1 as Q
        actor = TS.actor(1, "vehicle.one", 20., 0.)
        snapshot = Snapshot([actor])
        world = SimpleNamespace(get_actors=lambda: [], get_snapshot=lambda: snapshot)
        source = B.make_complete_scene_source(Q.QualitySceneSnapshotSource)(world, ego_id=99)
        source.refresh_static(force=True)
        with self.assertRaisesRegex(S.SourceGeometryError, "unregistered"):
            source.capture(snapshot)
        world.get_actors = lambda: [actor]
        world.get_snapshot = lambda: SimpleNamespace(frame=101)
        with self.assertRaisesRegex(S.SourceGeometryError, "advanced"):
            source.capture(snapshot)

    def test_fresh_catalogue_never_guesses_enablement_late(self):
        with tempfile.TemporaryDirectory() as root:
            catalogue = B.FreshCatalogue(Path(root)/"catalogue")
            occupied = SimpleNamespace(get_actors=lambda: [SimpleNamespace(type_id="vehicle.test")])
            with mock.patch.object(S, "capture_catalogue") as capture:
                with self.assertRaisesRegex(S.SourceGeometryError, "before dynamic"):
                    catalogue.after_load(occupied)
                capture.assert_not_called()
            with self.assertRaisesRegex(S.SourceGeometryError, "not owned"):
                catalogue.require_world(occupied)

    def test_fresh_load_hook_and_restore_even_on_failure(self):
        from data_collection.run_route_b_perception_collection_v2 import ClientProxy
        from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
        before = LG.write_avo_bundle
        fake = SimpleNamespace(load_world=mock.Mock(return_value="new world"))
        proxy = ClientProxy(lambda: fake, 1)
        with tempfile.TemporaryDirectory() as root, mock.patch.object(B.FreshCatalogue, "after_load", return_value="captured world") as capture:
            with B.install_complete_gt(attempt_dir=Path(root), max_frames=2):
                self.assertEqual(proxy.load_world("Town10HD_Opt", True), "captured world")
                capture.assert_called_once_with("new world")
                fake.load_world.assert_called_once_with("Town10HD_Opt", True)
            self.assertNotIn("load_world", ClientProxy.__dict__)
        self.assertIs(LG.write_avo_bundle, before)

    def test_real_avo_worker_complete_producer_transport_and_preservation(self):
        from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
        fixture = TS.SourceGeometryTests()
        with fixture.fixture(pitch=0) as (root, args):
            attempt = root / "attempt"; attempt.mkdir()
            args["rgb"]["_rgb_image"] = args["semantic_image"]  # owned same native geometry
            frozen = args["frozen_world"]
            # AVO's registered yaw-only geometry refuses tilted scoreable people.
            # Tilt remains covered by producer tests; this composition is upright.
            actors = TS.Actors((TS.actor(1, "walker.pedestrian.0001", 20., -3.),
                                TS.actor(2, "vehicle.test", 20., 4.)))
            frozen.get_actors = lambda: actors
            population = dict(args["dynamic_snapshot_evidence"], registered_population_ids=[1, 2],
                population={"registered_actors": [1, 2]}, population_sha256=B.sha({"registered_actors": [1, 2]}))
            frozen.population_evidence = population
            with B.install_complete_gt(attempt_dir=attempt, max_frames=2) as installed:
                shutil.copytree(root/"catalogue", installed["catalogue"].directory)
                class Base:
                    def _ground_truth(self, **kwargs): return []
                cls = LG.build_live_gt_collector(Base)
                obj = cls.__new__(cls)
                obj._complete_producer = None
                obj._quality_scene_lock = threading.Lock()
                obj._quality_scenes = {100: frozen}
                obj.live = SimpleNamespace(_run4_identity={100: {"reward_requested": True}}, last_datagram={100: 1},
                    identity_for_frame=lambda frame: args["identity"])
                obj.ego = SimpleNamespace(id=999)
                obj.edge_evidence_dir = root/"edge"; obj.edge_evidence_dir.mkdir()
                obj._avo_results, obj._avo_rejected_results = [], []
                obj._avo_sync = None
                obj._avo_sources = mock.Mock(return_value=(args["rgb"], args["semantic_image"], args["depth_image"]))
                kwargs = dict(frame_id=100, timestamp=5., camera_matrix=np.eye(4), camera_inverse=np.eye(4),
                              radar_points={"world_xyz": args["radar_world_xyz"]})
                obj._ground_truth(**kwargs)
                obj._avo_sources.assert_called_once_with(100)
                self.assertEqual(len(obj._avo_results), 1)
                record = obj._avo_results[0]
                restored = T.verify_full_inputs(obj.edge_evidence_dir, expected_identity=args["identity"],
                    expected_manifest_sha256=record["complete_target"]["manifest_sha256"])
                self.assertTrue(restored.native_inputs_verified)
                result = LG.preserve_avo_evidence(obj._avo_results, root/"preserved")
                self.assertTrue(result["complete_targets"]["complete"])
                self.assertTrue((root/"preserved"/"complete_targets"/"MANIFEST.json").is_file())
                obj.live.last_datagram = {}
                with self.assertRaisesRegex(Exception, "before last datagram"):
                    obj._ground_truth(**kwargs)
                obj._avo_sources.assert_called_once_with(100)


if __name__ == "__main__":
    unittest.main()
