"""Pre-frame startup and finite handoff capacity; no services or GPU."""
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from . import ue_live as U
from .test_async_ue_integration import SlowClient, setup_bridge, planned
from .test_source_codec import fixture


class UePrestartTests(unittest.TestCase):
    def test_prestart_uses_actual_registry_without_fabricating_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            bridge = U.NumericUeHandoff(attempt_dir=Path(temporary), registration={})
            population = {7: dict(actor_id=7, type_id="walker.pedestrian.0001")}
            bridge.collector = NS(ego=NS(id=8), scene_source=NS(
                ego_id=8, _parity_initialized=True,
                _parity_registry={7: ("walker.pedestrian.0001", object())},
                _parity_population=population))
            calls = []
            def start(value):
                self.assertEqual(bridge._tickets, {})
                calls.append(value)
                bridge.client = NS(ready={"status": "READY"})
            with mock.patch.object(bridge, "_start_client", side_effect=start):
                bridge.prestart_from_scene()
                with self.assertRaisesRegex(U.NumericUeError, "pre-frame"):
                    bridge.prestart_from_scene()
            expected = U.gt_bridge.sha(dict(schema="splitfusion.live_exact_population.v1",
                ego_id=8, registered_actors=list(population.values())))
            self.assertEqual(calls, [dict(dynamic_snapshot_evidence=dict(
                registered_population_ids=[7], population_sha256=expected))])
            record = U.C._parse((bridge.directory / "PRODUCER_PRESTART.json").read_bytes())
            self.assertTrue(record["no_stage_source_sent"])
            self.assertTrue(record["no_frame_admitted"])
            self.assertLessEqual(record["started_raw_ns"], record["completed_raw_ns"])
            self.assertEqual(bridge._tickets, {})

    def test_incomplete_or_foreign_registry_refused_before_child_start(self):
        for initialized, population in ((False, {7: {"actor_id": 7}}),
                                        (True, {7: {"actor_id": 9}})):
            with self.subTest(initialized=initialized), tempfile.TemporaryDirectory() as temporary:
                bridge = U.NumericUeHandoff(attempt_dir=Path(temporary), registration={})
                bridge.collector = NS(ego=NS(id=8), scene_source=NS(ego_id=8,
                    _parity_initialized=initialized, _parity_registry={7: object()},
                    _parity_population=population))
                with mock.patch.object(bridge, "_start_client") as start:
                    with self.assertRaises(U.NumericUeError):
                        bridge.prestart_from_scene()
                    start.assert_not_called()

    def test_source_memory_limit_refuses_before_plan_or_ipc(self):
        source, stamp = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            client = SlowClient()
            bridge, _, runtime, _, _, frame = setup_bridge(Path(temporary), source, stamp, client)
            bridge.registration["numeric_limits"]["retained_bytes"] = 1
            with self.assertRaisesRegex(U.NumericUeError, "capacity exceeded"):
                bridge.stage_before_action(runtime, frame_id=frame, carla_timestamp=stamp,
                    capture_timestamp_ns=source["identity"]["capture_timestamp_ns"], ego_pose=(0.,)*6)
            self.assertEqual(bridge._tickets, {})
            self.assertEqual(client.calls, [])

    def test_active_job_bound_refuses_before_send_or_ipc(self):
        source, stamp = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            client = SlowClient()
            bridge, _, runtime, _, _, frame = setup_bridge(Path(temporary), source, stamp, client)
            bridge.stage_before_action(runtime, frame_id=frame, carla_timestamp=stamp,
                capture_timestamp_ns=source["identity"]["capture_timestamp_ns"], ego_pose=(0.,)*6)
            bridge._tickets[-1] = NS(plan_bytes=b"prior1", retired=False)
            bridge._tickets[-2] = NS(plan_bytes=b"prior2", retired=False)
            runtime.pipeline.decisions.append(dict(frame_id=frame,
                action_open=dict(domain="CLOCK_MONOTONIC_RAW", ns=U.raw())))
            with self.assertRaisesRegex(U.NumericUeError, "capacity exceeded"):
                bridge.on_plan(runtime, planned(frame))
            self.assertEqual(client.calls, [])
            self.assertIsNone(bridge._tickets[frame].plan_bytes)


if __name__ == "__main__":
    unittest.main()
