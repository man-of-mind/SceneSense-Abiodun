"""Focused tests: the live-view taps never block, never raise into the pipeline and never change retained behaviour."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent.splitfusion_live_view_tap_v1 import tap as T  # noqa: E402


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


class BoundedTapTests(unittest.TestCase):
    def test_full_queue_drops_without_blocking(self):
        gate = threading.Event()

        def slow(seq, tap_ns, item, tap):
            gate.wait(5)             # a stalled disk / writer
            return dict(seq=seq, item=item)
        with tempfile.TemporaryDirectory() as d:
            tap = T.BoundedJsonlTap(Path(d) / "x.jsonl", role="t", build=slow, capacity=4).start()
            worst = 0.0
            for i in range(50):
                t0 = time.perf_counter()
                tap.offer(i)
                worst = max(worst, time.perf_counter() - t0)
            self.assertLess(worst, 0.005)
            counts = tap.snapshot_counts()
            self.assertEqual(counts["offered"], 50)
            self.assertGreaterEqual(counts["dropped_queue_full"], 45)
            gate.set()
            status = tap.close()
            self.assertEqual(status["counts"]["written"] + status["counts"]["dropped_queue_full"], 50)
            self.assertTrue((Path(d) / "t_tap_status.json").is_file())
            self.assertEqual(len(lines(Path(d) / "x.jsonl")), status["counts"]["written"])

    def test_build_error_and_existing_file_never_raise(self):
        with tempfile.TemporaryDirectory() as d:
            def bad(*_):
                raise ValueError("boom")
            tap = T.BoundedJsonlTap(Path(d) / "a.jsonl", role="a", build=bad).start()
            self.assertTrue(tap.offer(1))
            status = tap.close()
            self.assertEqual(status["counts"]["build_errors"], 1)
            (Path(d) / "b.jsonl").write_text("")
            tap = T.BoundedJsonlTap(Path(d) / "b.jsonl", role="b", build=bad).start()   # create-only refused
            self.assertFalse(tap.offer(1))
            self.assertEqual(tap.close()["counts"]["dropped_not_running"], 1)
            self.assertEqual(tap.offer(2), False)                                          # after close


class MapTapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cell = Path(self.tmp.name) / "cell"
        (cell / "direct_edge_map").mkdir(parents=True)
        history = {}

        class Service:
            def __init__(self, **kwargs):
                self.ingest_csv = cell / "direct_edge_map" / "direct_map_ingest.csv"

            def close(self):
                return "closed"

        def inner(document, ingest_at):
            if document.get("fail"):
                raise RuntimeError("install refused")
            history[(document["stream_id"], document["frame_id"])] = {"objects": [], "frame_id": document["frame_id"]}
            return {"install_timestamp": 12.5}
        server = types.SimpleNamespace(install_under_state_lock=inner, DirectMapIngestService=Service,
                                       baseline=types.SimpleNamespace(installed_frame_history=history))
        fake = types.ModuleType("phase6_map_server_v2")
        fake.install = lambda contract: server
        package = "rl_agent.splitfusion_hybrid_sac_live_route_b_v2"
        self.saved = sys.modules.get(package + ".phase6_map_server_v2")
        __import__(package)
        sys.modules[package + ".phase6_map_server_v2"] = fake
        setattr(sys.modules[package], "phase6_map_server_v2", fake)
        self.fake, self.cell = fake, cell

    def tearDown(self):
        package = "rl_agent.splitfusion_hybrid_sac_live_route_b_v2"
        if self.saved is not None:
            sys.modules[package + ".phase6_map_server_v2"] = self.saved
            setattr(sys.modules[package], "phase6_map_server_v2", self.saved)
        self.tmp.cleanup()

    def test_offer_only_after_successful_install(self):
        state = T.install_map_tap()
        with self.assertRaises(RuntimeError):
            T.install_map_tap()                                         # never double-wrapped
        server = self.fake.install(None)
        service = server.DirectMapIngestService()
        record = dict(world_x=1.0, yaw_sin=0.5, yaw_cos=0.8660254, score=0.3, class_name="vehicle", extra="dropped")
        result = server.install_under_state_lock(dict(stream_id="s", frame_id=7, records=[record],
                                                      capture_timestamp_ns=5), 1.0)
        self.assertEqual(result, {"install_timestamp": 12.5})           # returned unchanged
        with self.assertRaises(RuntimeError):
            server.install_under_state_lock(dict(stream_id="s", frame_id=8, fail=True), 1.0)
        self.assertEqual(service.close(), "closed")
        rows = lines(self.cell / "live_view" / "map_installs.jsonl")
        self.assertEqual([r["identity"]["frame_id"] for r in rows], [7])
        self.assertEqual(rows[0]["record"], {"objects": [], "frame_id": 7})
        self.assertEqual(rows[0]["producer_records"][0]["yaw_sin"], 0.5)
        self.assertNotIn("extra", rows[0]["producer_records"][0])
        self.assertIsNone(state["error"])
        status = json.loads((self.cell / "live_view" / "map_tap_status.json").read_text())
        self.assertEqual(status["counts"]["written"], 1)


class PoseTapTests(unittest.TestCase):
    def test_retained_archive_unchanged_and_ue_documents_copied(self):
        from rl_agent.splitfusion_live_parity_runtime_v1 import pose as POSE
        original = POSE.PoseArchive.add
        with tempfile.TemporaryDirectory() as d:
            with T.install_ue_pose_tap(d) as tap:
                ue = POSE.PoseArchive(Path(d) / "pose_archive", side="ue", max_frames=4)
                edge = POSE.PoseArchive(Path(d) / "edge_archive", side="edge", max_frames=4)
                ue.add("OWNED_SOURCE", source={"frame_id": 1}, source_sha256="ab")
                ue.add("UE_PREPARATION_DONE", frame_id=1, stream_id="s", sent=True)
                ue.add("UE_WIRE", context={"frame_id": 1}, source_sha256="ab")
                edge.add("EDGE_WIRE", context={"frame_id": 1})
                for _ in range(3):
                    ue.add("UE_WIRE", context={"frame_id": 2}, source_sha256="cd")
                with self.assertRaises(Exception):                       # retained budget error still raised
                    ue.add("UE_WIRE", context={"frame_id": 3}, source_sha256="ef")
                self.assertEqual(len(ue._events), 6)
            self.assertIs(POSE.PoseArchive.add, original)
            rows = lines(Path(d) / "live_view" / "ue_pose.jsonl")
            self.assertEqual([r["kind"] for r in rows], ["OWNED_SOURCE", "UE_WIRE"] + ["UE_WIRE"] * 3)
            self.assertEqual(rows[0]["source"], {"frame_id": 1})
            self.assertEqual(tap.status["counts"]["written"], 5)


if __name__ == "__main__":
    unittest.main()
