"""Focused live-adapter checks: bounded fan-out, exact heading join, ordering/session guards, history refusal."""
import json
import tempfile
import time
import unittest
from pathlib import Path

from spatial_map_coop.replay_ui_v1 import live_feed_v1 as L


def record(frame=5, n=2):
    objects = [dict(id=f"s:{frame}:{i}", source_stream_id="s", frame_id=frame, type="Vehicle", score=0.5 + i,
                    location=dict(x=1.0 + i, y=2.0, z=0.0), dimensions=dict(length=4.0, width=2.0, height=1.5))
               for i in range(n)]
    return dict(stream_id="s", frame_id=frame, objects=objects, capture_timestamp=1.0, install_timestamp=2.0)


def producer(rec):
    return [dict(stream_id="s", frame_id=rec["frame_id"], capture_timestamp_ns=7, class_name="vehicle",
                 score=o["score"], world_x=o["location"]["x"], world_y=2.0, world_z=0.0, size_x=4.0, size_y=2.0,
                 size_z=1.5, yaw_sin=1.0, yaw_cos=0.0) for o in rec["objects"]]


POSE = dict(capture_ns="7", binding=["r", "c", "s", 5, "7"])


class JoinTests(unittest.TestCase):
    def test_exact_join_gives_producer_world_yaw(self):
        rec = record()
        joined, error = L.join_producer(rec, producer(rec), POSE)
        self.assertIsNone(error)
        self.assertEqual(joined["s:5:1"]["heading"]["yaw_deg"], 90.0)
        self.assertTrue(joined["s:5:0"]["heading"]["valid"])

    def test_any_mismatch_refuses_the_whole_frame(self):
        rec = record()
        bad = producer(rec)
        bad[1]["world_x"] += 1e-9
        self.assertEqual(L.join_producer(rec, bad, POSE), ({}, "PRODUCER_RECORD_MISMATCH"))
        self.assertEqual(L.join_producer(rec, [], POSE)[1], "NO_PRODUCER_RECORDS")
        self.assertEqual(L.join_producer(rec, producer(rec), None)[1], "NO_BOUND_POSE")

    def test_filtered_index_gap_joins_by_original_index(self):
        rec = record(n=3)
        prod = producer(rec)
        rec["objects"].pop(1)                    # baseline score filter leaves a gap in ids
        joined, error = L.join_producer(rec, prod, POSE)
        self.assertIsNone(error)
        self.assertEqual(sorted(joined), ["s:5:0", "s:5:2"])


class FanOutTests(unittest.TestCase):
    def test_stalled_client_drops_without_blocking_others(self):
        with tempfile.TemporaryDirectory() as d:
            feed = L.LiveFeed(Path(d), log_path=Path(d) / "log.jsonl")
            slow, fast = feed.connect("a"), feed.connect("b")
            t0 = time.perf_counter()
            for i in range(500):
                feed._broadcast(json.dumps({"i": i}))
                fast.pop(0)
            self.assertLess(time.perf_counter() - t0, 1.0)
            self.assertEqual(len(slow.queue), L.CLIENT_QUEUE)
            self.assertGreaterEqual(slow.dropped, 500 - L.CLIENT_QUEUE)
            self.assertEqual(fast.dropped, 0)
            self.assertEqual(feed.counts["client_drops"], slow.dropped)
            feed.close()

    def test_old_installs_are_history_and_other_sessions_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            feed = L.LiveFeed(Path(d), log_path=Path(d) / "log.jsonl")
            old = dict(schema=L.MAP_SCHEMA, record_available=True, identity=dict(frame_id=1),
                       install=dict(install_timestamp=time.time() - 5), record={})
            feed._install(old)
            self.assertEqual(feed.counts["stale_at_read"], 1)
            self.assertEqual(len(feed.pending), 0)
            feed.session = "r/c"
            other = dict(identity=dict(run_id="OLD", cell_id="c", capture_timestamp_ns=1, stream_id="s", frame_id=2),
                         record=dict(frame_id=2), install=dict(install_timestamp=time.time()))
            feed._emit(other, 0, 0)
            self.assertEqual(feed.counts["other_session"], 1)
            self.assertEqual(feed.counts["emitted"], 0)
            feed.close()

    def test_restarted_viewer_skips_existing_backlog(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.jsonl"
            path.write_text('{"a":1}\n{"a":2}\n')
            time.sleep(0.01)
            tail = L.Tail(path, started_ns=time.time_ns())
            self.assertEqual(tail.read(), [])
            with path.open("a") as f:
                f.write('{"a":3}\n{"a":')
            self.assertEqual(tail.read(), [{"a": 3}])          # partial line waits for its newline
            with path.open("a") as f:
                f.write('4}\n')
            self.assertEqual(tail.read(), [{"a": 4}])


if __name__ == "__main__":
    unittest.main()
