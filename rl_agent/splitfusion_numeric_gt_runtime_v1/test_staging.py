"""Source staging must preserve exact native data and bind only the later action."""
import copy
import os
import time
import unittest

from .staging import encode_stage, decode_stage, bind_action
from .process import Engine
from .live_clock import read_boot_id
from .test_source_codec import fixture


class StagingTests(unittest.TestCase):
    def test_action_neutral_snapshot_rebinds_native_source_without_rounding(self):
        source, stamp = fixture()
        selected_identity = dict(source["identity"], action_id=7, profile_id="mode7")
        chunks, metadata = encode_stage(source, stamp)
        neutral = decode_stage(b"".join(chunks), metadata)
        self.assertIsNone(neutral["identity"]["action_id"])
        self.assertIsNone(neutral["identity"]["profile_id"])
        selected = bind_action(neutral, selected_identity)
        self.assertEqual(selected["identity"], selected_identity)
        for name in ("camera_matrix", "camera_inverse", "radar_world_xyz"):
            self.assertEqual(selected[name].tobytes(), source[name].tobytes())
        for name in ("semantic_image", "depth_image"):
            self.assertEqual(bytes(selected[name].raw_data), bytes(source[name].raw_data))

    def test_foreign_or_action_bearing_staged_packet_refused(self):
        source, stamp = fixture()
        chunks, extras = encode_stage(source, stamp)
        raw = b"".join(chunks)
        decoded = decode_stage(raw, extras)
        for field, value in (("frame_id", source["identity"]["frame_id"] + 1),
                             ("capture_timestamp_ns", source["identity"]["capture_timestamp_ns"] + 1),
                             ("stream_id", "foreign")):
            changed = dict(source["identity"], **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                bind_action(decoded, changed)
        tampered = copy.deepcopy(extras)
        tampered["capture_template_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            decode_stage(raw, tampered)

    def test_stage_receipt_is_stamped_by_accepting_process_after_decode(self):
        source, stamp = fixture()
        chunks, metadata = encode_stage(source, stamp)
        engine = Engine.__new__(Engine)
        engine.role, engine.staged_source, engine.tokens = "producer", None, {}
        before = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        result = engine.call("STAGE_SOURCE", dict(job=1, source_packet=b"".join(chunks),
                                                  stage_extras=metadata))
        after = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        self.assertLessEqual(before, result["source_stage_received_raw_ns"])
        self.assertLessEqual(result["source_stage_received_raw_ns"], after)
        self.assertEqual(result["source_stage_owner_pid"], os.getpid())
        self.assertEqual(result["source_stage_boot_id"], read_boot_id())
        self.assertEqual(result["source_bytes"], len(b"".join(chunks)))
        with self.assertRaises(ValueError):
            engine.call("STAGE_SOURCE", dict(job=2, source_packet=b"".join(chunks),
                                                   stage_extras=metadata))


if __name__ == "__main__":
    unittest.main()
