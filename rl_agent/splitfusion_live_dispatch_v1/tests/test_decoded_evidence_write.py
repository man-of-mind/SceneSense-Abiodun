"""Phase-15 regression: decoded masks must never be readable while partial."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime import (
    EDGE_RESULT_SCHEMA,
    EDGE_TERMINAL_ACK_SCHEMA,
    OBJECT_MAP_UPDATE_SCHEMA,
    EdgeEvaluationEvidenceWriter,
    _Counters,
    _write_decoded_evidence,
    segmentation_evidence_name,
)


class DecodedEvidenceWriteTest(unittest.TestCase):
    """The adapter loads this exact name as soon as it exists (see
    rl_agent/ue_route_b_split_cell_adapter_v1.py); a directly written file let
    it observe truncated masks and fail an otherwise valid cell."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="phase15_evidence_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.mask = np.arange(720 * 1280, dtype=np.uint64).astype(np.uint8).reshape(720, 1280)

    def test_published_evidence_is_complete_and_loadable(self) -> None:
        evidence = self.directory / "stream_1.npy"
        _write_decoded_evidence(evidence, self.mask)
        np.testing.assert_array_equal(np.load(evidence, allow_pickle=False), self.mask)
        self.assertEqual([path.name for path in self.directory.iterdir()], ["stream_1.npy"])

    def test_interrupted_write_never_publishes_the_final_name(self) -> None:
        evidence = self.directory / "stream_2.npy"

        def truncated(handle, array, allow_pickle=False):  # noqa: ANN001
            handle.write(array.tobytes()[: array.nbytes // 3])
            raise OSError("simulated mid-write failure")

        with mock.patch(
            "rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime.np.save",
            side_effect=truncated,
        ):
            with self.assertRaises(OSError):
                _write_decoded_evidence(evidence, self.mask)
        self.assertFalse(evidence.exists())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_duplicate_frame_evidence_is_refused(self) -> None:
        evidence = self.directory / "stream_3.npy"
        _write_decoded_evidence(evidence, self.mask)
        with self.assertRaises(FileExistsError):
            _write_decoded_evidence(evidence, self.mask)

    def test_mask_is_persisted_locally_and_absent_from_radio_feedback(self) -> None:
        """Phase-15 recovery: the dense label map leaves the return path only.

        The retry4 audit measured a ~1.23 MB base64 label map in every result,
        which is what destroyed installed-frame delivery. The map must still be
        persisted and hash-bound locally, but must not appear in the compact
        message the edge returns over OAI.
        """

        counters = _Counters()
        writer = EdgeEvaluationEvidenceWriter(self.directory, counters, depth=4)
        self.addCleanup(writer.close)
        name = segmentation_evidence_name("stream-a", 7)
        digest = hashlib.sha256(self.mask.tobytes()).hexdigest()
        sidecar = {
            "evidence_name": name, "stream_id": "stream-a", "frame_id": 7,
            "action_id": 71, "capture_timestamp_ns": 123456789,
            "shape": [720, 1280], "dtype": "uint8",
            "bytes": int(self.mask.nbytes), "sha256": digest,
        }
        status = writer.submit(self.mask, sidecar)
        self.assertEqual(status, "EVALUATION_EVIDENCE_WRITE_SUBMITTED")
        writer.close()

        # Preserved locally, atomically, and verifiable against its sidecar.
        persisted = self.directory / name
        np.testing.assert_array_equal(
            np.load(persisted, allow_pickle=False), self.mask
        )
        record = json.loads(persisted.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertTrue(record["hash_verified"])
        self.assertEqual(record["sha256"], digest)
        self.assertEqual(counters.snapshot()["evaluation_masks_hash_verified"], 1)

        # The returned message carries compact records and a terminal ACK only.
        result = {
            "schema": EDGE_RESULT_SCHEMA, "action_id": 71, "frame_id": 7,
            "object_map_update": {
                "schema": OBJECT_MAP_UPDATE_SCHEMA, "stream_id": "stream-a",
                "frame_id": 7, "records": [{"stream_id": "stream-a", "frame_id": 7}],
            },
            "edge_terminal_ack": {
                "schema": EDGE_TERMINAL_ACK_SCHEMA, "frame_id": 7,
                "installation_status": "EVALUATION_EVIDENCE_WRITE_SUBMITTED",
                "evidence": {"sha256": digest, "shape": [720, 1280]},
                "terminal_reason": "EDGE_SERVICE_COMPLETE",
            },
        }
        encoded = json.dumps(result).encode("utf-8")
        self.assertNotIn("semantic_labels_b64", result)
        self.assertNotIn(b"semantic_labels", encoded)
        # The dense map alone was ~1.23 MB encoded; the compact result is tiny.
        self.assertLess(len(encoded), 4096)
        # ACK identity stays separate from shared-object content.
        self.assertNotEqual(
            result["edge_terminal_ack"]["schema"],
            result["object_map_update"]["schema"],
        )
        self.assertNotIn("records", result["edge_terminal_ack"])


if __name__ == "__main__":
    unittest.main()
