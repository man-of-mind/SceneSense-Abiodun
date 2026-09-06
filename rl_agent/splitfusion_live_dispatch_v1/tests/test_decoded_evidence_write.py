"""Phase-15 regression: decoded masks must never be readable while partial."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime import (
    _write_decoded_evidence,
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


if __name__ == "__main__":
    unittest.main()
