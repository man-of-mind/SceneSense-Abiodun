"""Offline parity and integrity tests for the neutral tail evidence codec."""

from __future__ import annotations

import dataclasses
import hashlib
import unittest

import numpy as np

from rl_agent.splitfusion_quality_feedback_probe_v1 import scoring
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import (
    b_edge_process_v1 as E,
)
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import (
    tail_evidence_codec_v2 as C,
)


class NeutralOwnershipParityTest(unittest.TestCase):
    def test_rows_and_mask_are_bit_exact_to_historical_helpers(self) -> None:
        rows = [{"class_name": "person", "world_x": 1.25, "count": 2}]
        mask = np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.int64)[:, ::-1]
        self.assertEqual(C.immutable_rows(rows), scoring.immutable_rows(rows))
        actual = C.immutable_mask(mask)
        expected = scoring.immutable_mask(mask)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.flags.c_contiguous, expected.flags.c_contiguous)
        self.assertEqual(actual.flags.writeable, expected.flags.writeable)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_round_trip_is_exact_and_identity_bound(self) -> None:
        identity = E._fake_identity()
        mask = np.asarray([[0, 1], [2, 0]], dtype=np.uint8)
        payload = C.encode(identity=identity,
                           objects=({"class_name": "person"},),
                           semantic_mask=mask)
        decoded = C.decode(payload, expected_identity=identity)
        self.assertEqual(decoded.identity, identity)
        self.assertEqual(decoded.objects, ({"class_name": "person"},))
        np.testing.assert_array_equal(decoded.semantic_mask, mask)
        with self.assertRaisesRegex(C.TailEvidenceCodecError, "identity"):
            C.decode(payload, expected_identity=dataclasses.replace(
                identity, q_e4=identity.q_e4 + 1))

    def test_digest_and_class_corruption_fail_closed(self) -> None:
        identity = E._fake_identity()
        with self.assertRaisesRegex(C.TailEvidenceCodecError, "outside"):
            C.encode(identity=identity, objects=(),
                     semantic_mask=np.asarray([[3]], dtype=np.uint8))
        payload = bytearray(C.encode(
            identity=identity, objects=(),
            semantic_mask=np.asarray([[0]], dtype=np.uint8)))
        payload[-1] ^= 1
        with self.assertRaisesRegex(C.TailEvidenceCodecError, "digest"):
            C.decode(bytes(payload))


if __name__ == "__main__":
    unittest.main()
