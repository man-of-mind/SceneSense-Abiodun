from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from . import actor_timing_qualification as subject


class ActorTimingQualificationTests(unittest.TestCase):
    def test_representative_states_match_feature_contract(self) -> None:
        rows = subject.representative_states()
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(len(row) == 21 for row in rows))
        self.assertEqual(sum(rows[1][4:16]), 1.0)
        self.assertEqual(sum(rows[2][4:16]), 1.0)

    def test_small_cpu_measurement_includes_nonzero_actor_reserve(self) -> None:
        before = torch.cuda.is_initialized()
        result = subject.measure(
            subject.ActorTimingConfig(warmup=2, iterations=12, threads=1)
        )
        self.assertEqual(result["evidence_class"], subject.EVIDENCE_CLASS)
        self.assertGreater(result["timing_ns"]["p99"], 0)
        self.assertGreaterEqual(
            result["modeled_actor_reserve_ns"],
            subject.MINIMUM_MODELED_RESERVE_NS,
        )
        self.assertEqual(torch.cuda.is_initialized(), before)

    def test_create_only_writer_and_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence"
            result = {"schema": subject.SCHEMA, "value": 1}
            result_path, manifest_path = subject.write_create_only(output, result)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["result"]["sha256"], subject._sha256(result_path)
            )
            with self.assertRaises(FileExistsError):
                subject.write_create_only(output, result)


if __name__ == "__main__":
    unittest.main()
