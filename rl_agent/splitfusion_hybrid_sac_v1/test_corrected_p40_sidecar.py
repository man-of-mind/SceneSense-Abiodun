"""Focused tests for the read-only corrected P40 sidecar."""

from __future__ import annotations

import hashlib
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from . import corrected_p40_sidecar as p40


class CorrectedP40Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.episode = "episode_fixture"
        self.relative = "radar_points/sample.npz"
        self.path = (
            self.root
            / p40.ROUTE_B_ROOT_RELPATH
            / self.episode
            / self.relative
        )
        self.path.parent.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(
        self,
        ranges: np.ndarray,
        offsets: np.ndarray,
        *,
        include_ranges: bool = True,
        include_offsets: bool = True,
    ) -> None:
        fields = {}
        if include_ranges:
            fields["original_range_m"] = ranges
        if include_offsets:
            fields["sweep_offset"] = offsets
        np.savez(self.path, **fields)

    def _row(
        self,
        *,
        original_p40=None,
        original_valid: bool = False,
        original_status: str = "InvalidRadarRangesError",
        raw: int,
        valid: int,
        rejected: int,
        source_sha256: str | None = None,
        sample_id: str = "sample",
        split: str = "fit",
    ) -> dict:
        digest = source_sha256 or hashlib.sha256(self.path.read_bytes()).hexdigest()
        return {
            "sample_id": sample_id,
            "episode_id": self.episode,
            "grid_split": split,
            "frame_id": 10,
            "source_paths": {"radar_points_path": self.relative},
            "source_sha256": {"radar_points_path": digest},
            "radar_p40": original_p40,
            "radar_p40_valid": original_valid,
            "radar_p40_status": original_status,
            "current_sweep_raw_returns": raw,
            "current_sweep_valid_returns": valid,
            "current_sweep_rejected_returns": rejected,
        }


class RecordDerivationTest(CorrectedP40Fixture):
    def test_declared_filter_and_exact_formula(self) -> None:
        ranges = np.asarray([10.0, 20.0, 40.0, 120.0, 121.0, np.nan], dtype=np.float32)
        offsets = np.zeros(6, dtype=np.uint8)
        self._write(ranges, offsets)
        record = p40._derive_record(
            self.root, self._row(raw=6, valid=4, rejected=2)
        )
        self.assertEqual(record.current_sweep_raw_returns, 6)
        self.assertEqual(record.current_sweep_valid_returns, 4)
        self.assertEqual(record.current_sweep_rejected_returns, 2)
        self.assertEqual(record.current_sweep_rejected_fraction, 1.0 / 3.0)
        self.assertEqual(record.corrected_p40, (0.75 + 0.5 + 0.0 + 0.0) / 4.0)
        self.assertEqual(record.correction_classification, p40.REPAIR_CLASS_FILTERED)
        self.assertIsNone(record.original_p40)

    def test_only_current_sweep_is_used(self) -> None:
        ranges = np.asarray([10.0, 20.0, np.nan, -1.0], dtype=np.float32)
        offsets = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        self._write(ranges, offsets)
        expected = (0.75 + 0.5) / 2.0
        record = p40._derive_record(
            self.root,
            self._row(
                original_p40=expected,
                original_valid=True,
                original_status="VALID",
                raw=2,
                valid=2,
                rejected=0,
            ),
        )
        self.assertEqual(record.corrected_p40, expected)
        self.assertEqual(record.correction_classification, p40.REPAIR_CLASS_AUDIT)

    def test_malformed_range_array_fails_closed(self) -> None:
        self._write(
            np.asarray([[10.0, 20.0]], dtype=np.float32),
            np.asarray([0, 0], dtype=np.uint8),
        )
        with self.assertRaisesRegex(p40.SourceRadarError, "malformed radar provenance"):
            p40._derive_record(self.root, self._row(raw=2, valid=2, rejected=0))

    def test_missing_range_field_fails_closed(self) -> None:
        self._write(
            np.asarray([10.0], dtype=np.float32),
            np.asarray([0], dtype=np.uint8),
            include_ranges=False,
        )
        with self.assertRaisesRegex(p40.SourceRadarError, "lacks sweep_offset"):
            p40._derive_record(self.root, self._row(raw=1, valid=1, rejected=0))

    def test_empty_current_sweep_is_not_zero_imputed(self) -> None:
        self._write(
            np.asarray([10.0, 20.0], dtype=np.float32),
            np.asarray([1, 1], dtype=np.uint8),
        )
        with self.assertRaisesRegex(p40.SourceRadarError, "missing/empty"):
            p40._derive_record(self.root, self._row(raw=0, valid=0, rejected=0))

    def test_no_valid_current_return_is_not_zero_imputed(self) -> None:
        self._write(
            np.asarray([121.0, np.nan], dtype=np.float32),
            np.asarray([0, 0], dtype=np.uint8),
        )
        with self.assertRaisesRegex(p40.SourceRadarError, "no range-valid returns"):
            p40._derive_record(self.root, self._row(raw=2, valid=0, rejected=2))

    def test_source_hash_drift_fails_before_derivation(self) -> None:
        self._write(
            np.asarray([10.0], dtype=np.float32),
            np.asarray([0], dtype=np.uint8),
        )
        row = self._row(raw=1, valid=1, rejected=0)
        self._write(
            np.asarray([20.0], dtype=np.float32),
            np.asarray([0], dtype=np.uint8),
        )
        with self.assertRaisesRegex(p40.SourceRadarError, "source hash drift"):
            p40._derive_record(self.root, row)

    def test_source_path_escape_is_refused(self) -> None:
        with self.assertRaisesRegex(p40.SourceRadarError, "unsafe radar source path"):
            p40._safe_source_path(self.root, self.episode, "../foreign.npz")

    def test_manifest_count_drift_is_refused(self) -> None:
        self._write(
            np.asarray([10.0, 121.0], dtype=np.float32),
            np.asarray([0, 0], dtype=np.uint8),
        )
        with self.assertRaisesRegex(p40.SourceRadarError, "count drift"):
            p40._derive_record(self.root, self._row(raw=2, valid=2, rejected=0))

    def test_original_valid_value_must_rederive_exactly(self) -> None:
        self._write(
            np.asarray([10.0], dtype=np.float32),
            np.asarray([0], dtype=np.uint8),
        )
        with self.assertRaisesRegex(p40.SourceRadarError, "exactly rederive"):
            p40._derive_record(
                self.root,
                self._row(
                    original_p40=0.1,
                    original_valid=True,
                    original_status="VALID",
                    raw=1,
                    valid=1,
                    rejected=0,
                ),
            )


class InventoryAndCanonicalTest(CorrectedP40Fixture):
    def test_strict_json_rejects_duplicate_and_nonfinite_values(self) -> None:
        duplicate = self.root / "duplicate.json"
        duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
        with self.assertRaisesRegex(p40.BundleBindingError, "cannot parse strict"):
            p40._strict_json_object(duplicate, "fixture")
        nonfinite = self.root / "nonfinite.json"
        nonfinite.write_text('{"a":NaN}', encoding="utf-8")
        with self.assertRaisesRegex(p40.BundleBindingError, "cannot parse strict"):
            p40._strict_json_object(nonfinite, "fixture")

    def test_metadata_semantics_bind_range_and_filter_authority(self) -> None:
        document = {
            "experiment_id": self.episode,
            "radar": {
                "range_m": 120.0,
                "configured_attributes": {"range": "120.0"},
                "raw_range_note": p40.EXPECTED_RAW_RANGE_NOTE,
            },
        }
        p40._validate_radar_metadata_document(document, self.episode)
        wrong_range = {
            **document,
            "radar": {**document["radar"], "range_m": 80.0},
        }
        with self.assertRaisesRegex(p40.BundleBindingError, "not 120 m"):
            p40._validate_radar_metadata_document(wrong_range, self.episode)
        wrong_rule = {
            **document,
            "radar": {**document["radar"], "raw_range_note": "filter however you like"},
        }
        with self.assertRaisesRegex(p40.BundleBindingError, "authority drift"):
            p40._validate_radar_metadata_document(wrong_rule, self.episode)

    def test_duplicate_sample_ids_fail_before_second_read(self) -> None:
        self._write(
            np.asarray([10.0], dtype=np.float32),
            np.asarray([0], dtype=np.uint8),
        )
        row = self._row(
            original_p40=0.75,
            original_valid=True,
            original_status="VALID",
            raw=1,
            valid=1,
            rejected=0,
        )
        with self.assertRaises(p40.DuplicateSampleIdError):
            p40._derive_records(
                self.root,
                [row, dict(row)],
                expected_total=2,
                expected_split_counts={"fit": 2},
            )

    def test_immutable_provider_and_canonical_determinism(self) -> None:
        self._write(
            np.asarray([10.0], dtype=np.float32),
            np.asarray([0], dtype=np.uint8),
        )
        record = p40._derive_record(
            self.root,
            self._row(
                original_p40=0.75,
                original_valid=True,
                original_status="VALID",
                raw=1,
                valid=1,
                rejected=0,
            ),
        )
        first = p40.CorrectedP40Sidecar.create(
            [record], {"selection_manifest_file_sha256": "a" * 64}
        )
        second = p40.CorrectedP40Sidecar.create(
            [record], {"selection_manifest_file_sha256": "a" * 64}
        )
        self.assertEqual(first.sample_ids, ("sample",))
        self.assertEqual(first.lookup("sample"), 0.75)
        self.assertIs(first.record("sample"), record)
        self.assertEqual(first.binding_sha256, second.binding_sha256)
        self.assertEqual(first.canonical_report_bytes(), second.canonical_report_bytes())
        with self.assertRaises(TypeError):
            first.source_binding["new"] = "forbidden"  # type: ignore[index]
        with self.assertRaises(p40.UnknownSampleIdError):
            first.lookup("foreign")

    def test_create_only_writer_refuses_overwrite(self) -> None:
        (self.root / p40.DEFAULT_BUNDLE_RELPATH).mkdir(parents=True)
        (self.root / p40.ROUTE_B_ROOT_RELPATH).mkdir(parents=True, exist_ok=True)
        output = self.root / "report.json"
        digest = p40._write_create_only(output, b"{}\n", root=self.root)
        self.assertEqual(digest, hashlib.sha256(b"{}\n").hexdigest())
        with self.assertRaisesRegex(p40.CorrectedP40Error, "already exists"):
            p40._write_create_only(output, b"changed", root=self.root)
        self.assertEqual(output.read_bytes(), b"{}\n")

    def test_output_guard_creates_nothing_inside_immutable_trees(self) -> None:
        bundle = self.root / p40.DEFAULT_BUNDLE_RELPATH
        route_b = self.root / p40.ROUTE_B_ROOT_RELPATH
        bundle.mkdir(parents=True)
        route_b.mkdir(parents=True, exist_ok=True)
        with self.assertRaisesRegex(p40.CorrectedP40Error, "immutable evidence"):
            p40._write_create_only(bundle, b"{}\n", root=self.root)
        self.assertTrue(bundle.is_dir())
        forbidden_parent = bundle / "never_created"
        with self.assertRaisesRegex(p40.CorrectedP40Error, "immutable evidence"):
            p40._write_create_only(
                forbidden_parent / "report.json", b"{}\n", root=self.root
            )
        self.assertFalse(forbidden_parent.exists())

        alias = self.root / "route_alias"
        alias.symlink_to(route_b, target_is_directory=True)
        symlink_forbidden_parent = route_b / "also_never_created"
        with self.assertRaisesRegex(p40.CorrectedP40Error, "immutable evidence"):
            p40._write_create_only(
                alias / "also_never_created" / "report.json",
                b"{}\n",
                root=self.root,
            )
        self.assertFalse(symlink_forbidden_parent.exists())


class ExactCompletedBundleAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = p40.repository_root()
        bundle = root / p40.DEFAULT_BUNDLE_RELPATH
        if not bundle.is_dir():
            raise unittest.SkipTest("exact completed bundle is not present")
        cls.sidecar = p40.load_exact_corrected_p40_sidecar(root=root)

    def test_full_inventory_and_provider_contract(self) -> None:
        sidecar = self.sidecar
        self.assertEqual(len(sidecar.records), 768)
        self.assertEqual(len(sidecar.sample_ids), 768)
        self.assertEqual(len(set(sidecar.sample_ids)), 768)
        self.assertTrue(all(math.isfinite(sidecar.lookup(key)) for key in sidecar.sample_ids))
        self.assertTrue(all(0.0 <= sidecar.lookup(key) <= 1.0 for key in sidecar.sample_ids))
        split_counts = {
            split: sum(row.grid_split == split for row in sidecar.records)
            for split in ("fit", "held_scene")
        }
        self.assertEqual(split_counts, {"fit": 512, "held_scene": 256})

    def test_full_audit_explains_every_original_status(self) -> None:
        counts = {
            name: sum(row.correction_classification == name for row in self.sidecar.records)
            for name in (p40.REPAIR_CLASS_FILTERED, p40.REPAIR_CLASS_AUDIT)
        }
        self.assertEqual(counts[p40.REPAIR_CLASS_FILTERED], 729)
        self.assertEqual(counts[p40.REPAIR_CLASS_AUDIT], 39)
        self.assertEqual(sum(counts.values()), 768)
        self.assertEqual(
            sum(row.current_sweep_rejected_returns for row in self.sidecar.records),
            181_271,
        )

    def test_exact_bundle_binding_is_pinned_not_self_authorized(self) -> None:
        self.assertEqual(
            self.sidecar.source_binding["quality_database_file_sha256"],
            p40.EXACT_COMPLETED_GRID_BINDING.quality_database_file_sha256,
        )
        self.assertEqual(
            self.sidecar.source_binding["selected_npz_files_hash_verified"], 768
        )
        self.assertEqual(
            self.sidecar.source_binding["repair_implementation_relative_path"],
            p40.IMPLEMENTATION_RELPATH,
        )
        self.assertEqual(
            self.sidecar.source_binding["repair_implementation_sha256"],
            hashlib.sha256(Path(p40.__file__).read_bytes()).hexdigest(),
        )
        authority = self.sidecar.source_binding["repair_authority_metadata"]
        self.assertEqual(
            set(authority["episodes"]), set(p40.EXPECTED_EPISODE_METADATA_SHA256)
        )
        for episode_id, expected_sha256 in p40.EXPECTED_EPISODE_METADATA_SHA256.items():
            episode = authority["episodes"][episode_id]
            self.assertEqual(episode["metadata_sha256"], expected_sha256)
            self.assertEqual(episode["configured_range_m"], 120.0)
            self.assertEqual(episode["raw_range_note"], p40.EXPECTED_RAW_RANGE_NOTE)
        with self.assertRaises(TypeError):
            authority["episodes"][next(iter(authority["episodes"]))][
                "configured_range_m"
            ] = 80.0
        report = self.sidecar.canonical_report()
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["binding_sha256"], self.sidecar.binding_sha256)
        self.assertTrue(report["no_zero_imputation"])


if __name__ == "__main__":
    unittest.main()
