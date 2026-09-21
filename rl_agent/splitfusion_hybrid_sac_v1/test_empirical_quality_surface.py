"""Focused tests for the exact-grid empirical quality/payload surface."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import unittest
from dataclasses import fields
from pathlib import Path

from .corrected_p40_sidecar import load_exact_corrected_p40_sidecar
from .empirical_quality_surface import (
    BUNDLE_RELATIVE_PATH,
    DATABASE_FILE_SHA256,
    EXACT_GRID_ROW_EVIDENCE,
    MODELED_SAME_FRAME_EVIDENCE,
    EmpiricalQualitySurface,
    EvidenceBindingError,
    HiddenSurfaceRecord,
    InvalidSurfaceQuery,
    PolicySurfaceView,
    SplitAccessError,
    load_empirical_quality_surface,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


class EmpiricalQualitySurfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = _root()
        cls.bundle = cls.root / BUNDLE_RELATIVE_PATH
        cls.database = cls.bundle / "quality_rows.sqlite3"
        cls.before_stat = cls.database.stat()
        cls.sidecar = load_exact_corrected_p40_sidecar(root=cls.root)
        cls.surface = load_empirical_quality_surface(
            cls.sidecar, project_root=cls.root
        )
        cls.fit_record = next(
            record for record in cls.sidecar.records if record.grid_split == "fit"
        )
        cls.held_record = next(
            record for record in cls.sidecar.records if record.grid_split == "held_scene"
        )
        cls.readonly = sqlite3.connect(
            f"file:{cls.database}?mode=ro&immutable=1", uri=True
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.readonly.close()
        cls.surface.close()

    @classmethod
    def row(cls, sample_id: str, mode_id: int, q_e4: int) -> dict:
        found = cls.readonly.execute(
            "SELECT row_json FROM quality_rows WHERE sample_id=? AND mode_id=? AND q_e4=?",
            (sample_id, mode_id, q_e4),
        ).fetchone()
        assert found is not None
        return json.loads(found[0])

    def test_real_bundle_is_bound_complete_and_read_only(self) -> None:
        self.assertEqual(self.surface.fit_frame_count, 512)
        self.assertEqual(self.surface.held_scene_frame_count, 256)
        self.assertEqual(
            self.surface.binding.corrected_p40_binding_sha256,
            self.sidecar.binding_sha256,
        )
        hasher = hashlib.sha256()
        with self.database.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
        self.assertEqual(digest, DATABASE_FILE_SHA256)
        after = self.database.stat()
        self.assertEqual(after.st_size, self.before_stat.st_size)
        self.assertEqual(after.st_mtime_ns, self.before_stat.st_mtime_ns)

    def test_exact_anchor_returns_exact_row_values_and_provenance(self) -> None:
        source = self.row(self.fit_record.sample_id, 0, 3000)
        result = self.surface.query_fit(self.fit_record.sample_id, 0, 0.30)
        self.assertEqual(result.policy.evidence_status, EXACT_GRID_ROW_EVIDENCE)
        self.assertEqual(result.policy.q_e4, 3000)
        self.assertIsInstance(result.policy.payload.total_transmitted_bytes, int)
        self.assertEqual(
            result.policy.payload.scientific_inner_payload_bytes,
            source["scientific_inner_payload_bytes"],
        )
        self.assertEqual(
            result.policy.payload.total_transmitted_bytes,
            source["total_transmitted_bytes"],
        )
        self.assertEqual(
            result.policy.payload.udp_application_bytes,
            source["udp_application_bytes"],
        )
        self.assertEqual(result.policy.payload.datagram_count, source["datagram_count"])
        for name, source_name in (
            ("q_seg", "q_seg"),
            ("q_loc", "q_loc"),
            ("q_perc", "q_perc"),
            ("seg_vehicle_iou", "seg_vehicle_iou"),
            ("seg_person_iou", "seg_person_iou"),
            ("vehicle_recall", "loc_vehicle_recall"),
            ("person_recall", "loc_person_recall"),
        ):
            self.assertEqual(result.policy.component(name).value, source[source_name])
        self.assertEqual(len(result.hidden.endpoint_evidence), 1)
        self.assertEqual(result.hidden.endpoint_evidence[0].row_sha256, source["row_sha256"])

    def test_off_anchor_interpolates_same_frame_endpoints_only(self) -> None:
        lower = self.row(self.fit_record.sample_id, 0, 1500)
        upper = self.row(self.fit_record.sample_id, 0, 3000)
        result = self.surface.query_fit_q_e4(self.fit_record.sample_id, 0, 2000)
        alpha = (2000 - 1500) / (3000 - 1500)
        self.assertEqual(result.policy.evidence_status, MODELED_SAME_FRAME_EVIDENCE)
        self.assertIsNone(result.policy.payload.datagram_count)
        self.assertEqual(
            tuple(item.q_e4 for item in result.hidden.endpoint_evidence),
            (1500, 3000),
        )
        for field in (
            "scientific_inner_payload_bytes",
            "total_transmitted_bytes",
            "udp_application_bytes",
        ):
            expected = lower[field] + alpha * (upper[field] - lower[field])
            self.assertAlmostEqual(getattr(result.policy.payload, field), expected)
            self.assertGreater(lower[field], getattr(result.policy.payload, field))
            self.assertGreater(getattr(result.policy.payload, field), upper[field])
        expected_q = lower["q_perc"] + alpha * (upper["q_perc"] - lower["q_perc"])
        self.assertAlmostEqual(result.policy.component("q_perc").value, expected_q)
        # Quality is not monotonicized: it is exactly the signed endpoint line.
        expected_seg = lower["q_seg"] + alpha * (upper["q_seg"] - lower["q_seg"])
        self.assertAlmostEqual(result.policy.component("q_seg").value, expected_seg)

    def test_undefined_quality_is_never_zero_imputed(self) -> None:
        found = self.readonly.execute(
            "SELECT sample_id, mode_id FROM quality_rows "
            "WHERE grid_split='fit' AND json_extract(row_json, '$.quality_valid')=0 LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(found)
        sample_id, mode_id = str(found[0]), int(found[1])
        exact = self.surface.query_fit_q_e4(sample_id, mode_id, 3000)
        modeled = self.surface.query_fit_q_e4(sample_id, mode_id, 2000)
        for result in (exact, modeled):
            self.assertFalse(result.policy.component("q_perc").valid)
            self.assertIsNone(result.policy.component("q_perc").value)
            self.assertNotEqual(result.policy.component("q_perc").value, 0.0)

    def test_q_boundaries_and_invalid_queries_fail_closed(self) -> None:
        self.assertEqual(
            self.surface.query_fit(self.fit_record.sample_id, 0, 0.0).policy.q_e4,
            0,
        )
        self.assertEqual(
            self.surface.query_fit(self.fit_record.sample_id, 0, 0.98).policy.q_e4,
            9800,
        )
        for invalid in (-0.0001, 0.9801, math.nan, math.inf, True, "0.5"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(InvalidSurfaceQuery):
                    self.surface.query_fit(self.fit_record.sample_id, 0, invalid)
        with self.assertRaises(InvalidSurfaceQuery):
            self.surface.query_fit_q_e4(self.fit_record.sample_id, 0, 9801)
        with self.assertRaises(InvalidSurfaceQuery):
            self.surface.query_fit(self.fit_record.sample_id, 12, 0.5)

    def test_held_scene_cannot_enter_training_api(self) -> None:
        with self.assertRaises(SplitAccessError):
            self.surface.query_fit(self.held_record.sample_id, 0, 0.5)
        result = self.surface.evaluate_held(self.held_record.sample_id, 0, 0.5)
        self.assertEqual(result.hidden.grid_split, "held_scene")
        with self.assertRaises(SplitAccessError):
            self.surface.evaluate_held(self.fit_record.sample_id, 0, 0.5)

    def test_policy_view_has_no_hidden_identity_or_gt_counts(self) -> None:
        result = self.surface.query_fit(self.fit_record.sample_id, 0, 0.5)
        policy_fields = {field.name for field in fields(PolicySurfaceView)}
        hidden_fields = {field.name for field in fields(HiddenSurfaceRecord)}
        forbidden = {
            "episode_id",
            "sample_id",
            "frame_id",
            "grid_split",
            "selection_rank_within_split",
            "sampling_weight",
            "inclusion_probability",
        }
        self.assertTrue(forbidden.isdisjoint(policy_fields))
        self.assertTrue(forbidden.intersection(hidden_fields))
        serialized = json.dumps(result.policy.to_dict(), sort_keys=True)
        for name in forbidden | {"eligible_gt", "gt_pixels", "tp", "fp", "fn"}:
            self.assertNotIn(name, serialized)
        self.assertEqual(set(result.policy.scene.to_dict()), {"camera_si", "radar_p40"})

    def test_corrected_p40_inventory_is_mandatory(self) -> None:
        parent = self.sidecar

        class MissingOne:
            binding_sha256 = parent.binding_sha256
            sample_ids = parent.sample_ids[:-1]

            @staticmethod
            def lookup(sample_id: str) -> float:
                return parent.lookup(sample_id)

        with self.assertRaises(EvidenceBindingError):
            EmpiricalQualitySurface.load(self.bundle, MissingOne())

    def test_qualification_is_deterministic_and_reports_preregistered_gates(self) -> None:
        first = self.surface.qualify_leave_one_q_out()
        second = self.surface.qualify_leave_one_q_out()
        self.assertIs(first, second)
        self.assertEqual(first.report_sha256, second.report_sha256)
        document = first.document()
        self.assertEqual(len(document["slices_by_split_mode_q"]), 2 * 12 * 9)
        self.assertEqual(
            document["gate_results"]["payload_anchor_monotone_fraction"], 1.0
        )
        self.assertIn(document["status"], {"PASS", "FAIL"})
        self.assertEqual(
            document["thresholds_preregistered_before_held_inspection"]
            ["payload_held_weighted_p95_relative_error_max"],
            0.05,
        )
        # This assertion pins the currently measured scientific outcome; it is
        # not relaxed to force admission of the continuous surface.
        self.assertEqual(document["status"], "FAIL")
        self.assertTrue(document["gate_results"]["held_payload_p95"])
        self.assertFalse(document["gate_results"]["all_quality_fit_and_held"])
        self.assertFalse(
            document["gate_results"]["full_component_surface_qualified"]
        )
        self.assertTrue(
            document["gate_results"]["reward_target_q_perc_qualified"]
        )
        for split in ("fit", "held_scene"):
            component = document["gate_results"]["quality_component_by_split"][split]
            self.assertTrue(component["q_loc"])
            self.assertTrue(component["q_perc"])
            self.assertFalse(component["q_seg"])
        self.assertIn("d5d1e0d2", document["q_perc_scope"])


if __name__ == "__main__":
    unittest.main()
