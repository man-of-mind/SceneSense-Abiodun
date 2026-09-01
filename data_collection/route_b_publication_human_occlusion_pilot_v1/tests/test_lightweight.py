from __future__ import annotations

import ast
import csv
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from data_collection.route_b_publication_human_occlusion_pilot_v1 import build_pilot
from data_collection.route_b_publication_human_occlusion_pilot_v1 import score_agreement


class HumanOcclusionPilotChecks(unittest.TestCase):
    def test_1_selection_is_deterministic_balanced_and_exact(self) -> None:
        population, _hashes = build_pilot.load_population()
        self.assertEqual(len(population), 5276)
        first = build_pilot.select_examples(population, build_pilot.DEFAULT_SEED)
        second = build_pilot.select_examples(population, build_pilot.DEFAULT_SEED)
        identity = lambda rows: [(row["sample_id"], row["gt_actor_id"]) for row in rows]
        self.assertEqual(identity(first), identity(second))
        self.assertEqual(len(first), 100)
        self.assertEqual(len({row["sample_id"] for row in first}), 100)
        self.assertEqual(len({row["episode_id"] for row in first}), 2)
        self.assertEqual({row["distance_band"] for row in first}, {band[0] for band in build_pilot.DISTANCE_BANDS})
        group_counts = Counter((row["episode_id"], row["distance_band"]) for row in first)
        self.assertEqual(sorted(group_counts.values()), [12, 12, 12, 12, 13, 13, 13, 13])
        for group in group_counts:
            strata = {row["selection_stratum"] for row in first if (row["episode_id"], row["distance_band"]) == group}
            self.assertGreaterEqual(len(strata), 4)

    def test_2_builder_has_no_model_runtime_dependency(self) -> None:
        source = Path(build_pilot.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        self.assertTrue({"torch", "tensorflow", "carla"}.isdisjoint(imports))
        self.assertNotIn("/predictions/", source)
        self.assertNotIn("detections.csv", source)
        self.assertNotIn("checkpoints/", source)
        self.assertTrue(str(build_pilot.DEFAULT_SOURCE_ROOT).endswith("/views/val"))

    def test_3_generated_annotator_assets_hide_diagnostics_and_have_100_samples(self) -> None:
        raw = os.environ.get("HUMAN_OCCLUSION_PILOT_OUTPUT")
        if not raw:
            self.skipTest("set HUMAN_OCCLUSION_PILOT_OUTPUT to validate a generated run")
        output = Path(raw).resolve(strict=True)
        build_pilot.validate_annotator_assets(output)
        manifest = build_pilot.read_csv(output / "sample_manifest.csv")
        self.assertEqual(len(manifest), 100)
        self.assertEqual(len({row["sample_id"] for row in manifest}), 100)
        self.assertEqual(len(list((output / "panels").glob("*.png"))), 100)
        for name in ("annotator_A.csv", "annotator_B.csv"):
            rows = build_pilot.read_csv(output / name)
            self.assertEqual(len(rows), 100)
            self.assertTrue(all(not row["visibility_label"] and not row["truncation_label"] for row in rows))

    def test_4_annotation_schema_and_synthetic_agreement(self) -> None:
        sample_ids = [f"sample_{index:02d}" for index in range(8)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            with manifest.open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("sample_id",))
                writer.writeheader()
                writer.writerows({"sample_id": sample_id} for sample_id in sample_ids)
            rows = []
            labels = list(score_agreement.ORDERED_LABELS) * 2
            for sample_id, label in zip(sample_ids, labels):
                rows.append({"sample_id": sample_id, "visibility_label": label,
                             "truncation_label": "none", "notes": ""})
            paths = []
            for name in ("A.csv", "B.csv"):
                path = root / name
                with path.open("x", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=score_agreement.ANNOTATION_FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                paths.append(path)
            expected = score_agreement.read_manifest_ids(manifest)
            left = score_agreement.read_completed_annotations(paths[0], expected)
            right = score_agreement.read_completed_annotations(paths[1], expected)
            result = score_agreement.score_annotations(expected, left, right)
            self.assertEqual(result["linearly_weighted_cohens_kappa"], 1.0)
            self.assertTrue(result["pilot_qualified"])
            incomplete = root / "incomplete.csv"
            with incomplete.open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=score_agreement.ANNOTATION_FIELDS)
                writer.writeheader()
                writer.writerows(rows[:-1])
            with self.assertRaises(score_agreement.AnnotationError):
                score_agreement.read_completed_annotations(incomplete, expected)


if __name__ == "__main__":
    unittest.main()
