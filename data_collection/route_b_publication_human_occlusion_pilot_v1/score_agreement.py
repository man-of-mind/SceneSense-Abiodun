#!/usr/bin/env python3
"""Fail-closed agreement scoring for two completed human annotation files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ORDERED_LABELS = (
    "not_observable",
    "largely_occluded",
    "partly_occluded",
    "fully_visible",
)
AMBIGUOUS = "ambiguous"
VISIBILITY_LABELS = ORDERED_LABELS + (AMBIGUOUS,)
TRUNCATION_LABELS = ("none", "partial", "severe")
ANNOTATION_FIELDS = ("sample_id", "visibility_label", "truncation_label", "notes")


class AnnotationError(RuntimeError):
    """Raised when an annotation input fails closed validation."""


def read_manifest_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            raise AnnotationError("manifest is missing sample_id")
        values = [str(row.get("sample_id", "")).strip() for row in reader]
    if not values or any(not value for value in values):
        raise AnnotationError("manifest contains no samples or a blank sample_id")
    if len(values) != len(set(values)):
        raise AnnotationError("manifest contains duplicate sample IDs")
    return values


def read_completed_annotations(path: Path, expected_ids: Sequence[str]) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != ANNOTATION_FIELDS:
            raise AnnotationError(
                f"{path} fields must be exactly {ANNOTATION_FIELDS}, got {reader.fieldnames}"
            )
        rows = list(reader)
    if len(rows) != len(expected_ids):
        raise AnnotationError(
            f"{path} has {len(rows)} rows; expected exactly {len(expected_ids)}"
        )
    indexed: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, 2):
        sample_id = str(row.get("sample_id", "")).strip()
        visibility = str(row.get("visibility_label", "")).strip()
        truncation = str(row.get("truncation_label", "")).strip()
        notes = str(row.get("notes", ""))
        if not sample_id:
            raise AnnotationError(f"{path}:{row_number}: blank sample_id")
        if sample_id in indexed:
            raise AnnotationError(f"{path}:{row_number}: duplicate sample_id {sample_id}")
        if visibility not in VISIBILITY_LABELS:
            raise AnnotationError(
                f"{path}:{row_number}: invalid or incomplete visibility label {visibility!r}"
            )
        if truncation not in TRUNCATION_LABELS:
            raise AnnotationError(
                f"{path}:{row_number}: invalid or incomplete truncation label {truncation!r}"
            )
        indexed[sample_id] = {
            "visibility_label": visibility,
            "truncation_label": truncation,
            "notes": notes,
        }
    expected = set(expected_ids)
    actual = set(indexed)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AnnotationError(f"{path}: sample-ID mismatch; missing={missing}, extra={extra}")
    return indexed


def linear_weighted_kappa(pairs: Sequence[tuple[str, str]]) -> float:
    if not pairs:
        raise AnnotationError("no non-ambiguous samples remain for kappa")
    size = len(ORDERED_LABELS)
    indices = {label: index for index, label in enumerate(ORDERED_LABELS)}
    observed = [[0 for _ in range(size)] for _ in range(size)]
    for left, right in pairs:
        observed[indices[left]][indices[right]] += 1
    row_totals = [sum(row) for row in observed]
    col_totals = [sum(observed[row][col] for row in range(size)) for col in range(size)]
    count = float(len(pairs))
    max_distance = float(size - 1)
    observed_disagreement = sum(
        (abs(row - col) / max_distance) * observed[row][col] / count
        for row in range(size)
        for col in range(size)
    )
    expected_disagreement = sum(
        (abs(row - col) / max_distance)
        * (row_totals[row] / count)
        * (col_totals[col] / count)
        for row in range(size)
        for col in range(size)
    )
    if expected_disagreement <= 0.0:
        if observed_disagreement <= 0.0:
            return 1.0
        raise AnnotationError("weighted kappa has zero expected disagreement")
    value = 1.0 - observed_disagreement / expected_disagreement
    if not math.isfinite(value):
        raise AnnotationError("weighted kappa is non-finite")
    return float(value)


def score_annotations(
    manifest_ids: Sequence[str],
    annotator_a: Mapping[str, Mapping[str, str]],
    annotator_b: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    if set(annotator_a) != set(manifest_ids) or set(annotator_b) != set(manifest_ids):
        raise AnnotationError("annotation inputs do not exactly match the manifest")
    confusion = {
        left: {right: 0 for right in VISIBILITY_LABELS}
        for left in VISIBILITY_LABELS
    }
    exact_visibility = 0
    exact_truncation = 0
    ambiguous_any = 0
    ambiguous_a = 0
    ambiguous_b = 0
    ordered_pairs: list[tuple[str, str]] = []
    for sample_id in manifest_ids:
        left = str(annotator_a[sample_id]["visibility_label"])
        right = str(annotator_b[sample_id]["visibility_label"])
        confusion[left][right] += 1
        exact_visibility += int(left == right)
        exact_truncation += int(
            annotator_a[sample_id]["truncation_label"]
            == annotator_b[sample_id]["truncation_label"]
        )
        left_ambiguous = left == AMBIGUOUS
        right_ambiguous = right == AMBIGUOUS
        ambiguous_a += int(left_ambiguous)
        ambiguous_b += int(right_ambiguous)
        ambiguous_any += int(left_ambiguous or right_ambiguous)
        if not left_ambiguous and not right_ambiguous:
            ordered_pairs.append((left, right))
    count = len(manifest_ids)
    kappa = linear_weighted_kappa(ordered_pairs)
    return {
        "schema": "route_b_human_occlusion_agreement_v1",
        "sample_count": count,
        "exact_visibility_agreement_count": exact_visibility,
        "exact_visibility_agreement_rate": exact_visibility / count,
        "exact_truncation_agreement_count": exact_truncation,
        "exact_truncation_agreement_rate": exact_truncation / count,
        "visibility_confusion_matrix": {
            "labels": list(VISIBILITY_LABELS),
            "annotator_a_rows_annotator_b_columns": [
                [confusion[left][right] for right in VISIBILITY_LABELS]
                for left in VISIBILITY_LABELS
            ],
        },
        "ambiguous": {
            "either_annotator_count": ambiguous_any,
            "either_annotator_rate": ambiguous_any / count,
            "annotator_a_count": ambiguous_a,
            "annotator_a_rate": ambiguous_a / count,
            "annotator_b_count": ambiguous_b,
            "annotator_b_rate": ambiguous_b / count,
        },
        "kappa_non_ambiguous_count": len(ordered_pairs),
        "linearly_weighted_cohens_kappa": kappa,
        "qualification_threshold": 0.75,
        "pilot_qualified": bool(kappa >= 0.75),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--annotator-a", required=True, type=Path)
    parser.add_argument("--annotator-b", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest_ids = read_manifest_ids(args.manifest)
        annotator_a = read_completed_annotations(args.annotator_a, manifest_ids)
        annotator_b = read_completed_annotations(args.annotator_b, manifest_ids)
        result = score_annotations(manifest_ids, annotator_a, annotator_b)
        rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        sys.stdout.write(rendered)
        return 0
    except Exception as exc:  # fail closed at the CLI boundary
        sys.stderr.write(json.dumps({
            "pilot_qualified": False,
            "error": f"{type(exc).__name__}: {exc}",
        }, sort_keys=True) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
