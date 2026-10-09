"""Source-pinned, CPU-only invocation of the original complete v010 builder.

This is not a live evaluator. In particular, an AVO input bundle is NOT a
complete v010 input. Empty collections must be explicitly declared complete;
they are never manufactured for a missing static catalogue or quarantine.
The original build_contract function runs unchanged in a private module. Its
three image reads are redirected to validated arrays; its output is written
only into a TemporaryDirectory and returned as bytes and arrays.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import math
import re
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "splitfusion.complete_v010_inputs.v1"
CONTRACT = "ORIGINAL_ROUTE_B_V3_1_V010_COMPLETE"
SOURCES = {
    "builder": ("pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_clean_base_v1/build_contract_v1.py",
                "69eb5a8df84bd625af99a74d6ef7e8b50ac23ec73a7ee142c151725d2d55a264"),
    "visibility": ("data_collection/route_b_perception_v3/visibility_v1.py",
                   "4a7aa974ea6374eceff35c0fbd8261fba299b2c5de68297591f5ce9756cf980c"),
    "reconciliation": ("experiments/route_b_v3_vehicle_gt_reconciliation_audit_v1/20260828_000427/scripts/run_vehicle_gt_reconciliation_audit_v1.py",
                       "3172369c468a0a7940bad185589262969a2ec197fbd2336f123ad4b285a56022"),
    "scorer": ("pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_clean_base_v1/score_contract_v1.py",
               "14629e69a1617d05ca6dec2bad6901b69f83df96f2cf5543509dbe970d18069d"),
    "collector": ("data_collection/run_route_b_perception_collection_v3.py",
                  "09a373481d60e1630e69c1ea45403b3bc26104a58050e4159cac3c598304a7df"),
}
COMPLETENESS_KEYS = (
    "native_depth_and_semantic", "original_v3_person_mask", "all_dynamic_actor_boxes",
    "all_dynamic_actor_visibility", "all_static_candidate_rows", "static_catalogue_provenance",
    "identity_and_frame_conflicts", "registered_quarantine_scope",
)
QUARANTINE_KEYS = (
    "confirmed_secondary", "manual_static_ids", "dynamic_static_keys",
    "dynamic_actor_keys", "panel_static_keys", "confirmed_pairs",
)


class TargetInputError(ValueError):
    """Exact target inputs are absent, inconsistent, or outside their provenance."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(v) for v in value), key=lambda v: canonical(v))
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(root: Path = ROOT) -> dict[str, dict[str, str]]:
    result = {}
    for name, (relative, expected) in SOURCES.items():
        actual = sha256(Path(root) / relative)
        if actual != expected:
            raise TargetInputError(f"pinned {name} source changed: {actual}")
        result[name] = {"path": relative, "sha256": actual}
    return result


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _original_modules(root: Path):
    verify_sources(root)
    builder = _module("_private_original_v010_builder", root / SOURCES["builder"][0])
    visibility = _module("_private_original_v3_visibility", root / SOURCES["visibility"][0])
    # Import only this unchanged original function: the historical audit's
    # module top level imports unrelated model scorers. No numerical code is
    # rewritten, and the entire original source is pinned above.
    path = root / SOURCES["reconciliation"][0]
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "roi_mask")
    namespace = {"np": np, "math": math, "Any": Any, "Mapping": Mapping,
                 "reconstruct_consistent_mask": visibility.reconstruct_consistent_mask}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    recon = types.SimpleNamespace(roi_mask=namespace["roi_mask"],
        decode_depth_bgra=visibility.decode_depth_bgra,
        depth_is_plausible=visibility.depth_is_plausible)
    return builder, visibility, recon


@dataclass(frozen=True)
class CompleteTargetInputs:
    """All arguments have no defaults; absence must never mean an empty set.

    provenance contains schema, scope (HISTORICAL_REGISTERED_V010 or an explicit
    prospective scope), sample_id, evidence_sha256 mapping and all completeness
    declarations. These bind the collector/catalogue/conflict evidence, not
    merely the image arrays. This API does not itself qualify a new collector.
    """
    frame: Mapping[str, Any]
    split: str
    depth_bgra: np.ndarray
    raw_semantic: np.ndarray
    original_v3_mask: np.ndarray
    actor_boxes: Sequence[Mapping[str, Any]]
    actor_visibility: Sequence[Mapping[str, Any]]
    static_candidates: Sequence[Mapping[str, Any]]
    quarantine: Mapping[str, Any]
    provenance: Mapping[str, Any]


def input_content_sha256(inputs: CompleteTargetInputs) -> str:
    """Bind the exact native arrays, metadata, row order and quarantine sets."""
    arrays = {}
    for name in ("depth_bgra", "raw_semantic", "original_v3_mask"):
        array = getattr(inputs, name)
        arrays[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}
    metadata = {key: _json_value(getattr(inputs, key)) for key in
                ("frame", "split", "actor_boxes", "actor_visibility", "static_candidates", "quarantine")}
    return hashlib.sha256(canonical({"arrays": arrays, "metadata": metadata})).hexdigest()


def _validate(inputs: CompleteTargetInputs) -> None:
    p, f = inputs.provenance, inputs.frame
    if p.get("schema") != SCHEMA or not p.get("scope") or p.get("sample_id") != f.get("sample_id"):
        raise TargetInputError("missing or mismatched complete-input provenance")
    missing = [key for key in COMPLETENESS_KEYS if p.get("complete", {}).get(key) is not True]
    if missing:
        raise TargetInputError("incomplete exact v010 inputs: " + ", ".join(missing))
    if set(inputs.quarantine) != set(QUARANTINE_KEYS):
        raise TargetInputError("all explicit quarantine collections are required")
    evidence = p.get("evidence_sha256", {})
    if not evidence or any(len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in evidence.values()):
        raise TargetInputError("input provenance requires SHA-256 evidence bindings")
    for key in ("sample_id", "experiment_id", "frame_id", "camera_width", "camera_height", "mask_path", "instance_raw_path"):
        if key not in f:
            raise TargetInputError(f"missing original manifest field: {key}")
    for key in ("sample_id", "experiment_id"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(f[key])):
            raise TargetInputError(f"unsafe original manifest identity: {key}")
    if inputs.split not in ("train", "val"):
        raise TargetInputError("original builder split must be train or val")
    h, w = int(f["camera_height"]), int(f["camera_width"])
    for label, array, shape in (("depth", inputs.depth_bgra, (h, w, 4)),
                               ("raw semantic", inputs.raw_semantic, (h, w)),
                               ("original v3 mask", inputs.original_v3_mask, (h, w))):
        if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.shape != shape:
            raise TargetInputError(f"{label} must be native uint8 {shape}")
    if not set(np.unique(inputs.original_v3_mask)).issubset({0, 1, 2}):
        raise TargetInputError("original v3 mask has illegal labels")
    keys = lambda rows: [(str(r["sample_id"]), str(r["gt_actor_id"]), str(r["label"])) for r in rows]
    b, v = keys(inputs.actor_boxes), keys(inputs.actor_visibility)
    if set(b) != set(v) or len(set(b)) != len(b) or len(set(v)) != len(v):
        raise TargetInputError("actor box/visibility identity mismatch or duplicate")
    for rows in (inputs.actor_boxes, inputs.actor_visibility, inputs.static_candidates):
        if any(str(r["sample_id"]) != str(f["sample_id"]) for r in rows):
            raise TargetInputError("foreign-frame target input")
    if any(r["label"] not in ("person", "vehicle") for r in inputs.actor_boxes):
        raise TargetInputError("unexpected dynamic actor class")
    if p.get("input_content_sha256") != input_content_sha256(inputs):
        raise TargetInputError("complete-input content digest mismatch")


def build_complete_v010(inputs: CompleteTargetInputs, *, root: Path = ROOT) -> dict[str, Any]:
    """Run exact original target construction, including static and IGNORE rules."""
    _validate(inputs)
    builder, _visibility, recon = _original_modules(root)
    f = dict(inputs.frame)
    source_view = builder.FROZEN / "views" / inputs.split
    reads = {
        str(source_view / f["experiment_id"] / "depth" / f"{f['sample_id']}.png"): inputs.depth_bgra,
        str(source_view / f["instance_raw_path"]): inputs.raw_semantic,
        str(source_view / f["mask_path"]): inputs.original_v3_mask,
    }
    if len(reads) != 3:
        raise TargetInputError("three original image paths must be distinct")

    class Cv2Inputs:
        def __getattr__(self, name):
            return getattr(cv2, name)

        def imread(self, path, _flags):
            if str(path) not in reads:
                raise TargetInputError(f"unregistered original image read: {path}")
            return reads[str(path)].copy()

    builder.cv2 = Cv2Inputs()  # private module only; imported cv2 remains untouched
    with tempfile.TemporaryDirectory(prefix="v010_target_parity_") as temporary:
        output = Path(temporary)
        summary = builder.build_contract(contract="v010", split=inputs.split,
            manifest=[f], boxes=[dict(r) for r in inputs.actor_boxes],
            visibility=[dict(r) for r in inputs.actor_visibility],
            static_rows=[dict(r) for r in inputs.static_candidates],
            quarantine=inputs.quarantine, recon=recon, output_root=output)
        directory = output / "contracts" / "v010" / inputs.split
        seg = directory / "segmentation_masks" / f"{f['sample_id']}.png"
        obj = directory / "object_ignore_masks" / f"{f['sample_id']}.png"
        tables = {name: builder.read_csv(directory / name) for name in
                  ("object_boxes.csv", "object_ignore_regions.csv", "target_manifest.csv")}
        return {"contract": CONTRACT, "live_qualified": False, "sources": verify_sources(root),
                "provenance": dict(inputs.provenance), "summary": summary, "tables": tables,
                "segmentation": cv2.imread(str(seg), cv2.IMREAD_UNCHANGED),
                "object_ignore": cv2.imread(str(obj), cv2.IMREAD_UNCHANGED),
                "segmentation_png": seg.read_bytes(), "object_ignore_png": obj.read_bytes()}


def original_v3_person_mask(*, vehicle_mask: np.ndarray, depth_m: np.ndarray,
                           person_visibility: Sequence[Mapping[str, Any]], root: Path = ROOT) -> dict[str, Any]:
    """Reconstruct the original v3 person-paint step, explicitly NOT full v010.

    Uses the original helper for each retained interval and the collector's
    exact eligible/vehicle-preserving union rule. Callers generating new rows
    must use original visibility_from_corners with actual eight world corners.
    AVO masks/thresholds are not used anywhere in this operation.
    """
    _builder, visibility, _recon = _original_modules(root)
    if vehicle_mask.dtype != np.uint8 or vehicle_mask.ndim != 2 or vehicle_mask.shape != depth_m.shape:
        raise TargetInputError("native vehicle mask/depth dimensions mismatch")
    if not set(np.unique(vehicle_mask)).issubset({0, 1}):
        raise TargetInputError("base mask must contain only background and vehicle")
    result = vehicle_mask.copy()
    h, w = result.shape
    for row in person_visibility:
        if row.get("label") != "person":
            raise TargetInputError("person painting requires only person visibility rows")
        if str(row["eligible_visible_v010"]).lower() in ("true", "1", "yes"):
            support = visibility.reconstruct_consistent_mask(depth_m, dict(row), width=w, height=h)
            if int(np.count_nonzero(support)) != int(row["native_visible_px"]):
                raise TargetInputError("retained person support count mismatch")
            result[support & (result != 1)] = 2
    return {"contract": "ORIGINAL_V3_DEPTH_INTERVAL_PERSON_PAINT_ONLY_NOT_V010",
            "live_qualified": False, "mask": result, "sources": verify_sources(root)}


def audit_saved_live_bundle(document: Mapping[str, Any]) -> dict[str, Any]:
    """Explain why the current AVO-only recording cannot yield exact v010."""
    return {"status": "EXACT_V010_INPUTS_UNAVAILABLE", "can_score_corrected_v010": False,
            "available": [key for key in ("payloads", "camera_matrix", "camera_inverse", "intrinsics",
                          "raw_pedestrian_rows", "source_synchronization") if key in document],
            "missing": ["original_v3_mask_or_exact_all_actor_world_corners",
                        "complete_dynamic_vehicle_boxes_and_visibility",
                        "static_environment_catalogue_and_per_frame_candidates",
                        "registered_static_duplicates_and_manual_quarantines",
                        "frame_specific_static_dynamic_conflicts_and_quarantines"],
            "reason": "Pedestrian AVO volumes and raw tags do not reconstruct the full original v010 target."}


def score_complete_v010(target_result: Mapping[str, Any], prediction: np.ndarray,
                       *, root: Path = ROOT) -> dict[str, Any]:
    """Call the unchanged original pixel scorer: resize PREDICTION, ignore255.

    This is segmentation only, not a replacement for AVO/localization or the
    Q_perc combiner. The target must come from complete target construction.
    Missing predicted dense labels cannot be replaced by aggregate counts.
    """
    verify_sources(root)
    if target_result.get("contract") != CONTRACT:
        raise TargetInputError("segmentation scoring requires a complete v010 target")
    target = target_result["segmentation"]
    if target.shape != (432, 768) or target.dtype != np.uint8 or not set(np.unique(target)).issubset({0, 1, 2, 255}):
        raise TargetInputError("v010 target must be 768x432 with IGNORE=255")
    encoded = np.frombuffer(target_result["segmentation_png"], dtype=np.uint8)
    if not np.array_equal(cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED), target):
        raise TargetInputError("target array differs from built target PNG")
    if not isinstance(prediction, np.ndarray) or prediction.ndim != 2 or prediction.dtype != np.uint8:
        raise TargetInputError("predicted dense labels must be a 2-D uint8 array")
    if not set(np.unique(prediction)).issubset({0, 1, 2}):
        raise TargetInputError("illegal predicted label")
    scorer = _module("_private_original_v010_scorer", root / SOURCES["scorer"][0])
    sample = str(target_result["provenance"]["sample_id"])
    truth_path = Path("/offline_v010_inputs/contracts/v010/val/segmentation_masks") / f"{sample}.png"
    prediction_path = Path("/offline_v010_predictions/prediction.png")

    class Cv2Inputs:
        def __getattr__(self, name):
            return getattr(cv2, name)

        def imread(self, path, _flags):
            if Path(path) == truth_path:
                return target.copy()
            if Path(path) == prediction_path:
                return prediction.copy()
            raise TargetInputError("unexpected original scorer image path")

    scorer.cv2 = Cv2Inputs()
    scorer.read_csv = lambda _path: [{"sample_id": sample, "prediction_path": "prediction.png"}]
    result = scorer.score_segmentation(Path("/offline_v010_inputs"), "v010", [sample],
                                      prediction_path.parent, Path("/offline_v010_predictions/manifest.csv"))
    confusion = np.asarray(result["confusion_matrix"], dtype=np.int64)
    counts = {}
    for name, index in (("vehicle", 1), ("person", 2)):
        intersection = int(confusion[index, index])
        gt, pred = int(confusion[index, :].sum()), int(confusion[:, index].sum())
        counts[name] = {"gt_pixels": gt, "pred_pixels": pred, "intersection_pixels": intersection,
                        "union_pixels": gt + pred - intersection}
    return {"contract": CONTRACT, "scoring_contract": "ORIGINAL_PIXEL_CONFUSION_PREDICTION_NEAREST_TO_768x432_IGNORE255",
            "live_qualified": False, "scorer_sha256": SOURCES["scorer"][1],
            "original_score": result, "quality_grid_segmentation_counts": counts}


def required_collection_inputs() -> dict[str, Any]:
    """Executable interface inventory; it does not authorize or qualify live GT."""
    return {"schema": SCHEMA, "contract": CONTRACT, "live_qualified": False,
        "arrays": {"depth_bgra": "native HxWx4 uint8, original CARLA 24-bit encoded depth",
                   "raw_semantic": "native HxW uint8 raw tags, before palette conversion",
                   "original_v3_mask": "native HxW uint8 {0,1,2}, original depth-gated person paint",
                   "predicted_labels": "preserve actual 2-D uint8 {0,1,2} output before scoring"},
        "geometry": {"dynamic_actors": "all actor IDs/classes, actor+bbox transforms and exact eight world corners, not filtered localization records",
                     "visibility": "original visibility_from_corners rows for every original actor box; preserve depth interval, native support, v010 flags and projection",
                     "static_catalogue": "complete Town10HD environment-object catalogue with IDs, transforms, sizes and source hash",
                     "static_candidates": "original static projection/visibility rows including semantic support and full original static-positive fields"},
        "quarantine": {"required_keys": list(QUARANTINE_KEYS),
                       "scope": "explicit new-run identity/frame conflict adjudication; historical sample keys MUST NOT be reused for new frames",
                       "missing_or_unadjudicated": "refuse exact-v010 scoring; never default to empty"},
        "identity_and_provenance": ["same frame/time and owned-world snapshot for RGB/semantic/depth/actor geometry",
             "camera intrinsics, complete transforms and native dimensions", "all arrays and tables SHA-256 bound",
             "complete-input content digest", "original source pins", "explicit completeness declarations"],
        "prediction_evidence": ["frame/session/ticket-bound dense predicted labels", "full class/confidence/world-XY prediction records",
                                "their hashes and scorer-input identity"],
        "established_functions": [SOURCES["visibility"][0] + "::visibility_from_corners/reconstruct_consistent_mask",
             "data_collection/run_route_b_perception_collection_v3.py::_persist person paint",
             SOURCES["builder"][0] + "::source_state/build_contract/registered_quarantines",
             SOURCES["scorer"][0] + "::score_segmentation"],
        "target_grid": [432, 768], "ignore_label": 255,
        "resize": "nearest-neighbor prediction TO fixed target grid, not GT to prediction",
        "scope_limit": "This package proves offline mask/scorer parity only; no live producer or real-time cost qualification."}
