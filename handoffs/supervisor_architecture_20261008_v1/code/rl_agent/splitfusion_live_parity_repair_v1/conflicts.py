"""Prospective, conservative application of the ORIGINAL conflict detectors.

No historical sample-level quarantine or manual-panel decision is transferred.
This is a new live-input scope, not the historical v010 quarantine population.
Caller must supply a complete new-run catalogue and explicit causal history.
"""
from __future__ import annotations

import ast
from bisect import insort
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from rl_agent.splitfusion_live_target_parity_v1 import adapter as A

SCOPE = "PROSPECTIVE_LIVE_V010_CONSERVATIVE_CONFLICTS_V1"
COMPLETENESS = ("catalogue_complete", "dynamic_snapshot_complete", "visibility_complete",
                "static_projection_complete", "history_complete_for_declared_window")


class ConflictInputError(ValueError):
    pass


def _sha(value):
    return hashlib.sha256(A.canonical(A._json_value(value))).hexdigest()


def catalogue_content_sha256(rows):
    return _sha(rows)


@lru_cache(maxsize=1)
def _original_functions():
    A.verify_sources()
    source = A.ROOT / A.SOURCES["reconciliation"][0]
    names = {"box_iou_xywh", "yaw_delta_mod180", "rotated_rect", "footprint_iou",
             "static_geometry", "actor_geometry", "static_static_conflicts"}
    tree = ast.parse(source.read_text(), filename=str(source))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(nodes) != len(names):
        raise ConflictInputError("original conflict API changed")
    namespace = {"Any": Any, "Mapping": Mapping, "Sequence": Sequence,
                 "np": np, "cv2": cv2, "math": math, "median": median, "defaultdict": defaultdict}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    builder_path = A.ROOT / A.SOURCES["builder"][0]
    node = next(n for n in ast.parse(builder_path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "compute_train_dynamic_conflicts")
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(builder_path), "exec"), namespace)
    return namespace


def _finite_rows(rows, fields, label):
    for row in rows:
        for key in fields:
            try:
                number = float(row[key])
            except (KeyError, ValueError, TypeError) as error:
                raise ConflictInputError(f"missing/malformed {label}.{key}") from error
            if not math.isfinite(number):
                raise ConflictInputError(f"nonfinite {label}.{key}")


def derive_live_quarantine(*, frame: Mapping[str, Any], actor_boxes: Sequence[Mapping[str, Any]],
                           actor_visibility: Sequence[Mapping[str, Any]],
                           static_candidates: Sequence[Mapping[str, Any]],
                           static_catalogue: Sequence[Mapping[str, Any]],
                           prior_static_candidates: Sequence[Mapping[str, Any]],
                           provenance: Mapping[str, Any], _static_result=None) -> dict[str, Any]:
    """Return original-format quarantine sets and new-run adjudication evidence.

    ``provenance`` requires SCOPE, ``capture_identity`` (including matching
    frame_id), all five completeness declarations, the canonical catalogue
    content hash, and ``history_window`` with inclusive ``start_timestamp``
    and exclusive ``end_timestamp`` equal to the current frame timestamp.
    The history is caller-declared; this function cannot prove the collector
    observed every candidate. Missing inputs never become empty collections.
    """
    A.verify_sources()
    if provenance.get("scope") != SCOPE:
        raise ConflictInputError("explicit prospective conflict scope required")
    if any(provenance.get("complete", {}).get(k) is not True for k in COMPLETENESS):
        raise ConflictInputError("incomplete conflict input declaration")
    sid, episode = str(frame["sample_id"]), str(frame["experiment_id"])
    if "canonical_v3_" in sid or "canonical_v3_" in episode:
        raise ConflictInputError("historical sample keys forbidden for prospective conflicts")
    if int(provenance.get("capture_identity", {}).get("frame_id", -1)) != int(frame["frame_id"]):
        raise ConflictInputError("capture/frame identity mismatch")
    if provenance.get("catalogue_content_sha256") != catalogue_content_sha256(static_catalogue):
        raise ConflictInputError("catalogue digest mismatch")
    window = provenance.get("history_window", {})
    try:
        start, end, now = (float(window["start_timestamp"]), float(window["end_timestamp"]), float(frame["timestamp"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ConflictInputError("explicit causal history window required") from error
    if not all(math.isfinite(v) for v in (start, end, now)) or start > end or end != now:
        raise ConflictInputError("invalid causal history window")
    ids = [str(r["environment_object_id"]) for r in static_catalogue]
    if len(set(ids)) != len(ids):
        raise ConflictInputError("duplicate static catalogue identity")
    static_fields = [f"bbox_center_{a}_m" for a in "xyz"] + [f"bbox_extent_{a}_m" for a in "xyz"] + ["bbox_rotation_yaw_deg"]
    _finite_rows(static_catalogue, static_fields, "catalogue")
    for r in static_catalogue:
        if any(float(r[f"bbox_extent_{a}_m"]) <= 0 for a in "xyz"):
            raise ConflictInputError("nonpositive static extent")
    def keys(rows):
        return [(str(r["sample_id"]), str(r["gt_actor_id"]), str(r["label"])) for r in rows]
    box_keys, visibility_keys = keys(actor_boxes), keys(actor_visibility)
    if set(box_keys) != set(visibility_keys) or len(set(box_keys)) != len(box_keys) or len(set(visibility_keys)) != len(visibility_keys):
        raise ConflictInputError("dynamic box/visibility identity mismatch")
    for rows in (actor_boxes, actor_visibility, static_candidates):
        if any(str(r["sample_id"]) != sid or str(r["experiment_id"]) != episode or int(r["frame_id"]) != int(frame["frame_id"]) for r in rows):
            raise ConflictInputError("foreign current-frame conflict row")
        if any(float(r["timestamp"]) != now for r in rows):
            raise ConflictInputError("current-frame timestamp mismatch")
    current_and_prior = list(prior_static_candidates) + list(static_candidates)
    candidate_keys = [(str(r["sample_id"]), str(r["environment_object_id"])) for r in current_and_prior]
    if len(set(candidate_keys)) != len(candidate_keys):
        raise ConflictInputError("duplicate static observation")
    for row in current_and_prior:
        if str(row["environment_object_id"]) not in ids:
            raise ConflictInputError("static candidate absent from complete catalogue")
    for row in prior_static_candidates:
        stamp = float(row["timestamp"])
        if str(row["experiment_id"]) != episode or "canonical_v3_" in str(row["sample_id"]) or not start <= stamp < end or int(row["frame_id"]) >= int(frame["frame_id"]):
            raise ConflictInputError("noncausal/foreign static history")
    static_numeric = static_fields + ["clipped_bbox_x", "clipped_bbox_y", "clipped_bbox_w", "clipped_bbox_h", "range_m", "actor_near_depth_m"]
    _finite_rows(current_and_prior, static_numeric, "static candidate")
    _finite_rows([r for r in actor_boxes if r["label"] == "vehicle"],
                 [f"object_world_{a}" for a in "xyz"] + [f"gt_size_{a}_m" for a in "xyz"] + ["object_yaw_deg"], "dynamic vehicle")
    if any(float(r[f"gt_size_{a}_m"]) <= 0 for r in actor_boxes if r["label"] == "vehicle" for a in "xyz"):
        raise ConflictInputError("nonpositive dynamic vehicle size")
    _finite_rows(actor_visibility, ["clipped_bbox_x", "clipped_bbox_y", "clipped_bbox_w", "clipped_bbox_h"], "visibility")
    original = _original_functions()
    static_rows, secondary, manual = (original["static_static_conflicts"](static_catalogue, current_and_prior)
        if _static_result is None else _static_result)
    recon = SimpleNamespace(**{k: original[k] for k in
        ("static_geometry", "actor_geometry", "footprint_iou", "box_iou_xywh")})
    dynamic_static, dynamic_actor, dynamic_rows = original["compute_train_dynamic_conflicts"](
        [frame], actor_boxes, actor_visibility, static_candidates, recon)
    quarantine = {"confirmed_secondary": secondary, "manual_static_ids": manual,
        "dynamic_static_keys": dynamic_static, "dynamic_actor_keys": dynamic_actor,
        "panel_static_keys": set(),
        "confirmed_pairs": [row for row in static_rows if row["adjudication"] == "CONFIRMED_DUPLICATE"]}
    if set(quarantine) != set(A.QUARANTINE_KEYS):
        raise ConflictInputError("quarantine API mismatch")
    evidence = {"scope": SCOPE, "capture_identity": dict(provenance["capture_identity"]),
        "sample_id": sid, "experiment_id": episode,
        "sources": {k: {"path": A.SOURCES[k][0], "sha256": A.SOURCES[k][1]}
                    for k in ("builder", "reconciliation")},
        "provenance": dict(provenance),
        "input_content_sha256": _sha({"frame": frame, "actor_boxes": actor_boxes,
            "actor_visibility": actor_visibility, "static_candidates": static_candidates,
            "static_catalogue": static_catalogue, "prior_static_candidates": prior_static_candidates}),
        "static_static_adjudications": static_rows, "static_dynamic_adjudications": dynamic_rows,
        "quarantine": A._json_value(quarantine),
        "historical_manual_panel_policy": "NOT_APPLICABLE__NO_HISTORICAL_SAMPLE_KEYS_TRANSFERRED",
        "ambiguous_policy": "ORIGINAL_MANUAL_REVIEW_REQUIRED_IS_IGNORE_NOT_POSITIVE",
        "zero_semantic_and_unresolved_components": "UNCHANGED_ORIGINAL_BUILD_CONTRACT_RULES",
        "live_qualified": False}
    return {"scope": SCOPE, "quarantine": quarantine, "evidence": evidence,
            "evidence_sha256": _sha(evidence)}


@dataclass(frozen=True)
class _PairHistory:
    iou: tuple[float, ...] = ()
    ranges: tuple[float, ...] = ()
    near: tuple[float, ...] = ()
    overlap_count: int = 0
    best: tuple[float, str] = (-1., "")

    def appended(self, iou, distance, near, sample_id):
        # Exact sorted observations retain the original statistics.median
        # operation; no approximation, sample eviction or finite window.
        def added(values, value):
            result = list(values)
            insort(result, value)
            return tuple(result)
        return _PairHistory(added(self.iou, iou), added(self.ranges, distance),
            added(self.near, near), self.overlap_count + int(iou >= .10),
            max(self.best, (iou, sample_id)))


class IncrementalConflictState:
    """Result-equivalent full causal-history static adjudication.

    The registered maximum bounds memory; reaching it fails rather than
    discarding history. Each observation is incorporated once. Catalogue-only
    geometry is calculated once, and only changed pair decisions are rebuilt.
    The caller durably preserves each frame's current inputs and returned
    chain evidence; no raw-history copy is re-hashed on every frame.
    """
    def __init__(self, static_catalogue, *, max_frames):
        if type(max_frames) is not int or not 1 <= max_frames <= 100000:
            raise ConflictInputError("explicit bounded maximum frame count required")
        self.catalogue = A._json_value(static_catalogue)
        self.catalogue_digest = catalogue_content_sha256(self.catalogue)
        self.max_frames = max_frames
        self.original = _original_functions()
        self.by_id = {r["environment_object_id"]: r for r in self.catalogue}
        if len(self.by_id) != len(self.catalogue):
            raise ConflictInputError("duplicate static catalogue identity")
        rows, _, _ = self.original["static_static_conflicts"](self.catalogue, [])
        self.templates = {(r["environment_object_id_a"], r["environment_object_id_b"]): r for r in rows}
        self.decisions = dict(self.templates)
        self.geometry = {key: self.original["static_geometry"](row) for key, row in self.by_id.items()}
        self.pairs = {}
        self.frame_count = self.row_count = 0
        self.previous_frame = -1
        self.previous_timestamp = -math.inf
        self.first_timestamp = None
        self.episode = None
        self.seen_samples = set()
        self.chain = _sha({"schema": "splitfusion.full_causal_static_history.v1",
                          "catalogue_sha256": self.catalogue_digest})

    def _far_pair_geometry(self, key):
        # Literal original static geometry calculations. Unlike the original
        # candidate filter, geometry is cached even while a far pair has no
        # qualifying projection. Later full-history medians may admit it.
        first, second = (self.by_id[k] for k in key)
        ac, az, ayaw = self.geometry[key[0]]
        bc, bz, byaw = self.geometry[key[1]]
        dims = np.abs(az-bz)
        focus = []
        if {first["semantic_class"], second["semantic_class"]} == {"Bus"}:
            focus.append("prior_panels_13_14_bus_pair")
        if {key[0][-64:][:10], key[1][-64:][:10]} == {"9f9707ae5c", "cefe3933e3"}:
            focus.append("prior_panels_25_32_black_suv_pair")
        return dict(environment_object_id_a=key[0], native_id_a=first["carla_environment_object_id"],
            class_a=first["semantic_class"], object_name_a=first["object_name"],
            environment_object_id_b=key[1], native_id_b=second["carla_environment_object_id"],
            class_b=second["semantic_class"], object_name_b=second["object_name"],
            world_center_distance_m=float(np.linalg.norm(ac-bc)),
            world_center_xy_distance_m=float(np.linalg.norm(ac[:2]-bc[:2])),
            dimension_x_delta_m=float(dims[0]), dimension_y_delta_m=float(dims[1]), dimension_z_delta_m=float(dims[2]),
            yaw_delta_mod180_deg=self.original["yaw_delta_mod180"](ayaw, byaw),
            oriented_footprint_iou=self.original["footprint_iou"](ac, az, ayaw, bc, bz, byaw),
            focus_case=";".join(focus))

    def _classify(self, key, base, observation):
        first, second = (self.by_id[k] for k in key)
        ac, az, ayaw = self.geometry[key[0]]
        bc, bz, byaw = self.geometry[key[1]]
        center, center_xy = base["world_center_distance_m"], base["world_center_xy_distance_m"]
        fiou, yaw = base["oriented_footprint_iou"], base["yaw_delta_mod180_deg"]
        count = len(observation.iou)
        max_iou = observation.iou[-1] if count else 0.
        med_iou = float(median(observation.iou)) if count else 0.
        med_range = float(median(observation.ranges)) if count else None
        med_near = float(median(observation.near)) if count else None
        if not (center_xy <= 3. or fiou >= .05 or (max_iou >= .25 and med_range is not None and med_range <= 2.)):
            return None
        symmetric = max(az[0], az[1]) / max(1e-9, min(az[0], az[1])) <= 1.15
        confirmed = (fiou >= .70 and center <= 1.5 and (yaw <= 20. or symmetric)
                     and count >= 3 and med_iou >= .60 and med_range is not None and med_range <= 1.)
        manual = not confirmed and fiou >= .35 and center <= 2. and max_iou >= .50
        adjudication = "CONFIRMED_DUPLICATE" if confirmed else ("MANUAL_REVIEW_REQUIRED" if manual else "REJECTED_DISTINCT")
        volumes = (float(np.prod(az)), float(np.prod(bz)))
        if volumes[0] > volumes[1] or (math.isclose(volumes[0], volumes[1]) and int(first["carla_environment_object_id"]) < int(second["carla_environment_object_id"])):
            preferred, duplicate = key
        else:
            duplicate, preferred = key
        return dict(base, common_projected_frames=count,
            projected_overlap_frames_iou_ge_010=observation.overlap_count,
            median_projected_iou=med_iou, maximum_projected_iou=max_iou,
            median_range_delta_m=med_range if med_range is not None else "",
            median_near_depth_delta_m=med_near if med_near is not None else "",
            representative_sample_id=observation.best[1] if count else "",
            preferred_environment_object_id=preferred if confirmed else "",
            duplicate_environment_object_id=duplicate if confirmed else "",
            adjudication=adjudication,
            basis=("near-identical world footprint and repeated projected/depth agreement" if confirmed
                   else "strong but non-confirming overlap; human decision required" if manual
                   else "candidate geometry resolves as distinct"))

    def derive(self, *, frame, actor_boxes, actor_visibility, static_candidates, provenance):
        if self.frame_count >= self.max_frames:
            raise ConflictInputError("registered conflict frame bound reached; no history discarded")
        sid, episode = str(frame["sample_id"]), str(frame["experiment_id"])
        fid, stamp = int(frame["frame_id"]), float(frame["timestamp"])
        if fid <= self.previous_frame or stamp <= self.previous_timestamp or sid in self.seen_samples:
            raise ConflictInputError("incremental conflict inputs must be strictly causal and unique")
        if self.episode is not None and episode != self.episode:
            raise ConflictInputError("incremental conflict episode changed")
        if self.frame_count and provenance.get("history_window", {}).get("start_timestamp") != self.first_timestamp:
            raise ConflictInputError("causal history start changed")
        # Run every original input validation and unchanged dynamic detector
        # before any persistent state is changed. Omit only static recompute.
        result = derive_live_quarantine(frame=frame, actor_boxes=actor_boxes,
            actor_visibility=actor_visibility, static_candidates=static_candidates,
            static_catalogue=self.catalogue, prior_static_candidates=[],
            provenance=provenance, _static_result=([], set(), set()))
        updates, templates, decisions = {}, {}, {}
        for index, first in enumerate(static_candidates):
            for second in static_candidates[index + 1:]:
                key = tuple(sorted((first["environment_object_id"], second["environment_object_id"])))
                iou = self.original["box_iou_xywh"](
                    [first[k] for k in ("clipped_bbox_x", "clipped_bbox_y", "clipped_bbox_w", "clipped_bbox_h")],
                    [second[k] for k in ("clipped_bbox_x", "clipped_bbox_y", "clipped_bbox_w", "clipped_bbox_h")])
                observed = self.pairs.get(key, _PairHistory()).appended(iou,
                    abs(float(first["range_m"])-float(second["range_m"])),
                    abs(float(first["actor_near_depth_m"])-float(second["actor_near_depth_m"])), sid)
                updates[key] = observed
                base = self.templates.get(key)
                if base is None:
                    base = templates[key] = self._far_pair_geometry(key)
                decisions[key] = self._classify(key, base, observed)
        all_decisions = dict(self.decisions)
        for key, row in decisions.items():
            if row is None:
                all_decisions.pop(key, None)
            else:
                all_decisions[key] = row
        rows = sorted(all_decisions.values(), key=lambda r: (r["adjudication"], r["environment_object_id_a"], r["environment_object_id_b"]))
        secondary = {r["duplicate_environment_object_id"] for r in rows if r["adjudication"] == "CONFIRMED_DUPLICATE"}
        manual = {r[k] for r in rows if r["adjudication"] == "MANUAL_REVIEW_REQUIRED"
                  for k in ("environment_object_id_a", "environment_object_id_b")}
        quarantine = dict(result["quarantine"], confirmed_secondary=secondary, manual_static_ids=manual,
                          confirmed_pairs=[r for r in rows if r["adjudication"] == "CONFIRMED_DUPLICATE"])
        frame_digest = _sha({"frame": frame, "static_candidates": static_candidates})
        chain = _sha({"previous": self.chain, "current_frame_sha256": frame_digest})
        evidence = dict(result["evidence"], static_static_adjudications=rows,
            quarantine=A._json_value(quarantine),
            input_content_sha256=_sha({"current_input_sha256": result["evidence"]["input_content_sha256"],
                                       "prior_history_chain": self.chain}),
            causal_static_history={"schema": "splitfusion.full_causal_static_history.v1",
                "previous_sha256": self.chain, "current_frame_sha256": frame_digest, "chain_sha256": chain,
                "frames": self.frame_count+1, "candidate_rows": self.row_count+len(static_candidates),
                "max_frames": self.max_frames, "history_discarded": False})
        digest = _sha(evidence)
        self.templates.update(templates)
        self.pairs.update(updates)
        self.decisions = all_decisions
        self.frame_count += 1
        self.row_count += len(static_candidates)
        self.previous_frame, self.previous_timestamp, self.episode = fid, stamp, episode
        self.first_timestamp = stamp if self.first_timestamp is None else self.first_timestamp
        self.seen_samples.add(sid)
        self.chain = chain
        return {"scope": SCOPE, "quarantine": quarantine, "evidence": evidence, "evidence_sha256": digest}
