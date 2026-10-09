"""Apply a frozen class-specific range correction to detection records.

Correction: s = exp(clip(b + w . (f - mu)/sd, -c, c)) per class (s == 1 exactly for a
disabled class).  The sensor-frame vector is scaled along its own ray (bearing and image
projection unchanged): local' = s * local;  world' = o + s * (world - o), with o the camera
origin recovered from the frame's own records (rigid local->world fit) -- record fields only.
Scores, boxes, classes, ordering and segmentation are untouched.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .features import record_features

ORIGIN_FIT_TOL_M = 0.01


def frame_origin(records: Sequence[Mapping[str, Any]]) -> np.ndarray | None:
    """Camera origin in world from >=3 non-collinear records of one frame (3-D Kabsch)."""
    if len(records) < 3:
        return None
    A = np.array([[float(r["local_x"]), float(r["local_y"]), float(r["local_z"])] for r in records])
    B = np.array([[float(r["world_x"]), float(r["world_y"]), float(r["world_z"])] for r in records])
    ca, cb = A.mean(0), B.mean(0)
    U, sv, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    if sv[1] < 1e-6:
        return None
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    if np.abs(A @ R.T + t - B).max() > ORIGIN_FIT_TOL_M:
        return None
    return t


def load_frozen(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"frozen parameter sha mismatch {digest}")
    params = json.loads(raw)
    params["_sha256"] = digest
    return params


def log_scale(rec: Mapping[str, Any], params: Mapping[str, Any]) -> float:
    cls = params["classes"].get(str(rec["class_name"]))
    if cls is None or not cls["enabled"]:
        return 0.0
    f = record_features(rec)
    z = (f - np.asarray(cls["mean"])) / np.asarray(cls["std"])
    v = float(cls["intercept"]) + float(np.dot(np.asarray(cls["coef"]), z))
    c = float(params["clip_log_scale"])
    return min(max(v, -c), c)


def correct_record(rec: Mapping[str, Any], params: Mapping[str, Any], origin: np.ndarray | None) -> dict[str, Any]:
    out = dict(rec)
    ls = log_scale(rec, params)
    if ls == 0.0 or origin is None:
        return out  # identity: values returned bit-exact
    s = math.exp(ls)
    for k in ("local_x", "local_y", "local_z"):
        out[k] = s * float(rec[k])
    for k, o in zip(("world_x", "world_y", "world_z"), origin):
        out[k] = float(o) + s * (float(rec[k]) - float(o))
    if "actor_forward_depth_m" in rec:
        out["actor_forward_depth_m"] = out["local_x"]
    return out


def correct_frame(records: Sequence[Mapping[str, Any]], params: Mapping[str, Any],
                  origin: np.ndarray | None = None) -> list[dict[str, Any]]:
    """origin=None -> recover from the frame's own records."""
    if origin is None:
        origin = frame_origin(records)
    return [correct_record(r, params, origin) for r in records]
