"""Registered magnitude-aware float32 translation tolerance for reconstructed-vs-CARLA pose comparisons (M3.1 v2).

Why: CARLA keeps world transforms in float32. A pose reconstructed in float64 (ego Transform -> matrix, composed with a
static mount, or rebuilt from the wire FrameContext) differs from CARLA's own float32 sensor/actor matrix by up to about
one float32 ULP of the largest translation magnitude (v1 capture: max 0.997 ULP over 6,028 comparisons). The retained
fixed 1e-5 m absolute bound is 1.31 ULP below 128 m but only 0.66 ULP between 128 and 256 m, so it rejects correct poses
at those coordinates (v1 frame 3765, 1.043e-5 m at ~128.6 m).

Rule (``contract.POSE``), applied ONLY at the registered call sites (``contract.POSE_SUBSTITUTIONS``):
  * both matrices pass the retained ``pose._matrix`` checks (shape, finiteness, homogeneous row 1e-8, rigid rotation);
  * rotation 3x3 block max |delta| <= 1e-5 and homogeneous row max |delta| <= 1e-5 (the retained ATOL, unchanged);
  * M = max |translation component| over both matrices must be <= 512 m (registered coordinate range), else REJECT;
  * translation max |delta| <= max(1e-5, 2 x ULP32(M))  (<= 1.22e-4 m inside the range).
The return value is the overall max |delta| over all 16 elements, exactly what the retained ``_agree`` returned (it is
recorded unchanged, e.g. in UE_WIRE ``max_abs_matrix_delta``). Every call is counted per check name.
"""
import json
import math
import threading
from pathlib import Path

import numpy as np

from . import contract as C

_LOCK = threading.Lock()
STATS = {}


def ulp32(value):
    return float(np.spacing(np.float32(abs(float(value)))))


def translation_tolerance(magnitude):
    magnitude = float(magnitude)
    if not math.isfinite(magnitude) or magnitude > C.POSE["coordinate_range_m"]:
        raise ValueError(f"translation magnitude {magnitude!r} m outside the registered coordinate range")
    return max(C.POSE["floor_m"], C.POSE["ulps"] * ulp32(magnitude))


def agree_reconstructed(actual, expected, name):
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose as P
    a, e = P._matrix(actual, name), P._matrix(expected, name)
    rotation = float(np.max(np.abs(a[:3, :3] - e[:3, :3])))
    homogeneous = float(np.max(np.abs(a[3] - e[3])))
    P._require(rotation <= P.ATOL and homogeneous <= P.ATOL,
               f"{name} pose mismatch: rotation/homogeneous max_abs_delta={max(rotation, homogeneous)}")
    magnitude = float(max(np.max(np.abs(a[:3, 3])), np.max(np.abs(e[:3, 3]))))
    P._require(magnitude <= C.POSE["coordinate_range_m"],
               f"{name} translation {magnitude} m outside the registered coordinate range")
    tolerance = translation_tolerance(magnitude)
    translation = float(np.max(np.abs(a[:3, 3] - e[:3, 3])))
    with _LOCK:
        row = STATS.setdefault(name, dict(calls=0, rejected=0, max_translation_delta_m=0.0, max_ulp_ratio=0.0,
                                          max_magnitude_m=0.0, max_tolerance_m=0.0, max_rotation_delta=0.0,
                                          over_retained_1e5=0))
        row["calls"] += 1
        row["max_translation_delta_m"] = max(row["max_translation_delta_m"], translation)
        row["max_ulp_ratio"] = max(row["max_ulp_ratio"], translation / ulp32(magnitude) if magnitude else 0.0)
        row["max_magnitude_m"] = max(row["max_magnitude_m"], magnitude)
        row["max_tolerance_m"] = max(row["max_tolerance_m"], tolerance)
        row["max_rotation_delta"] = max(row["max_rotation_delta"], rotation)
        row["over_retained_1e5"] += int(translation > P.ATOL)
        row["rejected"] += int(translation > tolerance)
    P._require(translation <= tolerance,
               f"{name} pose mismatch: translation delta={translation} > tolerance={tolerance} at max|t|={magnitude}")
    return float(max(rotation, homogeneous, translation))


def snapshot():
    with _LOCK:
        return json.loads(json.dumps(STATS))


def write(path):
    Path(path).write_text(json.dumps(dict(schema="splitfusion.m3_route_b_v3.pose_tolerance.v1", rule=C.POSE,
                                          checks=snapshot()), indent=1, sort_keys=True) + "\n")
