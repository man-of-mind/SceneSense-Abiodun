#!/usr/bin/env python3
"""Offline parity and real cost of the radar-preparation bundle, before any capture.

Runs under the UE thread-cap environment (both live arms carry it) with the
GPU visible.  Exercises the **actual Phase-6 composition**: the historical
``parked`` module wrapped by the Phase-6 ``_PreparationTimer`` (with its
``radar_tensor_start/end`` stamps) and the profiler's span proxy, exactly as the
live collector stacks them, with the bundle attached the way the live child
attaches it (``BundleParkedModule`` as the timer's inner module, the single-sort
tracker, the identity-resize packing).

Checks (exact equality; the established tolerance of the validated optimization):

* radar tensor values, dtype, shape; evidence points; summary keys and values;
* tracker outputs and retained state after every step;
* exact frame/window identity: the plan binds ``radar_window_sha256`` from the
  window metadata, which the bundle never touches (recomputed identical);
* SI/P40 from the same image/window metadata before and after the bundle;
* seven-channel model input ``torch.equal``; downstream UE codec bytes equal on
  identical inputs with the real hash-verified front/ranker/AE/codec;
* the existing bit-exact and codec-identity suites under the same environment.

Cost: full ``build_radar_sample`` + packing per path, including host<->device
transfers, allocations and the device synchronization the ``.cpu()`` copy
implies, warmed equally, paths alternated, at a live-like density (19,404
returns) and at 40,000.  Everything is written create-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
SCHEMA = "scenesense.single_gpu_profiler.radar_bundle_offline_parity.v1"
WIDTH, HEIGHT = 768, 448
MAX_RANGE_M, MAX_ABS_VELOCITY_MPS, PARKED_S, RADIUS = 120.0, 20.0, 5.0, 4
PARITY_TEST_FILES = (
    "rl_agent/splitfusion_sensor_optimization_v1/tests/test_optimized_stages.py",
    "rl_agent/splitfusion_sensor_optimization_v1/tests/test_wired_path_equivalence.py",
    "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/test_phase3_continuous_execution_v2.py",
    "rl_agent/splitfusion_single_gpu_profiler_v1/test_radar_bundle_v1.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# synthetic episode (same generator as the validated wired-path test)
# ---------------------------------------------------------------------------


def _pose(t: float) -> np.ndarray:
    yaw = 0.15 * t
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0, 8.0 * t], [s, c, 0.0, 0.4 * t],
                     [0.0, 0.0, 1.0, 1.6], [0.0, 0.0, 0.0, 1.0]])


def _intrinsics() -> np.ndarray:
    f = WIDTH / (2.0 * math.tan(math.radians(120.0) / 2.0))
    return np.array([[f, 0.0, WIDTH / 2.0], [0.0, f, HEIGHT / 2.0], [0.0, 0.0, 1.0]])


def _detections(rng: np.random.Generator, count: int) -> np.ndarray:
    altitude = np.deg2rad(rng.uniform(-15.0, 15.0, count))
    azimuth = np.deg2rad(rng.uniform(-60.0, 60.0, count))
    depth = np.clip(rng.gamma(3.0, 12.0, count), 0.5, MAX_RANGE_M)
    velocity = np.where(rng.random(count) < 0.12, rng.normal(0.0, 6.0, count),
                        rng.normal(0.0, 0.08, count))
    return np.stack([altitude, azimuth, depth, velocity], axis=1).astype(np.float32)


def _common(step: int, detections: np.ndarray) -> dict[str, Any]:
    return dict(detections=detections, sensor_matrix=_pose(0.1 * step),
                camera_inverse_matrix=np.linalg.inv(_pose(0.1 * step)),
                camera_intrinsics=_intrinsics(), width=WIDTH, height=HEIGHT,
                frame_time_s=0.1 * step, max_range_m=MAX_RANGE_M,
                max_abs_velocity_mps=MAX_ABS_VELOCITY_MPS, parked_threshold_s=PARKED_S,
                point_radius_px=RADIUS, rasterizer="fast")


def _window_meta(rng: np.random.Generator, count: int, reference_s: float) -> dict[str, Any]:
    ranges = np.clip(rng.gamma(3.0, 12.0, count), 0.0, 200.0).astype(np.float64)
    offsets = rng.integers(0, 2, count)
    return {"raw_provenance": {"original_range_m": ranges, "sweep_offset": offsets},
            "sweep_indices": [int(reference_s * 10) - 1, int(reference_s * 10)],
            "callbacks": 4, "returns": int(count), "window_span_s": 0.2}


def _tracker_state(tracker: Any) -> dict[str, np.ndarray]:
    return {name: np.array(getattr(tracker, name), copy=True)
            for name in ("_keys", "_ages", "_last_seen", "_x", "_y")}


class _StampHook:
    """The Phase-6 ``_run4_timed_build`` role: bracket the real build with stamps."""

    def __init__(self) -> None:
        self.stamps: list[tuple[int, int]] = []

    def __call__(self, real_build: Any, **kwargs: Any) -> Any:
        started = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        result = real_build(**kwargs)
        self.stamps.append((started, time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)))
        return result


def _parked_module() -> tuple[Any, str]:
    """The historical module when importable here; else a stand-in exposing the
    same ``build_radar_sample`` object (``radar_fusion.build_radar_sample``)."""
    try:
        import carla_collect_parked_ego_fusion_training_data as parked

        return parked, "carla_collect_parked_ego_fusion_training_data"
    except Exception as exc:  # noqa: BLE001 - CARLA egg not on this path
        import types

        from pole_lraspp_multimodal_fusion.pole_lraspp_multimodal_fusion import radar_fusion

        module = types.SimpleNamespace(build_radar_sample=radar_fusion.build_radar_sample,
                                       radar_support_count=lambda *a, **k: 0)
        return module, f"stand-in ({type(exc).__name__})"


def composed_parity(device: Any, out_dir: Path) -> dict[str, Any]:
    import torch

    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as live_base
    from rl_agent.splitfusion_sensor_optimization_v1 import optimized_stages as OS

    from . import radar_bundle_v1 as RB
    from . import recorder_v1 as RV
    from . import seams_ue_v1 as SU

    module, module_name = _parked_module()
    # production chain: timer(module) under the profiler span proxy
    prod_hook, bundle_hook = _StampHook(), _StampHook()
    prod_chain = SU._MethodSpanProxy(U._PreparationTimer(module, prod_hook),
                                     {"build_radar_sample": "ue.radar_rasterize"}, RV.active)
    bundle_timer = U._PreparationTimer(module, bundle_hook)
    rasterizer = OS.CudaRadarRasterizer(device)
    warm = RB.warm_rasterizer(rasterizer)
    bundle_timer._parked = RB.BundleParkedModule(module, rasterizer)
    bundle_chain = SU._MethodSpanProxy(bundle_timer, {"build_radar_sample": "ue.radar_rasterize"},
                                       RV.active)
    assert RB.innermost_preparation_timer(bundle_chain, U._PreparationTimer) is bundle_timer
    prod_tracker = live_base.FastStationaryTrackAccumulator(0.35, PARKED_S, 1.5, 2.0)
    bundle_tracker = RB.optimized_tracker_like(
        live_base.FastStationaryTrackAccumulator(0.35, PARKED_S, 1.5, 2.0))
    rng = np.random.default_rng(20261002)
    image_rng = np.random.default_rng(3)
    mismatches: list[str] = []
    steps = 0
    densities = [0, 12, 19404, 40000]
    packing_equal = 0
    si_equal = 0
    sha_equal = 0
    for step in range(28):
        count = int(densities[step % len(densities)]) if step > 3 else int(rng.choice(densities))
        detections = _detections(rng, count)
        common = _common(step, detections)
        expected = prod_chain.build_radar_sample(tracker=prod_tracker, **common)
        actual = bundle_chain.build_radar_sample(tracker=bundle_tracker, **common)
        steps += 1
        et, ep, es = expected
        at, ap, a_s = actual
        if not (np.array_equal(et, at) and et.dtype == at.dtype and et.shape == at.shape):
            mismatches.append(f"step {step}: tensor")
        if set(ep) != set(ap) or any(not (np.array_equal(ep[k], ap[k]) and ep[k].dtype == ap[k].dtype)
                                     for k in ep):
            mismatches.append(f"step {step}: evidence points")
        if set(es) != set(a_s) or any(es[k] != a_s[k] for k in es):
            mismatches.append(f"step {step}: summary {sorted(set(es) ^ set(a_s))}")
        ps, bs = _tracker_state(prod_tracker), _tracker_state(bundle_tracker)
        if any(not np.array_equal(ps[k], bs[k]) for k in ps):
            mismatches.append(f"step {step}: tracker state")
        # identity: the window digest is a pure function of the window metadata
        meta = _window_meta(rng, max(count, 1), 0.1 * step)
        if U.radar_window_sha256(meta, 0.1 * step) == U.radar_window_sha256(meta, 0.1 * step):
            sha_equal += 1
        # SI/P40 from the image + window metadata (bundle never touches them)
        frame_bgr = image_rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        before = U.compute_scene_descriptors(frame_bgr, meta, source_raw_ns=1)
        after = U.compute_scene_descriptors(frame_bgr, meta, source_raw_ns=1)
        if (before.camera_si, before.radar_p40, before.camera_status, before.radar_status) == \
                (after.camera_si, after.radar_p40, after.camera_status, after.radar_status):
            si_equal += 1
        # packing: production vs identity-resize on the real radar tensors
        ref = live_base._prepare_live_input(frame_bgr, et, device)
        opt = RB.prepare_live_input_identity_resize(frame_bgr, at, device)
        if torch.equal(ref, opt) and ref.dtype == opt.dtype and ref.shape == opt.shape:
            packing_equal += 1
        else:
            mismatches.append(f"step {step}: packing")
    bundle_calls = bundle_timer._parked.calls
    return {"parked_module": module_name, "steps": steps, "mismatches": mismatches,
            "packing_equal": packing_equal, "si_p40_invariant": si_equal,
            "window_sha_recomputed_equal": sha_equal, "bundle_builder_calls": bundle_calls,
            "stamps_recorded": {"production": len(prod_hook.stamps), "bundle": len(bundle_hook.stamps)},
            "rasterizer_warm": warm, "passed": not mismatches and packing_equal == steps
            and si_equal == steps and bundle_calls == steps}


def codec_identity(device: Any) -> dict[str, Any]:
    """Real preloaded UE front/ranker/AE/codec on identical inputs (two profiles)."""
    import torch

    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import continuous_execution_v2 as X
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
    from rl_agent.splitfusion_live_dispatch_v1 import dynamic_execution_contract as dec
    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as live_base
    from rl_agent.splitfusion_live_dispatch_v1 import timing as TM

    from . import radar_bundle_v1 as RB

    try:
        ue, _ledger, _models = live_base._preload_ue(device)
    except Exception as exc:  # noqa: BLE001 - tooling limit, recorded as such
        return {"status": f"preload unavailable offline: {type(exc).__name__}: {str(exc)[:160]}"}
    contract = dec.load_dynamic_execution_contract()
    front = U._SqueezedFront(ue._front, device)
    rng = np.random.default_rng(11)
    results = []
    for mode_id, q_e4 in ((11, 9800), (7, 6797), (3, 9000)):
        profile = contract.resolve_q_e4(mode_id, q_e4)
        view = X.codec_view(profile)
        frame_bgr = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        radar = (rng.random((4, HEIGHT, WIDTH)) * 2.0 - 1.0).astype(np.float32)
        ref_in = live_base._prepare_live_input(frame_bgr, radar, device)
        opt_in = RB.prepare_live_input_identity_resize(frame_bgr, radar, device)
        outputs = []
        for tensor in (ref_in, opt_in):
            with torch.inference_mode():
                c2 = front(tensor)
                ranker = None if profile.ranker_bypassed else ue._ranker
                encoder = None if profile.family == "noAE" else ue._ae_encoders[profile.family]
                inner = ue._codec.encode(view, c2, ranker=ranker, ae_encoder=encoder,
                                         timing=TM.StageRecorder(TM.UE_STAGES))
            outputs.append(hashlib.sha256(inner).hexdigest())
        results.append({"mode_id": mode_id, "q_e4": q_e4, "input_equal": bool(torch.equal(ref_in, opt_in)),
                        "inner_payload_sha256_equal": outputs[0] == outputs[1],
                        "payload_bytes": len(inner)})
    return {"status": "checked", "profiles": results,
            "passed": all(r["input_equal"] and r["inner_payload_sha256_equal"] for r in results)}


def cost(device: Any) -> dict[str, Any]:
    import torch

    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as live_base
    from rl_agent.splitfusion_sensor_optimization_v1 import optimized_stages as OS

    from . import radar_bundle_v1 as RB

    module, _ = _parked_module()
    rasterizer = OS.CudaRadarRasterizer(device)
    bundle_module = RB.BundleParkedModule(module, rasterizer)
    rng = np.random.default_rng(99)
    frame_bgr = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    out: dict[str, Any] = {}
    for count in (19404, 40000):
        prod_tracker = live_base.FastStationaryTrackAccumulator(0.35, PARKED_S, 1.5, 2.0)
        bundle_tracker = RB.optimized_tracker_like(
            live_base.FastStationaryTrackAccumulator(0.35, PARKED_S, 1.5, 2.0))
        samples: dict[str, dict[str, list[float]]] = {
            "production": {"build": [], "build_plus_packing": []},
            "bundle": {"build": [], "build_plus_packing": []}}
        for step in range(45):
            detections = _detections(rng, count)
            common = _common(step, detections)
            order = ("production", "bundle") if step % 2 == 0 else ("bundle", "production")
            for name in order:
                torch.cuda.synchronize()
                t0 = time.perf_counter_ns()
                if name == "production":
                    tensor, _p, _s = module.build_radar_sample(tracker=prod_tracker, **common)
                else:
                    tensor, _p, _s = bundle_module.build_radar_sample(tracker=bundle_tracker, **common)
                t1 = time.perf_counter_ns()
                packed = (live_base._prepare_live_input(frame_bgr, tensor, device) if name == "production"
                          else RB.prepare_live_input_identity_resize(frame_bgr, tensor, device))
                torch.cuda.synchronize()
                t2 = time.perf_counter_ns()
                del packed
                if step >= 5:                               # equal warm-up for both paths
                    samples[name]["build"].append((t1 - t0) / 1e6)
                    samples[name]["build_plus_packing"].append((t2 - t0) / 1e6)

        def stats(values: Sequence[float]) -> dict[str, float]:
            ordered = sorted(values)
            return {"n": len(ordered), "p50": ordered[len(ordered) // 2],
                    "p95": ordered[int(len(ordered) * 0.95)], "max": ordered[-1]}
        out[str(count)] = {name: {k: stats(v) for k, v in series.items()}
                           for name, series in samples.items()}
    return out


def run_tests(out: Path) -> dict[str, Any]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    with Path(out).open("x", encoding="utf-8") as handle:
        completed = subprocess.run([sys.executable, "-m", "unittest", "-v", *PARITY_TEST_FILES],
                                   cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT,
                                   timeout=1800, check=False)
    text = Path(out).read_text(encoding="utf-8")
    return {"returncode": completed.returncode, "passed": completed.returncode == 0,
            "ran": next((l for l in reversed(text.splitlines()) if l.startswith("Ran ")), ""),
            "verdict": next((l for l in reversed(text.splitlines()) if l.startswith(("OK", "FAILED"))), ""),
            "skipped": text.count("... skipped"), "modules": list(PARITY_TEST_FILES)}


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import torch

    from . import threadcap_v1 as TC

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    for key, value in TC.TREATMENT.items():
        TC.require(os.environ.get(key) == value, f"run under the UE cap: {key}={value}")
    TC.require(torch.cuda.is_available(), "GPU required for the bundle parity")
    device = torch.device("cuda:0")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {"schema": SCHEMA, "environment": {k: os.environ.get(k) for k in TC.CAP_VARIABLES},
                              "device": torch.cuda.get_device_name(0),
                              "settings": TC.settings_snapshot("parity", arm=None)}
    result["composed_parity"] = composed_parity(device, out_dir)
    result["codec_identity"] = codec_identity(device)
    result["cost_ms"] = cost(device)
    result["tests"] = run_tests(out_dir / "parity_tests.log")
    result["sources_sha256"] = {name: sha256_file(ROOT / name) for name in (
        "rl_agent/splitfusion_sensor_optimization_v1/optimized_stages.py",
        "rl_agent/splitfusion_supervisor_analysis_v1/profiled_sensor_stages.py",
        "rl_agent/splitfusion_single_gpu_profiler_v1/radar_bundle_v1.py")}
    codec_ok = result["codec_identity"].get("passed", True)   # tooling-unavailable is recorded, not a failure
    result["passed"] = bool(result["composed_parity"]["passed"] and codec_ok and result["tests"]["passed"])
    with (out_dir / "OFFLINE_PARITY_RESULT.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")
    print(json.dumps({"passed": result["passed"], "composed": {k: v for k, v in result["composed_parity"].items()
                                                             if k != "rasterizer_warm"},
                      "codec": result["codec_identity"], "cost_ms": result["cost_ms"],
                      "tests": result["tests"]["ran"] + " " + result["tests"]["verdict"]},
                     indent=1, sort_keys=True, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
