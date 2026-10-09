"""Addendum 7: deterministic runtime pre-warming before READY / first frame.

Every registered executable path runs once, with exact live shapes and devices,
before the edge publishes READY and before the UE admits any scientific frame.
The paths are the 12 joint modes (noAE/AE128/AE64/AE32 × UINT4/UINT6/UINT8),
each at the lower, middle and upper registered ``q_e4`` of its support, since
``keep_count`` (and therefore tensor shapes) follows q.

* UE: :func:`warm_ue` drives the real 7-channel input preparation, front,
  ranker/top-k, AE encoder, quantizer and zstd codec through
  ``ContinuousUERuntimeV2.prepare`` with a synthetic, never-transmitted identity.
* Edge: :func:`warm_edge` builds a valid synthetic SFD4 wire per path and runs
  it through the real ``Run4EdgeProcessorV2.process``: codec inspect, AE decode
  (tensor reconstruction), model tail, post-processing and map-update
  construction. It never publishes and never submits an evaluation, because the
  warm frame is not reward-requested. The runtime's frame counters and
  context-session state are swapped out for warm-up and restored, and the
  deadline guard is bypassed so a cold first call is not aborted half-warm.

``torch.cuda.synchronize()`` brackets each timed path. Synthetic inputs come
from a local, seeded generator, so the global torch/numpy RNG and every policy
RNG are untouched. The reports are create-only JSON with per-path timings and
payload/update hashes.

Importing this module performs no I/O and initializes nothing.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

WARM_SESSION = "00000000-0000-4000-8000-00000000c0de"
WARM_LINEAGE = "0" * 64
WARM_STREAM = "run4_phase6_prewarm"
WARM_SEED = 20260930
FRAME_SHAPE = (720, 1280, 3)          # live RGB camera (image_size 1280x720)
RADAR_SHAPE = (4, 448, 768)           # live rasterized radar tensor
C2_SHAPE = (256, 112, 192)            # frozen split boundary
PASSES = ("first_pass", "hot_repeat")  # addendum 8: identical HOT repeat after pass 1
SCORE_SHAPE = (112, 192)              # frozen ranker score map


class PrewarmError(RuntimeError):
    """A registered path failed to warm; READY must not be published."""


def warm_paths(contract: Any) -> list[dict[str, Any]]:
    """Every (mode, q_e4) warm path: lower/mid/upper q of each mode's support."""
    from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import (
        MODELED_SMOKE_SUPPORT,
    )

    paths = []
    for mode_id, (lower, upper) in enumerate(MODELED_SMOKE_SUPPORT.mode_q_e4_bounds):
        lower, upper = int(lower), int(upper)
        for q_e4 in sorted({lower, (lower + upper) // 2, upper}):
            profile = contract.resolve_q_e4(mode_id, q_e4)
            paths.append({"mode_id": int(mode_id), "q_e4": int(profile.q_e4),
                          "family": profile.family, "quantizer": profile.quantizer,
                          "keep_count": int(profile.keep_count), "profile": profile})
    modes = {p["mode_id"] for p in paths}
    if modes != set(range(12)):
        raise PrewarmError(f"warm paths do not cover all 12 modes: {sorted(modes)}")
    return paths


def warm_identity(index: int, *, capture_ns: int) -> Any:
    """Synthetic hold identity: never enters the engine, controller or ledgers."""
    from . import continuous_execution_v2 as X

    return X.FrameIdentityV2(
        session_uuid=WARM_SESSION, controller_lineage_sha256=WARM_LINEAGE,
        decision_seq=0, ticket_seq=0, frame_id=int(index) + 1, tensor_seq=int(index) + 1,
        capture_timestamp_ns=int(capture_ns), reward_requested=False)


def _sync_default() -> None:
    import torch

    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def _timed(fn: Callable[[], Any], *, sync: Callable[[], None],
           clock: Callable[[], int]) -> tuple[Any, float]:
    sync()
    started = clock()
    result = fn()
    sync()
    return result, (clock() - started) / 1e6


def synthetic_sensor_inputs() -> tuple[Any, Any]:
    """Deterministic live-shaped RGB frame and radar tensor (local RNG only)."""
    import numpy as np

    rng = np.random.default_rng(WARM_SEED)
    frame = rng.integers(0, 256, size=FRAME_SHAPE, dtype=np.uint8)
    radar = rng.random(size=RADAR_SHAPE, dtype=np.float32)
    return frame, radar


def warm_ue(continuous_ue: Any, contract: Any, *,
            prepare_input: Callable[[Any, Any], Any],
            sync: Callable[[], None] = _sync_default,
            clock: Callable[[], int] = time.perf_counter_ns) -> dict[str, Any]:
    """Warm every UE path (first pass), then one identical HOT repeat (addendum 8)."""
    frame, radar = synthetic_sensor_inputs()
    input_7ch, input_ms = _timed(lambda: prepare_input(frame, radar), sync=sync, clock=clock)
    paths = warm_paths(contract)
    rows = [{k: path[k] for k in ("mode_id", "q_e4", "family", "quantizer", "keep_count")}
            for path in paths]
    for pass_index, label in enumerate(PASSES):
        for index, path in enumerate(paths):
            identity = warm_identity(pass_index * len(paths) + index,
                                     capture_ns=time.time_ns())
            try:
                prepared, elapsed = _timed(
                    lambda: continuous_ue.prepare(path["profile"], input_7ch, identity),
                    sync=sync, clock=clock)
            except Exception as exc:
                raise PrewarmError(f"UE {label} path mode {path['mode_id']} q_e4 "
                                   f"{path['q_e4']} failed: {type(exc).__name__}: {exc}") from exc
            inner = prepared.envelope.inner_payload
            rows[index][f"{label}_ms"] = elapsed
            rows[index][f"{label}_inner_sha256"] = hashlib.sha256(inner).hexdigest()
            rows[index]["inner_bytes"] = len(inner)
    for row in rows:
        row["elapsed_ms"] = row["first_pass_ms"]
        row["hot_identical_payload"] = row["first_pass_inner_sha256"] == row["hot_repeat_inner_sha256"]
    return {"schema": "scenesense.run4_live_v2.prewarm_ue.v2", "side": "UE",
            "input_preparation_ms": input_ms,
            "input_shape": list(getattr(input_7ch, "shape", ())),
            "paths": rows, "modes_warmed": sorted({r["mode_id"] for r in rows}),
            "timing_summary": timing_summary(rows),
            "completed": all("first_pass_ms" in r and "hot_repeat_ms" in r for r in rows)
                         and len(rows) == len(paths)}


class _WarmRanker:
    """Deterministic frozen-shape scores; the edge has no ranker model."""

    def score_cells(self, c2: Any) -> Any:
        import torch

        cells = SCORE_SHAPE[0] * SCORE_SHAPE[1]
        return torch.arange(cells, 0, -1, dtype=torch.float32,
                            device=c2.device).reshape(SCORE_SHAPE)


@contextlib.contextmanager
def isolated_edge_runtime(runtime: Any, *, unguarded_tail: Any = None) -> Iterator[None]:
    """Swap frame counters / context session (and the deadline guard) for warm-up."""
    from rl_agent.splitfusion_live_dispatch_v1.frame_context import (
        FrameContextSessionValidator,
    )

    saved = (runtime._counters, runtime._context_session, runtime._detached_tail)
    runtime._counters = copy.deepcopy(runtime._counters)
    runtime._context_session = FrameContextSessionValidator()
    if unguarded_tail is not None:
        runtime._detached_tail = unguarded_tail
    try:
        yield
    finally:
        runtime._counters, runtime._context_session, runtime._detached_tail = saved


def warm_edge(processor: Any, runtime: Any, contract: Any, *, device: Any,
              encoders: Mapping[str, Any], codec: Any, unguarded_tail: Any = None,
              sync: Callable[[], None] = _sync_default,
              clock: Callable[[], int] = time.perf_counter_ns) -> dict[str, Any]:
    """Warm every edge execution path through the real processor."""
    import torch

    from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1

    from . import continuous_execution_v2 as X
    from . import run4_live_wire_v2 as W

    generator = torch.Generator(device=device)
    generator.manual_seed(WARM_SEED)
    c2 = torch.rand(C2_SHAPE, generator=generator, device=device, dtype=torch.float32)
    synthesizer = X.ContinuousUERuntimeV2(contract, front=lambda _input: c2,
                                          ranker=_WarmRanker(), ae_encoders=dict(encoders),
                                          codec=codec)
    paths = warm_paths(contract)
    rows = [{k: path[k] for k in ("mode_id", "q_e4", "family", "quantizer", "keep_count")}
            for path in paths]
    with isolated_edge_runtime(runtime, unguarded_tail=unguarded_tail):
        for pass_index, label in enumerate(PASSES):
            for index, path in enumerate(paths):
                capture_ns = time.time_ns()
                identity = warm_identity(pass_index * len(paths) + index,
                                         capture_ns=capture_ns)
                try:
                    prepared = synthesizer.prepare(path["profile"], None, identity)
                    context = build_frame_context_v1(
                        stream_id=WARM_STREAM, frame_id=identity.frame_id,
                        sequence_id=identity.tensor_seq, capture_timestamp_ns=capture_ns,
                        ego_world_x=0.0, ego_world_y=0.0, ego_world_z=0.0,
                        ego_world_pitch=0.0, ego_world_yaw=0.0, ego_world_roll=0.0)
                    wire = W.pack_sfd4(prepared.envelope, context)
                    processed, elapsed = _timed(
                        lambda: processor.process(wire, edge_timing={}), sync=sync,
                        clock=clock)
                except Exception as exc:
                    raise PrewarmError(f"edge {label} path mode {path['mode_id']} q_e4 "
                                       f"{path['q_e4']} failed: {type(exc).__name__}: "
                                       f"{exc}") from exc
                if processed.evaluation is not None:
                    raise PrewarmError("a warm frame produced an evaluation ticket")
                update = json.dumps(processed.update, sort_keys=True, default=str).encode()
                row = rows[index]
                row[f"{label}_ms"] = elapsed
                row[f"{label}_inner_sha256"] = prepared.envelope.inner_payload_sha256
                row[f"{label}_wire_sha256"] = hashlib.sha256(wire).hexdigest()
                row[f"{label}_map_update_sha256"] = hashlib.sha256(update).hexdigest()
                row["wire_bytes"] = len(wire)
                row["record_count"] = len(processed.update.get("records") or ())
    for row in rows:
        row["elapsed_ms"] = row["first_pass_ms"]
        row["wire_sha256"] = row["first_pass_wire_sha256"]
        row["map_update_sha256"] = row["first_pass_map_update_sha256"]
        row["hot_identical_payload"] = row["first_pass_inner_sha256"] == row["hot_repeat_inner_sha256"]
    return {"schema": "scenesense.run4_live_v2.prewarm_edge.v2", "side": "EDGE",
            "paths": rows, "modes_warmed": sorted({r["mode_id"] for r in rows}),
            "timing_summary": timing_summary(rows),
            "completed": all("first_pass_ms" in r and "hot_repeat_ms" in r for r in rows)
                         and len(rows) == len(paths)}


def _stats(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return {"n": 0, "p50": None, "p95": None, "max": None}

    def pick(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]
    return {"n": len(ordered), "p50": pick(0.5), "p95": pick(0.95), "max": ordered[-1]}


def timing_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Diagnostic first-pass vs hot-repeat timing; no acceptance threshold."""
    summary: dict[str, Any] = {"overall": {}, "per_mode": {}}
    for label in PASSES:
        key = f"{label}_ms"
        summary["overall"][label] = _stats([r[key] for r in rows if key in r])
        for mode in sorted({r["mode_id"] for r in rows}):
            summary["per_mode"].setdefault(str(mode), {})[label] = _stats(
                [r[key] for r in rows if r["mode_id"] == mode and key in r])
    mode11 = [r for r in rows if r["mode_id"] == 11]
    if mode11:
        top = max(mode11, key=lambda r: r["q_e4"])
        summary["mode11_highest_q"] = {k: top.get(k) for k in (
            "q_e4", "family", "quantizer", "first_pass_ms", "hot_repeat_ms")}
    return summary


def write_report_create_only(path: Path, report: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(report), sort_keys=True, indent=1, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def publish_ready_after_warmup(warm: Callable[[], Mapping[str, Any]],
                               write_ready: Callable[[], None], *,
                               report_path: Optional[Path] = None) -> Mapping[str, Any]:
    """READY is written only after every registered path warmed successfully."""
    report = warm()
    if not report.get("completed") or sorted(report.get("modes_warmed") or ()) != list(
            range(12)):
        raise PrewarmError("warm-up incomplete; READY refused")
    if report_path is not None:
        write_report_create_only(report_path, report)
    write_ready()
    return report


__all__ = [
    "PrewarmError",
    "warm_paths",
    "warm_ue",
    "warm_edge",
    "isolated_edge_runtime",
    "publish_ready_after_warmup",
    "write_report_create_only",
    "synthetic_sensor_inputs",
]
