#!/usr/bin/env python3
"""Replay a valid SceneSense split-feature payload at fixed wall-clock FPS.

This diagnostic intentionally removes CARLA simulation/sensor timing from the
loop. It builds one syntactically valid no-AE fusion feature payload from the
front model, then repeatedly sends that payload with fresh frame ids and timing
fields to an already-running fusion back-half.
"""

from __future__ import annotations

import argparse
import csv
import math
import queue
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


ABIODUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ABIODUN_ROOT))
sys.path.insert(0, str(ABIODUN_ROOT / "pole_lraspp_multimodal_fusion"))
sys.path.insert(0, str(ABIODUN_ROOT / "rl_agent" / "feature_ae"))

import carla_split_inference_udp_data_collect as od_collect  # noqa: E402
from staleness import carla_fusion_staleness_scenario as fusion_scenario  # noqa: E402


DEFAULT_CKPT = (
    ABIODUN_ROOT
    / "experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt"
)

SEND_FIELDS = (
    "wall_time_iso",
    "elapsed_s",
    "frame_id",
    "scheduled_elapsed_s",
    "send_lag_ms",
    "payload_bytes",
    "payload_chunks",
)

RESULT_FIELDS = (
    "wall_time_iso",
    "elapsed_s",
    "frame_id",
    "server_ms",
    "round_trip_result_recv_ms",
    "tail_done_to_result_recv_ms",
    "result_send_to_recv_ms_perf",
    "result_send_to_recv_ms_wall",
    "result_payload_bytes_estimate",
    "result_payload_chunks_estimate",
    "object_count",
    "mask_present",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", default="10.0.0.2")
    parser.add_argument("--remote-host", default="192.168.70.140")
    parser.add_argument("--camera-source-port", type=int, default=52001)
    parser.add_argument("--remote-port", type=int, default=51002)
    parser.add_argument("--camera-result-port", type=int, default=52004)
    parser.add_argument("--chunk-bytes", type=int, default=60000)
    parser.add_argument("--socket-timeout", type=float, default=0.05)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--idle-before-s", type=float, default=10.0)
    parser.add_argument("--cooldown-s", type=float, default=120.0)
    parser.add_argument("--frame-id-start", type=int, default=900000000)
    parser.add_argument("--fusion-checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--model-input-width", type=int, default=768)
    parser.add_argument("--model-input-height", type=int, default=432)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fov", type=float, default=120.0)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--object-hidden-channels", type=int, default=128)
    parser.add_argument("--front-device", default="cuda")
    parser.add_argument(
        "--quantization-mode",
        choices=od_collect.QUANT_MODE_CHOICES,
        default=od_collect.QUANT_MODE_PER_CHANNEL_UINT8,
    )
    parser.add_argument(
        "--entropy-coder",
        choices=od_collect.ENTROPY_CODER_CHOICES,
        default=od_collect.ENTROPY_CODER_ZLIB,
    )
    parser.add_argument("--zstd-level", type=int, default=3)
    parser.add_argument("--roi-threshold", type=float, default=0.0)
    parser.add_argument("--ae-checkpoint", default="")
    parser.add_argument(
        "--run-dir",
        default="",
        help="Output directory. Defaults under abiodun/metrics_logs/oai_payload_replay/.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Run id used for output folder and summary.",
    )
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(name)


def _make_synthetic_inputs(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic, moderately structured RGB + radar tensors."""
    yy, xx = np.mgrid[0:height, 0:width]
    x_norm = xx.astype(np.float32) / max(1, width - 1)
    y_norm = yy.astype(np.float32) / max(1, height - 1)
    rgb = np.stack(
        [
            80.0 + 120.0 * x_norm,
            60.0 + 110.0 * y_norm,
            90.0 + 80.0 * np.sin(2.0 * math.pi * x_norm) * np.cos(math.pi * y_norm),
        ],
        axis=2,
    )
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    model_w, model_h = int(width), int(height)
    radar = np.zeros((4, model_h, model_w), dtype=np.float32)
    for cx, cy, radius in (
        (0.50, 0.55, 0.055),
        (0.36, 0.48, 0.030),
        (0.64, 0.43, 0.035),
    ):
        dist = np.sqrt((x_norm - cx) ** 2 + (y_norm - cy) ** 2)
        mask = dist < radius
        radar[0, mask] = 1.0
        radar[1, mask] = np.clip(1.0 - dist[mask] / max(radius, 1e-6), 0.0, 1.0)
        radar[2, mask] = 0.35
        radar[3, mask] = 0.5
    return np.ascontiguousarray(rgb[:, :, ::-1]), radar


def build_template_payload(args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    load_args = argparse.Namespace(
        fusion_checkpoint=str(args.fusion_checkpoint),
        fusion_experiment_dir="",
        model_input_width=int(args.model_input_width),
        model_input_height=int(args.model_input_height),
        num_classes=int(args.num_classes),
        object_hidden_channels=int(args.object_hidden_channels),
        roi_threshold=float(args.roi_threshold),
        ae_checkpoint=str(args.ae_checkpoint or ""),
    )
    split_model, model_input_size = fusion_scenario.load_fusion_model(load_args, device)
    model_w, model_h = int(model_input_size[0]), int(model_input_size[1])
    frame_bgr, radar_tensor = _make_synthetic_inputs(model_w, model_h)

    rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    rgb_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    with torch.inference_mode():
        fused = fusion_scenario.prepare_fusion_input(
            frame_bgr=frame_bgr,
            radar_tensor_chw=radar_tensor,
            model_size=(model_w, model_h),
            device=device,
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
        )
        features = split_model.encode(fused)
        features = fusion_scenario._front_compress(  # pylint: disable=protected-access
            split_model,
            features,
            tuple(int(v) for v in fused.shape[-2:]),
        )

    transport_cfg = od_collect.TransportConfig(
        quantization_mode=str(args.quantization_mode),
        entropy_coder_name=str(args.entropy_coder),
        zstd_level=int(args.zstd_level),
        roi_objectness_threshold=0.0,
        bypass_rcnn_transform=False,
    )
    feature_codecs: Dict[str, object] = OrderedDict()
    serialized_features, payload_bytes_uncompressed, _, _ = od_collect.serialize_feature_maps(
        features,
        feature_codecs,
        quantization_mode=transport_cfg.quantization_mode,
        per_level_compress_probe=False,
        entropy_coder=transport_cfg.make_entropy_coder(),
    )
    camera_matrix = np.eye(4, dtype=np.float64)
    camera_intrinsics_input = fusion_scenario.intrinsics_at(
        model_w,
        model_h,
        float(args.camera_fov),
    )
    return {
        "frame_id": int(args.frame_id_start),
        "batch_size": int(fused.shape[0]),
        "model_input_size": [model_w, model_h],
        "display_size": [int(args.camera_width), int(args.camera_height)],
        "feature_shapes": {
            name: tuple(int(v) for v in tensor.shape) for name, tensor in features.items()
        },
        "features": serialized_features,
        "camera_matrix": camera_matrix,
        "camera_intrinsics_input": camera_intrinsics_input.astype(np.float64),
        "camera_sent_perf": time.perf_counter(),
        "camera_sent_wall_s": time.time(),
        "_payload_bytes_uncompressed": int(payload_bytes_uncompressed),
    }


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def receiver_loop(
    *,
    sock: od_collect.UDPMessageSocket,
    stop_event: threading.Event,
    result_queue: "queue.Queue[Dict[str, object]]",
    run_start_perf: float,
) -> None:
    while not stop_event.is_set():
        payload = sock.receive()
        if payload is None:
            continue
        recv_perf = time.perf_counter()
        recv_wall = time.time()
        if isinstance(payload, dict):
            payload["car_result_recv_perf"] = recv_perf
            payload["car_result_recv_wall_s"] = recv_wall
            try:
                payload["round_trip_result_recv_ms"] = (
                    recv_perf - float(payload["camera_sent_perf"])
                ) * 1000.0
            except Exception:
                payload["round_trip_result_recv_ms"] = float("nan")
            try:
                payload["result_send_to_recv_ms_perf"] = (
                    recv_perf - float(payload["result_send_start_perf"])
                ) * 1000.0
            except Exception:
                payload["result_send_to_recv_ms_perf"] = float("nan")
            try:
                payload["result_send_to_recv_ms_wall"] = (
                    recv_wall - float(payload["result_send_start_wall_s"])
                ) * 1000.0
            except Exception:
                payload["result_send_to_recv_ms_wall"] = float("nan")
            try:
                payload["tail_done_to_result_recv_ms"] = (
                    recv_perf - float(payload["tail_done_perf"])
                ) * 1000.0
            except Exception:
                payload["tail_done_to_result_recv_ms"] = float("nan")
            payload["_elapsed_s"] = recv_perf - run_start_perf
            result_queue.put(payload)


def percentile(values: List[float], pct: float) -> float:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return float("nan")
    k = (len(values) - 1) * pct / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return values[int(k)]
    return values[lo] * (hi - k) + values[hi] * (k - lo)


def result_payload_to_row(payload: Dict[str, object]) -> Dict[str, object]:
    return {
        "wall_time_iso": datetime.now().isoformat(timespec="milliseconds"),
        "elapsed_s": _safe_float(payload.get("_elapsed_s")),
        "frame_id": _safe_int(payload.get("frame_id")),
        "server_ms": _safe_float(payload.get("server_ms")),
        "round_trip_result_recv_ms": _safe_float(payload.get("round_trip_result_recv_ms")),
        "tail_done_to_result_recv_ms": _safe_float(payload.get("tail_done_to_result_recv_ms")),
        "result_send_to_recv_ms_perf": _safe_float(
            payload.get("result_send_to_recv_ms_perf")
        ),
        "result_send_to_recv_ms_wall": _safe_float(
            payload.get("result_send_to_recv_ms_wall")
        ),
        "result_payload_bytes_estimate": _safe_int(
            payload.get("result_payload_bytes_estimate")
        ),
        "result_payload_chunks_estimate": _safe_int(
            payload.get("result_payload_chunks_estimate")
        ),
        "object_count": len(payload.get("objects", []))
        if isinstance(payload.get("objects"), list)
        else 0,
        "mask_present": isinstance(payload.get("mask"), np.ndarray),
    }


def drain_results(
    result_queue: "queue.Queue[Dict[str, object]]",
    result_rows: List[Dict[str, object]],
) -> int:
    drained = 0
    while True:
        try:
            payload = result_queue.get_nowait()
        except queue.Empty:
            break
        result_rows.append(result_payload_to_row(payload))
        drained += 1
    return drained


def write_summary(
    *,
    path: Path,
    run_id: str,
    args: argparse.Namespace,
    send_rows: List[Dict[str, object]],
    result_rows: List[Dict[str, object]],
) -> None:
    elapsed = [
        _safe_float(row["elapsed_s"])
        for row in send_rows
        if math.isfinite(_safe_float(row["elapsed_s"]))
    ]
    send_duration = (max(elapsed) - min(elapsed)) if len(elapsed) >= 2 else float("nan")
    actual_fps = (len(send_rows) / send_duration) if send_duration > 0 else float("nan")
    payload_bytes = [_safe_float(row["payload_bytes"]) for row in send_rows]
    rtt = [_safe_float(row["round_trip_result_recv_ms"]) for row in result_rows]
    server = [_safe_float(row["server_ms"]) for row in result_rows]
    down = [_safe_float(row["tail_done_to_result_recv_ms"]) for row in result_rows]
    nominal_mbps = float(args.fps) * percentile(payload_bytes, 50) * 8.0 / 1e6
    actual_mbps = actual_fps * percentile(payload_bytes, 50) * 8.0 / 1e6
    lines = [
        "# OAI Payload Replay Probe",
        "",
        f"Run id: `{run_id}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| target FPS | {float(args.fps):.3f} |",
        f"| sent frames | {len(send_rows)} |",
        f"| returned results | {len(result_rows)} |",
        f"| delivery | {(100.0 * len(result_rows) / max(1, len(send_rows))):.2f}% |",
        f"| actual send duration | {send_duration:.3f} s |",
        f"| actual FPS | {actual_fps:.3f} |",
        f"| payload p50 | {percentile(payload_bytes, 50):.1f} bytes |",
        f"| payload chunks p50 | {percentile([_safe_float(r['payload_chunks']) for r in send_rows], 50):.1f} |",
        f"| nominal target offered load | {nominal_mbps:.3f} Mbps |",
        f"| actual offered load | {actual_mbps:.3f} Mbps |",
        f"| RTT p50 | {percentile(rtt, 50):.3f} ms |",
        f"| RTT p95 | {percentile(rtt, 95):.3f} ms |",
        f"| back/server p50 | {percentile(server, 50):.3f} ms |",
        f"| downlink p50 | {percentile(down, 50):.3f} ms |",
        "",
        "Interpretation: this replay removes live CARLA frame generation. If actual FPS tracks target FPS, it is a cleaner OAI transport stress input than the live queue-probe mode.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_id = str(args.run_id or "").strip() or datetime.now().strftime(
        "oai_payload_replay_%Y%m%d_%H%M%S"
    )
    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if str(args.run_dir or "").strip()
        else ABIODUN_ROOT / "metrics_logs" / "oai_payload_replay" / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    device = _device(str(args.front_device))
    print(f"[Replay] building template on {device} from {args.fusion_checkpoint}")
    template = build_template_payload(args, device)

    send_sock = od_collect.UDPMessageSocket(
        bind_port=int(args.camera_source_port),
        remote_port=int(args.remote_port),
        chunk_bytes=int(args.chunk_bytes),
        socket_timeout=float(args.socket_timeout),
        host=str(args.bind_host),
        remote_host=str(args.remote_host),
        entropy_coder=od_collect.make_entropy_coder(
            str(args.entropy_coder),
            zstd_level=int(args.zstd_level),
        ),
    )
    recv_sock = od_collect.UDPMessageSocket(
        bind_port=int(args.camera_result_port),
        remote_port=None,
        chunk_bytes=int(args.chunk_bytes),
        socket_timeout=float(args.socket_timeout),
        host=str(args.bind_host),
        entropy_coder=od_collect.make_entropy_coder(
            str(args.entropy_coder),
            zstd_level=int(args.zstd_level),
        ),
    )

    send_path = run_dir / "send_events.csv"
    result_path = run_dir / "result_events.csv"
    summary_path = run_dir / "REPLAY_RESULTS.md"
    stop_event = threading.Event()
    result_queue: "queue.Queue[Dict[str, object]]" = queue.Queue()
    run_start_perf = time.perf_counter()
    receiver = threading.Thread(
        target=receiver_loop,
        kwargs={
            "sock": recv_sock,
            "stop_event": stop_event,
            "result_queue": result_queue,
            "run_start_perf": run_start_perf,
        },
        daemon=True,
    )
    receiver.start()

    send_rows: List[Dict[str, object]] = []
    result_rows: List[Dict[str, object]] = []
    print(f"[Replay] output: {run_dir}")
    print(
        f"[Replay] target={args.fps:.2f} FPS frames={args.frames} "
        f"idle={args.idle_before_s:.1f}s cooldown={args.cooldown_s:.1f}s"
    )
    try:
        if float(args.idle_before_s) > 0:
            time.sleep(max(0.0, float(args.idle_before_s)))
        send_start_perf = time.perf_counter()
        period = 1.0 / max(0.1, float(args.fps))
        for i in range(int(args.frames)):
            scheduled_elapsed = i * period
            scheduled_perf = send_start_perf + scheduled_elapsed
            sleep_s = scheduled_perf - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            frame_id = int(args.frame_id_start) + i
            payload = dict(template)
            payload.pop("_payload_bytes_uncompressed", None)
            payload["frame_id"] = frame_id
            payload["camera_sent_perf"] = time.perf_counter()
            payload["camera_sent_wall_s"] = time.time()
            payload_bytes, payload_chunks = send_sock.send(payload)
            elapsed_s = time.perf_counter() - run_start_perf
            row = {
                "wall_time_iso": datetime.now().isoformat(timespec="milliseconds"),
                "elapsed_s": elapsed_s,
                "frame_id": frame_id,
                "scheduled_elapsed_s": float(args.idle_before_s) + scheduled_elapsed,
                "send_lag_ms": (
                    time.perf_counter() - scheduled_perf
                )
                * 1000.0,
                "payload_bytes": int(payload_bytes),
                "payload_chunks": int(payload_chunks),
            }
            send_rows.append(row)
            drain_results(result_queue, result_rows)

        cooldown_deadline = time.perf_counter() + max(0.0, float(args.cooldown_s))
        while time.perf_counter() < cooldown_deadline:
            try:
                payload = result_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            result_rows.append(result_payload_to_row(payload))
    finally:
        stop_event.set()
        receiver.join(timeout=1.0)
        send_sock.close()
        recv_sock.close()

    # Drain anything received just before stop.
    drain_results(result_queue, result_rows)

    with send_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SEND_FIELDS)
        writer.writeheader()
        writer.writerows(send_rows)
    with result_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(result_rows)
    write_summary(
        path=summary_path,
        run_id=run_id,
        args=args,
        send_rows=send_rows,
        result_rows=result_rows,
    )
    print(f"[Replay] send CSV: {send_path}")
    print(f"[Replay] result CSV: {result_path}")
    print(f"[Replay] summary: {summary_path}")
    print(f"[Replay] delivery={len(result_rows)}/{len(send_rows)}")


if __name__ == "__main__":
    main()
