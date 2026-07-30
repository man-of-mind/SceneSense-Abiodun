#!/usr/bin/env python3
"""Replay synthetic edge->spatial-map packets at an exact offered load.

This isolates the spatial-map ingest/update path from CARLA sensor preparation.
Packets use the same zlib-compressed JSON schema consumed by
spatial_map_server_moving_ego_uplink_only_baseline.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List


SPATIAL_STREAM_SCHEMA = "fusion_object_spatial_map.v1"


SEND_FIELDS = (
    "wall_time_iso",
    "frame_index",
    "frame_id",
    "stream_id",
    "target_fps",
    "scheduled_elapsed_s",
    "actual_elapsed_s",
    "send_lag_ms",
    "packet_bytes",
    "object_count",
    "synthetic_backbone_to_front_send_ms",
    "synthetic_front_to_edge_ms",
    "synthetic_edge_queue_ms",
    "synthetic_tail_ms",
    "synthetic_model_to_map_publish_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Spatial-map UDP host.")
    parser.add_argument("--port", type=int, default=39201, help="Spatial-map UDP port.")
    parser.add_argument("--fps", type=float, required=True, help="Offered packet rate.")
    parser.add_argument("--frames", type=int, default=300, help="Number of packets to send.")
    parser.add_argument("--stream-id", default="replay_spatial", help="Spatial stream id.")
    parser.add_argument("--frame-id-start", type=int, default=1_000_000)
    parser.add_argument("--objects", type=int, default=3, help="Synthetic objects per packet.")
    parser.add_argument("--output-csv", required=True, help="Replay send metrics CSV.")
    parser.add_argument(
        "--backbone-to-front-send-ms",
        type=float,
        default=5.0,
        help="Synthetic front model+serialize time before feature send.",
    )
    parser.add_argument(
        "--front-to-edge-ms",
        type=float,
        default=7.0,
        help="Synthetic uplink/front-send to edge-receive time.",
    )
    parser.add_argument(
        "--edge-queue-ms",
        type=float,
        default=0.0,
        help="Synthetic edge queue wait before tail.",
    )
    parser.add_argument(
        "--tail-ms",
        type=float,
        default=10.0,
        help="Synthetic edge tail/model decode time.",
    )
    parser.add_argument(
        "--payload-pad-bytes",
        type=int,
        default=0,
        help="Optional JSON padding field size before zlib compression.",
    )
    return parser.parse_args()


def synthetic_objects(frame_id: int, count: int, stream_id: str) -> List[Dict[str, object]]:
    objects: List[Dict[str, object]] = []
    for idx in range(max(0, int(count))):
        x = 18.0 + 7.5 * idx + 0.1 * (frame_id % 17)
        y = -2.0 + 2.0 * idx
        objects.append(
            {
                "id": f"{stream_id}:{frame_id}:{idx}",
                "type": "Vehicle" if idx % 3 != 2 else "Pedestrian",
                "motion_state": "moving" if idx % 2 == 0 else "parked",
                "score": 0.72 - 0.04 * min(idx, 5),
                "location": {"x": x, "y": y, "z": 0.0},
                "dimensions": {
                    "length": 4.5 if idx % 3 != 2 else 0.5,
                    "width": 1.9 if idx % 3 != 2 else 0.5,
                    "height": 1.6 if idx % 3 != 2 else 1.7,
                },
                "yaw_deg": 0.0,
                "parked_score": 0.2 if idx % 2 == 0 else 0.8,
                "radar_support_score": 0.8,
                "bbox_xyxy": [320 + idx * 40, 250, 390 + idx * 40, 340],
            }
        )
    return objects


def build_packet(args: argparse.Namespace, frame_index: int, send_perf: float) -> bytes:
    frame_id = int(args.frame_id_start) + int(frame_index)
    backbone_to_front = max(0.0, float(args.backbone_to_front_send_ms)) / 1000.0
    front_to_edge = max(0.0, float(args.front_to_edge_ms)) / 1000.0
    edge_queue = max(0.0, float(args.edge_queue_ms)) / 1000.0
    tail = max(0.0, float(args.tail_ms)) / 1000.0

    t_map_publish = float(send_perf)
    t_tail_done = t_map_publish
    t_tail_start = t_tail_done - tail
    t_edge_recv = t_tail_start - edge_queue
    t_front_send = t_edge_recv - front_to_edge
    t_front_payload_ready = t_front_send
    t_front_model_done = t_front_payload_ready
    t_backbone_input = t_front_send - backbone_to_front
    t_capture = t_backbone_input

    synthetic_model_to_map_publish_ms = (t_map_publish - t_backbone_input) * 1000.0
    payload: Dict[str, object] = {
        "schema": SPATIAL_STREAM_SCHEMA,
        "source_script": Path(__file__).name,
        "stream_id": str(args.stream_id),
        "node_id": str(args.stream_id),
        "traffic_light_id": "",
        "traffic_light_actor_id": -1,
        "traffic_light_opendrive_id": "",
        "frame_id": frame_id,
        "timestamp": time.time(),
        "carla_timestamp": frame_id / max(0.1, float(args.fps)),
        "camera": {
            "x": 0.0,
            "y": 0.0,
            "z": 1.55,
            "pitch": -4.0,
            "yaw": 0.0,
            "roll": 0.0,
            "width": 1280,
            "height": 720,
            "fov": 120.0,
            "matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "segmentation": {"present": False},
        "objects": synthetic_objects(frame_id, int(args.objects), str(args.stream_id)),
        "latency": {
            "front_ms": float(args.backbone_to_front_send_ms),
            "back_ms": float(args.tail_ms),
            "round_trip_ms": 0.0,
            "payload_bytes": 0,
            "payload_bytes_uncompressed": 0,
            "payload_chunks": 0,
            "front_to_edge_ms": float(args.front_to_edge_ms),
            "capture_to_tail_done_ms": synthetic_model_to_map_publish_ms,
        },
        "timing": {
            "t_capture_perf": t_capture,
            "t_front_start_perf": t_backbone_input,
            "t_backbone_input_perf": t_backbone_input,
            "t_front_model_done_perf": t_front_model_done,
            "t_front_payload_ready_perf": t_front_payload_ready,
            "t_front_send_perf": t_front_send,
            "t_edge_recv_perf": t_edge_recv,
            "t_tail_start_perf": t_tail_start,
            "t_tail_done_perf": t_tail_done,
            "t_map_publish_perf": t_map_publish,
            "capture_to_backbone_input_ms": 0.0,
            "model_preprocess_ms": 0.0,
            "front_backbone_ms": float(args.backbone_to_front_send_ms),
            "feature_serialize_ms": 0.0,
            "backbone_input_to_front_send_ms": float(args.backbone_to_front_send_ms),
            "front_to_edge_ms": float(args.front_to_edge_ms),
            "edge_queue_ms": float(args.edge_queue_ms),
            "tail_ms": float(args.tail_ms),
            "edge_to_map_publish_ms": float(args.edge_queue_ms) + float(args.tail_ms),
            "backbone_input_to_edge_recv_ms": (
                float(args.backbone_to_front_send_ms) + float(args.front_to_edge_ms)
            ),
            "backbone_input_to_tail_done_ms": synthetic_model_to_map_publish_ms,
            "backbone_input_to_map_publish_ms": synthetic_model_to_map_publish_ms,
        },
    }
    if int(args.payload_pad_bytes) > 0:
        payload["padding"] = "x" * int(args.payload_pad_bytes)
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
    return zlib.compress(encoded, level=1)


def main() -> None:
    args = parse_args()
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    remote = (str(args.host), int(args.port))
    period_s = 1.0 / max(0.1, float(args.fps))
    start_perf = time.perf_counter()
    synthetic_model_to_map_publish_ms = (
        float(args.backbone_to_front_send_ms)
        + float(args.front_to_edge_ms)
        + float(args.edge_queue_ms)
        + float(args.tail_ms)
    )

    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(SEND_FIELDS))
        writer.writeheader()
        for frame_index in range(max(0, int(args.frames))):
            scheduled_elapsed_s = frame_index * period_s
            scheduled_perf = start_perf + scheduled_elapsed_s
            sleep_s = scheduled_perf - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            send_perf = time.perf_counter()
            packet = build_packet(args, frame_index, send_perf)
            sock.sendto(packet, remote)
            actual_elapsed_s = time.perf_counter() - start_perf
            writer.writerow(
                {
                    "wall_time_iso": datetime.now().isoformat(timespec="milliseconds"),
                    "frame_index": int(frame_index),
                    "frame_id": int(args.frame_id_start) + int(frame_index),
                    "stream_id": str(args.stream_id),
                    "target_fps": float(args.fps),
                    "scheduled_elapsed_s": float(scheduled_elapsed_s),
                    "actual_elapsed_s": float(actual_elapsed_s),
                    "send_lag_ms": float((actual_elapsed_s - scheduled_elapsed_s) * 1000.0),
                    "packet_bytes": int(len(packet)),
                    "object_count": int(args.objects),
                    "synthetic_backbone_to_front_send_ms": float(args.backbone_to_front_send_ms),
                    "synthetic_front_to_edge_ms": float(args.front_to_edge_ms),
                    "synthetic_edge_queue_ms": float(args.edge_queue_ms),
                    "synthetic_tail_ms": float(args.tail_ms),
                    "synthetic_model_to_map_publish_ms": float(synthetic_model_to_map_publish_ms),
                }
            )
            fp.flush()
    sock.close()
    print(
        f"sent {int(args.frames)} packets to {args.host}:{args.port} "
        f"at {float(args.fps):.1f} FPS; metrics={output}"
    )


if __name__ == "__main__":
    main()
