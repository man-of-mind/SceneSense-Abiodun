#!/usr/bin/env python3

"""
Multi-sensor UDP receiver -- OpenCV display variant.

Same shape as carla_multisensor_udp_gstreamer_receiver_v2.py with one
architectural change: the three camera streams are decoded by GStreamer
(software avdec_h265) but DISPLAYED by OpenCV (cv2.imshow). The pipeline
ends in `appsink` instead of `autovideosink`, and `rtpjitterbuffer` is
removed.

Why:
  - `rtpjitterbuffer` did strict per-packet validation that some
    FFmpeg-emitted RTP H.265 packets failed, producing constant
    "Received invalid RTP payload, dropping" warnings. Removing it
    eliminates those errors.
  - `autovideosink` typically picks GL-backed sinks (xvimagesink,
    glimagesink) that AnyDesk struggles to capture cleanly. cv2.imshow
    uses a plain framebuffer-style window AnyDesk handles much better.
  - `max-buffers=1 drop=true` on appsink means we always display the
    freshest decoded frame instead of building up a backlog.

Pairs with the existing GStreamer multi-sensor sender in
PythonAPI/neu_collab/.

Point-sensor (radar / semantic_lidar / lidar) windows are unchanged --
those use GstBgrVideoRenderer, left alone for now. They can be
migrated to cv2.imshow in a follow-up if needed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import signal
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


SENSOR_ORDER = ("rgb", "semantic", "instance", "radar", "semantic_lidar", "lidar")
CAMERA_SENSORS = frozenset(("rgb", "semantic", "instance"))
POINT_SENSORS = frozenset(("radar", "semantic_lidar", "lidar"))

SENSOR_LABELS = {
    "rgb": "RGB Camera",
    "semantic": "Semantic Segmentation Camera",
    "instance": "Instance Segmentation Camera",
    "radar": "Radar",
    "semantic_lidar": "Semantic LiDAR",
    "lidar": "LiDAR",
}

DEFAULT_PORT_OFFSETS = {
    "rgb": 0,
    "semantic": 1,
    "instance": 2,
    "radar": 3,
    "semantic_lidar": 4,
    "lidar": 5,
}

SENSOR_IDS = {name: index + 1 for index, name in enumerate(SENSOR_ORDER)}
SENSOR_NAMES_BY_ID = {value: key for key, value in SENSOR_IDS.items()}
PACKET_MAGIC = b"CMS1"
PACKET_VERSION = 1
PACKET_HEADER = struct.Struct("!4sBBHQQHHIH")
MAX_UDP_DATAGRAM = 65000

SEMANTIC_TAG_COLORS = np.array(
    [
        (0, 0, 0),
        (70, 70, 70),
        (190, 153, 153),
        (250, 170, 160),
        (220, 20, 60),
        (153, 153, 153),
        (157, 234, 50),
        (128, 64, 128),
        (244, 35, 232),
        (107, 142, 35),
        (0, 0, 142),
        (102, 102, 156),
        (220, 220, 0),
        (70, 130, 180),
        (0, 0, 230),
        (119, 11, 32),
    ],
    dtype=np.uint8,
)


@dataclass
class ReceivedFrame:
    sensor_name: str
    frame_id: int
    timestamp_us: int
    payload_size: int
    metadata: Dict[str, Any]
    payload: bytes


def display_available() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


_GST_MODULE = None
_GST_LOCK = threading.Lock()


def require_gst() -> Any:
    global _GST_MODULE
    with _GST_LOCK:
        if _GST_MODULE is not None:
            return _GST_MODULE
        try:
            import gi  # type: ignore

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "GStreamer Python bindings are required. Install PyGObject and "
                "plugins providing rtph265depay, h265parse, avdec_h265, and autovideosink."
            ) from exc
        Gst.init(None)
        _GST_MODULE = Gst
        return Gst


def sensor_ports_from_args(args: argparse.Namespace) -> Dict[str, int]:
    base = int(args.base_port)
    return {
        "rgb": int(args.rgb_port or base + DEFAULT_PORT_OFFSETS["rgb"]),
        "semantic": int(args.semantic_port or base + DEFAULT_PORT_OFFSETS["semantic"]),
        "instance": int(args.instance_port or base + DEFAULT_PORT_OFFSETS["instance"]),
        "radar": int(args.radar_port or base + DEFAULT_PORT_OFFSETS["radar"]),
        "semantic_lidar": int(args.semantic_lidar_port or base + DEFAULT_PORT_OFFSETS["semantic_lidar"]),
        "lidar": int(args.lidar_port or base + DEFAULT_PORT_OFFSETS["lidar"]),
    }


def put_latest(target_queue: "queue.Queue[Any]", item: Any) -> None:
    try:
        target_queue.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        target_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        target_queue.put_nowait(item)
    except queue.Full:
        pass


class GstBgrVideoRenderer:
    """Render BGR numpy frames through GStreamer instead of OpenCV HighGUI."""

    def __init__(self, name: str, width: int, height: int, fps: int = 20) -> None:
        self.Gst = require_gst()
        self.name = str(name)
        self.width = int(width)
        self.height = int(height)
        fps = max(1, int(fps))
        pipeline_desc = (
            "appsrc name=src is-live=true block=false format=time do-timestamp=true "
            f"caps=video/x-raw,format=BGR,width={self.width},height={self.height},framerate={fps}/1 "
            "! queue max-size-buffers=2 max-size-time=100000000 max-size-bytes=0 leaky=downstream "
            "! videoconvert "
            "! autovideosink sync=false"
        )
        self.pipeline = self.Gst.parse_launch(pipeline_desc)
        self.appsrc = self.pipeline.get_by_name("src")
        result = self.pipeline.set_state(self.Gst.State.PLAYING)
        if result == self.Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"Failed to start GStreamer renderer for {self.name}.")

    def push_frame(self, bgr: np.ndarray) -> bool:
        if bgr.ndim != 3 or bgr.shape[:2] != (self.height, self.width):
            return False
        frame = np.ascontiguousarray(bgr[:, :, :3], dtype=np.uint8)
        buf = self.Gst.Buffer.new_allocate(None, int(frame.nbytes), None)
        buf.fill(0, frame.tobytes())
        return self.appsrc.emit("push-buffer", buf) == self.Gst.FlowReturn.OK

    def poll_bus(self) -> None:
        bus = self.pipeline.get_bus()
        while True:
            message = bus.pop_filtered(
                self.Gst.MessageType.ERROR | self.Gst.MessageType.WARNING | self.Gst.MessageType.EOS
            )
            if message is None:
                break
            if message.type == self.Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                print(f"[GStreamer:{self.name}] ERROR: {err}; {debug}", file=sys.stderr)
            elif message.type == self.Gst.MessageType.WARNING:
                warn, debug = message.parse_warning()
                print(f"[GStreamer:{self.name}] WARNING: {warn}; {debug}", file=sys.stderr)

    def close(self) -> None:
        try:
            self.appsrc.emit("end-of-stream")
        except Exception:
            pass
        self.pipeline.set_state(self.Gst.State.NULL)


class GstRtpH265CameraReceiver:
    """H.265/RTP camera UDP receiver and display pipeline."""

    def __init__(
        self,
        sensor_name: str,
        port: int,
        plotter: "PayloadPlotter",
        *,
        display: bool,
        jitter_latency_ms: int,
        decoder_max_threads: int,
    ) -> None:
        self.Gst = require_gst()
        self.sensor_name = sensor_name
        self.port = int(port)
        self.plotter = plotter
        decoder_threads = max(1, int(decoder_max_threads))
        if display:
            tail = (
                "! rtph265depay "
                "! h265parse "
                f"! avdec_h265 max-threads={decoder_threads} "
                "! videoconvert "
                "! autovideosink sync=false"
            )
        else:
            tail = "! fakesink sync=false"
        pipeline_desc = (
            f'udpsrc name=src port={self.port} buffer-size=4194304 '
            'caps="application/x-rtp,media=video,encoding-name=H265,payload=96,clock-rate=90000" '
            f"! rtpjitterbuffer latency={int(jitter_latency_ms)} drop-on-latency=true do-lost=true "
            "! queue "
            f"{tail}"
        )
        self.pipeline = self.Gst.parse_launch(pipeline_desc)
        udpsrc = self.pipeline.get_by_name("src")
        if udpsrc is not None:
            srcpad = udpsrc.get_static_pad("src")
            if srcpad is not None:
                srcpad.add_probe(self.Gst.PadProbeType.BUFFER, self._payload_probe)
        result = self.pipeline.set_state(self.Gst.State.PLAYING)
        if result == self.Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"Failed to start GStreamer RTP receiver for {sensor_name}.")

    def _payload_probe(self, _pad: Any, info: Any) -> Any:
        buf = info.get_buffer()
        if buf is not None:
            self.plotter.add(self.sensor_name, int(buf.get_size()))
        return self.Gst.PadProbeReturn.OK

    def poll_bus(self) -> None:
        bus = self.pipeline.get_bus()
        while True:
            message = bus.pop_filtered(
                self.Gst.MessageType.ERROR | self.Gst.MessageType.WARNING | self.Gst.MessageType.EOS
            )
            if message is None:
                break
            if message.type == self.Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                print(f"[GStreamer:{self.sensor_name}] ERROR: {err}; {debug}", file=sys.stderr)
            elif message.type == self.Gst.MessageType.WARNING:
                warn, debug = message.parse_warning()
                print(f"[GStreamer:{self.sensor_name}] WARNING: {warn}; {debug}", file=sys.stderr)

    def close(self) -> None:
        self.pipeline.set_state(self.Gst.State.NULL)


class OpencvRtpH265CameraReceiver:
    """Drop-in replacement for GstRtpH265CameraReceiver. Decodes via GStreamer
    (avdec_h265) but DISPLAYS via OpenCV (cv2.imshow). No rtpjitterbuffer and
    no autovideosink in the pipeline.

    Pipeline (display=True):
        udpsrc -> caps -> rtph265depay -> h265parse -> avdec_h265
                -> videoconvert -> video/x-raw,format=BGR
                -> appsink (max-buffers=1 drop=true emit-signals=true)

    cv2.imshow / cv2.waitKey live on the main thread (see main loop).
    The GStreamer callback (_on_new_sample) only stores the latest frame
    under a lock, so cross-thread display races are avoided.
    """

    def __init__(
        self,
        sensor_name: str,
        port: int,
        plotter: "PayloadPlotter",
        *,
        display: bool,
        jitter_latency_ms: int,
        decoder_max_threads: int,
    ) -> None:
        self.Gst = require_gst()
        self.sensor_name = sensor_name
        self.port = int(port)
        self.plotter = plotter
        self.display = bool(display)
        # jitter_latency_ms accepted for API parity with GstRtpH265CameraReceiver
        # but intentionally unused -- the whole point of this class is to drop
        # the rtpjitterbuffer that needed it.
        _ = jitter_latency_ms
        decoder_threads = max(1, int(decoder_max_threads))

        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None

        if self.display:
            tail = (
                "! rtph265depay "
                "! h265parse "
                f"! avdec_h265 max-threads={decoder_threads} "
                "! videoconvert "
                "! video/x-raw,format=BGR "
                "! appsink name=appsink emit-signals=true sync=false "
                "max-buffers=1 drop=true"
            )
        else:
            tail = "! fakesink sync=false"

        pipeline_desc = (
            f"udpsrc name=src port={self.port} buffer-size=4194304 "
            'caps="application/x-rtp,media=video,encoding-name=H265,'
            'payload=96,clock-rate=90000" '
            f"{tail}"
        )
        self.pipeline = self.Gst.parse_launch(pipeline_desc)

        # Payload-size probe so the existing live plotter still sees per-frame
        # byte counts -- same behavior as GstRtpH265CameraReceiver.
        udpsrc = self.pipeline.get_by_name("src")
        if udpsrc is not None:
            srcpad = udpsrc.get_static_pad("src")
            if srcpad is not None:
                srcpad.add_probe(self.Gst.PadProbeType.BUFFER, self._payload_probe)

        if self.display:
            appsink = self.pipeline.get_by_name("appsink")
            if appsink is not None:
                appsink.connect("new-sample", self._on_new_sample)

        result = self.pipeline.set_state(self.Gst.State.PLAYING)
        if result == self.Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(
                f"Failed to start OpenCV-display receiver for {sensor_name}."
            )

    def _payload_probe(self, _pad: Any, info: Any) -> Any:
        buf = info.get_buffer()
        if buf is not None:
            self.plotter.add(self.sensor_name, int(buf.get_size()))
        return self.Gst.PadProbeReturn.OK

    def _on_new_sample(self, sink: Any) -> Any:
        """Runs on a GStreamer worker thread. Snapshot the decoded BGR frame
        into a numpy array under a lock; the main thread will display it."""
        sample = sink.emit("pull-sample")
        if sample is None:
            return self.Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        caps_struct = sample.get_caps().get_structure(0)
        width = caps_struct.get_value("width")
        height = caps_struct.get_value("height")
        ok, map_info = buf.map(self.Gst.MapFlags.READ)
        if not ok:
            return self.Gst.FlowReturn.ERROR
        try:
            frame = (
                np.frombuffer(map_info.data, dtype=np.uint8)
                  .reshape(height, width, 3)
                  .copy()
            )
        finally:
            buf.unmap(map_info)
        with self._frame_lock:
            self._latest_frame = frame
        return self.Gst.FlowReturn.OK

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Called from the main thread. Returns the most recent decoded frame,
        or None if no new frame has arrived since the last call."""
        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None
        return frame

    def poll_bus(self) -> None:
        bus = self.pipeline.get_bus()
        while True:
            message = bus.pop_filtered(
                self.Gst.MessageType.ERROR | self.Gst.MessageType.WARNING | self.Gst.MessageType.EOS
            )
            if message is None:
                break
            if message.type == self.Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                print(f"[GStreamer:{self.sensor_name}] ERROR: {err}; {debug}", file=sys.stderr)
            elif message.type == self.Gst.MessageType.WARNING:
                warn, debug = message.parse_warning()
                print(f"[GStreamer:{self.sensor_name}] WARNING: {warn}; {debug}", file=sys.stderr)

    def close(self) -> None:
        self.pipeline.set_state(self.Gst.State.NULL)


class PacketReassembler:
    def __init__(self, stale_seconds: float = 2.0) -> None:
        self.frames: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.stale_seconds = float(stale_seconds)
        self.last_cleanup = time.monotonic()

    def accept(self, datagram: bytes) -> Optional[ReceivedFrame]:
        if len(datagram) < PACKET_HEADER.size:
            return None
        try:
            (
                magic,
                version,
                sensor_id,
                _flags,
                frame_id,
                timestamp_us,
                chunk_index,
                chunk_count,
                payload_size,
                meta_len,
            ) = PACKET_HEADER.unpack_from(datagram)
        except struct.error:
            return None
        if magic != PACKET_MAGIC or version != PACKET_VERSION or sensor_id not in SENSOR_NAMES_BY_ID:
            return None
        if chunk_count <= 0 or chunk_index >= chunk_count:
            return None

        meta_start = PACKET_HEADER.size
        meta_end = meta_start + meta_len
        if len(datagram) < meta_end:
            return None
        try:
            metadata = json.loads(datagram[meta_start:meta_end].decode("utf-8"))
        except json.JSONDecodeError:
            return None

        now = time.monotonic()
        if now - self.last_cleanup > self.stale_seconds:
            self._cleanup(now)
        key = (int(sensor_id), int(frame_id))
        entry = self.frames.get(key)
        if entry is None:
            entry = {
                "created_at": now,
                "chunks": [None] * int(chunk_count),
                "metadata": metadata,
                "timestamp_us": int(timestamp_us),
                "payload_size": int(payload_size),
                "received": 0,
            }
            self.frames[key] = entry

        chunks = entry["chunks"]
        if chunks[chunk_index] is None:
            chunks[chunk_index] = datagram[meta_end:]
            entry["received"] += 1
        if int(entry["received"]) != int(chunk_count):
            return None

        del self.frames[key]
        payload = b"".join(part for part in chunks if part is not None)
        if len(payload) != int(payload_size):
            return None
        return ReceivedFrame(
            sensor_name=SENSOR_NAMES_BY_ID[int(sensor_id)],
            frame_id=int(frame_id),
            timestamp_us=int(timestamp_us),
            payload_size=int(payload_size),
            metadata=dict(entry["metadata"]),
            payload=payload,
        )

    def _cleanup(self, now: float) -> None:
        stale = [key for key, entry in self.frames.items() if now - float(entry.get("created_at", now)) > self.stale_seconds]
        for key in stale:
            self.frames.pop(key, None)
        self.last_cleanup = now


class UdpSensorReceiver(threading.Thread):
    def __init__(
        self,
        sensor_name: str,
        bind_ip: str,
        port: int,
        output_queue: "queue.Queue[ReceivedFrame]",
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.sensor_name = sensor_name
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.reassembler = PacketReassembler()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        self.sock.bind((str(bind_ip), int(port)))
        self.sock.settimeout(0.2)

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                try:
                    datagram, _addr = self.sock.recvfrom(MAX_UDP_DATAGRAM)
                except socket.timeout:
                    continue
                except OSError:
                    break
                frame = self.reassembler.accept(datagram)
                if frame is not None:
                    put_latest(self.output_queue, frame)
        finally:
            self.sock.close()


def draw_line(canvas: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], color_bgr: Tuple[int, int, int]) -> None:
    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    height, width = canvas.shape[:2]
    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            canvas[y0, x0] = color_bgr
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_filled_square(canvas: np.ndarray, center: Tuple[int, int], half_size: int, color_bgr: Tuple[int, int, int]) -> None:
    x, y = int(center[0]), int(center[1])
    height, width = canvas.shape[:2]
    canvas[max(0, y - half_size) : min(height, y + half_size + 1), max(0, x - half_size) : min(width, x + half_size + 1)] = color_bgr


BITMAP_FONT_5X7 = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "/": ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "11100"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01110"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
}


def draw_text(canvas: np.ndarray, text: str, origin: Tuple[int, int], color_bgr: Tuple[int, int, int], *, scale: int = 1) -> None:
    x0, y0 = int(origin[0]), int(origin[1])
    scale = max(1, int(scale))
    height, width = canvas.shape[:2]
    cursor_x = x0
    for char in text.upper():
        glyph = BITMAP_FONT_5X7.get(char, BITMAP_FONT_5X7["?"])
        for row_index, row in enumerate(glyph):
            for col_index, bit in enumerate(row):
                if bit != "1":
                    continue
                y_start = y0 + row_index * scale
                x_start = cursor_x + col_index * scale
                if x_start >= width or y_start >= height:
                    continue
                canvas[
                    max(0, y_start) : min(height, y_start + scale),
                    max(0, x_start) : min(width, x_start + scale),
                ] = color_bgr
        cursor_x += (len(glyph[0]) + 1) * scale


PLOT_COLORS_BGR = {
    "rgb": (80, 200, 255),
    "semantic": (120, 230, 120),
    "instance": (230, 160, 255),
    "radar": (70, 180, 255),
    "semantic_lidar": (255, 220, 80),
    "lidar": (210, 210, 210),
}

PLOT_LABELS = {
    "rgb": "RGB",
    "semantic": "SEM CAM",
    "instance": "INST CAM",
    "radar": "RADAR",
    "semantic_lidar": "SEM LIDAR",
    "lidar": "LIDAR",
}


def render_payload_plot(
    history: Mapping[str, List[Tuple[float, float]]],
    min_t: float,
    max_t: float,
    max_y: float,
    width: int,
    height: int,
    sensors: Sequence[str],
    title: str,
    y_label: str = "KB",
) -> np.ndarray:
    canvas = np.full((height, width, 3), (18, 18, 18), dtype=np.uint8)
    left, right, top, bottom = 64, width - 236, 46, height - 42
    draw_line(canvas, (left, top), (left, bottom), (90, 90, 90))
    draw_line(canvas, (left, bottom), (right, bottom), (90, 90, 90))
    draw_text(canvas, title, (left, 14), (225, 225, 225), scale=2)
    draw_text(canvas, y_label, (18, top - 2), (165, 165, 165), scale=1)
    draw_text(canvas, f"LAST {int(max_t - min_t)}S", (left, bottom + 14), (165, 165, 165), scale=1)
    for i in range(1, 5):
        y = bottom - int((bottom - top) * i / 4)
        draw_line(canvas, (left, y), (right, y), (48, 48, 48))
        draw_text(canvas, f"{max_y * i / 4:.0f}", (12, y - 4), (130, 130, 130), scale=1)
    for i in range(1, 6):
        x = left + int((right - left) * i / 6)
        draw_line(canvas, (x, top), (x, bottom), (48, 48, 48))

    span_t = max(1e-6, max_t - min_t)
    span_y = max(1e-6, max_y)
    for sensor_name in sensors:
        values = history.get(sensor_name, [])
        points = []
        for t, y_value in values:
            if t < min_t or t > max_t:
                continue
            x_px = left + int((t - min_t) / span_t * (right - left))
            y_px = bottom - int(min(y_value, max_y) / span_y * (bottom - top))
            points.append((x_px, y_px))
        color = PLOT_COLORS_BGR.get(sensor_name, (220, 220, 220))
        for p0, p1 in zip(points, points[1:]):
            draw_line(canvas, p0, p1, color)

    legend_left = right + 18
    legend_top = top
    legend_right = width - 16
    legend_bottom = bottom
    draw_line(canvas, (legend_left - 10, legend_top), (legend_right, legend_top), (58, 58, 58))
    draw_line(canvas, (legend_left - 10, legend_bottom), (legend_right, legend_bottom), (58, 58, 58))
    draw_line(canvas, (legend_left - 10, legend_top), (legend_left - 10, legend_bottom), (58, 58, 58))
    draw_line(canvas, (legend_right, legend_top), (legend_right, legend_bottom), (58, 58, 58))
    draw_text(canvas, "LEGEND", (legend_left, legend_top + 8), (210, 210, 210), scale=1)
    for index, sensor_name in enumerate(sensors):
        y = legend_top + 34 + index * 45
        color = PLOT_COLORS_BGR.get(sensor_name, (220, 220, 220))
        canvas[y : y + 10, legend_left : legend_left + 22] = color
        draw_text(canvas, PLOT_LABELS.get(sensor_name, sensor_name), (legend_left + 32, y - 1), color, scale=1)
        values = history.get(sensor_name, [])
        if values:
            draw_text(canvas, f"{values[-1][1]:.1f} {y_label}", (legend_left + 32, y + 17), (160, 160, 160), scale=1)
        else:
            draw_text(canvas, "NO DATA", (legend_left + 32, y + 17), (95, 95, 95), scale=1)
    return canvas


class PayloadPlotter:
    def __init__(self, title: str, sensors: Sequence[str], enabled: bool, window_seconds: float, average_window: float = 0.0) -> None:
        self.title = str(title)
        self.sensors = list(sensors)
        self.enabled = bool(enabled)
        self.window_seconds = float(window_seconds)
        self.average_window = float(average_window)
        self.history: Dict[str, List[Any]] = {name: [] for name in self.sensors}
        self.lock = threading.Lock()
        self.last_draw = 0.0
        self.start_time = time.monotonic()
        self.renderer: Optional[GstBgrVideoRenderer] = None
        if self.enabled:
            try:
                self.renderer = GstBgrVideoRenderer(self.title, 900, 420, fps=10)
            except Exception as exc:
                print(f"[WARN] Payload plot disabled: {exc}", file=sys.stderr)
                self.enabled = False

    def add(self, sensor_name: str, payload_size: int) -> None:
        if not self.enabled or sensor_name not in self.sensors:
            return
        t = time.monotonic() - self.start_time
        with self.lock:
            values = self.history.setdefault(sensor_name, [])
            
            if self.average_window > 0.0:
                bucket_t = math.floor(t / self.average_window) * self.average_window
                if not values:
                    values.append([bucket_t, payload_size])
                else:
                    last_bucket_t, _ = values[-1]
                    if bucket_t == last_bucket_t:
                        values[-1][1] += payload_size
                    elif bucket_t > last_bucket_t:
                        # Inject zeros for missing buckets to ensure gaps drop to 0 KB/s
                        curr = last_bucket_t + self.average_window
                        max_inserts = int(self.window_seconds / self.average_window) + 1
                        inserts = 0
                        while curr < bucket_t and inserts < max_inserts:
                            values.append([curr, 0])
                            curr += self.average_window
                            inserts += 1
                        values.append([bucket_t, payload_size])
                    else:
                        # If a packet arrived exceptionally late to a prior bucket, just append to latest sum
                        values[-1][1] += payload_size
            else:
                values.append((t, float(payload_size) / 1024.0))

            cutoff = t - self.window_seconds
            while values and values[0][0] < cutoff:
                values.pop(0)

    def maybe_draw(self) -> None:
        if not self.enabled or self.renderer is None:
            return
        now = time.monotonic()
        if now - self.last_draw < 0.25:
            return
        max_t = max(now - self.start_time, 1.0)
        min_t = max(0.0, max_t - self.window_seconds)
        
        with self.lock:
            snapshot = {}
            for name, values in self.history.items():
                if self.average_window > 0.0:
                    # Convert accumulated bytes in the bucket to KB/s
                    snapshot[name] = [(t, (b / 1024.0) / self.average_window) for t, b in values]
                else:
                    snapshot[name] = list(values)
                    
        max_y = 1.0
        for values in snapshot.values():
            if values:
                max_y = max(max_y, max(value[1] for value in values) * 1.15)
                
        y_label = "KB/S" if self.average_window > 0.0 else "KB"
        frame = render_payload_plot(snapshot, min_t, max_t, max_y, 900, 420, self.sensors, self.title.upper(), y_label)
        self.renderer.push_frame(frame)
        self.renderer.poll_bus()
        self.last_draw = now

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()


def topdown_points_image(
    points_xyz: np.ndarray,
    colors_bgr: np.ndarray,
    *,
    label: str,
    range_m: float,
    canvas_size: int,
    max_display_points: int,
) -> np.ndarray:
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    margin = 36
    bottom = canvas_size - margin
    center_x = canvas_size // 2
    scale = (canvas_size - 2 * margin) / max(1.0, float(range_m))
    canvas[0:28, :] = (42, 42, 42)
    draw_line(canvas, (center_x, margin), (center_x, bottom), (70, 70, 70))
    draw_line(canvas, (margin, bottom), (canvas_size - margin, bottom), (70, 70, 70))
    draw_filled_square(canvas, (center_x, bottom), 4, (255, 255, 255))
    if len(points_xyz):
        if max_display_points > 0 and len(points_xyz) > max_display_points:
            indices = np.linspace(0, len(points_xyz) - 1, max_display_points, dtype=np.int64)
            points_xyz = points_xyz[indices]
            colors_bgr = colors_bgr[indices]
        x = points_xyz[:, 0]
        y = points_xyz[:, 1]
        u = np.rint(center_x + y * scale).astype(np.int32)
        v = np.rint(bottom - x * scale).astype(np.int32)
        valid = (u >= 0) & (u < canvas_size) & (v >= 0) & (v < canvas_size) & (x >= 0)
        canvas[v[valid], u[valid]] = colors_bgr[valid]
    label_key = label.lower().replace(" ", "_")
    canvas[8:20, 12:44] = PLOT_COLORS_BGR.get(label_key, (220, 220, 220))
    draw_text(canvas, label, (54, 10), (220, 220, 220), scale=1)
    draw_text(canvas, "FRONT", (center_x - 18, margin - 18), (130, 130, 130), scale=1)
    draw_text(canvas, "EGO", (center_x + 9, bottom - 4), (220, 220, 220), scale=1)
    return canvas


def radar_payload_to_image(frame: ReceivedFrame, *, canvas_size: int, max_display_points: int) -> np.ndarray:
    points = np.frombuffer(frame.payload, dtype=np.float32)
    points = points[: (points.size // 4) * 4].reshape((-1, 4))
    if len(points):
        altitude = points[:, 0]
        azimuth = points[:, 1]
        depth = points[:, 2]
        velocity = points[:, 3]
        x = depth * np.cos(altitude) * np.cos(azimuth)
        y = depth * np.cos(altitude) * np.sin(azimuth)
        z = depth * np.sin(altitude)
        xyz = np.stack([x, y, z], axis=1)
        norm = np.clip((velocity + 20.0) / 40.0, 0.0, 1.0)
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[:, 0] = np.rint((1.0 - norm) * 255).astype(np.uint8)
        colors[:, 2] = np.rint(norm * 255).astype(np.uint8)
        colors[:, 1] = 180
    else:
        xyz = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.uint8)
    return topdown_points_image(
        xyz,
        colors,
        label="Radar",
        range_m=float(frame.metadata.get("range_m", 100.0)),
        canvas_size=canvas_size,
        max_display_points=max_display_points,
    )


def lidar_payload_to_image(frame: ReceivedFrame, *, semantic: bool, canvas_size: int, max_display_points: int) -> np.ndarray:
    shape = frame.metadata.get("shape") or [0, 4]
    cols = int(shape[1]) if len(shape) >= 2 else 4
    points = np.frombuffer(frame.payload, dtype=np.float32)
    points = points[: (points.size // max(1, cols)) * max(1, cols)].reshape((-1, max(1, cols)))
    if len(points):
        xyz = points[:, :3]
        if semantic and cols >= 6:
            tags = np.clip(points[:, 5].astype(np.int32), 0, len(SEMANTIC_TAG_COLORS) - 1)
            colors = SEMANTIC_TAG_COLORS[tags][:, ::-1].copy()
        elif cols >= 4:
            intensity = np.clip(points[:, 3], 0.0, 1.0)
            gray = np.rint(intensity * 255).astype(np.uint8)
            colors = np.stack([gray, gray, gray], axis=1)
        else:
            colors = np.full((len(points), 3), (190, 190, 190), dtype=np.uint8)
    else:
        xyz = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.uint8)
    return topdown_points_image(
        xyz,
        colors,
        label="Semantic LiDAR" if semantic else "LiDAR",
        range_m=float(frame.metadata.get("range_m", 100.0)),
        canvas_size=canvas_size,
        max_display_points=max_display_points,
    )


def run_receiver(args: argparse.Namespace) -> None:
    ports = sensor_ports_from_args(args)
    show = {name: not bool(getattr(args, f"no_show_{name}", False)) for name in SENSOR_ORDER}
    if any(show.values()) and not display_available() and not bool(args.force_display):
        print("[WARN] No DISPLAY/WAYLAND_DISPLAY detected; visualization windows disabled. Use --force-display to try anyway.", file=sys.stderr)
        show = {name: False for name in SENSOR_ORDER}
    plot_enabled = not bool(args.no_payload_plot)
    if plot_enabled and not display_available() and not bool(args.force_display):
        print("[WARN] Payload plot disabled because no display is available.", file=sys.stderr)
        plot_enabled = False

    point_sensors = [s for s in SENSOR_ORDER if s in POINT_SENSORS]

    camera_plotters = {
        name: PayloadPlotter(f"{PLOT_LABELS.get(name, name)} Payload Size", [name], plot_enabled, float(args.plot_window_seconds), average_window=1.0)
        for name in CAMERA_SENSORS
    }
    
    point_plotter = PayloadPlotter("Point Sensor Payload Size", point_sensors, plot_enabled, float(args.plot_window_seconds))

    output_queue: "queue.Queue[ReceivedFrame]" = queue.Queue(maxsize=int(args.receive_queue_size))
    stop_event = threading.Event()
    point_receivers = [
        UdpSensorReceiver(name, str(args.bind_ip), ports[name], output_queue, stop_event)
        for name in POINT_SENSORS
    ]
    for receiver in point_receivers:
        receiver.start()

    camera_receivers = [
        OpencvRtpH265CameraReceiver(
            name,
            ports[name],
            camera_plotters[name],
            display=bool(show.get(name, False)),
            jitter_latency_ms=int(args.rtp_jitter_latency_ms),
            decoder_max_threads=int(args.decoder_max_threads),
        )
        for name in CAMERA_SENSORS
        if show.get(name, False) or plot_enabled
    ]

    point_renderers: Dict[str, GstBgrVideoRenderer] = {}
    for name in POINT_SENSORS:
        if not show.get(name, False):
            continue
        try:
            point_renderers[name] = GstBgrVideoRenderer(
                SENSOR_LABELS[name],
                int(args.point_canvas_size),
                int(args.point_canvas_size),
                fps=20,
            )
        except Exception as exc:
            print(f"[WARN] {SENSOR_LABELS[name]} visualization disabled: {exc}", file=sys.stderr)
            show[name] = False

    print("Receiver listening on UDP ports:")
    for name in SENSOR_ORDER:
        status = "shown" if show.get(name, False) else "hidden"
        transport = "gst-rtp" if name in CAMERA_SENSORS else "python-udp"
        print(f"  {name:15s} <- {args.bind_ip}:{ports[name]} ({status}, {transport})")
    print("Press Ctrl+C in the terminal to stop.")

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        while not stop_event.is_set():
            for camera_receiver in camera_receivers:
                camera_receiver.poll_bus()
                # Pull the latest decoded frame (if any) and display it via
                # OpenCV. The GStreamer callback runs on a worker thread; the
                # imshow call here runs on the main thread, paired with the
                # single cv2.waitKey() below.
                latest = camera_receiver.get_latest_frame()
                if latest is not None:
                    cv2.imshow(
                        SENSOR_LABELS.get(
                            camera_receiver.sensor_name,
                            camera_receiver.sensor_name,
                        ),
                        latest,
                    )
            # Pump OpenCV's event loop (paints windows, captures keypresses).
            # 'q' or ESC = quit. waitKey returns -1 if no key pressed.
            if camera_receivers:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    stop_event.set()
                    continue
            for renderer in point_renderers.values():
                renderer.poll_bus()
            try:
                frame = output_queue.get(timeout=0.05)
            except queue.Empty:
                for plotter in camera_plotters.values():
                    plotter.maybe_draw()
                point_plotter.maybe_draw()
                continue
            
            point_plotter.add(frame.sensor_name, frame.payload_size)
            
            if show.get(frame.sensor_name, False):
                if frame.sensor_name == "radar":
                    image = radar_payload_to_image(
                        frame,
                        canvas_size=int(args.point_canvas_size),
                        max_display_points=int(args.max_display_points),
                    )
                elif frame.sensor_name == "semantic_lidar":
                    image = lidar_payload_to_image(
                        frame,
                        semantic=True,
                        canvas_size=int(args.point_canvas_size),
                        max_display_points=int(args.max_display_points),
                    )
                elif frame.sensor_name == "lidar":
                    image = lidar_payload_to_image(
                        frame,
                        semantic=False,
                        canvas_size=int(args.point_canvas_size),
                        max_display_points=int(args.max_display_points),
                    )
                else:
                    image = None
                renderer = point_renderers.get(frame.sensor_name)
                if image is not None and renderer is not None:
                    renderer.push_frame(image)
            
            for plotter in camera_plotters.values():
                plotter.maybe_draw()
            point_plotter.maybe_draw()
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        stop_event.set()
        for receiver in point_receivers:
            receiver.join(timeout=1.0)
        for camera_receiver in camera_receivers:
            camera_receiver.close()
        for renderer in point_renderers.values():
            renderer.close()
        for plotter in camera_plotters.values():
            plotter.close()
        point_plotter.close()
        print("Receiver shutdown complete.")


def add_common_port_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-port", type=int, default=47000, help="Base UDP port.")
    parser.add_argument("--rgb-port", type=int, default=0, help="Optional RGB UDP port override.")
    parser.add_argument("--semantic-port", type=int, default=0, help="Optional semantic-segmentation UDP port override.")
    parser.add_argument("--instance-port", type=int, default=0, help="Optional instance-segmentation UDP port override.")
    parser.add_argument("--radar-port", type=int, default=0, help="Optional radar UDP port override.")
    parser.add_argument("--semantic-lidar-port", type=int, default=0, help="Optional semantic-LiDAR UDP port override.")
    parser.add_argument("--lidar-port", type=int, default=0, help="Optional LiDAR UDP port override.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone UDP/GStreamer receiver for the CARLA multi-sensor demo.")
    add_common_port_args(parser)
    parser.add_argument("--bind-ip", default="0.0.0.0")
    parser.add_argument("--rtp-jitter-latency-ms", type=int, default=200)
    parser.add_argument("--decoder-max-threads", type=int, default=1)
    parser.add_argument("--receive-queue-size", type=int, default=64)
    parser.add_argument("--point-canvas-size", type=int, default=640)
    parser.add_argument("--max-display-points", type=int, default=50000)
    parser.add_argument("--no-payload-plot", action="store_true")
    parser.add_argument("--plot-window-seconds", type=float, default=60.0)
    parser.add_argument("--force-display", action="store_true")
    for name in SENSOR_ORDER:
        parser.add_argument(
            f"--no-show-{name.replace('_', '-')}",
            dest=f"no_show_{name}",
            action="store_true",
            help=f"Receive but do not visualize {name}.",
        )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    run_receiver(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())