#!/usr/bin/env python
"""
OAI variant of scan_receiver_v3.py — designed to run INSIDE the
oai-perception-rx container (see /abiodun/receiver_container/).

Differences vs scan_receiver_v3.py:
  - Port + bind interface are taken from env vars UDP_PORT / RX_BIND
    (set by /abiodun/receiver_container/docker-compose.yaml). CLI argv still
    works as a fallback.
  - udpsrc has an explicit `address=` so we only bind to the container's
    public_net interface (192.168.70.140), not loopback.
  - HUD includes the bind IP so we can visually confirm we received over
    the 5G data plane and not by accident on loopback.
"""

import os
import sys
import time

import cv2
import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst


UDP_PORT = int(os.environ.get('UDP_PORT', sys.argv[1] if len(sys.argv) > 1 else 65000))
RX_BIND = os.environ.get('RX_BIND', '0.0.0.0')
WINDOW_NAME = "scan_receiver_v3_oai"

GST_PIPELINE = (
    f"udpsrc address={RX_BIND} port={UDP_PORT} "
    f"! application/x-rtp,encoding-name=H265,media=video,clock-rate=90000,payload=96 "
    f"! rtph265depay "
    f"! h265parse "
    f"! avdec_h265 "
    f"! videoconvert "
    f"! video/x-raw,format=BGR "
    f"! appsink name=appsink emit-signals=true sync=false max-buffers=1 drop=true"
)


class FpsTracker:
    def __init__(self, window_seconds: float = 1.0):
        self._win = window_seconds
        self._stamps: list[float] = []

    def tick(self) -> float:
        now = time.perf_counter()
        self._stamps.append(now)
        cutoff = now - self._win
        while self._stamps and self._stamps[0] < cutoff:
            self._stamps.pop(0)
        return len(self._stamps) / self._win


def main() -> int:
    Gst.init(None)
    print(f"[receiver-oai] bind={RX_BIND}:{UDP_PORT}")
    print(f"[receiver-oai] pipeline: {GST_PIPELINE}")

    pipeline = Gst.parse_launch(GST_PIPELINE)
    appsink = pipeline.get_by_name('appsink')
    if appsink is None:
        print("[receiver-oai] ERROR: could not find appsink element.")
        return 1

    fps = FpsTracker(window_seconds=1.0)
    quit_requested = {'value': False}

    def on_new_sample(sink) -> Gst.FlowReturn:
        sample = sink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        width = caps.get_value('width')
        height = caps.get_value('height')

        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            frame = (
                np.frombuffer(map_info.data, dtype=np.uint8)
                  .reshape(height, width, 3)
                  .copy()
            )
        finally:
            buf.unmap(map_info)

        live_fps = fps.tick()
        cv2.putText(
            frame,
            f"{live_fps:5.1f} fps  bind={RX_BIND}:{UDP_PORT}  {width}x{height}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            quit_requested['value'] = True

        return Gst.FlowReturn.OK

    appsink.connect('new-sample', on_new_sample)

    pipeline.set_state(Gst.State.PLAYING)
    print(f"[receiver-oai] listening on {RX_BIND}:{UDP_PORT}")

    bus = pipeline.get_bus()
    try:
        while not quit_requested['value']:
            msg = bus.timed_pop_filtered(
                100 * Gst.MSECOND,
                Gst.MessageType.ERROR | Gst.MessageType.EOS,
            )
            if msg is None:
                continue
            if msg.type == Gst.MessageType.EOS:
                print("[receiver-oai] EOS received.")
                break
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                print(f"[receiver-oai] ERROR: {err} ({debug})")
                break
    except KeyboardInterrupt:
        print("\n[receiver-oai] Ctrl+C received.")
    finally:
        pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    sys.exit(main())
