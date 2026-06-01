#!/usr/bin/env python
"""
Phase-N receiver: H.265/RTP over UDP, decoded by GStreamer, displayed by OpenCV.

Architectural change vs scan_receiver_v1.py / v2.py:
  The pipeline ends in `appsink` instead of `autovideosink`. Decoded frames
  are pulled into Python as numpy arrays and shown via cv2.imshow, which
  lets us:
    1. Always render the freshest decoded frame (max-buffers=1 drop=true)
       -- no silent GStreamer-internal backlog when we can't keep up.
    2. Use a plain OpenCV window (simpler X path than GL-backed sinks ->
       AnyDesk capture is more predictable).
    3. Be in Python at display time, so we can timestamp, log, drop, or
       overlay metrics ourselves.

Decode still runs on the CPU (avdec_h265, libav software). If the system
is CPU-bound, this script alone will not fix that -- but it isolates the
decode cost from the display cost so we can measure them separately.

Pair with the existing PythonAPI/abiodun/scan_sender_v1.py (or later sender).

Run:
    python3 scan_receiver_v3.py            # listens on UDP :65000
    python3 scan_receiver_v3.py 65001      # custom port
    Press 'q' in the window, or Ctrl+C in the terminal, to quit.
"""

import sys
import time

import cv2
import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst


# --- Configuration ----------------------------------------------------------
UDP_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 65000
WINDOW_NAME = "scan_receiver_v3"

# --- GStreamer pipeline -----------------------------------------------------
# udpsrc           : receive raw UDP datagrams on the given port
# caps             : tell the pipeline what's inside those datagrams (RTP H.265,
#                    payload type 96, clock 90 kHz). Same as v2.
# rtph265depay     : reassemble RTP packets back into an H.265 elementary stream
# h265parse        : normalize the H.265 bytes for the decoder (frame boundaries,
#                    NAL units, etc.)
# avdec_h265       : SOFTWARE H.265 decoder (libav). This is the CPU-heavy step.
# videoconvert     : convert decoded pixel format to a format we want next
# video/x-raw,format=BGR : force BGR -- OpenCV's native byte order
# appsink ...      : the appsink hands every decoded frame to Python:
#                      emit-signals=true   -> fire on_new_sample callback
#                      sync=false          -> don't wait for presentation timestamps
#                      max-buffers=1       -> hold at most one frame internally
#                      drop=true           -> if Python is too slow, drop oldest
GST_PIPELINE = (
    f"udpsrc port={UDP_PORT} "
    f"! application/x-rtp,encoding-name=H265,media=video,clock-rate=90000,payload=96 "
    f"! rtph265depay "
    f"! h265parse "
    f"! avdec_h265 "
    f"! videoconvert "
    f"! video/x-raw,format=BGR "
    f"! appsink name=appsink emit-signals=true sync=false max-buffers=1 drop=true"
)


# --- Frame-rate tracker -----------------------------------------------------
# Simple sliding-window FPS, useful for spotting drops/lag visually in the
# overlay. Keep it cheap -- this runs once per frame.
class FpsTracker:
    def __init__(self, window_seconds: float = 1.0):
        self._win = window_seconds
        self._stamps: list[float] = []

    def tick(self) -> float:
        now = time.perf_counter()
        self._stamps.append(now)
        cutoff = now - self._win
        # drop stamps older than the window
        while self._stamps and self._stamps[0] < cutoff:
            self._stamps.pop(0)
        # frames per second = count over window length
        return len(self._stamps) / self._win


# --- Main -------------------------------------------------------------------
def main() -> int:
    Gst.init(None)
    print(f"[receiver] Pipeline: {GST_PIPELINE}")

    pipeline = Gst.parse_launch(GST_PIPELINE)
    appsink = pipeline.get_by_name('appsink')
    if appsink is None:
        print("[receiver] ERROR: could not find appsink element.")
        return 1

    fps = FpsTracker(window_seconds=1.0)
    quit_requested = {'value': False}  # mutable flag for the callback to set

    def on_new_sample(sink) -> Gst.FlowReturn:
        """Called by GStreamer every time a fresh decoded frame is ready."""
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
            # Wrap the raw decoded bytes as a [H, W, 3] uint8 numpy array.
            # .copy() is required because map_info.data goes invalid after unmap.
            frame = (
                np.frombuffer(map_info.data, dtype=np.uint8)
                  .reshape(height, width, 3)
                  .copy()
            )
        finally:
            buf.unmap(map_info)

        # Overlay a small HUD: live FPS in the corner. If this number is much
        # less than the sender's framerate (e.g. 20), frames are being dropped
        # or arriving late -- a sign of CPU saturation or network issues.
        live_fps = fps.tick()
        cv2.putText(
            frame, f"{live_fps:5.1f} fps  port={UDP_PORT}  {width}x{height}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, frame)
        # waitKey(1) pumps OpenCV's window event loop AND captures keypresses.
        # 'q' or ESC = quit; the bus loop in main() picks this up on next tick.
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            quit_requested['value'] = True

        return Gst.FlowReturn.OK

    appsink.connect('new-sample', on_new_sample)

    pipeline.set_state(Gst.State.PLAYING)
    print(f"[receiver] Listening on UDP :{UDP_PORT}. Press 'q' or Ctrl+C to stop.")

    bus = pipeline.get_bus()
    try:
        # Poll the bus for errors/EOS every 100 ms; check the quit flag too.
        while not quit_requested['value']:
            msg = bus.timed_pop_filtered(
                100 * Gst.MSECOND,
                Gst.MessageType.ERROR | Gst.MessageType.EOS,
            )
            if msg is None:
                continue
            if msg.type == Gst.MessageType.EOS:
                print("[receiver] EOS received.")
                break
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                print(f"[receiver] ERROR: {err} ({debug})")
                break
    except KeyboardInterrupt:
        print("\n[receiver] Ctrl+C received.")
    finally:
        print("[receiver] Shutting down pipeline...")
        pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()
        print("[receiver] Done.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
