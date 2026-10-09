"""UE-side emulator process for the composed replay (CPU only, no CARLA/radio/GPU).

Handoff mechanism under test (chosen, not assumed harmless): at the recorded action-open
instant, the UE worker thread stamps action-open and enqueues references to the already
owned, immutable same-frame source onto ONE persistent ``gt-freeze`` thread (queue.put
only). That thread does freeze + seal + descriptor send. Nothing GT-related runs before
action-open, so the capture-to-action-open interval is unchanged by construction; any
cost instead appears as contention with the UE worker thread after action-open.

To expose that contention the worker thread runs the corrected perception-input function
(``live_parity_repair_v1.perception.prepare_live_input``; 432-row content, PIL bilinear
resampling, normalization, 16-row padding) on CPU at the recorded input-stage offset of
every frame, and an uncontended probe of the same function on the same frame at a quiet
moment (alternating before/after). The emulated send is the recorded send offset plus any
measured slowdown versus the running median of uncontended probes, so contention
propagates to prediction timing. GPU encode/transmission are not emulated.
"""
import os
if __name__ == "__main__":
    os.environ.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})

import argparse
import ctypes
import json
import queue
import signal
import socket
import statistics
import threading
import time

import numpy as np


def raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def sleep_until(target_ns):
    while True:
        left = target_ns - raw()
        if left <= 0:
            return
        time.sleep(min(left / 1e9, 0.005) if left > 2_000_000 else 0)


def main(fd, parent):
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(1, signal.SIGTERM, 0, 0, 0)
    from . import channel as CH, freeze_codec as FZ, parity_common as P
    from rl_agent.splitfusion_live_parity_repair_v1 import perception
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
    import torch
    torch.set_num_threads(1)
    sock = socket.socket(fileno=fd)
    init, _ = CH.receive(sock, deadline=time.monotonic() + 120)
    cfg = init["config"]
    corpus = P.Corpus(49)
    cases = {c.frame_id: c for c in corpus.cases}
    radar = np.zeros((perception.CONTRACT.radar_channels, perception.CONTRACT.content_height,
                      perception.CONTRACT.content_width), np.float32)
    frames_bgr = {f: np.ascontiguousarray(np.frombuffer(c.kwargs["rgb"]["_rgb_image"].raw_data, np.uint8)
                                          .reshape(720, 1280, 4)[:, :, :3]) for f, c in cases.items()}
    nearest = sorted(frames_bgr)
    send_lock = threading.Lock()

    def send(value, fd_out=None):
        with send_lock:
            CH.send(sock, value, fd=fd_out, deadline=time.monotonic() + 30)

    def input_stage(frame_id):
        bgr = frames_bgr[frame_id] if frame_id in frames_bgr else frames_bgr[min(nearest, key=lambda f: abs(f - frame_id))]
        t0, c0 = raw(), time.thread_time_ns()
        perception.prepare_live_input(bgr, radar, "cpu")
        return (raw() - t0) / 1e6, (time.thread_time_ns() - c0) / 1e6

    handoff = queue.SimpleQueue()
    freeze_records = {}
    freeze_done = {}

    def freeze_worker():
        while True:
            item = handoff.get()
            if item is None:
                return
            k, sequence, case, enqueued = item
            start, c0 = raw(), time.thread_time_ns()
            descriptor, chunks = FZ.freeze(corpus.sourcekwargs(case), case.kwargs["policy_timestamp_s"],
                                           session_id=cfg["session_id"], sequence=sequence)
            frozen = raw()
            out, size = FZ.seal(chunks)
            sealed = raw()
            try:
                send(dict(op="FREEZE", k=k, descriptor=descriptor.decode(), enqueued_raw_ns=enqueued), out)
            finally:
                os.close(out)
            sent = raw()
            freeze_records[k] = dict(enqueued_raw_ns=enqueued, start_raw_ns=start, frozen_raw_ns=frozen,
                                     sealed_raw_ns=sealed, sent_raw_ns=sent, thread_cpu_ms=(time.thread_time_ns() - c0) / 1e6,
                                     sealed_bytes=size, descriptor_bytes=len(descriptor))
            freeze_done[k].set()

    worker = threading.Thread(target=freeze_worker, name="gt-freeze", daemon=False)
    worker.start()
    send(dict(op="READY", pid=os.getpid(), threads=threading.active_count()))
    start_msg, _ = CH.receive(sock, deadline=time.monotonic() + 600)
    t0 = start_msg["t0_raw_ns"]
    records, baselines = [], []
    for k, e in enumerate(cfg["events"]):
        scheduled = t0 + int(e["open_ms"] * 1e6)
        decision = e["kind"] == "POLICY_DECISION"
        row = dict(index=k, frame_id=e["frame_id"], kind=e["kind"], scheduled_open_raw_ns=scheduled)
        if decision:
            k_dec = e["decision_index"]
            freeze_done[k_dec] = threading.Event()
            probe_before = k_dec % 2 == 0
            if probe_before:
                sleep_until(scheduled - 80_000_000)
                row["baseline_ms"], row["baseline_cpu_ms"] = input_stage(e["frame_id"])
                row["baseline_position"] = "before"
        sleep_until(scheduled)
        opened = raw()
        row["action_open_raw_ns"] = opened
        if decision:
            case = cases[e["frame_id"]]
            handoff.put((k_dec, k_dec + 1, case, opened))         # the entire UE-thread handoff cost
            row["handoff_put_ns"] = raw() - opened
            send(dict(op="OPEN", k=k_dec, frame_id=e["frame_id"], action_open_raw_ns=opened))
            sleep_until(opened + int(e["plan_end_ms"] * 1e6))
            send(dict(op="PLAN", k=k_dec, plan=corpus.plan(case).decode(),
                      frame_context=PO.context_document(corpus.context(case)), plan_raw_ns=raw()))
        sleep_until(opened + int(e["input_start_ms"] * 1e6))
        row["input_start_raw_ns"] = raw()
        row["input_ms"], row["input_cpu_ms"] = input_stage(e["frame_id"])
        if decision:
            reference = statistics.median(baselines) if baselines else row.get("baseline_ms", row["input_ms"])
            slowdown = max(0.0, row["input_ms"] - reference)
            row["slowdown_vs_baseline_median_ms"] = slowdown
            sleep_until(opened + int((e["send_ms"] + slowdown) * 1e6))
            row["send_raw_ns"] = raw()
            send(dict(op="SEND", k=k_dec, send_raw_ns=row["send_raw_ns"], ok=True))
            if not probe_before:
                freeze_done[k_dec].wait(timeout=5)
                sleep_until(opened + int((e["send_ms"] + 60) * 1e6))
                row["baseline_ms"], row["baseline_cpu_ms"] = input_stage(e["frame_id"])
                row["baseline_position"] = "after"
            baselines.append(row["baseline_ms"])
            row["freeze_overlapped_input"] = None   # filled after the thread record exists
        records.append(row)
    handoff.put(None)
    worker.join(timeout=30)
    for row in records:
        if row["kind"] == "POLICY_DECISION":
            f = freeze_records[[e for e in cfg["events"] if e["frame_id"] == row["frame_id"]][0]["decision_index"]]
            row["freeze"] = f
            input_end = row["input_start_raw_ns"] + int(row["input_ms"] * 1e6)
            row["freeze_overlapped_input"] = f["start_raw_ns"] < input_end and f["sent_raw_ns"] > row["input_start_raw_ns"]
    with open(cfg["record_path"], "x") as stream:
        json.dump(dict(records=records, pid=os.getpid()), stream)
    send(dict(op="DONE"))
    sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--parent", type=int, required=True)
    a = parser.parse_args()
    main(a.fd, a.parent)
