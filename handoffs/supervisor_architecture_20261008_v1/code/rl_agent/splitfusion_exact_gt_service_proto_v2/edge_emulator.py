"""Edge-side emulator process for the composed replay (CPU only).

For each decision it waits for the coordinator's forwarded UE send instant, then submits the
archived native prediction at send + recorded (edge.compute end - last send): the earliest
usable CPU prediction boundary. Submission never waits for GT; map build/publication is not
on this path (the candidate hands the prediction to evaluation before the map branch).
Edge-side digest work (labels + canonical records) and the sealed labels reference are
performed here exactly as the prototype edge client would.
"""
import os
if __name__ == "__main__":
    os.environ.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})

import argparse
import ctypes
import json
import select
import signal
import socket
import time


def raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def main(fd, parent, cpu):
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(1, signal.SIGTERM, 0, 0, 0)
    from . import channel as CH, parity_common as P
    from .bench import edge_prediction
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
    sock = socket.socket(fileno=fd)
    init, _ = CH.receive(sock, deadline=time.monotonic() + 120)
    cfg = init["config"]
    corpus = P.Corpus(49)
    cases = {c.frame_id: c for c in corpus.cases}
    decisions = {e["decision_index"]: e for e in cfg["events"] if e["kind"] == "POLICY_DECISION"}
    CH.send(sock, dict(op="READY", pid=os.getpid()), deadline=time.monotonic() + 30)
    due, rows, results = {}, {}, 0
    while results < len(decisions):
        timeout = 1.0
        if due:
            timeout = max(0.0, (min(due.values()) - raw()) / 1e9)
        if select.select([sock], [], [], timeout)[0]:
            message, extra = CH.receive(sock, deadline=time.monotonic() + 30)
            if message["op"] == "SEND":
                k = message["k"]
                e = decisions[k]
                due[k] = message["send_raw_ns"] + int(e["compute_end_after_send_ms"] * 1e6)
                rows[k] = dict(k=k, frame_id=e["frame_id"], send_raw_ns=message["send_raw_ns"],
                               scheduled_prediction_raw_ns=due[k], forwarded_send_received_raw_ns=raw())
            elif message["op"] == "RESULT":
                rows[message["k"]].update(result_received_raw_ns=raw(), result=message["result"])
                results += 1
            continue
        for k in [k for k, t in due.items() if t <= raw()]:
            del due[k]
            e, row = decisions[k], rows[k]
            case = cases[e["frame_id"]]
            pred = corpus.predictions[case.frame_id]
            row["prediction_start_raw_ns"] = start = raw()
            c0 = time.thread_time_ns()
            identity, labels, records = edge_prediction(pred["identity"], pred["labels"], pred["records"])
            out, size = CH.seal_bytes([labels])
            try:
                CH.send(sock, dict(op="PREDICTION", k=k, identity=identity, records=records, size=size,
                                   frame_context=PO.context_document(corpus.context(case))),
                        fd=out, deadline=time.monotonic() + 30)
            finally:
                os.close(out)
            row.update(prediction_sent_raw_ns=raw(), edge_cpu_ms=(time.thread_time_ns() - c0) / 1e6,
                       lateness_vs_schedule_ms=(start - row["scheduled_prediction_raw_ns"]) / 1e6)
    with open(cfg["record_path"], "x") as stream:
        json.dump(dict(records=sorted(rows.values(), key=lambda r: r["k"]), pid=os.getpid()), stream)
    CH.send(sock, dict(op="DONE"), deadline=time.monotonic() + 30)
    sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--cpu", type=int, default=None)
    a = parser.parse_args()
    main(a.fd, a.parent, a.cpu)
