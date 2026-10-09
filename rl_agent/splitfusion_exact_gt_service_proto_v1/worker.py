"""Worker process for one service role: ``prep`` or ``score``.

Inherited private SEQPACKET socket, per-launch nonce, strictly sequential requests,
one outstanding request, at most one sealed fd per message. A ``Refused``
protocol outcome leaves state unchanged and the worker continues; any other
exception is reported and the worker exits (fail closed). CPU only; no affinity
or priority change: the child inherits the parent's scheduling as-is.
"""
import os
if __name__ == "__main__":
    os.environ.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})

import argparse
import ctypes
import select
import signal
import socket
import sys
import time

from . import channel as CH
from .scorer_core import ProtoScorer, Refused, triple_key


class ScoreCore:
    def __init__(self, config):
        self.core = ProtoScorer(**config)

    def ready(self):
        return dict(max_jobs=self.core.max_jobs, max_bytes=self.core.max_bytes)

    def handle(self, op, args, fd):
        c = self.core
        if op == "TARGET":
            packet = CH.read_sealed(fd, args["size"])
            return c.admit_target(args["identity"].encode(), args["semantic_sha256"], packet), None
        if op == "PREDICTION":
            labels = CH.read_sealed(fd, args["size"])
            return c.admit_prediction(args["identity"], args["frame_context"], args["records"], labels), None
        if op == "GET_RESULT":
            return dict(result=c.get_result(args["identity"].encode())), None
        if op == "RESULT_DELIVERED":
            return dict(ok=c.result_delivered(args["identity"].encode())), None
        if op == "ABANDON":
            return c.abandon(triple_key(args["capture"], args["action"], args["frame_context"]), args["reason"]), None
        if op == "ARCHIVE_BUILD":
            files = c.archive_build(args["identity"].encode())
            index, chunks = CH.pack_files(files)
            out, _ = CH.seal_bytes(chunks)
            return dict(index=index), out
        if op == "ARCHIVE_DURABLE":
            return dict(proof=c.archive_durable(args["identity"].encode(), args["receipt"], args["relative_prefix"])), None
        if op == "SNAPSHOT":
            return c.snapshot(), None
        if op == "CLOSE":
            return c.close(abandon=bool(args.get("abandon"))), None
        raise ValueError("unknown score operation " + str(op))


def serve(fd, parent_pid, cpu=None):
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})   # per-process pin of this worker only (benchmark parity with runtime children)
    libc = ctypes.CDLL(None, use_errno=True)
    CH.require(libc.prctl(1, signal.SIGTERM, 0, 0, 0) == 0, "cannot set parent-death signal")
    CH.require(os.getppid() == parent_pid, "wrong or dead parent")
    sock = socket.socket(fileno=fd)
    init, extra = CH.receive(sock, deadline=time.monotonic() + 60)
    CH.require(extra is None and set(init) == {"nonce", "role", "config"} and init["role"] in ("prep", "score"),
               "worker INIT differs")
    nonce = init["nonce"]
    if init["role"] == "prep":
        from .prep_core import PrepCore
        core = PrepCore(init["config"])
    else:
        core = ScoreCore(init["config"])
    import torch
    torch.set_num_threads(1)
    CH.send(sock, dict(status="READY", nonce=nonce, role=init["role"], pid=os.getpid(), parent_pid=os.getppid(),
                       cuda_initialized=torch.cuda.is_initialized(), ready=core.ready()),
            deadline=time.monotonic() + 60)
    last = 0
    while True:
        while not select.select([sock], [], [], 1.0)[0]:
            CH.require(os.getppid() == parent_pid, "parent changed while idle")
        message, incoming = CH.receive(sock, deadline=time.monotonic() + 60)
        outgoing = None
        try:
            CH.require(message.get("nonce") == nonce and message.get("seq") == last + 1, "worker session/sequence differs")
            last += 1
            begin, cpu_begin = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), time.thread_time_ns()
            try:
                result, outgoing = core.handle(message["op"], message["args"], incoming)
                reply = dict(status="OK", result=result)
            except Refused as refused:
                reply = dict(status="REFUSED", reason=refused.reason)
            except BaseException as error:
                reply = dict(status="FAILED", error=f"{type(error).__name__}: {error}"[:2000])
        finally:
            if incoming is not None:
                os.close(incoming)
        reply["timing"] = dict(begin_raw_ns=begin, end_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                               thread_cpu_ns=time.thread_time_ns() - cpu_begin)
        try:
            CH.send(sock, dict(reply, nonce=nonce, seq=last), fd=outgoing, deadline=time.monotonic() + 60)
        finally:
            if outgoing is not None:
                os.close(outgoing)
        if reply["status"] == "FAILED":
            sys.exit(1)
        if message["op"] == "CLOSE" and reply["status"] == "OK":
            sock.close()
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--cpu", type=int, default=None)
    arguments = parser.parse_args()
    serve(arguments.fd, arguments.parent, arguments.cpu)
