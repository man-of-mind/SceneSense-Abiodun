"""Dedicated CPU engines behind bounded inherited-socket capabilities.

One outstanding command per child, no hidden retry or unbounded command queue.
The parent may submit to another child while one computes. A failure invalidates
this client; shutdown reaps only its own child. This is not a live route launcher.
"""
import os
ENV = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
if __name__ == "__main__":
    os.environ.update(ENV)

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import secrets
import select
import signal
import socket
import struct
import subprocess
import sys
import time

from . import ipc as I

ROOT = Path(__file__).resolve().parents[2]
LIVE_PROCESS_CONTRACT = {
    "schema": "splitfusion.numeric_gt.live_process.v1",
    "clock": "CLOCK_MONOTONIC_RAW",
    "deadline_ns": 170_000_000,
    "inclusive_deadline": True,
    "scored_release": "SCORE_AND_OBSERVED_TERMINAL_AND_DURABLE_COMPLETION_SIDECAR",
    "unscored_release": "OBSERVED_UE_TIMEOUT_OR_PACKET_VERIFIED_SERVICE_FAULT_AND_DURABLE_EXPLICIT_TERMINAL",
    "legacy_recorded_archive_clocks": False,
}


def verify_sources(sources):
    I.require(type(sources) is dict and 1 <= len(sources) <= 1000, "bounded startup source binding required")
    for name, expected in sources.items():
        path = Path(name)
        I.require(not path.is_absolute() and ".." not in path.parts, "unsafe startup source path")
        path = ROOT / path
        I.require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT), "foreign startup source")
        I.require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "startup source changed: " + name)


def context(document):
    from rl_agent.splitfusion_live_dispatch_v1.frame_context import FrameContextV1, Pose6D, validate_frame_context
    result = FrameContextV1(**{k: v for k, v in document.items() if k != "ego_world"},
                            ego_world=Pose6D(*document["ego_world"]))
    validate_frame_context(result)
    return result


class Engine:
    def __init__(self, role, config):
        self.role, self.config = role, config
        self.tokens = {}
        self.staged_source = None
        if role == "producer":
            from .producer import ProducerEngine
            arguments = dict(config)
            if arguments.get("offline_replay") is False:
                I.require(arguments.pop("live_process_contract", None) == LIVE_PROCESS_CONTRACT,
                          "explicit observed-clock live lifecycle contract required")
            else:
                I.require(arguments.get("offline_replay") is True
                    and "live_process_contract" not in arguments,
                    "explicit offline replay or bound live process contract required")
            self.engine = ProducerEngine(**arguments)
        elif role == "scorer":
            from .scorer import ScorerEngine
            self.engine = ScorerEngine(**config)
        elif role == "echo_test_only":
            I.require(config == {"offline_test": True}, "test engine has no production authority")
            self.engine = None
        else:
            raise ValueError("unknown worker role")

    def call(self, operation, value):
        if operation == "SNAPSHOT":
            return {} if self.engine is None else self.engine.snapshot()
        if operation == "CLOSE":
            return {} if self.engine is None else self.engine.close()
        if self.role == "echo_test_only":
            I.require(operation == "ECHO", "unknown test operation")
            return value
        if self.role == "producer":
            if operation == "STAGE_SOURCE":
                from .staging import decode_stage
                from .live_clock import read_boot_id
                I.require(self.staged_source is None and type(value["job"]) is int
                    and value["job"] not in self.tokens, "one bounded source stage; no reused job")
                kwargs = decode_stage(value["source_packet"], value["stage_extras"])
                self.staged_source = (value["job"], kwargs)
                # The child owns this observation: it is made only after the
                # entire native packet has been decoded and accepted. The
                # caller can later bind the neutral capture to the admitted
                # full ReadyTarget identity without rereading the clock.
                received = I.raw()
                return dict(job=value["job"], capture_template=kwargs["identity"],
                            source_bytes=len(value["source_packet"]),
                            source_stage_received_raw_ns=received,
                            source_stage_owner_pid=os.getpid(),
                            source_stage_boot_id=read_boot_id())
            if operation == "CANCEL_STAGED":
                I.require(self.staged_source is not None
                    and self.staged_source[0] == value["job"], "foreign staged cancellation")
                self.staged_source = None
                return dict(job=value["job"], cancelled_before_plan=True)
            if operation == "PREPARE":
                from .source_codec import decode_source
                I.require(value["job"] not in self.tokens, "reused producer job")
                kwargs = decode_source(value["source_packet"], value["source_extras"])
            elif operation == "PREPARE_STAGED":
                from .staging import bind_action
                I.require(self.staged_source is not None and self.staged_source[0] == value["job"]
                    and value["job"] not in self.tokens, "unowned/duplicate staged preparation")
                kwargs = bind_action(self.staged_source[1], value["capture_identity"])
                self.staged_source = None
            else:
                kwargs = None
            if kwargs is not None:
                prepared = self.engine.prepare(value["plan_bytes"], kwargs,
                    frame_context=context(value["frame_context"]), population=value["population"],
                    archive_clocks=value["archive_clocks"], payload_filenames=value["payload_filenames"],
                    prescore_filename=value["prescore_filename"], replay_observed_raw_ns=value.get("replay_observed_raw_ns"))
                self.tokens[value["job"]] = prepared.token
                return dict(job=value["job"], identity_bytes=prepared.identity_bytes,
                            semantic_sha256=prepared.semantic_sha256)
            token = self.tokens[value["job"]]
            if operation == "CONFIRM_SEND":
                return self.engine.confirm_send(token, value["plan_bytes"])
            if operation == "CANCEL":
                result = self.engine.cancel(token, value["plan_bytes"], value["reason"])
                del self.tokens[value["job"]]
                return result
            if operation == "PUBLISH_READY":
                return self.engine.publish_ready(token)
            if operation == "EXPORT_LIVE_SOURCE":
                exported = self.engine.export_live_source(token, value["identity_bytes"],
                    value["semantic_sha256"])
                return {name: getattr(exported, name) for name in (
                    "schema", "identity_bytes", "semantic_sha256", "input_metadata_bytes",
                    "source_synchronization_bytes", "numerical_avo_bytes",
                    "producer_events_bytes", "payload_bytes")}
            if operation == "COMPLETE_LIVE_SCORED":
                result = self.engine.complete_live_scored(token, value["identity_bytes"],
                    value["semantic_sha256"], value["scorer_proof"])
                del self.tokens[value["job"]]
                return result
            if operation == "PREPARE_LIVE_UNSCORED_TERMINAL":
                result = self.engine.prepare_live_unscored_terminal(token, value["identity_bytes"],
                    value["semantic_sha256"], value["clock_descriptor_bytes"], value["reason"],
                    fault_feedback_packet_bytes=value.get("fault_feedback_packet_bytes"),
                    ue_received_packet_sha256=value.get("ue_received_packet_sha256"))
                return dict(identity_bytes=result.identity_bytes, semantic_sha256=result.semantic_sha256,
                    files=dict(result.files), terminal_sha256=result.terminal_sha256)
            if operation == "COMPLETE_LIVE_UNSCORED":
                result = self.engine.complete_live_unscored(token, value["identity_bytes"],
                    value["semantic_sha256"], value["terminal_sha256"], value["receipt"],
                    relative_prefix=value["relative_prefix"])
                del self.tokens[value["job"]]
                return result
            if operation == "SCORE_COMPLETED":
                return self.engine.score_completed(token, value["identity_bytes"], value["semantic_sha256"])
            if operation == "FEEDBACK_CONFIRMED":
                return self.engine.feedback_confirmed(token, value["identity_bytes"], value["semantic_sha256"])
            if operation == "BUILD_ARCHIVE":
                result = self.engine.build_archive(token)
                return dict(descriptor_bytes=result.descriptor_bytes, files=dict(result.files),
                    identity_bytes=result.identity_bytes, semantic_sha256=result.semantic_sha256,
                    archive_sha256=result.archive_sha256)
            if operation == "SCORER_ARCHIVE_COMPLETED":
                return self.engine.scorer_archive_completed(token, value["identity_bytes"], value["semantic_sha256"])
            if operation == "ARCHIVE_DURABLE":
                result = self.engine.archive_durable(token, value["receipt"], relative_prefix=value["relative_prefix"])
                del self.tokens[value["job"]]
                return result
            raise ValueError("unknown producer operation")
        if self.role == "scorer":
            key = value["key"]
            if operation == "REGISTER":
                expected = dict(value["expected"], frame_context=context(value["expected"]["frame_context"]))
                return self.engine.register(key, expected=expected, reservation_bytes=value["reservation_bytes"])
            if operation == "TARGET":
                return self.engine.put_target(key, value["packet"])
            if operation == "PREDICTION":
                return self.engine.put_prediction(key, identity=value["identity"], labels=value["labels"],
                    shape=tuple(value["shape"]), records=value["records"], prediction_contract=value["prediction_contract"])
            if operation == "SCORE":
                return self.engine.score_ready(key)
            if operation == "FEEDBACK_CONFIRMED":
                return self.engine.confirm_feedback(key)
            if operation == "ATTACH_ARCHIVE":
                return self.engine.attach_archive(key, **value["arguments"])
            if operation == "ATTACH_LIVE_ARCHIVE":
                return self.engine.attach_live_archive(key, **value["arguments"])
            if operation == "ARCHIVE_DURABLE":
                return self.engine.confirm_archive(key, receipt=value["receipt"], relative_prefix=value["relative_prefix"])
            if operation == "LIVE_ARCHIVE_DURABLE":
                return self.engine.confirm_live_archive(key, receipt=value["receipt"],
                    relative_prefix=value["relative_prefix"],
                    archive_submitted_event=value["archive_submitted_event"],
                    durable_completion_event=value["durable_completion_event"])
            if operation == "LIVE_COMPLETION_DURABLE":
                return self.engine.confirm_live_completion_durable(key, proof=value["proof"])
            raise ValueError("unknown scorer operation")


class Client:
    def __init__(self, role, config, *, directory, source_sha256, cpu=4, timeout_s=30,
                 writable_attempt_root=None):
        I.require(role in ("producer", "scorer", "echo_test_only") and 0 < timeout_s <= 60, "bounded role/timeout")
        I.require(type(cpu) is int and cpu in os.sched_getaffinity(0), "child CPU outside current allowed mask")
        self.role, self.timeout, self.sources = role, timeout_s, source_sha256
        self.failed = self.closed = False
        self.pending = None
        self.sequence = 0
        self.events = []
        self.nonce = secrets.token_hex(32)
        self.process = self.socket = self.log = None
        directory = Path(directory).absolute()
        root = ROOT if writable_attempt_root is None else Path(writable_attempt_root).absolute()
        I.require(root.is_dir() and not root.is_symlink() and root.stat().st_uid == os.getuid()
            and root.resolve() == root and not any(p.is_symlink() for p in root.parents),
            "explicit real owned writable attempt root required")
        I.require(not directory.exists() and directory.resolve().is_relative_to(root)
            and not any(p.is_symlink() for p in directory.parents if p.is_relative_to(root)),
            "fresh owned worker-log directory")
        directory.mkdir(parents=True, exist_ok=False)
        parent, child = socket.socketpair()
        self.socket = parent
        try:
            self.log = (directory / "worker.log").open("xb")
            environment = dict(os.environ, **ENV)
            environment.pop("PYTHONPATH", None)
            # The retained UE thread-cap experiment restores baseline caps in
            # Popen.__init__, including explicitly supplied environments. Apply
            # this CPU child's fixed caps at exec instead: numerical imports and
            # /proc/PID/environ then agree, without changing the parent's caps or
            # the retained restoration rule for any other child.
            command = ["/usr/bin/env", *(f"{key}={value}" for key, value in ENV.items()),
                       sys.executable, "-B", "-u", "-m", __name__]
            self.process = subprocess.Popen(command + ["--fd", str(child.fileno()),
                "--parent", str(os.getpid()), "--cpu", str(cpu)], pass_fds=(child.fileno(),),
                stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log, cwd=ROOT, env=environment)
            child.close()
            I.send(parent, dict(nonce=self.nonce, role=role, config=config, sources=source_sha256),
                   deadline=time.monotonic() + self.timeout)
            ready = I.receive(parent, deadline=time.monotonic() + self.timeout)
            I.require(ready.get("status") == "READY" and ready.get("nonce") == self.nonce
                and ready.get("pid") == self.process.pid and ready.get("parent_pid") == os.getpid()
                and ready.get("role") == role and ready.get("affinity") == [cpu]
                and ready.get("cuda_initialized") is False, "worker READY binding differs: " + str(ready))
            self.ready = ready
        except BaseException:
            child.close(); self.failed = True; self.abort()
            raise

    def begin(self, operation, value):
        I.require(not self.failed and not self.closed and self.pending is None, "worker busy/failed/closed")
        I.require(len(self.events) < 20000, "worker event bound")
        self.sequence += 1
        self.pending = dict(seq=self.sequence, operation=operation, parent_begin_raw_ns=I.raw())
        try:
            self.pending["sent_bytes"] = I.send(self.socket,
                dict(nonce=self.nonce, seq=self.sequence, operation=operation, value=value),
                deadline=time.monotonic() + self.timeout)
        except BaseException:
            self.failed = True
            raise

    def available(self):
        I.require(self.pending is not None and not self.closed, "no outstanding command")
        return bool(select.select([self.socket], [], [], 0)[0])

    def finish(self):
        I.require(self.pending is not None and not self.closed, "no outstanding command")
        try:
            result = I.receive(self.socket, deadline=time.monotonic() + self.timeout)
            I.require(result.get("nonce") == self.nonce and result.get("seq") == self.pending["seq"]
                and result.get("pid") == self.process.pid, "foreign worker response")
            event = dict(self.pending, parent_end_raw_ns=I.raw(), worker=result["timing"], status=result["status"])
            self.events.append(event)
            self.pending = None
            I.require(result["status"] == "OK", "worker failed: " + str(result.get("error")))
            return result["value"]
        except BaseException:
            self.failed = True
            raise

    def call(self, operation, value=None):
        self.begin(operation, {} if value is None else value)
        return self.finish()

    def close(self):
        if self.closed:
            I.require(not self.failed, "worker aborted/failed")
            return self.close_result
        try:
            self.close_result = self.call("CLOSE")
            self.process.wait(timeout=self.timeout)
            I.require(self.process.returncode == 0, "worker exit nonzero")
            self.closed = True
            self.socket.close(); self.log.close()
            return self.close_result
        except BaseException:
            self.failed = True; self.abort()
            raise

    def abort(self):
        self.failed = self.closed = True
        if self.socket is not None:
            self.socket.close()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait(timeout=2)
        if self.log is not None:
            self.log.close()


def receive_parent_command(sock, parent_pid):
    """Idle is not a broken IPC request: start the read bound at readability.

    The owner may legitimately await a late terminal/archive without sending
    a command. Its death remains guarded by PR_SET_PDEATHSIG, SO_PEERCRED and
    this parent check. EOF is readable and goes through the normal fail-closed
    decoder; a partial/malformed request still has a 60-second read bound.
    """
    while True:
        I.require(os.getppid() == parent_pid, "worker parent changed while idle")
        ready, _, _ = select.select([sock], [], [], 1.)
        if ready:
            return I.receive(sock, deadline=time.monotonic() + 60)


def serve(fd, parent_pid, cpu):
    I.require(os.getppid() == parent_pid, "wrong worker parent")
    libc = ctypes.CDLL(None, use_errno=True)
    I.require(libc.prctl(1, signal.SIGTERM, 0, 0, 0) == 0, "cannot set parent-death signal")
    I.require(os.getppid() == parent_pid, "parent died before setup")
    os.sched_setaffinity(0, {cpu})
    sock = socket.socket(fileno=fd)
    peer = struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    I.require(peer[0] == parent_pid and peer[1] == os.getuid(), "inherited channel creator identity differs")
    init = I.receive(sock, deadline=time.monotonic() + 60)
    nonce = init["nonce"]
    I.require(type(nonce) is str and len(nonce) == 64, "worker launch nonce")
    verify_sources(init["sources"])
    engine = Engine(init["role"], init["config"])
    import torch
    import cv2
    torch.set_num_threads(1); cv2.setNumThreads(1)
    I.require(not torch.cuda.is_initialized(), "CPU worker initialized CUDA")
    I.send(sock, dict(status="READY", nonce=nonce, role=init["role"], pid=os.getpid(), parent_pid=os.getppid(),
        affinity=sorted(os.sched_getaffinity(0)), cuda_initialized=False, torch_threads=torch.get_num_threads(),
        source_files=len(init["sources"]), authentication="INHERITED_SOCKET_CAPABILITY_CREATOR_UID_PID_AND_NONCE"),
        deadline=time.monotonic() + 60)
    last = 0
    while True:
        message = receive_parent_command(sock, parent_pid)
        I.require(message["nonce"] == nonce and type(message["seq"]) is int and message["seq"] == last + 1,
                  "worker sequence/session differs")
        last += 1
        begin, cpu_begin = I.raw(), time.thread_time_ns()
        try:
            result = engine.call(message["operation"], message["value"])
            response = dict(status="OK", value=result)
        except BaseException as error:
            response = dict(status="FAILED", error=f"{type(error).__name__}: {error}")
        timing = dict(begin_raw_ns=begin, end_raw_ns=I.raw(), thread_cpu_ns=time.thread_time_ns() - cpu_begin)
        I.send(sock, dict(response, timing=timing, nonce=nonce, seq=last, pid=os.getpid()),
               deadline=time.monotonic() + 60)
        if response["status"] != "OK":
            raise RuntimeError(response["error"])
        if message["operation"] == "CLOSE":
            verify_sources(init["sources"])
            I.require(not torch.cuda.is_initialized(), "CPU worker initialized CUDA")
            sock.close()
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    args = parser.parse_args()
    serve(args.fd, args.parent, args.cpu)
