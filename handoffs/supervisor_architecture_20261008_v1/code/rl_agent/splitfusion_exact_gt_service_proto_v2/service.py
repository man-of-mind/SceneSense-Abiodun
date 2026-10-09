"""Parent-side worker clients and an offline coordinator harness.

The harness plays the UE (freeze, plan, send confirmation), the edge (prediction
reference, result receipt) and the durable archive owner, using the unchanged
``ArchiveClient`` for real fsync. It decides nothing about the 170-ms deadline and
records no durations. It is a correctness harness, not the live coordinator.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import secrets
import select
import subprocess
import sys
import time

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_live_gt_handoff_v1.archive import ArchiveClient

from . import channel as CH, freeze_codec as FZ

ROOT = Path(__file__).resolve().parents[2]
ENV = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}


class WorkerFailed(RuntimeError):
    pass


@dataclass
class Reply:
    status: str
    result: dict | None
    reason: str | None
    fd: int | None
    timing: dict | None = None

    @property
    def ok(self):
        return self.status == "OK"


class WorkerClient:
    def __init__(self, role, config, *, log_path, timeout_s=120.0, cpu=None):
        self.role, self.timeout = role, float(timeout_s)
        self.nonce, self.seq, self.pending = secrets.token_hex(32), 0, None
        self.failed = self.closed = False
        parent, child = CH.pair()
        self.sock = parent
        self.log = open(log_path, "xb")
        env = dict(os.environ, **ENV)
        env.pop("PYTHONPATH", None)
        try:
            self.process = subprocess.Popen([sys.executable, "-B", "-u", "-m",
                "rl_agent.splitfusion_exact_gt_service_proto_v2.worker", "--fd", str(child.fileno()),
                "--parent", str(os.getpid()), *([] if cpu is None else ["--cpu", str(cpu)])], pass_fds=(child.fileno(),), cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log)
            child.close()
            CH.send(parent, dict(nonce=self.nonce, role=role, config=config), deadline=time.monotonic() + self.timeout)
            ready, extra = CH.receive(parent, deadline=time.monotonic() + self.timeout)
            C.require(extra is None and ready.get("status") == "READY" and ready.get("nonce") == self.nonce
                      and ready.get("pid") == self.process.pid and ready.get("role") == role
                      and ready.get("cuda_initialized") is False, "worker READY binding differs: " + str(ready))
            self.ready = ready["ready"]
        except BaseException:
            child.close()
            self.abort()
            raise

    def begin(self, op, args=None, fd=None):
        C.require(not self.failed and not self.closed and self.pending is None, "worker busy/failed/closed")
        self.seq += 1
        self.pending = (self.seq, op)
        CH.send(self.sock, dict(nonce=self.nonce, seq=self.seq, op=op, args=args or {}), fd=fd,
                deadline=time.monotonic() + self.timeout)

    def available(self):
        return bool(select.select([self.sock], [], [], 0)[0])

    def finish(self):
        C.require(self.pending is not None, "no outstanding request")
        seq, op = self.pending
        try:
            message, fd = CH.receive(self.sock, deadline=time.monotonic() + self.timeout)
        except BaseException:
            self.failed = True
            raise
        self.pending = None
        C.require(message.get("nonce") == self.nonce and message.get("seq") == seq, "foreign worker reply")
        if message["status"] == "FAILED":
            self.failed = True
            if fd is not None:
                os.close(fd)
            raise WorkerFailed(f"{self.role} {op}: {message.get('error')}")
        return Reply(message["status"], message.get("result"), message.get("reason"), fd, message.get("timing"))

    def call(self, op, args=None, fd=None):
        self.begin(op, args, fd)
        return self.finish()

    def close(self, *, abandon=False):
        reply = self.call("CLOSE", dict(abandon=abandon))
        self.process.wait(timeout=self.timeout)
        C.require(self.process.returncode == 0, "worker exited unsuccessfully")
        self.closed = True
        self.sock.close(); self.log.close()
        return reply

    def abort(self):
        self.failed = self.closed = True
        try:
            self.sock.close()
        except OSError:
            pass
        if getattr(self, "process", None) is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait(timeout=2)
        if not self.log.closed:
            self.log.close()


def ok(reply, what):
    C.require(reply.ok, f"{what} not OK: {reply.status} {reply.reason}")
    return reply.result


class Harness:
    """Offline UE/edge/archive roles around one prep and one score worker."""

    def __init__(self, directory, *, prep_config, score_config, archive_frames=200, archive_jobs=4):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.archive_root = self.directory / "durable"
        self.archive_root.mkdir()
        self.prep = self.score = self.archive = None
        try:
            self.prep = WorkerClient("prep", prep_config, log_path=self.directory / "prep.log")
            self.score = WorkerClient("score", score_config, log_path=self.directory / "score.log")
            self.archive = ArchiveClient(roots=[str(self.archive_root.resolve())], max_jobs=archive_jobs,
                max_bytes=256 * 1024 * 1024, max_frames=archive_frames,
                binding=dict(scope="exact_gt_service_proto_v2", run=prep_config["engine"]["run_binding"]))
        except BaseException:
            self.abort()
            raise
        self.session = self.prep.ready["session_id"]
        self.sequence = 0

    # ---------------- UE role ----------------
    def freeze(self, sourcekwargs, policy_timestamp_s, *, offline=None):
        self.sequence += 1
        descriptor, chunks = FZ.freeze(sourcekwargs, policy_timestamp_s, session_id=self.session,
                                       sequence=self.sequence)
        fd, _ = FZ.seal(chunks)
        try:
            args = dict(descriptor=descriptor.decode())
            if offline is not None:
                args["offline"] = offline
            self.prep.begin("FREEZE", args, fd=fd)
        finally:
            os.close(fd)
        return self.sequence

    def finish_freeze(self):
        return ok(self.prep.finish(), "FREEZE")

    def publish_to_scorer(self, job):
        reply = self.prep.call("PUBLISH", dict(job=job))
        published = ok(reply, "PUBLISH")
        try:
            target = self.score.call("TARGET", dict(identity=published["identity"],
                semantic_sha256=published["semantic_sha256"], size=published["size"]), fd=reply.fd)
        finally:
            os.close(reply.fd)
        return published, target

    # ---------------- edge role ----------------
    def predict(self, identity, frame_context_doc, records, labels: bytes, *, begin_only=False):
        fd, size = CH.seal_bytes([labels])
        try:
            self.score.begin("PREDICTION", dict(identity=identity, frame_context=frame_context_doc,
                                                records=records, size=size), fd=fd)
        finally:
            os.close(fd)
        return None if begin_only else self.score.finish()

    # ---------------- archive owner ----------------
    def files_of(self, reply):
        try:
            return CH.unpack_files(reply.result["index"], reply.fd)
        finally:
            os.close(reply.fd)

    def durable(self, identity, prefix, files):
        paths = {str(self.archive_root / prefix / name): raw for name, raw in files.items()}
        job = self.archive.submit(identity, paths)
        deadline = time.monotonic() + 60
        while True:
            self.archive.poll()
            done = [r for r in self.archive.completions if r["job_id"] == job]
            if done:
                C.require(len(done) == 1 and done[0]["durable"], "one durable completion required")
                for name, raw in paths.items():
                    with open(name, "rb") as stream:
                        C.require(stream.read() == raw, "durable readback differs")
                return done[0]
            C.require(time.monotonic() < deadline, "archive completion timeout")
            select.select([self.archive.socket], [], [], 0.5)

    def retire_scored(self, job, capture, scorer_identity, frame_id):
        """Off-path evidence + durability for one delivered scored frame, both sides."""
        sfiles = self.files_of(self.score.call("ARCHIVE_BUILD", dict(identity=scorer_identity)))
        sprefix = f"frame_{frame_id}/scorer"
        sreceipt = self.durable(dict(capture, evidence_role="scorer"), sprefix, sfiles)
        proof = ok(self.score.call("ARCHIVE_DURABLE", dict(identity=scorer_identity, receipt=sreceipt,
                                                            relative_prefix=sprefix)), "ARCHIVE_DURABLE")["proof"]
        pfiles = self.files_of(self.prep.call("EVIDENCE", dict(job=job)))
        pprefix = f"frame_{frame_id}/producer"
        preceipt = self.durable(capture, pprefix, pfiles)
        outcome = ok(self.prep.call("EVIDENCE_DURABLE", dict(job=job, receipt=preceipt, relative_prefix=pprefix,
                                                              scorer_proof=proof)), "EVIDENCE_DURABLE")
        return sfiles, pfiles, proof, outcome

    def retire_unscored(self, job, capture, scorer_identity, frame_id, reason, *, scorer_holds_target=True):
        if scorer_holds_target:
            sfiles = self.files_of(self.score.call("ARCHIVE_BUILD", dict(identity=scorer_identity)))
            sprefix = f"frame_{frame_id}/scorer"
            sreceipt = self.durable(dict(capture, evidence_role="scorer"), sprefix, sfiles)
            ok(self.score.call("ARCHIVE_DURABLE", dict(identity=scorer_identity, receipt=sreceipt,
                                                       relative_prefix=sprefix)), "ARCHIVE_DURABLE")
        pfiles = self.files_of(self.prep.call("UNSCORED_TERMINAL", dict(job=job, reason=reason)))
        pprefix = f"frame_{frame_id}/producer"
        preceipt = self.durable(capture, pprefix, pfiles)
        return ok(self.prep.call("UNSCORED_DURABLE", dict(job=job, receipt=preceipt, relative_prefix=pprefix)),
                  "UNSCORED_DURABLE")

    def snapshots(self):
        return ok(self.prep.call("SNAPSHOT"), "prep SNAPSHOT"), ok(self.score.call("SNAPSHOT"), "score SNAPSHOT")

    def close(self):
        out = dict(prep=ok(self.prep.close(), "prep CLOSE"), score=ok(self.score.close(), "score CLOSE"),
                   archive=self.archive.close())
        return out

    def abort(self):
        for client in (self.prep, self.score):
            if client is not None and not client.closed:
                client.abort()
        if self.archive is not None and not self.archive.closed:
            self.archive.abort()
