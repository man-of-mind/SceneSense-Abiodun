"""Host early-GT service: coordinator process for prep/score workers, UE and edge channels.

Spawned by the UE child (inherited SOCK_SEQPACKET pair). Listens for exactly one edge
connection on an AF_UNIX SEQPACKET socket inside the bind-mounted edge evidence directory;
the edge authenticates with SO_PEERCRED (uid 0 or the service uid) plus a per-launch nonce
from a 0600 capability file in the same directory.

Ordering and bounds:
* freeze admissions and refusal VOIDs reach the prep worker strictly in descriptor order;
* one unresolved producer transaction; freeze queue depth ``freeze_queue``; producer
  retention ``prep_jobs``; scorer slots ``scorer_jobs``; waiting predictions bounded in the
  score worker; archive outstanding <= 4; nothing is evicted;
* results go to the edge the moment the score worker replies; map publication is never
  involved; evidence/durability run at the lowest priority and only after the frame's real
  UE terminal is known.
Every decision frame ends in exactly one outcome: SCORED, UNSCORED(reason), CANCELLED or
NO_GT(reason). Unscored outcomes return an explicit fault to the edge (retained
GROUND_TRUTH_UNAVAILABLE semantics); no score or quality is ever fabricated.
"""
import os
if __name__ == "__main__":
    os.environ.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})

import argparse
import ctypes
import hashlib
import heapq
import itertools
import json
from pathlib import Path
import secrets
import select
import signal
import socket
import stat
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "splitfusion.exact_gt_live.service.v1"
CLOCK_SCHEMA = "splitfusion.exact_gt_live.frame_clock.v1"


def raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def verify_sources(sources):
    for relative, digest in sources.items():
        path = ROOT / relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts or not path.is_file()
                or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise ValueError("exact-GT service source drift: " + relative)


def rss_kib(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        return None


class Service:
    def __init__(self, ue_sock, config):
        from rl_agent.splitfusion_exact_gt_service_proto_v2 import service as SV, channel as CH
        from rl_agent.splitfusion_live_gt_handoff_v1.archive import ArchiveClient
        self.SV, self.CH = SV, CH
        self.cfg = config
        self.ue = ue_sock
        self.limits = config["limits"]
        self.out = Path(config["evidence_dir"])
        self.out.mkdir(parents=True, exist_ok=False)
        self.socket_path = Path(config["socket_path"])
        self.capability_path = Path(config["capability_path"])
        self.nonce = secrets.token_hex(32)
        self.counter = itertools.count()
        self.frames, self.order = {}, []
        self.queues = {"prep": [], "score": []}
        self.current = {"prep": None, "score": None}
        self.gate_ops = []              # ordered [("FREEZE"|"VOID", frame_id, args, fd)]
        self.prep_open = None
        self.archive_jobs, self.archive_pending = {}, []
        self.archive_seen = 0
        self.samples, self.memory, self.refusals, self.errors = [], [], [], []
        self.edge = None
        self.edge_pid = None
        self.ue_closed = self.edge_closed = False
        logs = self.out / "workers"
        logs.mkdir()
        self.prep = SV.WorkerClient("prep", dict(engine=config["engine"]), log_path=logs / "prep.log",
                                    cpu=self.limits["prep_cpu"])
        self.score = SV.WorkerClient("score", config["score"], log_path=logs / "score.log", cpu=self.limits["score_cpu"])
        (self.out / "archive").mkdir()
        self.archive_root = (self.out / "archive").resolve()
        self.archive = ArchiveClient(roots=[str(self.archive_root)], max_jobs=4, max_bytes=256 * 1024 * 1024,
                                     max_frames=4 * self.limits["max_frames"], binding=dict(config["binding"]))
        self.session = self.prep.ready["session_id"]
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        if self.socket_path.exists() or self.capability_path.exists():
            raise ValueError("fresh service socket/capability path required")
        from rl_agent.splitfusion_live_gt_handoff_v1 import broker as B
        with B._address(self.socket_path) as address:     # retained long-path-safe bind
            self.listener.bind(address)
        os.chmod(self.socket_path, 0o600)
        self.listener.listen(1)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        fd = os.open(self.capability_path, flags, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(dict(schema=SCHEMA, nonce=self.nonce, socket=self.socket_path.name, service_pid=os.getpid()), stream)
        self.started_raw_ns = raw()

    # ------------------------------------------------------------------ helpers
    def frame(self, frame_id):
        if frame_id not in self.frames:
            if len(self.frames) >= self.limits["max_frames"]:
                raise ValueError("decision frame budget exceeded")
            self.frames[frame_id] = dict(frame_id=frame_id, events=[])
            self.order.append(frame_id)
        return self.frames[frame_id]

    def stamp(self, f, role, name, raw_ns=None, **extra):
        f["events"].append(dict(role=role, name=name, raw_ns=raw() if raw_ns is None else int(raw_ns), **extra))

    def enqueue(self, worker, priority, op, args, frame_id, fd=None):
        heapq.heappush(self.queues[worker], (priority, next(self.counter), op, args, frame_id, fd))

    def send_ue(self, value):
        self.CH.send(self.ue, value, deadline=time.monotonic() + 10)

    def send_edge(self, value):
        if self.edge is not None and not self.edge_closed:
            self.CH.send(self.edge, value, deadline=time.monotonic() + 10)

    def counts(self):
        committed = sum(1 for f in self.frames.values() if f.get("committed") and not f.get("prep_retired"))
        slots = sum(1 for f in self.frames.values() if f.get("target_admitted") and not f.get("score_retired"))
        queued = sum(1 for op in self.gate_ops if op[0] == "FREEZE")
        return committed, slots, queued

    def sample(self, label):
        committed, slots, queued = self.counts()
        self.samples.append(dict(t=raw(), label=label, prep_queue=len(self.queues["prep"]),
            score_queue=len(self.queues["score"]), freeze_queue=queued, prep_open=self.prep_open,
            prep_retained=committed, scorer_slots=slots,
            waiting=sum(1 for f in self.frames.values() if f.get("waiting")),
            archive_outstanding=self.archive.outstanding_jobs + len(self.archive_pending)))

    # ------------------------------------------------------------------ UE messages
    def on_ue(self, message, fd):
        op = message["op"]
        if op == "CLOSE":
            self.ue_closed = True
            return
        f = self.frame(int(message["frame_id"]))
        if op == "OPEN":
            f.update(action_open_raw_ns=message["action_open_raw_ns"], decision=message["decision"])
            for name, value in message["events"].items():
                self.stamp(f, "ue", name, value)
            if message.get("freeze_refused"):
                f["no_gt"] = "UE_FREEZE_QUEUE_FULL"
                f["outcome"] = "NO_GT_UE_FREEZE_QUEUE_FULL"
                self.refusals.append((f["frame_id"], "UE_FREEZE_QUEUE_FULL"))
        elif op == "FREEZE":
            self.stamp(f, "service", "freeze_received")
            for name, value in message["events"].items():
                self.stamp(f, "ue_freeze_thread", name, value)
            f["freeze_stats"] = message["stats"]
            self.admit_freeze(f, message["descriptor"], fd)
        elif op == "PLAN":
            f.update(plan=message["plan"], frame_context=message["frame_context"])
            self.stamp(f, "service", "plan_received")
            self.try_bind(f)
        elif op == "SEND":
            self.stamp(f, "ue", "last_datagram_sent" if message["ok"] else "send_failed", message["raw_ns"])
            f["sent"] = message["ok"]
            if message["ok"]:
                self.try_confirm(f)
            else:
                self.cancel(f, "SEND_FAILED")
        elif op == "TERMINAL":
            if f.get("ue_terminal") is not None:
                raise ValueError("duplicate UE terminal")
            f["ue_terminal"] = message["name"]
            self.stamp(f, "ue", message["name"], message["raw_ns"])
            self.maybe_archive(f)

    def admit_freeze(self, f, descriptor, fd):
        committed, _, queued = self.counts()
        reason = None
        if self.prep_open is not None and queued >= self.limits["freeze_queue"]:
            reason = "FREEZE_QUEUE_FULL"
        elif committed + queued + (1 if self.prep_open is not None else 0) + 1 > self.limits["prep_jobs"] - 1:
            reason = "PRODUCER_CAPACITY"
        if reason is not None:
            os.close(fd)
            f["no_gt"] = reason
            self.refusals.append((f["frame_id"], reason))
            self.gate_ops.append(("VOID", f["frame_id"], dict(descriptor=descriptor), None))
        else:
            self.gate_ops.append(("FREEZE", f["frame_id"], dict(descriptor=descriptor), fd))
        self.pump_gate()

    def pump_gate(self):
        while self.gate_ops:
            kind, frame_id, args, fd = self.gate_ops[0]
            if kind == "FREEZE" and self.prep_open is not None:
                return
            self.gate_ops.pop(0)
            f = self.frames[frame_id]
            if kind == "FREEZE":
                if f.get("cancel_pending"):
                    # Unsent before admission: still advance the gate, nothing is computed.
                    os.close(fd)
                    f["no_gt"] = f.get("cancel_reason", "SEND_FAILED") + "_BEFORE_ADMISSION"
                    self.enqueue("prep", 0, "VOID", args, frame_id)
                    continue
                self.prep_open = frame_id
                self.stamp(f, "service", "freeze_dispatched")
                self.enqueue("prep", 0, "FREEZE", args, frame_id, fd)
            else:
                self.enqueue("prep", 0, "VOID", args, frame_id)

    def try_bind(self, f):
        if f.get("unbound") and f.get("plan") and not f.get("bind_queued") and not f.get("cancel_reason"):
            f["bind_queued"] = True
            self.enqueue("prep", 0, "BIND", dict(job=f["job"], plan=f["plan"], frame_context=f["frame_context"]), f["frame_id"])

    def try_confirm(self, f):
        if f.get("bound") and f.get("sent") and not f.get("confirm_queued") and not f.get("cancel_reason"):
            f["confirm_queued"] = True
            self.enqueue("prep", 0, "CONFIRM_SEND", dict(job=f["job"], plan=f["plan"]), f["frame_id"])

    def cancel(self, f, reason):
        if f.get("committed") or f.get("cancel_queued"):
            return
        f["cancel_reason"] = reason
        if f.get("bound") and not f.get("confirm_queued"):
            self.enqueue("prep", 0, "CANCEL_BOUND", dict(job=f["job"], plan=f["plan"], reason=reason), f["frame_id"])
            f["cancel_queued"] = True
        elif f.get("unbound") and not f.get("bound") and not f.get("bind_queued"):
            self.enqueue("prep", 0, "CANCEL_UNBOUND", dict(job=f["job"], reason=reason), f["frame_id"])
            f["cancel_queued"] = True
        else:
            f["cancel_pending"] = True     # applied when the freeze is admitted or bound

    # ------------------------------------------------------------------ edge messages
    def accept_edge(self):
        connection, _ = self.listener.accept()
        creds = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        pid, uid, gid = struct.unpack("3i", creds)
        hello, extra = self.CH.receive(connection, deadline=time.monotonic() + 10)
        if (extra is not None or self.edge is not None or uid not in (0, os.getuid())
                or hello.get("op") != "HELLO" or hello.get("nonce") != self.nonce):
            connection.close()
            self.errors.append(f"refused edge connection pid={pid} uid={uid}")
            return
        self.edge, self.edge_pid = connection, pid
        self.edge_hello = dict(pid=pid, uid=uid, gid=gid, raw_ns=raw(), info=hello.get("info"))
        self.CH.send(connection, dict(op="WELCOME", session=self.session), deadline=time.monotonic() + 10)

    def on_edge(self, message, fd):
        op = message["op"]
        if op == "CLOSE":
            self.edge_closed = True
            return
        f = self.frame(int(message["frame_id"]))
        if op == "PREDICTION":
            self.stamp(f, "service", "prediction_received")
            if not f.get("decision"):
                os.close(fd)
                self.stamp(f, "service", "prediction_for_unknown_decision")
                self.unscored_to_edge(f, "UNKNOWN_DECISION_FRAME")
                return
            for name, value in message["events"].items():
                self.stamp(f, "edge", name, value)
            f["prediction_received_raw_ns"] = raw()
            f["prediction_before_gt"] = not f.get("target_admitted")
            if f.get("outcome") is not None or f.get("no_gt") or f.get("cancel_reason"):
                os.close(fd)
                self.unscored_to_edge(f, f.get("no_gt") or f.get("cancel_reason") or f["outcome"])
                return
            self.enqueue("score", 0, "PREDICTION", dict(identity=message["identity"], records=message["records"],
                         size=message["size"], frame_context=message["frame_context"]), f["frame_id"], fd)
            f["prediction_queued_to_scorer"] = True
        elif op == "EMITTED":
            self.stamp(f, "edge", "feedback_emitted", message["raw_ns"], packet_sha256=message["packet_sha256"],
                       kind=message["kind"], reason=message["reason"])
            f["emitted"] = dict(message)
            self.maybe_archive(f)
        elif op == "EDGE_TERMINAL":
            self.stamp(f, "edge", "edge_terminal", message["raw_ns"], outcome=message["outcome"], stage=message["stage"])
            if not f.get("prediction_received_raw_ns"):
                self.abandon(f, "EDGE_TERMINAL_" + message["outcome"])

    def unscored_to_edge(self, f, reason):
        if f.get("edge_unscored_sent"):
            return
        f["edge_unscored_sent"] = True
        self.stamp(f, "service", "unscored_sent", reason=reason)
        self.send_edge(dict(op="UNSCORED", frame_id=f["frame_id"], reason=str(reason)))

    def abandon(self, f, reason):
        if f.get("outcome") is not None or f.get("result") is not None:
            return
        f["outcome"] = "UNSCORED_" + reason
        self.stamp(f, "service", "abandoned", reason=reason)
        if not f.get("committed") and not f.get("confirm_queued"):
            self.cancel(f, "ABANDONED")           # uncommitted transaction never advances history
        if f.get("identity") is not None:
            ident = json.loads(f["identity"])
            self.enqueue("score", 1, "ABANDON", dict(capture=ident["capture_identity"], action=ident["action_identity"],
                         frame_context=ident["frame_context"], reason=reason[:60].upper()), f["frame_id"])
            f["scorer_release_queued"] = True
        else:
            self.release_waiting_by_plan(f, reason)
        if f.get("prediction_received_raw_ns"):
            self.unscored_to_edge(f, reason)
        self.maybe_archive(f)

    def release_waiting_by_plan(self, f, reason):
        """No target identity yet: release a prediction already handed to the scorer by its plan triple."""
        if f.get("scorer_release_queued") or not f.get("prediction_queued_to_scorer") or not f.get("plan"):
            return
        plan = json.loads(f["plan"])
        self.enqueue("score", 1, "ABANDON", dict(capture=plan["capture"], action=plan["action"],
                     frame_context=plan["frame_context"], reason=str(reason)[:60].upper().replace("-", "_")), f["frame_id"])
        f["scorer_release_queued"] = True

    # ------------------------------------------------------------------ worker replies
    def dispatch(self):
        for name, client in (("prep", self.prep), ("score", self.score)):
            if client.pending is None and self.queues[name]:
                priority, _, op, args, frame_id, fd = heapq.heappop(self.queues[name])
                try:
                    client.begin(op, args, fd=fd)
                finally:
                    if fd is not None:
                        os.close(fd)
                self.current[name] = (op, frame_id, raw())

    def on_prep(self, reply):
        op, frame_id, began = self.current["prep"]
        self.current["prep"] = None
        f = self.frames[frame_id]
        f.setdefault("requests", []).append(dict(worker="prep", op=op, begin=began, end=raw(), timing=reply.timing,
                                                 status=reply.status, reason=reply.reason))
        if reply.status == "REFUSED":
            if op == "BIND" and str(reply.reason).startswith("DEFERRAL_REFUSED_"):
                self.bind_refused(f, reply.reason)
                self.prep_open = None
                self.pump_gate()
                return
            raise ValueError(f"unexpected prep refusal {op} {reply.reason}")
        result = self.SV.ok(reply, "prep " + op)
        if op == "FREEZE":
            f.update(job=result["job"], unbound=True)
            self.stamp(f, "prep", "source_decoded", result["decoded_raw_ns"])
            self.stamp(f, "prep", "unbound_ready", reply.timing["end_raw_ns"])
            if f.get("cancel_pending"):
                self.cancel(f, f.get("cancel_reason", "SEND_FAILED"))
            else:
                self.try_bind(f)
        elif op == "BIND":
            f.update(bound=True, identity=result["identity"], semantic=result["semantic_sha256"],
                     provenance=result.get("provenance"))
            self.stamp(f, "prep", "bound", reply.timing["end_raw_ns"])
            if f.get("cancel_pending"):
                self.cancel(f, f.get("cancel_reason", "SEND_FAILED"))
            else:
                self.try_confirm(f)
        elif op == "CONFIRM_SEND":
            f.update(committed=True, commit=result["commit"])
            self.stamp(f, "prep", "committed", reply.timing["end_raw_ns"])
            self.prep_open = None
            self.enqueue("prep", 0, "PUBLISH", dict(job=f["job"]), frame_id)
            self.pump_gate()
        elif op in ("CANCEL_BOUND", "CANCEL_UNBOUND"):
            if f.get("outcome") is None:
                f["outcome"] = "CANCELLED_" + f.get("cancel_reason", "UNKNOWN")
            f["history_after_cancel"] = result["history_sha256"]
            self.stamp(f, "prep", "cancelled", reply.timing["end_raw_ns"])
            self.prep_open = None
            self.pump_gate()
        elif op == "VOID":
            self.stamp(f, "prep", "voided", reply.timing["end_raw_ns"])
            if f.get("outcome") is None:            # exactly one registered outcome per decision
                f["outcome"] = "NO_GT_" + f.get("no_gt", "VOID")
            if f.get("prediction_received_raw_ns"):
                self.unscored_to_edge(f, f["outcome"])
        elif op == "PUBLISH":
            self.stamp(f, "prep", "published", reply.timing["end_raw_ns"])
            if f.get("outcome") is not None:          # abandoned before publication
                os.close(reply.fd)
                f["prep_unscored_reason"] = f["outcome"]
                self.maybe_archive(f)
                return
            self.enqueue("score", 1, "TARGET", dict(identity=result["identity"], semantic_sha256=result["semantic_sha256"],
                                                    size=result["size"], provenance=f.get("provenance")), frame_id, reply.fd)
        elif op == "RESULT_DELIVERED":
            f["prep_delivered"] = True
            self.maybe_prep_evidence(f)
        elif op in ("EVIDENCE", "UNSCORED_TERMINAL"):
            f["prep_files"] = self.CH.unpack_files(result["index"], reply.fd)
            os.close(reply.fd)
            self.archive_pending.append((frame_id, "producer"))
        elif op in ("EVIDENCE_DURABLE", "UNSCORED_DURABLE"):
            f["prep_retired"] = True
            f["prep_outcome"] = result["outcome"]
            self.stamp(f, "prep", "retired")

    def on_score(self, reply):
        op, frame_id, began = self.current["score"]
        self.current["score"] = None
        f = self.frames[frame_id]
        f.setdefault("requests", []).append(dict(worker="score", op=op, begin=began, end=raw(), timing=reply.timing,
                                                 status=reply.status, reason=reply.reason))
        if reply.status == "REFUSED":
            if op == "TARGET":
                f["target_refused"] = reply.reason
                self.refusals.append((frame_id, "TARGET_" + reply.reason))
                f["prep_unscored_reason"] = "SCORER_" + reply.reason
                if f.get("outcome") is None:
                    f["outcome"] = "UNSCORED_SCORER_" + reply.reason
                    self.stamp(f, "service", "abandoned", reason="SCORER_" + reply.reason)
                    if f.get("identity") is not None:
                        ident = json.loads(f["identity"])
                        self.enqueue("score", 1, "ABANDON", dict(capture=ident["capture_identity"],
                            action=ident["action_identity"], frame_context=ident["frame_context"],
                            reason="SCORER_CAPACITY"), frame_id)
                    if f.get("prediction_received_raw_ns"):
                        self.unscored_to_edge(f, f["outcome"])
                self.maybe_archive(f)
                return
            if op == "PREDICTION" and reply.reason == "WAITING_CAPACITY":
                # Expected bounded-capacity refusal: nothing was stored. Explicit no-score outcome once;
                # the frame's target (if it is ever admitted) is released through ABANDON below.
                self.refusals.append((frame_id, "SCORER_WAITING_CAPACITY"))
                self.stamp(f, "service", "score_refused", reason=reply.reason)
                if f.get("outcome") is None:
                    f["outcome"] = "UNSCORED_SCORER_WAITING_CAPACITY"
                    f["prep_unscored_reason"] = "SCORER_WAITING_CAPACITY"
                    if not f.get("committed") and not f.get("confirm_queued"):
                        self.cancel(f, "SCORER_WAITING_CAPACITY")   # uncommitted: never advances history
                self.unscored_to_edge(f, f["outcome"])
                if f.get("target_admitted") and f.get("score_abandon_status") is None and f.get("identity"):
                    self.abandon_target(f, "SCORER_WAITING_CAPACITY")
                self.maybe_archive(f)
                return
            if op in ("PREDICTION", "ABANDON") and reply.reason in ("LATE_AFTER_TERMINAL", "ALREADY_TERMINAL",
                                                                     "ALREADY_SCORED_OR_TERMINAL", "SLOT_ALREADY_TERMINAL"):
                self.stamp(f, "service", "score_refused", reason=reply.reason)
                if op == "PREDICTION":
                    self.unscored_to_edge(f, f.get("outcome") or reply.reason)
                return
            raise ValueError(f"unexpected score refusal {op} {reply.reason}")
        result = self.SV.ok(reply, "score " + op)
        if op == "TARGET":
            f["target_admitted"] = True
            self.stamp(f, "score", "target_admitted", reply.timing["end_raw_ns"])
            if result["joined_result"] is not None:
                f["joined_on_target"] = True
                self.deliver(f, result["joined_result"])
            elif f.get("outcome") is not None:
                # abandoned while publication was in flight: release the cached target as unscored
                ident = json.loads(f["identity"])
                self.enqueue("score", 1, "ABANDON", dict(capture=ident["capture_identity"], action=ident["action_identity"],
                             frame_context=ident["frame_context"], reason="ABANDONED_BEFORE_ADMISSION"), frame_id)
        elif op == "PREDICTION":
            self.stamp(f, "score", "prediction_admitted", reply.timing["end_raw_ns"])
            if result["status"] == "SCORED":
                self.deliver(f, result["result"])
            else:
                f["waiting"] = True
        elif op == "ABANDON":
            f["score_abandon_status"] = result["status"]
            if result["status"] == "WAITING_DROPPED":
                f["waiting"] = False
            self.maybe_archive(f)
        elif op == "RESULT_DELIVERED":
            f["score_delivered"] = True
            self.maybe_archive(f)
        elif op == "ARCHIVE_BUILD":
            f["score_files"] = self.CH.unpack_files(result["index"], reply.fd)
            os.close(reply.fd)
            self.archive_pending.append((frame_id, "scorer"))
        elif op == "ARCHIVE_DURABLE":
            f.update(score_retired=True, scorer_proof=result["proof"])
            self.stamp(f, "score", "retired")
            self.maybe_prep_evidence(f)

    def bind_refused(self, f, reason):
        """BIND refusal (explicit, registered): exactly one NO_GT outcome, no history advance (the prep
        engine already cancelled the unbound transaction), any waiting prediction released from the
        scorer, and the edge notified if its prediction is here. Covers both arrival orders."""
        if f.get("bind_refused"):
            raise ValueError("duplicate BIND refusal")
        f["bind_refused"] = reason
        f["no_gt"] = reason
        self.refusals.append((f["frame_id"], reason))
        self.stamp(f, "prep", "bind_refused", reason=reason)
        if f.get("outcome") is None:
            f["outcome"] = "NO_GT_" + reason
        self.release_waiting_by_plan(f, "BIND_REFUSED")
        if f.get("prediction_received_raw_ns"):
            self.unscored_to_edge(f, f["outcome"])

    def abandon_target(self, f, reason):
        ident = json.loads(f["identity"])
        self.enqueue("score", 1, "ABANDON", dict(capture=ident["capture_identity"], action=ident["action_identity"],
                     frame_context=ident["frame_context"], reason=reason[:60].upper()), f["frame_id"])

    def deliver(self, f, result):
        if f.get("result") is not None:
            raise ValueError("exactly one result per decision")
        f.update(result=result, waiting=False, outcome="SCORED")
        self.stamp(f, "service", "result_sent")
        self.send_edge(dict(op="RESULT", frame_id=f["frame_id"], measurement=result["measurement"],
                            q_perc=result["q_perc"], quality_defined=result["quality_defined"],
                            identity_sha256=result["identity_sha256"], ready_semantic_sha256=result["ready_semantic_sha256"]))
        self.memory.append(dict(frame_id=f["frame_id"], t=raw(), **{n: rss_kib(p) for n, p in self.pids().items()}))
        self.enqueue("score", 2, "RESULT_DELIVERED", dict(identity=f["identity"]), f["frame_id"])
        self.enqueue("prep", 2, "RESULT_DELIVERED", dict(job=f["job"]), f["frame_id"])

    # ------------------------------------------------------------------ evidence (after the real UE terminal)
    def terminal_known(self, f):
        return f.get("ue_terminal") is not None or self.ue_closed

    def maybe_archive(self, f):
        if f.get("archive_started") or not self.terminal_known(f):
            return
        if f.get("outcome") == "SCORED":
            if not (f.get("score_delivered") and (f.get("emitted") or self.edge_closed)):
                return
            f["archive_started"] = True
            self.enqueue("score", 3, "ARCHIVE_BUILD", dict(identity=f["identity"]), f["frame_id"])
        elif str(f.get("outcome", "")).startswith("UNSCORED"):
            if f.get("unbound") and not (f.get("committed") or f.get("bind_refused") or "history_after_cancel" in f):
                return                                  # producer transaction still unresolved (v2 guard)
            published = any(e["name"] == "published" for e in f["events"])
            if f.get("committed") and not published:
                return                                  # publication still in flight
            if f.get("target_admitted") and f.get("score_abandon_status") is None:
                return
            if f.get("prediction_received_raw_ns") and not (f.get("emitted") or self.edge_closed):
                return
            f["archive_started"] = True
            if f.get("target_admitted"):
                self.enqueue("score", 3, "ARCHIVE_BUILD", dict(identity=f["identity"]), f["frame_id"])
            else:
                f["score_retired"] = True
                self.maybe_prep_evidence(f)

    def maybe_prep_evidence(self, f):
        if f.get("prep_evidence_queued") or not f.get("score_retired"):
            return
        if f.get("outcome") == "SCORED":
            if not f.get("prep_delivered"):
                return
            f["prep_evidence_queued"] = True
            self.enqueue("prep", 3, "EVIDENCE", dict(job=f["job"]), f["frame_id"])
        elif f.get("committed"):
            f["prep_evidence_queued"] = True
            reason = str(f.get("prep_unscored_reason") or f["outcome"]).upper().replace("-", "_")[:63]
            self.enqueue("prep", 3, "UNSCORED_TERMINAL", dict(job=f["job"], reason=reason), f["frame_id"])
        else:
            f["prep_retired"] = True

    def clock_document(self, f):
        return json.dumps(dict(schema=CLOCK_SCHEMA, frame_id=f["frame_id"], clock="CLOCK_MONOTONIC_RAW",
            outcome=f.get("outcome"), ue_terminal=f.get("ue_terminal"), identity_sha256=None if f.get("identity") is None
            else hashlib.sha256(f["identity"].encode()).hexdigest(), events=f["events"],
            freeze_stats=f.get("freeze_stats"), commit=f.get("commit"), emitted=f.get("emitted"),
            prediction_before_gt=f.get("prediction_before_gt"), joined_on_target=f.get("joined_on_target", False)),
            sort_keys=True, separators=(",", ":")).encode()

    def pump_archive(self):
        while self.archive_pending and self.archive.outstanding_jobs < 4:
            frame_id, side = self.archive_pending.pop(0)
            f = self.frames[frame_id]
            capture = json.loads(f["identity"])["capture_identity"]
            if side == "scorer":
                files = dict(f.pop("score_files"))
                files["frame_clock.json"] = self.clock_document(f)
                identity = dict(capture, evidence_role="scorer")
            else:
                files, identity = f.pop("prep_files"), capture
            prefix = f"frame_{frame_id}/{side}"
            job = self.archive.submit(identity, {str(self.archive_root / prefix / n): r for n, r in files.items()})
            self.archive_jobs[job] = (frame_id, side, prefix, files)
        # The retained ArchiveClient.submit() polls internally and keeps those completions only in its
        # ``completions`` list; consume that list by index so no completion is ever lost (v2 fix).
        self.archive.poll()
        fresh = self.archive.completions[self.archive_seen:]
        self.archive_seen = len(self.archive.completions)
        for completion in fresh:
            frame_id, side, prefix, files = self.archive_jobs.pop(completion["job_id"])
            if not completion["durable"]:
                raise ValueError("durable archive completion required")
            for name, payload in files.items():
                if (self.archive_root / prefix / name).read_bytes() != payload:
                    raise ValueError("durable readback differs")
            f = self.frames[frame_id]
            if side == "scorer":
                scorer_files = {k: v for k, v in files.items() if k != "frame_clock.json"}
                receipt = dict(completion, written=[w for w in completion["written"] if not w["relative"].endswith("/frame_clock.json")])
                f["clock_sha256"] = hashlib.sha256(files["frame_clock.json"]).hexdigest()
                self.enqueue("score", 3, "ARCHIVE_DURABLE", dict(identity=f["identity"], receipt=receipt,
                                                                 relative_prefix=prefix), frame_id)
            else:
                op = "EVIDENCE_DURABLE" if f.get("outcome") == "SCORED" else "UNSCORED_DURABLE"
                args = dict(job=f["job"], receipt=completion, relative_prefix=prefix)
                if op == "EVIDENCE_DURABLE":
                    args["scorer_proof"] = f["scorer_proof"]
                self.enqueue("prep", 3, op, args, frame_id)

    def check_horizons(self):
        now = raw()
        horizon = int(self.limits["horizon_ms"] * 1e6)
        for f in self.frames.values():
            if (f.get("decision") and f.get("action_open_raw_ns") and f.get("outcome") is None
                    and now - f["action_open_raw_ns"] > horizon):
                if f.get("sent") is False:
                    continue
                self.abandon(f, "NO_RESULT_BY_HORIZON")

    def pids(self):
        values = dict(service=os.getpid(), prep=self.prep.process.pid, score=self.score.process.pid,
                      archive=self.archive.process.pid)
        return values

    def finished(self):
        if not self.ue_closed:
            return False
        for f in self.frames.values():
            if not f.get("decision"):
                continue
            outcome = f.get("outcome")
            if outcome is None:
                return False
            if outcome == "SCORED" or (outcome.startswith("UNSCORED") and f.get("committed")):
                if not f.get("prep_retired"):
                    return False
        return (not self.archive_jobs and not self.archive_pending and all(c is None for c in self.current.values())
                and not any(self.queues.values()) and not self.gate_ops)

    # ------------------------------------------------------------------ main loop
    def run(self):
        drain_deadline = None
        while not self.finished():
            if self.ue_closed and drain_deadline is None:
                drain_deadline = time.monotonic() + self.limits["drain_s"]
            if drain_deadline is not None and time.monotonic() > drain_deadline:
                self.errors.append("drain deadline expired with unfinished frames")
                break
            readers = [self.ue, self.listener]
            if self.edge is not None and not self.edge_closed:
                readers.append(self.edge)
            if self.prep.pending is not None:
                readers.append(self.prep.sock)
            if self.score.pending is not None:
                readers.append(self.score.sock)
            if self.archive_jobs:
                readers.append(self.archive.socket)
            for sock in select.select(readers, [], [], 0.02)[0]:
                if sock is self.ue and not self.ue_closed:
                    self.on_ue(*self.CH.receive(sock, deadline=time.monotonic() + 10)); self.sample("ue")
                elif sock is self.listener:
                    self.accept_edge()
                elif sock is self.edge:
                    try:
                        message, fd = self.CH.receive(sock, deadline=time.monotonic() + 10)
                    except self.CH.ChannelError:
                        self.edge_closed = True
                        continue
                    self.on_edge(message, fd); self.sample("edge")
                elif sock is self.prep.sock and self.prep.pending is not None:
                    self.on_prep(self.prep.finish()); self.sample("prep")
                elif sock is self.score.sock and self.score.pending is not None:
                    self.on_score(self.score.finish()); self.sample("score")
            self.check_horizons()
            for f in self.frames.values():
                self.maybe_archive(f)
            self.pump_archive()
            self.dispatch()

    def close(self):
        snapshot = {}
        try:
            snapshot = dict(prep=self.SV.ok(self.prep.call("SNAPSHOT"), "prep SNAPSHOT"),
                            score=self.SV.ok(self.score.call("SNAPSHOT"), "score SNAPSHOT"))
        except BaseException as error:
            self.errors.append(f"snapshot: {error}")
        abandon = bool(self.errors)
        closed = {}
        for name, client in (("prep", self.prep), ("score", self.score)):
            try:
                closed[name] = client.close(abandon=abandon).result
            except BaseException as error:
                self.errors.append(f"{name} close: {type(error).__name__}: {error}")
                client.abort()
        try:
            closed["archive"] = self.archive.close()["completed"]
        except BaseException as error:
            self.errors.append(f"archive close: {error}")
        for path in (self.socket_path, self.capability_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self.listener.close()
        frames = []
        for frame_id in self.order:
            f = self.frames[frame_id]
            frames.append({k: v for k, v in f.items() if k not in ("score_files", "prep_files", "plan")})
        with open(self.out / "FRAMES.jsonl", "x") as stream:
            for row in frames:
                stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        result = dict(schema=SCHEMA, session=self.session, pids=self.pids(), edge=getattr(self, "edge_hello", None),
            limits=self.limits, refusals=self.refusals, errors=self.errors, snapshot=snapshot, closed=closed,
            outcomes={o: sum(1 for f in self.frames.values() if f.get("outcome") == o)
                      for o in sorted({str(f.get("outcome")) for f in self.frames.values()})},
            decisions=sum(1 for f in self.frames.values() if f.get("decision")),
            samples=len(self.samples), started_raw_ns=self.started_raw_ns, closed_raw_ns=raw())
        for name, value in (("QUEUE_SAMPLES.json", self.samples), ("MEMORY_SAMPLES.json", self.memory),
                            ("SERVICE_RESULT.json", result)):
            with open(self.out / name, "x") as stream:
                json.dump(value, stream, indent=1, sort_keys=True, default=str)
        return result


def main(fd, parent):
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(1, signal.SIGTERM, 0, 0, 0)
    if os.getppid() != parent:
        raise SystemExit("parent changed before start")
    from rl_agent.splitfusion_exact_gt_service_proto_v2 import channel as CH
    sock = socket.socket(fileno=fd)
    init, _ = CH.receive(sock, deadline=time.monotonic() + 60)
    config = init["config"]
    verify_sources(config["sources"])
    service = None
    try:
        service = Service(sock, config)
        if config["limits"].get("service_cpu") is not None:
            os.sched_setaffinity(0, {config["limits"]["service_cpu"]})
        CH.send(sock, dict(op="READY", pid=os.getpid(), session=service.session,
                           socket=str(service.socket_path), capability=str(service.capability_path),
                           affinity=sorted(os.sched_getaffinity(0))), deadline=time.monotonic() + 10)
        service.run()
    except BaseException as error:
        if service is not None:
            service.errors.append(f"{type(error).__name__}: {error}"[:2000])
        else:
            CH.send(sock, dict(op="FAILED", error=f"{type(error).__name__}: {error}"[:2000]), deadline=time.monotonic() + 10)
            raise
    result = service.close()
    CH.send(sock, dict(op="CLOSED", result=dict(errors=result["errors"], outcomes=result["outcomes"],
                       decisions=result["decisions"], refusals=result["refusals"])), deadline=time.monotonic() + 30)
    sock.close()
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--parent", type=int, required=True)
    a = parser.parse_args()
    raise SystemExit(main(a.fd, a.parent))
