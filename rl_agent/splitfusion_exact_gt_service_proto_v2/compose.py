"""Executable bounded composition of the early-GT architecture (offline, CPU only).

Processes: UE emulator (unpinned, like the live UE), edge emulator (CPU 7), this coordinator
(main thread CPU 6), prep worker (CPU 4) and score worker (CPU 5) as in the live numeric-GT
registration, and the unchanged ArchiveClient (unpinned). CPUs 4-7 are the existing edge
partition; nothing in the retained partition or host scheduling policy changes.

The coordinator is an event loop. UE freeze/plan/send messages and edge predictions arrive
on their own recorded schedule; predictions are NEVER held back until GT is ready. Per worker
there is one outstanding request and a bounded priority queue:
  prep : FREEZE / BIND / CONFIRM_SEND / PUBLISH (0)  <  notifications (2)  <  evidence (3)
  score: PREDICTION (0)  <  TARGET (1)  <  notifications (2)  <  evidence (3)
Feedback (RESULT to the edge) is sent as soon as the score worker replies; map publication is
not involved and archival is deferred, bounded and durable (real fsync).

Usage (from abiodun/):
  python -m rl_agent.splitfusion_exact_gt_service_proto_v2.compose --name <dir> --variant recorded|spacing200
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import time

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_gt_handoff_v1.archive import ArchiveClient
from rl_agent.splitfusion_numeric_gt_runtime_v1 import producer as PR

from . import parity_common as P, channel as CH, service as SV, timeline as TL
from .lifecycle import expected_history

CPU_PREP, CPU_SCORE, CPU_COORD, CPU_EDGE = 4, 5, 6, 7
ENV = dict(SV.ENV)


def raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def rss_kib(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        return None


def spawn(module, out, name, extra=()):
    parent, child = CH.pair()
    env = dict(os.environ, **ENV)
    env.pop("PYTHONPATH", None)
    log = open(out / f"{name}.log", "xb")
    process = subprocess.Popen([sys.executable, "-B", "-u", "-m", f"rl_agent.splitfusion_exact_gt_service_proto_v2.{module}",
                                "--fd", str(child.fileno()), "--parent", str(os.getpid()), *extra],
                               pass_fds=(child.fileno(),), cwd=P.ROOT, env=env, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=log)
    child.close()
    return process, parent, log


class Composer:
    def __init__(self, out, events, *, prep_jobs=4, scorer_jobs=4, max_waiting=2, freeze_queue_max=1):
        self.out, self.events = out, events
        self.decisions = [e for e in events if e["kind"] == "POLICY_DECISION"]
        self.corpus = P.Corpus(49)
        self.case_by_frame = {c.frame_id: c for c in self.corpus.cases}
        self.freeze_queue_max = freeze_queue_max
        self.counter = itertools.count()
        engine = self.corpus.engine_config(offline=False)
        engine.pop("live_process_contract")
        engine.update(max_retained_jobs=prep_jobs, max_retained_bytes=prep_jobs * PR.RESERVATION)
        self.limits = dict(prep_jobs=prep_jobs, scorer_jobs=scorer_jobs, max_waiting=max_waiting,
                           freeze_queue=freeze_queue_max, archive_jobs=4)
        self.prep = SV.WorkerClient("prep", dict(engine=engine), log_path=out / "prep.log", cpu=CPU_PREP)
        self.score = SV.WorkerClient("score", dict(run_binding=self.corpus.binding,
            producer_binding=self.corpus.startup["binding_sha256"], max_jobs=scorer_jobs,
            max_bytes=scorer_jobs * 8 * 1024 * 1024, reservation_bytes=8 * 1024 * 1024,
            max_waiting=max_waiting, max_frames=200), log_path=out / "score.log", cpu=CPU_SCORE)
        (out / "durable").mkdir()
        self.archive_root = (out / "durable").resolve()
        self.archive = ArchiveClient(roots=[str(self.archive_root)], max_jobs=4, max_bytes=256 * 1024 * 1024,
                                     max_frames=400, binding=dict(scope="exact_gt_compose"))
        self.ue, self.ue_sock, self.ue_log = spawn("ue_emulator", out, "ue")
        self.edge, self.edge_sock, self.edge_log = spawn("edge_emulator", out, "edge", ("--cpu", str(CPU_EDGE)))
        self.original_affinity = sorted(os.sched_getaffinity(0))
        self.frames = {}
        self.queues = {"prep": [], "score": []}
        self.current = {"prep": None, "score": None}
        self.freeze_waiting = []
        self.prep_open = None
        self.archive_jobs, self.archive_pending = {}, []
        self.samples, self.memory, self.refusals = [], [], []
        self.done = {"ue": False, "edge": False}

    # ------------------------------------------------------------------ queues
    def enqueue(self, worker, priority, op, args, k, fd=None):
        heapq.heappush(self.queues[worker], (priority, next(self.counter), op, args, k, fd))

    def dispatch(self):
        for name in ("prep", "score"):
            client = self.prep if name == "prep" else self.score
            if client.pending is None and self.queues[name]:
                priority, _, op, args, k, fd = heapq.heappop(self.queues[name])
                try:
                    client.begin(op, args, fd=fd)
                finally:
                    if fd is not None:
                        os.close(fd)
                self.current[name] = (op, k, raw(), priority)

    def sample(self, label):
        committed = sum(1 for f in self.frames.values() if f.get("committed") and not f.get("prep_retired"))
        slots = sum(1 for f in self.frames.values() if f.get("gt_ready_raw_ns") and not f.get("score_retired"))
        self.samples.append(dict(t=raw(), label=label, prep_queue=len(self.queues["prep"]), score_queue=len(self.queues["score"]),
            freeze_queue=len(self.freeze_waiting), prep_open=self.prep_open is not None,
            waiting_predictions=sum(1 for f in self.frames.values() if f.get("waiting")),
            prep_retained=committed + (1 if self.prep_open is not None else 0), scorer_slots=slots,
            archive_outstanding=self.archive.outstanding_jobs + len(self.archive_pending)))

    def frame(self, k):
        return self.frames.setdefault(k, dict(k=k))

    # ------------------------------------------------------------------ UE / edge messages
    def on_ue(self, message, fd):
        k, op, now = message.get("k"), message["op"], raw()
        if op == "DONE":
            self.done["ue"] = True
            return
        f = self.frame(k)
        if op == "OPEN":
            f.update(frame_id=message["frame_id"], action_open_raw_ns=message["action_open_raw_ns"], open_received_raw_ns=now)
        elif op == "PLAN":
            f.update(plan=message["plan"], frame_context=message["frame_context"], plan_received_raw_ns=now)
            self.try_bind(k)
        elif op == "FREEZE":
            f["freeze_received_raw_ns"] = now
            item = (k, dict(descriptor=message["descriptor"]), fd)
            if self.prep_open is not None:
                if len(self.freeze_waiting) >= self.freeze_queue_max:
                    os.close(fd)
                    f["no_gt"] = "FREEZE_QUEUE_FULL"
                    self.refusals.append((k, "FREEZE_QUEUE_FULL"))
                    return
                self.freeze_waiting.append(item)
            else:
                self.start_freeze(item)
        elif op == "SEND":
            f.update(send_raw_ns=message["send_raw_ns"], send_received_raw_ns=now, sent=True)
            CH.send(self.edge_sock, dict(op="SEND", k=k, send_raw_ns=message["send_raw_ns"]), deadline=time.monotonic() + 10)
            self.try_confirm(k)

    def start_freeze(self, item):
        k, args, fd = item
        self.prep_open = k
        self.frame(k)["freeze_dispatched_raw_ns"] = raw()
        self.enqueue("prep", 0, "FREEZE", args, k, fd)

    def on_edge(self, message, fd):
        if message["op"] == "DONE":
            self.done["edge"] = True
            return
        k = message["k"]
        f = self.frame(k)
        f["prediction_received_raw_ns"] = raw()
        f["prediction_before_gt"] = "gt_ready_raw_ns" not in f
        self.enqueue("score", 0, "PREDICTION", dict(identity=message["identity"], records=message["records"],
                     size=message["size"], frame_context=message["frame_context"]), k, fd)

    def try_bind(self, k):
        f = self.frames[k]
        if f.get("unbound") and f.get("plan") and not f.get("bind_queued"):
            f["bind_queued"] = True
            self.enqueue("prep", 0, "BIND", dict(job=f["job"], plan=f["plan"], frame_context=f["frame_context"]), k)

    def try_confirm(self, k):
        f = self.frames[k]
        if f.get("bound") and f.get("sent") and not f.get("confirm_queued"):
            f["confirm_queued"] = True
            self.enqueue("prep", 0, "CONFIRM_SEND", dict(job=f["job"], plan=f["plan"]), k)

    # ------------------------------------------------------------------ worker replies
    def on_prep(self, reply):
        op, k, began, _ = self.current["prep"]
        self.current["prep"] = None
        f = self.frames[k]
        f.setdefault("prep_requests", []).append(dict(op=op, coordinator_begin_raw_ns=began, received_raw_ns=raw(),
                                                      timing=reply.timing, status=reply.status))
        result = SV.ok(reply, "prep " + op)
        if op == "FREEZE":
            f.update(job=result["job"], unbound=True, decoded_raw_ns=result["decoded_raw_ns"],
                     freeze_prepared_raw_ns=reply.timing["end_raw_ns"])
            self.try_bind(k)
        elif op == "BIND":
            f.update(bound=True, identity=result["identity"], semantic=result["semantic_sha256"],
                     bound_raw_ns=reply.timing["end_raw_ns"])
            self.try_confirm(k)
        elif op == "CONFIRM_SEND":
            f.update(committed=True, commit=result["commit"], committed_raw_ns=reply.timing["end_raw_ns"])
            self.prep_open = None
            self.enqueue("prep", 0, "PUBLISH", dict(job=f["job"]), k)
            if self.freeze_waiting:
                self.start_freeze(self.freeze_waiting.pop(0))
        elif op == "PUBLISH":
            f["published_raw_ns"] = reply.timing["end_raw_ns"]
            self.enqueue("score", 1, "TARGET", dict(identity=result["identity"], semantic_sha256=result["semantic_sha256"],
                                                    size=result["size"]), k, reply.fd)
        elif op == "RESULT_DELIVERED":
            f["prep_delivered"] = True
            self.maybe_prep_evidence(k)
        elif op == "EVIDENCE":
            f["prep_files"] = CH.unpack_files(result["index"], reply.fd)
            os.close(reply.fd)
            self.archive_pending.append((k, "producer"))
        elif op == "EVIDENCE_DURABLE":
            f.update(prep_retired=True, prep_outcome=result["outcome"], prep_retired_raw_ns=raw())

    def on_score(self, reply):
        op, k, began, _ = self.current["score"]
        self.current["score"] = None
        f = self.frames[k]
        f.setdefault("score_requests", []).append(dict(op=op, coordinator_begin_raw_ns=began, received_raw_ns=raw(),
                                                       timing=reply.timing, status=reply.status))
        if reply.status == "REFUSED":
            self.refusals.append((k, op + ":" + reply.reason))
            raise AssertionError(f"unexpected score refusal {op} {reply.reason} frame {k}")
        result = SV.ok(reply, "score " + op)
        if op == "TARGET":
            f["gt_ready_raw_ns"] = reply.timing["end_raw_ns"]
            if result["joined_result"] is not None:
                f["joined_on_target"] = True
                self.deliver(k, result["joined_result"])
        elif op == "PREDICTION":
            f["prediction_admitted_raw_ns"] = reply.timing["end_raw_ns"]
            if result["status"] == "SCORED":
                self.deliver(k, result["result"])
            else:
                f["waiting"] = True
        elif op == "RESULT_DELIVERED":
            f["score_delivered"] = True
            self.enqueue("score", 3, "ARCHIVE_BUILD", dict(identity=f["identity"]), k)
        elif op == "ARCHIVE_BUILD":
            f["score_files"] = CH.unpack_files(result["index"], reply.fd)
            os.close(reply.fd)
            self.archive_pending.append((k, "scorer"))
        elif op == "ARCHIVE_DURABLE":
            f.update(score_retired=True, scorer_proof=result["proof"])
            self.maybe_prep_evidence(k)

    def deliver(self, k, result):
        f = self.frames[k]
        C.require("result" not in f, "exactly one feedback per decision")
        f.update(result=result, waiting=False, result_sent_raw_ns=raw())
        CH.send(self.edge_sock, dict(op="RESULT", k=k, result=dict(q_perc=result["q_perc"],
                quality_defined=result["quality_defined"])), deadline=time.monotonic() + 10)
        self.memory.append(dict(k=k, t=raw(), **{n: rss_kib(p) for n, p in self.pids().items()}))
        self.enqueue("score", 2, "RESULT_DELIVERED", dict(identity=f["identity"]), k)
        self.enqueue("prep", 2, "RESULT_DELIVERED", dict(job=f["job"]), k)

    def maybe_prep_evidence(self, k):
        f = self.frames[k]
        if f.get("prep_delivered") and f.get("score_retired") and not f.get("prep_evidence_queued"):
            f["prep_evidence_queued"] = True
            self.enqueue("prep", 3, "EVIDENCE", dict(job=f["job"]), k)

    # ------------------------------------------------------------------ archival (bounded, deferred)
    def pump_archive(self):
        while self.archive_pending and self.archive.outstanding_jobs < 4:
            k, side = self.archive_pending.pop(0)
            f = self.frames[k]
            capture = self.case_by_frame[f["frame_id"]].kwargs["identity"]
            files = f.pop("score_files") if side == "scorer" else f.pop("prep_files")
            prefix = f"frame_{f['frame_id']}/{side}"
            identity = dict(capture, evidence_role="scorer") if side == "scorer" else capture
            job = self.archive.submit(identity, {str(self.archive_root / prefix / n): r for n, r in files.items()})
            self.archive_jobs[job] = (k, side, prefix, files)
        for completion in self.archive.poll():
            k, side, prefix, files = self.archive_jobs.pop(completion["job_id"])
            C.require(completion["durable"], "durable archive completion required")
            for name, payload in files.items():
                C.require((self.archive_root / prefix / name).read_bytes() == payload, "durable readback differs")
            f = self.frames[k]
            if side == "scorer":
                self.enqueue("score", 3, "ARCHIVE_DURABLE", dict(identity=f["identity"], receipt=completion,
                                                                 relative_prefix=prefix), k)
            else:
                self.enqueue("prep", 3, "EVIDENCE_DURABLE", dict(job=f["job"], receipt=completion, relative_prefix=prefix,
                                                                 scorer_proof=f["scorer_proof"]), k)

    def pids(self):
        return dict(prep=self.prep.process.pid, score=self.score.process.pid, ue=self.ue.pid, edge=self.edge.pid,
                    coordinator=os.getpid(), archive=self.archive.process.pid)

    # ------------------------------------------------------------------ run
    def run(self, record_dir):
        session = self.prep.ready["session_id"]
        events = []
        for index, e in enumerate(self.events):
            events.append(dict(e, decision_index=[d["frame_id"] for d in self.decisions].index(e["frame_id"])
                               if e["kind"] == "POLICY_DECISION" else None))
        CH.send(self.ue_sock, dict(nonce="ue", role="ue", config=dict(events=events, session_id=session,
                record_path=str(record_dir / "UE_RECORDS.json"))), deadline=time.monotonic() + 30)
        CH.send(self.edge_sock, dict(nonce="edge", role="edge", config=dict(events=events,
                record_path=str(record_dir / "EDGE_RECORDS.json"))), deadline=time.monotonic() + 30)
        for sock in (self.ue_sock, self.edge_sock):
            ready, _ = CH.receive(sock, deadline=time.monotonic() + 300)
            C.require(ready["op"] == "READY", "emulator READY required")
        os.sched_setaffinity(0, {CPU_COORD})
        t0 = raw() + 2_000_000_000
        CH.send(self.ue_sock, dict(op="START", t0_raw_ns=t0), deadline=time.monotonic() + 10)
        deadline = time.monotonic() + max(e["open_ms"] for e in self.events) / 1e3 + 120
        n = len(self.decisions)
        try:
            while not (self.done["ue"] and self.done["edge"] and sum(1 for f in self.frames.values()
                       if f.get("prep_retired")) == n and not self.archive_jobs and not self.archive_pending
                       and all(c is None for c in self.current.values()) and not any(self.queues.values())):
                C.require(time.monotonic() < deadline, "composition did not complete in time")
                readers = [self.ue_sock, self.edge_sock]
                if self.prep.pending is not None:
                    readers.append(self.prep.sock)
                if self.score.pending is not None:
                    readers.append(self.score.sock)
                if self.archive_jobs:
                    readers.append(self.archive.socket)
                ready = select.select(readers, [], [], 0.05)[0]
                for sock in ready:
                    if sock is self.ue_sock and not self.done["ue"]:
                        self.on_ue(*CH.receive(sock, deadline=time.monotonic() + 10)); self.sample("ue")
                    elif sock is self.edge_sock and not self.done["edge"]:
                        self.on_edge(*CH.receive(sock, deadline=time.monotonic() + 10)); self.sample("edge")
                    elif sock is self.prep.sock and self.prep.pending is not None:
                        self.on_prep(self.prep.finish()); self.sample("prep")
                    elif sock is self.score.sock and self.score.pending is not None:
                        self.on_score(self.score.finish()); self.sample("score")
                self.pump_archive()
                self.dispatch()
            snapshot = dict(prep=SV.ok(self.prep.call("SNAPSHOT"), "prep SNAPSHOT"),
                            score=SV.ok(self.score.call("SNAPSHOT"), "score SNAPSHOT"))
        finally:
            os.sched_setaffinity(0, set(self.original_affinity))
        closed = dict(prep=SV.ok(self.prep.close(), "prep CLOSE"), score=SV.ok(self.score.close(), "score CLOSE"),
                      archive=self.archive.close()["completed"])
        for name, process in (("ue", self.ue), ("edge", self.edge)):
            process.wait(timeout=30)
            closed[name] = process.returncode
        for log in (self.ue_log, self.edge_log):
            log.close()
        self.ue_sock.close(); self.edge_sock.close()
        return snapshot, closed


def stats(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return dict(n=0)
    n = len(v)
    import math
    return dict(n=n, p50=round(v[(n - 1) // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2, 3),
                p95=round(v[max(0, math.ceil(.95 * n) - 1)], 3), max=round(v[-1], 3), min=round(v[0], 3))


def analyse(composer, ue, edge, truth, history_ok):
    ms = lambda a, b: None if a is None or b is None else (b - a) / 1e6
    ue_by_frame = {r["frame_id"]: r for r in ue["records"]}
    edge_by_k = {r["k"]: r for r in edge["records"]}
    rows = []
    for k, f in sorted(composer.frames.items()):
        u, e = ue_by_frame[f["frame_id"]], edge_by_k[k]
        ao = f["action_open_raw_ns"]
        fr = u["freeze"]
        row = dict(k=k, frame_id=f["frame_id"],
            # UE side (measured)
            action_open_lateness_ms=ms(u["scheduled_open_raw_ns"], ao), handoff_put_ms=u["handoff_put_ns"] / 1e6,
            freeze_queue_wait_ms=ms(fr["enqueued_raw_ns"], fr["start_raw_ns"]), freeze_ms=ms(fr["start_raw_ns"], fr["frozen_raw_ns"]),
            seal_ms=ms(fr["frozen_raw_ns"], fr["sealed_raw_ns"]), freeze_send_ms=ms(fr["sealed_raw_ns"], fr["sent_raw_ns"]),
            freeze_thread_cpu_ms=fr["thread_cpu_ms"], freeze_overlapped_input=u["freeze_overlapped_input"],
            input_contended_ms=u["input_ms"], input_baseline_ms=u["baseline_ms"], baseline_position=u["baseline_position"],
            input_paired_delta_ms=u["input_ms"] - u["baseline_ms"], modelled_send_delay_ms=u["slowdown_vs_baseline_median_ms"],
            # service side, ms from action-open (measured in the replay)
            freeze_received=ms(ao, f["freeze_received_raw_ns"]), source_decoded=ms(ao, f["decoded_raw_ns"]),
            unbound_ready=ms(ao, f["freeze_prepared_raw_ns"]), bound=ms(ao, f["bound_raw_ns"]), send=ms(ao, f["send_raw_ns"]),
            committed=ms(ao, f["committed_raw_ns"]), published=ms(ao, f["published_raw_ns"]), gt_ready=ms(ao, f["gt_ready_raw_ns"]),
            prediction_submitted=ms(ao, e["prediction_start_raw_ns"]), prediction_received=ms(ao, f["prediction_received_raw_ns"]),
            prediction_lateness_vs_schedule_ms=e["lateness_vs_schedule_ms"],
            result_sent=ms(ao, f["result_sent_raw_ns"]), result_received_at_edge=ms(ao, e["result_received_raw_ns"]),
            prediction_to_result_at_edge_ms=ms(e["prediction_start_raw_ns"], e["result_received_raw_ns"]),
            gt_minus_prediction_ms=ms(f["prediction_received_raw_ns"], f["gt_ready_raw_ns"]),
            prediction_before_gt=f["prediction_before_gt"], joined_on_target=f.get("joined_on_target", False),
            q_perc=f["result"]["q_perc"])
        t = truth[f["frame_id"]]
        row["exact_quality"] = f["result"]["q_perc"] == t["q_perc"] and C.canonical(f["result"]["measurement"]) == t["measurement"]
        rows.append(row)
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and not isinstance(rows[0][k], bool) and k not in ("k", "frame_id")]
    summary = {k: stats([r[k] for r in rows]) for k in keys}
    summary["counts"] = dict(decisions=len(rows), exact_quality=sum(r["exact_quality"] for r in rows),
        prediction_before_gt=sum(r["prediction_before_gt"] for r in rows), joined_on_target=sum(r["joined_on_target"] for r in rows),
        freeze_overlapped_input=sum(bool(r["freeze_overlapped_input"]) for r in rows), history_exact=history_ok,
        refusals=composer.refusals)
    s = composer.samples
    summary["queues_max"] = {k: max(x[k] for x in s) for k in s[0] if k not in ("t", "label")}
    summary["memory_kib_max"] = {k: max(m[k] for m in composer.memory if m.get(k) is not None)
                                 for k in composer.memory[0] if k not in ("k", "t")}
    holds = [r for r in ue["records"] if r["kind"] != "POLICY_DECISION"]
    summary["hold_input_ms"] = stats([r["input_ms"] for r in holds])
    return rows, summary


def run(name, variant, limit=None):
    P.environment_ok()
    out = P.EVIDENCE_ROOT / name
    out.mkdir(parents=True, exist_ok=False)
    host_before = dict(loadavg=os.getloadavg(), raw_ns=raw())
    corpus_ids = [c.frame_id for c in P.Corpus(49).cases]
    events, timeline_inputs = TL.recorded(corpus_ids)
    if variant == "spacing200":
        events = TL.compressed(events, 200.0)
    if limit:
        last = [e for e in events if e["kind"] == "POLICY_DECISION"][limit - 1]["open_ms"]
        events = [e for e in events if e["open_ms"] <= last]
    composer = Composer(out, events)
    status, rows, summary = "STARTED", [], {}
    try:
        truth = {}
        for case in composer.corpus.cases:
            pred = composer.corpus.predictions[case.frame_id]
            original = E.evaluate_prepared(identity=pred["identity"]["action_identity"], prepared=case.bundle.prepared,
                predicted_labels=pred["labels"], predictions=pred["records"], avo=case.avo_document["avo"])
            truth[case.frame_id] = dict(q_perc=pred["q_perc"], measurement=C.canonical(original.measurement))
        snapshot, closed = composer.run(out)
        expected = expected_history(composer.corpus, list(range(len(composer.decisions))))
        history_ok = snapshot["prep"]["history_sha256"] == expected
        ue = json.loads((out / "UE_RECORDS.json").read_bytes())
        edge = json.loads((out / "EDGE_RECORDS.json").read_bytes())
        rows, summary = analyse(composer, ue, edge, truth, history_ok)
        exact = summary["counts"]["exact_quality"] == len(rows) == len(composer.decisions) and history_ok
        status = "COMPOSITION_COMPLETE_EXACT" if exact else "COMPOSITION_COMPLETE_NOT_EXACT"
        summary.update(final_snapshot=dict(prep=snapshot["prep"]["engine"]["retained_jobs"],
                       scorer_slots=snapshot["score"]["retained_slots"], waiting=snapshot["score"]["waiting_predictions"]),
                       closed=closed)
    except BaseException as error:
        status = f"STOPPED: {type(error).__name__}: {error}"[:2000]
        for client in (composer.prep, composer.score):
            if not client.closed:
                client.abort()
        for process in (composer.ue, composer.edge):
            if process.poll() is None:
                process.terminate()
        if not composer.archive.closed:
            composer.archive.abort()
        raise
    finally:
        shutil.rmtree(out / "durable", ignore_errors=True)
        def dump(path, value):
            with open(out / path, "x") as stream:
                json.dump(value, stream, indent=1, sort_keys=True, default=str)
        with open(out / "FRAMES.jsonl", "x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        dump("QUEUE_SAMPLES.json", composer.samples)
        dump("MEMORY_SAMPLES.json", composer.memory)
        dump("RESULT.json", dict(status=status, variant=variant, summary=summary, limits=composer.limits,
             host_before=host_before, host_after=dict(loadavg=os.getloadavg(), raw_ns=raw()),
             timeline_inputs=timeline_inputs, cpu=dict(prep=CPU_PREP, score=CPU_SCORE, coordinator=CPU_COORD,
             edge=CPU_EDGE, ue="unpinned", archive="unpinned"), measured_vs_reconstructed="all values here are measured in "
             "the offline replay; recorded v11 offsets only pace arrivals", live_claim=False))
        inventory = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
        dump("OUTPUT_SHA256.json", inventory)
    print(status)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--variant", choices=("recorded", "spacing200"), default="recorded")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.name, args.variant, args.limit)
