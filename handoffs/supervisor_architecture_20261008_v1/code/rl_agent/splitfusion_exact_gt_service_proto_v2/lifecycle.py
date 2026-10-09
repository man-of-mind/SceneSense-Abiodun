"""Event-ordered lifecycle driver for the two-process service (correctness only).

Event times are scenario coordinates used ONLY to order operations; nothing
waits, sleeps or measures. Each frame is one sealed v11 tick with a declared kind
(decision, hold, fallback). The coordinator enforces the producer's single
pending transaction with a bounded freeze queue (depth 1) and admission control
on producer retention, emits feedback only from scorer results, and retires
everything through real durable archival unless archival is stalled.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO

from . import parity_common as P, service as SV
from .scorer_core import triple_key

TIMINGS = P.ROOT / "experiments/splitfusion_numeric_gt_runtime_v1/completion_live100_timing_01/FRAME_TIMINGS.csv"
PREDICTION_FIELDS = ("gt_identity", "action_identity", "capture_timestamp_ns", "prediction_digest")
FREEZE_LEAD_MS, PLAN_MS, HORIZON_MS = 25.0, 5.7, 500.0


def recorded_rows():
    with open(TIMINGS, newline="") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: int(r["action_open_raw_ns"]))
    return [dict(open_ns=int(r["action_open_raw_ns"]), send=float(r["action_open_to_last_send_ms"]),
                 prediction=float(r["action_open_to_prediction_ready_ms"])) for r in rows]


@dataclass
class Frame:
    index: int
    kind: str                       # DECISION | HOLD | FALLBACK
    open_ms: float
    send_ms: float | None = None    # None = no send event; negative = send failure
    prediction_ms: float | None = None
    late_prediction_ms: float | None = None
    refuse_deferral: bool = False
    freeze: str | None = None       # QUEUED | PENDING | ADMITTED | REFUSED_<reason>
    job: int | None = None
    plan_seen: bool = False
    phase: str = "NEW"
    identity: str | None = None
    send_result: str | None = None
    scorer: str | None = None       # target admission / prediction state
    result: dict | None = None
    terminal: str | None = None
    feedback: int = 0
    retire: str | None = None       # SCORED | UNSCORED_<reason>
    scorer_holds_target: bool = False
    retired: bool = False
    notes: list = field(default_factory=list)


class Coordinator:
    def __init__(self, harness, corpus, frames, *, max_prep_retained, archive_stalls=()):
        self.h, self.c, self.frames = harness, corpus, frames
        self.max_prep_retained = max_prep_retained
        self.archive_stalls = list(archive_stalls)
        self.prep_open = None          # frame index with an unresolved producer transaction
        self.prep_pending = None       # frame index whose FREEZE request is in flight
        self.queue = []                # bounded freeze queue (depth 1)
        self.prep_retained = 0
        self.feedback = []
        self.refusals = []
        self.max_seen = dict(queue=0, prep_retained=0, scorer_slots=0, scorer_waiting=0)
        self.history_committed = []

    # ---------------- helpers ----------------
    def case(self, f):
        return self.c.cases[f.index]

    def context(self, f):
        return PO.context_document(self.c.context(self.case(f)))

    def action(self, f):
        action = dict(self.c.predictions[self.case(f).frame_id]["identity"]["action_identity"])
        if f.kind != "DECISION":
            action.update(frame_kind="POLICY_HOLD" if f.kind == "HOLD" else "FALLBACK", reward_requested=False)
        return action

    def plan(self, f):
        plan = C._parse(self.c.plan(self.case(f)))
        if f.refuse_deferral:
            plan["capture"] = dict(plan["capture"], action_id=71, profile_id="mode11_q9800_a71")
        return C.canonical(plan)

    def stalled(self, t):
        return any(a <= t < b for a, b in self.archive_stalls)

    def _prep_sync(self):
        if self.prep_pending is None:
            return
        f = self.frames[self.prep_pending]
        self.prep_pending = None
        reply = self.h.prep.finish()
        result = SV.ok(reply, "FREEZE")
        f.job, f.freeze, f.phase = result["job"], "ADMITTED", "UNBOUND"
        if f.plan_seen:
            self._apply_plan(f)

    def _prep_idle(self):
        while self.prep_pending is not None:
            self._prep_sync()

    def prep_call(self, op, args):
        self._prep_idle()
        return self.h.prep.call(op, args)

    def _observe(self):
        self.max_seen["queue"] = max(self.max_seen["queue"], len(self.queue))
        self.max_seen["prep_retained"] = max(self.max_seen["prep_retained"], self.prep_retained)
        snap = SV.ok(self.h.score.call("SNAPSHOT"), "score SNAPSHOT")
        C.require(snap["valid"] and snap["retained_slots"] <= snap["max_jobs"]
                  and snap["waiting_predictions"] <= snap["max_waiting"]
                  and snap["reserved_bytes"] <= snap["max_bytes"], "scorer bound violated")
        self.max_seen["scorer_slots"] = max(self.max_seen["scorer_slots"], snap["retained_slots"])
        self.max_seen["scorer_waiting"] = max(self.max_seen["scorer_waiting"], snap["waiting_predictions"])

    def _resolve_open(self, f):
        if self.prep_open == f.index:
            self.prep_open = None
        while self.queue and self.prep_open is None:
            self._dispatch(self.frames[self.queue.pop(0)])

    # ---------------- events ----------------
    def ev_freeze(self, f, t):
        if self.prep_open is not None:
            if len(self.queue) >= 1:
                f.freeze = "REFUSED_FREEZE_QUEUE_FULL"
                self.refusals.append((f.index, f.freeze))
                return
            f.freeze = "QUEUED"
            self.queue.append(f.index)
            return
        self._dispatch(f)

    def _dispatch(self, f):
        if self.prep_retained >= self.max_prep_retained:
            f.freeze = "REFUSED_PREP_CAPACITY"
            self.refusals.append((f.index, f.freeze))
            return
        self._prep_idle()
        C.require(self.prep_open is None, "single pending producer transaction violated")
        case = self.case(f)
        self.h.freeze(self.c.sourcekwargs(case), case.kwargs["policy_timestamp_s"])
        f.freeze, self.prep_open, self.prep_pending = "PENDING", f.index, f.index

    def ev_plan(self, f, t):
        f.plan_seen = True
        if f.freeze == "PENDING":
            self._prep_idle()            # binding depends on the finished freeze; applies the plan
        elif f.freeze == "ADMITTED" and f.phase == "UNBOUND":
            self._apply_plan(f)

    def _apply_plan(self, f):
        if f.kind != "DECISION":
            SV.ok(self.prep_call("CANCEL_UNBOUND", dict(job=f.job, reason=f.kind + "_NEVER_BOUND")), "CANCEL_UNBOUND")
            f.phase = "CANCELLED"
            self._resolve_open(f)
            return
        reply = self.prep_call("BIND", dict(job=f.job, plan=self.plan(f).decode(), frame_context=self.context(f)))
        if reply.status == "REFUSED":
            f.phase, f.terminal = "NO_GT_" + reply.reason, None
            self.refusals.append((f.index, reply.reason))
            self._resolve_open(f)
            return
        f.phase, f.identity = "BOUND", SV.ok(reply, "BIND")["identity"]
        if f.send_result is not None:
            self._apply_send(f)

    def ev_send(self, f, t):
        f.send_result = "FAILED" if f.send_ms < 0 else "SENT"
        if f.freeze == "PENDING":
            self._prep_idle()            # plan already seen: binds, then applies this send
        elif f.phase == "BOUND":
            self._apply_send(f)

    def _apply_send(self, f):
        if f.send_result == "FAILED":
            SV.ok(self.prep_call("CANCEL_BOUND", dict(job=f.job, plan=self.plan(f).decode(),
                                                      reason="SEND_FAILED_AFTER_BIND")), "CANCEL_BOUND")
            f.phase = "CANCELLED"
            self._resolve_open(f)
            return
        SV.ok(self.prep_call("CONFIRM_SEND", dict(job=f.job, plan=self.plan(f).decode())), "CONFIRM_SEND")
        self.history_committed.append(f.index)
        self.prep_retained += 1
        f.phase = "COMMITTED"
        self._resolve_open(f)
        self._prep_idle()
        published, target = self.h.publish_to_scorer(f.job)
        f.phase = "PUBLISHED"
        if target.status == "REFUSED":
            f.scorer = "TARGET_REFUSED_" + target.reason
            f.retire, f.scorer_holds_target = "UNSCORED_" + target.reason, False
            return
        f.scorer, f.scorer_holds_target = "TARGET_ADMITTED", True
        joined = target.result["joined_result"]
        if joined is not None:
            self._emit(f, joined)

    def _prediction(self, f):
        pred = self.c.predictions[self.case(f).frame_id]
        identity = {k: pred["identity"][k] for k in PREDICTION_FIELDS}
        identity["action_identity"] = self.action(f)
        return identity, pred

    def ev_predict(self, f, t, *, late=False):
        identity, pred = self._prediction(f)
        reply = self.h.predict(identity, self.context(f), pred["records"], pred["labels"].tobytes(order="C"))
        if reply.status == "REFUSED":
            f.notes.append(("PREDICTION_REFUSED", reply.reason, late))
            self.refusals.append((f.index, reply.reason))
            return
        C.require(not late and f.kind == "DECISION", "only timely decisions may be accepted")
        result = SV.ok(reply, "PREDICTION")
        if result["status"] == "SCORED":
            self._emit(f, result["result"])
        else:
            f.scorer = "PREDICTION_WAITING"

    def _emit(self, f, result):
        C.require(f.kind == "DECISION" and f.feedback == 0 and f.terminal is None, "feedback exactly once for a decision")
        f.result, f.feedback = result, 1
        self.feedback.append(f.index)
        SV.ok(self.h.score.call("RESULT_DELIVERED", dict(identity=f.identity)), "score RESULT_DELIVERED")
        SV.ok(self.prep_call("RESULT_DELIVERED", dict(job=f.job)), "prep RESULT_DELIVERED")
        f.retire = "SCORED"

    def ev_horizon(self, f, t):
        if f.kind != "DECISION" or f.feedback or f.send_result != "SENT" or f.terminal is not None:
            return
        case = self.case(f)
        reply = self.h.score.call("ABANDON", dict(capture=case.kwargs["identity"], action=self.action(f),
                                                  frame_context=self.context(f), reason="NO_RESULT_BY_HORIZON"))
        status = SV.ok(reply, "ABANDON")["status"]
        f.terminal = "UNSCORED_" + status
        if status == "UNSCORED":
            f.retire, f.scorer_holds_target = "UNSCORED_NO_RESULT_BY_HORIZON", True

    def ev_archive(self, f_unused, t):
        if self.stalled(t):
            return
        for f in self.frames:
            if f.retired or f.retire is None or f.phase != "PUBLISHED":
                continue
            case = self.case(f)
            if f.retire == "SCORED":
                self._prep_idle()
                self.h.retire_scored(f.job, case.kwargs["identity"], f.identity, case.frame_id)
            else:
                self._prep_idle()
                self.h.retire_unscored(f.job, case.kwargs["identity"], f.identity, case.frame_id,
                                       f.retire.removeprefix("UNSCORED_"), scorer_holds_target=f.scorer_holds_target)
            f.retired = True
            self.prep_retained -= 1

    # ---------------- run ----------------
    def events(self):
        out = []
        for f in self.frames:
            out.append((f.open_ms - FREEZE_LEAD_MS, 0, "freeze", f.index))
            out.append((f.open_ms + PLAN_MS, 1, "plan", f.index))
            if f.send_ms is not None:
                out.append((f.open_ms + abs(f.send_ms), 2, "send", f.index))
            if f.prediction_ms is not None:
                out.append((f.open_ms + f.prediction_ms, 3, "predict", f.index))
            if f.late_prediction_ms is not None:
                out.append((f.open_ms + f.late_prediction_ms, 5, "late", f.index))
            out.append((f.open_ms + HORIZON_MS, 4, "horizon", f.index))
        end = max(e[0] for e in out)
        out += [(t, 6, "archive", -1) for t in range(0, int(end) + 101, 100)]
        return sorted(out)

    def run(self):
        for t, _, kind, index in self.events():
            f = self.frames[index] if index >= 0 else None
            if f is not None and kind != "freeze" and f.freeze is not None and f.freeze.startswith("REFUSED") \
                    and kind in ("plan", "send"):
                if kind == "send":
                    f.send_result = "FAILED" if f.send_ms < 0 else "SENT"
                continue
            if kind == "late":
                self.ev_predict(f, t, late=True)
            else:
                getattr(self, "ev_" + kind)(f, t)
            self._observe()
        self._prep_idle()
        self.archive_stalls = []
        self.ev_archive(None, float("inf"))
        self._observe()
        return self.summary()

    def summary(self):
        decisions = [f for f in self.frames if f.kind == "DECISION"]
        sent = [f for f in decisions if f.send_result == "SENT"]
        return dict(frames=len(self.frames), decisions=len(decisions), sent_decisions=len(sent),
            feedback=len(self.feedback), feedback_frames=list(self.feedback),
            unscored=[(f.index, f.terminal or f.retire or f.phase) for f in sent if not f.feedback],
            no_feedback_non_decisions=all(f.feedback == 0 for f in self.frames if f.kind != "DECISION"),
            exactly_once=all(f.feedback <= 1 for f in self.frames),
            refusals=self.refusals, max_seen=self.max_seen, committed=list(self.history_committed),
            all_retired=all(f.retired for f in self.frames if f.phase == "PUBLISHED"),
            prep_retained=self.prep_retained)


def expected_history(corpus, committed):
    """Reference accepted history over exactly the committed frames (runtime producer, in-parent)."""
    ref = P.ReferenceDriver(corpus)
    try:
        last = None
        for index in committed:
            last = ref.run(corpus.cases[index])["history_after"]
        return last
    finally:
        ref.close()
