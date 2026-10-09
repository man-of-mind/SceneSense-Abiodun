"""Forced fixed action with the frozen actor kept in shadow.

``install_forced_action`` patches ``phase6_decision_engine_v2.Run4DecisionEngineV2`` (looked up at
runtime construction) with a subclass whose ``plan_frame`` temporarily wraps whatever actor the engine
holds -- including the profiler's ``agent.actor`` proxy -- OUTERMOST for the duration of one call:

* the real actor (and the profiler span) runs normally on the normally constructed state, so the span
  records the genuine proposal and the state hashes;
* the proposal is discarded and replaced by the registered (mode_id, q_e4); the retained engine then
  resolves the profile, opens the ticket, holds and previous-action features from that executed action;
* every POLICY_DECISION, POLICY_HOLD and FALLBACK plan is logged with proposed and executed actions.

Fallback frames (state unavailable, actor not called) keep the retained fallback semantics unchanged.
No retained source is edited.
"""
from __future__ import annotations

from contextlib import contextmanager
import dataclasses
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace


def _raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


class ForcedActionError(RuntimeError):
    pass


def _support(actor, mode_id):
    """The bounds the frozen actor itself enforces (through any forwarding proxy), if exposed."""
    try:
        lower, upper = actor._lower, actor._upper
    except AttributeError:
        return None
    return int(lower[mode_id]), int(upper[mode_id])


class _ForcedActor:
    """Forwards everything; ``act`` calls the real actor and returns the forced decision."""

    def __init__(self, inner, forced, sink):
        self._inner, self._forced, self._sink = inner, forced, sink

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def act(self, features):
        proposal = self._inner.act(features)
        mode, q = self._forced["mode_id"], self._forced["q_e4"]
        support = _support(self._inner, mode)
        if support is not None and not support[0] <= q <= support[1]:
            raise ForcedActionError(f"forced action escapes the actor's registered support {support}")
        if dataclasses.is_dataclass(proposal):
            changes = dict(mode_id=mode, q_e4=q)
            if support is not None and {"q_support_lower_e4", "q_support_upper_e4"} <= {f.name for f in dataclasses.fields(proposal)}:
                changes.update(q_support_lower_e4=support[0], q_support_upper_e4=support[1])
            executed = dataclasses.replace(proposal, **changes)
        else:
            executed = SimpleNamespace(**dict(vars(proposal), mode_id=mode, q_e4=q))
        self._sink.append(dict(proposed_mode_id=int(proposal.mode_id), proposed_q_e4=int(proposal.q_e4),
                               state_sha256=getattr(proposal, "state_sha256", None),
                               heads_sha256=getattr(proposal, "heads_sha256", None),
                               actor_raw_ns=_raw()))
        return executed


class ForcedLog:
    def __init__(self, path, forced):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = open(self.path, "x", buffering=1)
        self._lock = threading.Lock()
        self.forced = dict(forced)
        self.counters = dict(decisions=0, holds=0, fallbacks=0, proposal_differs=0, executed_mismatch=0)

    def write(self, row):
        with self._lock:
            self._stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def close(self):
        with self._lock:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()


def _profile_document(profile):
    return dict(mode_id=int(profile.mode_id), q_e4=int(profile.q_e4), action_id=getattr(profile, "action_id", None),
                profile_id=getattr(profile, "profile_id", None), keep_count=getattr(profile, "keep_count", None),
                execution_bundle_sha256=getattr(profile, "execution_bundle_sha256", None))


def make_engine_class(base, forced, log):
    class ForcedActionEngine(base):
        def plan_frame(self, **kwargs):
            sink = []
            held = self._actor
            self._actor = _ForcedActor(held, forced, sink)
            try:
                plan = super().plan_frame(**kwargs)
            finally:
                self._actor = held
            kind = plan.kind.value if hasattr(plan.kind, "value") else str(plan.kind)
            executed = _profile_document(plan.profile)
            row = dict(frame_id=int(kwargs["frame_id"]), kind=kind, executed=executed, forced=dict(forced),
                       proposed=sink[0] if sink else None, actor_calls_this_frame=len(sink), raw_ns=_raw())
            if kind == "POLICY_DECISION":
                log.counters["decisions"] += 1
                if sink and (sink[0]["proposed_mode_id"], sink[0]["proposed_q_e4"]) != (forced["mode_id"], forced["q_e4"]):
                    log.counters["proposal_differs"] += 1
            elif kind == "POLICY_HOLD":
                log.counters["holds"] += 1
            else:
                log.counters["fallbacks"] += 1
            if kind in ("POLICY_DECISION", "POLICY_HOLD") and (executed["mode_id"], executed["q_e4"]) != (forced["mode_id"], forced["q_e4"]):
                log.counters["executed_mismatch"] += 1
                log.write(dict(row, error="EXECUTED_ACTION_DIFFERS_FROM_FORCED"))
                raise ForcedActionError("executed action differs from the registered forced action")
            log.write(row)
            return plan

    return ForcedActionEngine


@contextmanager
def install_forced_action(*, log_path, forced):
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_decision_engine_v2 as E
    forced = dict(mode_id=int(forced["mode_id"]), q_e4=int(forced["q_e4"]), action_id=forced.get("action_id"),
                  profile_id=forced.get("profile_id"))
    original = E.Run4DecisionEngineV2
    if getattr(original, "_forced_action_installed", False):
        raise ForcedActionError("nested forced-action installation")
    log = ForcedLog(log_path, forced)
    engine_class = make_engine_class(original, forced, log)
    engine_class._forced_action_installed = True
    E.Run4DecisionEngineV2 = engine_class
    state = dict(schema="splitfusion.fixed_action_screen_v2.forced_scope.v1", forced=forced, log=str(log_path))
    try:
        yield state
    finally:
        E.Run4DecisionEngineV2 = original
        state["counters"] = dict(log.counters)
        log.close()
        summary = Path(log_path).with_name("FORCED_ACTION_SUMMARY.json")
        with open(summary, "x") as stream:
            json.dump(state, stream, indent=1, sort_keys=True)
