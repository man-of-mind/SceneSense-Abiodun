"""Full-route termination for M3.1 and the repair of the retained post-budget TypeError at its cause.

Cause (diagnosed from the retained sources, M1 capture ``route_detail.error``): ``splitfusion_exact_gt_live_v2.ue``
``EarlyGtCollector.finish`` does ``result = super().finish(); result["numeric_gt"] = handoff.close()``. The base
``PassiveSplitCollector.finish`` (reached through the Run-5 live-GT and owned-pose layers) returns ``bool``. The
handoff close completes (``EXACT_GT_UE_RESULT.json`` is written) and THEN the item assignment raises ``TypeError``.
Raised inside ``collecting_drive``'s ``finally`` it replaces the budget-stop exception; on a natural route end it
would discard the completed route result, so no durable route summary could ever be written.

Repair: ``M3CollectorLayer`` is inserted directly BELOW ``EarlyGtCollector`` (it wraps the builder that the early-GT
installation captures); its ``finish`` returns ``FinishResult`` -- a dict whose truth value is exactly the cleanup
flag -- so the retained assignment succeeds, ``bool(...)`` callers keep identical semantics, and nothing is caught or
suppressed. The same layer tees the exact installed map record (see ``map_journal``).

Termination: ``accept_route`` replaces only the retained budget-only stop rule of ``phase6_live_child_v2.run``
(registered source substitution ``R1_CHILD_ROUTE_STOP_RULE``). ``g1_route_branch`` and ``p3_final_cycle`` extend the
runner's G1 and Phase-6 P3 checks with the pre-registered ROUTE_COMPLETED branch / final-cycle exception.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time

from . import capacity as CAP
from . import contract as C
from . import map_journal as MJ

ROUTE_COMPLETED = "ROUTE_COMPLETED"
EVIDENCE_DIR = "m3_route"

CHILD_STOP_RULE = dict(
    id="R1_CHILD_ROUTE_STOP_RULE", process="UE child (entries.child)",
    target="rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_live_child_v2:run",
    replacements=[["""        stop_reason = str(collector.probe_stop_reason)
        # Addendum 6: a stop at a closed k_min decision cycle is registered;
        # it can only end at or below the transmitted budget.
        if stop_reason == "DECISION_CYCLE_BOUNDARY":
            require(1 <= int(collector.sent) <= int(args.transmitted_budget),
                    f"cycle-boundary stop outside the budget ({int(collector.sent)})")
        else:
            require(int(collector.sent) == int(args.transmitted_budget),
                    f"only {int(collector.sent)}/{int(args.transmitted_budget)} frames sent")
            require(stop_reason == "TRANSMITTED_BUDGET_REACHED",
                    f"unexpected stop reason {collector.probe_stop_reason!r}")
""", """        stop_reason = _M3_ROUTE_ACCEPT(collector=collector, detail=detail, accepted=_ok,
                                       ceiling=int(args.transmitted_budget), result=result)
"""]])
CAP.REGISTRY[CHILD_STOP_RULE["id"]] = CHILD_STOP_RULE


class IncompleteRoute(RuntimeError):
    """The planned loop did not complete (ceiling, timeout, watchdog, summary or collector failure)."""


class FinishResult(dict):
    """Mapping result of the collector finish chain whose truth value is exactly the base cleanup flag."""

    def __init__(self, cleanup_ok):
        if type(cleanup_ok) is not bool:
            raise TypeError("FinishResult wraps the retained bool cleanup flag only")
        super().__init__(cleanup_ok=cleanup_ok)
        self._cleanup_ok = cleanup_ok

    def __bool__(self):
        return self._cleanup_ok


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=1, default=str) + "\n")
    os.replace(temporary, path)


def build_layer(inherited):
    """The M3 collector layer placed directly below EarlyGtCollector."""

    class M3CollectorLayer(inherited):
        def __init__(self, **kwargs):
            # before the retained constructor: no retrieval can ever precede the journal
            self._m3_journal = MJ.MapJournal(Path(kwargs["attempt_dir"]) / C.MAP_JOURNAL_DIR)
            super().__init__(**kwargs)

        def _fetch_installed_record(self, expected_frame):
            capture = MJ.begin_capture()
            started = time.time_ns()
            try:
                outcome, predictions = super()._fetch_installed_record(expected_frame)
            finally:
                raw = MJ.end_capture(capture)
            self._m3_journal.record(frame_id=int(expected_frame), outcome=str(outcome), raw=raw,
                                    retrieval_start_wall_ns=started, retrieval_end_wall_ns=time.time_ns())
            return outcome, predictions

        def finish(self):
            inner = super().finish()
            evidence = dict(schema="splitfusion.m3_route_b_v1.finish_result.v1", inner_type=type(inner).__name__,
                            finished_wall_ns=time.time_ns())
            try:
                if type(inner) is not bool:
                    raise TypeError(f"retained finish returned {type(inner).__name__}, expected bool")
                result = FinishResult(inner)
                evidence["cleanup_ok"] = inner
                return result
            finally:
                evidence["map_journal"] = self._m3_journal.close()
                _write_json(Path(self.attempt_dir) / EVIDENCE_DIR / "FINISH_RESULT.json", evidence)

    return M3CollectorLayer


@contextmanager
def install_collector_layer():
    """Wrap the live-GT collector builder that ``install_early_gt_ue`` captures (enter between pose and early GT)."""
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
    inner = LG.build_live_gt_collector
    built = []

    def builder(base):
        layer = build_layer(inner(base))
        built.append(layer)
        return layer

    LG.build_live_gt_collector = builder
    MJ.install_tee()
    try:
        yield built
    finally:
        MJ.uninstall_tee()
        LG.build_live_gt_collector = inner


# ---------------------------------------------------------------------------------------------- UE child stop rule
def accept_route(*, collector, detail, accepted, ceiling, result):
    """Registered replacement of the retained budget-only stop rule (UE child, after run_route_b returned)."""
    detail = dict(detail or {})
    stop = str(collector.probe_stop_reason)
    sent = int(collector.sent)
    evidence = dict(
        collector_stop_reason=stop, transmitted_frames=sent, frame_ceiling=int(ceiling),
        run_route_b_accepted=bool(accepted), route_completed=detail.get("route_completed"),
        route_runner_returncode=detail.get("route_runner_returncode"), density_status=detail.get("density_status"),
        route_abort_reason=detail.get("route_abort_reason"), error=detail.get("error"),
        route_summary_persisted=detail.get("route_summary_persisted"),
        route_summary_identity_ok=detail.get("route_summary_identity_ok"),
        route_summary_sha256=detail.get("route_summary_sha256"),
        route_outcome=(detail.get("route_outcome") or {}).get("classification"))
    result["m3_route_completion"] = evidence
    if stop:
        evidence["terminal"] = "INCOMPLETE_ROUTE"
        raise IncompleteRoute(f"INCOMPLETE_ROUTE: collector stop {stop} at {sent}/{int(ceiling)} frames")
    complete = (bool(accepted) and detail.get("route_completed") is True and detail.get("route_summary_persisted") is True
                and detail.get("route_summary_identity_ok") is True and not detail.get("error")
                and 1 <= sent <= int(ceiling))
    if not complete:
        evidence["terminal"] = "INCOMPLETE_ROUTE"
        raise IncompleteRoute("INCOMPLETE_ROUTE: route runner/summary/collector criteria not met")
    collector.probe_stop_reason = ROUTE_COMPLETED
    collector._persist_probe_summary()
    result["collector"] = collector.probe_summary()
    evidence["terminal"] = ROUTE_COMPLETED
    return ROUTE_COMPLETED


@contextmanager
def install_child_stop_rule():
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_child_v2 as CHILD
    if hasattr(CHILD, "_M3_ROUTE_ACCEPT"):
        raise RuntimeError("route stop rule already installed")
    CHILD._M3_ROUTE_ACCEPT = accept_route
    installed = CAP.install((CHILD_STOP_RULE["id"],))
    try:
        yield installed
    finally:
        installed.uninstall()
        del CHILD._M3_ROUTE_ACCEPT


# ---------------------------------------------------------------------------------------------- runner-side gates
def g1_route_branch(original):
    """G1 with the registered ROUTE_COMPLETED branch; every other stop keeps the retained rule except that a frame-
    ceiling stop is explicitly an incomplete route."""

    def g1_frame_accounting(*, child, timing, analysis, transmitted_budget, boundary_budget=None, **kwargs):
        if boundary_budget is not None:
            kwargs["boundary_budget"] = boundary_budget
        base = original(child=child, timing=timing, analysis=analysis, transmitted_budget=transmitted_budget,
                        **kwargs)
        stop = child.get("stop_reason")
        if stop != ROUTE_COMPLETED:
            if stop == "TRANSMITTED_BUDGET_REACHED":
                return dict(base, branch="FRAME_CEILING_STOP_INCOMPLETE_ROUTE", passed=False)
            return base
        detail = dict(child.get("route_detail") or {})
        completion = dict(child.get("m3_route_completion") or {})
        sent, budget = base.get("sent"), int(transmitted_budget)
        route_checks = dict(
            child_terminal=completion.get("terminal") == ROUTE_COMPLETED,
            route_completed=detail.get("route_completed") is True,
            route_summary_persisted=detail.get("route_summary_persisted") is True,
            route_summary_identity_ok=detail.get("route_summary_identity_ok") is True,
            no_route_error=not detail.get("error"),
            within_ceiling=sent is not None and 1 <= int(sent) <= budget)
        checks = dict(base["checks"], **route_checks)
        return dict(base, branch=ROUTE_COMPLETED if all(route_checks.values()) else None, checks=checks,
                    passed=all(checks.values()))

    g1_frame_accounting.__m3_route_branch__ = True
    return g1_frame_accounting


def p3_final_cycle(original):
    """Phase-6 gates with the registered final-cycle exception for P3 only (all other gates/verdict rule unchanged)."""

    def evaluate_phase6(*, ue, edge, map_identity_rows, feedback_packets_on_ue_tunnel, cleanup_ok):
        result = original(ue=ue, edge=edge, map_identity_rows=map_identity_rows,
                          feedback_packets_on_ue_tunnel=feedback_packets_on_ue_tunnel, cleanup_ok=cleanup_ok)
        gates = dict(result["gates"])
        exception = final_cycle_exception(ue)
        result["m3_final_cycle_exception"] = exception
        if gates.get("P3_HOLD_TICKET") is False and exception["applies"]:
            gates["P3_HOLD_TICKET"] = True
            result["gates"] = gates
            result["verdict"] = ("PASSED" if all(gates.values())
                                 else "INCONCLUSIVE_OR_FAILED" if not gates["P0_POLICY_COVERAGE"] else "FAILED")
        return result

    evaluate_phase6.__m3_final_cycle__ = True
    return evaluate_phase6


def final_cycle_exception(ue):
    frames = list(ue.get("frames") or ())
    sessions = list(ue.get("sessions") or ())
    groups = {}
    for frame in frames:
        groups.setdefault((frame.get("session_uuid", ""), frame["ticket_seq"]), []).append(frame)
    short = [key for key, group in groups.items() if len(group) < 2]
    reward_ok = all(sum(1 for f in g if f["reward_requested"]) == 1 for g in groups.values())
    detail = dict(groups=len(groups), short_groups=[list(k) for k in short], reward_counts_all_one=reward_ok)
    if len(short) != 1 or not reward_ok or not sessions or not frames:
        return dict(detail, applies=False)
    session, ticket = short[0]
    last_session = sessions[-1]
    last_ticket = max(k[1] for k in groups if k[0] == last_session)
    last_frame = max(int(f["frame_id"]) for f in frames)
    only = groups[short[0]][0]
    applies = (session == last_session and ticket == last_ticket and int(only["frame_id"]) == last_frame
               and only["reward_requested"] is True)
    return dict(detail, applies=bool(applies), final_group=dict(session=session, ticket_seq=ticket,
                                                                 frame_id=int(only["frame_id"])))
