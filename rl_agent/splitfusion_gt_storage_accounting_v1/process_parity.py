"""Accounting parity on COMPLETE real ``Staged`` frames (offline, in-process, CPU only).

The in-process ``DeferredProducerEngine`` (``parity_common.DeferredDriver``, the prep worker's engine class) runs the
49-frame retained v11 corpus with the prep worker's substitutions installed (producer V2 switch, tilt-aware AVO switch).
``ownership.payload_storage_bytes`` is wrapped by a recording shadow that, for EVERY real ``_account`` call, evaluates
the original and the candidate on the identical arguments ``(row.plan, row.staged, row.products)`` and returns the
original value (the engine itself runs the original).  Actions include accept, hold (``cancel_unbound``: exact history
rollback before bind) and unsent (``cancel`` after bind).

COMPOSITE (labelled, not a producer output): for every call whose ``staged`` is a ``Staged``, the same call is also
evaluated on a shallow copy whose ``conflict`` and ``_restore["conflict"]`` are replaced by the replayed V8 conflict
history at a late decision (complete arrays/rows of a real frame + V8 late history), to measure complete-frame cost
and parity at late-route history size.  Usage:

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 taskset -c 4 /usr/bin/python3 -B -m \\
      rl_agent.splitfusion_gt_storage_accounting_v1.process_parity --cell <V8 cell dir> --late 1300 --out <new dir>
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN
from rl_agent.splitfusion_numeric_gt_producer_v1 import source as NS
from rl_agent.splitfusion_numeric_gt_producer_v2 import worker_switch as PV2
from rl_agent.splitfusion_avo_tilt_v1 import switch as TILT

from . import accounting as A, replay_parity as RP

OLD, NEW = OWN.payload_storage_bytes, A.payload_storage_bytes
T = time.thread_time_ns


def action_for(i):
    return "hold" if i % 7 == 3 else "unsent" if i % 11 == 5 else "accept"


def run(cell, late, out):
    out.mkdir(parents=True, exist_ok=False)
    late_state = None
    for i, fid, m, res, snap, state, q_ok, _d in RP.states(cell):
        if i == late:
            late_state = dict(decision=i, frame_id=fid, res=res, snap=snap, pairs=len(state.pairs),
                              observations=sum(len(p.iou) for p in state.pairs.values()), quarantine_equal=q_ok)
            break
    calls = []

    def shadow(*values):
        caller = sys._getframe(2).f_code.co_name
        t0 = T(); old = OLD(*values); t_old = T() - t0
        t0 = T(); new = NEW(*values); t_new = T() - t0
        row = dict(caller=caller, old=old, new=new, equal=old == new, old_ms=t_old / 1e6, new_ms=t_new / 1e6,
                   staged=type(values[1]).__name__ if len(values) > 1 else None, products=values[2] is not None)
        staged = values[1] if len(values) > 1 else None
        if isinstance(staged, NS.Staged):
            composite = copy.copy(staged)
            composite.conflict = late_state["res"]
            composite._restore = dict(staged._restore, conflict=late_state["snap"])
            args = (values[0], composite, values[2])
            t0 = T(); c_old = OLD(*args); tc_old = T() - t0
            t0 = T(); c_new = NEW(*args); tc_new = T() - t0
            row.update(composite_old=c_old, composite_new=c_new, composite_equal=c_old == c_new,
                       composite_old_ms=tc_old / 1e6, composite_new_ms=tc_new / 1e6,
                       restore_has_conflict=bool(staged._restore))
        calls.append(row)
        return old

    P.environment_ok()
    PV2.install(); TILT.install()
    OWN.payload_storage_bytes = shadow
    actions = []
    try:
        corpus = P.Corpus(49)
        driver = P.DeferredDriver(corpus)
        try:
            for i, case in enumerate(corpus.cases):
                start = len(calls)
                result = driver.run(case, action=action_for(i))
                actions.append(dict(index=i, frame_id=case.frame_id, action=result["action"], calls=len(calls) - start,
                                    history_restored=(result["history_after"] == result["history_before"])
                                    if result["action"] != "accept" else None))
        finally:
            driver.close()
    finally:
        OWN.payload_storage_bytes = OLD
        TILT.uninstall(); PV2.uninstall()

    def med(key, sel):
        values = sorted(r[key] for r in sel if key in r)
        return values[len(values) // 2] if values else None
    staged_calls = [r for r in calls if "composite_equal" in r]
    summary = dict(
        schema="splitfusion.gt_storage_accounting_v1.process_parity.v1", structure="COMPLETE_STAGED_FRAMES",
        corpus_frames=len(actions), actions={a: sum(x["action"] == a for x in actions) for a in ("accept", "hold", "unsent")},
        cancel_paths_restored_history=all(x["history_restored"] for x in actions if x["action"] != "accept"),
        calls=len(calls), callers={c: sum(r["caller"] == c for r in calls) for c in sorted({r["caller"] for r in calls})},
        accounting_equal=sum(r["equal"] for r in calls),
        composite=dict(label="COMPOSITE_REAL_FRAME_WITH_V8_LATE_CONFLICT_HISTORY", late_decision=late_state["decision"],
                       late_frame_id=late_state["frame_id"], pairs=late_state["pairs"],
                       observations=late_state["observations"], calls=len(staged_calls),
                       accounting_equal=sum(r["composite_equal"] for r in staged_calls)),
        cost_ms=dict(real_corpus_history=dict(old_p50=med("old_ms", calls), new_p50=med("new_ms", calls)),
                     composite_late_history=dict(old_p50=med("composite_old_ms", staged_calls),
                                                 new_p50=med("composite_new_ms", staged_calls))),
        bytes_range=[min(r["old"] for r in calls), max(r["old"] for r in calls)],
        composite_bytes_range=[min(r["composite_old"] for r in staged_calls), max(r["composite_old"] for r in staged_calls)],
        reservation_bytes=64 * 1024 * 1024, switches_restored=not PV2.installed() and not TILT.installed()
        and OWN.payload_storage_bytes is OLD, candidate_version=A.VERSION)
    (out / "CALLS.json").write_text(json.dumps(dict(actions=actions, calls=calls), indent=1) + "\n")
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    ok = (summary["accounting_equal"] == summary["calls"] > 0
          and summary["composite"]["accounting_equal"] == summary["composite"]["calls"] > 0
          and summary["cancel_paths_restored_history"] and summary["switches_restored"])
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", type=Path, required=True)
    p.add_argument("--late", type=int, default=1300)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    raise SystemExit(0 if run(a.cell, a.late, a.out) else 1)
