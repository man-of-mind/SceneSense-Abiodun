"""Complete-frame parity and same-process CPU/memory qualification (offline)."""
import argparse
import copy
import gc
import json
from pathlib import Path
import statistics
import sys
import time
import tracemalloc

from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN
from rl_agent.splitfusion_numeric_gt_producer_v1 import source as NS
from rl_agent.splitfusion_numeric_gt_producer_v2 import worker_switch as PV2
from rl_agent.splitfusion_avo_tilt_v1 import switch as TILT
from rl_agent.splitfusion_gt_storage_accounting_v1 import accounting as V1
from rl_agent.splitfusion_gt_storage_accounting_v1 import process_parity as V1P
from rl_agent.splitfusion_gt_storage_accounting_v1 import replay_parity as RP

from . import accounting as V2

SCHEMA = "splitfusion.gt_storage_accounting_v2.process_qualification.v1"
# Declared before timing: at least 25% lower P50 summed CPU on four-call
# complete synthetic-late decision frames, versus the deployed V1 function.
ACCEPTANCE = dict(complete_late_four_call_sum_v2_to_v1_max=0.75,
                  accounting_parity_required=True, capacity_and_rollback_required=True)
OLD = OWN.payload_storage_bytes
T = time.thread_time_ns


def _median(values):
    return statistics.median(values) if values else None


def _timed(function, args):
    begin = T()
    value = function(*args)
    return value, (T() - begin) / 1e6


def _memory(function_factory, args, calls=4):
    gc.collect()
    tracemalloc.start()
    function = function_factory()
    for _ in range(calls):
        function(*args)
    current, peak = tracemalloc.get_traced_memory()
    snapshot = function.snapshot() if isinstance(function, V2.AccountingSession) else None
    if isinstance(function, V2.AccountingSession):
        function.close()
    after_close = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    return dict(calls=calls, retained_bytes=current, peak_bytes=peak,
                temporary_above_retained_bytes=peak-current, after_close_bytes=after_close,
                candidate_snapshot=snapshot)


def run(cell, late, out):
    out.mkdir(parents=True, exist_ok=False)
    late_state = None
    for index, frame_id, metadata, result, snapshot, state, quarantine_ok, _digest in RP.states(cell):
        if index == late:
            late_state = dict(index=index, frame_id=frame_id, result=result, snapshot=snapshot,
                              pairs=len(state.pairs), observations=sum(len(h.iou) for h in state.pairs.values()),
                              quarantine_equal=quarantine_ok)
            break
    if late_state is None or not late_state["quarantine_equal"]:
        raise RuntimeError("late reconstructed history unavailable or not quarantine-exact")

    candidate_real = V2.AccountingSession()
    candidate_composite = V2.AccountingSession()
    calls, actions, late_snapshots, composites = [], [], {}, {}
    current_frame = {"index": None, "frame_id": None}
    memory_args = None

    def measure(args, label, caller):
        candidate = (candidate_real if label == "COMPLETE_REAL_49_FRAME_CORPUS" else candidate_composite)
        order = (("old", OLD), ("v1", V1.payload_storage_bytes), ("v2", candidate))
        offset = len(calls) % len(order)
        measured = {}
        for name, function in order[offset:] + order[:offset]:
            measured[name] = _timed(function, args)
        values = {value for value, _ms in measured.values()}
        row = dict(frame_index=current_frame["index"], frame_id=current_frame["frame_id"],
                   caller=caller, structure=label,
                   **{name + "_bytes": measured[name][0] for name in measured},
                   **{name + "_ms": measured[name][1] for name in measured},
                   equal=len(values) == 1)
        calls.append(row)
        return row

    def shadow(*values):
        nonlocal memory_args
        caller = sys._getframe(2).f_code.co_name
        real = measure(values, "COMPLETE_REAL_49_FRAME_CORPUS", caller)
        staged = values[1] if len(values) > 1 else None
        if isinstance(staged, NS.Staged):
            key = id(staged)
            snapshot = late_snapshots.setdefault(key, dict(late_state["snapshot"]))
            composite = composites.get(key)
            if composite is None:
                composite = composites[key] = copy.copy(staged)
                composite.conflict = late_state["result"]
                composite._restore = dict(staged._restore, conflict=snapshot)
            # These are the only Staged fields populated lazily by the real
            # assembly path. Preserve one object across its four owner stages.
            composite.evidence = staged.evidence
            composite.inputs = staged.inputs
            composite_args = (values[0], composite, values[2])
            measure(composite_args, "COMPLETE_REAL_FRAME_PLUS_SYNTHETIC_V8_LATE_HISTORY", caller)
            memory_args = composite_args
        return real["old_bytes"]

    P.environment_ok()
    corpus = P.Corpus(49)
    PV2.install(); TILT.install(); OWN.payload_storage_bytes = shadow
    try:
        driver = P.DeferredDriver(corpus)
        try:
            for index, case in enumerate(corpus.cases):
                current_frame.update(index=index, frame_id=case.frame_id)
                begin = len(calls)
                result = driver.run(case, action=V1P.action_for(index))
                frame_rows = calls[begin:]
                actions.append(dict(index=index, frame_id=case.frame_id, action=result["action"],
                                    accounting_rows=len(frame_rows),
                                    history_restored=(result["history_after"] == result["history_before"])
                                    if result["action"] != "accept" else None))
        finally:
            driver.close()
    finally:
        OWN.payload_storage_bytes = OLD
        TILT.uninstall(); PV2.uninstall()

    def selected(structure):
        return [row for row in calls if row["structure"] == structure]

    def per_call(rows):
        return {name + "_p50_ms": _median([row[name + "_ms"] for row in rows])
                for name in ("old", "v1", "v2")}

    def per_decision(rows, exactly_four=False):
        grouped = {}
        for row in rows:
            grouped.setdefault(row["frame_index"], []).append(row)
        if exactly_four:
            grouped = {key: value for key, value in grouped.items() if len(value) == 4}
        sums = {name: [sum(row[name + "_ms"] for row in group) for group in grouped.values()]
                for name in ("old", "v1", "v2")}
        return dict(frames=len(grouped), calls_per_frame=4 if exactly_four else "variable",
                    **{name + "_sum_p50_ms": _median(values) for name, values in sums.items()})

    real, composite = selected("COMPLETE_REAL_49_FRAME_CORPUS"), selected(
        "COMPLETE_REAL_FRAME_PLUS_SYNTHETIC_V8_LATE_HISTORY")
    composite_four = per_decision(composite, exactly_four=True)
    ratio = composite_four["v2_sum_p50_ms"] / composite_four["v1_sum_p50_ms"]
    memory = {
        "original": _memory(lambda: OLD, memory_args),
        "v1": _memory(lambda: V1.payload_storage_bytes, memory_args),
        "v2": _memory(lambda: V2.AccountingSession(), memory_args),
    }
    checks = dict(
        all_complete_calls_equal=all(row["equal"] for row in calls),
        all_four_stages_present={row["caller"] for row in real} ==
                                {"compute_unbound", "stage_products", "bind_plan", "set_ready"},
        cancel_paths_restore=all(action["history_restored"] for action in actions if action["action"] != "accept"),
        late_four_call_ratio_accepted=ratio <= ACCEPTANCE["complete_late_four_call_sum_v2_to_v1_max"],
        bounded_candidate_cache=all(candidate.snapshot()["entries"] <= candidate.max_entries
                                    and candidate.snapshot()["snapshots"] <= candidate.max_snapshots
                                    for candidate in (candidate_real, candidate_composite)),
        switches_restored=(OWN.payload_storage_bytes is OLD and not PV2.installed() and not TILT.installed()),
    )
    summary = dict(
        schema=SCHEMA, status="PASS" if all(checks.values()) else "FAIL", checks=checks,
        acceptance_rule=ACCEPTANCE, candidate_version=V2.VERSION, cpu_clock="time.thread_time_ns",
        cpu_affinity="external taskset; see reproduction command", corpus_frames=len(actions),
        actions={name: sum(action["action"] == name for action in actions)
                 for name in ("accept", "hold", "unsent")},
        calls=len(calls), calls_per_structure=len(real), callers={name: sum(row["caller"] == name for row in real)
                                                                 for name in sorted({row["caller"] for row in real})},
        complete_real_corpus=dict(label="COMPLETE retained 49-frame corpus", per_call=per_call(real),
                                  per_decision=per_decision(real)),
        composite_late=dict(label="SYNTHETIC: complete retained frame fields with reconstructed V8 decision-1300 "
                                   "conflict history; not a historical complete frame",
                            decision=late_state["index"], pairs=late_state["pairs"],
                            observations=late_state["observations"], per_call=per_call(composite),
                            four_call_decisions=composite_four, v2_to_v1_sum_ratio=ratio),
        parity=dict(equal=sum(row["equal"] for row in calls), total=len(calls),
                    bytes_range=[min(row["old_bytes"] for row in calls), max(row["old_bytes"] for row in calls)]),
        memory=memory, candidate_cache=dict(real=candidate_real.snapshot(),
                                            composite=candidate_composite.snapshot()),
        limits="CPU-only offline; synthetic late history is for accounting structure/cost, not a complete historical frame")
    candidate_real.close(); candidate_composite.close()
    (out / "CALLS.json").write_text(json.dumps(dict(actions=actions, calls=calls), indent=1) + "\n")
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1, sort_keys=True))
    return summary["status"] == "PASS"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--late", type=int, default=1300)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return 0 if run(args.cell, args.late, args.out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
