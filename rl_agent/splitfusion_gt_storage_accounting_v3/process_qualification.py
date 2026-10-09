"""(B) Complete-frame parity, owner-stage coverage, commit/cancel/rollback and complete-call CPU/wall/memory (offline).

Declared plan: ``experiments/splitfusion_gt_storage_accounting_v3/QUALIFICATION_PLAN.md`` (sealed before timing).

The real deferred engine (``parity_common.DeferredDriver``: producer V2 + tilt-aware AVO switches, as in the prep
worker) runs the retained 49-frame corpus with the V2 action schedule. ``ownership.payload_storage_bytes`` is a
shadow that, for every real owner call, evaluates in rotated order on the same arguments:
  original (oracle, identity-verified), V1, the unchanged V2 session and the V3 session
for three structures with one persistent session per arm and structure:
  COMPLETE_REAL            the retained complete frame;
  COMPOSITE_EARLY / _LATE  COMPOSITE (synthetic): the same complete frame with the reconstructed conflict history
                           (pre-derive snapshot + derive result) of decision W + 49 * pass + frame grafted in, so every
                           decision brings that decision's real history delta. Not a historical complete frame.
Passes 0..P-1 run the sequential driver; with ``--interleaved`` one more pass retires frame N-1 (score, feedback,
build_archive, durable) between frame N's FREEZE and BIND (correctness only). The V10 late composite also carries a
``v3_forced_fallback`` arm (max_objects=1; fallback cost, report only).

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    PYTHONDONTWRITEBYTECODE=1 taskset -c 4 /usr/bin/python3 -B -m \\
    rl_agent.splitfusion_gt_storage_accounting_v3.process_qualification --cell <cell> --label V10 \\
    --early 100 --late 1300 --passes 3 --interleaved --criteria --out <new dir>
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
from types import SimpleNamespace

from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN
from rl_agent.splitfusion_numeric_gt_producer_v1 import source as NS
from rl_agent.splitfusion_numeric_gt_producer_v2 import worker_switch as PV2
from rl_agent.splitfusion_avo_tilt_v1 import switch as TILT
from rl_agent.splitfusion_gt_storage_accounting_v1 import accounting as V1
from rl_agent.splitfusion_gt_storage_accounting_v1 import replay_parity as RP
from rl_agent.splitfusion_gt_storage_accounting_v1.process_parity import action_for
from rl_agent.splitfusion_gt_storage_accounting_v2 import accounting as V2

from . import accounting as V3, switch

SCHEMA = "splitfusion.gt_storage_accounting_v3.process_qualification.v1"
ORIGINAL = switch.ORIGINAL_PAYLOAD_STORAGE_BYTES
STAGES = ("compute_unbound", "stage_products", "bind_plan", "set_ready")
CRITERIA = dict(C1_first_call_cpu_p50_ratio_max=0.50, C2_four_call_cpu_p50_ratio_max=0.60,
                C3_each_stage_cpu_p50_v3_le_v2=True, C4_first_call_wall_p50_ratio_max=0.55,
                C4_four_call_wall_p50_ratio_max=0.65, C5_growth_ratio_max=0.5, C6_four_call_cpu_p95_v3_le_v2=True,
                C7_memory="memo<=max always; 0 after close; traced after close <= original + 64 KiB; "
                          "traced alive <= V2")
CPU, WALL = time.thread_time_ns, time.perf_counter_ns


def _q(values, q):
    values = sorted(values)
    return None if not values else values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def _p50(values):
    return statistics.median(values) if values else None


def _states(cell, wanted):
    found = {}
    for index, frame_id, metadata, result, snapshot, state, quarantine_ok, _digest in RP.states(cell):
        if index in wanted:
            if not quarantine_ok:
                raise RuntimeError(f"reconstructed decision {index} is not quarantine-exact")
            found[index] = dict(result=result, snapshot=snapshot, pairs=len(state.pairs),
                                observations=sum(len(h.iou) for h in state.pairs.values()))
        if len(found) == len(wanted):
            break
    if len(found) != len(wanted):
        raise RuntimeError("requested history decisions are not available in this capture")
    return found


def run(cell, label, early, late, passes, interleaved, criteria, out):
    out.mkdir(parents=True, exist_ok=False)
    if not switch.is_original(ORIGINAL) or OWN.payload_storage_bytes is not ORIGINAL:
        raise RuntimeError("oracle is not the original ownership.payload_storage_bytes")
    total_passes = passes + int(interleaved)
    windows = dict(COMPOSITE_EARLY=early, COMPOSITE_LATE=late)
    wanted = {start + 49 * p + k for start in windows.values() for p in range(total_passes) for k in range(49)}
    histories = _states(cell, wanted)
    structures = ("COMPLETE_REAL", "COMPOSITE_EARLY", "COMPOSITE_LATE")
    sessions = {s: dict(v2=V2.AccountingSession(), v3=V3.AccountingSession()) for s in structures}
    sessions["COMPOSITE_LATE"]["v3_forced_fallback"] = V3.AccountingSession(max_objects=1)
    if not all(session.fast_path for s in structures for name, session in sessions[s].items() if name == "v3"):
        raise RuntimeError("V3 fast-path capability missing")
    calls, actions, memo_trace = [], [], []
    state = dict(pass_index=None, schedule=None, frame_index=None, frame_id=None, counter=0)
    composites = {}                        # current frame's composite per structure only (frames hold images)
    memory = dict(key=None, current=[], sequence=None)   # only the latest complete late four-call decision

    def measure(structure, args, caller):
        arms = [("original", ORIGINAL), ("v1", V1.payload_storage_bytes)] + list(sessions[structure].items())
        offset = state["counter"] % len(arms)
        state["counter"] += 1
        measured = {}
        for name, function in arms[offset:] + arms[:offset]:
            wall, cpu = WALL(), CPU()
            value = function(*args)
            cpu, wall = CPU() - cpu, WALL() - wall
            measured[name] = (value, cpu / 1e6, wall / 1e6)
        v3 = sessions[structure]["v3"]
        memo_trace.append(v3.memo_objects())
        row = dict(structure=structure, pass_index=state["pass_index"], schedule=state["schedule"],
                   frame_index=state["frame_index"], frame_id=state["frame_id"], caller=caller,
                   **{n + "_bytes": v for n, (v, _c, _w) in measured.items()},
                   **{n + "_cpu_ms": c for n, (_v, c, _w) in measured.items()},
                   **{n + "_wall_ms": w for n, (_v, _c, w) in measured.items()},
                   equal=len({v for v, _c, _w in measured.values()}) == 1)
        calls.append(row)
        return row

    def shadow(*values):
        caller = sys._getframe(2).f_code.co_name
        real = measure("COMPLETE_REAL", values, caller)
        staged = values[1] if len(values) > 1 else None
        if isinstance(staged, NS.Staged):
            for structure, start in windows.items():
                decision = start + 49 * state["pass_index"] + state["frame_index"]
                composite = composites.get(structure)
                if composite is None or composite[0] is not staged:
                    history = histories[decision]
                    clone = copy.copy(staged)
                    clone.conflict = history["result"]
                    clone._restore = dict(staged._restore, conflict=history["snapshot"])
                    composites[structure] = composite = (staged, clone, decision)
                clone = composite[1]
                # The only Staged fields populated lazily by the real assembly path (as the V2 harness).
                clone.evidence, clone.inputs = staged.evidence, staged.inputs
                args = (values[0], clone, values[2])
                measure(structure, args, caller)
                if structure == "COMPOSITE_LATE":
                    key = (state["pass_index"], state["frame_index"])
                    if key != memory["key"]:
                        if [c for c, _a in memory["current"]] == list(STAGES):
                            memory["sequence"] = memory["current"]
                        memory.update(key=key, current=[])
                    memory["current"].append((caller, args))
        return real["original_bytes"]

    P.environment_ok()
    corpus = P.Corpus(49)
    PV2.install(); TILT.install(); OWN.payload_storage_bytes = shadow
    try:
        for pass_index in range(total_passes):
            schedule = "INTERLEAVED_RETIREMENT" if pass_index >= passes else "SEQUENTIAL"
            state.update(pass_index=pass_index, schedule=schedule)
            composites.clear()
            driver = P.DeferredDriver(corpus)
            try:
                pending = None
                for index, case in enumerate(corpus.cases):
                    state.update(frame_index=index, frame_id=case.frame_id)
                    begin, action = len(calls), action_for(index)
                    if schedule == "SEQUENTIAL":
                        result = driver.run(case, action=action)
                        restored = (result["history_after"] == result["history_before"]) if action != "accept" else None
                    else:
                        engine = driver.engine
                        before = P.history(engine)
                        handle = driver.unbound(case)
                        if pending is not None:                  # frame N-1 retires between FREEZE(N) and BIND(N)
                            driver.retire(*pending)
                            pending = None
                        if action == "hold":
                            engine.cancel_unbound(handle, "HOLD_NEVER_BOUND")
                        else:
                            prepared = engine.bind(handle, corpus.plan(case), frame_context=corpus.context(case))
                            if action == "unsent":
                                engine.cancel(prepared.token, corpus.plan(case), "SEND_FAILED_AFTER_BIND")
                            else:
                                engine.confirm_send(prepared.token, corpus.plan(case))
                                engine.publish_ready(prepared.token)
                                pending = (case, prepared)
                        restored = (P.history(engine) == before) if action != "accept" else None
                    actions.append(dict(pass_index=pass_index, schedule=schedule, index=index, frame_id=case.frame_id,
                                        action=action, accounting_rows=len(calls) - begin, history_restored=restored))
                if pending is not None:
                    driver.retire(*pending)
            finally:
                driver.close()
    finally:
        OWN.payload_storage_bytes = ORIGINAL
        TILT.uninstall(); PV2.uninstall()

    arms = ["original", "v1", "v2", "v3"]

    def select(structure, schedule="SEQUENTIAL"):
        return [r for r in calls if r["structure"] == structure and r["schedule"] == schedule]

    def stage_stats(rows):
        out = {}
        for stage in STAGES:
            chosen = [r for r in rows if r["caller"] == stage]
            out[stage] = dict(calls=len(chosen), **{
                f"{arm}_{clock}_{name}": fn([r[f"{arm}_{clock}_ms"] for r in chosen if f"{arm}_{clock}_ms" in r])
                for arm in arms + ["v3_forced_fallback"] for clock in ("cpu", "wall")
                for name, fn in (("p50", _p50), ("p95", lambda v: _q(v, .95)))
                if any(f"{arm}_{clock}_ms" in r for r in chosen)})
        return out

    def four_call(rows):
        grouped = {}
        for r in rows:
            grouped.setdefault((r["pass_index"], r["frame_index"]), []).append(r)
        groups = [g for g in grouped.values() if [r["caller"] for r in g] == list(STAGES)]
        result = dict(decisions=len(groups))
        for arm in arms + ["v3_forced_fallback"]:
            if not groups or f"{arm}_cpu_ms" not in groups[0][0]:
                continue
            for clock in ("cpu", "wall"):
                sums = [sum(r[f"{arm}_{clock}_ms"] for r in g) for g in groups]
                result[f"{arm}_{clock}_p50"] = _p50(sums)
                result[f"{arm}_{clock}_p95"] = _q(sums, .95)
        return result

    report = {}
    for structure in structures:
        rows = select(structure)
        report[structure] = dict(stages=stage_stats(rows), four_call=four_call(rows), calls=len(rows),
                                 equal=sum(r["equal"] for r in rows))
    interleaved_rows = [r for r in calls if r["schedule"] == "INTERLEAVED_RETIREMENT"]

    # Memory arm: one late composite four-call decision, replayed through stage-named owner functions.
    def owner_replay(function, sequence):
        owner = SimpleNamespace(_failure=None)
        owner._fail = lambda message, exc_type=OWN.OwnershipError: OWN.ProducerOwner._fail(owner, message, exc_type)
        row = SimpleNamespace(plan=None, staged=None, products=None, ready=None, reservation=10 ** 12,
                              payload_bytes=None)
        values = []
        saved = OWN.payload_storage_bytes
        OWN.payload_storage_bytes = function
        try:
            for caller, args in sequence:
                row.plan, row.staged, row.products = args
                MEMORY_CALLERS[caller](owner, row)
                values.append(row.payload_bytes)
        finally:
            OWN.payload_storage_bytes = saved
        return values

    if [c for c, _a in memory["current"]] == list(STAGES):
        memory["sequence"] = memory["current"]
    sequence = memory["sequence"]
    composites.clear()
    memory = {}
    for name, factory in (("original", lambda: ORIGINAL), ("v2", V2.AccountingSession), ("v3", V3.AccountingSession)):
        gc.collect()
        tracemalloc.start()
        function = factory()
        values = owner_replay(function, sequence)
        current, peak = tracemalloc.get_traced_memory()
        counters = function.snapshot() if hasattr(function, "snapshot") else None
        if hasattr(function, "close"):
            function.close()
        gc.collect()
        after_close = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory[name] = dict(values=values, traced_current_alive_bytes=current, traced_peak_bytes=peak,
                            traced_after_close_bytes=after_close, session=counters)
    memory_equal = memory["original"]["values"] == memory["v2"]["values"] == memory["v3"]["values"]

    late_v3 = sessions["COMPOSITE_LATE"]["v3"]
    session_reports = {s: {name: session.snapshot() for name, session in sessions[s].items()} for s in structures}
    for s in structures:
        for session in sessions[s].values():
            session.close()
    closed_empty = all(session.memo_objects() == 0 and not session._tuple_pins
                       for s in structures for name, session in sessions[s].items() if name.startswith("v3"))

    checks = dict(
        all_calls_equal=all(r["equal"] for r in calls),
        all_four_stages_exercised=all({r["caller"] for r in select(s)} == set(STAGES) for s in structures),
        interleaved_all_equal=(all(r["equal"] for r in interleaved_rows) and bool(interleaved_rows))
        if interleaved else True,
        cancel_paths_restore=all(a["history_restored"] for a in actions if a["action"] != "accept"),
        memory_arm_values_equal=memory_equal,
        switches_restored=OWN.payload_storage_bytes is ORIGINAL and not PV2.installed() and not TILT.installed(),
        v3_fast_path_used=session_reports["COMPOSITE_LATE"]["v3"]["snapshot_fast"] > 0)
    performance = None
    if criteria:
        late_r, early_r = report["COMPOSITE_LATE"], report["COMPOSITE_EARLY"]
        first = late_r["stages"]["compute_unbound"]
        four, four_early = late_r["four_call"], early_r["four_call"]
        growth_v2 = four["v2_cpu_p50"] - four_early["v2_cpu_p50"]
        growth_v3 = four["v3_cpu_p50"] - four_early["v3_cpu_p50"]
        values = dict(
            C1_first_call_cpu_p50_ratio=first["v3_cpu_p50"] / first["v2_cpu_p50"],
            C2_four_call_cpu_p50_ratio=four["v3_cpu_p50"] / four["v2_cpu_p50"],
            C3_stage_cpu_p50={stage: [late_r["stages"][stage]["v3_cpu_p50"], late_r["stages"][stage]["v2_cpu_p50"]]
                              for stage in STAGES},
            C4_first_call_wall_p50_ratio=first["v3_wall_p50"] / first["v2_wall_p50"],
            C4_four_call_wall_p50_ratio=four["v3_wall_p50"] / four["v2_wall_p50"],
            C5_growth_ms=dict(v2=growth_v2, v3=growth_v3, ratio=growth_v3 / growth_v2 if growth_v2 > 0 else None),
            C6_four_call_cpu_p95=[four["v3_cpu_p95"], four["v2_cpu_p95"]],
            C7=dict(max_memo_objects_seen=max(memo_trace), max_objects=late_v3.max_objects,
                    closed_empty=closed_empty,
                    after_close_minus_original=memory["v3"]["traced_after_close_bytes"]
                    - memory["original"]["traced_current_alive_bytes"],
                    alive=[memory["v3"]["traced_current_alive_bytes"], memory["v2"]["traced_current_alive_bytes"]]))
        verdicts = dict(
            C1=values["C1_first_call_cpu_p50_ratio"] <= CRITERIA["C1_first_call_cpu_p50_ratio_max"],
            C2=values["C2_four_call_cpu_p50_ratio"] <= CRITERIA["C2_four_call_cpu_p50_ratio_max"],
            C3=all(v3 <= v2 for v3, v2 in values["C3_stage_cpu_p50"].values()),
            C4=(values["C4_first_call_wall_p50_ratio"] <= CRITERIA["C4_first_call_wall_p50_ratio_max"]
                and values["C4_four_call_wall_p50_ratio"] <= CRITERIA["C4_four_call_wall_p50_ratio_max"]),
            C5=growth_v2 > 0 and growth_v3 <= CRITERIA["C5_growth_ratio_max"] * growth_v2,
            C6=values["C6_four_call_cpu_p95"][0] <= values["C6_four_call_cpu_p95"][1],
            C7=(values["C7"]["max_memo_objects_seen"] <= values["C7"]["max_objects"] and closed_empty
                and values["C7"]["after_close_minus_original"] <= 64 * 1024
                and values["C7"]["alive"][0] <= values["C7"]["alive"][1]))
        performance = dict(criteria=CRITERIA, values=values, verdicts=verdicts,
                           status="PASS" if all(verdicts.values()) else "FAIL")
    summary = dict(
        schema=SCHEMA, label=label, cell=str(cell), status="PASS" if all(checks.values()) else "FAIL",
        checks=checks, performance=performance, candidate_version=V3.VERSION, v2_version=V2.VERSION,
        settings=dict(cpu_clock="time.thread_time_ns", wall_clock="time.perf_counter_ns",
                      affinity=sorted(os.sched_getaffinity(0)), executable=sys.executable,
                      env={k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                         "OPENBLAS_NUM_THREADS", "PYTHONDONTWRITEBYTECODE")},
                      loadavg_at_end=open("/proc/loadavg").read().strip(), passes=passes, interleaved=interleaved,
                      windows=windows, rotation="all rotations of the arm order, advancing per measured call"),
        actions={schedule: {name: sum(a["action"] == name and a["schedule"] == schedule for a in actions)
                            for name in ("accept", "hold", "unsent")}
                 for schedule in sorted({a["schedule"] for a in actions})},
        history_windows={name: dict(first_decision=start, last_decision=start + 49 * total_passes - 1,
                                    pairs=[histories[start]["pairs"], histories[start + 49 * total_passes - 1]["pairs"]],
                                    observations=[histories[start]["observations"],
                                                  histories[start + 49 * total_passes - 1]["observations"]])
                         for name, start in windows.items()},
        structures=report,
        interleaved=dict(calls=len(interleaved_rows), equal=sum(r["equal"] for r in interleaved_rows),
                         stages=stage_stats([r for r in interleaved_rows if r["structure"] == "COMPOSITE_LATE"])),
        sessions=session_reports, memory=memory,
        labels=dict(COMPLETE_REAL="complete retained frame (short history)",
                    COMPOSITE="SYNTHETIC composite: complete frame + reconstructed (partial) history of a later "
                              "decision; not a historical complete frame",
                    timing="offline thread CPU on one pinned core; not a live prediction"))
    (out / "CALLS.json").write_text(json.dumps(dict(actions=actions, calls=calls), indent=0) + "\n")
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(dict(status=summary["status"], checks=checks, performance=performance,
                          four_call={s: report[s]["four_call"] for s in structures}), indent=1, default=str))
    return summary["status"] == "PASS" and (performance is None or performance["status"] == "PASS")


def compute_unbound(owner, row):
    OWN.ProducerOwner._account(owner, row)


def stage_products(owner, row):
    OWN.ProducerOwner._account(owner, row)


def bind_plan(owner, row):
    OWN.ProducerOwner._account(owner, row)


def set_ready(owner, row):
    OWN.ProducerOwner._account(owner, row)


MEMORY_CALLERS = dict(compute_unbound=compute_unbound, stage_products=stage_products, bind_plan=bind_plan,
                      set_ready=set_ready)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--early", type=int, default=100)
    parser.add_argument("--late", type=int, default=1300)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--interleaved", action="store_true")
    parser.add_argument("--criteria", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return 0 if run(args.cell, args.label, args.early, args.late, args.passes, args.interleaved, args.criteria,
                    args.out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
