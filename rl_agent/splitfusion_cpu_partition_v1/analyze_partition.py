#!/usr/bin/env python3
"""Analyze one partition pair using the retained Arm metrics and scheduler parser.

Usage: python analyze_partition.py EXPERIMENT_ROOT [--plots-only]
Inputs are read-only. Outputs are RESULT.json, per-arm tail CSVs, and PNG plots.
Scheduler execution coverage is compute/RU only; all-thread scope comes from
the apply journal and periodic affinity inventories, including future threads.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import csv
import importlib.util
import json
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2].parent / "abiodun_run4_single_gpu_profiler_v1"
PREVIOUS = WORKTREE / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261002T065135Z_compute_affinity_ab_v2"
sys.path.insert(0, str(WORKTREE))
from rl_agent.splitfusion_single_gpu_profiler_v1.compare_threadcap_ab_v1 import Arm, dist

_spec = importlib.util.spec_from_file_location("partition_retained_sched_tools", PREVIOUS / "sched_tools.py")
_scheduler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scheduler)
parse_perf = _scheduler.parse_perf


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, fields=None):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def overlap(a, b, c, d):
    return max(0, min(b, d) - max(a, c))


def placement(events, tid, lo, hi):
    start = cpu = None
    times, switches = collections.Counter(), collections.Counter()
    for event in events:
        if event["event"] != "sched:sched_switch":
            continue
        fields, now = event["fields"], event["ns"]
        if int(fields["prev_pid"]) == tid and start is not None:
            times[cpu] += overlap(start, now, lo, hi) / 1e6
            start = None
        if int(fields["next_pid"]) == tid:
            start, cpu = now, event["cpu"]
            if lo <= now <= hi:
                switches[cpu] += 1
    return dict(tid=tid, running_ms_by_cpu=dict(times), switch_ins_by_cpu=dict(switches),
                running_ms=sum(times.values()), coverage="Matched switch-in/out intervals within the live window")


def scope_result(directory, applied, window):
    spec = read(directory / "arm.json")
    journal = read(directory / "PARTITION_JOURNAL.json")
    plans = {str(row["pid"]): row for row in journal["processes"]}
    roles = applied["roles"]
    errors, newborn = [], {}
    baseline = {str(proc["pid"]): proc["threads"] for proc in applied["radio"]}
    edge_pid = str(applied["before"]["target"]["pid"])
    baseline[edge_pid] = applied["before"]["threads"]
    if collections.Counter(roles.values()) != {"edge": 1, "gnb": 3, "ue": 3}:
        errors.append(dict(reason="Expected exactly one edge, three gNB and three UE processes", roles=roles))
    if set(plans) != set(roles) or set(baseline) != set(roles):
        errors.append(dict(reason="Journal, role and baseline PID sets differ"))
    for pid, inventory in baseline.items():
        if not inventory or any(row["affinity"] != spec["original_cpus"] for row in inventory.values()):
            errors.append(dict(reason="Pre-application thread inventory is not the registered baseline", pid=pid))
    if set(spec["edge_cpus"]) & set(spec["radio_cpus"]):
        errors.append(dict(reason="Proposed edge and radio pools overlap"))
    if set(spec["edge_cpus"]) | set(spec["radio_cpus"]) != set(spec["original_cpus"]):
        errors.append(dict(reason="Partition does not cover the original CPU pool"))
    expected = {}
    for pid, role in roles.items():
        expected[pid] = sorted(spec["original_cpus"] if spec["arm"] == "control"
                               else spec["edge_cpus" if role == "edge" else "radio_cpus"])
        if pid not in plans or plans[pid]["target_cpus"] != expected[pid]:
            errors.append(dict(reason="Journal target disagrees with registered role mask", pid=pid))
    samples, first, last = 0, None, None
    tids = collections.defaultdict(set)
    for line in (directory / "target_samples.jsonl").read_text().splitlines():
        sample = json.loads(line)
        if not window[0] <= sample["raw_ns"] <= window[1]:
            continue
        samples += 1
        first = sample["raw_ns"] if first is None else first
        last = sample["raw_ns"]
        for issue in sample.get("errors", []):
            errors.append(dict(raw_ns=sample["raw_ns"], reason=issue))
        inventories = sample["processes"]
        if set(inventories) != set(roles):
            errors.append(dict(raw_ns=sample["raw_ns"], reason="Live inventory PID set differs", pids=sorted(inventories)))
        for pid, inventory in inventories.items():
            if pid not in expected:
                continue
            if not inventory:
                errors.append(dict(raw_ns=sample["raw_ns"], pid=pid, reason="Empty live thread inventory"))
            for tid, thread in inventory.items():
                tids[roles[pid]].add((pid, tid, thread["starttime"]))
                if thread["affinity"] != expected[pid]:
                    errors.append(dict(raw_ns=sample["raw_ns"], pid=pid, tid=tid,
                                       reason="Affinity differs from role mask", actual=thread["affinity"], expected=expected[pid]))
                old = baseline.get(pid, {}).get(tid)
                if old and old["starttime"] == thread["starttime"]:
                    if (old["policy"], old["priority"]) != (thread["policy"], thread["priority"]):
                        errors.append(dict(raw_ns=sample["raw_ns"], pid=pid, tid=tid, reason="Policy/priority changed"))
                else:
                    newborn.setdefault(f"{pid}:{tid}:{thread['starttime']}", dict(
                        pid=pid, tid=tid, role=roles[pid], first_seen_raw_ns=sample["raw_ns"], task=thread))
    footer = read(directory / "monitor_footer.json")
    restored = read(directory / "RESTORATION_LATEST.json")
    if samples == 0:
        errors.append(dict(reason="No affinity inventory within the live window"))
    elif first-window[0] > 2_500_000_000 or window[1]-last > 2_500_000_000:
        errors.append(dict(reason="Live-window endpoint lacks a nearby one-Hz affinity inventory"))
    return dict(scope="All threads of one edge and six radio processes; new threads intentionally included",
                roles=roles, expected_masks=expected, role_thread_identity_counts={k: len(v) for k, v in tids.items()},
                original=spec["original_cpus"], edge_cpus=spec["edge_cpus"], radio_cpus=spec["radio_cpus"],
                live_samples=samples, first_sample_raw_ns=first, last_sample_raw_ns=last,
                new_threads=newborn, scope_errors=errors, monitor=footer, restoration=restored,
                passed=not errors and applied["passed"] and footer.get("passed") is True and restored.get("passed") is True,
                limitation="One-Hz inventories do not prove every between-sample affinity or execution event; new-thread creator identities are not traced")


def radio_health(arm, cell, edge_report):
    with (arm.cell / "radio_trace.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    decisions = arm.ue.get("decisions", [])
    return dict(command_timing_status=dict(collections.Counter(row.get("command_timing_status", "") for row in rows)),
                radio_trace_rows=len(rows), command_latency_ms=dist(row.get("command_latency_ms") for row in rows),
                achieved_snr_db=dist(row.get("achieved_snr_db") for row in rows),
                target_snr_db=dist(row.get("target_snr_db") for row in rows),
                phase6=cell["phase6"], collector=cell["child"]["collector"],
                radio_cleanup=cell["cleanup"]["radio"], edge_counters=edge_report.get("counters"),
                evaluator=edge_report.get("evaluator"), ue_telemetry_counters=arm.ue.get("telemetry_counters"),
                ue_reader_counters=arm.ue.get("reader_counters"), ue_bridge_counters=arm.ue.get("bridge_counters"),
                decision_radio_reasons=dict(collections.Counter(str(reason) for row in decisions for reason in row.get("radio_reasons", []))),
                mcs_age_ms=dist(row["mcs_age_ns"] / 1e6 for row in decisions if row.get("mcs_age_ns") is not None),
                backlog_age_ms=dist(row["backlog_age_ns"] / 1e6 for row in decisions if row.get("backlog_age_ns") is not None))


def arm_result(root, label):
    directory = root / label
    arm = Arm(directory / "capture", label)
    result, (lo, hi) = arm.metrics(), arm.window
    applied, ready = read(directory / "AFFINITY_APPLIED.json"), read(directory / "worker_ready.json")
    target = applied["before"]["target"]
    tid = int(target["tid"])
    ru = [row["tid"] for proc in applied["radio"] if proc["comm"] == "nr-softmodem"
          for row in proc["threads"].values() if row["comm"] == "ru_thread"]
    assert len(ru) == 1, ru
    parsed = parse_perf(directory / "scheduler/sched.txt", tid)
    events, segments = parsed["events"], parsed["segments"]
    assert events, "Empty scheduler trace"
    ends = [segment["end"] for segment in segments]

    def partition(start, end):
        totals = dict(running=0, runnable=0, blocked=0, unknown=0)
        index = bisect.bisect_right(ends, start)
        while index < len(segments) and segments[index]["start"] < end:
            segment = segments[index]
            totals[segment["state"]] += overlap(start, end, segment["start"], segment["end"])
            index += 1
        totals["unknown"] += end - start - sum(totals.values())
        assert min(totals.values()) >= 0
        return {key: value / 1e6 for key, value in totals.items()}

    switches = {event["ns"]: event for event in events if event["event"] == "sched:sched_switch"
                and int(event["fields"]["prev_pid"]) == tid}
    episodes = [episode for episode in parsed["episodes"] if episode["prev_state"].startswith("R")]
    ru_next = [episode for episode in episodes if int(switches[episode["start"]]["fields"]["next_pid"]) in ru]
    frames = {frame["frame_id"]: frame for frame in arm.frames}
    tails = []
    for span in arm.spans:
        if span["stage"] != "edge.codec.frozen_tail" or span["phase"] != "live":
            continue
        assert int(span["tid"]) == ready["tid"], "Tail span compute identity mismatch"
        start, end = int(span["start_raw_ns"]), int(span["end_raw_ns"])
        frame, parts = frames.get(span["frame_id"], {}), partition(start, end)
        row = dict(arm=label, **{key: span[key] for key in ("frame_id", "tensor_seq", "session_uuid", "decision_seq", "ticket_seq", "mode_id", "q_e4")},
                   frame_kind=frame.get("frame_kind", span["frame_kind"]), start_raw_ns=start, end_raw_ns=end,
                   wall_ms=(end-start)/1e6, cuda_event_ms=float(span["cuda_ms"]) if span["cuda_ms"] else None,
                   **{key + "_ms": value for key, value in parts.items()},
                   ru_next_runnable_ms=sum(overlap(start, end, episode["start"], episode["end"]) for episode in ru_next)/1e6,
                   action_feedback_ms=float(frame["action_open_to_feedback_ms"]) if frame.get("action_open_to_feedback_ms") else None)
        candidates = [(overlap(start, end, episode["start"], episode["end"]), episode)
                      for episode in episodes if episode["start"] < end and episode["end"] > start]
        largest = max(candidates, key=lambda item: item[0]) if candidates else None
        details = switches[largest[1]["start"]]["fields"] if largest else {}
        row.update(largest_runnable_episode_overlap_ms=largest[0]/1e6 if largest else 0,
                   episode_out_cpu=largest[1]["out_cpu"] if largest else None,
                   episode_in_cpu=largest[1]["in_cpu"] if largest else None,
                   episode_next_tid=details.get("next_pid"), episode_next_comm=details.get("next_comm"),
                   episode_next_prio=details.get("next_prio"))
        assert abs(sum(parts.values()) - row["wall_ms"]) < 1e-6
        tails.append(row)
    assert tails, "No live frozen tails"
    write_csv(root / f"{label}_tails.csv", tails)
    write_csv(root / f"{label}_long_tails.csv", [row for row in tails if row["wall_ms"] > 100], list(tails[0]))
    scope = scope_result(directory, applied, arm.window)
    edge_place, ru_place = placement(events, tid, lo, hi), placement(events, ru[0], lo, hi)
    allowed_edge = set(scope["expected_masks"][str(target["pid"])] )
    radio_pid = next(str(proc["pid"]) for proc in applied["radio"] if str(ru[0]) in proc["threads"])
    allowed_ru = set(scope["expected_masks"][radio_pid])
    edge_place["outside_mask_running_ms"] = sum(value for cpu, value in edge_place["running_ms_by_cpu"].items() if cpu not in allowed_edge)
    ru_place["outside_mask_running_ms"] = sum(value for cpu, value in ru_place["running_ms_by_cpu"].items() if cpu not in allowed_ru)
    cell, edge_report = read(arm.cell / "CELL_RESULT.json"), read(arm.cell / "run4_phase6/edge_report.json")
    result.update(semantics=arm.analysis["semantics"], identity_checks=arm.analysis["identity"], cleanup=cell["cleanup"],
                  affinity=scope, monitor=scope["monitor"], radio_health=radio_health(arm, cell, edge_report),
                  tail_wall_ms=dist(row["wall_ms"] for row in tails),
                  tail_cuda_event_ms=dist(row["cuda_event_ms"] for row in tails),
                  tail_states_ms={key: dist(row[key+"_ms"] for row in tails) for key in ("running", "runnable", "blocked", "unknown", "ru_next_runnable")},
                  tail_counts=dict(n=len(tails), over_100_ms=sum(row["wall_ms"] > 100 for row in tails),
                                   over_170_ms=sum(row["wall_ms"] > 170 for row in tails),
                                   over_100_fraction=sum(row["wall_ms"] > 100 for row in tails)/len(tails)),
                  throughput_fps=result["workload"]["frames_sent"] / ((hi-lo)/1e9),
                  scheduler=dict(event_count=len(events), event_counts=dict(collections.Counter(event["event"] for event in events)),
                                 first_raw_ns=events[0]["ns"], last_raw_ns=events[-1]["ns"],
                                 parse_anomalies=parsed["parse_anomalies"], state_errors=parsed["state_errors"],
                                 tail_unknown_total_ms=sum(row["unknown_ms"] for row in tails),
                                 ru_next_runnable_total_ms=sum(row["ru_next_runnable_ms"] for row in tails),
                                 edge_placement=edge_place, ru_placement=ru_place,
                                 execution_coverage="Compute and gNB RU only; no complete placement trace for other edge/radio threads"))
    # Preserve the profiler's established distributions for every emitted stage.
    result["all_stage_durations"] = arm.analysis["durations"]
    result["workload"].update(mode_frame_counts=dict(collections.Counter(row["mode_id"] for row in arm.frames)),
                              mode_decision_counts=dict(collections.Counter(row["mode_id"] for row in arm.frames if row["frame_kind"] == "POLICY_DECISION")),
                              q_e4=dist(row["q_e4"] for row in arm.frames))
    decisions = [row for row in arm.frames if row["frame_kind"] == "POLICY_DECISION"]
    received = [row for row in decisions if row["feedback_receipt_raw_ns"]]
    timely = sum(0 <= int(row["feedback_receipt_raw_ns"])-int(row["action_open_raw_ns"]) <= 170_000_000 for row in received)
    primary = result["primary"]
    result["primary_raw_checks"] = dict(policy_decisions=len(decisions), received=len(received), missing=len(decisions)-len(received),
                                        inclusive_170ms_timely=timely,
                                        passed=(len(decisions) == primary["policy_decisions"] == primary["timely_feedback_le_170ms"]["n"]
                                                and len(decisions)-len(received) == primary["missing_feedback"]
                                                and timely == primary["timely_feedback_le_170ms"]["count"]))
    return result, tails


def comparison(control, treatment):
    def change(old, new):
        return new/old-1 if old not in (0, None) and new is not None else None
    def reduction(old, new):
        value = change(old, new)
        return -value if value is not None else None
    gain = 100*(treatment["primary"]["timely_feedback_le_170ms"]["fraction"]-control["primary"]["timely_feedback_le_170ms"]["fraction"])
    missing = 100*(treatment["primary"]["missing_feedback"]/treatment["primary"]["policy_decisions"]-control["primary"]["missing_feedback"]/control["primary"]["policy_decisions"])
    deltas = dict(timely_gain_pp=gain, missing_reply_change_pp=missing,
                  throughput_change_fraction=change(control["throughput_fps"], treatment["throughput_fps"]),
                  action_feedback_p50_change_fraction=change(control["primary"]["action_open_to_feedback_ms"]["p50"], treatment["primary"]["action_open_to_feedback_ms"]["p50"]),
                  map_p95_change_fraction=change(control["timing_ms"]["capture_to_map_install"]["p95"], treatment["timing_ms"]["capture_to_map_install"]["p95"]),
                  queue_p95_change_fraction=change(control["timing_ms"]["edge_queue"]["p95"], treatment["timing_ms"]["edge_queue"]["p95"]))
    mechanism = dict(runnable_p99_reduction=reduction(control["tail_states_ms"]["runnable"]["p99"], treatment["tail_states_ms"]["runnable"]["p99"]),
                     stall_rate_reduction=reduction(control["tail_counts"]["over_100_fraction"], treatment["tail_counts"]["over_100_fraction"]),
                     ru_next_runnable_mean_reduction=reduction(control["tail_states_ms"]["ru_next_runnable"]["mean"], treatment["tail_states_ms"]["ru_next_runnable"]["mean"]))
    flags = dict(missing=missing > 1,
                 throughput=deltas["throughput_change_fraction"] is None or deltas["throughput_change_fraction"] < -.05)
    for key, name, threshold in [("action_feedback_p50_change_fraction", "action_p50", .05),
                                 ("map_p95_change_fraction", "map_p95", .1), ("queue_p95_change_fraction", "queue_p95", .1)]:
        flags[name] = deltas[key] is None or deltas[key] > threshold
    def bad_command_rate(arm):
        health = arm["radio_health"]
        return sum(n for status, n in health["command_timing_status"].items() if status != "ACK_ON_TIME")/health["radio_trace_rows"] if health["radio_trace_rows"] else None
    command_rates = [bad_command_rate(arm) for arm in (control, treatment)]
    radio_flags = dict(command_failure_rate_increased=any(rate is None for rate in command_rates) or command_rates[1] > command_rates[0],
                       infrastructure_fault=any(arm["outcomes"]["terminals"].get("INFRASTRUCTURE_FAULT", 0) > 0 for arm in (control, treatment)),
                       lifecycle_failure=any(not arm["radio_health"]["radio_cleanup"].get("all_lifecycle_gates_passed")
                                             or not all(arm["radio_health"]["phase6"]["gates"].values()) for arm in (control, treatment)))
    flags["radio_health"] = any(radio_flags.values())
    validity_checks = {}
    for label, arm in (("control", control), ("treatment", treatment)):
        validity_checks[label] = dict(qualification=bool(arm["identity"]["gates"]) and all(arm["identity"]["gates"].values()),
                                      threadcap=bool(arm["identity"]["threadcap_checks"]) and all(arm["identity"]["threadcap_checks"].values()),
                                      scope=arm["affinity"]["passed"], primary_denominators=arm["primary_raw_checks"]["passed"],
                                      scheduler=not arm["scheduler"]["parse_anomalies"] and not arm["scheduler"]["state_errors"]
                                      and arm["scheduler"]["tail_unknown_total_ms"] == 0,
                                      recorded_placement=arm["scheduler"]["edge_placement"]["outside_mask_running_ms"] == 0
                                      and arm["scheduler"]["ru_placement"]["outside_mask_running_ms"] == 0)
    pair_masks_match = all(control["affinity"][key] == treatment["affinity"][key]
                          for key in ("original", "edge_cpus", "radio_cpus"))
    validity = all(all(checks.values()) for checks in validity_checks.values()) and pair_masks_match
    mechanism_pass = all(value is not None for value in mechanism.values()) and mechanism["runnable_p99_reduction"] >= .3 and mechanism["stall_rate_reduction"] >= .3 and mechanism["ru_next_runnable_mean_reduction"] > 0
    return dict(deltas=deltas, mechanism=mechanism, regression_flags=flags, radio_health_regression_flags=radio_flags,
                radio_command_failure_rates=dict(zip(("control", "treatment"), command_rates)),
                radio_health_check_scope="No new command-failure rate, infrastructure terminal, or lifecycle/integration failure; raw telemetry counters and age/latency distributions remain descriptive, not BLER",
                validity_checks=validity_checks, pair_registered_masks_match=pair_masks_match,
                validity_pass=validity, mechanism_thresholds_pass=mechanism_pass,
                timely_headline_threshold_pass=gain >= 5, regressions_pass=not any(flags.values()),
                strict_prospective_acceptance=validity and mechanism_pass and gain >= 5 and not any(flags.values()),
                status="PAIR_COMPLETED_VALID" if validity else "PAIR_COMPLETED_VALIDITY_FAILED",
                interpretation_limit="One sequential pair, temporal dependence, evolving workload/policy and no controlled tracer-overhead experiment. RU-next waits identify immediate successors, not exclusive RU occupancy. CUDA events are elapsed intervals, not kernel execution sums. Whole-runtime partition includes intentionally inherited new-thread masks.")


def plots(root, result, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"control": "#345995", "treatment": "#d45d30"}
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, key, title in zip(axes, ("wall_ms", "runnable_ms"), ("Frozen-tail wall time", "Compute runnable wait within tail")):
        for label in colors:
            values = sorted(float(row[key]) for row in rows[label])
            axis.step(values, [(i+1)/len(values) for i in range(len(values))], where="post", color=colors[label], label=label)
        axis.set(xlabel="Milliseconds", ylabel="Cumulative fraction", title=title, ylim=(0, 1.01))
        axis.grid(alpha=.2); axis.legend()
    fig.tight_layout(); fig.savefig(root / "01_tail_distributions.png"); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, sharey=True)
    for axis, label in zip(axes, colors):
        values = rows[label]; origin = int(values[0]["start_raw_ns"])
        times = [(int(row["start_raw_ns"])-origin)/1e9 for row in values]
        axis.plot(times, [float(row["wall_ms"]) for row in values], color=colors[label], lw=.8, label="Tail wall")
        axis.plot(times, [float(row["runnable_ms"]) for row in values], color="#293132", lw=.8, label="Runnable wait")
        axis.axhline(100, color="gray", ls="--", lw=.8)
        axis.set(title=label.capitalize(), ylabel="Milliseconds"); axis.grid(alpha=.2); axis.legend()
    axes[-1].set_xlabel("Seconds after first live edge tail")
    fig.tight_layout(); fig.savefig(root / "02_tail_timeline.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, key, title in zip(axes, ("edge_placement", "ru_placement"), ("Compute-thread execution frequency", "gNB RU execution frequency")):
        cpus = result["control"]["affinity"]["original"]
        for offset, label in enumerate(colors):
            values = {int(cpu): ms for cpu, ms in result[label]["scheduler"][key]["running_ms_by_cpu"].items()}
            total = sum(values.values())
            axis.bar([cpu+(offset-.5)*.4 for cpu in cpus], [100*values.get(cpu, 0)/total if total else 0 for cpu in cpus], width=.4, color=colors[label], label=label)
        axis.set(title=title, xlabel="Host CPU", ylabel="Share of recorded running time (%)")
        axis.set_xticks(cpus[::2]); axis.legend(); axis.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(root / "03_cpu_placement.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, key, title in zip(axes, ("wall_ms", "runnable_ms"), ("Frozen-tail wall frequency", "Compute runnable-wait frequency")):
        maximum = max(float(row[key]) for label in colors for row in rows[label])
        bins = [maximum*i/40 for i in range(41)] if maximum else [0, 1]
        for label in colors:
            values = [float(row[key]) for row in rows[label]]
            axis.hist(values, bins=bins, weights=[100/len(values)]*len(values), histtype="step", color=colors[label], label=label)
        axis.set(title=title, xlabel="Milliseconds", ylabel="Tails per bin (%)"); axis.legend(); axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(root / "04_tail_frequency.png"); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.experiment_root.resolve()
    if args.plots_only:
        result = read(root / "RESULT.json")
        rows = {}
        for label in ("control", "treatment"):
            with (root / f"{label}_tails.csv").open(newline="") as handle:
                rows[label] = list(csv.DictReader(handle))
    else:
        result, rows = {}, {}
        for label in ("control", "treatment"):
            print("Analyzing", label, flush=True)
            result[label], rows[label] = arm_result(root, label)
        result["comparison"] = comparison(result["control"], result["treatment"])
        result["analysis_sources"] = dict(arm_metrics=str(WORKTREE / "rl_agent/splitfusion_single_gpu_profiler_v1/compare_threadcap_ab_v1.py"), scheduler=str(PREVIOUS / "sched_tools.py"))
        write_json(root / "RESULT.json", result)
    plots(root, result, rows)
    print(json.dumps(result["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
