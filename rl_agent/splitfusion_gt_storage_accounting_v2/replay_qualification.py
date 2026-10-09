"""All-state parity and scaling on the partial reconstructed V8 history."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time

from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN
from rl_agent.splitfusion_gt_storage_accounting_v1 import accounting as V1
from rl_agent.splitfusion_gt_storage_accounting_v1 import replay_parity as RP

from . import accounting as V2

SCHEMA = "splitfusion.gt_storage_accounting_v2.replay_qualification.v1"
T = time.thread_time_ns


def _median(values):
    return statistics.median(values)


def run(cell, out):
    out.mkdir(parents=True, exist_ok=False)
    candidate = V2.AccountingSession()
    functions = (("original", OWN.payload_storage_bytes), ("v1", V1.payload_storage_bytes), ("v2", candidate))
    rows = []
    for index, frame_id, metadata, result, snapshot, state, quarantine_ok, digest_ok in RP.states(cell):
        plan = b'{"plan":"' + str(frame_id).encode() + b'"}'
        args = (plan, RP.staged_part(metadata, result, snapshot), None)
        timings, values = {name: [] for name, _function in functions}, []
        for repeat in range(4):
            offset = (index + repeat) % len(functions)
            for name, function in functions[offset:] + functions[:offset]:
                begin = T(); value = function(*args); elapsed = (T() - begin) / 1e6
                timings[name].append(elapsed); values.append(value)
        rows.append(dict(index=index, frame_id=frame_id, quarantine_equal=int(quarantine_ok),
                         evidence_digest_equal=int(digest_ok), equal=int(len(set(values)) == 1),
                         logical_bytes=values[0], pairs=len(state.pairs),
                         pair_observations=sum(len(history.iou) for history in state.pairs.values()),
                         **{name + "_per_call_p50_ms": _median(timings[name]) for name, _function in functions},
                         **{name + "_four_call_sum_ms": sum(timings[name]) for name, _function in functions}))
    candidate_snapshot = candidate.snapshot()
    candidate.close()
    with (out / "REPLAY_PARITY.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    def window(selected):
        return dict(decisions=len(selected), pairs_p50=_median([row["pairs"] for row in selected]),
                    observations_p50=_median([row["pair_observations"] for row in selected]),
                    **{name + "_per_call_p50_ms": _median([row[name + "_per_call_p50_ms"] for row in selected])
                       for name, _function in functions},
                    **{name + "_four_call_sum_p50_ms": _median([row[name + "_four_call_sum_ms"] for row in selected])
                       for name, _function in functions})
    windows = dict(early_0_199=window(rows[:200]), late_1100_1699=window(rows[1100:1700]),
                   tail_last_30=window(rows[-30:]))
    checks = dict(accounting_equal_all=sum(row["equal"] for row in rows) == len(rows),
                  quarantine_equal_all=sum(row["quarantine_equal"] for row in rows) == len(rows),
                  bounded_cache=candidate_snapshot["entries"] <= candidate_snapshot["max_entries"]
                                and candidate_snapshot["snapshots"] <= candidate_snapshot["max_snapshots"])
    summary = dict(schema=SCHEMA, status="PASS" if all(checks.values()) else "FAIL", checks=checks,
                   candidate_version=V2.VERSION, structure="PARTIAL_STAGED_PART",
                   calls_per_function_per_decision=4, decisions=len(rows), windows=windows,
                   scaling_samples=[rows[index] for index in (0, 100, 900, 1200, len(rows)-1)],
                   candidate_cache=candidate_snapshot,
                   remaining_growth=dict(
                       v1_four_call_late_over_early=(windows["late_1100_1699"]["v1_four_call_sum_p50_ms"] /
                                                     windows["early_0_199"]["v1_four_call_sum_p50_ms"]),
                       v2_four_call_late_over_early=(windows["late_1100_1699"]["v2_four_call_sum_p50_ms"] /
                                                     windows["early_0_199"]["v2_four_call_sum_p50_ms"])),
                   limitation="PARTIAL reconstruction: exact conflict history/quarantine and retained per-decision "
                              "rows, but no complete historical image/array frame; digest clocks are not reproduced")
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1, sort_keys=True))
    return summary["status"] == "PASS"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return 0 if run(args.cell, args.out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
