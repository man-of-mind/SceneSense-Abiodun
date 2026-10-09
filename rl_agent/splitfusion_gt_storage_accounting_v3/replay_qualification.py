"""(A) Accounting parity on every reconstructed history state of a capture (offline, CPU only, read-only inputs).

PARTIAL reconstruction, not a historical producer replay: the producer-v2 private ``IncrementalConflictState`` is
advanced over the capture's retained per-decision inputs (``splitfusion_gt_storage_accounting_v1.replay_parity``);
the replayed quarantine must equal the archived one on every decision. The accounted structure is the partial
staged part used by the V1/V2 qualifications (retained rows + derive result + rollback snapshot; no images), so the
real alias context between the derive result and the snapshot is present.

Per decision, the live caller order through the REAL ``ProducerOwner._account`` (functions named like the owner
stages, so the owner-shortcut logic runs exactly as in production):
  compute_unbound (None, part, None) -> stage_products (None, part, P) -> bind_plan (plan, part, P)
  -> set_ready (plan, part, P) + ready lengths,
where P is a small SYNTHETIC products stub that aliases three snapshot decision rows. Then two direct (non-owner)
calls in both alias orders: (derive result, snapshot) and (snapshot, derive result). Every call is evaluated by the
original (oracle), the unchanged V2 session and the V3 session, in rotated order; all integers must be equal.
Timings here are partial-structure (snapshot-dominated) and are reported separately from complete-call results.

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 taskset -c 4 /usr/bin/python3 -B -m \\
      rl_agent.splitfusion_gt_storage_accounting_v3.replay_qualification --cell <cell> --label V10 --out <new dir>
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN
from rl_agent.splitfusion_gt_storage_accounting_v1 import replay_parity as RP
from rl_agent.splitfusion_gt_storage_accounting_v2 import accounting as V2

from . import accounting as V3, switch

SCHEMA = "splitfusion.gt_storage_accounting_v3.replay_qualification.v1"
ORIGINAL = switch.ORIGINAL_PAYLOAD_STORAGE_BYTES
T = time.thread_time_ns
STAGES = ("compute_unbound", "stage_products", "bind_plan", "set_ready")
READY = SimpleNamespace(metadata_bytes=b"m" * 64, segmentation_bytes=b"s" * 128, object_ignore_bytes=b"o" * 16)


def _owner():
    owner = SimpleNamespace(_failure=None)
    owner._fail = lambda message, exc_type=OWN.OwnershipError: OWN.ProducerOwner._fail(owner, message, exc_type)
    return owner


def compute_unbound(owner, row):
    OWN.ProducerOwner._account(owner, row)


def stage_products(owner, row):
    OWN.ProducerOwner._account(owner, row)


def bind_plan(owner, row):
    OWN.ProducerOwner._account(owner, row)


def set_ready(owner, row):
    OWN.ProducerOwner._account(owner, row)


CALLERS = dict(compute_unbound=compute_unbound, stage_products=stage_products, bind_plan=bind_plan,
               set_ready=set_ready)


def run(cell, label, out):
    out.mkdir(parents=True, exist_ok=False)
    if not switch.is_original(ORIGINAL) or OWN.payload_storage_bytes is not ORIGINAL:
        raise RuntimeError("oracle is not the original ownership.payload_storage_bytes")
    v2, v3 = V2.AccountingSession(), V3.AccountingSession()
    if isinstance(ORIGINAL, (V2.AccountingSession, V3.AccountingSession)) or not v3.fast_path:
        raise RuntimeError("oracle/candidate identity or capability check failed")
    arms = (("original", ORIGINAL), ("v2", v2), ("v3", v3))
    rows, current, counter = [], {}, [0]

    def shadow(*values):
        offset = counter[0] % len(arms)
        counter[0] += 1
        results = {}
        for name, function in arms[offset:] + arms[:offset]:
            begin = T()
            value = function(*values)
            results[name] = (value, (T() - begin) / 1e6)
        current["calls"].append(dict(caller=current["caller"], **{n + "_bytes": v for n, (v, _t) in results.items()},
                                     **{n + "_ms": t for n, (_v, t) in results.items()},
                                     equal=len({v for v, _t in results.values()}) == 1))
        return results["original"][0]

    OWN.payload_storage_bytes = shadow
    try:
        for index, frame_id, metadata, result, snapshot, state, quarantine_ok, digest_ok in RP.states(cell):
            part = RP.staged_part(metadata, result, snapshot)
            products = ({"decision_rows": list(snapshot["decisions"].values())[:3]}, np.zeros(8))
            row = SimpleNamespace(plan=None, staged=part, products=None, ready=None, reservation=10 ** 12,
                                  payload_bytes=None)
            owner = _owner()
            current["calls"] = []
            for stage in STAGES:
                if stage == "stage_products":
                    row.products = products
                if stage == "bind_plan":
                    row.plan = b'{"plan":"' + str(frame_id).encode() + b'"}'
                if stage == "set_ready":
                    row.ready = READY
                current["caller"] = stage
                CALLERS[stage](owner, row)
            for direct in ((result, snapshot), (snapshot, result)):
                current["caller"] = "direct_" + ("result_first" if direct[0] is result else "snapshot_first")
                shadow(*direct)
            calls = current["calls"]
            rows.append(dict(index=index, frame_id=frame_id, quarantine_equal=int(quarantine_ok),
                             evidence_digest_equal=int(digest_ok), calls=len(calls),
                             equal=int(all(c["equal"] for c in calls)),
                             pairs=len(state.pairs), observations=sum(len(h.iou) for h in state.pairs.values()),
                             **{f"{c['caller']}_{arm}_ms": c[arm + "_ms"] for c in calls for arm, _f in arms},
                             **{f"{c['caller']}_bytes": c["original_bytes"] for c in calls}))
    finally:
        OWN.payload_storage_bytes = ORIGINAL
    v2_snapshot, v3_snapshot = v2.snapshot(), v3.snapshot()
    v2.close(); v3.close()
    with (out / "REPLAY_QUALIFICATION.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    def window(selected):
        return dict(decisions=len(selected), pairs_p50=statistics.median(r["pairs"] for r in selected),
                    observations_p50=statistics.median(r["observations"] for r in selected),
                    **{f"{stage}_{arm}_p50_ms": statistics.median(r[f"{stage}_{arm}_ms"] for r in selected)
                       for stage in STAGES for arm, _f in arms},
                    **{f"four_call_{arm}_p50_ms": statistics.median(
                        sum(r[f"{stage}_{arm}_ms"] for stage in STAGES) for r in selected) for arm, _f in arms})
    quarter = len(rows) // 4
    windows = dict(first_quarter=window(rows[:quarter]), last_quarter=window(rows[-quarter:]))
    checks = dict(accounting_equal_all_calls=all(r["equal"] for r in rows),
                  six_calls_per_decision=all(r["calls"] == 6 for r in rows),
                  quarantine_equal_all=all(r["quarantine_equal"] for r in rows),
                  v3_fast_path_used=v3_snapshot["snapshot_fast"] > 0,
                  v3_bounded=v3_snapshot["peak_memo_objects"] <= v3_snapshot["max_objects"],
                  v3_closed_empty=v3.memo_objects() == 0 and not v3._tuple_pins,
                  oracle_restored=OWN.payload_storage_bytes is ORIGINAL)
    summary = dict(schema=SCHEMA, label=label, status="PASS" if all(checks.values()) else "FAIL", checks=checks,
                   cell=str(cell), structure="PARTIAL_STAGED_PART (reconstructed history; not a historical producer "
                                             "replay; products = synthetic stub aliasing snapshot rows)",
                   decisions=len(rows), calls=sum(r["calls"] for r in rows),
                   equal_calls=sum(r["calls"] for r in rows if r["equal"]),
                   historical_evidence_digest_equal=sum(r["evidence_digest_equal"] for r in rows),
                   windows=windows, v2_session=v2_snapshot, v3_session=v3_snapshot,
                   timing_label="partial-structure, snapshot-dominated; NOT complete-call cost",
                   candidate_version=V3.VERSION)
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(dict(status=summary["status"], checks=checks, decisions=len(rows),
                          calls=summary["calls"], windows=windows), indent=1))
    return summary["status"] == "PASS"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return 0 if run(args.cell, args.label, args.out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
