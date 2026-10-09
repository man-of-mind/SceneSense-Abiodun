"""Accounting parity + evolving-sequence cost on the 1,786 V8 conflict-history states (offline, CPU only, read-only).

Reconstruction (as in ``experiments/splitfusion_m3_gt_prep_growth_diag_v1``): the producer-v2 private
``IncrementalConflictState`` is advanced over the retained per-decision ``input_metadata.json`` of the V8 capture in
FRAMES.jsonl order.  The replayed quarantine is required to equal the archived quarantine on every decision.  The
historical conflict ``evidence_sha256`` is NOT reproduced (known limitation, reported); this is not a full historical
GT replay.

The walked structure is PARTIAL, not a complete ``Staged`` frame (images and per-frame arrays were not retained).  It
contains exactly the conflict-owned and retained per-frame parts of what ``_account`` walks for that decision:

  STAGED_PART = {"frame", "boxes", "visibility", "static"   (retained per-decision rows),
                 "conflict"  (derive result: quarantine + evidence),
                 "_restore": {"ages": {}, "last": {}, "conflict": <NumericSourceProducer._conflict_snapshot body>}}

and is passed as ``(plan_bytes, STAGED_PART, None)`` like ``_account(row)`` before products exist.  For every decision
the old and candidate integers must be equal; both are timed (thread CPU), in alternating order, on the evolving
state.  Usage:

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 taskset -c 4 /usr/bin/python3 -B -m \\
      rl_agent.splitfusion_gt_storage_accounting_v1.replay_parity --cell <V8 cell dir> --out <new dir>
"""
import argparse
import csv
import json
from pathlib import Path
import time
import tracemalloc

from rl_agent.splitfusion_numeric_gt_producer_v1 import originals as O
from rl_agent.splitfusion_live_gt_optimization_v1 import source_geometry as V1S
from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as S, conflicts as CF
from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN

from . import accounting as A

OLD, NEW = OWN.payload_storage_bytes, A.payload_storage_bytes
T = time.thread_time_ns


def snapshot(s):  # literal NumericSourceProducer._conflict_snapshot body (source.py:526-531)
    return dict(templates=dict(s.templates), pairs=dict(s.pairs), decisions=dict(s.decisions),
                frame_count=s.frame_count, row_count=s.row_count, previous_frame=s.previous_frame,
                previous_timestamp=s.previous_timestamp, first_timestamp=s.first_timestamp, episode=s.episode,
                seen_samples=set(s.seen_samples), chain=s.chain)


def states(cell):
    """Yield (index, frame_id, metadata, derive result, pre-derive snapshot, state) over the retained sequence."""
    bound = O.bound()
    catalogue, _s, _e = S._load_catalogue(cell / "parity_static_catalogue")
    state = V1S._conflict_namespace(bound.adapter).IncrementalConflictState(catalogue, max_frames=100000)
    svc = cell / "numeric_gt" / "service"
    frames = sorted(json.loads(line)["frame_id"] for line in open(svc / "FRAMES.jsonl"))
    start = None
    for i, fid in enumerate(frames):
        m = json.load(open(svc / "archive" / f"frame_{fid}" / "producer" / "input_metadata.json"))
        stamp = float(m["frame"]["timestamp"])
        start = stamp if start is None else start
        snap = snapshot(state)
        provenance = {"scope": CF.SCOPE, "capture_identity": dict(m["provenance"]["capture_identity"]),
                      "complete": dict.fromkeys(CF.COMPLETENESS, True),
                      "catalogue_content_sha256": state.catalogue_digest,
                      "history_population": "EVERY_ACCEPTED_REWARD_FRAME_OF_THIS_PRODUCER",
                      "history_window": {"start_timestamp": start, "end_timestamp": stamp}}
        res = state.derive(frame=m["frame"], actor_boxes=m["actor_boxes"], actor_visibility=m["actor_visibility"],
                           static_candidates=m["static_candidates"], provenance=provenance)
        quarantine_equal = (bound.adapter.canonical(bound.adapter._json_value(res["quarantine"]))
                            == bound.adapter.canonical(bound.adapter._json_value(m["quarantine"])))
        digest_equal = res["evidence_sha256"] == m["provenance"]["evidence_sha256"]["conflicts"]
        yield i, fid, m, res, snap, state, quarantine_equal, digest_equal


def staged_part(m, res, snap):
    return {"frame": m["frame"], "boxes": m["actor_boxes"], "visibility": m["actor_visibility"],
            "static": m["static_candidates"], "conflict": res,
            "_restore": {"ages": {}, "last": {}, "conflict": snap}}


def timed(fn, args):
    t0 = T()
    value = fn(*args)
    return value, (T() - t0) / 1e6


def run(cell, out):
    out.mkdir(parents=True, exist_ok=False)
    rows, mem = [], []
    for i, fid, m, res, snap, state, q_ok, d_ok in states(cell):
        plan = b'{"plan":"' + str(fid).encode() + b'"}'
        args = (plan, staged_part(m, res, snap), None)
        if i % 2:
            new, t_new = timed(NEW, args); old, t_old = timed(OLD, args)
        else:
            old, t_old = timed(OLD, args); new, t_new = timed(NEW, args)
        rows.append(dict(i=i, frame_id=fid, quarantine_equal=int(q_ok), evidence_digest_equal=int(d_ok),
                         old_bytes=old, new_bytes=new, equal=int(old == new), old_ms=t_old, new_ms=t_new,
                         pairs=len(state.pairs), pair_observations=sum(len(p.iou) for p in state.pairs.values()),
                         seen_samples=len(state.seen_samples)))
        if i in (0, 100, 900, 1200, 1785):
            for name, fn in (("old", OLD), ("new", NEW)):
                tracemalloc.start()
                fn(*args)
                peak = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                mem.append(dict(i=i, function=name, transient_peak_bytes=peak))
    with open(out / "REPLAY_PARITY.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def med(sel, key):
        values = sorted(r[key] for r in sel)
        return values[len(values) // 2]
    early, late, tail = rows[:200], rows[1100:1700], rows[-30:]
    summary = dict(
        schema="splitfusion.gt_storage_accounting_v1.replay_parity.v1", structure="PARTIAL_STAGED_PART",
        decisions=len(rows), accounting_equal=sum(r["equal"] for r in rows),
        quarantine_equal=sum(r["quarantine_equal"] for r in rows),
        historical_conflict_evidence_digest_equal=sum(r["evidence_digest_equal"] for r in rows),
        cost_ms={name: dict(old_p50=med(sel, "old_ms"), new_p50=med(sel, "new_ms"), pairs=med(sel, "pairs"),
                            observations=med(sel, "pair_observations"))
                 for name, sel in (("decisions_0_199", early), ("decisions_1100_1699", late), ("last_30", tail))},
        transient_memory=mem, retained_cache_bytes=0, candidate_version=A.VERSION,
        limitation="partial structure (no images/arrays); historical conflict evidence digest not reproduced; "
                   "quarantine reproduced on every decision")
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    s = run(a.cell, a.out)
    raise SystemExit(0 if s["accounting_equal"] == s["decisions"] == s["quarantine_equal"] else 1)
