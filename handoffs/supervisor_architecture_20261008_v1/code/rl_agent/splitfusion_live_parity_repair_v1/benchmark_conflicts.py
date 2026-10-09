"""Reproducible CPU-only scaling check on repeated retained geometry.

Not a live latency or policy-performance measurement. Predictions, actors,
models, CUDA and services are never invoked. The repeated-frame construction
tests exact cumulative conflict semantics and bounds per-frame CPU work.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from rl_agent.splitfusion_live_target_parity_v1.fixtures import FIXTURES, load_retained_fixture
from . import conflicts as C


def run(*, frames=1000):
    if type(frames) is not int or not 1 <= frames <= 1000:
        raise ValueError("bounded 1..1000 CPU replay frames required")
    fixture, _ = load_retained_fixture(next(iter(FIXTURES)))
    path = A.ROOT / fixture.provenance["evidence_paths"]["static_catalogue"]
    if A.sha256(path) != fixture.provenance["evidence_sha256"]["static_catalogue"]:
        raise ValueError("retained catalogue changed")
    with path.open(newline="") as stream:
        catalogue = list(csv.DictReader(stream))
    cache = C.IncrementalConflictState(catalogue, max_frames=frames)
    history, elapsed, checkpoints = [], [], []
    selected = {i for i in (0, 10, 100, 500, frames-1) if i < frames}
    for index in range(frames):
        frame = dict(fixture.frame, sample_id=f"cpu_scaling_frame_{index}",
                     experiment_id="cpu_scaling_retained_geometry", frame_id=index+1, timestamp=float(index+1))
        def rebound(rows):
            return [dict(r, **{k:frame[k] for k in ("sample_id", "experiment_id", "frame_id", "timestamp")}) for r in rows]
        args = dict(frame=frame, actor_boxes=rebound(fixture.actor_boxes),
            actor_visibility=rebound(fixture.actor_visibility), static_candidates=rebound(fixture.static_candidates),
            provenance={"scope":C.SCOPE, "capture_identity":{"frame_id":index+1},
                "complete":dict.fromkeys(C.COMPLETENESS,True),
                "catalogue_content_sha256":C.catalogue_content_sha256(catalogue),
                "history_window":{"start_timestamp":1.,"end_timestamp":float(index+1)}})
        start = time.perf_counter_ns()
        observed = cache.derive(**args)
        incremental_ms = (time.perf_counter_ns()-start)/1e6
        elapsed.append(incremental_ms)
        if index in selected:
            start = time.perf_counter_ns()
            expected = C.derive_live_quarantine(**args, static_catalogue=catalogue,
                                               prior_static_candidates=history)
            original_ms = (time.perf_counter_ns()-start)/1e6
            same = (expected["quarantine"] == observed["quarantine"] and
                    expected["evidence"]["static_static_adjudications"] == observed["evidence"]["static_static_adjudications"] and
                    expected["evidence"]["static_dynamic_adjudications"] == observed["evidence"]["static_dynamic_adjudications"])
            if not same:
                raise AssertionError(f"original cumulative conflict mismatch at prefix {index+1}")
            checkpoints.append(dict(prior_frames=index, original_full_recompute_ms=original_ms,
                incremental_ms=incremental_ms, exact_quarantine_and_adjudications=True))
        history.extend(args["static_candidates"])
    source_files = [Path(__file__), Path(C.__file__)]
    return {"schema":"splitfusion.incremental_conflicts_cpu_benchmark.v1",
        "scope":"REPEATED_RETAINED_GEOMETRY_CPU_ONLY__NOT_LIVE_PERFORMANCE",
        "source_sha256":{str(p.relative_to(A.ROOT)):A.sha256(p) for p in source_files},
        "original_source_pins":A.verify_sources(),
        "fixture_sample":fixture.frame["sample_id"], "fixture_source_sha256":fixture.provenance["evidence_sha256"],
        "frames":frames, "catalogue_objects":len(catalogue),
        "static_candidates_per_frame":len(fixture.static_candidates), "dynamic_boxes":len(fixture.actor_boxes),
        "checkpoints":checkpoints, "incremental_frame_ms":elapsed,
        "incremental_ms_p50_p95_max":dict(zip(("p50","p95","max"), map(float,np.percentile(elapsed,[50,95,100])))),
        "history_pairs":len(cache.pairs), "observations":cache.row_count,
        "history_chain_sha256":cache.chain, "history_discarded":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames",type=int,default=1000)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("benchmark output is create-only")
    result=run(frames=args.frames)
    payload=A.canonical(result)
    with args.output.open("xb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"output":str(args.output),"sha256":hashlib.sha256(payload).hexdigest(),
                      "checkpoints":result["checkpoints"],"timing":result["incremental_ms_p50_p95_max"]},indent=2))


if __name__=="__main__":
    main()
