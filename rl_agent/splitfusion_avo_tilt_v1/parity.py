"""Upright parity and AVO-step cost on the 49 retained real frames (all pedestrians upright), old vs candidate.

Per frame, both ``numeric_avo`` implementations run on identical inputs (corpus source, producer depth decode,
owned camera matrices).  Score content is compared exactly (qualified ids/AVO/support/points/world xy, structural
ids/reasons/world xy, row geometry fields); metadata differences (version labels, ``avo_geometry``, extra row
fields) are listed separately.  The old result is also compared with the retained recorded AVO document.  Cost:
alternating repetitions per frame, per-call wall time (``CLOCK_MONOTONIC_RAW``); the median of per-frame paired
differences is reported.

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -m rl_agent.splitfusion_avo_tilt_v1.parity --name <dir>
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
from rl_agent.splitfusion_numeric_gt_producer_v1 import avo as NA, originals as O

from . import geometry as G, quality as Q, switch

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_avo_tilt_v1"
SCORE_QUALIFIED = ("gt_actor_id", "world_x", "world_y", "frame_id", "qualified", "avo", "no_support",
                   "retained_actor_point_count")
METADATA_ROW_FIELDS = set(Q.ROW_FIELDS)


def raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def arguments(case):
    kw = case.kwargs
    depth_raw = np.frombuffer(kw["depth_image"].raw_data, np.uint8).reshape(720, 1280, 4).copy()
    return dict(identity=kw["identity"], rgb=kw["rgb"], semantic_image=kw["semantic_image"],
                depth_image=kw["depth_image"], frozen_world=kw["frozen_world"],
                camera_matrix=np.asarray(kw["camera_matrix"]), camera_inverse=np.asarray(kw["camera_inverse"]),
                depth=O.bound().visibility.decode_depth_bgra(depth_raw), sync_validated=True)


def compare(old, new):
    oq = [{k: r[k] for k in SCORE_QUALIFIED} for r in old.avo["qualified"]]
    nq = [{k: r[k] for k in SCORE_QUALIFIED} for r in new.avo["qualified"]]
    rows_equal = len(old.rows) == len(new.rows) and all(
        {k: v for k, v in n.items() if k not in METADATA_ROW_FIELDS} == o for o, n in zip(old.rows, new.rows))
    score = dict(qualified_identical=oq == nq, structural_identical=old.avo["structural_ignored"] == new.avo["structural_ignored"],
                 threshold_identical=old.avo["threshold"] == new.avo["threshold"],
                 frame_timestamp_identical=(old.avo["frame_id"], old.avo["timestamp"]) == (new.avo["frame_id"], new.avo["timestamp"]),
                 row_geometry_identical=rows_equal)
    metadata = sorted({*(k for k in new.avo if k not in old.avo), *(k for k in old.avo if k not in new.avo)})
    labels = sorted({(o.get("algorithm_version"), n.get("algorithm_version"))
                     for o, n in zip(old.avo["qualified"], new.avo["qualified"])})
    return score, dict(top_level_keys_added_or_removed=metadata, qualified_version_labels=labels,
                       row_fields_added=sorted({k for r in new.rows for k in r} - {k for r in old.rows for k in r}))


def run(name, repetitions):
    P.environment_ok()
    if switch.installed():
        raise RuntimeError("parity compares both geometries; the process switch must be off")
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    corpus = P.Corpus(49)
    frames, deltas, old_ms, new_ms = [], [], [], []
    for case in corpus.cases:
        args = arguments(case)
        old, new = NA.numeric_avo(**args), Q.numeric_avo(**args)
        score, metadata = compare(old, new)
        recorded = case.avo_document["avo"]
        recorded_equal = (json.dumps(recorded["qualified"], sort_keys=True) == json.dumps(old.avo["qualified"], sort_keys=True)
                          and recorded["structural_ignored"] == old.avo["structural_ignored"])
        times = {"old": [], "new": []}
        for rep in range(repetitions):
            for label, function in ((("old", NA.numeric_avo), ("new", Q.numeric_avo)) if rep % 2 == 0
                                    else (("new", Q.numeric_avo), ("old", NA.numeric_avo))):
                begin = raw()
                function(**args)
                times[label].append((raw() - begin) / 1e6)
        o, n = statistics.median(times["old"]), statistics.median(times["new"])
        old_ms.append(o); new_ms.append(n); deltas.append(n - o)
        frames.append(dict(frame_id=case.frame_id, people=len(old.rows), qualified=len(old.avo["qualified"]),
                           score=score, metadata=metadata, old_equals_recorded_document=recorded_equal,
                           old_ms_median=o, new_ms_median=n))
        del args
    def pct(values, q):
        return float(np.percentile(np.asarray(values), q))
    checks = dict(
        score_identical_all_frames=all(all(f["score"].values()) for f in frames),
        old_reproduces_recorded_documents=all(f["old_equals_recorded_document"] for f in frames),
        metadata_differences_are_labels_only=all(
            f["metadata"]["top_level_keys_added_or_removed"] == ["avo_geometry"]
            and all(n == G.GEOMETRY_VERSION for _o, n in f["metadata"]["qualified_version_labels"])
            and f["metadata"]["row_fields_added"] == sorted(Q.ROW_FIELDS) for f in frames))
    cost = dict(repetitions_per_frame=repetitions, frames=len(frames),
                old_ms=dict(p50=pct(old_ms, 50), p95=pct(old_ms, 95)), new_ms=dict(p50=pct(new_ms, 50), p95=pct(new_ms, 95)),
                paired_delta_ms=dict(p50=pct(deltas, 50), p95=pct(deltas, 95), min=min(deltas), max=max(deltas)),
                paired_delta_relative_p50=pct([d / o for d, o in zip(deltas, old_ms)], 50))
    result = dict(schema="splitfusion.avo_tilt_v1.upright_parity.v1", status="PASS" if all(checks.values()) else "FAIL",
                  checks=checks, qualified_actor_frames=sum(f["qualified"] for f in frames),
                  people_rows=sum(f["people"] for f in frames), cost=cost, frames=frames,
                  geometry_version=G.GEOMETRY_VERSION, live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(
        {"RESULT.json": hashlib.sha256((out / "RESULT.json").read_bytes()).hexdigest()}, indent=1) + "\n")
    print(json.dumps(dict(status=result["status"], checks=checks, qualified=result["qualified_actor_frames"],
                          rows=result["people_rows"], cost=cost), indent=1))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--repetitions", type=int, default=6)
    args = parser.parse_args()
    run(args.name, args.repetitions)
