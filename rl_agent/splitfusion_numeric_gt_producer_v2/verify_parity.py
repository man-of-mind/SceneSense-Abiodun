"""Exactness proof of NumericSourceProducerV2 against v1 on the 49 retained v11 frames (CPU only).

  cd <abiodun>; env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /usr/bin/python3 -B -m rl_agent.splitfusion_numeric_gt_producer_v2.verify_parity --out <dir>

Pass A (all frames committed): per frame, v1 and v2 (alternating order) must give identical staged outputs (actor
boxes, visibility rows, static rows and summary, conflict quarantine/evidence/digest, training mask, vehicle mask,
complete-input digest, history before/after), identical target arrays/tables, AVO and payload digests, and both must
equal the RETAINED v11 digests (prepared target content, AVO document). Pass B: identical commit/cancel pattern
(cancel every 6th frame) on fresh producers; history digests, cancel/commit receipts and every output must agree.
Pass C: refusal boundaries (compute while pending, non-increasing frame) raise the same error in both and leave
history unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rl_agent.splitfusion_numeric_gt_producer_v1 import benchmark as B, cases as K, replay_v11 as RP, source as NS
from rl_agent.splitfusion_run5_live_avo_v1 import gt_source_sync as SY

from .source_v2 import NumericSourceProducerV2, package_sources


def canon_sha(value):
    return hashlib.sha256(RP.canon(value) if isinstance(RP.canon(value), bytes) else json.dumps(
        value, sort_keys=True, default=str).encode()).hexdigest()


def arr_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.dtype, a.shape)).encode() + a.tobytes()).hexdigest()


def staged_digest(s):
    return dict(boxes=canon_sha(s.boxes), visibility=canon_sha(s.visibility), static=canon_sha(s.static),
                static_summary=canon_sha(s.static_summary), conflict_evidence=s.conflict["evidence_sha256"],
                quarantine=canon_sha(s.conflict["quarantine"]), mask=arr_sha(s.mask), vehicle_mask=arr_sha(s.vehicle_mask),
                depth=arr_sha(s.depth), input_content=s.inputs.provenance["input_content_sha256"],
                history_before=s.history_before_sha256, history_after=s.history_after_sha256,
                corners=canon_sha(s.corners_evidence), evidence=canon_sha(s.evidence))


def run_frame(producer, case, *, commit=True):
    kw = {key: case.kwargs[key] for key in RP.KEYS}
    observed = case.evidence["source_synchronization"]["observed_raw_ns"]
    original = SY.source_alignment_record

    def alignment(*a, _o=observed, **k):
        value = original(*a, **k)
        value["observed_raw_ns"] = _o
        return value
    population = case.evidence["dynamic_snapshot_evidence"]
    if producer.population != frozenset(population["registered_population_ids"]):
        producer.register_population(population["registered_population_ids"], population["population_sha256"],
                                     before_frame_id=case.frame_id)
    captured = {}
    compute = producer.compute

    def capture(**k):
        captured["staged"] = compute(**k)
        captured["digest"] = staged_digest(captured["staged"])
        return captured["staged"]
    with patch.object(SY, "source_alignment_record", alignment):
        if commit:
            producer.compute = capture
            try:
                payload, prepared, documents = B.run_numeric(producer, case, kw, {})
            finally:
                del producer.compute
            return dict(staged=captured["digest"], prepared=prepared.content_sha256,
                        input_content=payload.input_content_sha256, avo=canon_sha(payload.avo),
                        avo_document=documents["document_sha256"], history=producer.history_sha256(),
                        retained_prepared=case.ready["prepared_content_sha256"], retained_avo=case.avo_sha256)
        staged = producer.compute(**kw)
        digest = staged_digest(staged)
        receipt = producer.cancel(staged)
        return dict(staged=digest, cancel=receipt, history=producer.history_sha256())


def run(out):
    RP.verify_environment()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cases = K.load_cases()
    args = K.producer_arguments(cases[0])
    report = dict(schema="splitfusion.numeric_gt_producer_v2.parity.v1", frames=len(cases), package=package_sources())
    # Pass A
    v1, v2 = NS.NumericSourceProducer(**args), NumericSourceProducerV2(**args)
    rows, tree_built, bbox = [], [], []
    for i, case in enumerate(cases):
        order = (v1, v2) if i % 2 == 0 else (v2, v1)
        res = {id(p): run_frame(p, case) for p in order}
        a, b = res[id(v1)], res[id(v2)]
        tree_built.append(v2.last_frame_stats["radar_tree_built"])
        bbox.append((v2.last_frame_stats["bbox_calls"], v2.last_frame_stats["bbox_computed"]))
        rows.append(dict(frame_id=case.frame_id, identical=a == b,
                         retained=a["prepared"] == a["retained_prepared"] and a["avo_document"] == a["retained_avo"]
                         and b["prepared"] == b["retained_prepared"] and b["avo_document"] == b["retained_avo"],
                         static_rows=json.loads(json.dumps(cases[i].evidence["static_summary"]))["projected_row_count"],
                         radar_tree_built=tree_built[-1], bbox_calls=bbox[-1][0], bbox_computed=bbox[-1][1]))
    report["pass_a"] = dict(identical=sum(r["identical"] for r in rows), retained_digests=sum(r["retained"] for r in rows),
                            frames=len(rows), radar_tree_built_frames=sum(tree_built),
                            bbox_calls_median=float(np.median([x for x, _ in bbox])),
                            bbox_computed_median=float(np.median([y for _, y in bbox])), rows=rows)
    # Pass B: commit/cancel pattern
    v1, v2 = NS.NumericSourceProducer(**args), NumericSourceProducerV2(**args)
    brow = []
    for i, case in enumerate(cases):
        commit = i % 6 != 2
        a, b = run_frame(v1, case, commit=commit), run_frame(v2, case, commit=commit)
        brow.append(dict(frame_id=case.frame_id, commit=commit, identical=a == b))
    report["pass_b"] = dict(identical=sum(r["identical"] for r in brow), frames=len(brow),
                            cancelled=sum(not r["commit"] for r in brow), rows=brow)
    # Pass C: refusals
    refusals = {}
    for name, cls in (("v1", NS.NumericSourceProducer), ("v2", NumericSourceProducerV2)):
        p = cls(**args)
        run_frame(p, cases[0])
        before = p.history_sha256()
        kw = {key: cases[1].kwargs[key] for key in RP.KEYS}
        observed = cases[1].evidence["source_synchronization"]["observed_raw_ns"]
        original = SY.source_alignment_record

        def alignment(*a, _o=observed, **k):
            value = original(*a, **k)
            value["observed_raw_ns"] = _o
            return value
        out_errors = []
        with patch.object(SY, "source_alignment_record", alignment):
            staged = p.compute(**kw)
            try:
                p.compute(**kw)
            except Exception as error:
                out_errors.append(f"{type(error).__name__}: {error}")
            p.cancel(staged)
            kw0 = {key: cases[0].kwargs[key] for key in RP.KEYS}
            try:
                p.compute(**kw0)                      # frame not after the last committed one
            except Exception as error:
                out_errors.append(f"{type(error).__name__}: {error}")
        refusals[name] = dict(errors=out_errors, history_unchanged=p.history_sha256() == before)
    report["pass_c"] = dict(identical=refusals["v1"] == refusals["v2"], **refusals)
    report["status"] = "PASS" if (report["pass_a"]["identical"] == report["pass_a"]["retained_digests"] == len(cases)
                                  and report["pass_b"]["identical"] == len(cases) and report["pass_c"]["identical"]
                                  and all(r["history_unchanged"] and len(r["errors"]) == 2 for r in refusals.values())) else "FAIL"
    (out / "PARITY.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: (v if k not in ("pass_a", "pass_b") else {kk: vv for kk, vv in v.items() if kk != "rows"})
                      for k, v in report.items() if k != "package"}, indent=1, default=str))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    run(p.parse_args().out)
