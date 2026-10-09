"""Step 1: exact parity of deferred action binding on all 49 sealed v11 frames.

Run A: reference runtime producer vs deferred engine, every frame accepted.
Run B: deferred engine with holds, a fallback, a send failure after binding and a
refused non-template plan, versus a reference that never sees those frames.

Writes create-only evidence under experiments/splitfusion_exact_gt_service_proto_v2/.
Correctness only: no duration is recorded or interpreted.

    cd abiodun && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -m rl_agent.splitfusion_exact_gt_service_proto_v2.verify_deferred --name step1_deferred_01
"""
from __future__ import annotations

import argparse
import json

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C

from . import parity_common as P

COMPARED = ("identity_bytes", "semantic_sha256", "commit", "packet", "archive_files", "history_after")


def comparable(row):
    return dict(identity_sha256=P.sha(row["identity_bytes"]), semantic_sha256=row["semantic_sha256"],
                commit=row["commit"], packet_sha256=P.sha(row["packet"]), archive_files=row["archive_files"],
                archive_descriptor_sha256=P.sha(C.canonical(row["archive_descriptor"])),
                history_before=row["history_before"], history_after=row["history_after"])


def cancellation_plan(corpus):
    """Deterministic non-accepted subset, including first frame and a registration frame."""
    populations = [tuple(c.evidence["dynamic_snapshot_evidence"]["registered_population_ids"]) for c in corpus.cases]
    registration = [i for i in range(1, len(populations)) if populations[i] != populations[i - 1]]
    plan = {0: "hold", 3: "hold", 7: "unsent", 12: "refuse", 20: "hold", 21: "fallback", 30: "unsent", 41: "hold"}
    chosen = next((i for i in registration if i not in plan and i + 1 < len(populations)), None)
    if chosen is not None:
        plan[chosen] = "hold"
    return dict(sorted(plan.items())), registration, chosen


def refuse_plan(corpus, case):
    plan = C._parse(corpus.plan(case))
    plan["capture"] = dict(plan["capture"], action_id=71, profile_id="mode11_q9800_a71")
    return C.canonical(plan)


def run(name):
    P.environment_ok()
    corpus = P.Corpus(49)
    output = P.create_output(name)
    P.write_json(output / "PLAN.json", dict(schema="splitfusion.exact_gt_service_proto.step1_plan.v1",
        frames=len(corpus.cases), run_binding=corpus.binding, source_sha256=corpus.sources,
        reference="runtime process.Engine producer role in-process: STAGE_SOURCE, PREPARE_STAGED, CONFIRM_SEND, PUBLISH_READY",
        candidate="DeferredProducerEngine: prepare_unbound(decoded staged template) -> bind -> confirm_send -> publish_ready",
        mode="offline replay with recorded sync/AVO clocks (bit-exact convention of the sealed proofs)",
        archive_receipts="structural harness receipts, no fsync; retirement exercised, durability not claimed",
        timing_recorded=False, live=False, gpu=False))
    rows, failures = [], []
    try:
        # ---------- Run A: all accepted ----------
        ref = P.ReferenceDriver(corpus)
        ref_rows = {c.frame_id: ref.run(c) for c in corpus.cases}
        ref.close()
        cand = P.DeferredDriver(corpus)
        for case in corpus.cases:
            r = cand.run(case)
            a, b = comparable(ref_rows[case.frame_id]), comparable(r)
            exact = {k: a[k] == b[k] for k in a}
            retained = P.archive_checks(case, r["archive_raw"], r["archive_descriptor"])
            score = P.score_exact(corpus, case, r["packet"])
            ok = all(exact.values()) and all(retained.values()) and score["q_perc_exact"] \
                and score["measurement_exact"] and score["identity_key_exact"]
            rows.append(dict(run="A_all_accepted", frame_id=case.frame_id, exact_vs_reference=exact,
                retained_v11=retained, score=score, ok=ok, candidate=b))
            if not ok:
                failures.append(rows[-1])
                break
        snap_a = cand.engine.snapshot()
        cand.close()
        if failures:
            raise AssertionError("Run A parity failure at frame %s" % failures[0]["frame_id"])

        # ---------- Run B: cancellations never advance accepted history ----------
        plan, registration, chosen = cancellation_plan(corpus)
        accepted = [c for i, c in enumerate(corpus.cases) if i not in plan]
        ref = P.ReferenceDriver(corpus)
        sub_rows = {c.frame_id: ref.run(c) for c in accepted}
        ref.close()
        cand = P.DeferredDriver(corpus)
        for index, case in enumerate(corpus.cases):
            action = plan.get(index, "accept")
            r = cand.run(case, action=action, refuse_plan=refuse_plan(corpus, case) if action == "refuse" else None)
            if action == "accept":
                a, b = comparable(sub_rows[case.frame_id]), comparable(r)
                exact = {k: a[k] == b[k] for k in a}
                ok = all(exact.values())
                rows.append(dict(run="B_with_cancellations", frame_id=case.frame_id, action=action,
                                 exact_vs_reference_without_cancelled=exact, ok=ok))
            else:
                restored = r["history_after"] == r["history_before"]
                ok = restored and (action != "refuse" or r.get("refused") == "DeferralRefused")
                rows.append(dict(run="B_with_cancellations", frame_id=case.frame_id, action=action,
                                 history_restored=restored, refused=r.get("refused"),
                                 staged_history_unchanged=r.get("history_staged", r["history_before"]) == r["history_before"],
                                 ok=ok))
            if not ok:
                failures.append(rows[-1])
                break
        snap_b = cand.engine.snapshot()
        cand.close()
        if failures:
            raise AssertionError("Run B failure at frame %s" % failures[0]["frame_id"])

        a_rows = [r for r in rows if r["run"] == "A_all_accepted"]
        b_rows = [r for r in rows if r["run"] == "B_with_cancellations"]
        result = dict(schema="splitfusion.exact_gt_service_proto.step1_result.v1",
            status="STEP1_DEFERRED_BINDING_EXACT",
            run_a=dict(frames=len(a_rows), all_exact=all(r["ok"] for r in a_rows),
                       undefined_quality=sum(r["score"]["q_perc"] is None for r in a_rows),
                       compared_fields=sorted(a_rows[0]["exact_vs_reference"]),
                       retained_v11_checks=sorted(a_rows[0]["retained_v11"]),
                       engine_snapshot_deferral=snap_a["deferral"]),
            run_b=dict(frames=len(b_rows), cancellation_plan={str(corpus.cases[i].frame_id): a for i, a in plan.items()},
                       population_registration_frames=[corpus.cases[i].frame_id for i in registration],
                       cancelled_registration_frame=None if chosen is None else corpus.cases[chosen].frame_id,
                       accepted_exact=all(r["ok"] for r in b_rows if r["action"] == "accept"),
                       cancelled_history_restored=all(r["ok"] for r in b_rows if r["action"] != "accept"),
                       engine_snapshot_deferral=snap_b["deferral"]),
            timing_recorded=False, performance_claim=False, live_qualified=False)
        P.write_json(output / "RESULT.json", result)
        print(json.dumps({k: result[k] for k in ("status", "run_a", "run_b")}, indent=1, default=str)[:3000])
    except BaseException as error:
        P.write_json(output / "FAILURE.json", dict(error=f"{type(error).__name__}: {error}",
                                                   first_failure=failures[0] if failures else None, completed=len(rows)))
        raise
    finally:
        with (output / "FRAMES.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        P.seal_output(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    run(parser.parse_args().name)
