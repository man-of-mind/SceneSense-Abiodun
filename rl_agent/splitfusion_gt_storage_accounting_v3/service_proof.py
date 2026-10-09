"""Real-service old-vs-candidate proof of the exact storage-accounting optimisation (offline, CPU only; no CARLA server,
OAI, GPU or radio).

Every arm runs UE seam -> GT service trampoline -> real ``exact_gt_live_v2`` service -> GT worker trampolines -> real
prep and score workers, registered M3 limits, GC arm ``candidate``, producer v2.  Paired arms use IDENTICAL retained
inputs and scenario; the only difference is the path:

  v8_*         the V8 deployment path (tilt-aware AVO + V7 FREEZE lifetime repair), unchanged
  candidate_*  this package: prep installs the accounting switch, then the UNCHANGED V8 worker trampoline

  *_normal     49-frame retained corpus, v3 rehearsal scenario (11 scored, send-failure cancel, edge terminal,
               horizon timeout then late prediction)
  *_tilt       synthetic tilts on frames 1531/1535/1539 (live V7 tilt, 6 deg, sign-flipped) + upright 1527
  *_envelope   upright 1527 then a 20-deg target on 1543 -> fail closed with the original error + FREEZE snapshot

  v8_normal_repeat  a second V8 normal arm: the run-to-run baseline on identical inputs

The live service embeds two monotonic clock readings in the GT source evidence
(``source_synchronization.observed_raw_ns``, ``sources/frozen_world/snapshot_observed_raw_ns``); every digest computed
over that evidence (``owned_source`` -> producer/ready semantic digests -> scorer proof, feedback packet) therefore
differs between ANY two live runs, V8 against V8 included.  Byte-identity of those digests is proven separately in
offline replay with recorded clocks (``output_parity``).  Here, per pair:
* identical outcomes, q, feedback kind, commit/history digests, measurement digests, exact rescoring, and scores exact
  against the in-process reference producer with the same (tilt-aware) geometry;
* identical bytes for every clock-free archive file (masks, labels, AVO);
* every other differing leaf (after order-normalising the asynchronous ``events``/``requests`` lists) is either a
  clock reading (``TIMING_LEAF``) or a declared clock-derived digest/pid (``CLOCK_DERIVED``), and its pattern also
  differs between the two V8 runs; the V8-vs-V8 baseline itself may contain only those two classes.
Anything else fails the proof.

  cd <checkout>; env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -m rl_agent.splitfusion_gt_storage_accounting_v3.service_proof --name <dir>
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as CC
from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
from rl_agent.splitfusion_exact_gt_live_v2 import contract as EC
from rl_agent.splitfusion_exact_gt_live_v2.edge import install_early_gt_edge, EarlyGtEvaluatorV1
from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import quality_bridge
from rl_agent.splitfusion_m3_route_b_v7.freeze_service_proof import Rig, _normal, _json
from rl_agent.splitfusion_m3_route_b_v8 import contract as V8C, gc_seams as V8_SEAM
from rl_agent.splitfusion_avo_tilt_v1 import switch as TILT
from rl_agent.splitfusion_avo_tilt_v1.fixtures import TiltCorpus
from rl_agent.splitfusion_avo_tilt_v1.service_proof import TILTS, TILT_SCENARIO, ENVELOPE_SCENARIO, _scored_sequence

from . import accounting as A, contract as C, gc_seams as SEAM

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_gt_storage_accounting_v3"
BUDGET = C.FRAME_CEILING
ARM, PRODUCER = "candidate", "v2"
PATHS = dict(v8=(V8C, V8_SEAM), candidate=(C, SEAM))
ENVELOPE_ERROR = ("WorkerFailed: prep FREEZE: LiveGtContractError: tilt-aware AVO v1 does not support actor 101: "
                  "TILT_OUTSIDE_GROUND_RULE_ENVELOPE_TARGET")
# Clock readings / CPU durations of THIS run (exact leaf names observed in service_proof_02).
TIMING_LEAF = re.compile(r"(raw_ns|thread_cpu_ms|thread_cpu_ns)$|/requests\[\d+\]/(begin|end)$")
# Digests computed over the clock-bearing live source evidence, and process ids (declared; see module docstring).
CLOCK_DERIVED = re.compile(r"(/provenance/evidence_sha256/owned_source|/producer_semantic_sha256|/ready_semantic_sha256"
                           r"|producer_events\.json/semantic_sha256|producer_events\.json/pid|/semantic|:semantic"
                           r"|/scorer_proof/(archive_manifest_sha256|ready_semantic_sha256|receipt_sha256)"
                           r"|/clock_sha256|/packet_sha256)$")
CLOCK_FREE_FILES = ("producer/numerical_avo.json", "scorer/segmentation.u8", "scorer/object_ignore.u8",
                    "scorer/predicted_labels.u8")


def _record(path):
    """(document or None, state): a fatally failed prep worker can be SIGTERMed while writing its exit records
    (known V7 limitation), leaving a record missing or truncated; that is reported, never parsed as valid."""
    path = Path(path)
    if not path.exists():
        return None, "missing"
    try:
        return json.loads(path.read_text()), "ok"
    except ValueError:
        return None, f"truncated_{path.stat().st_size}_bytes"


def arm_run(out, corpus, limits, spec, *, path, scenario, expect_failure):
    contract, seam = PATHS[path]
    out.mkdir()
    registration = dict(id=contract.ID, transmitted_budget=BUDGET, gt_gc=contract.gc_registration(ARM, BUDGET, PRODUCER))
    state, rig = {}, None
    with seam.install_ue_service_seam(registration=registration, output=out) as trace_dir:
        try:
            rig = Rig(out, corpus, limits, contract.exact_sources(ROOT), BUDGET)
            with install_early_gt_edge(socket_path=rig.evidence / EC.SOCKET_NAME,
                                       capability_path=rig.evidence / EC.CAPABILITY_NAME, connect_timeout_s=30):
                rig.ev = EarlyGtEvaluatorV1(spec=spec, read_ground_truth=None, send=rig.payloads.put,
                                            match_distance_m=2., gt_timeout_s=2.)
                rig.ev.start()
                run = _normal(rig, limits) if scenario == "normal" else _scored_sequence(rig, scenario, state)
                try:
                    rig.ev.close()
                except BaseException as error:
                    if not expect_failure:
                        raise
                    state["edge_close_error"] = f"{type(error).__name__}: {error}"[:300]
            try:
                ue_result = rig.h.close()
            except BaseException as error:
                if not expect_failure:
                    raise
                state["close_error"], ue_result = f"{type(error).__name__}: {error}"[:600], None
        except BaseException as error:
            if rig is not None and rig.h.process is not None and rig.h.process.poll() is None:
                rig.h.abort(error)
            raise
    service_dir = rig.ue_dir / "numeric_gt/service"
    rows = [json.loads(line) for line in open(service_dir / "FRAMES.jsonl")] if (service_dir / "FRAMES.jsonl").exists() else []
    service = _json(service_dir / "SERVICE_RESULT.json") or {}
    from rl_agent.splitfusion_fixed_action_screen_analysis_v2.analyze import rescore
    index = {c.frame_id: i for i, c in enumerate(rig.cases)}
    frames = {}
    TILT.install()                                # both paths are tilt-aware; the reference uses the same geometry
    try:
        ref = P.AnchorReferenceDriver(corpus)
        try:
            for row in rows:
                fid = row["frame_id"]
                fb = rig.feedback.get(fid) or {}
                entry = dict(outcome=row.get("outcome"), q_service=(row.get("result") or {}).get("q_perc"),
                             q_emitted=fb.get("q_perc"), feedback=fb.get("kind"), committed=bool(row.get("committed")),
                             commit=row.get("commit"), semantic=row.get("semantic"), identity=row.get("identity"))
                scorer_dir = service_dir / "archive" / f"frame_{fid}" / "scorer"
                if row.get("outcome") == "SCORED":
                    case = rig.cases[index[fid]]
                    reference = P.score_exact(corpus, case, ref.run(case)["packet"])
                    score = json.loads((scorer_dir / "numerical_score.json").read_bytes())
                    entry.update(q_reference=reference["q_perc"],
                                 exact=entry["q_service"] == reference["q_perc"] == fb.get("q_perc"),
                                 archive_rescore_exact=rescore(scorer_dir),
                                 measurement_sha256=CC.sha256(CC.canonical(score["measurement"])),
                                 measurement_reference_sha256=reference["measurement_sha256"])
                frames[str(fid)] = entry
        finally:
            ref.close()
    finally:
        TILT.uninstall()
    records = {n: _record(trace_dir / n) for n in ("prep_gc_trace.json", "score_gc_trace.json", "service_substitution.json",
                                                   C.PREP_BINDING_RECORD, C.AVO_BINDING_RECORD,
                                                   C.ACCOUNTING_BINDING_RECORD)}
    docs = {n: v[0] for n, v in records.items()}
    snapshot_dir = trace_dir / C.FAILURE_SNAPSHOT_DIR
    snapshot = None
    if (snapshot_dir / "READY.json").exists():
        ready = json.loads((snapshot_dir / "READY.json").read_bytes())
        error = json.loads((snapshot_dir / "ERROR.json").read_bytes())
        snapshot = dict(hashes_verify=all(hashlib.sha256((snapshot_dir / n).read_bytes()).hexdigest() == d
                                          for n, d in ready["sha256"].items()),
                        exception_type=error["exception_type"], exception=error["exception"],
                        frame_id=json.loads(json.loads((snapshot_dir / "REQUEST.json").read_bytes())["descriptor"])["frame_id"])
    ops = {}
    for o in (docs["prep_gc_trace.json"] or {}).get("ops", []):
        if o.get("cpu_ns") is not None:
            ops.setdefault(o["op"], []).append(o["cpu_ns"] / 1e6)
    sub = docs["service_substitution.json"] or {}
    return dict(path=path, frames=frames, rows=rows, scenario=run, state=state, errors=list(service.get("errors") or []),
                outcomes=service.get("outcomes"), service_returncode=rig.h.process.returncode,
                ue_result_ok=ue_result is not None, service_dir=str(service_dir),
                worker_entries=sorted({a[a.index("-m") + 1] for a in (s.get("argv", []) for s in sub.get("substitutions", []))
                                       if "-m" in a}),
                prep_lifetime=docs[C.PREP_BINDING_RECORD], avo_binding=docs[C.AVO_BINDING_RECORD],
                accounting_binding=docs[C.ACCOUNTING_BINDING_RECORD], snapshot=snapshot,
                record_states={n: v[1] for n, v in records.items()},
                prep_cpu_ms_p50={k: statistics.median(v) for k, v in ops.items()})


def _files(service_dir, fid):
    root = Path(service_dir) / "archive" / f"frame_{fid}"
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()} if root.exists() else {}


def _normalise(value):
    """Order-normalise the asynchronous event/request logs (their order is a race between threads/processes)."""
    if isinstance(value, dict):
        out = {k: _normalise(v) for k, v in value.items()}
        for key, order in (("events", ("name", "role", "kind", "reason")), ("requests", ("worker", "op", "status"))):
            if isinstance(out.get(key), list) and all(isinstance(e, dict) for e in out[key]):
                out[key] = sorted(out[key], key=lambda e: tuple(str(e.get(k)) for k in order))
        return out
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    return value


def _leaf_diffs(a, b):
    try:
        return P.json_diff(_normalise(json.loads(a)), _normalise(json.loads(b)))
    except (ValueError, UnicodeDecodeError):
        return ["<binary differs>"]


def _classify(leaf):
    return "timing" if TIMING_LEAF.search(leaf) else "clock_derived" if CLOCK_DERIVED.search(leaf) else "VALUE"


def pattern(leaf):
    return re.sub(r"\[\d+\]", "[i]", re.sub(r"frame_\d+|FRAMES\[\d+\]", "*", leaf))


def compare(old, new):
    """Every difference between two arms on identical inputs: (leaf, class)."""
    diffs = []
    keys = ("outcome", "q_service", "q_emitted", "feedback", "committed", "commit", "identity",
            "measurement_sha256", "exact", "archive_rescore_exact")
    if sorted(old["frames"]) != sorted(new["frames"]):
        diffs.append(("frames", "VALUE"))
    for fid in sorted(set(old["frames"]) & set(new["frames"])):
        for k in keys:
            if old["frames"][fid].get(k) != new["frames"][fid].get(k):
                diffs.append((f"frame_{fid}:{k}", "VALUE"))
        if old["frames"][fid].get("semantic") != new["frames"][fid].get("semantic"):
            diffs.append((f"frame_{fid}:semantic", "clock_derived"))
        fo, fn = _files(old["service_dir"], fid), _files(new["service_dir"], fid)
        if sorted(fo) != sorted(fn):
            diffs.append((f"frame_{fid}:archive_file_set", "VALUE"))
        for name in sorted(set(fo) & set(fn)):
            if fo[name] != fn[name]:
                if name in CLOCK_FREE_FILES:
                    diffs.append((f"frame_{fid}/{name}", "VALUE"))
                    continue
                for leaf in _leaf_diffs(fo[name], fn[name]):
                    diffs.append((f"frame_{fid}/{name}{leaf}", _classify(f"frame_{fid}/{name}{leaf}")))
    rows_old = {r["frame_id"]: _normalise(r) for r in old["rows"]}
    rows_new = {r["frame_id"]: _normalise(r) for r in new["rows"]}
    for fid in sorted(set(rows_old) | set(rows_new)):
        for leaf in P.json_diff(rows_old.get(fid, {}), rows_new.get(fid, {})):
            diffs.append((f"FRAMES[{fid}]{leaf}", _classify(f"FRAMES[{fid}]{leaf}")))
    return diffs


def run(name):
    P.environment_ok()
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    limits = C.limits(BUDGET)
    spec, _ = quality_bridge()
    base, tilted = P.Corpus(49), TiltCorpus(TILTS)
    f = {i: tilted.cases[i].frame_id for i in (19, 20, 21, 22, 23)}
    arms, result = {}, None
    try:
        for kind, corpus, scenario, failure in (("normal", base, "normal", False),
                                                ("tilt", tilted, TILT_SCENARIO, False),
                                                ("envelope", tilted, ENVELOPE_SCENARIO, True)):
            for path in ("v8", "candidate"):
                arms[f"{path}_{kind}"] = arm_run(out / f"{path}_{kind}", corpus, limits, spec, path=path,
                                                 scenario=scenario, expect_failure=failure)
        arms["v8_normal_repeat"] = arm_run(out / "v8_normal_repeat", base, limits, spec, path="v8", scenario="normal",
                                           expect_failure=False)
        baseline = compare(arms["v8_normal"], arms["v8_normal_repeat"])
        baseline_patterns = {pattern(d[0]) for d in baseline}
        pairs = {kind: compare(arms[f"v8_{kind}"], arms[f"candidate_{kind}"]) for kind in ("normal", "tilt", "envelope")}
        value_diffs = {k: [d for d in v if d[1] == "VALUE"] for k, v in pairs.items()}
        outside_baseline = {k: sorted({pattern(d[0]) for d in v if d[1] != "VALUE"} - baseline_patterns)
                            for k, v in pairs.items()}

        def scored(arm):
            return {k: v for k, v in arm["frames"].items() if v["outcome"] == "SCORED"}

        def exact(arm):
            return bool(scored(arm)) and all(v["exact"] and v["archive_rescore_exact"]
                                              and v["measurement_sha256"] == v["measurement_reference_sha256"]
                                              for v in scored(arm).values())
        cand = [arms[f"candidate_{k}"] for k in ("normal", "tilt", "envelope")]
        v8 = [arms[f"v8_{k}"] for k in ("normal", "tilt", "envelope", "normal_repeat")]
        cn, ct, ce = cand
        vn, vt, ve, _vr = v8
        checks = dict(
            baseline_only_clock_classes=bool(baseline) and all(d[1] != "VALUE" for d in baseline),
            no_value_difference=all(not v for v in value_diffs.values()),
            candidate_differences_within_v8_run_to_run_baseline=all(not v for v in outside_baseline.values()),
            normal_both_exact_11_scored=exact(vn) and exact(cn) and len(scored(vn)) == len(scored(cn)) == 11,
            normal_cancel_and_terminal_paths_identical=all(
                vn["frames"][str(x)]["outcome"] == cn["frames"][str(x)]["outcome"]
                and vn["frames"][str(x)]["commit"] == cn["frames"][str(x)]["commit"]
                for x in (vn["scenario"]["send_failure_cancel"], vn["scenario"]["edge_terminal_before_prediction"],
                          vn["scenario"]["horizon_timeout_then_late_prediction"])),
            tilt_both_all_scored_exact=all(sorted(scored(a)) == sorted(str(f[i]) for i in TILT_SCENARIO) and exact(a)
                                           for a in (vt, ct)),
            envelope_both_fail_closed_with_original_error=all(
                bool(a["errors"]) and a["errors"][0].startswith(ENVELOPE_ERROR) and "BufferError" not in " ".join(a["errors"])
                and a["service_returncode"] not in (0, None) and sorted(scored(a)) == [str(f[19])] for a in (ve, ce)),
            envelope_both_snapshot_preserved=all(bool(a["snapshot"]) and a["snapshot"]["hashes_verify"]
                                                 and a["snapshot"]["frame_id"] == f[23]
                                                 and a["snapshot"]["exception_type"] == "LiveGtContractError"
                                                 for a in (ve, ce)),
            clean_shutdown_normal_tilt=all(a["service_returncode"] == 0 and not a["errors"] and a["ue_result_ok"]
                                           for a in (vn, cn, vt, ct)),
            candidate_paths_started_candidate_worker=all(a["worker_entries"] == [C.WORKER_ENTRY] for a in cand),
            v8_paths_started_v8_worker=all(a["worker_entries"] == [V8C.WORKER_ENTRY] for a in v8),
            candidate_accounting_installed_and_restored=all(
                (a["accounting_binding"] or {}).get("installed") is True
                and (a["accounting_binding"] or {}).get("restored") is True
                and (a["accounting_binding"] or {}).get("version") == A.VERSION for a in (cn, ct)),
            v8_paths_have_no_accounting_switch=all(a["accounting_binding"] is None for a in v8),
            candidate_fast_path_capability_and_use=all(
                all(reason is None for reason in ((a["accounting_binding"] or {}).get("capability") or {"x": 1}).values())
                and ((a["accounting_binding"] or {}).get("accounting_session") or {}).get("snapshot_fast", 0) > 0
                and ((a["accounting_binding"] or {}).get("accounting_session") or {}).get("owner_shortcut_hits", 0) > 0
                for a in (cn, ct)),
            tilt_and_freeze_repair_installed_both_paths=all(
                (a["avo_binding"] or {}).get("installed") is True and (a["prep_lifetime"] or {}).get("installed") is True
                for a in (vn, cn, vt, ct)))
        status = "PASS" if all(checks.values()) else "FAIL"
        result = dict(schema="splitfusion.gt_storage_accounting_v3.service_proof.v1", status=status, checks=checks,
                      baseline=dict(arms=["v8_normal", "v8_normal_repeat"], differences=len(baseline),
                                    value=[d for d in baseline if d[1] == "VALUE"],
                                    patterns=sorted(baseline_patterns)),
                      differences={k: dict(timing=sum(d[1] == "timing" for d in v),
                                           clock_derived=sum(d[1] == "clock_derived" for d in v), value=value_diffs[k],
                                           outside_baseline=outside_baseline[k],
                                           patterns=sorted({pattern(d[0]) for d in v}))
                                   for k, v in pairs.items()},
                      prep_cpu_ms_p50={k: a["prep_cpu_ms_p50"] for k, a in arms.items()},
                      arms={k: {kk: vv for kk, vv in a.items() if kk != "rows"} for k, a in arms.items()},
                      candidate_version=A.VERSION, limits=limits, sources=C.exact_sources(ROOT),
                      live_services_started=False, carla_server_started=False, oai_started=False,
                      scope="real GT service/worker path on retained v11 inputs (49 frames) and synthetic tilts; "
                            "short history: proves output/semantic parity, not late-route cost")
        print(json.dumps(dict(status=status, failed=[k for k, v in checks.items() if not v],
                              value_diffs=value_diffs, outside_baseline=outside_baseline,
                              baseline_value=result["baseline"]["value"],
                              counts={k: (v["timing"], v["clock_derived"]) for k, v in result["differences"].items()}),
                         indent=1, default=str))
    except BaseException as error:
        result = dict(status="FAIL", error=f"{type(error).__name__}: {error}", arms_completed=sorted(arms),
                      live_services_started=False)
        raise
    finally:
        (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
        manifest = {str(p.relative_to(out)): CC.sha256(p.read_bytes()) for p in sorted(out.rglob("*"))
                    if p.is_file() and p.name != "OUTPUT_SHA256.json" and "archive" not in p.parts}
        (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    if result["status"] != "PASS":
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    run(parser.parse_args().name)
