"""Real-service proof of the v7 GT prep FREEZE lifetime repair (offline, CPU only; no CARLA, OAI, GPU or radio).

Every arm runs the production chain: UE seam -> GT service trampoline -> real ``exact_gt_live_v2`` service -> GT
worker trampolines -> real prep and score workers, with the registered M3 limits, the registered GC arm and producer
v2, the offline 49-frame corpus and the harness of the sealed v3/v4 real-service rehearsals (``exact_gt_live_v2``
``Rig``; reference scorer; archive rescore).  Nothing older is rebound: each path is started through its own seam.

  normal_v6         V6 path (v6 seam -> v4 service -> v4 worker)        v3 rehearsal scenario: 11 scored frames,
  normal_v7         V7 path (v7 seam -> v7 service -> v7 worker)        send-failure cancel, edge terminal before
                                                                       prediction, horizon timeout + late prediction
  inject_v6         V6 path, engine frame budget 2: frames 0-1 scored, then frame 2's FREEZE fails inside the real
  inject_v7         V7 path, same                   ``prepare_unbound`` ("runtime frame budget exhausted", a genuine
                                                    fail-closed ProducerError raised while the decoded source is
                                                    borrowed from the sealed mapping)
  inject_v7_snapshot_write_failure   V7 path, same, with ``<trace>/freeze_failure_v1`` pre-created so the fatal
                                     snapshot cannot be written

Registered expectations (all must hold): normal arms exact against the reference and identical to each other; the V6
injection reproduces the V6 live signature ``prep FREEZE: BufferError``; both V7 injections report the original
``ProducerError`` and never BufferError; the V7 snapshot (ERROR/REQUEST/SOURCE/READY) is complete after service
teardown and SOURCE.bin matches every sealed buffer digest of the failed frame's descriptor; snapshot-write failure
is reported and does not replace the original error; the failed frame is never committed or scored and the service
fails closed.  The engine budget is the injection only: it is a test configuration, not a registered limit.

  cd <deployment worktree>; env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -B -m rl_agent.splitfusion_m3_route_b_v7.freeze_service_proof --name <dir>
"""
import argparse
import hashlib
import json
from pathlib import Path
import queue
import time

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as CC
from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
from rl_agent.splitfusion_exact_gt_live_v2 import contract as EC
from rl_agent.splitfusion_exact_gt_live_v2 import rehearsal as R
from rl_agent.splitfusion_exact_gt_live_v2.edge import install_early_gt_edge, EarlyGtEvaluatorV1
from rl_agent.splitfusion_exact_gt_live_v2.ue import EarlyGtUeHandoff
from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import quality_bridge
from rl_agent.splitfusion_numeric_gt_runtime_v1.async_live_rehearsal import _owned_collector
from rl_agent.splitfusion_m3_route_b_v6 import contract as V6
from rl_agent.splitfusion_m3_route_b_v6 import gc_seams as V6_SEAM

from . import contract as C
from . import gc_seams as V7_SEAM
from .host import FREEZE_PROOF_SCHEMA

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_m3_route_b_v7"
BUDGET = C.FRAME_CEILING
ARM, PRODUCER = "candidate", "v2"                  # the registered GC arm / numeric producer
INJECT_ENGINE_FRAMES = 2
INJECTED = "ProducerError: runtime frame budget exhausted"
V6_SIGNATURE = "WorkerFailed: prep FREEZE: BufferError: cannot close exported pointers exist"
PATHS = dict(v6=(V6, V6_SEAM), v7=(C, V7_SEAM))
KEYS = ("outcome", "q_service", "q_emitted", "feedback", "committed", "commit")


class Rig(R.Rig):
    """The sealed rehearsal rig, with the path's own registered sources and an explicit engine frame budget."""

    def __init__(self, out, corpus, limits, sources, engine_frames):
        self.c, self.cases = corpus, corpus.cases
        self.ue_dir = out / "ue_attempt"
        self.ue_dir.mkdir()
        self.evidence = out / "edge_evidence"
        self.evidence.mkdir(parents=True)
        registration = dict(exact_gt_limits=limits, exact_gt_sources=sources,
            numeric_run_binding=corpus.binding, numeric_producer_binding=corpus.startup["binding_sha256"],
            numeric_prediction_contract=corpus.contract, exact_gt_socket_name=EC.SOCKET_NAME,
            exact_gt_capability_name=EC.CAPABILITY_NAME)
        self.h = EarlyGtUeHandoff(attempt_dir=self.ue_dir, registration=registration, service_config={})
        self.h.collector = _owned_collector(self.cases)
        self.h.evidence_dir = self.evidence
        self.engine = dict(producer_args=dict(corpus.producer_args(), max_frames=int(limits["max_frames"])),
            run_binding=corpus.binding, producer_binding=corpus.startup["binding_sha256"],
            prediction_contract=corpus.contract, max_frames=int(engine_frames),
            max_retained_jobs=int(limits["prep_jobs"]), max_retained_bytes=int(limits["prep_jobs"]) * 64 * 1024 * 1024,
            offline_replay=False)
        self.h.start_service(self.engine)
        self.payloads, self.feedback, self.rows = queue.Queue(), {}, {}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def _normal(rig, limits):
    f = [c.frame_id for c in rig.cases]
    scenario = {}

    def scored(i, prediction_first):
        rig.admit(i)
        if not prediction_first:
            time.sleep(0.3)
        rig.predict(f[i]); rig.await_feedback(f[i]); rig.terminal(f[i])
        time.sleep(0.12)
    for i in range(8):
        scored(i, prediction_first=bool(i % 2))
    scenario["scored_first"] = f[:8]
    rig.admit(8, send=False); time.sleep(0.4); rig.terminal(f[8])
    scenario["send_failure_cancel"] = f[8]
    rig.admit(9); time.sleep(0.2); rig.ev.notify_terminal(f[9], "STALE_BEFORE_MAP", "DECODE")
    time.sleep(0.5); rig.terminal(f[9])
    scenario["edge_terminal_before_prediction"] = f[9]
    rig.admit(10); time.sleep(limits["horizon_ms"] / 1e3 + 0.6); rig.predict(f[10])
    rig.await_feedback(f[10]); rig.terminal(f[10])
    scenario["horizon_timeout_then_late_prediction"] = f[10]
    for i in (11, 12, 13):
        scored(i, prediction_first=bool(i % 2))
    scenario["scored_after"] = [f[11], f[12], f[13]]
    return scenario


def _inject(rig, state):
    f = [c.frame_id for c in rig.cases]
    for i in range(INJECT_ENGINE_FRAMES):          # bound + committed + scored: the engine budget is now used up
        rig.admit(i, send=True); rig.predict(f[i]); rig.await_feedback(f[i]); rig.terminal(f[i])
        time.sleep(0.12)
    state["injected_frame"] = f[INJECT_ENGINE_FRAMES]
    try:                                           # its FREEZE reaches prepare_unbound in the real prep worker
        rig.admit(INJECT_ENGINE_FRAMES, send=True)
    except BaseException as error:
        state["ue_admit_error"] = f"{type(error).__name__}: {error}"[:1000]
    deadline = time.monotonic() + 30
    while rig.h.process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    state["service_exited_by_itself"] = rig.h.process.poll() is not None
    return dict(scored_before=f[:INJECT_ENGINE_FRAMES], injected=state["injected_frame"])


def arm_run(out, name, corpus, limits, spec, *, path, inject, block_snapshot=False):
    contract, seam = PATHS[path]
    out.mkdir()
    registration = dict(id=contract.ID, transmitted_budget=BUDGET, gt_gc=contract.gc_registration(ARM, BUDGET, PRODUCER))
    rig, scenario, state = None, {}, dict(ue_admit_error=None, close_error=None)
    with seam.install_ue_service_seam(registration=registration, output=out) as trace_dir:
        if block_snapshot:
            (trace_dir / C.FAILURE_SNAPSHOT_DIR).mkdir()
        try:
            rig = Rig(out, corpus, limits, contract.exact_sources(ROOT), INJECT_ENGINE_FRAMES if inject else BUDGET)
            with install_early_gt_edge(socket_path=rig.evidence / EC.SOCKET_NAME,
                                       capability_path=rig.evidence / EC.CAPABILITY_NAME, connect_timeout_s=30):
                rig.ev = EarlyGtEvaluatorV1(spec=spec, read_ground_truth=None, send=rig.payloads.put,
                                            match_distance_m=2., gt_timeout_s=2.)
                rig.ev.start()
                scenario = _inject(rig, state) if inject else _normal(rig, limits)
                try:
                    edge_report = rig.ev.close()
                except BaseException as error:     # injections: the service (edge peer) has already failed
                    if not inject:
                        raise
                    state["edge_close_error"], edge_report = f"{type(error).__name__}: {error}"[:1000], None
            try:
                ue_result = rig.h.close()
            except BaseException as error:         # expected for injections: close() aborts and re-raises
                if not inject:
                    raise
                state["close_error"] = f"{type(error).__name__}: {error}"[:1000]
                ue_result = None
        except BaseException as error:
            if rig is not None and rig.h.process is not None and rig.h.process.poll() is None:
                rig.h.abort(error)
            raise
    # ---- everything below runs after service teardown (service and both workers have exited)
    service_dir = rig.ue_dir / "numeric_gt/service"
    rows = [json.loads(line) for line in open(service_dir / "FRAMES.jsonl")] if (service_dir / "FRAMES.jsonl").exists() else []
    service_result = _json(service_dir / "SERVICE_RESULT.json") or {}
    from rl_agent.splitfusion_fixed_action_screen_analysis_v2.analyze import rescore
    ref = P.AnchorReferenceDriver(corpus)
    index = {c.frame_id: i for i, c in enumerate(rig.cases)}
    frames = {}
    try:
        for row in rows:
            fid = row["frame_id"]
            fb = rig.feedback.get(fid) or {}
            entry = dict(outcome=row.get("outcome"), q_service=(row.get("result") or {}).get("q_perc"),
                         q_emitted=fb.get("q_perc"), feedback=fb.get("kind"), committed=bool(row.get("committed")),
                         commit=digest(row.get("commit")))
            if row.get("outcome") == "SCORED":
                expected = P.score_exact(corpus, rig.cases[index[fid]], ref.run(rig.cases[index[fid]])["packet"])["q_perc"]
                entry.update(q_reference=expected, exact=entry["q_service"] == expected == fb.get("q_perc"),
                             archive_rescore_exact=rescore(service_dir / "archive" / f"frame_{fid}" / "scorer"))
            frames[str(fid)] = entry
    finally:
        ref.close()
    docs = {n: _json(trace_dir / n) for n in ("prep_gc_trace.json", "score_gc_trace.json", "service_substitution.json",
                                              "ue_substitution.json", C.PREP_BINDING_RECORD)}
    prep_log = (service_dir / "workers/prep.log")
    prep_log_text = prep_log.read_text(errors="replace") if prep_log.exists() else ""
    errors = list(service_result.get("errors") or [])
    service_sub = docs["service_substitution.json"] or {}
    worker_argv = [s.get("argv", []) for s in service_sub.get("substitutions", [])]
    observed = dict(
        service_errors=errors, service_returncode=rig.h.process.returncode, ue_admit_error=state["ue_admit_error"],
        ue_close_error=state["close_error"], service_exited_by_itself=state.get("service_exited_by_itself"),
        edge_close_error=state.get("edge_close_error"),
        prep_gc_trace_written=docs["prep_gc_trace.json"] is not None,
        prep_gc_trace_exit=(docs["prep_gc_trace.json"] or {}).get("exit"),
        prep_binding_record=docs[C.PREP_BINDING_RECORD], worker_entry=sorted({a[a.index("-m") + 1] for a in worker_argv
                                                                              if "-m" in a}),
        prep_log_tail=prep_log_text[-4000:])
    scored = {k: v for k, v in frames.items() if v["outcome"] == "SCORED"}
    checks = dict(
        started_this_paths_entries=sorted({a[a.index("-m") + 1] for a in worker_argv if "-m" in a})
            == [contract.gc_registration(ARM, BUDGET, PRODUCER)["trampolines"]["worker"]],
        every_scored_frame_exact=bool(scored) and all(v["exact"] and v["archive_rescore_exact"] for v in scored.values()))
    snapshot = None
    if not inject:
        checks.update(
            eleven_scored=len(scored) == 11,
            unscored_scenarios_unscored=all(frames[str(x)]["outcome"] != "SCORED" for x in (
                scenario["send_failure_cancel"], scenario["edge_terminal_before_prediction"],
                scenario["horizon_timeout_then_late_prediction"])),
            service_exit_zero=ue_result is not None and ue_result["service_returncode"] == 0 and not errors
                and edge_report["failure"] is None,
            no_failure_snapshot=not (trace_dir / C.FAILURE_SNAPSHOT_DIR).exists())
    else:
        failed = str(scenario["injected"])
        checks.update(
            frames_before_injection_scored=sorted(scored) == sorted(str(x) for x in scenario["scored_before"]),
            injected_frame_not_committed_or_scored=(failed not in frames) or (
                frames[failed]["outcome"] != "SCORED" and not frames[failed]["committed"]),
            no_frame_after_failure_scored=all(int(k) < int(failed) for k in scored),
            service_failed_closed=rig.h.process.returncode not in (0, None) and bool(errors)
                and ue_result is None and state["close_error"] is not None)
        if path == "v6":
            checks["v6_live_signature_reproduced"] = bool(errors) and errors[0] == V6_SIGNATURE
            checks["no_v7_snapshot_on_v6_path"] = not (trace_dir / C.FAILURE_SNAPSHOT_DIR).exists()
        else:
            original = f"WorkerFailed: prep FREEZE: {INJECTED}"
            checks.update(
                original_error_reported=bool(errors) and errors[0] == original,
                no_buffer_error_anywhere=not any("BufferError" in e for e in errors)
                    and "BufferError" not in prep_log_text,
                chained_traceback_printed="Traceback (most recent call last)" in prep_log_text
                    and INJECTED in prep_log_text)
            directory = trace_dir / C.FAILURE_SNAPSHOT_DIR
            if block_snapshot:
                checks.update(
                    snapshot_write_failure_reported="FREEZE_DIAGNOSTIC_WRITE_FAILED: FileExistsError" in prep_log_text,
                    blocked_snapshot_left_empty=directory.is_dir() and not any(directory.iterdir()))
            else:
                snapshot = _snapshot(directory, failed)
                checks.update({f"snapshot_{k}": v for k, v in snapshot["checks"].items()})
    return dict(arm=name, path=path, inject=inject, block_snapshot=block_snapshot,
                status="PASS" if all(checks.values()) else "FAIL", checks=checks, scenario=scenario, frames=frames,
                observed=observed, snapshot=snapshot, outcomes=service_result.get("outcomes"),
                limits=service_result.get("limits"))


def _snapshot(directory, failed_frame):
    """Verify the fatal snapshot after teardown; SOURCE.bin must carry every sealed buffer of the failed descriptor."""
    ready = _json(directory / "READY.json") or {}
    hashes = ready.get("sha256") or {}
    hashes_ok = bool(hashes) and all((directory / n).is_file() and CC.sha256((directory / n).read_bytes()) == d
                                     for n, d in hashes.items())
    error = _json(directory / "ERROR.json") or {}
    request = _json(directory / "REQUEST.json") or {}
    descriptor = json.loads(request.get("descriptor", "{}"))
    source = (directory / "SOURCE.bin").read_bytes() if (directory / "SOURCE.bin").is_file() else b""
    buffers = descriptor.get("buffers") or []
    buffers_ok = bool(buffers) and len(source) == descriptor.get("total_bytes") and all(
        hashlib.sha256(source[b["offset"]:b["offset"] + b["nbytes"]]).hexdigest() == b["sha256"] for b in buffers)
    checks = dict(ready_complete=ready.get("complete") is True and set(hashes) == {"ERROR.json", "REQUEST.json",
                                                                                     "SOURCE.bin"},
                  hashes_verify=hashes_ok,
                  original_exception_recorded=error.get("exception_type") == "ProducerError"
                      and error.get("exception") == INJECTED.split(": ", 1)[1]
                      and "BufferError" not in error.get("traceback", ""),
                  request_is_failed_frame=str(descriptor.get("frame_id")) == str(failed_frame),
                  source_matches_sealed_buffers=buffers_ok, source_within_bound=0 < len(source) <= 32 * 1024 * 1024)
    return dict(checks=checks, files={n: (directory / n).stat().st_size for n in hashes if (directory / n).is_file()},
                exception_type=error.get("exception_type"), exception=error.get("exception"),
                descriptor_frame_id=descriptor.get("frame_id"), source_bytes=len(source))


ARMS = (("normal_v6", dict(path="v6", inject=False)), ("normal_v7", dict(path="v7", inject=False)),
        ("inject_v6", dict(path="v6", inject=True)), ("inject_v7", dict(path="v7", inject=True)),
        ("inject_v7_snapshot_write_failure", dict(path="v7", inject=True, block_snapshot=True)))


def run(name):
    P.environment_ok()
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    corpus = P.Corpus(49)
    limits = C.limits(BUDGET)
    spec, _ = quality_bridge()
    result = None
    try:
        arms = {arm: arm_run(out / arm, arm, corpus, limits, spec, **kwargs) for arm, kwargs in ARMS}
        a, b = arms["normal_v6"]["frames"], arms["normal_v7"]["frames"]
        identical = a.keys() == b.keys() and all({k: a[f][k] for k in KEYS} == {k: b[f][k] for k in KEYS} for f in a)
        i6, i7 = arms["inject_v6"]["frames"], arms["inject_v7"]["frames"]
        checks = {f"{arm}:{k}": v for arm, value in arms.items() for k, v in value["checks"].items()}
        checks["normal_v7_identical_to_normal_v6"] = identical
        checks["inject_scored_frames_identical_v6_v7"] = {f: {k: i6[f][k] for k in KEYS} for f in i6
                                                          if i6[f]["outcome"] == "SCORED"} == \
            {f: {k: i7[f][k] for k in KEYS} for f in i7 if i7[f]["outcome"] == "SCORED"}
        status = "PASS" if all(checks.values()) else "FAIL"
        result = dict(schema=FREEZE_PROOF_SCHEMA, status=status, checks=checks, arms=arms, limits=limits,
                      engine_frames_injection=INJECT_ENGINE_FRAMES, injected_error=INJECTED, v6_signature=V6_SIGNATURE,
                      sources=C.exact_sources(ROOT), launcher_binding=C.launcher_binding(),
                      live_services_started=False, carla_started=False, oai_started=False, performance_claim=False,
                      scope="service-path behaviour of the repair; NOT the initiating error of V6 frame 3557")
        print(json.dumps(dict(status=status, failed=[k for k, v in checks.items() if not v],
                              v6_errors=arms["inject_v6"]["observed"]["service_errors"],
                              v7_errors=arms["inject_v7"]["observed"]["service_errors"]), indent=1))
    except BaseException as error:
        result = dict(schema=FREEZE_PROOF_SCHEMA, status="FAIL", error=f"{type(error).__name__}: {error}",
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
