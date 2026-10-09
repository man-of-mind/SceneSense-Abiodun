"""Paced whole-worker replay: unchanged prep/score worker protocol, v1 vs v2 numeric producer.

  cd <abiodun>; env -u PYTHONPATH CUDA_VISIBLE_DEVICES= /usr/bin/python3 -B -m \
    rl_agent.splitfusion_numeric_gt_producer_v2.replay_producer --out <dir> --order 1,2,1,2,1,2,1,2

Identical in both arms: requests (sealed 49 v11 frames, FREEZE..EVIDENCE_DURABLE), pacing (240 ms), CPU placement
(prep 4, score 5, parent 6), thread caps, and GC policy (the adopted candidate GC policy via its trampoline, with
its tracing). Only NUMERIC_PRODUCER differs. Summary via ``splitfusion_gt_worker_spike_fix_v1.summarize``.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_live_gt_handoff_v1.archive import ArchiveClient
from rl_agent.splitfusion_exact_gt_service_proto_v2 import bench as B, service as SV, channel as CH, parity_common as P
from rl_agent.splitfusion_gt_worker_spike_fix_v1 import replay as R

ENTRY = "rl_agent.splitfusion_numeric_gt_producer_v2.worker_entry"


class Client(SV.WorkerClient):
    def __init__(self, role, config, *, log_path, variant, trace_path, timeout_s=120.0, cpu=None):
        import secrets
        self.role, self.timeout = role, float(timeout_s)
        self.nonce, self.seq, self.pending = secrets.token_hex(32), 0, None
        self.failed = self.closed = False
        parent, child = CH.pair()
        self.sock = parent
        self.log = open(log_path, "xb")
        env = dict(os.environ, **SV.ENV, SPIKE_FIX_MODE="candidate", SPIKE_FIX_TRACE=str(trace_path), NUMERIC_PRODUCER=variant)
        env.pop("PYTHONPATH", None)
        try:
            self.process = subprocess.Popen([sys.executable, "-B", "-u", "-m", ENTRY, "--fd", str(child.fileno()),
                "--parent", str(os.getpid()), *([] if cpu is None else ["--cpu", str(cpu)])], pass_fds=(child.fileno(),),
                cwd=SV.ROOT, env=env, stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log)
            child.close()
            CH.send(parent, dict(nonce=self.nonce, role=role, config=config), deadline=time.monotonic() + self.timeout)
            ready, extra = CH.receive(parent, deadline=time.monotonic() + self.timeout)
            C.require(extra is None and ready.get("status") == "READY" and ready.get("nonce") == self.nonce
                      and ready.get("pid") == self.process.pid and ready.get("role") == role
                      and ready.get("cuda_initialized") is False, "worker READY binding differs: " + str(ready))
            self.ready = ready["ready"]
        except BaseException:
            child.close()
            self.abort()
            raise


class Arm(R.Arm):
    def __init__(self, corpus, directory, *, mode, trace_path, scorer_jobs=4, prep_jobs=4):
        self.c, self.dir = corpus, directory
        directory.mkdir(parents=True)
        self.name = mode
        engine = dict(corpus.engine_config(offline=True), max_retained_jobs=prep_jobs,
                      max_retained_bytes=prep_jobs * B.RESERVATION)
        self.prep = Client("prep", dict(engine=engine), log_path=directory / "prep.log", cpu=B.CPU_PREP,
                           variant=mode, trace_path=trace_path)
        self.session = self.prep.ready["session_id"]
        run, producer = corpus.binding, corpus.startup["binding_sha256"]
        self.score = SV.WorkerClient("score", dict(run_binding=run, producer_binding=producer, max_jobs=scorer_jobs,
            max_bytes=scorer_jobs * 8 * 1024 * 1024, reservation_bytes=8 * 1024 * 1024, max_waiting=2, max_frames=200),
            log_path=directory / "score.log", cpu=B.CPU_SCORE)
        (directory / "durable").mkdir()
        self.h = SV.Harness.__new__(SV.Harness)
        self.h.prep, self.h.score, self.h.archive_root = self.prep, self.score, (directory / "durable").resolve()
        self.h.archive = self.archive = ArchiveClient(roots=[str(self.h.archive_root)], max_jobs=4,
            max_bytes=256 * 1024 * 1024, max_frames=200, binding=dict(arm=mode))
        self.sequence = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--order", default="1,2,1,2,1,2,1,2")
    parser.add_argument("--period-ms", type=float, default=240.0)
    parser.add_argument("--max-busy", type=float, default=0.15)
    a = parser.parse_args()
    P.environment_ok()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    host = B.host_check()
    host["busy_fraction_1s"] = R.busy_fraction()
    (out / "HOST_BEFORE.json").write_text(json.dumps(host, default=str, indent=1))
    C.require(host["busy_fraction_1s"] <= a.max_busy and not host["forbidden_processes"], "host is not quiet")
    corpus = P.Corpus(49)
    original = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {B.CPU_PARENT})
    results = []
    try:
        for index, code in enumerate(a.order.split(",")):
            variant = {"1": "v1", "2": "v2"}[code]
            r = session(corpus, out, index, variant, int(a.period_ms * 1e6))
            results.append(r)
            print(f"session {index} {variant} {r['status']} history={r['history_sha256'][:12]}", flush=True)
    finally:
        os.sched_setaffinity(0, set(original))
        (out / "SESSIONS.json").write_text(json.dumps(results, default=str, indent=1))
        (out / "HOST_AFTER.json").write_text(json.dumps(dict(B.host_check(), busy_fraction_1s=R.busy_fraction()), default=str, indent=1))
        leftovers = subprocess.run(["pgrep", "-f", "splitfusion_numeric_gt_producer_v2.worker_entry|splitfusion_exact_gt_service_proto_v2.worker"],
                                   capture_output=True, text=True).stdout.split()
        (out / "CLEANUP.json").write_text(json.dumps(dict(leftover_pids=leftovers), indent=1))


def session(corpus, out, index, variant, period_ns):
    """Body of the spike-fix ``replay.session`` with the producer variant as the arm label and this Arm."""
    import hashlib, shutil
    d = out / f"session_{index:02d}_{variant}"
    trace = d.parent / f"trace_{index:02d}_{variant}.json"
    arm = Arm(corpus, d, mode=variant, trace_path=trace)
    rows = []
    try:
        st = B.proc_status(arm.prep.process.pid)
        C.require(st["Cpus_allowed_list"] == str(B.CPU_PREP) and B.environ_caps(arm.prep.process.pid)["OMP_NUM_THREADS"] == "1",
                  "prep affinity/caps differ")
        settings = dict(prep=dict(status=st, caps=B.environ_caps(arm.prep.process.pid)),
                        score=dict(status=B.proc_status(arm.score.process.pid)), parent_affinity=sorted(os.sched_getaffinity(0)))
        next_start = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        for k, case in enumerate(corpus.cases):
            now = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            if now < next_start:
                time.sleep((next_start - now) / 1e9)
            start = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            next_start = max(next_start + period_ns, start)
            result, state, m = arm.process(case)
            files = arm.retire(case, state, m)
            pred = corpus.predictions[case.frame_id]
            rows.append(dict(session=index, mode=variant, k=k, frame_id=case.frame_id, start_raw_ns=start,
                             end_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                             identity=state["identity"], semantic=state["semantic"], commit=R.digest(state["committed"]),
                             q_perc=result["q_perc"], q_perc_matches_recorded=result["q_perc"] == pred["q_perc"],
                             measurement=R.digest(result["measurement"]), producer_files=files,
                             metrics={k2: v for k2, v in m.items() if not k2.endswith("_on")}))
        snap = arm.snapshot()["producer"]
        closed = arm.close()
        durable_bytes = sum(f.stat().st_size for f in (d / "durable").rglob("*") if f.is_file())
        shutil.rmtree(d / "durable")
        return dict(index=index, mode=variant, status="COMPLETE", rows=rows, settings=settings,
                    history_sha256=snap["history_sha256"], engine=snap["engine"], gate=snap["gate"],
                    jobs_phase=sorted(set(snap["jobs"].values())), close=closed,
                    prep_returncode=arm.prep.process.returncode, trace=trace.name, durable_bytes_removed=durable_bytes)
    except BaseException as error:
        try:
            arm.close(abort=True)
        except BaseException:
            pass
        raise RuntimeError(f"STOPPED: {type(error).__name__}: {error}"[:2000]) from error


if __name__ == "__main__":
    main()
