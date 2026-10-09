"""Quiet-host CPU-only benchmark: existing runtime path (control) vs prototype, paired by exact frame.

Usage (from abiodun/, CPU-only env):
    python -m ...bench --seal-plan <dir>      # writes PLAN.json binding this harness and all sources
    python -m ...bench --run <dir> --plan <plan-dir>

v11 passes (49 frames): both arms alive, frames interleaved, per-frame order alternating
(k + pass) % 2; pass 0 is warm-up for both arms; fresh processes per pass. Stages:
  UE        parent work to hand the frozen source to the service (control: encode_stage + join +
            IPC begin; prototype: freeze + seal + send). This happens before action-open if run on
            the UE worker thread, so it is reported on its own and never folded into later stages.
  DELIVERY  end of UE work -> source decoded in the child.
  PREPARE   source decoded -> bound Prepared in the child (control PREPARE_STAGED; prototype
            prepare_unbound + BIND).
  TARGET    CONFIRM_SEND begin -> target admitted in the scorer (control includes PUBLISH_READY,
            edge inbox decode, evaluator decode, REGISTER, TARGET; prototype PUBLISH + TARGET).
  PREDICT   prediction in hand at the edge -> result in hand (edge digest work, transfer,
            admission, join, score, quality, return).
  ARCHIVE   result -> both sides durably retired (deferred; off the feedback path).
live100 passes (48 frames, scorer only): TARGET and PREDICT on archived targets/predictions.
Retention passes (lag 2, one per arm): retained bytes, RSS and queue occupancy only.
Every frame is checked for exact outputs; any mismatch or settings failure stops the run.
Quiet-host results are not loaded-live performance and not a 170-ms qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import select
import shutil
import subprocess
import sys
import time

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as R
from rl_agent.splitfusion_numeric_gt_producer_v1 import cases as K
from rl_agent.splitfusion_numeric_gt_runtime_v1 import process as RP, staging as ST, scorer as SC, edge_live as EL, producer as PR
from rl_agent.splitfusion_minimal_quality_scorer_v1 import scorer as MS
from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_gt_handoff_v1.archive import ArchiveClient

from . import parity_common as P, freeze_codec as FZ, channel as CH, service as SV
from .verify_composition import live100_frames, PREDICTION_FIELDS

PACKAGE = Path(__file__).resolve().parent
CPU_PREP, CPU_SCORE, CPU_PARENT = 4, 5, 6
PASSES = 4                      # pass 0 = warm-up
COLD_FRAMES = 2                 # first frames of each fresh-process pass, reported separately
RESERVATION = PR.RESERVATION
MAX_LOAD1 = 1.5
FORBIDDEN = ("CarlaUnreal", "CarlaUE4", "CarlaUE5", "nr-softmodem", "nr-uesoftmodem")


def raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def ms(a, b):
    return (b - a) / 1e6


def stats(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return dict(n=0)
    return dict(n=len(v), p50=round(v[(len(v) - 1) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2, 4),
                p95=round(v[max(0, math.ceil(.95 * len(v)) - 1)], 4), max=round(v[-1], 4), min=round(v[0], 4))


# ---------------------------------------------------------------- host / settings evidence
def proc_status(pid):
    out = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM", "Threads", "Cpus_allowed_list", "voluntary_ctxt_switches",
                   "nonvoluntary_ctxt_switches"):
            out[key] = value.strip()
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    out.update(processor=int(stat[36]), nice=int(stat[16]), policy=int(stat[38]))
    return out


def thread_cpu_now():
    return int(Path("/proc/thread-self/stat").read_text().rsplit(")", 1)[1].split()[36])


def environ_caps(pid):
    env = dict(item.split("=", 1) for item in Path(f"/proc/{pid}/environ").read_bytes().decode(errors="replace").split("\0")
               if "=" in item)
    return {k: env.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}


def host_check():
    load = os.getloadavg()
    names = []
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                comm = Path(f"/proc/{pid}/comm").read_text().strip()
            except OSError:
                continue
            if any(comm.startswith(f[:15]) for f in FORBIDDEN):
                names.append((int(pid), comm))
    try:
        docker = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True,
                                timeout=20).stdout.split()
    except Exception as error:  # noqa: BLE001
        docker = [f"unavailable: {error}"]
    freq = {}
    for cpu in (CPU_PREP, CPU_SCORE, CPU_PARENT):
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
        freq[cpu] = {n: (base / n).read_text().strip() for n in ("scaling_governor", "scaling_cur_freq")
                     if (base / n).exists()}
    return dict(raw_ns=raw(), loadavg=load, forbidden_processes=names, docker_containers=docker, cpufreq=freq)


# ---------------------------------------------------------------- edge-side prediction work (identical in both arms)
def edge_prediction(identity_doc, labels, records):
    raw_labels = labels.tobytes(order="C") if isinstance(labels, np.ndarray) else bytes(labels)
    owned = C._json_owned(list(records))
    digest = hashlib.sha256(raw_labels + MS.canonical(owned)).hexdigest()
    identity = {k: identity_doc[k] for k in ("gt_identity", "action_identity", "capture_timestamp_ns")}
    identity["prediction_digest"] = digest
    C.require(digest == identity_doc["prediction_digest"], "edge prediction digest differs from archived digest")
    return identity, raw_labels, owned


def child_cpu(timing):
    return None if timing is None else timing["thread_cpu_ns"] / 1e6


# ---------------------------------------------------------------- arms
class ControlArm:
    name = "control"

    def __init__(self, corpus, directory, *, scorer_jobs=4, producer_jobs=4, with_producer=True, live100=None):
        self.c, self.dir = corpus, directory
        directory.mkdir(parents=True)
        root = directory.resolve()
        self.producer = None
        if with_producer:
            config = dict(corpus.engine_config(offline=True), max_retained_jobs=producer_jobs,
                          max_retained_bytes=producer_jobs * RESERVATION)
            self.producer = RP.Client("producer", config, directory=root / "producer_process",
                source_sha256=corpus.sources, cpu=CPU_PREP, timeout_s=60, writable_attempt_root=root)
        run, producer = (corpus.binding, corpus.startup["binding_sha256"]) if live100 is None else live100
        self.scorer = RP.Client("scorer", dict(run_binding=run, producer_binding=producer, max_jobs=scorer_jobs,
            max_bytes=scorer_jobs * 32 * 1024 * 1024), directory=root / "scorer_process",
            source_sha256=corpus.sources, cpu=CPU_SCORE, timeout_s=60, writable_attempt_root=root)
        (root / "durable").mkdir()
        self.archive_root = root / "durable"
        self.archive = ArchiveClient(roots=[str(self.archive_root)], max_jobs=4, max_bytes=256 * 1024 * 1024,
                                     max_frames=200, binding=dict(arm="control"))
        self.job = 0

    def pids(self):
        return {k: v.process.pid for k, v in (("producer", self.producer), ("scorer", self.scorer)) if v is not None}

    def process(self, case):
        c, m = self.c, {}
        self.job += 1
        job = self.job
        sk = c.sourcekwargs(case)
        cpu0, t0 = time.thread_time_ns(), raw()
        chunks, extras = ST.encode_stage(sk, case.kwargs["policy_timestamp_s"])
        packet = b"".join(chunks)
        self.producer.begin("STAGE_SOURCE", dict(job=job, source_packet=packet, stage_extras=extras))
        t1, cpu1 = raw(), time.thread_time_ns()
        m["UE_ms"], m["UE_cpu_ms"] = ms(t0, t1), (cpu1 - cpu0) / 1e6
        m["UE_parent_cpu_on"] = thread_cpu_now()
        self.producer.finish()
        stage = self.producer.events[-1]
        m["DELIVERY_ms"] = ms(t1, stage["worker"]["end_raw_ns"])
        m["DELIVERY_child_cpu_ms"] = stage["worker"]["thread_cpu_ns"] / 1e6
        request = dict(job=job, plan_bytes=c.plan(case), capture_identity=case.kwargs["identity"],
                       frame_context=PO.context_document(c.context(case)),
                       population=case.evidence["dynamic_snapshot_evidence"], **c.offline_inputs(case))
        prepared = self.producer.call("PREPARE_STAGED", request)
        prep = self.producer.events[-1]
        m["PREPARE_ms"] = ms(stage["worker"]["end_raw_ns"], prep["worker"]["end_raw_ns"])
        m["PREPARE_child_cpu_ms"] = prep["worker"]["thread_cpu_ns"] / 1e6
        # target readiness
        expected = c.expected(case)
        key = prepared["identity_bytes"]
        cpu0, t0 = time.thread_time_ns(), raw()
        committed = self.producer.call("CONFIRM_SEND", dict(job=job, plan_bytes=c.plan(case)))
        e_conf = self.producer.events[-1]
        ready = self.producer.call("PUBLISH_READY", dict(job=job))
        e_pub = self.producer.events[-1]
        inbox_expected = EL._expected(key)
        R.decode_target(ready, **SC._expected(inbox_expected))            # edge inbox admission
        R.decode_target(ready, **SC._expected(inbox_expected))            # evaluator thread
        self.scorer.call("REGISTER", dict(key=key, expected=dict(expected,
            frame_context=PO.context_document(expected["frame_context"])), reservation_bytes=RESERVATION // 2))
        e_reg = self.scorer.events[-1]
        self.scorer.call("TARGET", dict(key=key, packet=ready))
        e_tgt = self.scorer.events[-1]
        t1 = raw()
        m["TARGET_ms"], m["TARGET_parent_cpu_ms"] = ms(t0, t1), (time.thread_time_ns() - cpu0) / 1e6
        m["TARGET_child_cpu_ms"] = sum(e["worker"]["thread_cpu_ns"] for e in (e_conf, e_pub, e_reg, e_tgt)) / 1e6
        # prediction
        pred = c.predictions[case.frame_id]
        result = self.predict(key, pred["identity"], pred["labels"], pred["records"], m)
        state = dict(job=job, key=key, expected=expected, committed=committed, prepared=prepared, packet_sha=P.sha(ready))
        return result, state, m

    def predict(self, key, identity_doc, labels, records, m, *, contract=None):
        cpu0, t0 = time.thread_time_ns(), raw()
        identity, raw_labels, owned = edge_prediction(identity_doc, labels, records)
        self.scorer.call("PREDICTION", dict(key=key, identity=identity, labels=raw_labels, shape=[720, 1280],
                                            records=owned, prediction_contract=contract or self.c.contract))
        e_p = self.scorer.events[-1]
        score = self.scorer.call("SCORE", dict(key=key))
        e_s = self.scorer.events[-1]
        t1 = raw()
        m["PREDICT_ms"], m["PREDICT_parent_cpu_ms"] = ms(t0, t1), (time.thread_time_ns() - cpu0) / 1e6
        m["PREDICT_child_cpu_ms"] = (e_p["worker"]["thread_cpu_ns"] + e_s["worker"]["thread_cpu_ns"]) / 1e6
        return score

    def retire(self, case, state, m):
        key, job = state["key"], state["job"]
        cpu0, t0 = time.thread_time_ns(), raw()
        self.scorer.call("FEEDBACK_CONFIRMED", dict(key=key))
        note = dict(job=job, identity_bytes=key, semantic_sha256=state["prepared"]["semantic_sha256"])
        self.producer.call("SCORE_COMPLETED", note)
        self.producer.call("FEEDBACK_CONFIRMED", note)
        archival = self.producer.call("BUILD_ARCHIVE", dict(job=job))
        descriptor = json.loads(archival["descriptor_bytes"])
        pfiles = archival["files"]
        arguments = {n: descriptor[n] for n in ("ready_semantic_sha256", "input_metadata", "avo_document",
                                                "legacy_target_meta", "old_prepared_sha256")}
        arguments.update(segmentation_png=pfiles["segmentation.png"], object_ignore_png=pfiles["object_ignore.png"])
        sfiles = self.scorer.call("ATTACH_ARCHIVE", dict(key=key, arguments=arguments))["files"]
        pfx, sfx = f"frame_{case.frame_id}/producer", f"frame_{case.frame_id}/scorer/"
        files = {str(self.archive_root / pfx / n): r for n, r in pfiles.items()}
        files.update({str(self.archive_root / sfx / n): r for n, r in sfiles.items()})
        receipt = durable(self.archive, case.kwargs["identity"], files)
        self.scorer.call("ARCHIVE_DURABLE", dict(key=key, receipt=receipt, relative_prefix=sfx))
        self.producer.call("SCORER_ARCHIVE_COMPLETED", note)
        self.producer.call("ARCHIVE_DURABLE", dict(job=job, receipt=receipt, relative_prefix=pfx))
        m["ARCHIVE_ms"], m["ARCHIVE_parent_cpu_ms"] = ms(t0, raw()), (time.thread_time_ns() - cpu0) / 1e6
        m["ARCHIVE_bytes"] = sum(map(len, files.values()))
        return {k: P.sha(v) for k, v in pfiles.items()}

    def snapshot(self):
        out = dict(scorer=self.scorer.call("SNAPSHOT"))
        if self.producer is not None:
            out["producer"] = self.producer.call("SNAPSHOT")
        return out

    def close(self, *, abort=False):
        result = {}
        for name, client in (("producer", self.producer), ("scorer", self.scorer)):
            if client is None:
                continue
            if abort:
                client.abort(); result[name] = dict(aborted=True, returncode=client.process.poll())
            else:
                client.close(); result[name] = dict(returncode=client.process.returncode)
        result["archive"] = self.archive.close()["completed"]
        return result


class ProtoArm:
    name = "prototype"

    def __init__(self, corpus, directory, *, scorer_jobs=4, prep_jobs=4, with_prep=True, live100=None):
        self.c, self.dir = corpus, directory
        directory.mkdir(parents=True)
        self.prep = None
        if with_prep:
            engine = dict(corpus.engine_config(offline=True), max_retained_jobs=prep_jobs,
                          max_retained_bytes=prep_jobs * RESERVATION)
            self.prep = SV.WorkerClient("prep", dict(engine=engine), log_path=directory / "prep.log", cpu=CPU_PREP)
            self.session = self.prep.ready["session_id"]
        run, producer = (corpus.binding, corpus.startup["binding_sha256"]) if live100 is None else live100
        self.score = SV.WorkerClient("score", dict(run_binding=run, producer_binding=producer, max_jobs=scorer_jobs,
            max_bytes=scorer_jobs * 8 * 1024 * 1024, reservation_bytes=8 * 1024 * 1024, max_waiting=2, max_frames=200),
            log_path=directory / "score.log", cpu=CPU_SCORE)
        (directory / "durable").mkdir()
        self.h = SV.Harness.__new__(SV.Harness)
        self.h.prep, self.h.score, self.h.archive_root = self.prep, self.score, (directory / "durable").resolve()
        self.h.archive = self.archive = ArchiveClient(roots=[str(self.h.archive_root)], max_jobs=4,
            max_bytes=256 * 1024 * 1024, max_frames=200, binding=dict(arm="prototype"))
        self.sequence = 0

    def pids(self):
        return {k: v.process.pid for k, v in (("prep", self.prep), ("score", self.score)) if v is not None}

    def process(self, case):
        c, m = self.c, {}
        self.sequence += 1
        cpu0, t0 = time.thread_time_ns(), raw()
        descriptor, chunks = FZ.freeze(c.sourcekwargs(case), case.kwargs["policy_timestamp_s"],
                                       session_id=self.session, sequence=self.sequence)
        tf = raw()
        fd, size = FZ.seal(chunks)
        ts = raw()
        try:
            self.prep.begin("FREEZE", dict(descriptor=descriptor.decode(), offline=c.offline_inputs(case)), fd=fd)
        finally:
            os.close(fd)
        t1, cpu1 = raw(), time.thread_time_ns()
        m["UE_ms"], m["UE_cpu_ms"] = ms(t0, t1), (cpu1 - cpu0) / 1e6
        m["UE_freeze_ms"], m["UE_seal_ms"], m["UE_send_ms"] = ms(t0, tf), ms(tf, ts), ms(ts, t1)
        m["UE_parent_cpu_on"] = thread_cpu_now()
        m["UE_descriptor_bytes"], m["UE_sealed_bytes"] = len(descriptor), size
        reply = self.prep.finish()
        frozen = SV.ok(reply, "FREEZE")
        m["DELIVERY_ms"] = ms(t1, frozen["decoded_raw_ns"])
        m["DELIVERY_child_cpu_ms"] = None   # decode CPU is inside FREEZE; reported via PREPARE split below
        freeze_cpu = child_cpu(reply.timing)
        plan = c.plan(case)
        bind = self.prep.call("BIND", dict(job=frozen["job"], plan=plan.decode(),
                                           frame_context=PO.context_document(c.context(case))))
        bound = SV.ok(bind, "BIND")
        m["PREPARE_ms"] = ms(frozen["decoded_raw_ns"], bind.timing["end_raw_ns"])
        m["FREEZE_REQUEST_child_cpu_ms"], m["BIND_child_cpu_ms"] = freeze_cpu, child_cpu(bind.timing)
        m["PREPARE_child_cpu_ms"] = freeze_cpu + child_cpu(bind.timing)   # includes decode; see DELIVERY note
        cpu0, t0 = time.thread_time_ns(), raw()
        conf = self.prep.call("CONFIRM_SEND", dict(job=frozen["job"], plan=plan.decode()))
        committed = SV.ok(conf, "CONFIRM_SEND")["commit"]
        pub = self.prep.call("PUBLISH", dict(job=frozen["job"]))
        published = SV.ok(pub, "PUBLISH")
        try:
            tgt = self.score.call("TARGET", dict(identity=published["identity"], semantic_sha256=published["semantic_sha256"],
                                                 size=published["size"]), fd=pub.fd)
        finally:
            os.close(pub.fd)
        SV.ok(tgt, "TARGET")
        t1 = raw()
        m["TARGET_ms"], m["TARGET_parent_cpu_ms"] = ms(t0, t1), (time.thread_time_ns() - cpu0) / 1e6
        m["TARGET_child_cpu_ms"] = sum(child_cpu(x.timing) for x in (conf, pub, tgt))
        pred = c.predictions[case.frame_id]
        result = self.predict(published["identity"], pred["identity"], pred["labels"], pred["records"],
                              PO.context_document(c.context(case)), m)
        state = dict(job=frozen["job"], key=published["identity"].encode(), identity=published["identity"],
                     semantic=published["semantic_sha256"], committed=committed, chunks=chunks)
        return result, state, m

    def predict(self, identity_str, identity_doc, labels, records, context_doc, m):
        cpu0, t0 = time.thread_time_ns(), raw()
        identity, raw_labels, owned = edge_prediction(identity_doc, labels, records)
        fd, size = CH.seal_bytes([raw_labels])
        try:
            reply = self.score.call("PREDICTION", dict(identity=identity, frame_context=context_doc, records=owned,
                                                       size=size), fd=fd)
        finally:
            os.close(fd)
        out = SV.ok(reply, "PREDICTION")
        t1 = raw()
        C.require(out["status"] == "SCORED", "prediction must find its cached target")
        m["PREDICT_ms"], m["PREDICT_parent_cpu_ms"] = ms(t0, t1), (time.thread_time_ns() - cpu0) / 1e6
        m["PREDICT_child_cpu_ms"] = child_cpu(reply.timing)
        return out["result"]

    def retire(self, case, state, m):
        cpu0, t0 = time.thread_time_ns(), raw()
        SV.ok(self.score.call("RESULT_DELIVERED", dict(identity=state["identity"])), "score DELIVERED")
        SV.ok(self.prep.call("RESULT_DELIVERED", dict(job=state["job"])), "prep DELIVERED")
        sfiles, pfiles, proof, outcome = self.h.retire_scored(state["job"], case.kwargs["identity"], state["identity"],
                                                              case.frame_id)
        m["ARCHIVE_ms"], m["ARCHIVE_parent_cpu_ms"] = ms(t0, raw()), (time.thread_time_ns() - cpu0) / 1e6
        m["ARCHIVE_bytes"] = sum(map(len, sfiles.values())) + sum(map(len, pfiles.values()))
        return {k: P.sha(v) for k, v in pfiles.items()}

    def snapshot(self):
        out = dict(scorer=SV.ok(self.score.call("SNAPSHOT"), "SNAPSHOT"))
        if self.prep is not None:
            out["producer"] = SV.ok(self.prep.call("SNAPSHOT"), "SNAPSHOT")
        return out

    def close(self, *, abort=False):
        result = {}
        for name, client in (("prep", self.prep), ("score", self.score)):
            if client is None:
                continue
            if abort:
                client.abort(); result[name] = dict(aborted=True, returncode=client.process.poll())
            else:
                client.close(abandon=False); result[name] = dict(returncode=client.process.returncode)
        result["archive"] = self.archive.close()["completed"]
        return result


def durable(archive, identity, files):
    job = archive.submit(identity, files)
    deadline = time.monotonic() + 60
    while True:
        archive.poll()
        done = [r for r in archive.completions if r["job_id"] == job]
        if done:
            C.require(len(done) == 1 and done[0]["durable"], "one durable completion")
            for name, payload in files.items():
                C.require(Path(name).read_bytes() == payload, "durable readback differs")
            return done[0]
        C.require(time.monotonic() < deadline, "archive completion timeout")
        select.select([archive.socket], [], [], 0.5)


# ---------------------------------------------------------------- run
class Runner:
    def __init__(self, out):
        self.out = out
        self.original_affinity = sorted(os.sched_getaffinity(0))
        self.rows, self.settings, self.cleanup = [], [], []

    def pin_parent(self, on):
        os.sched_setaffinity(0, {CPU_PARENT} if on else set(self.original_affinity))

    def record_settings(self, label, arms):
        entry = dict(label=label, parent=dict(pid=os.getpid(), affinity=sorted(os.sched_getaffinity(0)),
                                              status=proc_status(os.getpid()), caps=environ_caps(os.getpid())))
        for arm in arms:
            for role, pid in arm.pids().items():
                status, caps = proc_status(pid), environ_caps(pid)
                entry[f"{arm.name}.{role}"] = dict(pid=pid, status=status, caps=caps)
                expected = str(CPU_PREP if role in ("producer", "prep") else CPU_SCORE)
                C.require(status["Cpus_allowed_list"] == expected, f"{arm.name}.{role} affinity differs: {status}")
                C.require(caps["CUDA_VISIBLE_DEVICES"] == "" and caps["OMP_NUM_THREADS"] == "1",
                          f"{arm.name}.{role} thread caps differ")
                C.require(status["nice"] == 0 and status["policy"] == 0, "default scheduling policy required")
        self.settings.append(entry)

    def rss(self, arms):
        return {f"{a.name}.{r}": {k: proc_status(p)[k] for k in ("VmRSS", "VmHWM")} for a in arms for r, p in a.pids().items()}

    def v11_pass(self, corpus, index, truth):
        d = self.out / f"pass_{index}"
        order_arms = (ControlArm, ProtoArm) if index % 2 == 0 else (ProtoArm, ControlArm)
        arms = {}
        for cls in order_arms:
            arms[cls.name] = cls(corpus, d / cls.name)
        self.pin_parent(True)
        try:
            self.record_settings(f"v11_pass_{index}", arms.values())
            for k, case in enumerate(corpus.cases):
                first = ("control", "prototype") if (k + index) % 2 == 0 else ("prototype", "control")
                outputs = {}
                for name in first:
                    arm = arms[name]
                    result, state, m = arm.process(case)
                    m["parent_cpu_after"] = thread_cpu_now()
                    archive_files = arm.retire(case, state, m)
                    outputs[name] = (result, state, archive_files)
                    snap = arm.snapshot()
                    row = dict(kind="v11", pass_index=index, warmup=index == 0, cold=k < COLD_FRAMES, frame_id=case.frame_id,
                               arm=name, position=first.index(name), metrics=m, rss=self.rss([arm]),
                               queue=dict(scorer_retained=snap["scorer"].get("retained_slots", snap["scorer"].get("retained_jobs")),
                                          producer_retained=snap["producer"]["retained_jobs"] if "retained_jobs" in snap["producer"]
                                          else snap["producer"]["engine"]["retained_jobs"]))
                    self.rows.append(row)
                c_res, c_state, c_files = outputs["control"]
                p_res, p_state, p_files = outputs["prototype"]
                t = truth[case.frame_id]
                checks = dict(identity=c_state["key"] == p_state["key"],
                              semantic=c_state["prepared"]["semantic_sha256"] == p_state["semantic"],
                              commit=c_state["committed"] == p_state["committed"],
                              producer_archive_files=c_files == p_files,
                              q_perc=c_res["q_perc"] == p_res["q_perc"] == t["q_perc"],
                              measurement=C.canonical(c_res["measurement"]) == C.canonical(p_res["measurement"])
                                  == t["measurement"],
                              registration_key=c_state["key"] == SC.registration_key(c_state["expected"]))
                self.rows[-1]["exact"] = checks
                if not all(checks.values()):
                    raise AssertionError(f"exactness failure pass {index} frame {case.frame_id}: {checks}")
                # Attribution only (outside every measured interval): digest share of prototype freeze.
                h0 = raw()
                for chunk in p_state["chunks"]:
                    view = memoryview(chunk).cast("B")
                    if len(view) >= 64:
                        hashlib.sha256(view).hexdigest()
                for row in self.rows[-2:]:
                    if row["arm"] == "prototype":
                        row["metrics"]["UE_digest_only_ms"] = ms(h0, raw())
            self.record_settings(f"v11_pass_{index}_end", arms.values())
        finally:
            self.pin_parent(False)
        self.cleanup.append(dict(label=f"v11_pass_{index}", **{n: a.close() for n, a in arms.items()}))
        shutil.rmtree(d)   # durable payloads already read back and hash-bound by receipts

    def retention_pass(self, corpus, cls, lag=2):
        d = self.out / f"retention_{cls.name}"
        arm = cls(corpus, d)
        self.pin_parent(True)
        try:
            pending = []
            for k, case in enumerate(corpus.cases):
                result, state, m = arm.process(case)
                pending.append((case, state, m))
                snap = arm.snapshot()
                self.rows.append(dict(kind="retention", arm=cls.name, frame_id=case.frame_id, lag=lag, rss=self.rss([arm]),
                    producer={k2: (snap["producer"] if cls is ControlArm else snap["producer"]["engine"]).get(k2)
                              for k2 in ("retained_jobs", "retained_payload_bytes")},
                    scorer={k2: snap["scorer"].get(k2) for k2 in ("retained_jobs", "retained_slots", "reserved_bytes",
                            "logical_bytes", "waiting_predictions", "states")}, archive_outstanding=arm.archive.outstanding_jobs))
                if len(pending) > lag:
                    case0, state0, m0 = pending.pop(0)
                    arm.retire(case0, state0, m0)
            for case0, state0, m0 in pending:
                arm.retire(case0, state0, m0)
        finally:
            self.pin_parent(False)
        self.cleanup.append(dict(label=f"retention_{cls.name}", result=arm.close()))
        shutil.rmtree(d)

    def live100_pass(self, corpus, frames, index):
        d = self.out / f"live100_pass_{index}"
        binding = (frames[0]["metadata"]["run_binding"], frames[0]["metadata"]["producer_binding"])
        order = (ControlArm, ProtoArm) if index % 2 == 0 else (ProtoArm, ControlArm)
        arms = {}
        for cls in order:
            kw = dict(scorer_jobs=48, live100=binding)
            kw["with_producer" if cls is ControlArm else "with_prep"] = False
            arms[cls.name] = cls(corpus, d / cls.name, **kw)
        self.pin_parent(True)
        try:
            self.record_settings(f"live100_pass_{index}", arms.values())
            for k, f in enumerate(frames):
                first = ("control", "prototype") if (k + index) % 2 == 0 else ("prototype", "control")
                results = {}
                s = f["score"]
                labels = np.frombuffer(f["raw"]["predicted_labels.u8"], np.uint8).reshape(720, 1280)
                context = f["metadata"]["frame_context"]
                contract = f["metadata"]["prediction_contract"]
                for name in first:
                    m = {}
                    if name == "control":
                        arm = arms[name]
                        key = f["identity"]
                        cpu0, t0 = time.thread_time_ns(), raw()
                        inbox_expected = EL._expected(key)
                        R.decode_target(f["packet"], **SC._expected(inbox_expected))
                        R.decode_target(f["packet"], **SC._expected(inbox_expected))
                        arm.scorer.call("REGISTER", dict(key=key, expected=inbox_expected,
                                                         reservation_bytes=16 * 1024 * 1024))
                        arm.scorer.call("TARGET", dict(key=key, packet=f["packet"]))
                        m["TARGET_ms"], m["TARGET_parent_cpu_ms"] = ms(t0, raw()), (time.thread_time_ns() - cpu0) / 1e6
                        m["TARGET_child_cpu_ms"] = sum(e["worker"]["thread_cpu_ns"] for e in arm.scorer.events[-2:]) / 1e6
                        results[name] = arm.predict(key, s["prediction_identity"], labels, s["prediction_records"], m,
                                                    contract=contract)
                    else:
                        arm = arms[name]
                        cpu0, t0 = time.thread_time_ns(), raw()
                        fd, size = CH.seal_bytes([f["packet"]])
                        try:
                            reply = arm.score.call("TARGET", dict(identity=f["identity"].decode(),
                                semantic_sha256=f["semantic"], size=size), fd=fd)
                        finally:
                            os.close(fd)
                        SV.ok(reply, "TARGET")
                        m["TARGET_ms"], m["TARGET_parent_cpu_ms"] = ms(t0, raw()), (time.thread_time_ns() - cpu0) / 1e6
                        m["TARGET_child_cpu_ms"] = child_cpu(reply.timing)
                        results[name] = arm.predict(f["identity"].decode(), s["prediction_identity"], labels,
                                                    s["prediction_records"], context, m)
                    self.rows.append(dict(kind="live100", pass_index=index, warmup=index == 0, cold=k < COLD_FRAMES,
                                          frame=f["directory"], arm=name, position=first.index(name), metrics=m))
                checks = {n: (r["q_perc"] == s["q_perc"] and C.canonical(r["measurement"]) == C.canonical(s["measurement"]))
                          for n, r in results.items()}
                self.rows[-1]["exact"] = checks
                if not all(checks.values()):
                    raise AssertionError(f"live100 exactness failure {f['directory']}: {checks}")
            self.record_settings(f"live100_pass_{index}_end", arms.values())
        finally:
            self.pin_parent(False)
        # No archival path exists for live100 targets in the control scorer; both arms are aborted symmetrically.
        self.cleanup.append(dict(label=f"live100_pass_{index}", **{n: a.close(abort=True) for n, a in arms.items()}))
        shutil.rmtree(d)


def summarize(rows):
    out = {}
    for kind, stages in (("v11", ("UE", "DELIVERY", "PREPARE", "TARGET", "PREDICT", "ARCHIVE")), ("live100", ("TARGET", "PREDICT"))):
        sel = [r for r in rows if r["kind"] == kind and not r["warmup"]]
        block = {}
        for arm in ("control", "prototype"):
            mine = [r for r in sel if r["arm"] == arm]
            warm = [r for r in mine if not r["cold"]]
            block[arm] = {}
            for metric in sorted({k for r in mine for k in r["metrics"]}):
                if metric.endswith("_on"):
                    continue
                block[arm][metric] = dict(steady=stats([r["metrics"].get(metric) for r in warm]),
                                          cold=stats([r["metrics"].get(metric) for r in mine if r["cold"]]))
        key = "frame_id" if kind == "v11" else "frame"
        paired = {}
        for stage in stages:
            diffs = []
            for r in sel:
                if r["arm"] != "prototype" or r["cold"]:
                    continue
                c = next(x for x in sel if x["arm"] == "control" and x["pass_index"] == r["pass_index"] and x[key] == r[key])
                diffs.append(r["metrics"][stage + "_ms"] - c["metrics"][stage + "_ms"])
            paired[stage + "_ms_prototype_minus_control"] = stats(diffs)
        block["paired"] = paired
        block["position_effect"] = {arm: {pos: stats([r["metrics"].get("PREDICT_ms") for r in sel if r["arm"] == arm
                                                      and r["position"] == pos and not r["cold"]]) for pos in (0, 1)}
                                    for arm in ("control", "prototype")}
        out[kind] = block
    parent_cpus = sorted({r["metrics"].get("UE_parent_cpu_on") for r in rows if r["kind"] == "v11"} - {None})
    out["parent_cpu_observed"] = parent_cpus
    return out


def write(path, value):
    with open(path, "x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, default=str)
        stream.write("\n")


def harness_sources():
    return {str(p.relative_to(P.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PACKAGE.glob("*.py"))}


def seal_plan(directory):
    out = (P.EVIDENCE_ROOT / directory)
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(schema="splitfusion.exact_gt_service_proto.benchmark_plan.v1", harness_sources=harness_sources(),
        cpu=dict(prep_or_producer=CPU_PREP, scorer=CPU_SCORE, parent_thread=CPU_PARENT, archive_child="unpinned (inherits)"),
        passes=dict(v11=PASSES, live100=PASSES, warmup_pass=0, cold_frames_per_pass=COLD_FRAMES,
                    retention=["control lag 2", "prototype lag 2"]),
        ordering="per frame (k + pass) % 2; arm construction order alternates by pass",
        quiet_host=dict(max_load1=MAX_LOAD1, forbidden=list(FORBIDDEN)),
        stop_rules=["any exactness check fails", "any settings check fails", "host check fails", "no automatic retry"],
        exclusions=["control EXPORT_LIVE_SOURCE (live-only; offline recorded-clock mode used for exact outputs)",
                    "edge envelope ticket matching and quality_feedback formula check (identical in both arms, not reproducible offline)"],
        claims=dict(loaded_live=False, deadline_qualification=False))
    write(out / "PLAN.json", plan)
    (out / "PLAN_SHA256.txt").write_text(hashlib.sha256((out / "PLAN.json").read_bytes()).hexdigest() + "\n")
    return out


def run(directory, plan_dir):
    plan_path = P.EVIDENCE_ROOT / plan_dir / "PLAN.json"
    sealed = (P.EVIDENCE_ROOT / plan_dir / "PLAN_SHA256.txt").read_text().strip()
    C.require(hashlib.sha256(plan_path.read_bytes()).hexdigest() == sealed, "plan seal differs")
    plan = json.loads(plan_path.read_bytes())
    C.require(plan["harness_sources"] == harness_sources(), "harness sources differ from the sealed plan")
    P.environment_ok()
    out = P.EVIDENCE_ROOT / directory
    out.mkdir(parents=True, exist_ok=False)
    before = host_check()
    write(out / "HOST_BEFORE.json", before)
    C.require(before["loadavg"][0] <= MAX_LOAD1 and not before["forbidden_processes"]
              and not [n for n in before["docker_containers"] if any(t in n.lower() for t in ("oai", "carla", "perception"))],
              "host is not quiet")
    runner = Runner(out)
    fds_before = len(os.listdir("/proc/self/fd"))
    status = "STARTED"
    try:
        corpus = P.Corpus(49)
        truth = {}
        for case in corpus.cases:
            pred = corpus.predictions[case.frame_id]
            original = E.evaluate_prepared(identity=pred["identity"]["action_identity"], prepared=case.bundle.prepared,
                predicted_labels=pred["labels"], predictions=pred["records"], avo=case.avo_document["avo"])
            truth[case.frame_id] = dict(q_perc=pred["q_perc"], measurement=C.canonical(original.measurement))
        frames, _ = live100_frames()
        for index in range(PASSES):
            runner.v11_pass(corpus, index, truth)
        for index in range(PASSES):
            runner.live100_pass(corpus, frames, index)
        runner.retention_pass(corpus, ControlArm)
        runner.retention_pass(corpus, ProtoArm)
        status = "BENCHMARK_COMPLETE_ALL_EXACT"
    except BaseException as error:
        status = f"STOPPED: {type(error).__name__}: {error}"[:2000]
        raise
    finally:
        after = host_check()
        leftovers = subprocess.run(["pgrep", "-f", "splitfusion_exact_gt_service_proto_v2.worker|splitfusion_numeric_gt_runtime_v1.process|splitfusion_live_gt_handoff_v1.archive"],
                                   capture_output=True, text=True).stdout.split()
        with open(out / "FRAMES.jsonl", "x") as stream:
            for row in runner.rows:
                stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        write(out / "SETTINGS.json", runner.settings)
        write(out / "CLEANUP.json", dict(passes=runner.cleanup, leftover_pids=leftovers, parent_fds_before=fds_before,
                                        parent_fds_after=len(os.listdir("/proc/self/fd")),
                                        parent_affinity_restored=sorted(os.sched_getaffinity(0)) == runner.original_affinity))
        write(out / "HOST_AFTER.json", after)
        write(out / "RESULT.json", dict(status=status, summary=summarize(runner.rows) if status.startswith("BENCHMARK") else None,
            plan_sha256=sealed, python=sys.version, platform=platform.platform(), numpy=np.__version__,
            claims=dict(loaded_live=False, deadline_qualification=False, quiet_host_only=True)))
        write(out / "OUTPUT_SHA256.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in sorted(out.iterdir()) if p.is_file()})
    print(status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal-plan")
    parser.add_argument("--run")
    parser.add_argument("--plan")
    args = parser.parse_args()
    if args.seal_plan:
        print(seal_plan(args.seal_plan))
    else:
        run(args.run, args.plan)
