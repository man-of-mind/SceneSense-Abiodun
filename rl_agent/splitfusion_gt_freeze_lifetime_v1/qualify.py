"""Offline real-worker parity and 600-request lifetime soak. No radio/GPU.

The soak reindexes one retained source, prepares and cancels each transaction.
It is a lifetime/resource test, NOT 600 distinct scenes or a full route replay.
Only the 49-frame parity pass commits/scores/retires genuine retained frames.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

from rl_agent.splitfusion_exact_gt_service_proto_v2 import channel as CH, freeze_codec as FZ
from rl_agent.splitfusion_exact_gt_service_proto_v2 import service as SV, parity_common as P
from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C

ENTRY = "rl_agent.splitfusion_gt_freeze_lifetime_v1.worker_entry"
BASE = "rl_agent.splitfusion_m3_route_b_v4.worker_entry"


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


class Client(SV.WorkerClient):
    def __init__(self, config, directory, entry):
        directory.mkdir()
        self.role, self.timeout = "prep", 120.0
        self.nonce, self.seq, self.pending = secrets.token_hex(32), 0, None
        self.failed = self.closed = False
        parent, child = CH.pair()
        self.sock = parent
        self.log = (directory / "worker.log").open("xb")
        env = dict(os.environ, **SV.ENV)
        env.pop("PYTHONPATH", None)
        self.command = [sys.executable, "-B", "-u", "-m", entry,
                        "--fd", str(child.fileno()), "--parent", str(os.getpid()),
                        "--role", "prep", "--gc-policy", "candidate", "--trace-dir", str(directory),
                        "--max-frames", "3900", "--producer", "v2"]
        try:
            self.process = subprocess.Popen(self.command, cwd=SV.ROOT, env=env,
                pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log)
            child.close()
            CH.send(parent, dict(nonce=self.nonce, role="prep", config=config), deadline=time.monotonic() + self.timeout)
            ready, extra = CH.receive(parent, deadline=time.monotonic() + self.timeout)
            C.require(extra is None and ready.get("status") == "READY" and ready.get("nonce") == self.nonce
                      and ready.get("pid") == self.process.pid and ready.get("role") == "prep"
                      and ready.get("cuda_initialized") is False, "worker READY mismatch")
            self.ready = ready["ready"]
        except BaseException:
            child.close()
            self.abort()
            raise


def files(reply):
    try:
        return CH.unpack_files(reply.result["index"], reply.fd)
    finally:
        os.close(reply.fd)


def parity(corpus, directory, entry):
    client = Client(dict(engine=corpus.engine_config(offline=True)), directory, entry)
    rows = []
    try:
        for seq, case in enumerate(corpus.cases, 1):
            descriptor, chunks = FZ.freeze(corpus.sourcekwargs(case), case.kwargs["policy_timestamp_s"],
                                           session_id=client.ready["session_id"], sequence=seq)
            fd, _ = FZ.seal(chunks)
            try:
                SV.ok(client.call("FREEZE", dict(descriptor=descriptor.decode(), offline=corpus.offline_inputs(case)),
                                  fd=fd), "FREEZE")
            finally:
                os.close(fd)
            bound = SV.ok(client.call("BIND", dict(job=seq, plan=corpus.plan(case).decode(),
                frame_context=PO.context_document(corpus.context(case)))), "BIND")
            commit = SV.ok(client.call("CONFIRM_SEND", dict(job=seq, plan=corpus.plan(case).decode())), "CONFIRM_SEND")
            reply = client.call("PUBLISH", dict(job=seq))
            try:
                packet = CH.read_sealed(reply.fd, reply.result["size"])
            finally:
                os.close(reply.fd)
            scored = P.score_exact(corpus, case, packet)
            C.require(scored["q_perc_exact"] and scored["measurement_exact"] and scored["identity_key_exact"],
                      "retained score mismatch")
            SV.ok(client.call("RESULT_DELIVERED", dict(job=seq)), "RESULT_DELIVERED")
            archive = files(client.call("EVIDENCE", dict(job=seq)))
            prefix = f"frame_{case.frame_id}/producer"
            SV.ok(client.call("EVIDENCE_DURABLE", dict(job=seq, relative_prefix=prefix,
                receipt=P.harness_receipt(case.kwargs["identity"], prefix, archive))), "EVIDENCE_DURABLE")
            rows.append(dict(frame_id=case.frame_id, bound=bound, commit=commit,
                             packet_sha256=C.sha256(packet), score=scored,
                             archive_sha256={k:C.sha256(v) for k,v in archive.items()}))
        snapshot = SV.ok(client.call("SNAPSHOT"), "SNAPSHOT")
        SV.ok(client.close(), "CLOSE")
        return dict(rows=rows, history=snapshot["history_sha256"], returncode=client.process.returncode)
    finally:
        if not client.closed:
            client.abort()


def reindex(descriptor, frame, sequence):
    value = json.loads(descriptor)
    value["sequence"] = sequence
    value["frame_id"] = value["capture_template"]["frame_id"] = frame
    value["rgb"]["frame_id"] = value["world"]["frame_id"] = frame
    value["dynamic_snapshot_evidence"]["frame_id"] = frame
    for role in ("semantic_callback", "depth_callback", "radar_callback"):
        value["rgb"][role]["frame_id"] = frame
    for image in value["images"]:
        image["frame"] = frame
    return C.canonical(value).decode()


def resources(pid):
    root = Path("/proc") / str(pid)
    maps = (root / "maps").read_text()
    status = dict(line.split(":",1) for line in (root / "status").read_text().splitlines() if ":" in line)
    return dict(fd_count=len(list((root / "fd").iterdir())),
                mapped_source_count=maps.count("splitfusion_gt_snapshot_v4"), rss=status.get("VmRSS", "").strip())


def soak(corpus, directory, count):
    config = corpus.engine_config(offline=True)
    config["max_frames"] = 3900
    config["producer_args"]["max_frames"] = 3900
    client = Client(dict(engine=config), directory, ENTRY)
    case = corpus.cases[0]
    descriptor, chunks = FZ.freeze(corpus.sourcekwargs(case), case.kwargs["policy_timestamp_s"],
                                   session_id=client.ready["session_id"], sequence=1)
    fd, _ = FZ.seal(chunks)
    samples = []
    initial = SV.ok(client.call("SNAPSHOT"), "SNAPSHOT")["history_sha256"]
    try:
        for index in range(count):
            args = dict(descriptor=reindex(descriptor, case.frame_id + index, index+1),
                        offline=corpus.offline_inputs(case))
            SV.ok(client.call("FREEZE", args, fd=fd), "FREEZE")
            result = SV.ok(client.call("CANCEL_UNBOUND", dict(job=index+1, reason="SYNTHETIC_LIFETIME_SOAK")), "CANCEL")
            C.require(result["history_sha256"] == initial, "cancel changed history")
            if index in (0, 9, 99, 529, count-1):
                samples.append(dict(request=index+1, **resources(client.process.pid)))
                print(json.dumps(dict(stage="real_worker_soak", **samples[-1])), flush=True)
        # Deliberately fail AFTER decode at request count+1, through the real
        # preparation implementation. The original error must survive cleanup.
        args = dict(descriptor=reindex(descriptor, case.frame_id+count, count+1),
                    offline=dict(archive_clocks={"invalid": 0}))
        error = None
        try:
            client.call("FREEZE", args, fd=fd)
        except SV.WorkerFailed as exc:
            error = str(exc)
        C.require(error is not None and "seven actual recorded AVO clocks required" in error
                  and "BufferError" not in error, "original failure not preserved: " + str(error))
        client.process.wait(timeout=30)  # allow the retained worker to write its trace
        failure = json.loads((directory / "freeze_failure_v1" / "ERROR.json").read_bytes())
        trace = json.loads((directory / "prep_gc_trace.json").read_bytes())
        C.require(trace["restored_equals_initial"] is True, "worker GC state not restored")
        C.require(all(s["mapped_source_count"] == 0 for s in samples), "mapping leak in soak")
        C.require(len({s["fd_count"] for s in samples}) == 1, "FD growth in soak")
        return dict(prepared_and_cancelled=count, source_scenes=1, committed_and_scored=0,
                    synthetic_reindexed=True, samples=samples, original_failure=error,
                    failure_snapshot_exception=failure["exception"], worker_returncode=client.process.returncode,
                    gc_restored=trace["restored_equals_initial"], trace_written=True)
    finally:
        os.close(fd)
        client.abort()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--soak", type=int, default=600)
    args=parser.parse_args()
    P.environment_ok()
    C.require(531 <= args.soak <= 3900, "bounded soak beyond observed failure required")
    out=args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    corpus=P.Corpus(49)
    result=dict(status="STARTED", live_run=False, radio_changed=False)
    try:
        old=parity(corpus, out / "control", BASE)
        write(out / "CONTROL.json", old)
        new=parity(corpus, out / "candidate", ENTRY)
        write(out / "CANDIDATE.json", new)
        C.require(old == new, "prep lifetime repair changed retained parity")
        result.update(parity_frames=len(old["rows"]), parity_exact=True,
                      soak=soak(corpus, out / "soak", args.soak), status="PASS")
    except BaseException as exc:
        result.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["sources"]={p.name:C.sha256(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")}
        write(out / "RESULT.json", result)
        write(out / "OUTPUT_SHA256.json", {str(p.relative_to(out)):C.sha256(p.read_bytes())
              for p in out.rglob("*") if p.is_file() and p.name != "OUTPUT_SHA256.json"})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
