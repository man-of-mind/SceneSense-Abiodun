"""Pre-launch prerequisite evidence (create-only, CPU only, no CARLA/OAI/GPU).

  offline   : all unit suites + fresh step-1..3 parity runs, bound to exact sources
  (startup proof: provided by the fixed-action screen v2, which owns the live entry modules)
  container : the admitted edge image (no GPU) connects to the real host service over the
              bind-mounted socket, sends a real v11 prediction as a sealed memfd, and the
              exact result returns; ownership, peer credentials and cleanup are verified
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import contract as C

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_exact_gt_live_v2"
ENV = dict(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def _out(name):
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    return out


def _seal(out, exclude=("OUTPUT_SHA256.json",)):
    manifest = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*"))
                if p.is_file() and p.name not in exclude and "archive" not in p.parts and "durable" not in p.parts}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))


def _env():
    env = dict(os.environ, **ENV)
    env.pop("PYTHONPATH", None)
    return env


def offline(name):
    out = _out(name)
    sources = C.exact_sources(ROOT)
    suites = ["rl_agent.splitfusion_exact_gt_service_proto_v2.test_deferred",
              "rl_agent.splitfusion_exact_gt_service_proto_v2.test_freeze_codec",
              "rl_agent.splitfusion_exact_gt_service_proto_v2.test_lifecycle",
              "rl_agent.splitfusion_exact_gt_service_proto_v2.test_anchor",
              "rl_agent.splitfusion_ue_input_prep_v1.test_candidate"]
    rows, tests, failures, errors, skipped = [], 0, 0, 0, 0
    scratch = out / "tmp"
    scratch.mkdir()
    for suite in suites:
        log = out / (suite.rsplit(".", 1)[-1] + ".log")
        code = subprocess.run([sys.executable, "-W", "ignore::ResourceWarning", "-m", "unittest", "-v", suite], cwd=ROOT,
                              env=dict(_env(), TMPDIR=str(scratch)), stdout=open(log, "x"), stderr=subprocess.STDOUT).returncode
        text = log.read_text()
        ran = int(text.rsplit("Ran ", 1)[1].split()[0]) if "Ran " in text else 0
        f = int(text.split("failures=")[1].split(")")[0].split(",")[0]) if "failures=" in text else 0
        e = int(text.split("errors=")[1].split(")")[0].split(",")[0]) if "errors=" in text else 0
        s = int(text.split("skipped=")[1].split(")")[0].split(",")[0]) if "skipped=" in text else 0
        rows.append(dict(suite=suite, returncode=code, ran=ran, failures=f, errors=e, skipped=s))
        tests, failures, errors, skipped = tests + ran, failures + f, errors + e + (code != 0 and f == 0 and e == 0), skipped + s
    shutil.rmtree(scratch)
    parity = {}
    for module, label in (("verify_deferred", "step1"), ("verify_freeze", "step2"), ("verify_composition", "step3"),
                          ("verify_anchor", "anchor")):
        target = f"prereq_{name}_{label}"
        log = out / f"{label}.log"
        code = subprocess.run([sys.executable, "-W", "ignore::ResourceWarning", "-m",
                               f"rl_agent.splitfusion_exact_gt_service_proto_v2.{module}", "--name", target], cwd=ROOT,
                              env=_env(), stdout=open(log, "x"), stderr=subprocess.STDOUT).returncode
        status = json.loads((ROOT / "experiments/splitfusion_exact_gt_service_proto_v2" / target / "RESULT.json").read_bytes()).get("status")
        parity[label] = dict(returncode=code, status=status, evidence=target)
    ok = (failures == 0 and errors == 0 and all(r["returncode"] == 0 for r in rows)
          and all(p["returncode"] == 0 and p["status"].endswith("EXACT") for p in parity.values())
          and C.exact_sources(ROOT) == sources)
    result = dict(schema="splitfusion.exact_gt_live_v2.offline_proof.v1", status="PASS" if ok else "FAIL", tests=tests,
                  failures=failures, errors=errors, skipped=skipped, suites=rows, parity=parity, sources=sources,
                  live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    _seal(out)
    print(json.dumps({k: result[k] for k in ("status", "tests", "failures", "errors", "skipped", "parity")}, indent=1))


CONTAINER_SCRIPT = r'''
import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
base = Path("/work/torch_cache/check")
cfg = json.loads((base / "ticket.json").read_text())
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from rl_agent.splitfusion_live_dispatch_v1.frame_context import FrameContextV1, Pose6D, validate_frame_context
def context_from_document(document):
    value = FrameContextV1(**{k: v for k, v in document.items() if k != "ego_world"}, ego_world=Pose6D(*document["ego_world"]))
    validate_frame_context(value)
    return value
from rl_agent.splitfusion_exact_gt_live_v2.edge import install_early_gt_edge, EarlyGtEvaluatorV1
spec = W.load_run4_quality_spec(Path("/work/abiodun"))
payloads = []
ev_dir = Path("/work/torch_cache/segmentation_evidence")
with install_early_gt_edge(socket_path=ev_dir / cfg["socket"], capability_path=ev_dir / cfg["capability"], connect_timeout_s=60):
    ev = E.Run4EvaluatorV2(spec=spec, read_ground_truth=None, send=payloads.append, match_distance_m=2., gt_timeout_s=2.)
    ev.start()
    envelope = NS(**cfg["action"], capture_timestamp_ns=cfg["capture"]["capture_timestamp_ns"])
    labels = np.load(base / "labels.npy", allow_pickle=False)
    ticket = E.EvaluationTicketV2(envelope=envelope, context=context_from_document(cfg["context"]), records=tuple(cfg["records"]),
                                  predicted_mask=labels, gt_identity=cfg["capture"], enqueued_wall_ns=time.time_ns())
    ev._materialized[int(cfg["capture"]["frame_id"])] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    ev.submit(ticket)
    deadline = time.monotonic() + 60
    while not payloads and time.monotonic() < deadline:
        time.sleep(0.01)
    report = ev.close()
feedback, reason = W.decode_feedback(payloads[0]) if payloads else (None, None)
out = dict(uid=os.getuid(), pid=os.getpid(), payloads=len(payloads), q_perc=None if feedback is None else feedback.q_perc,
           kind=None if feedback is None else feedback.kind, reason=None if reason is None else reason.name,
           failure=report.get("failure"), service_results=report.get("service_results"))
(base / "container_result.json").write_text(json.dumps(out))
'''


def container(name, mount_root=None):
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_edge_launch_v2 import ADMITTED_IMAGE_ID
    from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
    from .rehearsal import Rig
    import numpy as np
    out = _out(name)
    mount = Path(mount_root or ROOT).resolve(strict=True)
    sources = C.exact_sources(ROOT)
    if mount != ROOT and C.exact_sources(mount) != sources:
        raise ValueError("mounted tree differs from the tested exact sources")
    corpus = P.Corpus(49)
    limits = dict(C.LIMITS, max_frames=49, horizon_ms=10000, drain_s=60, prep_cpu=None, score_cpu=None, service_cpu=None)
    cache = out / "torch_cache"
    rig = Rig(out, corpus, limits, evidence=cache / "segmentation_evidence")
    result, error = None, None
    try:
        row = rig.admit(0)
        time.sleep(1.0)
        check = cache / "check"
        check.mkdir()
        case = corpus.cases[0]
        pred = corpus.predictions[case.frame_id]
        np.save(check / "labels.npy", pred["labels"], allow_pickle=False)
        (check / "ticket.json").write_text(json.dumps(dict(action=row["action"], capture=row["capture"],
            context=PO.context_document(row["context"]), records=pred["records"], socket=C.SOCKET_NAME,
            capability=C.CAPABILITY_NAME)))
        (check / "script.py").write_text(CONTAINER_SCRIPT)
        os.chmod(cache, 0o777); os.chmod(check, 0o777)
        command = ["sudo", "-n", "docker", "run", "--rm", "--network", "none", "--entrypoint", "python3",
                   "-v", f"{mount}:/work/abiodun:ro", "-v", f"{cache.resolve()}:/work/torch_cache:rw", "--workdir", "/work/abiodun",
                   *sum((["-e", f"{k}={v}"] for k, v in ENV.items()), []), "-e", "PYTHONDONTWRITEBYTECODE=1", ADMITTED_IMAGE_ID,
                   "-c", "exec(open('/work/torch_cache/check/script.py').read())"]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=300)
        (out / "container.log").write_text(proc.stdout[-20000:] + proc.stderr[-40000:])
        inside = json.loads((check / "container_result.json").read_text()) if (check / "container_result.json").exists() else {}
        rig.feedback[case.frame_id] = dict(q_perc=inside.get("q_perc"), received=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))
        rig.terminal(case.frame_id)
        ue_result = rig.h.close()
        service = json.loads((rig.ue_dir / "numeric_gt/service/SERVICE_RESULT.json").read_text())
        frames = [json.loads(l) for l in open(rig.ue_dir / "numeric_gt/service/FRAMES.jsonl")]
        checks = dict(container_exit_zero=proc.returncode == 0, container_root=inside.get("uid") == 0,
            exact_quality=inside.get("q_perc") == pred["q_perc"] and frames[0].get("outcome") == "SCORED"
                and frames[0]["result"]["q_perc"] == pred["q_perc"],
            peer_credentials_recorded=(service.get("edge") or {}).get("uid") == 0,
            socket_and_capability_removed=not (cache / "segmentation_evidence" / C.SOCKET_NAME).exists()
                and not (cache / "segmentation_evidence" / C.CAPABILITY_NAME).exists(),
            service_clean=not ue_result["errors"] and ue_result["service_returncode"] == 0 and not service["errors"],
            edge_no_failure=inside.get("failure") is None and inside.get("service_results") == 1)
        result = dict(schema="splitfusion.exact_gt_live_v2.container_proof.v1", status="PASS" if all(checks.values()) else "FAIL",
            checks=checks, inside=inside, image=ADMITTED_IMAGE_ID, command=command, sources=sources, mounted_tree=str(mount),
            live_services_started=False,
            gpu=False, network="none")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        if rig.h.process is not None and rig.h.process.poll() is None:
            rig.h.abort(exc)
        result = dict(status="FAIL", error=error, live_services_started=False)
        raise
    finally:
        subprocess.run(["sudo", "-n", "chown", "-R", f"{os.getuid()}:{os.getgid()}", str(cache)], timeout=60)
        leftovers = subprocess.run(["sudo", "-n", "docker", "ps", "-q", "--filter", f"ancestor={ADMITTED_IMAGE_ID}"],
                                   capture_output=True, text=True).stdout.split()
        result = dict(result or {}, leftover_containers=leftovers,
                      root_owned_after_chown=[str(p) for p in cache.rglob("*") if p.stat().st_uid == 0])
        if leftovers or result["root_owned_after_chown"]:
            result["status"] = "FAIL"
        (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
        _seal(out)
    print(json.dumps({k: result.get(k) for k in ("status", "checks", "error")}, indent=1, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("offline", "container"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--mount-root", default=None)
    args = parser.parse_args()
    if args.kind == "container":
        container(args.name, args.mount_root)
    else:
        offline(args.name)
