"""M3.1: run the ONE registered full Route-B GOOD_ALL capture exactly once, then the M3 integration verdict and the
retention postflight. No retry on any outcome; an integrity/infrastructure failure keeps its evidence and stops.

Before launch: FREEZE.json binds the prepared registration/prospective/prerequisite/acceptance/plan files, the complete
runtime source inventory, this driver, the M3 verdict/geometry modules, the acceptance source and the storage plan.
Preconditions: fresh root bound to GOOD_ALL + frame ceiling 3,900; free disk >= the registered requirement; host
MemAvailable floor; no leftover GT/trampoline process; exclusive live window (polled up to 10 min, else hard blocker,
nothing launched). Header of the copied channel-response campaign follows.

Channel response v1 (pinned FADE_RECOVERY a71 cell; launch refused unless the registration binds it). Original header:

Run the ONE registered 1,000-frame retention capture (gc_v2 configuration + pre-registered paired-input retention),
exactly once, and stop. After the retained integration verdict, a CPU-only postflight seals the retention manifest
digest (read from the child's RETENTION_RESULT.json and re-hashed), verifies the archive (strict=False: missing samples
are reported, never repaired) and resolves every retained record to the numeric-GT archive with exact rescoring
(GT_REFERENCES.json, written into this evidence directory; the capture is read-only).

Original gc_v2 header: Run the ONE registered 1,000-frame validation (adopted GC policy + numeric producer v2), exactly once, and stop.
Requires explicit approval. Versioned copy of the GC-pair driver with a single root "validation"; original header:

Run the ONE registered control/candidate pair of the GT GC live qualification, exactly once, and stop.

Before any frame: FREEZE.json binds both prepared registrations/prospective records, the complete runtime source
inventory, this driver and the acceptance file. Per arm, in order control -> candidate:
  root fresh; storage (registered plan) and host MemAvailable floor; exclusive live window (polled up to 10 min,
  else hard blocker, no launch); ONE launch through the retained host lifecycle; the retained actor-driven
  integration verdict (unchanged) and the GC-trace check in separate CPU-only processes; post-run window and leftover
  GT/trampoline processes. Between arms: settle >= 120 s and until the live window is exclusive with loadavg <= 2.0
  (up to 15 min). A control integration STOP, a crash, leftovers or a non-exclusive window stop the campaign
  (completed evidence kept, no retry). No implementation or acceptance change after frames are transmitted.
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

from rl_agent.splitfusion_fixed_action_screen_analysis_v3.campaign import live_window

MAIN = Path(__file__).resolve().parents[2]
WT = MAIN.parent / "abiodun_run4_single_gpu_profiler_v1"
ROOTS = WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261007_m3_route_b_v3"
EVIDENCE = MAIN / "experiments/splitfusion_m3_route_b_v3"
STRIP = ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD")
ARMS = ("good_all",)
GT_PATTERN = "splitfusion_m3_route_b_v3|splitfusion_exact_gt_live_v2.service|splitfusion_exact_gt_service_proto_v2.worker"
POSTFLIGHT = ("import json,sys,hashlib\n"
              "from pathlib import Path\n"
              "from rl_agent.splitfusion_m3_route_b_v3 import contract as C\n"
              "from rl_agent.splitfusion_paired_retention_v2 import validate as V, gt_adapter as G\n"
              "root, out = Path(sys.argv[1]), Path(sys.argv[2])\n"
              "reg = json.loads((root / 'registration.json').read_bytes())\n"
              "plan_sha = reg['paired_retention']['plan_sha256']\n"
              "res = json.loads((root / C.RETENTION_RESULT_FILE).read_bytes())\n"
              "man = root / C.RETENTION_DIR / 'MANIFEST.json'\n"
              "digest = hashlib.sha256(man.read_bytes()).hexdigest() if man.exists() else None\n"
              "doc = dict(retention_result=res, manifest_sha256=digest, manifest_matches_child=digest == res.get('manifest_sha256'),\n"
              "           plan_sha256=plan_sha)\n"
              "if digest is not None and doc['manifest_matches_child']:\n"
              "    m = V.verify_archive_v2(root / C.RETENTION_DIR, expected_manifest_sha256=digest, expected_plan_sha256=plan_sha,\n"
              "                            plan_path=root / C.RETENTION_PLAN_FILE, strict=False)\n"
              "    doc['verify_strict_false'] = dict(status_counts=m['status_counts'], records=len(m['records']),\n"
              "                                      not_retained=m['not_retained'], observed=m['observed_policy_decisions'])\n"
              "    try:\n"
              "        V.verify_archive_v2(root / C.RETENTION_DIR, expected_manifest_sha256=digest, expected_plan_sha256=plan_sha)\n"
              "        doc['verify_strict'] = 'PASS'\n"
              "    except Exception as exc:\n"
              "        doc['verify_strict'] = f'REFUSED: {exc}'\n"
              "    cell = next((root / 'capture/cells').iterdir())\n"
              "    refs = G.resolve_gt_references(root / C.RETENTION_DIR, manifest_sha256=digest, plan_sha256=plan_sha,\n"
              "                                   archive_root=cell / 'numeric_gt/service/archive', output_path=out / 'GT_REFERENCES.json')\n"
              "    doc['gt_references'] = dict(sha256=refs['references_sha256'], status_counts=refs['status_counts'],\n"
              "                                ordinal_crosscheck_available=refs['ordinal_crosscheck_available'])\n"
              "(out / 'RETENTION_POSTFLIGHT.json').write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))\n"
              "print(json.dumps({k: v for k, v in doc.items() if k != 'retention_result'}, default=str))\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event(log, **value):
    value["wall"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    log.write(json.dumps(value, sort_keys=True) + "\n")
    log.flush()
    print(json.dumps(value), flush=True)


def mem_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024


def leftovers():
    p = subprocess.run(["pgrep", "-af", GT_PATTERN], capture_output=True, text=True)
    return [l for l in p.stdout.splitlines() if "pgrep" not in l and "campaign" not in l]


def wait_window(log, *, timeout_s, max_load=None, min_wait_s=0):
    t0 = time.monotonic()
    while True:
        w = live_window()
        ok = w["exclusive"] and (max_load is None or w["loadavg_1min"] <= max_load)
        if ok and time.monotonic() - t0 >= min_wait_s:
            return w, True
        if time.monotonic() - t0 > timeout_s:
            return w, False
        time.sleep(30)


def freeze(out):
    from rl_agent.splitfusion_m3_route_b_v3 import contract as C   # main copy == worktree copy (deploy verified)
    record = dict(schema="splitfusion.m3_route_b_v3.freeze.v1", worktree=str(WT),
                  runtime_sources=C.source_inventory(WT),
                  arms={arm: {name: sha(ROOTS / arm / name) for name in
                              ("registration.json", "PROSPECTIVE.json", "PREREQUISITE.json", "ACCEPTANCE.json",
                               "arm.json", "compose_partition.yaml", C.RETENTION_PLAN_FILE)} for arm in ARMS},
                  campaign_sha256=sha(__file__), integration_sha256=sha(Path(__file__).with_name("integration.py")),
                  route_geometry_sha256=sha(Path(__file__).with_name("route_geometry.py")),
                  acceptance_source_sha256=sha(EVIDENCE / "ACCEPTANCE.json"),
                  storage_plan_sha256=sha(EVIDENCE / "STORAGE_PLAN.json"), frozen_before_launch=True)
    (out / "FREEZE.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    return record


def run(name):
    out = EVIDENCE / name
    out.mkdir(parents=False, exist_ok=False)
    log = open(out / "CAMPAIGN.jsonl", "x", buffering=1)
    frozen = freeze(out)
    event(log, kind="frozen", sources=len(frozen["runtime_sources"]))
    plan = json.loads((EVIDENCE / "STORAGE_PLAN.json").read_bytes())
    env = {k: v for k, v in os.environ.items() if k not in STRIP}
    check_env = dict(env, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    results = {}
    for index, arm in enumerate(ARMS):
        root = ROOTS / arm
        reg = json.loads((root / "registration.json").read_bytes())
        if (root / "capture").exists() or (root / "gt_gc_trace").exists() or reg["gt_gc"]["arm"] != "candidate" \
                or reg["gt_gc"]["producer"] != "v2" or (root / "paired_retention_v2").exists() \
                or (root / "RETENTION_RESULT.json").exists() or not reg.get("paired_retention") \
                or reg.get("transmitted_budget") != 3900 \
                or ((reg.get("channel") or {}).get("schedule") or {}).get("schedule_id") != "GOOD_ALL" \
                or ((reg.get("channel") or {}).get("schedule") or {}).get("target_snr_db") != 21.5:
            event(log, kind="stop", arm=arm, reason="ROOT_NOT_FRESH_OR_MISMATCHED")
            return 2
        free = shutil.disk_usage("/").free
        need = plan["required_before_launch"]
        mem = mem_available()
        event(log, kind="preflight", arm=arm, free_bytes=free, required_bytes=need, mem_available=mem,
              leftover_gt=leftovers())
        if free < need or mem < reg["gt_gc"]["safety"]["host_mem_available_min_bytes"] or leftovers():
            event(log, kind="stop", arm=arm, reason="STORAGE_MEMORY_OR_LEFTOVER_PRECONDITION")
            return 2
        if index == 0:
            window, ok = wait_window(log, timeout_s=600)
        else:
            window, ok = wait_window(log, timeout_s=900, max_load=2.0, min_wait_s=120)
        (out / f"window_{arm}.json").write_text(json.dumps(window, indent=1, sort_keys=True))
        event(log, kind="live_window", arm=arm, exclusive=window["exclusive"], loadavg=window["loadavg_1min"],
              gpu_mib=window["gpu_memory_used_mib"], containers=len(window["containers"]), ok=ok)
        if not ok:
            event(log, kind="stop", arm=arm, reason="LIVE_WINDOW_NOT_EXCLUSIVE_HARD_BLOCKER_NOT_LAUNCHED")
            return 2
        token = reg["arms"]["C2"]["execute_token"]
        event(log, kind="launch_start", arm=arm, root=str(root), budget=reg["transmitted_budget"], token=token)
        with open(out / f"launch_{arm}.log", "x") as stream:
            rc = subprocess.run([sys.executable, "-m", "rl_agent.splitfusion_m3_route_b_v3.host", "launch",
                                 "--output", str(root), "--execute", token], cwd=WT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT).returncode
        event(log, kind="launch_end", arm=arm, returncode=rc)
        code = ("import json,sys\n"
                "from pathlib import Path\n"
                "from rl_agent.splitfusion_m3_route_b_v3_analysis.integration import evaluate\n"
                "v = evaluate(sys.argv[1], launch_returncode=int(sys.argv[2]), launch_log=sys.argv[3], root=Path(sys.argv[4]))\n"
                "print(json.dumps(v, default=str, indent=1))\n"
                "sys.exit(0 if v['verdict'] == 'PASS_M3_FULL_ROUTE_INTEGRATION' else 3)\n")
        with open(out / f"VERDICT_{arm}.json", "x") as stream, open(out / f"verdict_{arm}.stderr", "x") as err:
            crc = subprocess.run([sys.executable, "-B", "-c", code, str(root), str(rc), str(out / f"launch_{arm}.log"),
                                  str(MAIN)],
                                 cwd=MAIN, env=check_env, stdout=stream, stderr=err).returncode
        left = leftovers()
        after = live_window()
        (out / f"window_after_{arm}.json").write_text(json.dumps(after, indent=1, sort_keys=True))
        event(log, kind="verdict_end", arm=arm, verdict_returncode=crc, leftover_gt=left,
              post_exclusive=after["exclusive"], post_containers=len(after["containers"]))
        results[arm] = crc
        with open(out / f"postflight_{arm}.log", "x") as stream:
            prc = subprocess.run([sys.executable, "-B", "-c", POSTFLIGHT, str(root), str(out)], cwd=MAIN, env=check_env,
                                 stdout=stream, stderr=subprocess.STDOUT).returncode
        event(log, kind="retention_postflight_end", arm=arm, returncode=prc)
        if crc != 0 or left:
            event(log, kind="stop", arm=arm, reason="INTEGRATION_OR_CLEANUP_FAILURE_COMPLETED_EVIDENCE_KEPT_NO_RETRY")
            return 3
    event(log, kind="capture_complete", results=results)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    raise SystemExit(run(parser.parse_args().name))
