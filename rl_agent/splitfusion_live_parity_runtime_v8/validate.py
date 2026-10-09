"""Source-bound fresh-process regressions before any live handoff launch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from . import contract as C
from .startup import write_x
from rl_agent.splitfusion_live_parity_runtime_v1.deploy import CPU_FIXTURE_FILES
from rl_agent.splitfusion_live_parity_runtime_v7.validate import CHILD

GROUPS = {name: {"directory": "rl_agent/"+package} for name, package in (
    ("integration_v8", "splitfusion_live_parity_runtime_v8"),
    ("handoff_v1", "splitfusion_live_gt_handoff_v1"),
    ("optimization_v5", "splitfusion_live_gt_optimization_v5"),
    ("optimization_v6", "splitfusion_live_gt_optimization_v6"),
    ("integration_v7", "splitfusion_live_parity_runtime_v7"),
    ("integration_v6", "splitfusion_live_parity_runtime_v6"),
    ("optimization_v3", "splitfusion_live_gt_optimization_v3"),
    ("optimization_v4", "splitfusion_live_gt_optimization_v4"),
    ("paired_replay_v1", "splitfusion_paired_quality_replay_v1"),
    ("paired_replay_v2", "splitfusion_paired_quality_replay_v2"),
    ("integration_v5", "splitfusion_live_parity_runtime_v5"),
    ("optimization_v2", "splitfusion_live_gt_optimization_v2"),
    ("integration_v4", "splitfusion_live_parity_runtime_v4"),
    ("optimization", "splitfusion_live_gt_optimization_v1"),
    ("integration_v3", "splitfusion_live_parity_runtime_v3"),
    ("integration_v2", "splitfusion_live_parity_runtime_v2"),
    ("integration_v1", "splitfusion_live_parity_runtime_v1"),
    ("repair_v2", "splitfusion_live_parity_repair_v2"),
    ("repair", "splitfusion_live_parity_repair_v1"),
    ("existing_run5", "splitfusion_run5_live_avo_v1"))}
GROUPS["frozen_semantics"] = {"names": [
    "rl_agent.splitfusion_live_target_parity_v1.test_adapter",
    "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase3_continuous_execution_v2",
    "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase4_reward_hold_v2",
    "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_live_integration_v2",
    "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_prewarm_gt_priority_v2"]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    root=Path(__file__).resolve().parents[2]
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    sources=C.source_inventory(root)
    fixtures={name:digest for name,(_size,digest) in CPU_FIXTURE_FILES.items()}
    if not all(not (root/name).is_symlink() and C.sha(root/name)==digest for name,digest in fixtures.items()):
        raise ValueError("finite CPU fixture drift")
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",
             OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH",None)
    results,logs={},{}
    with tempfile.TemporaryDirectory(prefix="parity-v8-tests-") as scratch:
        env["MPLCONFIGDIR"]=scratch
        for name,spec in GROUPS.items():
            cmd=[sys.executable,"-c",CHILD,json.dumps(spec)]
            done=subprocess.run(cmd,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                text=True,timeout=300)
            path=output/(name+".log")
            write_x(path,done.stdout.encode())
            records=[line[len("RESULT_JSON="):] for line in done.stdout.splitlines() if line.startswith("RESULT_JSON=")]
            result=json.loads(records[-1]) if records else dict(tests=0,skipped=0,skips=[])
            results[name]={**result,"returncode":done.returncode,"command":cmd}
            logs[path.name]=C.sha(path)
            print(json.dumps(dict(suite=name,**result,returncode=done.returncode)),flush=True)
    if sources!=C.source_inventory(root) or not all(C.sha(root/n)==d for n,d in fixtures.items()):
        raise ValueError("sources or fixtures changed during regression")
    report=dict(schema="splitfusion.parity_runtime.offline.v8",sources=sources,fixtures=fixtures,
        suites=results,logs=logs,passed=all(r["returncode"]==0 and r["tests"]>0 for r in results.values()),
        live_started=False,training_started=False,total_tests=sum(r["tests"] for r in results.values()))
    write_x(output/"OFFLINE_VERIFICATION.json",report)
    from .host_capture import verify_offline
    verify_offline(output/"OFFLINE_VERIFICATION.json",root)
    print(json.dumps(dict(status="OFFLINE_PASS",total_tests=report["total_tests"])))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
