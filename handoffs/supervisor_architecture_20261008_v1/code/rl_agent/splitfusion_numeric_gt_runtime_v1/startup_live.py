"""CPU-only startup import/IPC checks before spending a live attempt."""
import argparse
import importlib
import json
import os
from pathlib import Path
from . import live_contract as C
from .live_host import write


def run(output):
    root = Path(__file__).resolve().parents[2]
    output = Path(output).absolute()
    if output.exists() or not output.resolve().is_relative_to(root):
        raise ValueError("fresh owned CPU startup proof required")
    sources = C.source_inventory(root)
    checks = {}
    # Import every actual entry from the deployed tree, never from PYTHONPATH.
    for name in (C.CHILD, C.EDGE, C.RUNNER, C.PARTITION,
                 "rl_agent.splitfusion_numeric_gt_runtime_v1.ue_live",
                 "rl_agent.splitfusion_numeric_gt_runtime_v1.edge_live"):
        module = importlib.import_module(name)
        checks[name] = Path(module.__file__).resolve().is_relative_to(root)
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("CPU startup import initialized CUDA")
    from .process import Client, LIVE_PROCESS_CONTRACT
    output.mkdir(parents=True, exist_ok=False)
    from .profiled_startup import check as profiled_check
    composition = profiled_check(output / "profiled_composition")
    checks["actual_profiled_numeric_class_and_empty_drain"] = composition["actual_profiled_numeric_class"]
    worker = Client("echo_test_only", {"offline_test": True},
        directory=output / "worker", source_sha256=C.worker_sources(),
        cpu=min(os.sched_getaffinity(0)))
    try:
        expected = dict(process_contract=LIVE_PROCESS_CONTRACT, payload=b"bounded-ipc" * 1000)
        checks["actual_cpu_child_ipc"] = worker.call("ECHO", expected) == expected
        closed = worker.close()
        checks["actual_cpu_child_exit"] = worker.process.returncode == 0 and closed == {}
    finally:
        if not worker.closed:
            worker.abort()
    checks["sources_unchanged"] = sources == C.source_inventory(root)
    checks["cuda_uninitialized"] = not torch.cuda.is_initialized()
    result = dict(schema="splitfusion.numeric_gt.live_startup.v1", passed=all(checks.values()),
        checks=checks, sources=sources, worker_ready=worker.ready,
        profiler_composition=composition,
        live_services_started=False, cuda_initialized=torch.cuda.is_initialized(),
        proof_scope="DEPLOYED_IMPORTS_AND_REAL_CPU_IPC__NOT_A_LIVE_OR_GPU_QUALIFICATION")
    write(output / "RESULT.json", result)
    print(json.dumps(dict(passed=result["passed"], checks=checks)))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    value = run(parser.parse_args().output)
    raise SystemExit(0 if value["passed"] else 1)
