"""Candidate UE seam: start the unchanged host early-GT service through the candidate service trampoline.

Versioned copy of ``splitfusion_m3_route_b_v7.gc_seams``; only the contract it reads differs (candidate id and
trampolines).  Popen rewriting, restoration and the substitution record are unchanged.  Offline proof use only.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import types

from . import contract as C


def seam_parameters(registration):
    """The seam's registration check (also run by the verification-only child entry)."""
    gc_reg = registration["gt_gc"]
    arm, producer = gc_reg["arm"], gc_reg.get("producer")
    if arm not in C.ARMS or producer not in C.PRODUCERS or registration.get("id") != C.ID:
        raise ValueError("GC-qualification seam requires its sealed registration")
    if gc_reg.get("trampolines") != dict(service=C.SERVICE_ENTRY, worker=C.WORKER_ENTRY):
        raise ValueError("registered GT service/worker trampolines differ from the candidate contract")
    return dict(arm=arm, producer=producer, budget=int(registration["transmitted_budget"]),
                service_entry=C.SERVICE_ENTRY, substituted=C.SERVICE_MODULE)


def rewrite_service_argv(argv, *, arm, trace_dir, max_frames, producer):
    argv = list(argv)
    hits = [i for i in range(len(argv) - 1) if argv[i] == "-m" and argv[i + 1] == C.SERVICE_MODULE]
    if len(hits) != 1:
        return None
    i = hits[0]
    return argv[:i] + ["-m", C.SERVICE_ENTRY] + argv[i + 2:] + [
        "--gc-policy", arm, "--trace-dir", str(trace_dir), "--max-frames", str(int(max_frames)), "--producer", producer]


@contextmanager
def install_ue_service_seam(*, registration, output):
    from rl_agent.splitfusion_exact_gt_live_v2 import ue as UE
    parameters = seam_parameters(registration)
    arm, producer, budget = parameters["arm"], parameters["producer"], parameters["budget"]
    trace_dir = Path(output) / C.TRACE_DIR
    trace_dir.mkdir(exist_ok=False)
    records = []
    original = UE.subprocess

    def popen(argv, *args, **kwargs):
        rewritten = rewrite_service_argv(argv, arm=arm, trace_dir=trace_dir, max_frames=budget, producer=producer)
        if rewritten is not None:
            records.append(dict(argv=rewritten[1:]))
            argv = rewritten
        return subprocess.Popen(argv, *args, **kwargs)

    proxy = types.ModuleType("subprocess_service_substitution")
    proxy.__dict__.update({k: v for k, v in vars(subprocess).items() if not k.startswith("__")})
    proxy.Popen = popen
    UE.subprocess = proxy
    error = None
    try:
        yield trace_dir
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"[:500]
        raise
    finally:
        UE.subprocess = original
        with open(trace_dir / "ue_substitution.json", "x") as stream:
            json.dump(dict(schema="splitfusion.m3_route_b_v3.ue_substitution.v1", arm=arm, producer=producer,
                           substitutions=records, restored=UE.subprocess is subprocess, error=error,
                           seam_module=__name__, contract_module=C.__name__), stream, indent=1, sort_keys=True)
