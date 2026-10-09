"""Candidate trampoline for the UNCHANGED host early-GT service: versioned copy of the V8 ``service_entry``.

    python -m rl_agent.splitfusion_gt_storage_accounting_v3.service_entry --fd N --parent P
        --gc-policy control|candidate --trace-dir DIR --max-frames N --producer v1|v2

Identical to the V8 service trampoline except that it reads the candidate contract, so the retained ``WorkerClient``
spawn of ``-m rl_agent.splitfusion_exact_gt_service_proto_v2.worker`` is rewritten to the candidate ``worker_entry``
(prep: exact storage-accounting switch, then the UNCHANGED V8 worker trampoline -> tilt-aware AVO -> V7 FREEZE
lifetime repair -> V4 trampoline; score: the V8 trampoline).  Spawn-shape refusal, environment, capacity substitution
and the substitution record are unchanged.  Offline proof use only.
"""
import os
if __name__ == "__main__":
    os.environ.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})

import argparse
import json
from pathlib import Path
import subprocess
import threading
import types

from . import contract as C


class WorkerSubstitution:
    """Rewrites only the retained worker spawn; records every substitution."""

    def __init__(self, *, arm, trace_dir, max_frames, producer):
        self.arm, self.trace_dir, self.max_frames = arm, Path(trace_dir), int(max_frames)
        if producer not in C.PRODUCERS:
            raise ValueError("unknown producer")
        self.producer = producer
        self.records, self._role = [], threading.local()
        self._saved = None

    def rewrite(self, argv):
        argv = list(argv)
        hits = [i for i in range(len(argv) - 1) if argv[i] == "-m" and argv[i + 1] == C.WORKER_MODULE]
        role = getattr(self._role, "value", None)
        if len(hits) != 1 or role not in ("prep", "score"):
            raise RuntimeError(f"REFUSED: unexpected worker spawn shape/role ({len(hits)} module hits, role={role!r})")
        i = hits[0]
        policy = self.arm if role == "prep" else "control"
        producer = self.producer if role == "prep" else "v1"     # the score worker constructs no producer
        out = argv[:i] + ["-m", C.WORKER_ENTRY] + argv[i + 2:] + [
            "--role", role, "--gc-policy", policy, "--trace-dir", str(self.trace_dir), "--max-frames", str(self.max_frames),
            "--producer", producer]
        self.records.append(dict(role=role, gc_policy=policy, producer=producer, argv=out[1:]))
        return out

    def install(self):
        from rl_agent.splitfusion_exact_gt_service_proto_v2 import service as SV
        if self._saved is not None:
            raise RuntimeError("worker substitution already installed")
        original_init, original_subprocess = SV.WorkerClient.__init__, SV.subprocess
        substitution = self

        def init(client, role, config, **kwargs):
            substitution._role.value = role
            try:
                return original_init(client, role, config, **kwargs)
            finally:
                substitution._role.value = None

        def popen(argv, *args, **kwargs):
            return subprocess.Popen(substitution.rewrite(argv), *args, **kwargs)

        proxy = types.ModuleType("subprocess_worker_substitution")
        proxy.__dict__.update({k: v for k, v in vars(subprocess).items() if not k.startswith("__")})
        proxy.Popen = popen
        SV.WorkerClient.__init__, SV.subprocess = init, proxy
        self._saved = (SV, original_init, original_subprocess)
        return self

    def uninstall(self):
        if self._saved is None:
            return
        SV, original_init, original_subprocess = self._saved
        SV.WorkerClient.__init__, SV.subprocess = original_init, original_subprocess
        self._saved = None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--fd", type=int, required=True)
    p.add_argument("--parent", type=int, required=True)
    p.add_argument("--gc-policy", choices=C.ARMS, required=True)
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--max-frames", type=int, required=True)
    p.add_argument("--producer", choices=C.PRODUCERS, required=True)
    a = p.parse_args(argv)
    if not a.trace_dir.is_dir() or a.trace_dir.is_symlink():
        raise SystemExit("REFUSED: trace directory must exist (created by the UE seam)")
    from rl_agent.splitfusion_exact_gt_live_v2 import service as LS
    substitution = WorkerSubstitution(arm=a.gc_policy, trace_dir=a.trace_dir, max_frames=a.max_frames,
                                      producer=a.producer).install()
    from rl_agent.splitfusion_m3_route_b_v3 import capacity as CAP
    capacity = CAP.install(CAP.GT_SERVICE)     # M3 C3: archive sized at its unchanged 4,000-job cap
    rc, error = None, None
    try:
        rc = LS.main(a.fd, a.parent)
        return rc
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"[:500]
        raise
    finally:
        substitution.uninstall()
        capacity.uninstall()
        with open(a.trace_dir / "service_substitution.json", "x") as stream:
            json.dump(dict(schema="splitfusion.m3_route_b_v3.service_substitution.v1", arm=a.gc_policy, producer=a.producer,
                           substitutions=substitution.records, m3_capacity=capacity.document(), returncode=rc, error=error,
                           pid=os.getpid(), entry_module=C.SERVICE_ENTRY, contract_module=C.__name__), stream,
                      indent=1, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
