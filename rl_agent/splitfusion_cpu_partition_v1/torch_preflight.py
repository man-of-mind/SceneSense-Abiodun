"""Disposable-container test including Torch/CUDA and late native workers."""
import json
import os
import sys
import threading
from pathlib import Path
import torch
from cpu_partition import PartitionSession, ProcessPlan, ProcBackend


def main():
    output, arm = Path(sys.argv[1]), sys.argv[2]
    original = tuple(sorted(os.sched_getaffinity(0)))
    assert original == tuple(range(24)), original
    expected = original if arm == 'control' else (4, 5, 6, 7, 20, 21, 22, 23)
    torch.ones(8, device='cuda').sum().item()
    backend = ProcBackend()
    before = {t: backend.snapshot(os.getpid(), t) for t in backend.list_tids(os.getpid())}
    session = PartitionSession(max_passes=8)
    error = None
    ready, release = threading.Event(), threading.Event()
    worker = None
    try:
        applied = session.apply([ProcessPlan(os.getpid(), expected, original)])
        failures = []
        def work():
            try:
                value = torch.randn(256, 256)
                assert torch.isfinite(value @ value).all().item()
                assert torch.ones(16, device='cuda').sum().item() == 16
            except BaseException as exc:
                failures.append(repr(exc))
            finally:
                ready.set()
                release.wait(20)
        worker = threading.Thread(target=work)
        worker.start()
        assert ready.wait(15) and not failures, failures
        after = {t: backend.snapshot(os.getpid(), t) for t in backend.list_tids(os.getpid())}
        assert all(t.affinity == expected for t in after.values())
        assert all((t.policy, t.priority) == (before[k].policy, before[k].priority)
                   for k, t in after.items() if k in before)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        restored = session.restore()
        final = [backend.snapshot(os.getpid(), t) for t in backend.list_tids(os.getpid())]
        release.set()
        if worker is not None:
            worker.join(10)
        passed = restored['converged'] and all(t.affinity == original for t in final) and error is None
        record = dict(passed=passed, error=error, arm=arm, original=original,
            expected=expected, before_threads=len(before), after_threads=len(after) if 'after' in locals() else None,
            new_threads=sorted(set(after)-set(before)) if 'after' in locals() else [],
            torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
            ordinary_cupti_loaded='libcupti' in Path('/proc/self/maps').read_text(),
            restoration=restored)
        with output.open('x') as stream:
            json.dump(record, stream, indent=2)
        assert passed, ('Torch preflight failed; preserved detailed record', str(output), error)


if __name__ == '__main__':
    main()
