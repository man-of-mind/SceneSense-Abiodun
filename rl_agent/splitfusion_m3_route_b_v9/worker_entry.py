"""M3.1 v9 GT worker trampoline: the qualified exact storage accounting (with bounded per-call timing) in the prep role
only, then the UNCHANGED v8 worker trampoline (tilt-aware AVO -> v7 FREEZE lifetime repair -> v4 trampoline -> v3).

    python -m rl_agent.splitfusion_m3_route_b_v9.worker_entry <the v3 worker_entry arguments>

prep:  ``splitfusion_gt_storage_accounting_v1.switch.install()`` (``ownership.payload_storage_bytes`` ->
       ``accounting.payload_storage_bytes``, the qualified candidate), then the owner global is pointed at an
       ``AccountingTimer`` wrapping exactly that function (same integer, same exceptions; bounded buffers), then
       ``splitfusion_m3_route_b_v8.worker_entry.main``.  On exit the timer is removed, the switch restored, and
       ``storage_accounting_binding.json`` / ``storage_accounting_timing.json`` are written.
score: ``splitfusion_m3_route_b_v8.worker_entry.main`` unchanged (nothing installed here).

Mirrors the qualified candidate trampoline (``splitfusion_gt_storage_accounting_v1.worker_entry``) with the v9
contract and the timer added.  Only ``--role`` and ``--trace-dir`` are read here; the retained parser validates every
argument.  A record-write failure is reported on stderr and never replaces the worker's own exception.  As in v7/v8,
the exit records can be lost when the service SIGTERMs the worker after a fatal failure.
"""
import os
if __name__ == "__main__":
    os.environ.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})

from pathlib import Path
import sys

from . import contract as C


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    from rl_agent.splitfusion_m3_route_b_v7 import worker_entry as V7W
    from rl_agent.splitfusion_m3_route_b_v8 import worker_entry as V8W
    role, trace_dir = V7W.role_and_trace(args)
    if role != "prep":
        return V8W.main(args)
    from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN
    from rl_agent.splitfusion_gt_storage_accounting_v1 import accounting as A, switch
    from .accounting_timing import AccountingTimer
    record = dict(schema="splitfusion.m3_route_b_v9.storage_accounting_binding.v1", role=role, pid=os.getpid(),
                  entry_module=C.WORKER_ENTRY, v8_worker_entry=V8W.__name__, version=A.VERSION,
                  substitution=dict(switch.SUBSTITUTION),
                  sources={name: C.sha(Path(module.__file__)) for name, module in (("accounting", A), ("switch", switch))},
                  timer_capacity=C.ACCOUNTING_TIMING_CAPACITY, installed=None, timer_installed=None,
                  timer_removed=None, restored=None, exit=None)
    timer = AccountingTimer(A.payload_storage_bytes, C.ACCOUNTING_TIMING_CAPACITY)
    switch.install()
    try:
        record["installed"] = switch.installed() and OWN.payload_storage_bytes is A.payload_storage_bytes
        if not record["installed"]:
            raise RuntimeError("REFUSED: storage-accounting switch is not the owner accounting function")
        OWN.payload_storage_bytes = timer
        record["timer_installed"] = OWN.payload_storage_bytes is timer and timer.__wrapped__ is A.payload_storage_bytes
        return V8W.main(args)
    except BaseException as error:
        record["exit"] = f"{type(error).__name__}: {error}"[:500]
        raise
    finally:
        if OWN.payload_storage_bytes is timer:
            OWN.payload_storage_bytes = A.payload_storage_bytes
        record["timer_removed"] = OWN.payload_storage_bytes is not timer
        switch.uninstall()
        record["restored"] = (not switch.installed()
                              and OWN.payload_storage_bytes is switch.ORIGINAL_PAYLOAD_STORAGE_BYTES)
        record["timing_summary"] = dict(calls=timer.calls, recorded=len(timer.cpu_ns), dropped=timer.dropped,
                                        errors=timer.errors, foreign_frames=timer.foreign_frames)
        V7W._write_record(trace_dir / C.ACCOUNTING_TIMING_RECORD, timer.document())
        V7W._write_record(trace_dir / C.ACCOUNTING_BINDING_RECORD, record)


if __name__ == "__main__":
    raise SystemExit(main())
