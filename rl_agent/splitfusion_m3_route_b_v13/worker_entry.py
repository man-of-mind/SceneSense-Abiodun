"""M3.3 v13 GT worker trampoline: corrected exact storage accounting V3.v2 (with bounded per-call timing) in the
prep role only, then the UNCHANGED v8 worker trampoline (tilt-aware AVO -> v7 FREEZE lifetime repair -> v4 -> v3).

    python -m rl_agent.splitfusion_m3_route_b_v13.worker_entry <the v3 worker_entry arguments>

prep:  ``splitfusion_gt_storage_accounting_v3.switch.install()`` returns ONE bounded ``AccountingSession`` and makes
       it ``ownership.payload_storage_bytes``; the owner global is then pointed at an ``AccountingTimer`` wrapping
       exactly that session (same integer, same exceptions; bounded buffers), and
       ``splitfusion_m3_route_b_v8.worker_entry.main`` runs.  On exit the timer is removed, the session's bounded
       cache statistics are recorded, the session is restored as the owner global so the switch's own uninstall sees
       exactly what it installed, ``switch.uninstall()`` closes and clears it, and
       ``storage_accounting_binding.json`` / ``storage_accounting_timing.json`` are written.
score: ``splitfusion_m3_route_b_v8.worker_entry.main`` unchanged (nothing installed here).

Uses the native V10-derived worker path with V3's reviewed production switch and the timer added. The V3 fast path
requires all advertised capabilities; this entry REFUSES to start the prep worker if any capability is unavailable.
Only ``--role`` and ``--trace-dir`` are read here; the retained parser validates every argument. A
record-write failure is reported on stderr and never replaces the worker's own exception.  As in v7/v8/v10/v11, the exit
records can be lost when the service SIGTERMs the worker after a fatal failure.
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
    from rl_agent.splitfusion_gt_storage_accounting_v3 import accounting as A, switch
    from .accounting_timing import AccountingTimer
    capabilities = A.capabilities()
    fast_path = all(reason is None for reason in capabilities.values())
    record = dict(schema="splitfusion.m3_route_b_v13.storage_accounting_binding.v1", role=role, pid=os.getpid(),
                  entry_module=C.WORKER_ENTRY, v8_worker_entry=V8W.__name__, version=A.VERSION,
                  substitution=dict(switch.SUBSTITUTION),
                  sources={name: C.sha(Path(module.__file__)) for name, module in (("accounting", A), ("switch", switch))},
                  timer_capacity=C.ACCOUNTING_TIMING_CAPACITY, max_objects=A.DEFAULT_MAX_OBJECTS,
                  capabilities=capabilities, fast_path_available=fast_path, interpreter=sys.executable,
                  python_version=sys.version.split()[0], installed=None, timer_installed=None,
                  timer_removed=None, restored=None, accounting_session=None, exit=None)
    if tuple(capabilities) != C.ACCOUNTING_REQUIRED_CAPABILITIES or not fast_path:
        V7W._write_record(trace_dir / C.ACCOUNTING_BINDING_RECORD, record)
        raise RuntimeError("REFUSED: registered V3 storage-accounting capabilities unavailable: "
                           + repr(capabilities))
    session = switch.install(max_objects=C.ACCOUNTING_MAX_OBJECTS)
    timer = AccountingTimer(session, C.ACCOUNTING_TIMING_CAPACITY)
    try:
        record["installed"] = switch.installed() and OWN.payload_storage_bytes is session
        if not record["installed"]:
            raise RuntimeError("REFUSED: storage-accounting switch is not the owner accounting function")
        OWN.payload_storage_bytes = timer
        record["timer_installed"] = OWN.payload_storage_bytes is timer and timer.__wrapped__ is session
        return V8W.main(args)
    except BaseException as error:
        record["exit"] = f"{type(error).__name__}: {error}"[:500]
        raise
    finally:
        if OWN.payload_storage_bytes is timer:
            OWN.payload_storage_bytes = session
        record["timer_removed"] = OWN.payload_storage_bytes is not timer
        record["accounting_session"] = session.snapshot()
        switch.uninstall()
        record["restored"] = (not switch.installed()
                              and OWN.payload_storage_bytes is switch.ORIGINAL_PAYLOAD_STORAGE_BYTES)
        record["accounting_session_closed"] = session.snapshot()["closed"]
        record["timing_summary"] = dict(calls=timer.calls, recorded=len(timer.cpu_ns), dropped=timer.dropped,
                                        errors=timer.errors, foreign_frames=timer.foreign_frames)
        V7W._write_record(trace_dir / C.ACCOUNTING_TIMING_RECORD, timer.document())
        V7W._write_record(trace_dir / C.ACCOUNTING_BINDING_RECORD, record)


if __name__ == "__main__":
    raise SystemExit(main())
