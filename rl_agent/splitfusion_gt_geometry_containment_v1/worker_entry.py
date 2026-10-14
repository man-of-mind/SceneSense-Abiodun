"""Containment-candidate GT worker trampoline (OFFLINE ONLY).

    python -m rl_agent.splitfusion_gt_geometry_containment_v1.worker_entry <the v3 worker_entry arguments>

prep (correction 01: the CURRENT V16 prep stack, composed explicitly so containment sits INSIDE the lifetime repair):
  1. V16 accounting layer, copied from ``splitfusion_m3_route_b_v16.worker_entry.main``: capability check (REFUSES to
     start without every registered V3 capability), ``splitfusion_gt_storage_accounting_v3.switch.install`` -> ONE
     bounded ``AccountingSession``, wrapped by the V16 ``AccountingTimer`` as ``ownership.payload_storage_bytes``;
     on exit timer removed, session restored, switch uninstalled, binding + timing records written (V16 schemas)
  2. V8 layer (``splitfusion_m3_route_b_v8.worker_entry.main`` body): tilt-aware AVO switch + ``avo_tilt_binding.json``
  3. V7 layer (``splitfusion_m3_route_b_v7.worker_entry.main`` body): FREEZE lifetime repair + ``prep_lifetime_binding.json``
  4. ``core.install``: PrepCore -> PrepCoreContained + ``geometry_containment_binding.json``
  5. ``splitfusion_m3_route_b_v4.worker_entry.main`` (V3 worker; unchanged)
score: ``splitfusion_m3_route_b_v8.worker_entry.main`` unchanged (= V16's score role).
A record-write failure is reported on stderr and never replaces the worker's own exception.
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
    from rl_agent.splitfusion_gt_storage_accounting_v3 import accounting as A, switch as ACCT
    from rl_agent.splitfusion_m3_route_b_v16.accounting_timing import AccountingTimer
    capabilities = A.capabilities()
    fast_path = all(reason is None for reason in capabilities.values())
    accounting = dict(schema="splitfusion.m3_route_b_v16.storage_accounting_binding.v1", role=role, pid=os.getpid(),
                      entry_module=C.WORKER_ENTRY, v8_worker_entry=V8W.__name__, version=A.VERSION,
                      substitution=dict(ACCT.SUBSTITUTION),
                      sources={name: C.sha(Path(module.__file__)) for name, module in (("accounting", A), ("switch", ACCT))},
                      timer_capacity=C.ACCOUNTING_TIMING_CAPACITY, max_objects=A.DEFAULT_MAX_OBJECTS,
                      capabilities=capabilities, fast_path_available=fast_path, interpreter=sys.executable,
                      python_version=sys.version.split()[0], installed=None, timer_installed=None,
                      timer_removed=None, restored=None, accounting_session=None, exit=None)
    if tuple(capabilities) != C.ACCOUNTING_REQUIRED_CAPABILITIES or not fast_path:
        V7W._write_record(trace_dir / C.ACCOUNTING_BINDING_RECORD, accounting)
        raise RuntimeError("REFUSED: registered V3 storage-accounting capabilities unavailable: " + repr(capabilities))
    session = ACCT.install(max_objects=C.ACCOUNTING_MAX_OBJECTS)
    timer = AccountingTimer(session, C.ACCOUNTING_TIMING_CAPACITY)
    try:
        accounting["installed"] = ACCT.installed() and OWN.payload_storage_bytes is session
        if not accounting["installed"]:
            raise RuntimeError("REFUSED: storage-accounting switch is not the owner accounting function")
        OWN.payload_storage_bytes = timer
        accounting["timer_installed"] = OWN.payload_storage_bytes is timer and timer.__wrapped__ is session
        return _avo_lifetime_containment(args, role, trace_dir, V7W)
    except BaseException as error:
        accounting["exit"] = f"{type(error).__name__}: {error}"[:500]
        raise
    finally:
        if OWN.payload_storage_bytes is timer:
            OWN.payload_storage_bytes = session
        accounting["timer_removed"] = OWN.payload_storage_bytes is not timer
        accounting["accounting_session"] = session.snapshot()
        ACCT.uninstall()
        accounting["restored"] = (not ACCT.installed()
                                  and OWN.payload_storage_bytes is ACCT.ORIGINAL_PAYLOAD_STORAGE_BYTES)
        accounting["accounting_session_closed"] = session.snapshot()["closed"]
        accounting["timing_summary"] = dict(calls=timer.calls, recorded=len(timer.cpu_ns), dropped=timer.dropped,
                                            errors=timer.errors, foreign_frames=timer.foreign_frames)
        V7W._write_record(trace_dir / C.ACCOUNTING_TIMING_RECORD, timer.document())
        V7W._write_record(trace_dir / C.ACCOUNTING_BINDING_RECORD, accounting)


def _avo_lifetime_containment(args, role, trace_dir, V7W):
    from rl_agent.splitfusion_avo_tilt_v1 import geometry as G, quality as Q, switch
    from rl_agent.splitfusion_exact_gt_service_proto_v2 import prep_core as PC
    from rl_agent.splitfusion_gt_freeze_lifetime_v1 import repair as R
    from rl_agent.splitfusion_m3_route_b_v4 import worker_entry as retained
    from rl_agent.splitfusion_numeric_gt_producer_v1 import avo as NA
    from . import classify as CL, core as K, engine as E
    pid = os.getpid()
    avo = dict(schema="splitfusion.avo_tilt_v1.binding.v1", role=role, pid=pid, entry_module=C.WORKER_ENTRY,
               v7_worker_entry=V7W.__name__, geometry_version=G.GEOMETRY_VERSION,
               sources={name: C.sha(Path(module.__file__)) for name, module in
                        (("geometry", G), ("quality", Q), ("switch", switch))},
               installed=None, restored=None, exit=None)
    lifetime = dict(schema="splitfusion.m3_route_b_v7.prep_lifetime_binding.v1", role=role, pid=pid,
                    entry_module=C.WORKER_ENTRY, contract_module=C.__name__, retained_trampoline=retained.__name__,
                    repair_module=R.__name__, repair_file=str(Path(R.__file__).resolve()),
                    repair_sha256=C.sha(Path(R.__file__)), failure_directory=str(trace_dir),
                    installed=None, restored=None, exit=None)
    containment = dict(schema=CL.VERSION + ".binding", role=role, pid=pid, entry_module=C.WORKER_ENTRY,
                       base_stack=C.BASE_WORKER_ENTRY,
                       sources={name: C.sha(Path(module.__file__)) for name, module in
                                (("classify", CL), ("core", K), ("engine", E))},
                       diagnostics_directory=str(trace_dir), installed=None, restored=None, exit=None)
    switch.install()
    try:
        avo["installed"] = switch.installed() and NA.numeric_avo is Q.numeric_avo
        if not avo["installed"]:
            raise RuntimeError("REFUSED: tilt-aware AVO switch is not the producer AVO step")
        with R.install(str(trace_dir)):
            lifetime["installed"] = (PC.PrepCore is R.PrepCoreLifetime
                                     and R.PrepCoreLifetime.failure_directory == str(trace_dir))
            if not lifetime["installed"]:
                raise RuntimeError("REFUSED: prep FREEZE lifetime repair is not the constructed core class")
            with K.install(trace_dir):
                containment["installed"] = (PC.PrepCore is K.PrepCoreContained
                                            and issubclass(K.PrepCoreContained, R.PrepCoreLifetime))
                if not containment["installed"]:
                    raise RuntimeError("REFUSED: containment PrepCore is not the constructed core class")
                return retained.main(args)
    except BaseException as error:
        text = f"{type(error).__name__}: {error}"[:500]
        avo["exit"] = lifetime["exit"] = containment["exit"] = text
        raise
    finally:
        containment["restored"] = PC.PrepCore is not K.PrepCoreContained and K.PrepCoreContained.diagnostics_directory is None
        lifetime["restored"] = PC.PrepCore is R.ORIGINAL_PREP_CORE and R.PrepCoreLifetime.failure_directory is None
        switch.uninstall()
        avo["restored"] = not switch.installed() and NA.numeric_avo is switch.ORIGINAL_NUMERIC_AVO
        V7W._write_record(trace_dir / C.AVO_BINDING_RECORD, avo)
        V7W._write_record(trace_dir / C.PREP_BINDING_RECORD, lifetime)
        V7W._write_record(trace_dir / C.CONTAINMENT_BINDING_RECORD, containment)


if __name__ == "__main__":
    raise SystemExit(main())
