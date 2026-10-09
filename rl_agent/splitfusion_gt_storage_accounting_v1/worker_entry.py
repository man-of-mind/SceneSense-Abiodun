"""Candidate GT worker trampoline: exact storage accounting in the prep role, then the UNCHANGED V8 worker trampoline.

    python -m rl_agent.splitfusion_gt_storage_accounting_v1.worker_entry <the v3 worker_entry arguments>

prep:  ``switch.install()`` (``ownership.payload_storage_bytes`` -> ``accounting.payload_storage_bytes``), then
       ``splitfusion_m3_route_b_v8.worker_entry.main`` (tilt-aware AVO switch -> V7 FREEZE lifetime repair -> V4/V3
       worker); the switch is restored on exit and ``storage_accounting_binding.json`` records both facts.
score: ``splitfusion_m3_route_b_v8.worker_entry.main`` unchanged (nothing installed here).
Only ``--role`` and ``--trace-dir`` are read here; the retained parser validates every argument.  A record-write
failure is reported on stderr and never replaces the worker's own exception.
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
    from . import accounting as A, switch
    record = dict(schema="splitfusion.gt_storage_accounting_v1.binding.v1", role=role, pid=os.getpid(),
                  entry_module=C.WORKER_ENTRY, v8_worker_entry=V8W.__name__, version=A.VERSION,
                  substitution=dict(switch.SUBSTITUTION),
                  sources={name: C.sha(Path(module.__file__)) for name, module in (("accounting", A), ("switch", switch))},
                  installed=None, restored=None, exit=None)
    switch.install()
    try:
        record["installed"] = switch.installed() and OWN.payload_storage_bytes is A.payload_storage_bytes
        if not record["installed"]:
            raise RuntimeError("REFUSED: storage-accounting switch is not the owner accounting function")
        return V8W.main(args)
    except BaseException as error:
        record["exit"] = f"{type(error).__name__}: {error}"[:500]
        raise
    finally:
        switch.uninstall()
        record["restored"] = (not switch.installed()
                              and OWN.payload_storage_bytes is switch.ORIGINAL_PAYLOAD_STORAGE_BYTES)
        V7W._write_record(trace_dir / C.ACCOUNTING_BINDING_RECORD, record)


if __name__ == "__main__":
    raise SystemExit(main())
