"""Candidate GT worker trampoline: tilt-aware AVO in the prep role, then the unchanged V7 worker trampoline.

    python -m rl_agent.splitfusion_avo_tilt_v1.worker_entry <the v3 worker_entry arguments>

prep:  ``switch.install()`` (producer ``numeric_avo`` -> tilt-aware), then ``v7.worker_entry.main`` (FREEZE lifetime
       repair + V4/V3 worker); the switch is restored on exit and ``avo_tilt_binding.json`` records both facts.
score: ``v7.worker_entry.main`` unchanged (nothing installed).
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
    role, trace_dir = V7W.role_and_trace(args)
    if role != "prep":
        return V7W.main(args)
    from rl_agent.splitfusion_numeric_gt_producer_v1 import avo as NA
    from . import geometry as G, quality as Q, switch
    record = dict(schema="splitfusion.avo_tilt_v1.binding.v1", role=role, pid=os.getpid(), entry_module=C.WORKER_ENTRY,
                  v7_worker_entry=V7W.__name__, geometry_version=G.GEOMETRY_VERSION,
                  sources={name: C.sha(Path(module.__file__)) for name, module in
                           (("geometry", G), ("quality", Q), ("switch", switch))},
                  installed=None, restored=None, exit=None)
    switch.install()
    try:
        record["installed"] = switch.installed() and NA.numeric_avo is Q.numeric_avo
        if not record["installed"]:
            raise RuntimeError("REFUSED: tilt-aware AVO switch is not the producer AVO step")
        return V7W.main(args)
    except BaseException as error:
        record["exit"] = f"{type(error).__name__}: {error}"[:500]
        raise
    finally:
        switch.uninstall()
        record["restored"] = not switch.installed() and NA.numeric_avo is switch.ORIGINAL_NUMERIC_AVO
        V7W._write_record(trace_dir / C.AVO_BINDING_RECORD, record)


if __name__ == "__main__":
    raise SystemExit(main())
