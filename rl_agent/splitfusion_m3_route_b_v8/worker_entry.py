"""M3.1 v8 GT worker trampoline: the tilt-aware pedestrian AVO in the prep role only, then the unchanged v7 worker
trampoline (prep FREEZE lifetime repair -> v4 trampoline -> v3 worker implementation).

    python -m rl_agent.splitfusion_m3_route_b_v8.worker_entry <the v3 worker_entry arguments>

prep:  ``splitfusion_avo_tilt_v1.switch.install()`` makes ``splitfusion_numeric_gt_producer_v1.avo.numeric_avo``
       resolve to the qualified tilt-aware ``quality.numeric_avo`` in this process (its only production caller is
       ``numeric_gt_integration_v2.composition.assemble`` in ``prepare_unbound``), then ``v7.worker_entry.main``;
       the switch is restored on exit and ``avo_tilt_binding.json`` records both facts.
score: ``v7.worker_entry.main`` unchanged (nothing installed).

``main`` is the qualified candidate trampoline (``splitfusion_avo_tilt_v1.worker_entry.main``, service_proof_02)
reading the v8 contract instead of the candidate one (checked by the binding regression).  Only ``--role`` and
``--trace-dir`` are read here; the retained parser validates every argument.  A record-write failure is reported on
stderr and never replaces the worker's own exception.
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
    from rl_agent.splitfusion_avo_tilt_v1 import geometry as G, quality as Q, switch
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
