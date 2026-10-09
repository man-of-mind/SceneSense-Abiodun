"""Prep-worker entry for the whole-worker benchmark: the adopted GC-policy trampoline
(``splitfusion_gt_worker_spike_fix_v1.worker_entry``, identical in both arms) plus the producer switch.

    env NUMERIC_PRODUCER=v1|v2 SPIKE_FIX_MODE=candidate SPIKE_FIX_TRACE=<path> \
        python -m rl_agent.splitfusion_numeric_gt_producer_v2.worker_entry --fd N --parent P [--cpu C]
"""
import os
if __name__ == "__main__":
    os.environ.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})


def main():
    variant = os.environ.get("NUMERIC_PRODUCER", "v1")
    if variant not in ("v1", "v2"):
        raise SystemExit("unknown NUMERIC_PRODUCER")
    from . import worker_switch as W
    from rl_agent.splitfusion_gt_worker_spike_fix_v1 import worker_entry as GC
    if variant == "v2":
        W.install()
    try:
        GC.main()
    finally:
        W.uninstall()


if __name__ == "__main__":
    main()
