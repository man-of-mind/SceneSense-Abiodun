"""Candidate worker wrapper; a future launcher must explicitly register it.

No existing launcher or source manifest is modified by this package. Production
use is blocked until this wrapper and the repair are included in a new identity
and pass the actual process-role startup checks.
"""
import os

if __name__ == "__main__":
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")


def main(argv=None):
    import sys
    from rl_agent.splitfusion_m3_route_b_v4 import worker_entry as retained
    from rl_agent.splitfusion_m3_route_b_v3.worker_entry import parse
    from .repair import install

    args = list(sys.argv[1:] if argv is None else argv)
    parsed = parse(args)
    if parsed.role == "prep":
        with install(parsed.trace_dir):
            return retained.main(args)
    return retained.main(args)


if __name__ == "__main__":
    main()
