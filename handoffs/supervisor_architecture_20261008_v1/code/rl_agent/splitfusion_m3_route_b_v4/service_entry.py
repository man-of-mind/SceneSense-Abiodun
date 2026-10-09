"""V4 identity wrapper for the unchanged v3 service trampoline."""
import os
if __name__ == "__main__":
    os.environ.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
from . import contract as C


def main(argv=None):
    from rl_agent.splitfusion_m3_route_b_v3 import service_entry as S
    S.C = C
    return S.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
