"""Retained CPU partition plus setup-only passive diagnostic barriers."""
import os
from pathlib import Path
import sys


def main():
    code = Path(os.environ['SCENESENSE_TRANSPORT_CODE'])
    sys.path.insert(0, str(code))
    import startup_gate as G
    from . import child_v1_cpu_partition_v1 as C
    from . import child_v1_threadcap as TC
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
    from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
    output = Path(os.environ['SCENESENSE_PARTITION_OUTPUT'])
    build = U.build_run4_runtime_class

    def factory(*args, **kwargs):
        return G.wrap_runtime_class(build(*args, **kwargs), output)

    U.build_run4_runtime_class = factory
    original = TC.install_ordering_route

    def ordering(*args, **kwargs):
        result = original(*args, **kwargs)
        pinned.stop_tail = G.teardown_gate(pinned.stop_tail, output)
        return result

    TC.install_ordering_route = ordering
    return C.main()


if __name__ == '__main__':
    raise SystemExit(main())
