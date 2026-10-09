"""Cap-only UE with diagnostic NVTX and container profiler launch mounts."""
import os
from pathlib import Path

def main(argv=None):
    from . import child_v1_threadcap as C
    from . import nsys_markers_diagnostic as N
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_launch_v2 as EL
    out = Path(os.environ['SCENESENSE_NSYS_OUTPUT'])
    N.install(out / 'ue_nvtx_cost.json')
    old_command = EL.up_command
    def command(variables):
        args = old_command(variables)
        i = args.index('up')
        return args[:i] + ['-f', str(out / 'compose_nsys.yaml')] + args[i:]
    EL.up_command = command
    old_ordering = C.install_ordering_route
    def ordering(*args, **kwargs):
        old_ordering(*args, **kwargs)
        from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as D
        D.DIRECT_EDGE_MODULE = '-m rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_nsys_diagnostic'
    C.install_ordering_route = ordering
    return C.main(argv)

if __name__ == '__main__':
    raise SystemExit(main())

