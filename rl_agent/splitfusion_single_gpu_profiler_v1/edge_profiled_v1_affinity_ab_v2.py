"""Retained serial edge with a thread-only, restored affinity experiment."""
import sys

def main():
    sys.path.insert(0,'/affinity_output')
    import affinity_lib
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
    affinity_lib.install('/affinity_output',PW)
    from . import edge_profiled_v1_ordering as E
    return E.main()
if __name__=='__main__':raise SystemExit(main())
