"""Retained serial edge, with setup-only partition verification barrier."""
import sys


def main():
    sys.path.insert(0, '/cpu_partition_code')
    import edge_gate
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
    edge_gate.install('/partition_output', PW)
    from . import edge_profiled_v1_ordering as E
    return E.main()


if __name__ == '__main__':
    raise SystemExit(main())
