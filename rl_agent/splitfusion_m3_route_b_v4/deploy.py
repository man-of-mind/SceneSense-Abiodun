"""Deploy only the prospective v4 package/shims; verify the sealed v3 base in place."""
import argparse
from pathlib import Path

from rl_agent.splitfusion_m3_route_b_v3 import deploy as D

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "rl_agent/splitfusion_m3_route_b_v4"
EVIDENCE = ROOT / "experiments/splitfusion_m3_route_b_v4"
SHIMS = {
    f"{PACKAGE}/runtime_shims/child_v1_m3_route_b_v4.py":
        "rl_agent/splitfusion_single_gpu_profiler_v1/child_v1_m3_route_b_v4.py",
    f"{PACKAGE}/runtime_shims/edge_profiled_v1_m3_route_b_v4.py":
        "rl_agent/splitfusion_single_gpu_profiler_v1/edge_profiled_v1_m3_route_b_v4.py",
}


def run(name, target):
    saved = D.PACKAGE, D.PACKAGES, D.EVIDENCE, D.SHIMS, D.PIPELINE
    D.PACKAGE = PACKAGE
    D.PACKAGES = (PACKAGE, "rl_agent/splitfusion_paired_retention_v2")
    D.EVIDENCE = EVIDENCE
    D.SHIMS = dict(SHIMS)
    D.PIPELINE = (*saved[4], "rl_agent/splitfusion_m3_route_b_v3")
    try:
        return D.run(name, target)
    finally:
        D.PACKAGE, D.PACKAGES, D.EVIDENCE, D.SHIMS, D.PIPELINE = saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    run(args.name, args.target)
