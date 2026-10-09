"""V5 launcher identity over the unchanged sealed v4 partition mechanics."""
from pathlib import Path
import sys


def apply(output, retained_code):
    from rl_agent.splitfusion_m3_route_b_v4.partition import apply as retained_apply
    return retained_apply(output, retained_code)


if __name__ == "__main__":
    apply(Path(sys.argv[1]), Path(sys.argv[2]))
