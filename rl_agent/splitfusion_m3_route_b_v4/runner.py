"""V4-bound facade over the unchanged v3 live runner."""
from . import contract as C
from rl_agent.splitfusion_m3_route_b_v3 import runner as _R


def main(argv=None):
    _R.C = C
    return _R.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
