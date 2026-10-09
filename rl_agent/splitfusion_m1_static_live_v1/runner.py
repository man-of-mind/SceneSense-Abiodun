"""M1 runner: retained CPU-partition lifecycle with the M1 contract bound."""
from pathlib import Path
import sys


def main(argv=None):
    from . import contract as C
    from rl_agent.splitfusion_run5_full_live_retention_v1 import runner as retained_runner
    from rl_agent.splitfusion_run5_full_live_retention_v1 import host as retained_host
    old_runner, old_host = retained_runner.C, retained_host.C
    retained_runner.C = retained_host.C = C
    args = list(sys.argv[1:] if argv is None else argv)
    before = sys.argv
    sys.argv = [before[0], *args]
    try:
        return retained_runner.main(args)
    finally:
        sys.argv = before
        retained_runner.C, retained_host.C = old_runner, old_host


if __name__ == "__main__":
    raise SystemExit(main())
