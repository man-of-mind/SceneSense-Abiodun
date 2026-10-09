"""Bind the retained CPU-partition runner to one concrete fixed contract."""
import sys


def main(C, argv=None):
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

