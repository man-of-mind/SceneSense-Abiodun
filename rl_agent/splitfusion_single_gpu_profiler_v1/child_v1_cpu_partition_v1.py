"""Cap-only baseline with an explicitly scoped radio/edge CPU partition."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    from . import child_v1_threadcap as C
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_launch_v2 as EL
    from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned

    output = Path(os.environ['SCENESENSE_PARTITION_OUTPUT'])
    code = Path(os.environ['SCENESENSE_PARTITION_CODE'])
    original_up = EL.up_command

    def command(values):
        argv = original_up(values)
        i = argv.index('up')
        return argv[:i] + ['-f', str(output / 'compose_partition.yaml')] + argv[i:]

    EL.up_command = command
    C.ORDERING_EDGE_MODULE = '-m rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_cpu_partition_v1'
    original_ordering = C.install_ordering_route

    def ordering(*args, **kwargs):
        original_ordering(*args, **kwargs)
        original_stop = pinned.stop_tail

        def stop_tail():
            (output / 'collection_stop').touch()
            try:
                if (output / 'PARTITION_JOURNAL.json').exists():
                    result = subprocess.run(
                        ['sudo', '-n', sys.executable, str(code / 'host_partition.py'),
                         'restore', str(output)], capture_output=True, text=True, timeout=20)
                    (output / 'prestop_restore_command.json').write_text(json.dumps({
                        'returncode': result.returncode, 'stdout': result.stdout,
                        'stderr': result.stderr,
                        'raw_ns': time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)}, indent=2) + '\n')
            finally:
                stopped = original_stop()
            return stopped

        pinned.stop_tail = stop_tail

    C.install_ordering_route = ordering
    return C.main()


if __name__ == '__main__':
    raise SystemExit(main())
