"""Original profiler runner with durable radio ownership and restoration hooks."""
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    from rl_agent import ue_288_campaign_supervisor as S
    from . import runner_v1 as R
    output = Path(os.environ['SCENESENSE_PARTITION_OUTPUT'])
    code = Path(os.environ['SCENESENSE_PARTITION_CODE'])
    original_start, original_stop = S._start_live_radio, S._stop_live_radio

    def start(*args, **kwargs):
        namespace, state, attached = original_start(*args, **kwargs)
        identities = {}
        for role in ('gnb', 'ue'):
            topology = attached[role + '_process_topology']
            for pid in (topology['main_pid'], *topology['worker_pids']):
                identities[str(pid)] = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        with (output / 'RADIO_AUTHORITY.json').open('x') as stream:
            json.dump({'namespace': str(namespace), 'radio_state': str(state),
                       'attached': attached, 'pid_starttimes': identities}, stream, indent=2)
        return namespace, state, attached

    def stop(*args, **kwargs):
        try:
            if (output / 'PARTITION_JOURNAL.json').exists():
                result = subprocess.run(['sudo', '-n', sys.executable,
                    str(code / 'host_partition.py'), 'restore', str(output)],
                    capture_output=True, text=True, timeout=20)
                (output / 'radio_prestop_restore_command.json').write_text(json.dumps({
                    'returncode': result.returncode, 'stdout': result.stdout,
                    'stderr': result.stderr}, indent=2) + '\n')
        finally:
            stopped = original_stop(*args, **kwargs)
        return stopped

    S._start_live_radio, S._stop_live_radio = start, stop
    return R.main()


if __name__ == '__main__':
    raise SystemExit(main())
