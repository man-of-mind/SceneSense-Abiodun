"""Exact retained affinity application, with the new edge entry-point identity.

Restoration and monitoring use the retained helper unchanged. This wrapper
only substitutes its exact command-line matcher; PID/TID/starttime, namespace,
cgroup, ownership, policy and journal checks are unchanged.
"""
import importlib.util
import sys
from pathlib import Path

EDGE_MODULE = 'rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_cpu_partition_v1_run5_live_avo'


def apply(output, retained_code):
    code = Path(retained_code).resolve(strict=True)
    sys.path.insert(0, str(code))
    import host_partition as H

    def locate(ready):
        matches = []
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
                if f' -m {EDGE_MODULE} ' not in command:
                    continue
                inventory = H.tasks(int(entry.name))
                for thread in inventory.values():
                    if thread['ns_tid'][-1] == ready['tid'] and thread['starttime'] == ready['starttime']:
                        matches.append(dict(target=thread, threads=inventory, cmd=command,
                                            cgroups=H.cgroups(entry.name)))
            except (OSError, ProcessLookupError):
                continue
        if len(matches) != 1:
            raise RuntimeError(f'Expected one identity-matched Run-5 edge worker; got {len(matches)}')
        return matches[0]

    H.locate_edge = locate
    H.apply(Path(output))


if __name__ == '__main__':
    apply(Path(sys.argv[1]), Path(sys.argv[2]))
