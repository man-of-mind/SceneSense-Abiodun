"""Apply an explicit whole-edge/whole-radio partition at the READY barrier."""
import json
import hashlib
import os
import sys
import time
from pathlib import Path

from cpu_partition import PartitionSession, ProcessPlan, ProcBackend, validate_disjoint_cpu_sets
from edge_gate import write, raw

PREVIOUS = Path(__file__).resolve().parents[2].parent / 'abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261002T065135Z_compute_affinity_ab_v2'
sys.path.insert(0, str(PREVIOUS))
from injection_guard_v2 import collect
from host_affinity import task, tasks, cgroups


def locate_edge(ready):
    matches = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if ' -m rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_cpu_partition_v1 ' not in command:
                continue
            inventory = tasks(int(entry.name))
            for thread in inventory.values():
                if (thread['ns_tid'][-1] == ready['tid']
                        and thread['starttime'] == ready['starttime']):
                    matches.append(dict(target=thread, threads=inventory, cmd=command,
                                        cgroups=cgroups(entry.name)))
        except (OSError, ProcessLookupError):
            continue
    if len(matches) != 1:
        raise RuntimeError(f'Expected one identity-matched edge worker, got {len(matches)}')
    return matches[0]


def owned_radio(authority):
    result = []
    for role, executable in [('gnb', 'nr-softmodem'), ('ue', 'nr-uesoftmodem')]:
        topology = authority['attached'][role + '_process_topology']
        assert topology['endpoint_roles_verified'] is True
        assert topology['same_executable_process_count'] == 3
        pids = [topology['main_pid'], *topology['worker_pids']]
        assert len(set(pids)) == 3
        for pid in pids:
            directory = Path('/proc') / str(pid)
            assert directory.joinpath('exe').resolve().name == executable
            assert os.getpgid(pid) == topology['process_group_id']
            fields = directory.joinpath('stat').read_text().rsplit(')', 1)[1].split()
            assert fields[19] == authority['pid_starttimes'][str(pid)], 'Radio PID reused after attachment'
            command = directory.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode().strip()
            assert hashlib.sha256(command.encode()).hexdigest() == authority['attached'][role + '_command_sha256'], 'Radio argv changed'
            if pid != topology['main_pid']:
                assert int(fields[1]) == topology['main_pid']
            result.append(dict(pid=pid, role=role, comm=executable,
                cmd=command,
                threads=tasks(pid), cgroups=cgroups(pid)))
    assert len({p['pid'] for p in result}) == 6
    return result


def apply(output):
    assert not (output / 'PARTITION_JOURNAL.json').exists()
    spec = json.loads((output / 'arm.json').read_text())
    authority = json.loads((output / 'RADIO_AUTHORITY.json').read_text())
    ready = json.loads((output / 'worker_ready.json').read_text())
    edge, radio = locate_edge(ready), owned_radio(authority)
    guard = collect(edge['target']['pid'])
    write(output / 'GUARD_EVIDENCE.json', guard)
    assert guard['verdict']['passed'], guard['verdict']
    all_cpus = tuple(spec['original_cpus'])
    validate_disjoint_cpu_sets({'radio': spec['radio_cpus'], 'edge': spec['edge_cpus']}, all_cpus)
    plans, roles = [], {}
    backend = ProcBackend()
    for process in [*radio, dict(pid=edge['target']['pid'], role='edge', threads=edge['threads'],
                                  cgroups=edge['cgroups'])]:
        assert process['threads']
        assert all(tuple(t['affinity']) == all_cpus for t in process['threads'].values()), 'Pre-existing nonbaseline mask'
        desired = all_cpus if spec['arm'] == 'control' else tuple(spec[process['role'] == 'edge' and 'edge_cpus' or 'radio_cpus'])
        plans.append(ProcessPlan(process['pid'], desired, all_cpus,
                                 expected_starttime=backend.process_starttime(process['pid'])))
        roles[str(process['pid'])] = process['role']
    write(output / 'AFFINITY_BEFORE.json', dict(edge=edge, radio=radio, roles=roles, spec=spec, raw_ns=raw()))
    session = PartitionSession(max_passes=8,
        persist=lambda value: write(output / 'PARTITION_JOURNAL.json', value))
    result = session.apply(plans, available_cpus=all_cpus)
    write(output / 'PARTITION_APPLY_RESULT.json', result)
    # Compatibility fields are explicit, allowing the established scheduler analyzer to be reused.
    write(output / 'AFFINITY_APPLIED.json', dict(arm=spec['arm'], raw_ns=raw(), before=edge,
        radio=radio, roles=roles, original_affinity=list(all_cpus),
        expected_affinity=list(plans[-1].cpus), passed=result['converged'],
        scope='ALL_THREADS_OF_VERIFIED_EDGE_AND_SIX_RADIO_PROCESSES_INCLUDING_FUTURE_THREADS'))


def restore(output):
    journal = output / 'PARTITION_JOURNAL.json'
    if not journal.exists():
        return
    session = PartitionSession.from_dict(json.loads(journal.read_text()),
        persist=lambda value: write(journal, value))
    was_restored = session.status == 'restored'
    result = session.restore()
    target = output / ('PARTITION_RESTORED_' + str(raw()) + '.json')
    write(target, result)
    assert result['converged'], result['errors']
    assert was_restored or not result['skipped'], ('First restoration must verify every process', result['skipped'])
    write(output / 'RESTORATION_LATEST.json', dict(passed=True, evidence=target.name, raw_ns=raw()))


def monitor(output):
    info = json.loads((output / 'AFFINITY_APPLIED.json').read_text())
    journal = json.loads((output / 'PARTITION_JOURNAL.json').read_text())
    plans = {p['pid']: p for p in journal['processes']}
    baseline = {int(tid): row for p in info['radio'] for tid, row in p['threads'].items()}
    baseline.update({int(k): v for k, v in info['before']['threads'].items()})
    backend = ProcBackend()
    errors, n, start = [], 0, time.process_time()
    deadline = time.monotonic() + 600
    with (output / 'target_samples.jsonl').open('x') as stream:
        while not (output / 'collection_stop').exists() and time.monotonic() < deadline:
            record = dict(raw_ns=raw(), processes={}, errors=[])
            for pid, plan in plans.items():
                try:
                    assert backend.process_starttime(pid) == plan['pid_starttime'], 'PID identity changed'
                    inventory = tasks(pid)
                    record['processes'][str(pid)] = inventory
                    for tid, row in inventory.items():
                        if row['affinity'] != plan['target_cpus']:
                            record['errors'].append(f'affinity drift {pid}/{tid}')
                        previous = baseline.get(int(tid))
                        if previous and previous['starttime'] == row['starttime']:
                            if (row['policy'], row['priority']) != (previous['policy'], previous['priority']):
                                record['errors'].append(f'priority drift {pid}/{tid}')
                except (OSError, AssertionError) as exc:
                    if not (output / 'collection_stop').exists():
                        record['errors'].append(f'{pid}: {exc}')
            # A sample crossing the stop/restoration boundary is not live scope evidence.
            if (output / 'collection_stop').exists():
                break
            # Keep prior analyzer's edge/radio inventory views.
            record['edge'] = record['processes'].get(str(info['before']['target']['pid']), {})
            record['radio'] = [r for pid, ts in record['processes'].items()
                               if info['roles'][pid] != 'edge' for r in ts.values() if r['comm'] == 'ru_thread']
            stream.write(json.dumps(record) + '\n'); stream.flush(); n += 1
            if record['errors']:
                errors.extend(record['errors'])
                write(output / 'SCOPE_VIOLATION.json', record)
                break
            time.sleep(1)
    write(output / 'monitor_footer.json', dict(samples=n, cpu_s=time.process_time()-start,
        errors=errors, passed=not errors, raw_ns=raw()))


if __name__ == '__main__':
    action, location = sys.argv[1:]
    {'apply': apply, 'restore': restore, 'monitor': monitor}[action](Path(location))
