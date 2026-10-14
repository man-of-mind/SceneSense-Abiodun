"""User-approved, exact-instance host baseline for the first M4 smoke.

Preserve idle pre-existing MPS and remote desktop, not arbitrary GPU clients.
No process is stopped and no GPU exclusivity/performance claim is made.
Approval: 2026-10-13, explicit baseline exception in the startup/smoke turn.
After a reboot or service restart, another ownership review is required.
"""
import os
from pathlib import Path
import subprocess

from .identity import require

BOOT_ID = '0b44342f-aec1-429a-bb76-bf4cc251f369'
UID = 200171
INSTANCES = {
    3031: dict(exe='/usr/libexec/gnome-remote-desktop-daemon', start_ticks=856,
               cmdline='/usr/libexec/gnome-remote-desktop-daemon'),
    794777: dict(exe='/usr/bin/nvidia-cuda-mps-server', start_ticks=755744669,
                 cmdline='nvidia-cuda-mps-server'),
    2880716: dict(exe='/usr/bin/nvidia-cuda-mps-control', start_ticks=167960061,
                  cmdline='nvidia-cuda-mps-control -d'),
}
GPU_PIDS = {3031, 794777}


def validate_records(boot_id, records, server_ids, clients):
    require(boot_id == BOOT_ID, 'host baseline belongs to another boot')
    require(set(records) <= set(INSTANCES), 'unapproved host baseline process')
    for pid, actual in records.items():
        require(actual == dict(INSTANCES[pid], uid=UID),
                'host baseline identity/start time differs: ' + str(pid))
    expected_servers = [794777] if 794777 in records else []
    require(server_ids == expected_servers, 'foreign/missing MPS server')
    require(not server_ids or 2880716 in records, 'MPS control owner missing')
    require(not clients, 'pre-existing MPS has an active client')
    return set(records)


def inspect():
    records = {}
    for pid in INSTANCES:
        path = Path('/proc') / str(pid)
        if not path.exists():
            continue
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        records[pid] = dict(exe=os.readlink(path / 'exe'), start_ticks=int(fields[19]),
            uid=path.stat().st_uid,
            cmdline=(path / 'cmdline').read_bytes().replace(b'\0', b' ').decode().strip())
    # Verify identities BEFORE contacting the pre-existing daemon.
    validate_records(Path('/proc/sys/kernel/random/boot_id').read_text().strip(), records,
                     [794777] if 794777 in records else [], [])
    replies = {}
    def query(command):
        env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY='/tmp/nvidia-mps')
        result = subprocess.run(['/usr/bin/nvidia-cuda-mps-control'], input=command+'\n',
            text=True, capture_output=True, timeout=10, env=env)
        require(result.returncode == 0, 'MPS ownership query failed')
        replies[command] = dict(stdout=result.stdout, stderr=result.stderr)
        return [int(v) for v in result.stdout.split()]
    servers = query('get_server_list') if 2880716 in records else []
    clients = query('get_client_list 794777') if servers == [794777] else []
    allowed = validate_records(BOOT_ID, records, servers, clients)
    return dict(policy='EXPLICIT_PREEXISTING_BASELINE_20261013',
        exclusive_gpu_claim=False, allowed_process_pids=sorted(allowed),
        allowed_gpu_pids=sorted(allowed & GPU_PIDS), identities=records,
        mps_queries=replies, mps_client_count=len(clients))


def validate_gpu_rows(text, allowed):
    rows = []
    for line in text.splitlines():
        fields = line.split(',', 1)
        require(len(fields) == 2 and fields[0].strip().isdigit(), 'malformed GPU client row')
        pid = int(fields[0])
        require(pid in allowed, 'GPU has an unapproved compute client: '+str(pid))
        # NVML reports the MPS server by basename. The actual executable, UID,
        # boot and start time were already checked via /proc before admission.
        exe = INSTANCES[pid]['exe']
        require(fields[1].strip() in (exe, Path(exe).name), 'GPU executable identity differs')
        rows.append(pid)
    return rows
