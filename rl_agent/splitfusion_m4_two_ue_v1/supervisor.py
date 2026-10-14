"""External parent for one M4 live attempt. Importing this module starts nothing.

Same pattern as data_collection/run_route_b_collection_supervised_v1: the
CARLA client runs in a child process; this parent never imports carla/torch,
records the child's real exit status (including a terminating signal), and
only then stops the owned CARLA group. A native abort in the client can no
longer skip process cleanup or erase the recorded outcome. Cleanup is the
existing ownership-checked resources.recover path; nothing is guessed.
"""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import configuration as C

SCHEMA = 'splitfusion.m4.supervised_attempt.v1'
CHILD_COMMAND = 'supervised-child'


def _read(path, default=None):
    try: return json.loads(Path(path).read_bytes())
    except (FileNotFoundError, ValueError): return default


def _phases(path):
    rows = []
    try:
        for line in Path(path).read_text().splitlines():
            try: rows.append(json.loads(line))
            except ValueError: rows.append(dict(event='UNPARSEABLE', raw=line[:200]))
    except FileNotFoundError: pass
    return rows


def _stop_remaining_carla(capture, port):
    """Escalate only on the birth-certified CARLA group, as stop_carla does."""
    from data_collection import run_route_b_collection_supervised_v1 as S
    from .resources import process_birth
    record = _read(Path(capture)/'CARLA_STARTED.json')
    if record is None:
        return dict(certificate=False, owner_record=(Path(capture)/'CARLA_OWNER.json').is_file())
    pgid = int(record['pgid']); known = {p['pid']: p['birth'] for p in record['members']}
    def certified(live):
        out = []
        for p in live:
            try:
                if p['pid'] in known and process_birth(p['pid']) == known[p['pid']]: out.append(p)
            except (FileNotFoundError, ProcessLookupError): pass
        return out
    result = dict(certificate=True, pgid=pgid, members_before=S.process_group_members(pgid))
    if result['members_before']:
        if not certified(result['members_before']):
            result['refused'] = 'CARLA group has no birth-certified member; not signalled'
            return result
        result['sigkill'] = S.signal_group(pgid, signal.SIGKILL)
        deadline = time.monotonic() + 15
        while S.process_group_members(pgid) and time.monotonic() < deadline: time.sleep(.5)
    result['members_after'] = S.process_group_members(pgid)
    result['rpc_port_listening'] = S.rpc_port_listening(int(port))
    result['stopped'] = not result['members_after'] and not result['rpc_port_listening']
    return result


def _cleanup(capture, port):
    """Ownership-checked recovery after the child has exited (always run)."""
    from .resources import recover, check_owned_state
    report = dict(lock=None)
    lock = open('/tmp/splitfusion-live-host.lock', 'a+')
    try:
        for _ in range(40):
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); report['lock'] = 'HELD'; break
            except BlockingIOError: time.sleep(.25)
        else: report['lock'] = 'NOT_ACQUIRED_OWNERSHIP_CHECKS_STILL_APPLY'
        try: report['recover'] = recover(capture, execute=True)
        except BaseException as error: report['recover_error'] = repr(error)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            owned = check_owned_state(capture)
            if not any(r['still_owned'] for r in owned): break
            time.sleep(1.)
        report['owned_after'] = [dict(name=r['name'], pid=r['pid'], still_owned=r['still_owned'])
                                 for r in check_owned_state(capture)]
        report['carla'] = _stop_remaining_carla(capture, port)
        container = _read(Path(capture)/'EDGE_CONTAINER.json')
        if container is not None:
            probe = subprocess.run(['sudo', '-n', 'docker', 'inspect', container['name']],
                                   capture_output=True, text=True, timeout=20)
            report['edge_container_present'] = probe.returncode == 0
    finally:
        if report['lock'] == 'HELD': fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    report['verified'] = bool('recover_error' not in report
        and not any(r['still_owned'] for r in report['owned_after'])
        and report['carla'].get('stopped', not report['carla'].get('certificate')
                                and not report['carla'].get('owner_record'))
        and not report.get('edge_container_present', False))
    return report


def run(preparation, output, *, smoke, token, child_argv=None):
    """child_argv is for offline supervisor checks only; live uses the CLI child."""
    from data_collection.run_route_b_collection_supervised_v1 import describe_exit
    preparation, output = Path(preparation).absolute(), Path(output).absolute()
    if output.exists():
        raise ValueError('capture output already exists')
    argv = [sys.executable, '-B', '-m', 'rl_agent.splitfusion_m4_two_ue_v1.cli', CHILD_COMMAND,
            'smoke' if smoke else 'capture', '--preparation', str(preparation), '--output', str(output),
            '--execute-token', token] if child_argv is None else list(child_argv)
    started = time.time()
    child = subprocess.Popen(argv, cwd=str(C.ROOT), stdin=subprocess.DEVNULL)  # inherits retained stdout/stderr
    forwarded = []
    def forward(sig, _):
        forwarded.append(int(sig))
        if child.poll() is None: child.send_signal(sig)
    previous = {s: signal.signal(s, forward) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        returncode = child.wait()
    finally:
        for s, handler in previous.items(): signal.signal(s, handler)
    ended = time.time()
    value = dict(schema=SCHEMA, child_argv=argv, child_pid=child.pid, child_exit=describe_exit(returncode),
                 started_wall_s=started, child_ended_wall_s=ended, forwarded_signals=forwarded,
                 parent_imports_carla='carla' in sys.modules, parent_imports_torch='torch' in sys.modules)
    if not output.is_dir():
        value.update(status='REFUSED_BEFORE_CAPTURE', verdict='FAIL')
        print(json.dumps(value, sort_keys=True), file=sys.stderr)
        return returncode if returncode > 0 else 2
    config = _read(preparation)['config']
    phases = _phases(output/'PHASES.jsonl')
    value['child_records'] = {name: (output/name).is_file() for name in
        ('RUN_RESULT.json', 'CLEANUP.json', 'FIRST_FAILURE.json', 'ORIGINAL_FAILURE.json', 'SMOKE.json')}
    value['first_failure'] = _read(output/'FIRST_FAILURE.json')
    value['original_failure'] = _read(output/'ORIGINAL_FAILURE.json')
    value['run_result'] = _read(output/'RUN_RESULT.json')
    value['smoke'] = _read(output/'SMOKE.json')
    value['last_phase_markers'] = phases[-3:]
    entered = [r for r in phases if r.get('event') == 'ENTER']
    closed = {(r.get('phase'), r.get('thread')) for r in phases if r.get('event') in ('DONE', 'ERROR')}
    value['phases_entered_not_closed'] = [r['phase'] for r in entered if (r['phase'], r['thread']) not in closed]
    value['cleanup'] = _cleanup(output, config['carla_port'])
    run_result = value['run_result'] or {}
    reasons = []
    if returncode != 0: reasons.append('child exit ' + json.dumps(value['child_exit'], sort_keys=True))
    if run_result.get('error') or not value['child_records']['RUN_RESULT.json']: reasons.append('child run not clean or unrecorded')
    if run_result.get('cleanup_failures'): reasons.append('child cleanup failures')
    if smoke and not (value['smoke'] or {}).get('pass_'): reasons.append('smoke gates not met')
    if not value['cleanup']['verified']: reasons.append('supervisor cleanup not verified')
    value.update(verdict='PASS' if not reasons else 'FAIL', reasons=reasons, finished_wall_s=time.time())
    C.write_new(output/'SUPERVISOR.json', value)
    print(json.dumps(dict(verdict=value['verdict'], reasons=reasons, child_exit=value['child_exit'],
                          cleanup_verified=value['cleanup']['verified']), sort_keys=True))
    if not reasons: return 0
    if returncode < 0: return 128 - returncode  # shell convention for a signalled child
    return returncode or 3
