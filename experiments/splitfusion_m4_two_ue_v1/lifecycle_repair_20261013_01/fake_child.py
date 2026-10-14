"""Offline supervisor probe child: real phases/OwnedProcesses records, synthetic CARLA (sleep group)."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT, capture, scenario = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(ROOT))
from rl_agent.splitfusion_m4_two_ue_v1 import configuration as C, phases
from rl_agent.splitfusion_m4_two_ue_v1.resources import OwnedProcesses, process_birth
from data_collection.run_route_b_collection_supervised_v1 import process_group_members

capture.mkdir()
C.write_new(capture/'PREPARATION_REFERENCE.json', dict(path='synthetic', sha256='0'*64))
P = phases.install(capture)
owner = OwnedProcesses(capture)
C.write_new(capture/'HOST_OWNER.json', dict(pid=os.getpid(), birth=process_birth(os.getpid())))
P.step('startup.radio_attach')
owner.start('fake_radio', ['sleep', '300'], cwd=str(capture))
P.step('startup.carla_start')
carla_argv = ['bash', '-c', 'trap "" TERM; sleep 300'] if scenario == 'ignore_term' else ['sleep', '300']
carla = subprocess.Popen(carla_argv, start_new_session=True, stdin=subprocess.DEVNULL)
pgid = os.getpgid(carla.pid)
C.write_new(capture/'CARLA_OWNER.json', dict(pid=carla.pid, pgid=pgid))
import time; time.sleep(.3)
C.write_new(capture/'CARLA_STARTED.json', dict(pgid=pgid, rpc_port=2999,
    members=[dict(row, birth=process_birth(row['pid'])) for row in process_group_members(pgid)]))
P.step(None)
if scenario == 'abort':
    try:
        with P.phase('run.world_and_routes'): raise ValueError('synthetic first failure')
    except ValueError as error:
        C.write_new(capture/'ORIGINAL_FAILURE.json', dict(error=repr(error)))
    with P.phase('teardown.gt_m4-ue0', stage='teardown'): pass
    P.event('teardown.carla_delegated_to_supervisor', pgid=pgid)
    with P.phase('teardown.processes', stage='teardown'):
        os.abort()  # native abort with owned radio still running
with P.phase('teardown.processes', stage='teardown'):
    closed = owner.close()
C.write_new(capture/'CLEANUP.json', dict(processes=closed, carla=dict(delegated_to_supervisor=True, pgid=pgid)))
C.write_new(capture/'RUN_RESULT.json', dict(status='SMOKE_ONLY', error=None, returncode=0, cleanup_failures=[]))
C.write_new(capture/'SMOKE.json', dict(pass_=True))
sys.exit(0)
