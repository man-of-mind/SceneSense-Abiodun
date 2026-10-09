"""Bounded controlled radio/edge CPU partition, retaining scientific pipeline."""
import argparse
import hashlib
import json
import os
import signal
import subprocess as S
import sys
import time
from pathlib import Path

from edge_gate import raw, write, wait
from host_partition import PREVIOUS

CODE = Path(__file__).resolve().parent
WORKTREE = CODE.parents[1].parent / 'abiodun_run4_single_gpu_profiler_v1'
sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(PREVIOUS))
from sched_tools import export_perf

ALL = list(range(24))
RADIO = list(range(4)) + list(range(8, 20))
EDGE = list(range(4, 8)) + list(range(20, 24))
PKG = WORKTREE / 'rl_agent/splitfusion_single_gpu_profiler_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def execute(command, log, timeout=90, cwd=None, env=None):
    with Path(log).open('x') as stream:
        result = S.run(command, stdout=stream, stderr=S.STDOUT, timeout=timeout, cwd=cwd, env=env)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}); inspect {log}')


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    assert not set(RADIO) & set(EDGE) and set(RADIO) | set(EDGE) == set(ALL)
    sources = list(CODE.glob('*.py')) + [PKG / name for name in (
        'runner_cpu_partition_v1.py', 'child_v1_cpu_partition_v1.py', 'edge_profiled_v1_cpu_partition_v1.py')]
    sources += [PREVIOUS / name for name in ('host_affinity.py', 'injection_guard_v2.py',
                                             'sched_tools.py', 'perf_sentinel.py', 'affinity_lib.py')]
    evidence = dict(scope='ALL_THREADS_OF_VERIFIED_EDGE_AND_SIX_RADIO_PROCESSES_INCLUDING_FUTURE_THREADS',
        sequence=['control', 'treatment'], budget=1000, boundary_exception=999,
        radio_cpus=RADIO, edge_cpus=EDGE, original_cpus=ALL,
        cpu_choice_rationale='Split includes lower and higher CPU IDs for both roles. 100ms retained samples show combined OAI mean2.65/max4.31 CPU and edge mean0.61/max1.30; instantaneous capacity remains unproven.',
        unchanged=['actor','state','reward','170ms_deadline','radio_configuration','OAI_priorities',
                   'CPU_radar','serial_map_then_evaluator','UE_thread_cap','precision','MPS','queues'],
        changed='Only process/thread CPU-affinity masks for the attached six radio PIDs and verified edge PID.',
        acceptance=dict(timely_gain_pp_min=5, runnable_p99_reduction_fraction_min=.3,
            tail_over_100_rate_reduction_fraction_min=.3,
            regression_limits=dict(missing_reply_pp_increase=1, throughput_decrease_fraction=.05,
                action_feedback_p50_increase_fraction=.05, map_p95_increase_fraction=.1, edge_queue_p95_increase_fraction=.1)),
        sources={str(p): sha(p) for p in sources}, parent_report_sha256=sha(PREVIOUS/'REPORT.md'),
        git_head=S.check_output(['git','rev-parse','HEAD'],cwd=WORKTREE,text=True).strip(),
        registered_before_live=True, no_heavy_nsight=True)
    write(root/'PROSPECTIVE.json', evidence)
    execute(['lscpu','--json'], root/'lscpu.json')
    execute(['git','status','--short'], root/'git_before.txt', cwd=WORKTREE)
    for arm in ['control','treatment']:
        directory = root/arm; directory.mkdir()
        spec=dict(arm=arm, original_cpus=ALL, radio_cpus=RADIO, edge_cpus=EDGE)
        write(directory/'arm.json',spec)
        (directory/'compose_partition.yaml').write_text('services:\n  oai-perception-rx:\n    volumes:\n'
            f'      - {directory}:/partition_output:rw\n      - {CODE}:/cpu_partition_code:ro\n')
        registration=json.loads((PREVIOUS/'control/registration.json').read_text())
        registration['id']='OAI_EDGE_DISJOINT_CPU_PARTITION_V1_'+arm.upper()
        registration['child_module']='rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_cpu_partition_v1'
        registration['arms']['C2']['execute_token']='CPU_PARTITION_'+arm.upper()+'_ONCE_EXECUTE'
        registration['claim_scope']=evidence['scope']+'; exploratory controlled pair, no statistical proof'
        registration['experiment_binds']={p.name+':'+str(i):dict(path=str(p),sha256=sha(p)) for i,p in enumerate(sources)}
        registration.pop('registration_sha256',None)
        registration['registration_sha256']=canonical(registration)
        write(directory/'registration.json',registration)
    print(root,flush=True)


def postflight(root, label):
    from rl_agent.splitfusion_single_gpu_profiler_v1 import runner_v1 as R
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_runner_v2 as L
    from rl_agent import ue_288_campaign_supervisor as SUP
    from rl_agent.splitfusion_quality_feedback_probe_v1 import live_probe as LP
    from injection_guard_v2 import collect
    guard=collect(os.getpid()); assert guard['verdict']['passed'],guard['verdict']
    config,_,_=L.offline_preflight(L.DEFAULT_CONFIG,output_root=root/(label+'_unused'),transmitted_budget=1000)
    cold=SUP._require_phase15_application_cold(LP._probe_campaign(config,run_id='cpu_partition_'+label))
    audit=R.run_audit('rl_agent.splitfusion_hybrid_sac_checkpoint_sidecar_v1.audit_run4_selected_actor_v1',root/(label+'_actor.json'))
    write(root/(label+'.json'),dict(cold=cold,audit=audit,guard=guard,raw_ns=raw()))
    assert audit['returncode']==0 and all(audit['document']['checks'].values())


def run_pair(root):
    prospective=json.loads((root/'PROSPECTIVE.json').read_text())
    assert all(sha(p)==digest for p,digest in prospective['sources'].items()),'Bound source drift'
    assert all(json.loads((root/('torch_'+a+'.json')).read_text())['passed'] for a in ['control','treatment'])
    assert not (root/'controller_events.jsonl').exists()
    def event(kind,**fields):
        row=dict(kind=kind,raw_ns=raw(),wall_ns=time.time_ns(),**fields)
        with (root/'controller_events.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    for arm in ['control','treatment']:
        directory=root/arm; assert not (directory/'capture').exists()
        execute([sys.executable,str(CODE/'experiment.py'),'postflight',str(root),'--label',arm+'_pre'],root/(arm+'_pre.log'),cwd=WORKTREE)
        event('settle',arm=arm,seconds=60)
        samples=[]; start=time.monotonic()
        while time.monotonic()-start<60:
            samples.append(dict(raw_ns=raw(),load=Path('/proc/loadavg').read_text().strip()));time.sleep(5)
        assert os.getloadavg()[0]<=4,os.getloadavg()
        assert not S.check_output(['docker','ps','--format','{{.Names}}'],text=True).strip()
        write(root/(arm+'_settle.json'),dict(samples=samples,gpu=S.check_output(['nvidia-smi','--query-gpu=temperature.gpu,utilization.gpu,memory.used','--format=csv,noheader'],text=True)))
        env=dict(os.environ,SCENESENSE_PARTITION_OUTPUT=str(directory),SCENESENSE_PARTITION_CODE=str(CODE),PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH',None)
        assert not any(env.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','LD_PRELOAD'])
        argv=[sys.executable,'-m','rl_agent.splitfusion_single_gpu_profiler_v1.runner_cpu_partition_v1',
            '--registration',str(directory/'registration.json'),'--arm','C2','--output-root',str(directory/'capture'),
            '--execute','CPU_PARTITION_'+arm.upper()+'_ONCE_EXECUTE']
        (directory/'scheduler').mkdir()
        log=(directory/'runner.log').open('x'); app=S.Popen(argv,cwd=WORKTREE,env=env,stdout=log,stderr=S.STDOUT,start_new_session=True)
        event('launch',arm=arm,pid=app.pid,argv=argv); perf=monitor=None
        try:
            deadline=time.monotonic()+240
            while not (directory/'worker_ready.json').exists():
                if app.poll() is not None: raise RuntimeError('runner ended before edge ready')
                if time.monotonic()>deadline: raise TimeoutError('edge worker READY')
                time.sleep(.1)
            execute(['sudo','-n',sys.executable,str(CODE/'host_partition.py'),'apply',str(directory)],directory/'apply.log',30)
            info=json.loads((directory/'AFFINITY_APPLIED.json').read_text()); tid=info['before']['target']['tid']
            ru=[t['tid'] for p in info['radio'] for t in p['threads'].values() if t['comm']=='ru_thread']; assert len(ru)==1,ru
            switch=' || '.join(f'prev_pid == {t} || next_pid == {t}' for t in [tid,*ru])
            argv_perf=['sudo','-n','perf','record','-a','--clockid','CLOCK_MONOTONIC_RAW','--synth=no',
                '--no-buildid','--no-buildid-cache','--no-bpf-event','-m','128','--max-size','128M',
                '-o',str(directory/'scheduler/sched.data'),'-e','sched:sched_switch','--filter',switch,
                '-e','sched:sched_wakeup','--filter',f'pid == {tid}',
                '-e','sched:sched_wakeup_new','--filter',f'pid == {tid}',
                '--',sys.executable,str(PREVIOUS/'perf_sentinel.py'),str(directory)]
            write(directory/'perf_command.json',argv_perf)
            perf=S.Popen(argv_perf,stdout=(directory/'scheduler/perf.log').open('x'),stderr=S.STDOUT,start_new_session=True)
            wait(directory/'perf_ready.json',15); assert perf.poll() is None
            monitor=S.Popen(['sudo','-n',sys.executable,str(CODE/'host_partition.py'),'monitor',str(directory)],stdout=(directory/'monitor.log').open('x'),stderr=S.STDOUT,start_new_session=True)
            write(directory/'release.json',dict(expected_affinity=info['expected_affinity'],raw_ns=raw()))
            wait(directory/'ready_released.json',5)
            event('live',arm=arm,compute_tid=tid,ru_tids=ru,mask=info['expected_affinity'])
            deadline=time.monotonic()+1000; last=time.monotonic()
            while app.poll() is None:
                if (directory/'SCOPE_VIOLATION.json').exists(): raise RuntimeError('Scope drift; see SCOPE_VIOLATION')
                if time.monotonic()>deadline: raise TimeoutError('route lifecycle')
                if time.monotonic()-last>20: event('running',arm=arm); last=time.monotonic()
                time.sleep(.25)
            assert app.returncode==0,('runner return',app.returncode)
            (directory/'collection_stop').touch()
            assert perf.wait(35)==0 and monitor.wait(10)==0
            export_perf(directory/'scheduler/sched.data')
            result=json.loads((directory/'capture/QUALIFICATION_RESULT.json').read_text())
            assert result['passed'] and result['threadcap']['all_passed'],result['gates']
            assert json.loads((directory/'monitor_footer.json').read_text())['passed']
            assert json.loads((directory/'RESTORATION_LATEST.json').read_text())['passed']
            # Before container/radio removal, all surviving target threads must have been restored.
            for name in ['prestop_restore_command.json','radio_prestop_restore_command.json']:
                assert json.loads((directory/name).read_text())['returncode']==0,name
            execute([sys.executable,str(CODE/'experiment.py'),'postflight',str(root),'--label',arm+'_post'],root/(arm+'_post.log'),cwd=WORKTREE)
            event('arm_complete',arm=arm,sent=result['g1_detail']['sent'],verdict=result['verdict'])
        except BaseException as exc:
            event('failure',arm=arm,error=repr(exc)); (directory/'collection_stop').touch()
            if (directory/'PARTITION_JOURNAL.json').exists():
                try:
                    restored = S.run(['sudo','-n',sys.executable,str(CODE/'host_partition.py'),'restore',str(directory)],timeout=25)
                    event('failure_path_restore',returncode=restored.returncode)
                except BaseException as restore_exc:
                    event('failure_path_restore_error',error=repr(restore_exc))
            if app.poll() is None:
                os.killpg(app.pid,signal.SIGINT)
                try: app.wait(timeout=180)
                except S.TimeoutExpired: event('cleanup_timeout',arm=arm,pid=app.pid)
            for proc in [perf,monitor]:
                if proc:
                    try: proc.wait(timeout=35)
                    except S.TimeoutExpired:
                        event('collector_timeout',pid=proc.pid)
                        S.run(['sudo','-n','kill','-TERM','--',str(-proc.pid)],timeout=5)
                        try: proc.wait(timeout=10)
                        except S.TimeoutExpired:
                            S.run(['sudo','-n','kill','-KILL','--',str(-proc.pid)],timeout=5)
                            proc.wait(timeout=5)
            raise
    event('pair_complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['prepare','postflight','run'])
    parser.add_argument('root',type=Path);parser.add_argument('--label');args=parser.parse_args()
    root=args.root.resolve()
    if args.operation=='prepare': prepare(root)
    elif args.operation=='postflight': postflight(root,args.label)
    else: run_pair(root)
