"""One bounded passive diagnostic on the accepted CPU-partition configuration."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess as S
import sys
import time

CODE = Path(__file__).resolve().parent
CPU = CODE.parent / 'splitfusion_cpu_partition_v1'
sys.path.insert(0, str(CPU))
import experiment as E
from verify_retained_selection import verify
from startup_gate import raw, write
from packet_capture import start_capture

TOKEN = 'RETAINED_TRANSPORT_LISTENER_DIAGNOSTIC_300_ONCE'
RUNNER = 'rl_agent.splitfusion_single_gpu_profiler_v1.runner_transport_diagnostic_v1'
CHILD = 'rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_transport_diagnostic_v1'


def prepare(root):
    selection = verify()
    root.mkdir(parents=True, exist_ok=False)
    arm = root / 'treatment'; arm.mkdir()
    registration = json.loads((Path(selection['evidence_root']) / 'treatment/registration.json').read_text())
    sources = {v['path']: v['sha256'] for v in registration['experiment_binds'].values()}
    for path in [*CODE.glob('*.py'), E.PKG / 'runner_transport_diagnostic_v1.py',
                 E.PKG / 'child_v1_transport_diagnostic_v1.py']:
        sources[str(path)] = E.sha(path)
    write(root / 'PROSPECTIVE.json', dict(selection=selection, sources=sources,
        budget=300, boundary_exception=299, single_attempt_no_live_retry=True,
        purpose='DIAGNOSTIC_NOT_FINAL_AFTER_PERFORMANCE_MEASUREMENT',
        changed='Passive endpoint packet capture and targeted listener scheduler/recvfrom trace; setup-only listener barrier and durable existing modem logs.',
        unchanged=['actor','state','reward','170ms inclusive deadline','OAI source/config/priorities',
                   'CPU partition','CPU radar','UE thread cap','precision','MPS','queue semantics'],
        exact_radio_bsr_crc_available=False, broad_radio_debug_enabled=False,
        maximum_packet_capture_seconds=600, maximum_scheduler_bytes=134217728,
        diagnostic_clock='PCAP wall timestamps bridged by qualified periodic wall/RAW anchors; scheduler CLOCK_MONOTONIC_RAW',
        snapshot_prefix_only=True, full_payload_integrity_claim=False))
    write(arm / 'arm.json', dict(arm='treatment', original_cpus=E.ALL, radio_cpus=E.RADIO, edge_cpus=E.EDGE))
    (arm / 'compose_partition.yaml').write_text('services:\n  oai-perception-rx:\n    volumes:\n'
        f'      - {arm}:/partition_output:rw\n      - {CPU}:/cpu_partition_code:ro\n')
    registration.update(id='RETAINED_TRANSPORT_LISTENER_DIAGNOSTIC_V1', child_module=CHILD,
        transmitted_budget=300, g1_boundary_budget=300,
        claim_scope='Passive diagnostic, no optimization or performance qualification claim')
    registration['arms']['C2']['execute_token'] = TOKEN
    registration['experiment_binds'] = {str(i):dict(path=p,sha256=h) for i,(p,h) in enumerate(sources.items())}
    registration.pop('registration_sha256', None)
    registration['registration_sha256'] = E.canonical(registration)
    write(arm / 'registration.json', registration)
    E.execute(['git','status','--short'],root/'git_before.txt',cwd=E.WORKTREE)
    print(root, flush=True)


def environment(arm):
    env = dict(os.environ, SCENESENSE_PARTITION_OUTPUT=str(arm), SCENESENSE_PARTITION_CODE=str(CPU),
               SCENESENSE_TRANSPORT_CODE=str(CODE), PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None)
    assert not any(env.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','LD_PRELOAD'))
    assert env.get('CUDA_VISIBLE_DEVICES', 'unset') != '', 'GPU must be visible'
    return env


def run(root):
    prospective = json.loads((root/'PROSPECTIVE.json').read_text())
    assert all(E.sha(p)==h for p,h in prospective['sources'].items()), 'Source drift'
    verify()
    arm=root/'treatment'; env=environment(arm)
    assert not (root/'controller_events.jsonl').exists() and not (arm/'capture').exists()
    def event(kind, **fields):
        row=dict(kind=kind,raw_ns=raw(),wall_ns=time.time_ns(),**fields)
        with (root/'controller_events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    E.execute([sys.executable,'-m',RUNNER,'--registration',str(arm/'registration.json'),
        '--arm','C2','--output-root',str(arm/'capture'),'--preflight'],root/'preflight.log',
        cwd=E.WORKTREE,env=env,timeout=120)
    E.execute([sys.executable,str(CPU/'experiment.py'),'postflight',str(root),'--label','before'],
              root/'before.log',cwd=E.WORKTREE)
    for name in ('sys_enter_recvfrom','sys_exit_recvfrom'):
        S.run(['sudo','-n','test','-r',f'/sys/kernel/tracing/events/syscalls/{name}/format'],check=True)
    event('settle',seconds=60)
    samples=[]
    for _ in range(12):
        samples.append(dict(raw_ns=raw(),load=Path('/proc/loadavg').read_text().strip()));time.sleep(5)
    assert os.getloadavg()[0]<=4 and not S.check_output(['docker','ps','--format','{{.Names}}'],text=True).strip()
    write(root/'settle.json',samples)
    argv=[sys.executable,'-m',RUNNER,'--registration',str(arm/'registration.json'),'--arm','C2',
          '--output-root',str(arm/'capture'),'--execute',TOKEN]
    (arm/'scheduler').mkdir()
    app=S.Popen(argv,cwd=E.WORKTREE,env=env,stdout=(arm/'runner.log').open('x'),stderr=S.STDOUT,start_new_session=True)
    event('launch',pid=app.pid,argv=argv)
    perf=monitor=session=None; stopped=False

    def wait_live(path,seconds):
        deadline=time.monotonic()+seconds
        while not path.exists():
            if app.poll() is not None:raise RuntimeError(f'Runner exited before {path.name}')
            if time.monotonic()>deadline:raise TimeoutError(str(path))
            time.sleep(.05)

    def drain():
        nonlocal stopped
        if stopped:return json.loads((arm/'COLLECTORS_STOPPED.json').read_text())
        (arm/'collection_stop').touch()
        errors=[]; packet=None
        if session:
            try:
                packet=session.stop()
                if not packet['passed']:errors.extend(packet['errors'])
            except BaseException as exc:errors.append(repr(exc))
        for label,proc in [('perf',perf),('monitor',monitor)]:
            if proc:
                try:
                    rc=proc.wait(timeout=12)
                    if rc:errors.append(f'{label} returned {rc}')
                except S.TimeoutExpired:
                    errors.append(f'{label} drain timeout')
                    try:
                        S.run(['sudo','-n','kill','-TERM','--',str(-proc.pid)],timeout=5,check=True)
                        proc.wait(timeout=5)
                    except BaseException as exc:
                        errors.append(f'{label} terminate: {exc!r}')
                        try:
                            S.run(['sudo','-n','kill','-KILL','--',str(-proc.pid)],timeout=5,check=True)
                            proc.wait(timeout=5)
                        except BaseException as kill_exc:errors.append(f'{label} kill: {kill_exc!r}')
        result=dict(raw_ns=raw(),errors=errors,packet=packet,passed=not errors)
        write(arm/'COLLECTORS_STOPPED.json',result);stopped=True
        event('collectors_drained',passed=result['passed'])
        return result

    try:
        wait_live(arm/'worker_ready.json',240)
        E.execute(['sudo','-n',sys.executable,str(CPU/'host_partition.py'),'apply',str(arm)],arm/'apply.log',30)
        info=json.loads((arm/'AFFINITY_APPLIED.json').read_text())
        tid=info['before']['target']['tid'];edge_pid=info['before']['target']['pid']
        monitor=S.Popen(['sudo','-n',sys.executable,str(CPU/'host_partition.py'),'monitor',str(arm)],
                        stdout=(arm/'monitor.log').open('x'),stderr=S.STDOUT,start_new_session=True)
        write(arm/'release.json',dict(expected_affinity=info['expected_affinity'],raw_ns=raw()))
        wait_live(arm/'LISTENER_READY.json',180)
        listener=json.loads((arm/'LISTENER_READY.json').read_text());lid=listener['listener']['tid']
        assert Path(f'/proc/{listener["pid"]}/task/{lid}/stat').read_text().rsplit(')',1)[1].split()[19] == listener['listener']['starttime']
        journal=json.loads((arm/'PARTITION_JOURNAL.json').read_text())
        identity=next(p for p in journal['processes'] if p['pid']==edge_pid)
        session=start_capture(arm/'packets',edge_pid=edge_pid,edge_starttime=identity['pid_starttime'],
            ue_ip=listener['receiver'][0],edge_ip=listener['remote'][0],
            edge_port=listener['remote'][1],feedback_port=listener['receiver'][1])
        ru=[t['tid'] for p in info['radio'] for t in p['threads'].values() if t['comm']=='ru_thread']
        assert len(ru)==1
        tids=[tid,lid,*ru]
        switch=' || '.join(f'prev_pid == {t} || next_pid == {t}' for t in tids)
        wake=' || '.join(f'pid == {t}' for t in [lid,tid])
        command=['sudo','-n','perf','record','-a','--clockid','CLOCK_MONOTONIC_RAW','--synth=no',
            '--no-buildid','--no-buildid-cache','--no-bpf-event','-m','128','--max-size','128M',
            '-o',str(arm/'scheduler/sched.data'),'-e','sched:sched_switch','--filter',switch,
            '-e','sched:sched_wakeup','--filter',wake,'-e','sched:sched_wakeup_new','--filter',wake]
        for action in ('enter','exit'):
            command+=['-e',f'syscalls:sys_{action}_recvfrom','--filter',f'common_pid == {lid}']
        command+=['--',sys.executable,str(E.PREVIOUS/'perf_sentinel.py'),str(arm)]
        write(arm/'perf_command.json',command)
        perf=S.Popen(command,stdout=(arm/'scheduler/perf.log').open('x'),stderr=S.STDOUT,start_new_session=True)
        wait_live(arm/'perf_ready.json',15);assert perf.poll() is None
        write(arm/'LISTENER_RELEASE.json',dict(raw_ns=raw()))
        wait_live(arm/'LISTENER_RELEASED.json',5)
        event('live',listener_tid=lid,edge_tid=tid,ru_tids=ru)
        deadline=time.monotonic()+900;last=time.monotonic()
        while app.poll() is None:
            if (arm/'SCOPE_VIOLATION.json').exists():raise RuntimeError('Partition scope drift')
            if time.monotonic()>deadline:raise TimeoutError('Route lifecycle')
            if (arm/'collection_stop').exists() or (arm/'TRANSPORT_STOP_REQUESTED.json').exists():drain()
            elif session:
                session.check_health()
                if perf.poll() is not None or monitor.poll() is not None:
                    raise RuntimeError('Scheduler/scope collector exited before route teardown')
            if time.monotonic()-last>20:event('running');last=time.monotonic()
            time.sleep(.1)
        drained=drain()
        assert app.returncode==0,('runner return',app.returncode)
        assert drained['passed'],drained
        assert json.loads((arm/'perf_sentinel_footer.json').read_text())['stop_marker']
        E.export_perf(arm/'scheduler/sched.data')
        result=json.loads((arm/'capture/QUALIFICATION_RESULT.json').read_text())
        assert result['passed'] and result['threadcap']['all_passed'],result['gates']
        assert json.loads((arm/'monitor_footer.json').read_text())['passed']
        assert json.loads((arm/'RESTORATION_LATEST.json').read_text())['passed']
        for name in ('COLLECTOR_DRAIN_TIMEOUT.json','RADIO_LOG_PRESERVATION_ERROR.json'):
            assert not (arm/name).exists(),name
        assert (arm/'radio_logs/gnb.log').is_file() and (arm/'radio_logs/ue.log').is_file()
        E.execute([sys.executable,str(CPU/'experiment.py'),'postflight',str(root),'--label','after'],
                  root/'after.log',cwd=E.WORKTREE)
        event('complete',sent=result['g1_detail']['sent'],capture_verdict=result['verdict'])
    except BaseException as exc:
        event('failure',error=repr(exc))
        try:drain()
        except BaseException as cleanup_exc:event('collector_cleanup_error',error=repr(cleanup_exc))
        try:
            if (arm/'PARTITION_JOURNAL.json').exists():
                restored=S.run(['sudo','-n',sys.executable,str(CPU/'host_partition.py'),'restore',str(arm)],timeout=25)
                event('failure_restore',returncode=restored.returncode)
        except BaseException as cleanup_exc:event('restore_cleanup_error',error=repr(cleanup_exc))
        finally:
            if app.poll() is None:
                os.killpg(app.pid,signal.SIGINT)
                try:app.wait(timeout=180)
                except S.TimeoutExpired:event('runner_cleanup_timeout',pid=app.pid)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('prepare','run'))
    parser.add_argument('root',type=Path);args=parser.parse_args()
    (prepare if args.operation=='prepare' else run)(args.root.resolve())
