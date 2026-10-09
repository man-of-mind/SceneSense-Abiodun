"""Narrow tracepoints, RAW clock, durable bounded collection, conservative state parsing."""
import collections,json,os,re,signal,subprocess as S,time
from pathlib import Path

def raw(): return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
def write(p,x): Path(p).write_text(json.dumps(x,indent=2)+'\n')
def command(argv,p,timeout=30):
    with Path(p).open('x') as f:
        proc=S.Popen(argv,stdout=f,stderr=S.STDOUT,start_new_session=True)
        try: rc=proc.wait(timeout=timeout)
        except S.TimeoutExpired:
            # Exact owned process group, never a name-wide cleanup.
            S.run(['sudo','-n','kill','-TERM','--',str(-proc.pid)],timeout=5)
            try: proc.wait(timeout=5)
            except S.TimeoutExpired:S.run(['sudo','-n','kill','-KILL','--',str(-proc.pid)],timeout=5);proc.wait(timeout=5)
            write(str(p)+'.timeout.json',dict(argv=argv,preserved=True,raw_ns=raw()));raise
    if rc:raise RuntimeError((argv,rc,str(p)))

def perf_argv(tid,path,duration):
    a=['sudo','-n','perf','record','-a','--clockid','CLOCK_MONOTONIC_RAW','--synth=no','--no-buildid','--no-buildid-cache','--no-bpf-event','-m','128','--max-size','64M','-o',str(path)]
    for event,filt in [('sched:sched_switch',f'prev_pid == {tid} || next_pid == {tid}'),('sched:sched_wakeup',f'pid == {tid}'),('sched:sched_wakeup_new',f'pid == {tid}')]+[(f'syscalls:sys_{action}_{call}',f'common_pid == {tid}') for call in ['futex','ioctl','clock_nanosleep','read','poll','ppoll'] for action in ['enter','exit']]:
        a+=['-e',event,'--filter',filt]
    return a+['--','sleep',str(duration)]

def export_perf(path):
    path=Path(path)
    command(['sudo','-n','perf','script','-i',str(path),'--ns','--show-lost-events','-F','trace:tid,cpu,time,event,trace'],path.with_suffix('.txt'),30)
    command(['sudo','-n','perf','script','-i',str(path),'--header-only'],path.with_suffix('.header.txt'),15)
    command(['sudo','-n','perf','script','-i',str(path),'-d'],path.with_suffix('.debug.txt'),30)

def parse_perf(path,tid):
    events=[]; anomalies=[]
    pat=re.compile(r'^\s*(-?\d+)\s+\[(\d+)\]\s+(\d+)\.(\d+):\s+(\S+):\s*(.*)$')
    for line in Path(path).read_text().splitlines():
        m=pat.match(line)
        if not m:
            if line.strip() and not (line.startswith("'trace' not valid for " ) and line.endswith(' Ignoring.')):anomalies.append(line)
            continue
        source,cpu,sec,frac,event,trace=m.groups(); ns=int(sec)*10**9+int(frac.ljust(9,'0'))
        fields=dict(re.findall(r'(\w+)=([^ ]+)',trace))
        events.append(dict(ns=ns,source_tid=int(source),cpu=int(cpu),event=event,trace=trace,fields=fields))
    events.sort(key=lambda x:x['ns'])
    # Off-CPU episodes: state at switch-out, then wakeup(s), then next switch-in.
    segments=[]; episodes=[]; last=None; state='unknown'; out=None; wakes=[]; errors=[]
    def seg(end):
        if last is not None and end>last:segments.append(dict(start=last,end=end,state=state))
    for x in events:
        e=x['event'];f=x['fields'];t=x['ns']
        if e=='sched:sched_switch':
            if int(f['prev_pid'])==tid:
                if state=='running':seg(t)
                elif last is not None:
                    seg(t);errors.append(dict(reason='out_without_running',ns=t,state=state))
                last=t;state='runnable' if f['prev_state'].startswith('R') else 'blocked'
                out=x;wakes=[]
            if int(f['next_pid'])==tid:
                if out is not None:
                    if state=='blocked':
                        # Lost/unobserved wake: last blocked segment cannot be cleanly partitioned.
                        state='unknown';errors.append(dict(reason='blocked_in_without_wakeup',ns=t))
                    seg(t);episodes.append(dict(start=out['ns'],end=t,prev_state=out['fields']['prev_state'],out_cpu=out['cpu'],in_cpu=x['cpu'],wakeups=wakes))
                elif last is not None:seg(t)
                state='running';last=t;out=None;wakes=[]
        elif e in ['sched:sched_wakeup','sched:sched_wakeup_new']:
            if int(f['pid'])!=tid:raise AssertionError('filter mismatch')
            if out is not None:wakes.append(x)
            if state=='blocked':seg(t);last=t;state='runnable'
    return dict(events=events,segments=segments,episodes=episodes,parse_anomalies=anomalies,state_errors=errors)

def partition(segments,a,b):
    d=dict(running=0,runnable=0,blocked=0,unknown=0)
    for x in segments:d[x['state']]+=max(0,min(b,x['end'])-max(a,x['start']))
    d['unknown']+=b-a-sum(d.values())
    assert min(d.values())>=0 and sum(d.values())==b-a
    return {k:v/1e6 for k,v in d.items()}
