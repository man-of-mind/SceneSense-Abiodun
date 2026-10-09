"""Thread-only setup barrier and unconditional affinity restoration; no hot-path patches."""
import json,os,threading,time
from pathlib import Path
def raw():return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
def write(p,x):
    p=Path(p);q=p.with_suffix(p.suffix+'.pending');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def snapshot(tid):
    q=Path('/proc/self/task')/str(tid)
    return dict(pid=os.getpid(),tid=tid,raw_ns=raw(),affinity=sorted(os.sched_getaffinity(tid)),policy=os.sched_getscheduler(tid),priority=os.sched_getparam(tid).sched_priority,starttime=q.joinpath('stat').read_text().rsplit(')',1)[1].split()[19],status=q.joinpath('status').read_text(),cgroup=Path('/proc/self/cgroup').read_text())
def wait(p,timeout=30):
    end=time.monotonic()+timeout
    while not Path(p).exists():
        if time.monotonic()>end:raise TimeoutError(str(p))
        time.sleep(.02)
def run_gate(path,work):
    p=Path(path);tid=threading.get_native_id();original=snapshot(tid);write(p/'worker_ready.json',original)
    try:
        wait(p/'release.json');release=json.loads((p/'release.json').read_text());actual=snapshot(tid)
        assert actual['affinity']==release['expected_affinity'];assert actual['policy']==original['policy'] and actual['priority']==original['priority']
        write(p/'worker_verified.json',actual);return work()
    except BaseException as exc:
        write(p/'worker_failure.json',dict(error=repr(exc),raw_ns=raw()));raise
    finally:
        os.sched_setaffinity(tid,original['affinity']);after=snapshot(tid)
        write(p/'worker_restored.json',dict(original=original,restored=after,passed=after['affinity']==original['affinity']))
def install(path,PW):
    p=Path(path);pending=[];orig_pub=PW.publish_ready_after_warmup;orig_run=threading.Thread.run;orig_start=threading.Thread.start
    def publish(warm,ready,*a,**kw):return orig_pub(warm,lambda:pending.append(ready),*a,**kw)
    PW.publish_ready_after_warmup=publish
    def target(t):
        f=getattr(t,'_target',None)
        return t.name=='run4-edge-process' and getattr(f,'__name__','')=='process_loop' and getattr(f,'__module__','').endswith('phase6_edge_runtime_v2')
    def run(t):
        if not target(t):return orig_run(t)
        return run_gate(p,lambda:orig_run(t))
    def start(t):
        chosen=target(t);result=orig_start(t)
        if chosen:
            wait(p/'worker_verified.json',35)
            assert len(pending)==1,('deferred READY callbacks',len(pending))
            pending.pop()();write(p/'ready_released.json',dict(raw_ns=raw()))
        return result
    threading.Thread.run=run;threading.Thread.start=start
