"""Read-only identity/cgroup/policy inventory; setter restricted to verified compute TID."""
import json,os,subprocess as S,sys,time
from pathlib import Path
from affinity_lib import write,raw
from injection_guard_v2 import collect
def task(pid,tid):
    q=Path('/proc')/str(pid)/'task'/str(tid);st=(q/'status').read_text();fields=(q/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=int(pid),tid=int(tid),comm=(q/'comm').read_text().strip(),ns_tid=[int(v) for l in st.splitlines() if l.startswith('NSpid:') for v in l.split()[1:]],starttime=fields[19],cpu=int(fields[36]),state=fields[0],affinity=sorted(os.sched_getaffinity(int(tid))),policy=os.sched_getscheduler(int(tid)),priority=os.sched_getparam(int(tid)).sched_priority)
def tasks(pid):
    d={}
    for q in (Path('/proc')/str(pid)/'task').iterdir():
        try:d[q.name]=task(pid,q.name)
        except (OSError,ProcessLookupError):pass
    return d
def cgroups(pid):
    text=(Path('/proc')/str(pid)/'cgroup').read_text();path=text.strip().split(':',2)[-1];base=Path('/sys/fs/cgroup');q=base/path.lstrip('/');out=[]
    while True:
        row=dict(path=str(q))
        for f in ['cpuset.cpus','cpuset.cpus.effective','cpu.max','cpu.weight','cpu.stat']:
            try:row[f]=(q/f).read_text().strip()
            except OSError:row[f]=None
        out.append(row)
        if q==base:break
        assert base in q.parents;q=q.parent
    return dict(proc_cgroup=text,ancestors=out)
def locate(ready):
    rows=[]
    for q in Path('/proc').iterdir():
        if not q.name.isdigit():continue
        try:
            cmd=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if 'edge_profiled_v1_affinity_ab' not in cmd and 'synthetic_worker.py' not in cmd:continue
            ts=tasks(int(q.name))
            for t in ts.values():
                if t['ns_tid'][-1]==ready['tid'] and t['starttime']==ready['starttime']:
                    rows.append(dict(target=t,cmd=cmd,threads=ts,cgroups=cgroups(q.name),maps=(q/'maps').read_text()))
        except OSError:pass
    assert len(rows)==1,[(x['target']['pid'],x['target']['tid']) for x in rows];return rows[0]
def radio():
    rows=[]
    for q in Path('/proc').iterdir():
        if not q.name.isdigit():continue
        try:
            comm=(q/'comm').read_text().strip()
            if comm not in ['nr-softmodem','nr-uesoftmodem']:continue
            rows.append(dict(pid=int(q.name),comm=comm,cmd=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode(),threads=tasks(int(q.name)),cgroups=cgroups(q.name)))
        except OSError:pass
    return rows
def apply(p,arm):
    ready=json.loads((p/'worker_ready.json').read_text());before=locate(ready);t=before['target']
    evidence=collect(t['pid'],before.pop('maps'))
    write(p/'GUARD_EVIDENCE.json',evidence)
    write(p/'AFFINITY_BEFORE.json',dict(before=before,guard=evidence['verdict'],raw_ns=raw()))
    assert evidence['verdict']['passed'],evidence['verdict']['reasons']
    mask=set(t['affinity']);assert sorted(mask)==ready['affinity']
    desired=mask-{4,6} if arm=='treatment' else mask;assert desired,'empty treatment mask'
    # The effective affinity already reflects cpuset restrictions; retain cgroup ancestry to verify it.
    if arm=='treatment':os.sched_setaffinity(t['tid'],desired)
    after=tasks(t['pid']);assert after[str(t['tid'])]['affinity']==sorted(desired)
    for k,old in before['threads'].items():
        if k in after and int(k)!=t['tid']:assert after[k]['affinity']==old['affinity'],('other thread changed',k)
        if k in after:assert (after[k]['policy'],after[k]['priority'])==(old['policy'],old['priority'])
    doc=dict(arm=arm,raw_ns=raw(),before=before,after_threads=after,original_affinity=sorted(mask),expected_affinity=sorted(desired),radio=radio(),known_injection_checks_passed=True,passed=True)
    write(p/'AFFINITY_APPLIED.json',doc);return doc
def restore(p):
    d=json.loads((p/'AFFINITY_APPLIED.json').read_text());old=d['before']['target']
    try:
        now=task(old['pid'],old['tid']);assert now['starttime']==old['starttime'],'TID reuse'
        os.sched_setaffinity(old['tid'],d['original_affinity']);after=task(old['pid'],old['tid']);assert after['affinity']==d['original_affinity'];write(p/'host_restore.json',dict(passed=True,raw_ns=raw(),after=after))
    except (FileNotFoundError,ProcessLookupError):
        doc=json.loads((p/'worker_restored.json').read_text());assert doc['passed'];write(p/'host_restore.json',dict(passed=True,already_exited=True,raw_ns=raw()))
if __name__=='__main__':
    action,p,*args=sys.argv[1:];p=Path(p)
    if action=='apply':print(json.dumps(apply(p,args[0])['expected_affinity']))
    elif action=='restore':restore(p)
    elif action=='radio':write(p/'radio_snapshot.json',radio())
