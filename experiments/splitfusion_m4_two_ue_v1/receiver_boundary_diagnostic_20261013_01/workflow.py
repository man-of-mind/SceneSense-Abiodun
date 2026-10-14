"""Create-only command/evidence wrapper: M4 observational receiver boundaries + one supervised smoke (no retry)."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
DEP=ROOT.parent/'abiodun_run4_single_gpu_profiler_v1'
E='experiments/splitfusion_m4_two_ue_v1'
OLD_PREP=DEP/E/'preparation_smoke_20261013_10'
PREP=DEP/E/'preparation_smoke_20261013_11'
CAP=DEP/E/'smoke_live_20261013_08'
CHANGED={'rl_agent/splitfusion_m4_two_ue_v1/'+n for n in ('edge_entry.py','datapath.py','frontend.py')}
ADDED={'rl_agent/splitfusion_m4_two_ue_v1/receiver_timing.py'}
sys.path.insert(0,str(ROOT))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def write(name,value):
    with (OUT/name).open('x') as f: json.dump(value,f,indent=2,sort_keys=True)


def run():
    name,cwd=sys.argv[2:4]; cmd=sys.argv[4:]; begin=time.time()
    with (OUT/(name+'.stdout')).open('x') as so,(OUT/(name+'.stderr')).open('x') as se:
        p=subprocess.run(cmd,cwd=cwd,stdout=so,stderr=se)
    write(name+'.json',dict(command=cmd,cwd=cwd,returncode=p.returncode,start_wall_s=begin,end_wall_s=time.time()))
    print('COMMAND',name,'RC',p.returncode)
    print((OUT/(name+('.stderr' if p.returncode else '.stdout'))).read_text()[-2500:])
    raise SystemExit(p.returncode)


def deploy():
    old=json.loads((OLD_PREP/'PREPARATION.json').read_text())
    new=json.loads((OUT/'main_preparation/PREPARATION.json').read_text())
    assert new['config']==old['config']|{'repository':str(ROOT)}
    assert set(new['sources'])==set(old['sources'])|ADDED, set(new['sources'])^set(old['sources'])
    changed={n for n in old['sources'] if new['sources'][n]!=old['sources'][n]}
    assert changed==CHANGED,changed
    for name,digest in old['sources'].items(): assert sha(DEP/name)==digest,('deployment drift',name)
    for name,digest in new['sources'].items(): assert sha(ROOT/name)==digest,('main drift',name)
    for name in ADDED: assert not (DEP/name).exists(),name
    backup=OUT/'before'
    for name in sorted(CHANGED):
        assert (backup/Path(name).name).read_bytes()==(DEP/name).read_bytes(),('backup differs',name)
    write('DEPLOY_PLAN.json',dict(changed=sorted(CHANGED),added=sorted(ADDED),
        old={n:old['sources'][n] for n in CHANGED},new={n:new['sources'][n] for n in CHANGED|ADDED}))
    print('DEPLOY_PLAN_VERIFIED; apply the four exact files, then verify_deploy')


def verify_deploy():
    new=json.loads((OUT/'main_preparation/PREPARATION.json').read_text())
    for name,digest in new['sources'].items():
        assert sha(DEP/name)==sha(ROOT/name)==digest,name
    write('DEPLOY_RESULT.json',dict(status='PASS',verified=len(new['sources']),changed=sorted(CHANGED),added=sorted(ADDED)))
    print('DEPLOYED',sorted(CHANGED|ADDED))


def postflight():
    from rl_agent.splitfusion_m4_two_ue_v1 import host_baseline as B
    from rl_agent.splitfusion_m4_two_ue_v1.resources import check_owned_state
    baseline=B.inspect()
    def command(args):return subprocess.run(args,check=True,text=True,capture_output=True,timeout=20).stdout
    gpu=command(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'])
    B.validate_gpu_rows(gpu,set(baseline['allowed_gpu_pids']))
    owned=check_owned_state(CAP)
    live=[]
    for p in Path('/proc').glob('[0-9]*/comm'):
        try:
            name=p.read_text().strip()
            if name.startswith(('CarlaUnreal','nr-softmodem','nr-uesoftmodem')):live.append(dict(pid=int(p.parent.name),comm=name))
        except OSError:pass
    containers=command(['sudo','-n','docker','ps','--format','{{json .}}'])
    interfaces=json.loads(command(['ip','-j','link','show']))
    tunnels=[r['ifname'] for r in interfaces if r['ifname'].startswith('oaitun_ue')]
    clean=not (live or containers.strip() or tunnels or any(r['still_owned'] for r in owned))
    value=dict(clean=clean,owned=owned,live=live,running_containers=containers,ue_tunnels=tunnels,
        approved_baseline=baseline,gpu_clients=gpu,disk_free_bytes=shutil.disk_usage(ROOT).free)
    write(sys.argv[2] if len(sys.argv)>2 else 'POSTFLIGHT.json',value)
    print('HOST_CLEAN' if clean else 'HOST_NOT_CLEAN',value['disk_free_bytes'],containers.strip()[:500],live,tunnels)


def before():
    import fcntl
    from rl_agent.splitfusion_m4_two_ue_v1 import host_baseline as B
    from rl_agent.splitfusion_m4_two_ue_v1 import configuration as C
    C.validate(PREP/'PREPARATION.json')
    baseline=B.inspect()
    def read(args):return subprocess.run(args,check=True,text=True,capture_output=True,timeout=20).stdout
    gpu=read(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'])
    B.validate_gpu_rows(gpu,set(baseline['allowed_gpu_pids']))
    processes=[];live=[]
    for p in Path('/proc').glob('[0-9]*/comm'):
        try:
            comm=p.read_text().strip();cmd=(p.parent/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if comm.startswith(('CarlaUnreal','nr-softmodem','nr-uesoftmodem')):live.append(dict(pid=int(p.parent.name),comm=comm))
            if comm=='firefox' or (comm.startswith('python') and any(v in cmd.lower() for v in ('ui','http.server','streamlit'))):
                processes.append(dict(pid=int(p.parent.name),comm=comm,command=cmd))
        except OSError:pass
    assert not live,live
    free=shutil.disk_usage(ROOT).free;assert free>=C.DISK_GATE,free
    with open('/tmp/splitfusion-live-host.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    existing=read(['sudo','-n','docker','ps','-aq','--no-trunc']).split()
    core_ids=read(['sudo','-n','docker','compose','--project-directory',str(DEP/'OAI/oai-cn5g'),'ps','-aq']).split()
    core=json.loads(read(['sudo','-n','docker','inspect',*core_ids])) if core_ids else []
    value=dict(baseline=baseline,gpu=gpu,free_bytes=free,minimum_bytes=C.DISK_GATE,lock_free=True,
        live=live,preserved_processes=processes,core=[dict(id=c['Id'],name=c['Name'],running=c['State']['Running']) for c in core],
        existing_container_ids=existing,running_containers=read(['sudo','-n','docker','ps','--format','{{json .}}']),
        browser_exception='User explicitly requested leaving pre-existing Firefox; any timing effect is unmeasured.')
    write(sys.argv[2],value)
    print(json.dumps(value,indent=2))


def core_owned():
    ids=subprocess.run(['sudo','-n','docker','compose','ps','-aq'],cwd=DEP/'OAI/oai-cn5g',check=True,text=True,capture_output=True).stdout.split()
    initial=json.loads((OUT/'HOST_BEFORE.json').read_text())
    assert not initial['running_containers'].strip()
    assert not set(ids)&set(initial['existing_container_ids']), 'expected only task-created core containers'
    value=json.loads(subprocess.run(['sudo','-n','docker','inspect',*ids],check=True,text=True,capture_output=True).stdout)
    write('CORE_OWNERSHIP.json',[dict(id=c['Id'],name=c['Name'],started=c['State']['StartedAt']) for c in value])
    print('OWNED_CORE',len(value))


def core_stop():
    initial=json.loads((OUT/'HOST_BEFORE.json').read_text())
    assert not initial['running_containers'].strip()
    owned=json.loads((OUT/'CORE_OWNERSHIP.json').read_text())
    ids=subprocess.run(['sudo','-n','docker','compose','ps','-aq'],cwd=DEP/'OAI/oai-cn5g',check=True,text=True,capture_output=True).stdout.split()
    current=json.loads(subprocess.run(['sudo','-n','docker','inspect',*ids],check=True,text=True,capture_output=True).stdout)
    assert {c['Id'] for c in current}=={c['id'] for c in owned}
    assert {(c['Id'],c['State']['StartedAt']) for c in current}=={(c['id'],c['started']) for c in owned}
    command=['sudo','-n','docker','compose','stop','--timeout','20']
    begin=time.time();p=subprocess.run(command,cwd=DEP/'OAI/oai-cn5g',capture_output=True,text=True)
    write('core_stop.json',dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
        start_wall_s=begin,end_wall_s=time.time(),ownership_checked=True))
    print(p.returncode,p.stdout,p.stderr);raise SystemExit(p.returncode)


def finalize():
    v=json.loads((PREP/'PREPARATION.json').read_text())
    assert all(sha(ROOT/n)==d and sha(DEP/n)==d for n,d in v['sources'].items())
    write('FINAL_SHA256.json',dict(sources=v['sources'],
        audit={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()},
        capture={str(p.relative_to(CAP)):sha(p) for p in CAP.rglob('*') if p.is_file() and os.access(p,os.R_OK)},
        capture_unreadable_not_hashed=sorted(str(p.relative_to(CAP)) for p in CAP.rglob('*') if p.is_file() and not os.access(p,os.R_OK)),
        preparation={str(p.relative_to(PREP)):sha(p) for p in PREP.rglob('*') if p.is_file()}))
    print('FINAL_SOURCES_AND_EVIDENCE_VERIFIED')


if __name__=='__main__':
    dict(run=run,deploy=deploy,verify_deploy=verify_deploy,before=before,core_owned=core_owned,core_stop=core_stop,postflight=postflight,finalize=finalize)[sys.argv[1]]()
