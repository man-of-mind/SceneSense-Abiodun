"""Create-only command/evidence wrapper: M4 lifecycle repair + one supervised smoke (no retry)."""
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
OLD_PREP=DEP/E/'preparation_smoke_20261013_08'
PREP=DEP/E/'preparation_smoke_20261013_09'
CAP=DEP/E/'smoke_live_20261013_06'
CHANGED={'rl_agent/splitfusion_m4_two_ue_v1/'+n for n in ('host.py','cli.py','world.py','composed_checks.py')}
ADDED={'rl_agent/splitfusion_m4_two_ue_v1/supervisor.py'}
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
    backup=OUT/'before'; backup.mkdir()
    for name in sorted(CHANGED):
        with (backup/Path(name).name).open('xb') as f: f.write((DEP/name).read_bytes())
    write('DEPLOY_PLAN.json',dict(changed=sorted(CHANGED),added=sorted(ADDED),
        old={n:old['sources'][n] for n in CHANGED},new={n:new['sources'][n] for n in CHANGED|ADDED}))
    for name in sorted(CHANGED|ADDED):
        shutil.copyfile(ROOT/name,DEP/name)
        assert sha(DEP/name)==new['sources'][name]
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
    dict(run=run,deploy=deploy,postflight=postflight,finalize=finalize)[sys.argv[1]]()
