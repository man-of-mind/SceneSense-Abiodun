"""Create-only evidence commands and one ownership-checked smoke orchestration."""
import fcntl,importlib.util,json,os,shutil,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2];DEP=ROOT.parent/'abiodun_run4_single_gpu_profiler_v1'
sys.path.insert(0,str(BASE));import launch as L
from rl_agent.splitfusion_m4_two_ue_v1.binding import bind
spec=importlib.util.spec_from_file_location('prior_workflow',BASE.parent/'receiver_boundary_diagnostic_20261013_01/workflow.py')
W=importlib.util.module_from_spec(spec);spec.loader.exec_module(W)
def write(n,v):
 with (BASE/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
def run(name,cmd,cwd=DEP,check=True):
 start=time.time()
 with (BASE/(name+'.stdout')).open('x') as so,(BASE/(name+'.stderr')).open('x') as se:
  p=subprocess.run(cmd,cwd=cwd,stdout=so,stderr=se)
 write(name+'.json',dict(command=list(map(str,cmd)),cwd=str(cwd),returncode=p.returncode,start_wall_s=start,end_wall_s=time.time()))
 print(name,p.returncode,flush=True)
 if check:p.check_returncode()
 return p.returncode
def read(cmd):return subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=30).stdout
ENV=['env','-u','PYTHONPATH','-u','CUDA_VISIBLE_DEVICES','-u','LD_PRELOAD','PYTHONDONTWRITEBYTECODE=1',
     'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','python3','-B']
def offline():
 run('offline_preflight',ENV+[str(BASE/'launch.py'),'preflight'])
 # Unchanged slim-container entry proof is source-bound; use its existing
 # import-only startup gate against the SAME qualified baseline preparation.
 cmd=json.loads((BASE.parent/'receiver_boundary_diagnostic_20261013_01/edge_preflight.json').read_text())['command']
 # Match the actual retained slim CLI (no actors/services are constructed).
 run('edge_preflight',cmd)
def capture():
 L.validate();assert not L.CAP.exists()
 # Retained read-only baseline check preserves Firefox/MPS/UI; no GPU exclusivity.
 sys.argv=['work.py','before','HOST_BEFORE.json']
 bind(W.before,OUT=BASE,PREP=L.PREP.parent,ROOT=ROOT,write=write)()
 before=json.loads((BASE/'HOST_BEFORE.json').read_text())
 assert not before['running_containers'].strip(),'unexpected running container owner'
 lock=open('/tmp/splitfusion-live-host.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 started=[]
 try:
  ids=read(['sudo','-n','docker','compose','--project-directory',str(DEP/'OAI/oai-cn5g'),'ps','-aq','--all']).split()
  assert len(ids)==10,('expected retained ten core containers',ids)
  initial=json.loads(read(['sudo','-n','docker','inspect',*ids]))
  assert not any(c['State']['Running'] for c in initial)
  write('CORE_BEFORE.json',initial)
  run('core_start',['sudo','-n','docker','compose','start'],cwd=DEP/'OAI/oai-cn5g')
  started=json.loads(read(['sudo','-n','docker','inspect',*ids]));write('CORE_OWNERSHIP.json',started)
  deadline=time.monotonic()+60
  while True:
   needed=json.loads(read(['sudo','-n','docker','inspect','oai-amf','oai-smf','oai-upf','mysql']))
   if all(c['State']['Running'] and c['State'].get('Health',{}).get('Status','healthy')=='healthy' for c in needed):break
   assert time.monotonic()<deadline,'core health prerequisite failed'
   time.sleep(1)
  run('host_preflight',ENV+[str(BASE/'launch.py'),'preflight','--host'])
  free=shutil.disk_usage(ROOT).free;assert free>=16*1024**3
  binding=L.validate();write('IMMEDIATE_PRELAUNCH.json',dict(free_bytes=free,gate_bytes=16*1024**3,
       exclusive_experiment_lock=True,exclusive_gpu=False,binding_sha256=binding['sha256'],wall_s=time.time()))
  # The retained host owns the lock across radio startup, capture and teardown.
  fcntl.flock(lock,fcntl.LOCK_UN)
  run('smoke_launch',ENV+[str(BASE/'launch.py'),'run'])
 except BaseException as e:
  write('FIRST_FAILURE.json',dict(error=repr(e),wall_s=time.time(),live_capture_exists=L.CAP.exists()))
  raise
 finally:
  lock.close()
  if started:
   ids=[c['Id'] for c in started];current=json.loads(read(['sudo','-n','docker','inspect',*ids]))
   assert {(c['Id'],c['State']['StartedAt']) for c in current}=={(c['Id'],c['State']['StartedAt']) for c in started}
   run('core_stop',['sudo','-n','docker','stop','--time','20',*ids])
  sys.argv=['work.py','postflight','POSTFLIGHT.json']
  bind(W.postflight,OUT=BASE,CAP=L.CAP,ROOT=ROOT,write=write)()
if __name__=='__main__':
 {'offline':offline,'capture':capture}[sys.argv[1]]()
