"""One record-only diagnostic binding around existing M4 host/supervisor.
No production modules are edited or mutated. Existing function binding helpers
substitute only AttachedRadio and its argv dependency in private namespaces.
"""
import hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2];DEP=ROOT.parent/'abiodun_run4_single_gpu_profiler_v1'
PREP=DEP/'experiments/splitfusion_m4_two_ue_v1/preparation_smoke_20261013_11/PREPARATION.json'
CAP=DEP/'experiments/splitfusion_m4_two_ue_v1/smoke_live_20261013_09'
OBS=BASE/'observer';TOKEN='M4_TWO_UE_SMOKE_SEPARATELY_AUTHORIZED'
sys.path.insert(0,str(DEP))
from rl_agent.splitfusion_m4_two_ue_v1 import configuration as C,radio_runtime as R
from rl_agent.splitfusion_m4_two_ue_v1.binding import bind,construct
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,v):
 with (BASE/name).open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
def diagnostic_argv(preparation):
 commands=R.argv(preparation)
 for role,a in commands.items():
  # Existing loader supports a MODULE-specific path. All other libraries,
  # binaries, RF parameters and environmental policy remain unchanged.
  exe=next(i for i,x in enumerate(a) if x.endswith('/nr-softmodem') or x.endswith('/nr-uesoftmodem'))
  a.insert(exe,'M4_RFSIM_DIAGNOSTIC_DIR='+str(OBS/'maps'))
  a.extend(['--loader.rfsimulator.shlibpath',str(BASE/'native')])
 return commands
def validate():
 v=json.loads((BASE/'BINDING.json').read_text())
 assert v['sha256']==C.canonical({k:x for k,x in v.items() if k!='sha256'})
 assert v['preparation_sha256']==C.validate(PREP)['sha256']
 assert v['capture']==str(CAP) and v['sampler_cpu']==22 and v['smoke_sim_seconds']==20
 for p,h in v['sources'].items():assert sha(p)==h,('diagnostic source drift',p)
 assert v['commands']==diagnostic_argv(PREP.parent)
 build=json.loads((BASE/'BUILD.json').read_text())
 for p,h in build['original_unchanged'].items():assert sha(p)==h,('qualified source/binary drift',p)
 return v
class DiagnosticRadio(R.AttachedRadio):
 def __init__(self,**kw):
  super().__init__(**kw)
  C.write_new(self.output/'DIAGNOSTIC_BINDING_REFERENCE.json',dict(path=str(BASE/'BINDING.json'),sha256=validate()['sha256']))
 start=bind(R.AttachedRadio.start,argv=diagnostic_argv)
def child():
 validate()
 from rl_agent.splitfusion_m4_two_ue_v1 import host
 diagnostic_run=construct(host.run,[('from .radio_runtime import AttachedRadio, GoodAll',
  'from .radio_runtime import GoodAll\n    AttachedRadio = diagnostic_radio')],diagnostic_radio=DiagnosticRadio)
 return diagnostic_run(PREP,CAP,smoke=True,carla_stop_delegated=True)
def prepare():
 assert not CAP.exists();old=C.validate(PREP)
 files=[BASE/n for n in ('launch.py','sample.py','build.py','NATIVE_PATCH.json','BUILD.json',
  'native/diagnostic.h','native/simulator.cpp','native/librfsimulator.so')]
 v=dict(preparation_sha256=old['sha256'],preparation=str(PREP),capture=str(CAP),sampler_cpu=22,
  smoke_sim_seconds=20,sources={str(p):sha(p) for p in files},commands=diagnostic_argv(PREP.parent),
  scope='Record-only instrumented RFsim library and external sampler; all baseline preparation sources unchanged')
 v['sha256']=C.canonical(v);write('BINDING.json',v);print(v['sha256'])
def run():
 validate();assert not CAP.exists()
 from rl_agent.splitfusion_m4_two_ue_v1 import supervisor as S
 from rl_agent.splitfusion_m4_two_ue_v1.resources import OwnedProcesses
 OBS.mkdir();(OBS/'maps').mkdir();owned=OwnedProcesses(OBS)
 sampler=owned.start('thread_sampler',['sudo','-n','env','PYTHONDONTWRITEBYTECODE=1','/usr/bin/python3','-B',
   str(BASE/'sample.py'),str(CAP),str(OBS)],cwd=DEP,cpus=[22])
 started=time.time();rc=2
 try:
  time.sleep(.6);assert sampler.poll() is None,'external sampler failed before live invocation'
  rc=S.run(PREP,CAP,smoke=True,token=TOKEN,child_argv=[sys.executable,'-B',str(BASE/'launch.py'),'child'])
  return rc
 finally:
  (OBS/'STOP').touch(exist_ok=False)
  try:sampler.wait(timeout=15)
  except subprocess.TimeoutExpired:pass
  cleanup=owned.close();write('LAUNCH.json',dict(returncode=rc,started_wall_s=started,ended_wall_s=time.time(),sampler_cleanup=cleanup))
def preflight():
 v=validate()
 from rl_agent.splitfusion_m4_two_ue_v1.preflight import report
 result=report(PREP,host='--host' in sys.argv)
 result.update(diagnostic_binding_sha256=v['sha256'],commands=diagnostic_argv(PREP.parent),
               production_source_changes=False,private_binding_helpers=True)
 print(json.dumps(result,sort_keys=True))
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='prepare':prepare()
 elif mode=='preflight':preflight()
 elif mode=='child':raise SystemExit(child())
 elif mode=='run':raise SystemExit(run())
 else:raise ValueError(mode)
