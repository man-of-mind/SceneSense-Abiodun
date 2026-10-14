"""Focused record-only probes, old radio-clock audit and private binding checks."""
import ctypes,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 os.sched_setaffinity(0,{22});os.nice(15)
 out=BASE/(sys.argv[1] if len(sys.argv)>1 else 'checks_01');out.mkdir();maps=out/'maps';maps.mkdir()
 compile=['g++','-std=c++11','-O2','-o',str(out/'test_native'),str(BASE/'native/test_native.cpp')]
 subprocess.run(compile,check=True,capture_output=True)
 run=subprocess.run([str(out/'test_native'),str(maps)],capture_output=True,text=True)
 with (out/'NATIVE_PROCESS.json').open('x') as f:json.dump(dict(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr),f)
 run.check_returncode()
 test=json.loads(run.stdout);sample=load(BASE/'sample.py','external_sampler')
 native=sample.snapshot(next(maps.glob('*.bin')))
 assert test['struct_bytes']==ctypes.sizeof(sample.Data)==4736
 assert native['counters'][8]['count']==1 and native['counters'][7]['count']==100000
 assert native['rx_ts']==1000 and native['peers'][0]['required_ts']==150
 # Match actual emitter layout, TCP peer identities, clocks and histogram bins.
 test['reader']='PASS';test['tcp_peer_ports']=[native['peers'][0]['local_port'],native['peers'][0]['remote_port']]
 old=load(BASE.parent/'uplink_service_audit_20261013_01/analyze.py','prior_readers')
 ticks=old.ticks(old.CAP/'native_radio/NRUE_MAC_RLC_BUFFER_STATUS_live.csv')
 result={}
 for ue in (0,1):
  seq=[x for x in ticks if x['ue_id']==ue]
  diffs=[b['radio_slot']-a['radio_slot'] for a,b in zip(seq,seq[1:])]
  wall=[b['wall_ns']-a['wall_ns'] for a,b in zip(seq,seq[1:])]
  assert min(diffs)>=0 and min(wall)>=0
  old_summary=json.loads((BASE.parent/'uplink_service_audit_20261013_01/SUMMARY.json').read_text())
  v=old_summary['per_ue']['m4-ue'+str(ue)]['resources'];a=old.prior_tick(seq,v['clock_start_wall_ns']);b=old.prior_tick(seq,v['clock_end_wall_ns'])
  ratio=(b['radio_slot']-a['radio_slot'])*.0005/((b['wall_ns']-a['wall_ns'])/1e9)
  assert abs(ratio-v['radio_per_wall'])<1e-12
  result[str(ue)]=dict(ticks=len(seq),min_slot_step=min(diffs),max_slot_step=max(diffs),min_wall_step_ns=min(wall),
     max_wall_step_ns=max(wall),radio_per_wall=ratio,backward_clock_steps=0,
     unwrapping='20 slots per 10-ms NR frame; SFN wrap adds 20480 slots, not wall time.')
 # Direct mathematical wrap/control case, same retained unwrap rule.
 previous=None;offset=0;unwrapped=[]
 for n in (20478,20479,0,1):
  if previous is not None and n<previous-10240:offset+=20480
  unwrapped.append(n+offset);previous=n
 assert unwrapped==[20478,20479,20480,20481]
 launcher=load(BASE/'launch.py','diagnostic_launcher')
 from rl_agent.splitfusion_m4_two_ue_v1 import host,radio_runtime as R
 from rl_agent.splitfusion_m4_two_ue_v1.binding import construct
 original=R.argv(launcher.PREP.parent);instrumented=launcher.diagnostic_argv(launcher.PREP.parent)
 for role,a in instrumented.items():
  normalized=[x for x in a[:-2] if not x.startswith('M4_RFSIM_DIAGNOSTIC_DIR=')]
  assert normalized==original[role] and a[-2]=='--loader.rfsimulator.shlibpath'
 assert R.argv.__module__=='rl_agent.splitfusion_m4_two_ue_v1.radio_runtime'
 assert launcher.DiagnosticRadio.start.__code__ is R.AttachedRadio.start.__code__
 modified=construct(host.run,[('from .radio_runtime import AttachedRadio, GoodAll',
   'from .radio_runtime import GoodAll\n    AttachedRadio = diagnostic_radio')],diagnostic_radio=launcher.DiagnosticRadio)
 value=dict(status='PASS',native=test,old_clock=result,synthetic_wrap=unwrapped,private_binding=modified.__m4_binding__,
  unchanged_radio_argv_except_module_path_and_diagnostic_directory=True,source_modules_unmodified=True,
  compile_command=compile,native_snapshot=native)
 with (out/'RESULT.json').open('x') as f:json.dump(value,f,indent=2)
 print(json.dumps({k:v for k,v in value.items() if k not in ('native_snapshot','compile_command','private_binding')}))
if __name__=='__main__':main()
