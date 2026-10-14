"""Compile only the isolated RFsim translation unit; never invoke shared ninja."""
import hashlib,json,os,re,shlex,subprocess,time
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
OAI=ROOT/'OAI/openairinterface5g';BUILD=OAI/'cmake_targets/ran_build/build';OUT=BASE/'native'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 os.sched_setaffinity(0,{22});os.nice(15);start=time.monotonic()
 original=OAI/'radio/rfsimulator/simulator.cpp';text=original.read_text()
 for old,new in json.loads((BASE/'NATIVE_PATCH.json').read_text()):
  assert text.count(old)==1,old;text=text.replace(old,new)
 assert text.rstrip()==(OUT/'simulator.cpp').read_text().rstrip(),'instrumentation patch drift'
 ninja=(BUILD/'build.ninja').read_text()
 block=re.search(r'build radio/rfsimulator/CMakeFiles/rfsimulator.dir/simulator.cpp.o:.*?(?=\n\n)',ninja,re.S).group()
 opts={k:shlex.split(re.search(r'^  '+k+r' = (.*)$',block,re.M).group(1)) for k in ('DEFINES','FLAGS','INCLUDES')}
 # Preserve the original local include resolution after moving this source.
 compile=['g++',*opts['DEFINES'],*opts['FLAGS'],*opts['INCLUDES'],'-I'+str(original.parent),'-c',str(OUT/'simulator.cpp'),'-o',str(OUT/'simulator.o')]
 other=[BUILD/'radio/rfsimulator/CMakeFiles/rfsimulator.dir/apply_channelmod.c.o',
        BUILD/'radio/rfsimulator/CMakeFiles/rfsimulator.dir/__/__/openair1/PHY/TOOLS/signal_energy.c.o',BUILD/'libSIMU.a']
 baseline=[BUILD/'librfsimulator.so',BUILD/'nr-softmodem',BUILD/'nr-uesoftmodem',original,*other]
 before={str(p):sha(p) for p in baseline}
 commands=[compile,['g++','-shared','-o',str(OUT/'librfsimulator.so'),str(OUT/'simulator.o'),*map(str,other)]]
 for i,cmd in enumerate(commands):
  with (BASE/f'build_{i}.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 assert all(sha(p)==h for p,h in before.items())
 result=dict(original_unchanged=before,instrumented_library_sha256=sha(OUT/'librfsimulator.so'),
  instrumented_source_sha256=sha(OUT/'simulator.cpp'),header_sha256=sha(OUT/'diagnostic.h'),commands=commands,elapsed_s=time.monotonic()-start)
 with (BASE/'BUILD.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k not in ('commands','original_unchanged')}))
if __name__=='__main__':main()
