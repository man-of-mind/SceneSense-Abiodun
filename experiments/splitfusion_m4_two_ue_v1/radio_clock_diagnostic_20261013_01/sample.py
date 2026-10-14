"""External 500-ms /proc + bounded RFsim mmap sampler. Root only for owned OAI.
No live launch here. A stop marker or 240-s bound ends sampling and owned perf.
"""
import ctypes as C,json,os,signal,subprocess,sys,time
from pathlib import Path
U=C.c_uint64
class Counter(C.Structure):_fields_=[(n,U) for n in ('count','wall','cpu','max_wall','active_wall')]+[('hist',U*10)]
class Peer(C.Structure):_fields_=[('fd',C.c_int64)]+[(n,U) for n in ('local_port','remote_port','received_ts','required_ts')]+[(n,C.c_char*k) for n,k in [('model',64),('local_ip',48),('remote_ip',48)]]
class Data(C.Structure):_fields_=[(n,U) for n in ('seq','magic','pid','tid','ru','role','sample_rate','mono_origin','wall_origin','rx_ts','rx_calls','overflow')]+[('peers',Peer*4),('counters',Counter*32)]
def serial(v):
 if isinstance(v,bytes):return v.decode(errors='replace')
 if isinstance(v,C.Array):return [serial(x) for x in v]
 if isinstance(v,C.Structure):return {n:serial(getattr(v,n)) for n,*_ in v._fields_}
 return v
def snapshot(path):
 # Re-open/pread avoids copying a mmap while its writer updates the sequence.
 with path.open('rb',buffering=0) as f:
  for _ in range(100):
   f.seek(0);a=int.from_bytes(f.read(8),'little')
   if a%2:continue
   f.seek(0);raw=f.read(C.sizeof(Data));f.seek(0);b=int.from_bytes(f.read(8),'little')
   if len(raw)==C.sizeof(Data) and a==b==int.from_bytes(raw[:8],'little'):
    d=Data.from_buffer_copy(raw)
    if d.magic==0:return None # Newly truncated file, not yet published.
    assert d.magic==0x4d34524644494131;return serial(d)
 return None
def stat(path):
 text=path.read_text();a=text.rsplit(')',1)[1].split()
 return dict(comm=text.split('(',1)[1].rsplit(')',1)[0],state=a[0],ppid=int(a[1]),pgrp=int(a[2]),
  minor_faults=int(a[7]),major_faults=int(a[9]),utime=int(a[11]),stime=int(a[12]),priority=int(a[15]),nice=int(a[16]),
  start_ticks=int(a[19]),cpu=int(a[36]),rt_priority=int(a[37]),policy=int(a[38]))
def read(path):
 try:return path.read_text().strip()
 except OSError:return None
def cgroups(pid):
 text=read(Path(f'/proc/{pid}/cgroup'));out=dict(membership=text,ancestors={})
 for line in (text or '').splitlines():
  if line.startswith('0::'):
   p=Path('/sys/fs/cgroup')/line.split('::',1)[1].lstrip('/')
   while str(p).startswith('/sys/fs/cgroup'):
    out['ancestors'][str(p)]={n:read(p/n) for n in ('cpu.max','cpu.stat','cpu.weight','cpuset.cpus.effective','cpu.pressure')}
    if p==Path('/sys/fs/cgroup'):break
    p=p.parent
 return out
def owned_radios(cap):
 owners={}
 for role in ('gnb','ue'):
  try:
   o=json.loads((cap/(role+'.ownership.json')).read_text());s=stat(Path(f'/proc/{o["pid"]}/stat'))
   if str(s['start_ticks'])==o['birth'] and s['pgrp']==o['pgid']:owners[role]=o
  except (OSError,ValueError):pass
 found={}
 for p in Path('/proc').glob('[0-9]*'):
  try:
   comm=read(p/'comm')
   if comm not in ('nr-softmodem','nr-uesoftmodem'):continue
   s=stat(p/'stat');role='gnb' if comm=='nr-softmodem' else 'ue'
   if role in owners and s['pgrp']==owners[role]['pgid']:found[int(p.name)]=dict(role=role,**s)
  except (OSError,ValueError):pass
 return found
def main():
 cap,out=map(Path,sys.argv[1:3]);os.sched_setaffinity(0,{22});os.nice(19)
 start=time.monotonic();cpu_start=time.process_time();proc={};perf=None;perf_log=None;samples=0;misses=0;cost=[];identities={}
 stopped=False
 def stop(*_):
  nonlocal stopped;stopped=True
 signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
 with (out/'THREADS.jsonl').open('x') as tf,(out/'NATIVE.jsonl').open('x') as nf,(out/'SAMPLER_TICKS.jsonl').open('x') as sf:
  tick=time.monotonic()
  try:
   while not stopped and not (out/'STOP').exists() and time.monotonic()-start<240:
    w0=time.monotonic_ns();c0=time.process_time_ns();wall=time.time_ns();raw=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    discovered=owned_radios(cap);proc.update(discovered)
    for pid,info in discovered.items():
     if str(pid) not in identities:
      identities[str(pid)]=dict(info,exe=os.readlink(f'/proc/{pid}/exe'),cgroups=cgroups(pid),
         cmdline=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(),
         maps=read(Path(f'/proc/{pid}/maps')))
     for td in Path(f'/proc/{pid}/task').glob('[0-9]*'):
      try:
       row=dict(pid=pid,tid=int(td.name),role=info['role'],mono_ns=time.monotonic_ns(),wall_ns=wall,raw_ns=raw,**stat(td/'stat'))
       status={k:v.strip() for k,v in (l.split(':',1) for l in (td/'status').read_text().splitlines() if ':' in l)}
       row.update({k:status.get(k) for k in ('Cpus_allowed_list','voluntary_ctxt_switches','nonvoluntary_ctxt_switches')})
       row.update(schedstat=read(td/'schedstat'),wchan=read(td/'wchan'))
       tf.write(json.dumps(row,separators=(',',':'))+'\n')
      except (OSError,ValueError):pass
    for p in sorted((out/'maps').glob('*.bin')):
     d=snapshot(p)
     if d is None:misses+=1;continue
     if str(d['pid']) in identities and 'instrumented_maps' not in identities[str(d['pid'])]:
      identities[str(d['pid'])]['instrumented_maps']=read(Path(f'/proc/{d["pid"]}/maps'))
     nf.write(json.dumps(dict(mono_ns=time.monotonic_ns(),wall_ns=time.time_ns(),path=p.name,**d),separators=(',',':'))+'\n')
    if perf is None and len(discovered)==2:
     cmd=['perf','record','--clockid','mono','-e','cpu-clock:u','-F','49','--call-graph','dwarf,4096',
          '-m','64','-p',','.join(map(str,sorted(discovered))),'-o',str(out/'perf.data'),'--','sleep','220']
     perf_log=(out/'perf.log').open('x');perf=subprocess.Popen(cmd,stdout=perf_log,stderr=subprocess.STDOUT)
     identities['perf']=dict(pid=perf.pid,command=cmd,start_mono_ns=time.monotonic_ns())
    elapsed=time.monotonic_ns()-w0;cpu=time.process_time_ns()-c0;cost.append((elapsed,cpu));samples+=1
    sf.write(json.dumps(dict(mono_ns=w0,wall_ns=wall,raw_ns=raw,elapsed_ns=elapsed,cpu_ns=cpu,
                            cgroups={str(pid):cgroups(pid) for pid in discovered}))+'\n')
    tf.flush();nf.flush();sf.flush();tick+=.5
    time.sleep(max(0,tick-time.monotonic()))
  finally:
   if perf is not None:
    if perf.poll() is None:perf.send_signal(signal.SIGINT)
    try:perf.wait(timeout=10)
    except subprocess.TimeoutExpired:perf.terminate();perf.wait(timeout=5)
    identities['perf']['returncode']=perf.returncode;perf_log.close()
   result=dict(identities=identities,samples=samples,unstable_native_snapshots=misses,elapsed_s=time.monotonic()-start,
    cpu_s=time.process_time()-cpu_start,sampling_interval_s=.5,cpu_affinity=sorted(os.sched_getaffinity(0)),
    nice=os.nice(0),sched_schedstats=read(Path('/proc/sys/kernel/sched_schedstats')),
    proc_tick_hz=os.sysconf('SC_CLK_TCK'),sample_wall_ns_sum=sum(w for w,c in cost),sample_cpu_ns_sum=sum(c for w,c in cost),
    native_record_bytes=C.sizeof(Data),max_duration_s=240,stopped_by_marker=(out/'STOP').exists())
   with (out/'SAMPLER_RESULT.json').open('x') as f:json.dump(result,f,indent=2)
if __name__=='__main__':main()
