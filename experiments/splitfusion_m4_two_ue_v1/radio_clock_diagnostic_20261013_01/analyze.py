"""Saved-only CPU/thread/peer audit. Reuses retained radio and clock readers."""
import collections as C,csv,hashlib,importlib.util,json,math,os,re,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2];DEP=ROOT.parent/'abiodun_run4_single_gpu_profiler_v1'
CAP=DEP/'experiments/splitfusion_m4_two_ue_v1/smoke_live_20261013_09';OBS=BASE/'observer'
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else BASE/'analysis_01'
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def module(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
A=module(BASE.parent/'uplink_service_audit_20261013_01/analyze.py','retained_uplink');H=A.H;R=A.R
os.environ['MPLCONFIGDIR']=str(OUT/'mpl_cache')
import numpy as np
doc,lines,q,rows=H.doc,H.lines,H.q,H.rows
STAGES={0:'READ inclusive',2:'FLUSH inclusive',3:'EPOLL syscall',4:'AVAILABLE processing inclusive',5:'WRITE inclusive',6:'SOCKET_WRITE inclusive',
        8:'BLOCKING_PEER_0',9:'BLOCKING_PEER_1',10:'BLOCKING_PEER_2',11:'BLOCKING_PEER_3',16:'CHANNEL_PEER_0',17:'CHANNEL_PEER_1',
        18:'CHANNEL_PEER_2',19:'CHANNEL_PEER_3',24:'MERGE_PEER_0',25:'MERGE_PEER_1',26:'MERGE_PEER_2',27:'MERGE_PEER_3'}
LIMITS=[.1,.25,.5,1,2,5,10,25,100,None]
def write(n,v):
 with (OUT/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False)
def table(n,rr):
 keys=list(dict.fromkeys(k for r in rr for k in r))
 with (OUT/n).open('x') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rr)
def quantile_bin(hist,p):
 acc=0;target=sum(hist)*p
 for i,n in enumerate(hist):
  acc+=n
  if acc>=target:return [0 if i==0 else LIMITS[i-1],LIMITS[i]]
def native_delta(a,b):
 result={};duration=(b['mono_ns']-a['mono_ns'])/1e9
 for k,(aa,bb) in enumerate(zip(a['counters'],b['counters'])):
  if not bb['count']-aa['count']:continue
  vals={n:bb[n]-aa[n] for n in ('count','wall','cpu')};assert all(x>=0 for x in vals.values())
  hist=[y-x for x,y in zip(aa['hist'],bb['hist'])];assert min(hist)>=0 and sum(hist)==vals['count']
  result[k]=dict(stage=STAGES.get(k,str(k)),count=vals['count'],wall_s=vals['wall']/1e9,cpu_s=vals['cpu']/1e9,
    wall_fraction=vals['wall']/1e9/duration,cpu_fraction=vals['cpu']/1e9/duration,
    p95_wall_ms_bin=quantile_bin(hist,.95),p99_wall_ms_bin=quantile_bin(hist,.99),
    max_wall_ms_since_process_start=bb['max_wall']/1e6,hist=hist,
    active_at_start_ns=aa['active_wall'],active_at_end_ns=bb['active_wall'])
 return result
def main():
 start=time.monotonic();os.sched_setaffinity(0,{22});os.nice(19);assert not H.live_busy();OUT.mkdir()
 H.bound(Path(__file__));attached=doc(CAP/'ATTACHED.json')['sources'];audit=lines(CAP/'edge/RECEIVER_TIMING.jsonl')
 sizes={s['stream_id']:doc(CAP/s['stream_id']/'SENT_FRAME_SIZES.json') for s in attached}
 clock=R.Clock(audit,[r['pair'] for ss in sizes.values() for r in ss]);frames=[]
 completed={};datagrams={}
 for r in audit:
  if r['kind']=='datagram':datagrams[r['stream_id'],r['recv_seq']]=r
  if r['kind']=='assembly' and r['completed']:completed[r['stream_id'],r['frame_id']]=r
 for s in attached:
  stream=s['stream_id'];e=doc(CAP/stream/'run4_phase6/PHASE6_UE_EVIDENCE.json');z={r['frame_id']:r for r in sizes[stream]}
  for d in e['decisions']:
   f=d['frame_id'];st=d['stages'];v=z[f];assert v['source']==s
   comp=completed.get((stream,f));dg=datagrams.get((stream,comp['recv_seq'])) if comp else None
   last=clock.to_wall(st['last_packet_send_raw_ns'])
   frames.append(dict(stream_id=stream,source_session_id=s['session_id'],frame_id=f,kind=d['kind'],mode_id=d['mode_id'],q_e4=d['q_e4'],
    capture_wall_ns=v['capture_wall_ns'],first_send_wall_ns=clock.to_wall(st['first_packet_send_raw_ns']),last_send_wall_ns=last,
    payload_bytes=v['payload_bytes'],chunk_count=v['chunk_count'],udp_bytes=v['payload_bytes']+8*v['chunk_count'],
    completing_kernel_wall_ns=dg['kernel_wall_ns'] if dg else None,last_send_to_kernel_ms=(dg['kernel_wall_ns']-last)/1e6 if dg else None))
 lo=min(f['first_send_wall_ns'] for f in frames);hi=max(f['last_send_wall_ns'] for f in frames)
 ticks=lines(OBS/'SAMPLER_TICKS.jsonl');offsets=[x['wall_ns']-x['mono_ns'] for x in ticks]
 offset=sorted(offsets)[len(offsets)//2];mlo=lo-offset;mhi=hi-offset
 native=lines(OBS/'NATIVE.jsonl');thread=lines(OBS/'THREADS.jsonl');ng=C.defaultdict(list);tg=C.defaultdict(list)
 for r in native:
  assert not r['overflow'] and not r['seq']%2
  ng[r['pid'],r['tid'],r['ru']].append(r)
 for r in thread:tg[r['pid'],r['tid']].append(r)
 tids={k:rr[-1]['comm'] for k,rr in tg.items()}
 sr=doc(OBS/'SAMPLER_RESULT.json');hz=sr['proc_tick_hz'];thread_table=[];thread_bins=[]
 for key,allr in tg.items():
  rr=[r for r in allr if lo<=r['wall_ns']<=hi]
  if len(rr)<2:continue
  a,b=rr[0],rr[-1];dt=(b['mono_ns']-a['mono_ns'])/1e9
  cpu=(b['utime']+b['stime']-a['utime']-a['stime'])/hz
  ss0=list(map(int,a['schedstat'].split()));ss1=list(map(int,b['schedstat'].split()))
  sched=(ss1[1]-ss0[1])/1e9;assert sched>=0
  r=dict(pid=key[0],tid=key[1],role=a['role'],comm=b['comm'],samples=len(rr),start_mono_ns=a['mono_ns'],end_mono_ns=b['mono_ns'],
   duration_s=dt,user_cpu_s=(b['utime']-a['utime'])/hz,system_cpu_s=(b['stime']-a['stime'])/hz,cpu_s=cpu,cpu_fraction=cpu/dt,
   sched_runtime_s=(ss1[0]-ss0[0])/1e9,runqueue_delay_s=sched,runqueue_fraction=sched/dt,sched_timeslices=ss1[2]-ss0[2],
   voluntary_switches=int(b['voluntary_ctxt_switches'])-int(a['voluntary_ctxt_switches']),
   involuntary_switches=int(b['nonvoluntary_ctxt_switches'])-int(a['nonvoluntary_ctxt_switches']),
   minor_faults=b['minor_faults']-a['minor_faults'],major_faults=b['major_faults']-a['major_faults'],
   allowed_cpus=b['Cpus_allowed_list'],observed_cpus=sorted(set(x['cpu'] for x in rr)),policy=b['policy'],rt_priority=b['rt_priority'],
   sampled_states=dict(C.Counter(x['state'] for x in rr)),sampled_wchan=dict(C.Counter(x['wchan'] for x in rr)))
  thread_table.append(r)
  for x,y in zip(rr,rr[1:]):
   duration=(y['mono_ns']-x['mono_ns'])/1e9
   thread_bins.append(dict(pid=key[0],tid=key[1],comm=b['comm'],seconds=(x['wall_ns']-lo)/1e9,
    duration_s=duration,cpu_fraction=((y['utime']+y['stime'])-(x['utime']+x['stime']))/hz/duration))
 native_table=[];peer_table=[];native_summary=[];native_bins=[]
 for key,allr in ng.items():
  rr=[r for r in allr if lo<=r['wall_ns']<=hi]
  if len(rr)<2:continue
  a,b=rr[0],rr[-1];dt=(b['mono_ns']-a['mono_ns'])/1e9;stages=native_delta(a,b)
  record=dict(pid=key[0],tid=key[1],ru=key[2],comm=tids.get(key[:2],'exited startup thread'),role=b['role'],
   duration_s=dt,start_mono_ns=a['mono_ns'],end_mono_ns=b['mono_ns'],sample_rate=b['sample_rate'],
   radio_elapsed_s=(b['rx_ts']-a['rx_ts'])/b['sample_rate'],radio_per_wall=(b['rx_ts']-a['rx_ts'])/b['sample_rate']/dt,
   samples=len(rr),stages=stages,peers=b['peers'])
  native_summary.append(record)
  for stage,v in stages.items():native_table.append(dict(pid=key[0],tid=key[1],ru=key[2],comm=record['comm'],duration_s=dt,
    stage_id=stage,radio_per_wall=record['radio_per_wall'],**v))
  if 0 not in stages:continue
  for i,p in enumerate(b['peers']):
   if not p['local_port']:continue
   matches=[]
   for otherkey,otherr in ng.items():
    ob=otherr[-1]
    if otherkey[0]==key[0]:continue
    for op in ob['peers']:
     if op['local_port']==p['remote_port'] and op['remote_port']==p['local_port']:
      matches.append((ob['pid'],ob['tid'],ob['ru']))
   assert len(set((pid,ru) for pid,tid,ru in matches))==1,('ambiguous peer',key,i,matches)
   other_pid,other_ru=next(iter(set((pid,ru) for pid,tid,ru in matches)))
   ue_id=other_ru if b['role']==1 else b['ru'];source=next(s for s in attached if s['ue_id']==ue_id)
   wait=stages.get(8+i,{});channel=stages.get(16+i,{})
   peer_table.append(dict(pid=key[0],tid=key[1],ru=key[2],peer_slot=i,role='gnb' if b['role']==1 else 'ue',
    direction='UL samples UE → gNB' if b['role']==1 else 'DL samples gNB → UE',stream_id=source['stream_id'],rnti=source['rnti'],
    local_port=p['local_port'],peer_port=p['remote_port'],model=p['model'],other_pid=other_pid,other_ru=other_ru,
    ip_note='UE endpoints are IPv4 loopback; gNB log identifies IPv4-mapped IPv6. Native IPv4-only address text invalid on gNB, ignored; matched unique reversed ports + owned process/RU + activation log.',
    wait_count=wait.get('count',0),selected_blocker_wall_s=wait.get('wall_s',0),selected_blocker_cpu_s=wait.get('cpu_s',0),
    wait_p95_ms_bin=wait.get('p95_wall_ms_bin'),wait_p99_ms_bin=wait.get('p99_wall_ms_bin'),
    wait_max_ms_since_start=wait.get('max_wall_ms_since_process_start'),channel_wall_s=channel.get('wall_s',0),channel_cpu_s=channel.get('cpu_s',0)))
  for x,y in zip(rr,rr[1:]):
   duration=(y['mono_ns']-x['mono_ns'])/1e9;v=native_delta(x,y)
   native_bins.append(dict(pid=key[0],tid=key[1],ru=key[2],comm=record['comm'],seconds=(x['wall_ns']-lo)/1e9,duration_s=duration,
    radio_per_wall=(y['rx_ts']-x['rx_ts'])/y['sample_rate']/duration,
    channel_cpu_fraction=sum(c['cpu_fraction'] for k,c in v.items() if 16<=k<20),
    epoll_wall_fraction=v.get(3,{}).get('wall_fraction',0),available_cpu_fraction=v.get(4,{}).get('cpu_fraction',0)))
 radio=A.ticks(CAP/'native_radio/NRUE_MAC_RLC_BUFFER_STATUS_live.csv')
 grants=H.radio_reader(CAP/'native_radio/NRUE_MAC_DCI_GRANT_live.csv',True)
 pusch=H.radio_reader(CAP/'pusch/ttracer/live_pusch_with_ingest.csv');per_ue={};rlc_bins=[]
 for s in attached:
  u=s['stream_id'];ff=[r for r in frames if r['stream_id']==u];rr=[r for r in radio if r['ue_id']==s['ue_id'] and r['rnti']==s['rnti']]
  assert all(b['radio_slot']>=a['radio_slot'] for a,b in zip(rr,rr[1:]))
  gg=[r for r in grants if int(r['rnti'])==s['rnti']];pp=[r for r in pusch if int(r['rnti'])==s['rnti'] and lo<=r['event_wall_ns']<=hi]
  received=[r for r in datagrams.values() if r['stream_id']==u and lo<=r['kernel_wall_ns']<=hi]
  rate=lambda b:b*8/(hi-lo)*1e3
  a=A.prior_tick(rr,lo);b=A.prior_tick(rr,hi)
  per_ue[u]=dict(source=s,sent=len(ff),kinds=dict(C.Counter(r['kind'] for r in ff)),payload_bytes=q(r['payload_bytes'] for r in ff),
   offered_Mbps=rate(sum(r['payload_bytes'] for r in ff)),received_chunk_Mbps=rate(sum(r['datagram_bytes']-8 for r in received)),
   rlc_delta_MB=(b['bytes']-a['bytes'])/1e6,rlc_median_MB=q(r['bytes']/1e6 for r in rr if lo<=r['wall_ns']<=hi)['p50'],
   sender_to_kernel_ms=q(r['last_send_to_kernel_ms'] for r in ff),radio=A.resources(gg,rr,lo,hi),
   snr_db=q(float(r['snrx10'])/10 for r in pp),mcs=q(r['mcs'] for r in pp))
  for r in rr:
   if lo<=r['wall_ns']<=hi:rlc_bins.append(dict(stream_id=u,seconds=(r['wall_ns']-lo)/1e9,bytes=r['bytes'],radio_slot=r['radio_slot']))
 processes=[]
 for pid in sorted({r['pid'] for r in thread_table}):
  ts=[r for r in thread_table if r['pid']==pid]
  processes.append(dict(pid=pid,role=ts[0]['role'],threads=len(ts),user_cpu_s=sum(r['user_cpu_s'] for r in ts),system_cpu_s=sum(r['system_cpu_s'] for r in ts),
   runqueue_thread_seconds=sum(r['runqueue_delay_s'] for r in ts),note='Sum across concurrent threads measures work/delay burden, not critical-path wall time.'))
 # Preserve the supplemental perf coverage failure explicitly; no capture retry.
 perf_pids=[int(x) for x in sr['identities']['perf']['command'][sr['identities']['perf']['command'].index('-p')+1].split(',')]
 native_pids={r['pid'] for r in native_summary if r['stages']}
 summary=dict(status='UE_RFSIM_CHANNEL_CPU_BOTTLENECK_GNB_PEER_WAIT_DOWNSTREAM',window_wall_ns=[lo,hi],window_s=(hi-lo)/1e9,
  performance_sample_window_mono_ns=[mlo,mhi],native=native_summary,threads=sorted(thread_table,key=lambda r:-r['cpu_s']),processes=processes,peers=peer_table,
  per_ue=per_ue,clock_mapping=clock.summary,mono_wall_offset_range_ns=max(offsets)-min(offsets),sampler=sr,
  perf=dict(recorded_pids=perf_pids,rf_device_pids_not_profiled=sorted(native_pids-set(perf_pids)),
   limitation='Discovery counted a gNB tracer helper as second process and started supplemental perf before UE launch. /proc and native instrumentation cover both logical UEs; UE instruction-level profile unavailable. No retry.'),
  interval_note='First/last 500-ms samples inside common first-to-last send window, ~41.5 s. Cumulative stages include completed calls; in-flight boundary calls are retained separately. Histograms give bins, maxima are since startup.',
  overlap_note='READ includes AVAILABLE and selected peer waits; FLUSH includes EPOLL; AVAILABLE includes CHANNEL/MERGE. UE threads and gNB overlap. Never sum these to an end-to-end time.',
  inference='Both UEthread_0/1 spend most available CPU inside rxAddInput. Unsynchronised process-global Gaussian RNG state is a source-supported candidate for amplification, not instruction-level attribution or a controlled causal test.')
 write('SUMMARY.json',summary);write('SOURCE_IDENTITIES.json',dict(attached=attached,peer_table=peer_table,process_identities=sr['identities']))
 table('THREADS.csv',thread_table);table('PROCESSES.csv',processes);table('NATIVE_STAGES.csv',native_table);table('PEERS.csv',peer_table)
 table('NATIVE_ALIGNED.csv',native_bins);table('THREAD_ALIGNED.csv',thread_bins);table('RLC_ALIGNED.csv',rlc_bins);table('FRAMES.csv',frames)
 plots(summary,native_bins,rlc_bins)
 write('INPUT_SHA256.json',H.INPUTS)
 write('REPRODUCIBILITY.json',dict(elapsed_s=time.monotonic()-start,command='env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B '+str(Path(__file__).relative_to(ROOT))+' /tmp/m4_radio_clock_reproduction_01',cpu=22,nice=19))
 print(json.dumps(dict(status=summary['status'],window_s=summary['window_s'],per_ue={u:{k:v for k,v in x.items() if k not in ('source','radio')} for u,x in per_ue.items()},perf=summary['perf']),indent=2))
def plots(s,nn,rr):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':17,'axes.titlesize':21,'axes.labelsize':17,'legend.fontsize':14,'axes.spines.top':False,'axes.spines.right':False})
 colors={'UEthread_0':'#146C9E','UEthread_1':'#D47720','ru_thread':'#50616B'}
 fig,axes=plt.subplots(2,1,figsize=(19.2,10.8),dpi=200,sharex=True)
 for comm in ('ru_thread','UEthread_0','UEthread_1'):
  v=[r for r in nn if r['comm']==comm]
  axes[0].plot([r['seconds'] for r in v],[r['radio_per_wall'] for r in v],label=comm,color=colors[comm],lw=2,alpha=.85)
 axes[0].axhline(.08532,color='#9A9A9A',ls='--',label='Previous smoke mean 0.0853 (different session)')
 axes[0].set_ylabel('Radio seconds / wall second');axes[0].set_title('Independent sample and frame clocks agree: common radio progress remains slow');axes[0].legend(ncol=2);axes[0].grid(alpha=.2)
 for i,u in enumerate(('m4-ue0','m4-ue1')):
  v=[r for r in rr if r['stream_id']==u];axes[1].plot([r['seconds'] for r in v],[r['bytes']/1e6 for r in v],color=colors['UEthread_'+str(i)],label=f'UE{i} RLC queue',lw=2)
 axes[1].set_ylabel('RLC queued bytes (MB)');axes[1].set_xlabel('Wall seconds since first send');axes[1].legend();axes[1].grid(alpha=.2)
 fig.suptitle('Two-UE radio progress and backlog under the unchanged smoke configuration',fontsize=25,y=.985)
 fig.text(.5,.025,'208 frames per UE; ~41.7-s send window. Native progress: cumulative samples / 122.88 Msamples/s; RLC: source-bound frame/slot trace.\nRecord-only instrumentation; an unmatched difference from the previous session is not a measured improvement.',ha='center',fontsize=15)
 fig.subplots_adjust(left=.08,right=.98,top=.88,bottom=.14,hspace=.27);fig.savefig(OUT/'01_RADIO_PROGRESS_BACKLOG.png',dpi=200);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(19.2,10.8),dpi=200,gridspec_kw={'width_ratios':[1.4,1]})
 comms=['UEthread_0','UEthread_1','ru_thread'];metrics=[]
 for name in comms:
  th=next(r for r in s['threads'] if r['comm']==name);na=next(r for r in s['native'] if r['comm']==name)
  st=na['stages'];metrics.append([100*th['cpu_fraction'],100*sum(v['cpu_fraction'] for k,v in st.items() if 16<=k<20),100*st[3]['wall_fraction'],100*th['runqueue_fraction']])
 x=np.arange(3);labels=['All thread CPU','Channel CPU (subset)','epoll wall wait','Runnable delay'];palette=['#50616B','#146C9E','#D47720','#AC4160']
 for j,label in enumerate(labels):
  vals=[m[j] for m in metrics];bars=axes[0].bar(x+(j-1.5)*.19,vals,.18,label=label,color=palette[j])
  for bar,v in zip(bars,vals):axes[0].text(bar.get_x()+bar.get_width()/2,v+1.3,f'{v:.1f}',ha='center',fontsize=14)
 axes[0].set_xticks(x,['UE0 radio\nthread','UE1 radio\nthread','gNB radio\nthread']);axes[0].set_ylim(0,114)
 axes[0].set_ylabel('Percent of each thread’s ~41.5-s window');axes[0].set_title('Both UE channel computations consume nearly a core');axes[0].legend(loc='upper center',ncol=2,fontsize=12)
 pp=sorted([r for r in s['peers'] if r['role']=='gnb'],key=lambda r:r['stream_id'])
 vals=[r['selected_blocker_wall_s'] for r in pp]
 axes[1].bar(['UE0 first\ninsufficient peer','UE1 first\ninsufficient peer'],vals,color=['#146C9E','#D47720'],width=.6)
 for i,(v,r) in enumerate(zip(vals,pp)):axes[1].text(i,v+.6,f'{v:.2f} s\n{r["wait_count"]:,} wait-loop calls',ha='center',fontsize=17)
 axes[1].set_ylim(0,max(vals)*1.25);axes[1].set_ylabel('gNB selected-blocker wait-loop wall seconds');axes[1].set_title('gNB stalls downstream of busy UE peers')
 fig.suptitle('Limiting stage: UE RFsim channel processing; gNB waiting is a consequence',fontsize=25,y=.985)
 fig.text(.5,.03,'Non-additive bars: channel CPU is inside total CPU; epoll is inside peer-wait loops. Concurrent UE/gNB times overlap.\nPeer labels identify the first insufficient socket, not exclusive causation. Native counters + ~84 external snapshots per thread.',ha='center',fontsize=15)
 fig.subplots_adjust(left=.07,right=.98,top=.86,bottom=.19,wspace=.27);fig.savefig(OUT/'02_CPU_PEER_WAIT.png',dpi=200);plt.close(fig)
if __name__=='__main__':main()
