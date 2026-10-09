"""Known-injection checks, not a universal detector of every possible profiler."""
import os
from pathlib import Path

def selected_env(raw):
    env=dict(x.split('=',1) for x in raw.split('\0') if '=' in x)
    keys={'LD_PRELOAD','LD_AUDIT','CUDA_INJECTION64_PATH','CUDA_INJECTION32_PATH','NVTX_INJECTION64_PATH','CUDA_LAUNCH_BLOCKING','KINETO_LOG_LEVEL','TORCH_PROFILER_LOG_LEVEL','VK_INSTANCE_LAYERS','VK_LAYER_PATH'}
    return {k:v for k,v in env.items() if k in keys or k.startswith(('NSYS','NSIGHT','NVPROF','CUPTI_'))}

def classify(e):
    reasons=[]
    hits=[x for x in e.get('maps','').splitlines() if any(y in x.lower() for y in ('libtoolsinjection','libcuinj','libnsys'))]
    if hits:reasons.append('known injected library maps')
    env=e.get('environment',{})
    blocked={k:v for k,v in env.items() if v and (k in {'LD_PRELOAD','LD_AUDIT','CUDA_INJECTION64_PATH','CUDA_INJECTION32_PATH','NVTX_INJECTION64_PATH'} or k.startswith(('NSYS','NSIGHT','NVPROF','CUPTI_')) or (k.startswith('VK_') and any(x in v.lower() for x in ['nsight','nsys'])))}
    if blocked:reasons.append('launch environment injection/profiling controls')
    collectors=e.get('collector_processes',[])
    if collectors:reasons.append('active Nsight collector process')
    for proc in e.get('ancestry',[]):
        a=proc.get('argv',[])
        if a and Path(a[0]).name in {'nsys','ncu','nvprof'}:reasons.append('profiler launch ancestry')
    if e.get('tracer_pid',0):reasons.append('process has TracerPid')
    return dict(passed=not reasons,reasons=sorted(set(reasons)),injection_maps=hits,blocked_environment=blocked,
        ordinary_cupti_present='libcupti' in e.get('maps','').lower(),
        scope='Known Nsight/injection indicators plus launch/process evidence; absence is not universal proof that no profiling exists.')

def collect(pid,maps=None):
    q=Path('/proc')/str(pid)
    status=q.joinpath('status').read_text()
    e=dict(maps=q.joinpath('maps').read_text() if maps is None else maps,
        environment=selected_env(q.joinpath('environ').read_bytes().decode(errors='replace')),
        tracer_pid=int(next(l.split(':')[1] for l in status.splitlines() if l.startswith('TracerPid:'))),ancestry=[],collector_processes=[])
    parent=pid;seen=set()
    while parent and parent not in seen:
        seen.add(parent);d=Path('/proc')/str(parent)
        try:
            argv=[x.decode(errors='replace') for x in d.joinpath('cmdline').read_bytes().split(b'\0') if x]
            fields=d.joinpath('stat').read_text().rsplit(')',1)[1].split()
            e['ancestry'].append(dict(pid=parent,argv=argv,starttime=fields[19]));parent=int(fields[1])
        except FileNotFoundError:break
    for d in Path('/proc').iterdir():
        if not d.name.isdecimal():continue
        try:
            argv=[x.decode(errors='replace') for x in d.joinpath('cmdline').read_bytes().split(b'\0') if x]
            if argv and Path(argv[0]).name in {'nsys','ncu','nvprof','nsys-agent','nsys-launcher'}:
                e['collector_processes'].append(dict(pid=int(d.name),argv=argv))
        except FileNotFoundError:pass
    e['verdict']=classify(e);return e
