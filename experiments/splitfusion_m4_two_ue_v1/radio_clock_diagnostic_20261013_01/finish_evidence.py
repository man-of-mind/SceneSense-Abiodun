"""Offline supplemental evidence only. Never starts a radio or capture."""
import hashlib, importlib.util, json, os, re, shutil, socket, subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
DEP = ROOT.parent / 'abiodun_run4_single_gpu_profiler_v1'
CAP = DEP / 'experiments/splitfusion_m4_two_ue_v1/smoke_live_20261013_09'
os.sched_setaffinity(0, {22})
os.nice(19)
def doc(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n, v):
    with (BASE / n).open('x') as f:
        json.dump(v, f, indent=2, sort_keys=True, allow_nan=False)

s = doc(BASE / 'analysis_01/SUMMARY.json')
lo, hi = s['window_wall_ns']
ticks = [json.loads(x) for x in (BASE / 'observer/SAMPLER_TICKS.jsonl').read_text().splitlines()]
tt = [x for x in ticks if lo <= x['wall_ns'] <= hi]
quotas = set()
throttles = []
for t in tt:
    for c in t['cgroups'].values():
        for path, v in c['ancestors'].items():
            if v['cpu.max'] is not None:
                quotas.add((path, v['cpu.max']))
            st = dict(line.split() for line in v['cpu.stat'].splitlines())
            if 'nr_throttled' in st:
                throttles.append((int(st['nr_throttled']), int(st['throttled_usec'])))
assert all(v == 'max 100000' for _, v in quotas)
assert set(throttles) == {(0, 0)}
probe_ns = doc(BASE / 'checks_02/RESULT.json')['native']['empty_scope_ns']
scopes = sum(v['count'] for n in s['native'] for v in n['stages'].values())
overhead = dict(sampler_total_cpu_s=s['sampler']['cpu_s'], sampler_total_wall_s=s['sampler']['elapsed_s'],
    capture_sample_bodies=len(tt), capture_body_cpu_s=sum(t['cpu_ns'] for t in tt)/1e9,
    capture_body_wall_s=sum(t['elapsed_ns'] for t in tt)/1e9,
    capture_body_max_wall_ms=max(t['elapsed_ns'] for t in tt)/1e6,
    capture_body_cpu_percent=sum(t['cpu_ns'] for t in tt)/1e9/s['window_s']*100,
    cpu=22, nice=19, interval_ms=500, native_record_bytes=4736,
    maps=len(list((BASE/'observer/maps').glob('*.bin'))), empty_scope_ns=probe_ns,
    completed_scope_count=scopes, empty_scope_cost_projection_s=scopes*probe_ns/1e9,
    limitations='Sample-body costs exclude subsequent cgroup sampling, flushing and sleep bookkeeping. Total sampler CPU includes these. Empty-scope projection excludes device lookup, peer metadata and perturbation; not a measured whole-instrumentation overhead bound. Supplemental perf overhead not isolated. CPU22 shares front/map work, not protected radio/inference CPUs.',
    cgroup_cpu_max=sorted(quotas), observed_throttle_counter_pairs=sorted(set(throttles)))
write('OVERHEAD_AND_SCHEDULING.json', overhead)

oai = ROOT/'OAI/openairinterface5g'
source_specs = [
    (oai/'openair1/SIMULATION/TOOLS/rangen_double.c', [(120,157),(189,201)]),
    (oai/'radio/rfsimulator/apply_channelmod.c', [(216,264)]),
    (oai/'radio/rfsimulator/simulator.cpp', [])]
source_evidence = []
for p, spans in source_specs:
    lines=p.read_text().splitlines()
    source_evidence.append(dict(path=str(p), sha256=sha(p), excerpts=[dict(start=a,end=b,
        text='\n'.join(f'{i+1}: {lines[i]}' for i in range(a-1,min(b,len(lines))))) for a,b in spans]))
binary = DEP/'OAI/openairinterface5g/cmake_targets/ran_build/build/nr-uesoftmodem'
commands=[]
for symbol in ('gaussZiggurat','nfix'):
    cmd=['objdump','-d','--disassemble='+symbol,str(binary)]
    r=subprocess.run(cmd,text=True,capture_output=True,check=True)
    with (BASE/('UE_'+symbol+'_disassembly.txt')).open('x') as f: f.write(r.stdout)
    commands.append(cmd)
gnb_log=CAP/'gnb.log'
connections=[dict(line=i+1,text=line,actual_host_order_port=socket.ntohs(int(re.search(r'127\.0\.0\.1:(\d+)',line)[1])))
    for i,line in enumerate(gnb_log.read_text().splitlines()) if 'Client connects from ::ffff:127.0.0.1:' in line]
assert sorted(x['actual_host_order_port'] for x in connections)==[36238,36246]
assert sorted(x['peer_port'] for x in s['peers'] if x['role']=='gnb')==[36238,36246]
write('SOURCE_EVIDENCE.json',dict(sources=source_evidence,binary=str(binary),binary_sha256=sha(binary),
    disassembly_commands=commands,gnb_log=str(gnb_log),gnb_log_sha256=sha(gnb_log),connections=connections,
    peer_join='Native unique reversed TCP port pair, owned PID and RU index, ATTACHED source/RNTI. Original gNB connection log prints sin6_port in network order; ntohs reconciles it. Ignore gNB native IPv4-only address strings.',
    inference='Shared writable non-TLS Gaussian RNG state in one UE process is a candidate for coherence cost/data races. Stage cost is measured; the share due to RNG and cache coherence is not.'))

spec=importlib.util.spec_from_file_location('diagnostic_launch_readonly',BASE/'launch.py')
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
binding=launch.validate()  # Hash/registration verification only; no main/start/run.
assert doc(CAP/'SMOKE.json')['pass_']
assert doc(BASE/'POSTFLIGHT.json')['clean']
sizes={str(p):dict(allocated_bytes=int(subprocess.check_output(['du','-s','-B1',str(p)],text=True).split()[0]),
    apparent_bytes=int(subprocess.check_output(['du','-s','--apparent-size','-B1',str(p)],text=True).split()[0]))
    for p in (BASE,CAP)}
write('FINAL_CHECK.json',dict(status='PASS',registration_and_diagnostic_binding='unchanged and valid',
    binding_sha256=binding['sha256'],preparation_sha256=binding['preparation_sha256'],
    original_unchanged=doc(BASE/'BUILD.json')['original_unchanged'],
    smoke_pass=True,postflight_clean=True,actual_live_attempts=1,
    disk_free_bytes=shutil.disk_usage(ROOT).free,directory_sizes=sizes,
    prior_host_free_bytes=doc(BASE/'HOST_BEFORE.json')['free_bytes'],
    prelaunch_free_bytes=doc(BASE/'IMMEDIATE_PRELAUNCH.json')['free_bytes'],
    no_repair_implemented=True))
print(json.dumps(dict(overhead=overhead,final=doc(BASE/'FINAL_CHECK.json')),indent=2))
