"""Create-only final integrity inventory of this diagnosis and its one capture."""
import hashlib, json, os, subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
CAP=BASE.parents[2].parent/'abiodun_run4_single_gpu_profiler_v1/experiments/splitfusion_m4_two_ue_v1/smoke_live_20261013_09'
os.sched_setaffinity(0,{22});os.nice(19)
out=BASE/'FINAL_SHA256.json'
assert not out.exists()
hashes={}
for directory in (BASE,CAP):
    for p in sorted(directory.rglob('*')):
        if not p.is_file() or p==out:continue
        try:
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            digest=h.hexdigest()
        except PermissionError:
            # Owned recording and capture-local telnet histories are root-readable.
            assert p in (BASE/'observer/perf.data',
                         CAP/'radio_work/gnb/oaitelnet.history',
                         CAP/'radio_work/ue/oaitelnet.history'),p
            digest=subprocess.check_output(['sudo','-n','sha256sum',str(p)],text=True).split()[0]
        hashes[str(p)]=digest
with out.open('x') as f:json.dump(hashes,f,indent=2,sort_keys=True)
print(json.dumps(dict(files=len(hashes),manifest=str(out),manifest_sha256=hashlib.sha256(out.read_bytes()).hexdigest())))
