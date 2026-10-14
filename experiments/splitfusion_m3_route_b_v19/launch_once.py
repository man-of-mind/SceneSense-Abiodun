"""One-shot evidence wrapper around the existing production host launch; no retry."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

MAIN = Path(__file__).resolve().parents[2]
WORK = MAIN.parent/'abiodun_run4_single_gpu_profiler_v1'
OUT = Path(__file__).resolve().parent/'capture_01'
PROSPECTIVE = WORK/'rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261012_m3_route_b_v19/spatial_prospective_01'
STARTUP = WORK/'experiments/splitfusion_m3_route_b_v19/startup_verification_01/RESULT.json'


def save(path, value):
    with path.open('x') as f:
        json.dump(value,f,indent=1,sort_keys=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    OUT.mkdir(exist_ok=False)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    for name in ('PYTHONPATH','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LD_PRELOAD'):
        env.pop(name,None)
    sys.path.insert(0,str(WORK))
    from rl_agent.splitfusion_m3_route_b_v19 import contract as C,host as H
    from rl_agent.splitfusion_m3_route_b_v19.startup_verification import processes
    before=processes()
    containers=subprocess.run(['sudo','-n','docker','ps','-q'],capture_output=True,text=True,check=True).stdout.split()
    if before['processes'] or before['edge_image_containers'] or containers:
        save(OUT/'REFUSED.json',dict(reason='host not idle',processes=before,containers=containers))
        return 95
    verified=H.verify(PROSPECTIVE,STARTUP)
    offline=json.loads((WORK/'experiments/splitfusion_m3_route_b_v19/offline_binding_02/RESULT.json').read_text())
    if verified['status']!='PASS' or offline['status']!='PASS':
        raise RuntimeError('mandatory binding/startup proof did not pass')
    if (PROSPECTIVE/'capture').exists() or (PROSPECTIVE/'controller_events.jsonl').exists():
        raise FileExistsError('attempt already used; never retry')
    free=shutil.disk_usage(PROSPECTIVE).free
    if free < 17179869184:
        save(OUT/'REFUSED.json',dict(reason='unchanged 16-GiB disk gate',free_bytes=free))
        return 95
    registration=json.loads((PROSPECTIVE/'registration.json').read_text())
    argv=['/usr/bin/python3','-B','-m',C.HOST,'launch','--output',str(PROSPECTIVE),
          '--startup-proof',str(STARTUP),'--execute',registration['arms']['C2']['execute_token']]
    start=now();tick=time.monotonic()
    save(OUT/'PRELAUNCH.json',dict(started_utc=start,free_bytes=free,gate_bytes=17179869184,host_idle=before,
         all_containers=containers,verification=verified,command=argv,cwd=str(WORK),
         registration_sha256=C.sha(PROSPECTIVE/'registration.json'),startup_sha256=C.sha(STARTUP),
         approved_png_sha256=C.sha(MAIN/C.APPROVED_DIR/'01_proposed_spatial_snr_region.png'),
         preparation_source_start_epoch=int(subprocess.check_output(['stat','-c','%W',str(MAIN/C.PACKAGE/'__init__.py')],text=True)),
         retries_allowed=0))
    print(json.dumps(dict(status='LAUNCHING_SINGLE_ATTEMPT',free_bytes=free,started_utc=start)),flush=True)
    with (OUT/'launch.log').open('x') as stdout,(OUT/'launch.stderr').open('x') as stderr:
        result=subprocess.run(argv,cwd=WORK,env=env,stdout=stdout,stderr=stderr)
    # Numeric exit status is captured, never inferred. Preserve all evidence even on failure.
    with (OUT/'launch.rc').open('x') as f:f.write(str(result.returncode)+'\n')
    save(OUT/'LAUNCH_RESULT.json',dict(returncode=result.returncode,started_utc=start,finished_utc=now(),
         duration_s=time.monotonic()-tick,free_bytes_after=shutil.disk_usage(PROSPECTIVE).free,
         process_snapshot_after=processes(),
         containers_after=subprocess.run(['sudo','-n','docker','ps','-q'],capture_output=True,text=True,check=True).stdout.split(),
         retry=False))
    print(json.dumps(dict(status='ATTEMPT_FINISHED',returncode=result.returncode,duration_s=time.monotonic()-tick)),flush=True)
    return result.returncode


if __name__=='__main__':
    raise SystemExit(main())
