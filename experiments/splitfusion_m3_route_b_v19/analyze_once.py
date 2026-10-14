"""Record one pass through existing registered assessment/readers and spatial presentation."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
PROS=ROOT.parent/'abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261012_m3_route_b_v19/spatial_prospective_01'

def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=1,sort_keys=True)

def main():
    result=json.loads((BASE/'capture_01/LAUNCH_RESULT.json').read_text())
    out=BASE/'analysis_execution_01';out.mkdir(exist_ok=False)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(out/'mplconfig'))
    env.pop('PYTHONPATH',None);env.pop('LD_PRELOAD',None)
    py=[sys.executable,'-B'];assessment=BASE/'registered_assessment_01'
    analysis=BASE/'analysis_01';layers=BASE/'route_layers_01';regions=BASE/'regions_01'
    commands=[
      ('registered_assessment',py+['-m','rl_agent.splitfusion_m3_route_b_v19_offline_v1.postflight','--arm',str(PROS),'--out',str(assessment),'--launch-log',str(BASE/'capture_01/launch.log'),'--launch-rc',str(result['returncode'])]),
      ('retention',py+['-m','rl_agent.splitfusion_m3_route_b_v19.postflight_p3','retention',str(PROS),str(BASE/'retention_01')]),
      ('analysis',py+['-m','rl_agent.splitfusion_m3_route_b_v17_analysis.analyze','--root',str(PROS),'--verdict',str(assessment/'VERDICT_registered_binding.json'),'--out',str(analysis)]),
      ('layers',py+[str(BASE/'join_layers.py'),'--capture',str(PROS),'--analysis',str(analysis),'--out',str(layers)]),
      ('regions',py+[str(BASE/'regions.py'),'--capture',str(PROS),'--analysis',str(analysis),'--layers',str(layers),'--out',str(regions)]),
      ('presentation',py+[str(BASE/'render_spatial.py'),'--analysis',str(analysis),'--layers',str(layers),'--regions',str(regions),'--out',str(BASE/'presentation_01')]),
    ]
    started=datetime.now(timezone.utc).isoformat();tick=time.monotonic();records=[]
    save(out/'COMMANDS.json',dict(commands=commands,cwd=str(ROOT),cpu_only=True,started_utc=started))
    for name,argv in commands:
        at=time.monotonic();print(json.dumps(dict(stage=name,status='STARTED')),flush=True)
        with (out/(name+'.log')).open('x') as f:
            p=subprocess.run(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        r=dict(stage=name,returncode=p.returncode,duration_s=time.monotonic()-at)
        records.append(r);save(out/(name+'.json'),r);print(json.dumps(r),flush=True)
        if p.returncode and not (name=='registered_assessment' and (assessment/'ASSESSMENT.json').exists()):
            break
    save(out/'RESULT.json',dict(started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),duration_s=time.monotonic()-tick,stages=records))
    return 0 if len(records)==len(commands) and all(r['returncode']==0 for r in records) else 3

if __name__=='__main__':raise SystemExit(main())
