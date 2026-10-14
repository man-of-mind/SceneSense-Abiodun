"""Spatial assessment-binding negatives in new fixtures; retained evidence is read-only."""
import json
from pathlib import Path
import shutil
import sys

MAIN=Path(__file__).resolve().parents[2]
WORK=MAIN.parent/'abiodun_run4_single_gpu_profiler_v1'
sys.path.insert(0,str(WORK))
from rl_agent.splitfusion_m3_route_b_v19 import contract as C,host as H
from rl_agent.splitfusion_m3_route_b_v19_offline_v1 import binding as B


def main():
    pros=WORK/'rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261012_m3_route_b_v19/spatial_prospective_01'
    out=WORK/'experiments/splitfusion_m3_route_b_v19/offline_binding_02';out.mkdir()
    valid=B.resolve(pros)
    checks={'native_spatial_binding':valid.SCHEDULE==C.channel_registration(WORK)['schedule']};errors={}
    cases={'shifted_core':lambda r:r['channel']['schedule']['center_xy_m'].__setitem__(0,-54.0),
           'constant_v17_channel':lambda r:r.__setitem__('channel',C.V17.channel_registration(WORK)),
           'wrong_actor_feature':lambda r:r['channel']['actor_input'].__setitem__('feature_index',23)}
    for name,mutate in cases.items():
        f=out/name;f.mkdir()
        for p in pros.iterdir():
            if p.is_file() and p.name not in ('registration.json','PROSPECTIVE.json','compose_partition.yaml'):
                shutil.copy2(p,f/p.name)
        reg=json.loads((pros/'registration.json').read_text());mutate(reg)
        reg.pop('registration_sha256');reg['registration_sha256']=C.canonical(reg)
        H.write(f/'registration.json',reg)
        record=json.loads((pros/'PROSPECTIVE.json').read_text())
        with (f/'compose_partition.yaml').open('x') as stream:stream.write(H.compose_text(f.resolve(),record['retained_code']))
        record.update(registration_file_sha256=C.sha(f/'registration.json'),compose_file_sha256=C.sha(f/'compose_partition.yaml'))
        H.write(f/'PROSPECTIVE.json',record)
        try:B.resolve(f)
        except Exception as e:checks[name]=True;errors[name]=str(e)
        else:checks[name]=False
    H.write(out/'RESULT.json',dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,errors=errors,
                                 binding=B.document(valid),live_services_started=False,script_sha256=C.sha(__file__)))
    H.write(out/'OUTPUT_SHA256.json',{str(p.relative_to(out)):C.sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(dict(checks=checks,errors=errors)))
    return 0 if all(checks.values()) else 1


if __name__=='__main__':raise SystemExit(main())
