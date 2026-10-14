"""Read-only post-capture binding/restoration/disk audit; writes only this report directory."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
WORK=ROOT.parent/'abiodun_run4_single_gpu_profiler_v1'
PROS=WORK/'rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261012_m3_route_b_v19/spatial_prospective_01'
sys.path.insert(0,str(WORK))
from rl_agent.splitfusion_m3_route_b_v19 import contract as C,host as H
from rl_agent.splitfusion_m3_route_b_v19.startup_verification import processes

def read(p):return json.loads(p.read_text())
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=1,sort_keys=True)

def main():
    out=BASE/'final_audit_02';out.mkdir(exist_ok=False)
    verified=H.verify(PROS,WORK/'experiments/splitfusion_m3_route_b_v19/startup_verification_01/RESULT.json')
    C.approved_design(ROOT)
    snapshot=processes()
    containers=subprocess.run(['sudo','-n','docker','ps','-q'],capture_output=True,text=True,check=True).stdout.split()
    assert not snapshot['processes'] and not snapshot['edge_image_containers'] and not containers
    cell=next((PROS/'capture/cells').iterdir())
    with (cell/'radio_trace.csv').open() as f:radio=list(csv.DictReader(f))
    spatial={}
    for status in sorted({r['pose_status'] for r in radio}):
        rows=[r for r in radio if r['pose_status']==status]
        spatial[status]=dict(samples=len(rows),max_selected_source_age_ms=max(float(r['pose_age_ms']) for r in rows),
            max_source_to_command_ms=max((int(r['command_send_monotonic_ns'])-int(r['source_monotonic_ns']))/1e6 for r in rows if r['command_send_monotonic_ns']))
    storage={}
    for name,path in [('attempt',PROS),('report',BASE),('preparation_proofs',WORK/'experiments/splitfusion_m3_route_b_v19')]:
        storage[name]=dict(path=str(path),allocated_bytes=int(subprocess.check_output(['du','-s','-B1',str(path)],text=True).split()[0]))
    before=read(BASE/'capture_01/PRELAUNCH.json');launched=read(BASE/'capture_01/LAUNCH_RESULT.json')
    registration=read(PROS/'PROSPECTIVE.json')
    restored=read(PROS/'RESTORATION_LATEST.json');radio_restore=read(PROS/'radio_prestop_restore_command.json')
    assert restored['passed'] and radio_restore['returncode']==0
    doc=dict(status='PASS',audited_utc=datetime.now(timezone.utc).isoformat(),binding=verified,
        registered_sources=len(registration['sources']),registration_sha256=C.sha(PROS/'registration.json'),
        source_inventory_sha256=C.canonical(registration['sources']),
        approved_png_sha256=C.sha(ROOT/C.APPROVED_DIR/'01_proposed_spatial_snr_region.png'),
        host_idle=snapshot,containers=containers,partition_restore=restored,radio_restore=radio_restore,
        spatial_pose_ages=spatial,storage=storage,free_bytes=shutil.disk_usage(ROOT).free,
        capture_disk_free_decrease_bytes=before['free_bytes']-launched['free_bytes_after'],
        timed_implementation_and_checks_s=datetime.fromisoformat(before['started_utc']).timestamp()-before['preparation_source_start_epoch'],
        preparation_timing_limit='First implementation source creation to launch; preceding read-through not instrumented.',
        evidence_deleted=False,evidence_archived_to_make_space=False,retries=0,commit=False,push=False)
    save(out/'RESULT.json',doc)
    print(json.dumps(dict(status=doc['status'],free_bytes=doc['free_bytes'],registered_sources=doc['registered_sources'],spatial_pose_ages=spatial)))

if __name__=='__main__':main()
