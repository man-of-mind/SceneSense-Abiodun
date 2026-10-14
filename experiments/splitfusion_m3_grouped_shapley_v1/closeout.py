"""Create-only final provenance and storage inventory; no model queries."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
from common import BASE,ROOT,old,write

result=json.loads((BASE/'attribution_01/RESULTS.json').read_text())
inputs=dict(result['inputs_sha256'])
for p in [ROOT/'experiments/splitfusion_decision_cases_v1/MEMO.md',
          ROOT/'experiments/splitfusion_decision_cases_v1/PLAN.md',
          ROOT/'experiments/splitfusion_m3_actor_diagnosis_v1/MEMO.md',
          ROOT/'experiments/splitfusion_m3_actor_diagnosis_v1/diagnose.py',
          Path('/home/shr_aisvcs/.codex/attachments/b89eb487-b939-42ad-aaf1-d1c4a3f6ce5a/Pasted text.txt')]:
    inputs[str(p)]=old.sha(p)
for p,h in inputs.items():assert old.sha(p)==h,p
paths=sorted(p for p in BASE.rglob('*') if p.is_file())
assert not (BASE/'MANIFEST.json').exists()
size=sum(p.stat().st_size for p in paths);assert size<500*1024**2
disk=shutil.disk_usage(BASE);now=datetime.now(timezone.utc)
manifest=dict(schema='splitfusion.grouped_shapley.closeout.v1',utc=now.isoformat(),
    scope='CPU actor interpretation only; no live/GPU/training/production/evidence mutation',
    weights_sha256=result['weights_sha256'],inputs_sha256=inputs,
    files={str(p.relative_to(BASE)):dict(bytes=p.stat().st_size,sha256=old.sha(p)) for p in paths},
    total_artifact_bytes_before_manifest=size,limit_bytes=500*1024**2,
    available_disk_bytes=disk.free,available_disk_GiB=disk.free/1024**3,
    start_available_disk_bytes=14324477952,
    storage_note='No cleanup, deletion or archiving. Filesystem headroom change also reflects other host activity. Manifest excludes its own hash/size.',
    start_utc='2026-10-12T06:37:03+00:00',elapsed_seconds=(now-datetime.fromisoformat('2026-10-12T06:37:03+00:00')).total_seconds(),
    attribution_seconds=result['seconds'],cuda_initialized=result['cuda_initialized'],
    final_figures='figures_02',draft_figures_retained='figures_01',
    visual_inspection='All six final PNGs inspected; first draft label clipping and legend collision corrected in final render; no attribution rerun.',
    python=sys.version,additive_error=result['maximum_reconstruction_error'])
write(BASE/'MANIFEST.json',manifest)
print(json.dumps({k:manifest[k] for k in ('utc','total_artifact_bytes_before_manifest','available_disk_bytes','available_disk_GiB','elapsed_seconds','additive_error')},indent=1))
