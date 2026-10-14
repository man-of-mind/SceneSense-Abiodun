"""Verify existing output hashes and bind the final report; no live or inference operations."""
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import shutil
import sys
from PIL import Image

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
WORK=ROOT.parent/'abiodun_run4_single_gpu_profiler_v1'
PROS=WORK/'rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261012_m3_route_b_v19/spatial_prospective_01'
sys.path.insert(0,str(ROOT))
from rl_agent.splitfusion_m3_route_b_v19 import contract as C

def read(p):return json.loads(p.read_text())

def main():
    record=read(PROS/'PROSPECTIVE.json')
    assert C.canonical(record['runtime_sources'])=='f184f30a603ccf6ce69a11871308f0ce02ebb26babab30a4e5ba40d321549e51'
    assert C.canonical(record['sources'])=='5493f6eb27c91ce4e2100b3f9b7c044376b91b3512bf24528d5f89efa34bdfb8'
    for source,digest in record['sources'].items():assert C.sha(WORK/source)==digest,source
    for source,digest in record['runtime_sources'].items():assert C.sha(WORK/source)==digest,source
    checked={}
    for name in ('registered_assessment_01','analysis_01','regions_01','presentation_01','presentation_02'):
        manifest=BASE/name/'OUTPUT_SHA256.json'
        for relative,digest in read(manifest).items():assert C.sha(manifest.parent/relative)==digest,relative
        checked[name]=C.sha(manifest)
    for name in ('02_channel_mode_q_payload.png','03_prediction_feedback_cdf.png','04_map_freshness_cdf.png','05_latency_map_freshness_time.png'):
        # The first invocation used absolute analysis paths; the second used relative
        # paths. Existing PNG Source metadata consequently differs, not the pixels.
        with Image.open(BASE/'presentation_01'/name) as a,Image.open(BASE/'presentation_02'/name) as b:
            assert a.size==b.size and a.mode==b.mode and a.tobytes()==b.tobytes(),name
            assert (ROOT/a.info['Source']).resolve()==(ROOT/b.info['Source']).resolve()
    for target in re.findall(r'\]\(([^)]+)\)',(BASE/'REPORT.md').read_text()):assert (BASE/target).exists(),target
    C.approved_design(ROOT)
    now=datetime.now(timezone.utc)
    started=datetime.fromisoformat(read(BASE/'analysis_execution_01/RESULT.json')['started_utc'])
    external={str(p):C.sha(p) for p in [PROS/'registration.json',PROS/'PROSPECTIVE.json',PROS/'PREREQUISITE.json',PROS/'RETENTION_RESULT.json',PROS/'RESTORATION_LATEST.json',WORK/'experiments/splitfusion_m3_route_b_v19/startup_verification_01/RESULT.json',WORK/'experiments/splitfusion_m3_route_b_v19/binding_regression_01/RESULT.json',WORK/'experiments/splitfusion_m3_route_b_v19/offline_binding_02/RESULT.json',ROOT/C.APPROVED_DIR/'01_proposed_spatial_snr_region.png']}
    files={str(p.relative_to(BASE)):C.sha(p) for p in sorted(BASE.rglob('*')) if p.is_file() and 'mplconfig' not in p.parts and '__pycache__' not in p.parts and p.name!='FINAL_MANIFEST.json'}
    doc=dict(status='PASS',sealed_utc=now.isoformat(),verdict=read(BASE/'registered_assessment_01/ASSESSMENT.json')['verdict'],
        files=files,external_evidence=external,verified_output_manifests=checked,
        runtime_inventory_sha256=C.canonical(record['runtime_sources']),prospective_inventory_sha256=C.canonical(record['sources']),
        source_inventory_verified_unchanged=True,approved_png_unchanged=True,final_presentation='presentation_02',
        other_four_presentation_pngs_pixels_identical=True,analysis_and_report_wall_s=(now-started).total_seconds(),
        automated_analysis_s=read(BASE/'analysis_execution_01/RESULT.json')['duration_s'],free_bytes_at_seal=shutil.disk_usage(ROOT).free,
        attempts=1,retries=0,stopped_after_report=True)
    with (BASE/'FINAL_MANIFEST.json').open('x') as f:json.dump(doc,f,indent=1,sort_keys=True)
    print(json.dumps({k:v for k,v in doc.items() if k not in ('files','external_evidence','verified_output_manifests')}))

if __name__=='__main__':main()
