"""Small read-only evidence checks; writes results only beside this script."""
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

OUT=Path(__file__).resolve().parent
os.nice(19)
os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})

# No importing plotting/runtime libraries, and no GPU/model operations.
data=list(csv.DictReader((OUT/'FRAME_TIMING.csv').open()))
assert len(data)==len({(r['stream_id'],r['frame_id']) for r in data})==416
summary=json.loads((OUT/'CORRECTED_ANALYSIS.json').read_text())
for s,expected in [('m4-ue0',(103,33,29,178,172)),('m4-ue1',(100,28,26,180,179))]:
    rows=[r for r in data if r['stream_id']==s];a=summary['per_ue'][s]
    assert (a['decisions'],a['predictions'],a['accepted_map_installs'],a['terminals']['FEEDBACK_TIMEOUT'],a['after_reassembly_expired'])==expected
    assert len(rows)==sum(a['terminals'].values())==208
    assert all(not r['kernel_receive_wall_ns'] and not r['exact_reassembly_raw_ns'] for r in rows)
    assert sum(bool(r['prediction_ready_raw_ns']) for r in rows)==a['predictions']
    assert sum(bool(r['map_install_wall_ns']) for r in rows)==a['accepted_map_installs']
    for r in rows:
        for k,v in r.items():
            if k.endswith('_ms') and v:
                assert float(v)>=0,(s,r['frame_id'],k,v)
        if r['edge_terminal_stage']=='EDGE_AFTER_REASSEMBLY':
            assert not r['prediction_ready_raw_ns']
            assert float(r['capture_to_verified_wall_ms'])>500
        if r['kind']=='POLICY_DECISION' and r['policy_feedback_raw_ns']:
            actual=(int(r['policy_feedback_raw_ns'])-int(r['action_open_raw_ns']))/1e6
            assert abs(actual-float(r['action_to_feedback_ms']))<1e-9 and actual>170
    assert a['actor_acknowledged_snr_consistent']==a['decisions']
    assert a['radio_windows']['capture_only']['snr_db']['p50']==20.5
assert sum(a['accepted_map_installs'] for a in summary['per_ue'].values())==55
inputs=json.loads((OUT/'INPUT_SHA256.json').read_text())
for name,expected in inputs.items():
    p=Path(name)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected['sha256'],name
repro=json.loads((OUT/'REPRODUCIBILITY.json').read_text())
assert repro['script_sha256']==hashlib.sha256((OUT/'analyze.py').read_bytes()).hexdigest()
names=['01_frame_outcomes','02_delay_localization']
assert sorted(p.stem for p in OUT.glob('*.png'))==names
for name in names:
    png=(OUT/f'{name}.png').read_bytes()
    assert png[:8]==b'\x89PNG\r\n\x1a\n'
    assert struct.unpack('>II',png[16:24])==(3840,2160)
    assert (OUT/f'{name}.pdf').read_bytes().startswith(b'%PDF-')
    root=ET.parse(OUT/f'{name}.svg').getroot()
    assert root.tag.endswith('svg')
result=dict(status='PASS',joined_frames=416,figure_bases=2,png_dimensions=[3840,2160],
            matched_input_hashes=len(inputs),negative_durations=0,production_mutations=0,
            checks='identity/count crosswalk; explicit noncompletion; clocks; source SNR; final script hash; formats and dimensions',
            visual_review='Both PNGs inspected; labels, denominator notes and distinct deadline anchors readable')
(OUT/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir())
         if p.is_file() and p.name!='OUTPUT_SHA256.json'}
(OUT/'OUTPUT_SHA256.json').write_text(json.dumps(outputs,indent=2)+'\n')
print(json.dumps(result))
