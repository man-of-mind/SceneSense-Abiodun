# DIAGNOSIS ONLY: GT reference for display-heading analysis. Never fed to the UI.
import csv, json, sys
from pathlib import Path
C=Path(sys.argv[1]); out=Path(sys.argv[2])
static=[]
with (C/'parity_static_catalogue/static_environment_objects.csv').open() as f:
    for r in csv.DictReader(f):
        if r['semantic_class'] in ('Car','Truck','Bus','Motorcycle','Bicycle') and r['enabled']=='true':
            static.append(dict(x=float(r['bbox_center_x_m']),y=float(r['bbox_center_y_m']),yaw=float(r['transform_yaw_deg']),cls=r['semantic_class'],L=2*float(r['bbox_extent_x_m'])))
frames={}
for d in (C/'numeric_gt/service/archive').iterdir():
    p=d/'scorer/target_metadata.json'
    if not p.exists(): continue
    t=json.loads(p.read_text()); fid=int(d.name.split('_')[1])
    frames[fid]=[dict(id=b['gt_actor_id'],label=b['label'],x=float(b['object_world_x']),y=float(b['object_world_y']),
        yaw=float(b['object_yaw_deg'] or 0),speed=float(b['object_speed_mps'] or 0),parked=b.get('parked_label'),src=b.get('source_kind'),state=b.get('contract_state'))
        for b in t['object_boxes']]
out.write_text(json.dumps(dict(use='OFFLINE_DIAGNOSIS_ONLY',static=static,frames=frames)))
print(len(static),len(frames))
