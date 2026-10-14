"""Validate the existing Town10 world-coordinate approximation, offline."""
import json
import numpy as np
from .recording import STATIC, sha_file

def validated_geometry(recording):
    try:
        return _validated_geometry(recording)
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return dict(roads=[], buildings=[], map_name=None, error="STATIC_GEOMETRY_UNAVAILABLE",
                    validation=dict(valid=False, reason="MISSING_OR_INVALID_GEOMETRY"))

def _validated_geometry(recording):
    data = json.loads(STATIC.read_text())
    registered = json.loads((recording.root/'registration.json').read_text())
    registration_text = json.dumps(registered)
    town_ok = data.get('map_name') == 'Carla/Maps/Town10HD_Opt' and 'Town10HD_Opt Route B' in registration_text
    valid_polys = all(len(b)>=3 and all(len(p)==2 and all(np.isfinite(p)) for p in b) for b in data['buildings'])
    segments = np.array([[a,b] for road in data['roads'] for a,b in zip(road,road[1:])])
    a,b = segments[:,0],segments[:,1];v=b-a;den=np.maximum(1e-9,(v*v).sum(axis=1));dist=[]
    # Independent consistency check against capture camera/ego world metres,
    # never a transform fitted to observations or an actor identity join.
    for wire in list(recording.wires.values())[::10]:
        p=np.array(wire['context']['ego_world'][:2]);t=np.clip(((p-a)*v).sum(axis=1)/den,0,1)
        dist.append(float(np.linalg.norm(p-(a+t[:,None]*v),axis=1).min()))
    p95=float(np.percentile(dist,95))
    data['validation']=dict(valid=bool(town_ok and valid_polys and p95<5 and not data.get('error')),
        geometry_sha256=sha_file(STATIC),source=str(STATIC),registered_town=town_ok,
        coordinates='CARLA world XY metres; no fitted transform',pose_to_lane_p95_m=p95,
        meaning='Filtered environment-object bounding-box footprints, not facade meshes',
        extraction='extract_traffic_lights.py:_environment_object_bbox_footprint / _build_precise_static_map',
        limitations='2D static buildings only; size/road-proximity filtered; no heights, transparency, bridges or dynamic occluders; asset version not retained')
    return data
