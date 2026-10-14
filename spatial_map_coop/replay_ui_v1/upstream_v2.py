"""Exact retained prediction join. No GT records or matches participate."""
import json
from .heading_ingest_v2 import heading
from .recording import digest, sha_file

def join_predictions(recording, record, pose):
    if not pose:
        return {}, 'NO_BOUND_POSE'
    path = recording.cell / f"numeric_gt/service/archive/frame_{record['frame_id']}/scorer/numerical_score.json"
    if not path.exists():
        return {}, 'NO_RETAINED_PREDICTION'
    d = json.loads(path.read_text())
    ident = d['prediction_identity']; binding = ident['gt_identity']
    expected = pose['binding']
    if [binding[k] for k in ('run_id','cell_id','stream_id','frame_id')] + [str(binding['capture_timestamp_ns'])] != expected:
        return {}, 'PREDICTION_IDENTITY_MISMATCH'
    if str(ident['capture_timestamp_ns']) != pose['capture_ns']:
        return {}, 'PREDICTION_TIMESTAMP_MISMATCH'
    preds = d['prediction_records']
    if len(preds) != len(record['objects']):
        return {}, 'PREDICTION_COUNT_MISMATCH'
    # Exact record ordinal, full frame identity, coordinates, dimensions, score
    # and class are all required. Never spatial/nearest-neighbor yaw assignment.
    if len({p['prediction_index'] for p in preds}) != len(preds) or len({p['candidate_identity'] for p in preds}) != len(preds):
        return {}, 'AMBIGUOUS_PREDICTION_IDENTITY'
    result = {}; source_hash = sha_file(path)
    for i, (p, raw) in enumerate(zip(preds, record['objects'])):
        if (raw['id'] != f"{record['stream_id']}:{record['frame_id']}:{i}" or
            p['stream_id'] != record['stream_id'] or p['frame_id'] != record['frame_id'] or
            str(p['capture_timestamp_ns']) != pose['capture_ns'] or p['score'] != raw['score'] or
            {'vehicle':'Vehicle','person':'Pedestrian'}.get(p['class_name']) != raw['type'] or
            any(p['world_'+a] != raw['location'][a] for a in 'xyz') or
            any(p['size_'+a] != raw['dimensions'][b] for a,b in zip('xyz',('length','width','height')))):
            return {}, 'PREDICTION_RECORD_MISMATCH'
        result[raw['id']] = dict(heading=heading(p), candidate={
            k:p[k] for k in ('candidate_identity','prediction_index','fpn_level','point_index','internal_class',
                            'bbox_x0','bbox_x1','bbox_y0','bbox_y1','local_x','local_y','local_z')},
            provenance=dict(path=str(path), sha256=source_hash, record_sha256=digest(p),
                            binding=expected, prediction_index=p['prediction_index'], installed_ordinal=i, join='EXACT_IDENTITY_XYZ_DIMENSIONS_SCORE_CLASS'))
    return result, None
