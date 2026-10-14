"""Small synchronized evidence audit. Reads only prediction fields, not GT identities.

Writes derived UI evidence into a new presentation directory; never changes
records, scores, scorer state, or historical metrics. No model execution.
"""
import argparse
from collections import Counter
import io
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .recording import Recording, ROOT, sha_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'artifacts')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    r=Recording();summaries=[];sources={};event_count=0;observations=0;all_headings=Counter()
    for clip_id,frame in [('departure',1480),('turn',2020),('stop',2748)]:
        clip=r.clip(clip_id);event=next(e for e in clip['events'] if e['frame_id']==frame)
        path=r.cell/f'numeric_gt/service/archive/frame_{frame}/scorer/numerical_score.json'
        # No actor boxes, matches, GT IDs or quality numbers are used for display/audit joins.
        score_doc=json.loads(path.read_text())
        predictions=score_doc['prediction_records'];identity=score_doc['prediction_identity']
        assert str(identity['capture_timestamp_ns'])==event['capture_ns']
        assert len(predictions)==len(event['objects'])
        matrix=np.array(event['pose']['camera_matrix']);errors=[];rows=[]
        before=[o['raw'] for o in event['objects']]
        camera=Image.open(io.BytesIO(r.camera(frame))).convert('RGB')
        draw=ImageDraw.Draw(camera)
        for i,(prediction,entry) in enumerate(zip(predictions,event['objects'])):
            raw=entry['raw'];world=np.array([prediction['world_'+a] for a in 'xyz'])
            installed=np.array([raw['location'][a] for a in 'xyz'])
            assert np.array_equal(world,installed)
            assert prediction['score']==raw['score']
            assert prediction['frame_id']==frame and prediction['stream_id']==entry['source']
            local=np.array([prediction['local_'+a] for a in 'xyz']+[1.])
            error=float(np.max(np.abs((matrix@local)[:3]-world)));errors.append(error)
            delta=world[:2]-np.array([event['pose']['sensor'][a] for a in 'xy'])
            angle=math.degrees(math.atan2(delta[1],delta[0]))-event['pose']['sensor']['yaw']
            angle=(angle+180)%360-180
            rows.append(dict(index=i,record_id=raw['id'],track_id=entry['track_id'],
                source=entry['source'],frame=frame,capture_ns=entry['capture_ns'],
                score=raw['score'],measured_xyz=world.tolist(),local_xyz=local[:3].tolist(),
                world_transform_error_m=error,sensor_bearing_deg=angle,range_m=float(np.linalg.norm(delta)),
                upstream_yaw_deg=math.degrees(math.atan2(prediction['yaw_sin'],prediction['yaw_cos'])),
                installed_yaw_deg=raw['yaw_deg'],installed_model_yaw_deg=raw['model_yaw_deg'],
                visibility='unknown',raw=raw))
            if i<6:
                box=[prediction[k] for k in ['bbox_x0','bbox_y0','bbox_x1','bbox_y1']]
                color=(110,224,202) if raw['type']=='Vehicle' else (228,165,207)
                draw.rectangle(box,outline=color,width=1);draw.text((box[0]+2,max(0,box[1]-12)),f'#{i} {raw["score"]:.2f}',fill=color)
        assert before==[o['raw'] for o in event['objects']]
        camera.save(args.out/f'camera-{frame}-prediction-boxes.png')
        rimage=args.out/f'camera-{frame}.png';rimage.write_bytes(r.camera(frame))
        pairs=[]
        for i,a in enumerate(rows):
            for b in rows[i+1:]:
                distance=float(np.linalg.norm(np.array(a['measured_xyz'][:2])-b['measured_xyz'][:2]))
                if distance<1:pairs.append(dict(a=a['index'],b=b['index'],distance_m=distance))
        summaries.append(dict(frame=frame,clip=clip_id,objects=len(rows),
            capture_to_install_ms=event['arrival_ms']-event['capture_ms'],
            pose_binding=event['pose']['binding'],pose=event['pose'],
            exact_xyz_score_join=True,maximum_camera_transform_error_m=max(errors),
            nearby_candidate_pairs_under_1m=pairs,records=rows,
            camera_note='Restored retained normalized RGB, exact frame; top six model boxes only in annotation. No GT/visibility label inferred.'))
        sources[str(path)]=sha_file(path)
    # Reproduce the legacy camera-centre-as-ego and missing source/FoV filtering.
    legacy_path=ROOT/'spatial_map_coop/recordings/two_ego_occl.jsonl'
    legacy=[]
    with legacy_path.open() as f:
        for line in f:
            snap=json.loads(line)['snap'];focus=snap['metadata']['focus_view'];source=focus['follow_stream_id']
            stream=next((s for s in snap['active_streams'] if s['stream_id']==source and s.get('sensor_pose')),None)
            if not stream:continue
            sensor=stream['sensor_pose'];ego=focus['ego_pose']
            dx=ego['x']-sensor['x'];dy=ego['y']-sensor['y'];yaw=math.radians(sensor['yaw_deg'])
            raw=snap['raw_spatial_map_objects'];remote=[o['id'] for o in raw if o['source_stream_id']!=source]
            behind=[o['id'] for o in raw if o['source_stream_id']==source and
                (o['location']['x']-sensor['x'])*math.cos(yaw)+(o['location']['y']-sensor['y'])*math.sin(yaw)<0]
            item=dict(frame=snap['frame_id'],stream=source,raw_count=len(raw),remote_records=remote,
                behind_reported_sensor=behind,camera_center_minus_sensor_m=[dx,dy],
                along_heading_offset_m=dx*math.cos(yaw)+dy*math.sin(yaw),
                limitation='Legacy retained snapshot has no validated capture-clock/pose binding; geometry diagnostics are not visibility truth.')
            if not legacy or remote:legacy.append(item)
            if remote:break
    # Count the retained installed-event inventory, without retaining a capture copy.
    with r.path.open() as f:
        for line in f:
            d=json.loads(line)
            if d['outcome']!='OK':continue
            event_count+=1
            for o in d['map_document']['record']['objects']:
                observations+=1;all_headings[str(o['yaw_deg'])]+=1
    sources[str(r.path)]=sha_file(r.path)
    sources[str(r.cell/'pose_archive/pose_evidence.jsonl')]=sha_file(r.cell/'pose_archive/pose_evidence.jsonl')
    sources[str(legacy_path)]=sha_file(legacy_path)
    result=dict(schema='scenesense.replay_ui_audit.v1',installed_events=event_count,
                installed_objects=observations,installed_heading_counts=dict(all_headings),
                legacy_reproductions=legacy,synchronized_samples=summaries,source_sha256=sources,
                measurement_or_scientific_state_modified=False,
                interpretation='Source/range/FoV policy errors and heading loss are demonstrated. Model false positives, occlusion, and physical identity of overlapping candidates are not resolved by this audit.')
    (args.out/'OBJECT_TRACE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['synchronized_samples','source_sha256']},indent=2))


if __name__=='__main__':main()
