"""Bounded derived evidence, hashes and join coverage; no experiment writes."""
import json
import argparse
from pathlib import Path
from collections import Counter
from .recording import Recording, ROOT, sha_file
from .static_geometry_v2 import validated_geometry
from .heading_ingest_v2 import heading

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips-dir',type=Path,help='optional bounded clip exports for the Node comparison')
    args=parser.parse_args()
    if args.clips_dir:args.clips_dir.mkdir(parents=True,exist_ok=True)
    r=Recording();out=Path(__file__).parent/'artifacts_v2';out.mkdir(exist_ok=True)
    sources=[
        'carla_collect_parked_ego_fusion_training_data.py',
        'pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_clean_base_v1/build_contract_v1.py',
        'pole_lraspp_multimodal_fusion/object_head_pilot_v1/splitfusion_fcos_r50_fpn_p2_p7_v1/data.py',
        'pole_lraspp_multimodal_fusion/object_head_pilot_v1/splitfusion_fcos_r50_fpn_p2_p7_v1/model.py',
        'pole_lraspp_multimodal_fusion/object_head_pilot_v1/splitfusion_fcos_r50_fpn_p2_p7_v1/infer.py',
        'uplink_only_spatial_map_pipeline/spatial_map_server_moving_ego_uplink_only_baseline.py',
        'rl_agent/splitfusion_direct_edge_map_v1/spatial_map_direct_server_v1.py',
        'rl_agent/splitfusion_direct_edge_map_v1/protocol.py','extract_traffic_lights.py']
    report=dict(schema='replay_heading_geometry_evidence.v2',producer_source_hashes={p:sha_file(ROOT/p) for p in sources},
        convention=dict(space='CARLA world XY',positive_yaw='+X toward +Y',length='size_x along local +X',width='size_y along local +Y',
                        conversion='wrap(degrees(atan2(yaw_sin,yaw_cos))); no camera rotation, no 90-degree offset',
                        trace=['actor transform.rotation.yaw / static bbox_rotation_yaw_deg -> object_yaw_deg',
                               'FCOS data.py targets sin/cos of world yaw; dimensions unchanged',
                               'model.py geometry_decode rotates local_xyz using camera extrinsic; yaw only normalized',
                               'infer.py record copies yaw[0]/yaw[1], dimensions[0:3]',
                               'legacy map normalizer defaults model yaw to zero, then adds configured offset']),
        static_geometry=validated_geometry(r)['validation'],clips={},examples=[])
    for name in ['departure','turn','stop','avenue']:
        clip=r.clip(name)
        if args.clips_dir:(args.clips_dir/f'replay-{name}.json').write_text(json.dumps(clip))
        counts=Counter(o['heading_reason'] for e in clip['events'] for o in e['objects'])
        report['clips'][name]=dict(events=len(clip['events']),objects=sum(len(e['objects']) for e in clip['events']),
                                  joined_frames=sum(any(o['upstream'] for o in e['objects']) for e in clip['events']),heading_reasons=dict(counts))
    for frame in (1480,2020,2748):
        doc=json.loads((r.cell/f'numeric_gt/service/archive/frame_{frame}/scorer/numerical_score.json').read_text())
        report['examples'].append(dict(frame=frame,record_count=len(doc['prediction_records']),
            headings=[dict(installed_ordinal=i,prediction_index=p['prediction_index'],candidate_identity=p['candidate_identity'],heading=heading(p)) for i,p in enumerate(doc['prediction_records'])]))
    report['historical_input_sha256']={str(p):sha_file(p) for p in [r.path,r.cell/'pose_archive/pose_evidence.jsonl',r.cell/'direct_edge_map/direct_map_ingest.csv']}
    old=json.loads((Path(__file__).parent/'artifacts/VALIDATION.json').read_text())['source_sha256'];baseline=Path(__file__).parent/'baseline_v1';verified={}
    for p in baseline.iterdir():
        key=('spatial_map_coop/replay_ui_v1/' if p.name=='recording.py' else 'spatial_map_coop/viewer_assets/')+p.name
        if key in old:
            verified[p.name]=sha_file(p)==old[key]
            assert verified[p.name],p
    report['baseline_hash_matches_prior_validation']=verified
    (out/'HEADING_GEOMETRY_EVIDENCE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['clips'],indent=2))

if __name__=='__main__':main()
