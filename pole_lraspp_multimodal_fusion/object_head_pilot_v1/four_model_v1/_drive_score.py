import sys, math, json, pickle
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
import visibility_audit_v1 as V

RUN = Path('/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_noae_precision_full_v1/20260825_195301')
D = pickle.load(open('_gt.pkl','rb'))
gts, valids = D['gts'], D['valids']
TAGS = [('frozen historical noAE','baseline_frozen_noae'),
        ('Stage-1 selected e18','curriculum_stage1_objhead_v1_epoch_018'),
        ('Stage-2 selected e13','curriculum_stage2_joint_v1_epoch_013')]

def score(preds_by_frame, rule, md):
    V.MATCH_DISTANCE_M = md
    return V.score_rule(preds_by_frame, gts, rule, valids, honour_ignore=True)

print('=== RULES A / B / C ARE IDENTICAL (proved: 0 IGNORE objects) -> one recall per match radius ===')
print()
hdr='%-24s %-6s %8s %8s %8s %9s %9s %9s'%('checkpoint','matchR','ov_rec','veh_rec','per_rec','ov_prec','FP/frame','ov_F1')
print(hdr); print('-'*len(hdr))
store={}
for name,tag in TAGS:
    preds = V.load_predictions(RUN/'eval'/tag/'detections.csv')
    for md in (3.0, 5.0):
        r = score(preds, 'A', md)
        store[(tag,md)] = r
        print('%-24s %-6s %8.4f %8.4f %8.4f %9.4f %9.4f %9.4f'%(
            name, '%.1f m'%md, r['overall_recall'], r['vehicle_recall'], r['person_recall'],
            r['overall_precision'], r['overall_fp_per_frame'], r['overall_f1']))
    print()
pickle.dump(store, open('_scores.pkl','wb'))

print('=== HISTORICAL M-PRIME REFERENCE (retained, test split of the M-prime corpus) ===')
h=json.load(open('/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/ae_integrated_20260710/sweeps_permodel/noae__clean/metrics/test_fusion_evaluation_metrics.json'))
gt_h=h['learned_object_tp']+h['learned_object_fn']
print('  noae__clean  ov_rec=%.4f veh=%.4f per=%.4f prec=%.4f  GT=%d over %d frames = %.2f GT/frame'%(
    h['learned_object_recall'],h['learned_vehicle_object_recall'],h['learned_person_object_recall'],
    h['learned_object_precision'],gt_h,h['samples'],gt_h/h['samples']))
print('  match radius used by every historical sweep script: 5.0 m  (rl_agent/run_*.sh)')
print()
r2=store[('curriculum_stage2_joint_v1_epoch_013',3.0)]
gt_rb=r2['overall_tp']+r2['overall_fn']
print('  Route B val  GT=%d over %d frames = %.2f GT/frame  (%.2fx denser)'%(
    gt_rb,r2['frames'],gt_rb/r2['frames'],(gt_rb/r2['frames'])/(gt_h/h['samples'])))
