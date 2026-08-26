import sys, math, json, pickle
from pathlib import Path
sys.path.insert(0,'.')
import visibility_audit_v1 as V

RUN = Path('/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_noae_precision_full_v1/20260825_195301')
D = pickle.load(open('_gt.pkl','rb'))
gts, valids = D['gts'], D['valids']
TAGS = [('frozen historical noAE','baseline_frozen_noae'),
        ('Stage-1 selected e18','curriculum_stage1_objhead_v1_epoch_018'),
        ('Stage-2 selected e13','curriculum_stage2_joint_v1_epoch_013')]
RADII = [1.0,2.0,3.0,4.0,5.0]

def run(preds, md):
    V.MATCH_DISTANCE_M = md
    return V.score_rule(preds, gts, 'A', valids, honour_ignore=True)

# Also collect the matched-distance histogram so 3-5 m recoveries are explicit.
def matched_dists(preds_by_frame, md):
    V.MATCH_DISTANCE_M = md
    out={'vehicle':[], 'person':[]}
    for sid in valids:
        preds = preds_by_frame.get(sid,[])
        ev=[g for g in gts.get(sid,[]) if g['A']=='EVALUABLE']
        m=V.greedy(preds,ev,set(range(len(preds))),set(range(len(ev))))
        for pi,gi,d in m: out[ev[gi]['label']].append(d)
    return out

ALL={}
for name,tag in TAGS:
    preds = V.load_predictions(RUN/'eval'/tag/'detections.csv')
    ALL[tag]={'name':name,'r':{md:run(preds,md) for md in RADII},'md5':matched_dists(preds,5.0)}

def hdr(t): print('\n'+t); print('-'*104)

for tag,blk in ALL.items():
    hdr('%s   [%s]'%(blk['name'],tag))
    print('%-8s %-7s %8s %8s %8s %7s %7s %7s %9s'%('class','matchR','precision','recall','F1','TP','FP','FN','XY MAE m'))
    for cls in ('vehicle','person'):
        for md in (3.0,5.0):
            r=blk['r'][md]
            print('%-8s %-7s %9.4f %8.4f %8.4f %7d %7d %7d %9.4f'%(
                cls,'%.1f m'%md,r[cls+'_precision'],r[cls+'_recall'],r[cls+'_f1'],
                r[cls+'_tp'],r[cls+'_fp'],r[cls+'_fn'],r[cls+'_xy_mae_m']))
        r3,r5=blk['r'][3.0],blk['r'][5.0]
        print('%-8s %-7s newly accepted matches in 3-5 m band = %d   (recall +%.4f)'%(
            cls,'delta',r5[cls+'_tp']-r3[cls+'_tp'],r5[cls+'_recall']-r3[cls+'_recall']))
    print()
    print('  recall vs match radius (EVALUATION TOLERANCE CURVE):')
    print('  %-8s '%'class' + ' '.join('%9s'%('%.0f m'%md) for md in RADII))
    for cls in ('vehicle','person','overall'):
        print('  %-8s '%cls + ' '.join('%9.4f'%blk['r'][md][cls+'_recall'] for md in RADII))
    print('  matched-distance distribution of the 5 m match set (share of matches):')
    for cls in ('vehicle','person'):
        ds=sorted(blk['md5'][cls]); n=len(ds)
        bands=[(0,1),(1,2),(2,3),(3,4),(4,5)]
        parts=[]
        for lo,hi in bands:
            k=sum(1 for d in ds if lo<=d<hi); parts.append('%d-%dm:%5.1f%%'%(lo,hi,100*k/max(1,n)))
        print('    %-8s n=%5d  '%(cls,n)+'  '.join(parts))

pickle.dump(ALL,open('_radius.pkl','wb'))

# ---- gap attribution ----
H=json.load(open('/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/ae_integrated_20260710/sweeps_permodel/noae__clean/metrics/test_fusion_evaluation_metrics.json'))
hist={'vehicle':H['learned_vehicle_object_recall'],'person':H['learned_person_object_recall'],'overall':H['learned_object_recall']}
hdr('FRACTION OF THE HISTORICAL -> ROUTE B RECALL GAP EXPLAINED BY 3 m -> 5 m')
print('historical M-prime noae__clean (5 m, in-domain test split): veh=%.4f per=%.4f ov=%.4f'%(hist['vehicle'],hist['person'],hist['overall']))
print()
print('%-24s %-8s %8s %8s %8s %10s %12s'%('checkpoint','class','hist','RB@3m','RB@5m','gap(3m)','%gap by 3->5'))
for tag,blk in ALL.items():
    for cls in ('vehicle','person','overall'):
        r3,r5=blk['r'][3.0][cls+'_recall'],blk['r'][5.0][cls+'_recall']
        gap=hist[cls]-r3; expl=(r5-r3)/gap*100 if gap>0 else float('nan')
        print('%-24s %-8s %8.4f %8.4f %8.4f %10.4f %11.1f%%'%(blk['name'],cls,hist[cls],r3,r5,gap,expl))
