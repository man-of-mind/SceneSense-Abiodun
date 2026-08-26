import sys, math, json, statistics as st
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0,'.')
import visibility_audit_v1 as V

RUN = Path('/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_noae_precision_full_v1/20260825_195301')
man = V.load_csv(RUN/'dataset/manifest.csv')
val = {r['sample_id']: r for r in man if r['split']=='val'}
print('val frames in manifest:', len(val))
wh = {sid: (float(r['camera_width']), float(r['camera_height'])) for sid,r in val.items()}
print('camera sizes:', Counter(wh.values()))
assign = V.region_assigner()
region = {sid: assign(float(r['camera_x']), float(r['camera_y'])) for sid,r in val.items()}
print('frames per region:', Counter(region.values()))

boxes = V.load_csv(RUN/'dataset/object_boxes.csv')
print('object_boxes rows total:', len(boxes))
gts = defaultdict(list)
allrec = []
for row in boxes:
    sid = row['sample_id']
    if sid not in val: continue
    w,h = wh[sid]
    g = V.classify_gt(row, w, h)
    g['region'] = region[sid]
    if not g['eligible_class']: continue
    allrec.append(g); gts[sid].append(g)
json.dump({'n':len(allrec)}, open('_n.json','w'))
import pickle
pickle.dump({'gts':dict(gts),'region':region,'wh':wh,'valids':set(val)}, open('_gt.pkl','wb'))

print()
print('=== TARGET-CLASS ACTOR GT ON VAL SPLIT ===')
print('total target GT rows      :', len(allrec))
inr = [g for g in allrec if g['in_range']]
oor = [g for g in allrec if not g['in_range']]
print('within 40 m               :', len(inr))
print('OUT_OF_RANGE (>40 m)      :', len(oor), '(%.1f%%)'%(100*len(oor)/len(allrec)))
for cls in ('vehicle','person'):
    c=[g for g in inr if g['label']==cls]; print('  within 40 m %-8s: %d'%(cls,len(c)))

print()
print('=== GEOMETRY / SUPPORT DISTRIBUTIONS, GT WITHIN 40 m (n=%d) ==='%len(inr))
def q(vals,name,fmt='%.2f'):
    v=sorted(x for x in vals if not math.isnan(x))
    if not v: print('  %-26s (no data)'%name); return
    ps=[0,5,10,25,50,75,90,100]
    print('  %-26s'%name, ' '.join(('p%d='+fmt)%(p,v[min(len(v)-1,int(p/100*len(v)))]) for p in ps))
q([g['distance_m'] for g in inr],'gt_distance_m')
q([g['depth_m'] for g in inr],'gt_depth_m')
q([g['proj_h_px'] for g in inr],'projected box height px')
q([g['clipped_h_px'] for g in inr],'CLIPPED box height px')
q([g['area_recorded_px'] for g in inr],'gt_bbox_area_px (recorded)')
q([g['clipped_area_px'] for g in inr],'clipped area px')
q([g['truncation'] for g in inr],'truncation frac','%.3f')
q([g['radar_support_points'] for g in inr],'radar_support_points','%.0f')

print()
print('=== BOOLEAN FACTS, GT WITHIN 40 m ===')
def pct(f,name):
    n=sum(1 for g in inr if f(g)); print('  %-42s %6d  %5.1f%%'%(name,n,100*n/max(1,len(inr))))
pct(lambda g: g['depth_positive'],'positive camera depth')
pct(lambda g: not g['depth_positive'],'BEHIND camera (depth<=0)')
pct(lambda g: g['centre_in_image'],'projected CENTRE in image (rule A test)')
pct(lambda g: g['intersects_image'],'clipped box intersects image')
pct(lambda g: g['intersects_image'] and not g['centre_in_image'],'intersects but centre OUTSIDE image')
pct(lambda g: g['clipped_area_px']>=V.MIN_GT_AREA_PX,'clipped area >= 12 px')
pct(lambda g: g['area_recorded_px']>=V.MIN_GT_AREA_PX,'recorded area >= 12 px')
pct(lambda g: g['radar_ok'],'radar_support_points > 0')
pct(lambda g: g['truncation']>0.5,'truncation > 50%')
pct(lambda g: g['truncation']>=1.0,'fully outside image (truncation=100%)')

print()
print('=== recorded gt_bbox_area_px vs computed clipped area (is recorded clipped?) ===')
same=sum(1 for g in inr if not math.isnan(g['area_recorded_px']) and abs(g['area_recorded_px']-g['clipped_area_px'])<=1.0)
full=sum(1 for g in inr if not math.isnan(g['area_recorded_px']) and abs(g['area_recorded_px']-g['proj_h_px']*0)<=0)
print('  recorded == clipped (within 1 px^2): %d / %d = %.1f%%'%(same,len(inr),100*same/len(inr)))
unc=[g for g in inr if g['truncation']>0.01]
same_t=sum(1 for g in unc if abs(g['area_recorded_px']-g['clipped_area_px'])<=1.0)
print('  among truncated (>1%%) objects      : %d / %d = %.1f%% match clipped'%(same_t,len(unc),100*same_t/max(1,len(unc))))

print()
print('=== A / B / C CLASSIFICATION, GT WITHIN 40 m ===')
for rule in ('A','B','C'):
    c=Counter(g[rule] for g in inr)
    ev,ig=c['EVALUABLE'],c['IGNORE']
    print('  rule %s: EVALUABLE=%6d  IGNORE=%6d   (evaluable %.1f%% of within-40m)'%(rule,ev,ig,100*ev/max(1,ev+ig)))
    for cls in ('vehicle','person'):
        cc=Counter(g[rule] for g in inr if g['label']==cls)
        print('        %-8s EVALUABLE=%6d  IGNORE=%6d'%(cls,cc['EVALUABLE'],cc['IGNORE']))
