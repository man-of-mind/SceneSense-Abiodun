import json,collections,math,sys
def ax(a,b):
    d=((b-a)%360+540)%360-180;return abs(d-180 if d>90 else d+180 if d<-90 else d)
def axm(ys):
    sx=sum(math.cos(math.radians(2*y)) for y in ys);sy=sum(math.sin(math.radians(2*y)) for y in ys);return math.degrees(math.atan2(sy,sx))/2,math.hypot(sx,sy)/len(ys)
rows=[json.loads(l) for l in open(sys.argv[1])]
ids=collections.defaultdict(set)
for r in rows:
    if r['gt']: ids[r['key']].add(str(r['gt']['id']))  if r['gt']['id']!='static' else None
base=[r for r in rows if r['tier']=='established' and r['motion']!='moving' and r['gt']]
bad=[r for r in base if r['yaw'] is not None and ax(r['yaw'],r['gt']['yaw'])>20]
cat=collections.Counter()
for r in bad:
    g=r['gt']['yaw']
    if r['road'] and ax(g,r['road']['yaw'])>20: c='E legitimate angle (GT off lane tangent)'
    elif r['gt']['speed']>.5: c='D GT moving/turning, display stationary'
    else:
        raws=r['raw']
        if len(raws)>=2:
            m,R=axm(raws)
            if ax(m,g)<=20: c='A held stale/initial axis; recent raw mean was right'
            elif R>=.8: c='B persistent producer bias (consistent wrong raw axis)'
            else: c='C producer jitter (no consistent raw axis)'
        elif len(raws)==1: c='A held stale/initial axis; recent raw mean was right' if ax(raws[0],g)<=20 else 'B persistent producer bias (consistent wrong raw axis)'
        else: c='F no recent qualified yaw; held axis'
    cat[c]+=1
n=len(bad);out=dict(samples=len(base),diagonal=n,categories={k:v for k,v in sorted(cat.items())},
  multi_gt_identity=sum(1 for r in bad if len(ids[r['key']])>1),
  first_axis_still_shown=sum(1 for r in bad if r.get('firstYaw') is not None and r.get('assocYaw',r['yaw']) is not None and ax(r.get('assocYaw',r['yaw']),r['firstYaw'])<3))
print(json.dumps(out,indent=1))
