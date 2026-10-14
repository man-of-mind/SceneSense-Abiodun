import json,sys,statistics as st,collections
def ax(a,b):
    d=((b-a)%360+540)%360-180;return abs(d-180 if d>90 else d+180 if d<-90 else d)
def ev(path):
    rows=[json.loads(l) for l in open(path)]
    out={}
    for nm,f in [('all',lambda r:True),('est-stationary',lambda r:r['tier']=='established' and r['motion']!='moving'),('est-moving',lambda r:r['tier']=='established' and r['motion']=='moving'),('tentative',lambda r:r['tier']=='tentative'),('conflict',lambda r:r['conflict'])]:
        sel=[r for r in rows if f(r)];g=[r for r in sel if r['gt']];o=[r for r in g if r['yaw'] is not None]
        e=[ax(r['yaw'],r['gt']['yaw']) for r in o]
        out[nm]=dict(n=len(sel),gt=len(g),oriented_frac=round(len(o)/max(1,len(g)),3),
          diagonal_gt20=sum(x>20 for x in e),diagonal_share_of_matched=round(sum(x>20 for x in e)/max(1,len(g)),3),
          err_p50=round(st.median(e),1) if e else None,err_p90=round(sorted(e)[int(.9*len(e))],1) if e else None)
    # stability: total abs change of drawn axis per stationary track-second
    by=collections.defaultdict(list)
    for r in rows: by[r['key']].append(r)
    tv=0;dur=0;flips=0
    for k,a in by.items():
        for p,q in zip(a,a[1:]):
            if p['motion']!='moving' and q['motion']!='moving' and q['ms']-p['ms']<400:
                dur+=(q['ms']-p['ms'])/1000
                if p['yaw'] is not None and q['yaw'] is not None: tv+=ax(p['yaw'],q['yaw'])
                if (p['yaw'] is None)!=(q['yaw'] is None): flips+=1
    out['stationary_axis_change_deg_per_s']=round(tv/dur,3);out['stationary_neutral_oriented_switches_per_min']=round(flips/dur*60,2)
    out['sources']=collections.Counter(r['dsrc'] for r in rows if r['tier']=='established').most_common(8)
    return out
for p in sys.argv[1:]:
    print(p);[print(' ',k,v) for k,v in ev(p).items()]
