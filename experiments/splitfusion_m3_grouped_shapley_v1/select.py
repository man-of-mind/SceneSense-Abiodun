"""Freeze target/background identities without computing new attributions."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from common import ROOT,BASE,old,write,load_records,state_id,record_document

def main(out):
    out.mkdir(exist_ok=False)
    runs,records,inputs=load_records()
    cases_path=ROOT/'experiments/splitfusion_decision_cases_v1/results_01/SELECTION.json'
    cases=json.loads(cases_path.read_text());inputs[str(cases_path)]=old.sha(cases_path)
    selected={};backgrounds={'primary':[],'alternative':[]};transitions={}
    for ver,rr in runs.items():
        reps=np.linspace(0,len(rr)-1,100).astype(int).tolist();assert len(set(reps))==100
        changes=[i for i in range(1,len(rr)) if rr[i]['condition']!=rr[i-1]['condition']]
        assert len(changes)==(1 if ver in (13,14) else 0)
        k=changes[0] if changes else None;transitions[str(ver)]=k
        windows=list(range(k-20,k+40)) if k is not None else []
        ci=[c['index'] for c in cases if c['run']==ver]
        selected[str(ver)]=dict(representative=reps,transition_window=windows,illustrative=ci,
            target_union=sorted(set(reps+windows+ci)))
        if ver in (11,12):
            backgrounds['primary'] += [rr[int(f*(len(rr)-1))] for f in (.25,.75)]
            backgrounds['alternative'] += [rr[int(f*(len(rr)-1))] for f in (.125,.625)]
        else:
            for phase in ('GOOD','BAD'):
                pool=[r for r in rr if r['condition']==phase]
                backgrounds['primary'].append(pool[int(.5*(len(pool)-1))])
                f=.25 if ver==13 else .75
                backgrounds['alternative'].append(pool[int(f*(len(pool)-1))])
    for bg in backgrounds.values():
        assert collections.Counter(r['run'] for r in bg)=={11:2,12:2,13:2,14:2}
        assert collections.Counter(r['condition'] for r in bg)=={'GOOD':4,'BAD':4}
    used=set()
    for ver,sel in selected.items():
        for i in sel['target_union']:
            used.add((int(ver),i))
            if i:used.add((int(ver),i-1))
    for bg in backgrounds.values():used.update((r['run'],r['index']) for r in bg)
    doc=dict(schema='splitfusion.grouped_shapley.selection.v1',selection=selected,transitions=transitions,cases=cases,
        backgrounds={k:[dict(record_document(r),weight=1/8) for r in v] for k,v in backgrounds.items()},
        endpoint_ids=[state_id(runs[v][i]) for v,i in sorted(used)],
        inputs_sha256=inputs,plan_sha256=old.sha(BASE/'PLAN.md'),selection_source_sha256=old.sha(__file__),
        common_source_sha256=old.sha(BASE/'common.py'),recorded_states=len(records),
        history_counts={str(v):dict(collections.Counter(r['history'] for r in rr)) for v,rr in runs.items()})
    write(out/'SELECTION.json',doc)
    write(out/'RECORDED_STATES.json',[record_document(r) for r in records])
    print(json.dumps(dict(targets={v:len(s['target_union']) for v,s in selected.items()},
        endpoints=len(used),backgrounds={k:[(r['run'],r['index'],r['condition'],r['history']) for r in bg] for k,bg in backgrounds.items()},
        history_counts=doc['history_counts']),indent=1))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
