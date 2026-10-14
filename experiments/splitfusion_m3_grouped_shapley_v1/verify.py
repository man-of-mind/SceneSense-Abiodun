"""Focused artifact checks only: no actor queries or runtime qualification."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from common import BASE,ROOT,old,GROUPS,GN,write,csvout

def main(base):
    out=base/'checks_01';out.mkdir(exist_ok=False)
    sel=json.loads((base/'selection_01/SELECTION.json').read_text())
    states=json.loads((base/'selection_01/RECORDED_STATES.json').read_text());ids={r['id']:r for r in states}
    result=json.loads((base/'attribution_01/RESULTS.json').read_text())
    for p,h in result['inputs_sha256'].items():assert old.sha(p)==h,p
    assert old.sha(base/'select.py')==sel['selection_source_sha256']
    assert old.sha(base/'common.py')==sel['common_source_sha256']
    for n,h in result['outputs_sha256'].items():assert old.sha(base/'attribution_01'/n)==h,n
    for bg in sel['backgrounds'].values():
        assert Counter(b['run'] for b in bg)=={11:2,12:2,13:2,14:2}
        assert Counter(b['condition'] for b in bg)=={'GOOD':4,'BAD':4}
        assert all(b['features']==ids[b['id']]['features'] and b['weight']==1/8 for b in bg)
    for v in (11,12,13,14):
        rr=[r for r in states if r['run']==v]
        assert sel['selection'][str(v)]['representative']==np.linspace(0,len(rr)-1,100).astype(int).tolist()
    with gzip.open(base/'attribution_01/QUERY_CACHE.jsonl.gz','rt') as f:
        cache={tuple(c['features']):c for c in map(json.loads,f)}
    assert len(cache)==result['queries'] and all(old.valid(v) for v in cache)
    for key in sel['endpoint_ids']:
        r=ids[key];y=cache[tuple(r['features'])]['outputs']
        assert int(np.argmax(y[:12]))==r['mode']
        assert int(round(y[13+r['mode']]*10000))==r['q_e4']
    pairs={};err=0.;case_count=0
    previous=json.loads((ROOT/'experiments/splitfusion_decision_cases_v1/results_01/ATTRIBUTIONS.json').read_text())['cases']
    with gzip.open(base/'attribution_01/PAIR_COALITIONS.jsonl.gz','rt') as f:
        for line in f:
            p=json.loads(line);a=ids[p['reference']]['features'];b=ids[p['target']]['features']
            assert len(p['coalitions'])==16
            for mask,c in enumerate(p['coalitions']):
                expected=list(a)
                for g,ix in enumerate(GROUPS.values()):
                    if mask&(1<<g):
                        for j in ix:expected[j]=b[j]
                assert c['mask']==mask and c['features']==expected
                cached=cache[tuple(expected)]
                assert c['outputs']==cached['outputs'] and c['near']==cached['near'] and c['nearest']==cached['nearest']
            vals=np.asarray([c['outputs'] for c in p['coalitions']]);m=p['selected_mode']
            margin=np.log(vals[:,m]/np.max(np.delete(vals[:,:12],m,axis=1),axis=1))
            outputs=np.column_stack((vals[:,12],vals[:,24],margin,vals[:,13+m]))
            ph=np.zeros((4,4))
            for g in range(4):
                for mask in range(16):
                    if not mask&(1<<g):
                        k=mask.bit_count();ph[g]+=math.factorial(k)*math.factorial(3-k)/24*(outputs[mask|1<<g]-outputs[mask])
            assert np.allclose(ph,p['selected_output_phi'],rtol=0,atol=1e-12)
            err=max(err,float(np.max(abs(ph.sum(0)-(outputs[15]-outputs[0])))))
            far=sum(not c['near'] for c in p['coalitions'][1:15]);assert far==p['nonendpoint_far']
            if p['reused_case']:
                prior=previous[p['reused_case']]
                for field in ('coalitions','phi','reconstruction_max_error','nonendpoint_far'):assert p[field]==prior[field]
                case_count+=1
            pairs[p['pair_id']]=(ph,outputs[0],outputs[15],far)
    with gzip.open(base/'attribution_01/ATTRIBUTIONS.json.gz','rt') as f:rows=json.load(f)
    for r in rows:
        ps=[pairs[i] for i in r['pair_ids']]
        ph=np.mean([p[0] for p in ps],axis=0);ref=np.mean([p[1] for p in ps],axis=0)
        for j,metric in enumerate(result['outputs']):
            assert abs(r[metric+'_reference']-ref[j])<1e-12 and abs(r[metric+'_target']-ps[0][2][j])<1e-12
            assert max(abs(r[metric+'_'+g]-ph[k,j]) for k,g in enumerate(GN))<1e-12
        if r['question'].startswith('common_'):
            bg=r['question'].removeprefix('common_')
            assert r['reference_ids']==[b['id'] for b in sel['backgrounds'][bg]]
        else:assert r['reference_ids']==[f"V{r['run']}:{r['index']-1}"]
    summary=[]
    for s in result['summary']:
        rs=[r for r in rows if r['run']==s['run'] and r['question']==s['question'] and r['representative']]
        assert len(rs)==s['n']==(99 if s['question']=='adjacent' else 100)
        for g in GN:assert abs(np.mean([abs(r[s['output']+'_'+g]) for r in rs])-s['mean_absolute'][g])<1e-12
        summary.append(dict(run=s['run'],question=s['question'],output=s['output'],n=s['n'],
            **{'mean_abs_'+g:s['mean_absolute'][g] for g in GN},
            **{'mean_signed_'+g:s['mean_signed'][g] for g in GN},
            reference_mean=s['reference_mean'],target_mean=s['target_mean'],far_slots=s['far_slots'],
            nonendpoint_slots=s['slots'],all_near_targets=s['all_near_targets']))
    csvout(out/'SUMMARY.csv',summary)
    findings={'runs':{},'backgrounds':{k:[dict(id=b['id'],frame=b['frame'],condition=b['condition'],history=b['history'],backlog_bytes=b['backlog_bytes']) for b in bg] for k,bg in sel['backgrounds'].items()},'cases':[]}
    for v in (11,12,13,14):
        rr=[r for r in states if r['run']==v];rep=[r for r in rows if r['run']==v and r['question']=='common_primary' and r['representative']]
        findings['runs'][v]=dict(n=len(rr),modes=dict(Counter(r['mode'] for r in rr)),
            representative_modes=dict(Counter(r['mode'] for r in rep)),history=dict(Counter(r['history'] for r in rr)),
            explained_non11=sum(r['mode']!=11 for r in rows if r['run']==v and r['question']=='common_primary'),
            representative_q11_range=[min(r['q11_target'] for r in rep),max(r['q11_target'] for r in rep)])
    for c in sel['cases']:
        r=next(r for r in rows if r['question']=='adjacent' and r['run']==c['run'] and r['index']==c['index'])
        findings['cases'].append(dict(case=c['case'],**r))
    write(out/'FINDINGS.json',findings)
    manifest=json.loads((base/'figures_02/OUTPUT_SHA256.json').read_text());assert old.sha(base/'render.py')==manifest['source_sha256']
    for n,h in manifest['png_sha256'].items():
        path=base/'figures_02'/n;assert old.sha(path)==h
        with Image.open(path) as im:assert im.size==(3840,2160) and im.format=='PNG'
    assert len(manifest['png_sha256'])==6 and case_count==16 and len(pairs)==7903 and err<1e-10
    assert len(rows)==1442
    size=sum(p.stat().st_size for p in base.rglob('*') if p.is_file());assert size<500*1024**2
    checks=dict(passed=True,recorded_states=len(states),representative_targets=400,
        attribution_rows=len(rows),pairs_checked=len(pairs),coalition_slots_checked=16*len(pairs),
        valid_unique_query_states=len(cache),exact_prior_case_matches=case_count,
        maximum_reconstruction_error=err,endpoint_coverage=result['endpoint_coverage'],
        fresh_replay=result['replay']['freshly_replayed'],reused_replay=result['replay']['reused_count'],
        input_hashes_match=True,summary_cohorts_separate=True,same_balanced_backgrounds=True,
        png_count=6,png_size=[3840,2160],bytes_before_closeout=size,
        no_actor_queries_in_check=True,source_sha256=old.sha(__file__))
    write(out/'CHECKS.json',checks);print(json.dumps(checks,indent=1))
    print(json.dumps({'runs':findings['runs'],'backgrounds':findings['backgrounds']},indent=1))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=BASE);main(p.parse_args().base.resolve())
