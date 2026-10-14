"""One bounded CPU pass, reusing the sealed query and exact 16-coalition implementation."""
import argparse
import collections
import csv
import gzip
import json
import math
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from common import ROOT,BASE,OLD_PATH,old,GROUPS,GN,write,csvout,extract_function,load_records,state_id

OUTPUTS=['margin11','q11','selected_margin','selected_conditional_q']

def check_hashes(mapping):
    for p,h in mapping.items():assert old.sha(Path(p))==h, 'Changed evidence/source: '+p

def main(selection_path,out):
    started=time.monotonic();out.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    sel=json.loads(selection_path.read_text());check_hashes(sel['inputs_sha256'])
    assert old.sha(BASE/'PLAN.md')==sel['plan_sha256']
    runs,records,inputs=load_records();check_hashes(inputs)
    ids={state_id(r):r for r in records}
    prior_path=ROOT/'experiments/splitfusion_m3_actor_diagnosis_v1/results_02/RESULTS.json'
    prior=json.loads(prior_path.read_text());check_hashes(prior['inputs_sha256'])
    casesdir=ROOT/'experiments/splitfusion_decision_cases_v1/results_01'
    cresults=json.loads((casesdir/'RESULTS.json').read_text());check_hashes(cresults['inputs_sha256'])
    for n in ('ATTRIBUTIONS.json','SELECTION.json','CASES.csv'):
        assert old.sha(casesdir/n)==cresults['outputs_sha256'][n]
    old_selected_path=prior_path.with_name('SELECTED_STATES.csv')
    om=json.loads(prior_path.with_name('OUTPUT_SHA256.json').read_text())
    assert old.sha(old_selected_path)==om['SELECTED_STATES.csv']
    old_selected=list(csv.DictReader(old_selected_path.open()))
    reused={}
    for r in old_selected:
        key=f"V{r['run']}:{r['index']}";now=ids[key]
        assert json.loads(r['features'])==now['v'] and int(r['frame'])==now['frame']
        assert int(r['mode'])==now['mode'] and int(r['q'])==now['q_e4']
        assert prior['replay'][r['run']]['passed'];reused[key]='prior_diagnosis_exact_replay'
    old_cases=json.loads((casesdir/'ATTRIBUTIONS.json').read_text())['cases']
    for c in sel['cases']:
        e=old_cases[c['case']]
        for index,coal in ((c['index']-1,e['coalitions'][0]),(c['index'],e['coalitions'][15])):
            key=f"V{c['run']}:{index}";assert ids[key]['v']==coal['features']
            assert cresults['replay'][str(c['run'])]['passed'];reused[key]='prior_cases_exact_replay'
    endpoint_ids=set(sel['endpoint_ids']);reused={k:v for k,v in reused.items() if k in endpoint_ids}
    fresh={}
    for ver in runs:
        rr=[ids[k] for k in sorted(endpoint_ids) if ids[k]['run']==ver and k not in reused]
        result=old.actor_replay([r['span'] for r in rr],
            [dict(frame_id=r['frame'],mode_id=r['mode'],q_e4=r['q_e4'],kind='POLICY_DECISION') for r in rr],
            loader='rl_agent.splitfusion_run5_full_live_v1.actor:load_registered_actor',method='act_on_vector')
        assert result['passed'];fresh[str(ver)]=result
    replay=dict(endpoint_count=len(endpoint_ids),reused_count=len(reused),reused=reused,
        freshly_replayed=sum(r['replayed'] for r in fresh.values()),fresh=fresh,mismatches=0,
        rule='Exact mode/q/state hash/heads hash; prior proofs reused only after all bound inputs and sources match.')
    assert replay['reused_count']+replay['freshly_replayed']==replay['endpoint_count']
    write(out/'EXACT_REPLAY.json',replay)
    actor=old.load_registered_actor();assert old.WEIGHTS_SHA256=='bb9b0d256de79b4498cd622e1daa59edcf43ff3072dc9f675b265aac579ed3ce'
    X=np.asarray([r['v'] for r in records]);scale=np.asarray(prior['support']['scales']);radius=prior['support']['radius']
    assert len(X)==prior['support']['recorded_reference_n']
    def distance(v):
        parts=[np.mean(((X[:,ix]-np.asarray(v)[ix])/scale[ix])**2,axis=1) for ix in old.CONT.values()]
        d=np.sqrt(np.mean(parts,axis=0));d[np.any(X[:,old.CAT]!=np.asarray(v)[old.CAT],axis=1)]=np.inf
        return d
    cache={}
    for e in old_cases.values():
        for c in e['coalitions']:
            key=tuple(map(float,c['features']));assert old.valid(key)
            if key in cache:assert np.array_equal(cache[key]['y'],np.array(c['outputs']))
            cache[key]=dict(y=np.array(c['outputs']),near=c['near'],nearest=c['nearest'])
    seeded_queries=len(cache)
    env=dict(np=np,torch=torch,math=math,old=old,actor=actor,cache=cache,distance=distance,radius=radius,GROUPS=GROUPS)
    ask=extract_function('ask',env);explain=extract_function('explain',env)
    for key in endpoint_ids:
        r=ids[key];y=ask(r['v'])['y'];assert int(np.argmax(y[:12]))==r['mode']
        assert int(round(y[13+r['mode']]*10000))==r['q_e4']
    pair_cache={};errors=[];max_selected_err=0.;reuse_pairs=0
    # Retain every coalition, including far ones; compression is file storage only.
    pairfile=gzip.open(out/'PAIR_COALITIONS.jsonl.gz','xt')
    def pair(a,b):
        nonlocal max_selected_err,reuse_pairs
        key=(state_id(a),state_id(b))
        if key in pair_cache:return pair_cache[key]
        case=next((c for c in sel['cases'] if c['run']==b['run'] and c['index']==b['index'] and
            a['run']==b['run'] and a['index']==b['index']-1),None)
        if case:
            e=old_cases[case['case']];reuse_pairs+=1
        else:e=explain(a['v'],b['v'])
        vals=np.asarray([c['outputs'] for c in e['coalitions']]);m=b['mode']
        margin=np.log(vals[:,m]/np.max(np.delete(vals[:,:12],m,axis=1),axis=1))
        phm=np.zeros(4)
        for g in range(4):
            for mask in range(16):
                if mask&(1<<g):continue
                k=mask.bit_count();phm[g]+=math.factorial(k)*math.factorial(3-k)/24*(margin[mask|1<<g]-margin[mask])
        err=abs(phm.sum()-(margin[-1]-margin[0]));assert err<1e-10
        max_selected_err=max(max_selected_err,float(err));errors.append(e['reconstruction_max_error'])
        ph=np.asarray(e['phi'])
        pp=np.column_stack((ph[:,12],ph[:,24],phm,ph[:,13+m]))
        yy=np.column_stack((vals[:,12],vals[:,24],margin,vals[:,13+m]))
        pid=len(pair_cache)
        packed=dict(pair_id=pid,reference=key[0],target=key[1],selected_mode=m,
            reused_case=None if not case else case['case'],selected_output_phi=pp.tolist(),outputs=OUTPUTS,**e)
        pairfile.write(json.dumps(packed,separators=(',',':'),allow_nan=False)+'\n')
        ans=dict(pair_id=pid,phi=pp,reference=yy[0],target=yy[-1],far=e['nonendpoint_far'],
            nearest_max=max((c['nearest'] for c in e['coalitions'][1:15] if c['nearest'] is not None),default=0))
        pair_cache[key]=ans
        return ans
    rows=[];details=[]
    def aggregate_target(r,refs,question):
        es=[pair(ref,r) for ref in refs];phi=np.mean([e['phi'] for e in es],axis=0)
        ref=np.mean([e['reference'] for e in es],axis=0);target=es[0]['target']
        er=float(np.max(abs(phi.sum(0)-(target-ref))));assert er<1e-10;errors.append(er)
        vs=sel['selection'][str(r['run'])]
        row=dict(id=state_id(r),run=r['run'],index=r['index'],frame=r['frame'],t_s=r['t_s'],
            question=question,representative=r['index'] in vs['representative'],
            transition_window=r['index'] in vs['transition_window'],illustrative=r['index'] in vs['illustrative'],
            mode=r['mode'],executed_q=r['q_e4']/10000,payload_sfd4_bytes=int(r['row']['payload_l2_bytes']),
            condition=r['condition'],history=r['history'],preceding_outcome=r['preceding_outcome'],
            far_nonendpoint_slots=sum(e['far'] for e in es),nonendpoint_slots=14*len(es),
            all_coalitions_near=all(e['far']==0 for e in es),near_reference_pairs=sum(e['far']==0 for e in es),
            references=len(es),reconstruction_error=er,
            changed_history_indices=json.dumps([j for j in range(4,21) if refs[0]['v'][j]!=r['v'][j]]) if question=='adjacent' else '')
        for j,name in enumerate(OUTPUTS):
            row[name+'_reference']=float(ref[j]);row[name+'_target']=float(target[j])
            for g,gn in enumerate(GN):row[name+'_'+gn]=float(phi[g,j])
        rows.append(row);details.append(dict(**row,pair_ids=[e['pair_id'] for e in es],reference_ids=[state_id(ref) for ref in refs]))
    for ver,rr in runs.items():
        selected=sel['selection'][str(ver)]
        for i in selected['target_union']:
            r=rr[i]
            aggregate_target(r,[ids[b['id']] for b in sel['backgrounds']['primary']],'common_primary')
            if i in selected['representative']:
                aggregate_target(r,[ids[b['id']] for b in sel['backgrounds']['alternative']],'common_alternative')
            if i:aggregate_target(r,[rr[i-1]],'adjacent')
        print('ATTRIBUTED V',ver,'targets',len(selected['target_union']),'queries',len(cache),flush=True)
    pairfile.close();csvout(out/'ATTRIBUTIONS.csv',rows)
    with gzip.open(out/'ATTRIBUTIONS.json.gz','xt') as f:json.dump(details,f,separators=(',',':'),allow_nan=False)
    summaries=[]
    for ver in runs:
        for question in ('common_primary','common_alternative','adjacent'):
            rs=[r for r in rows if r['run']==ver and r['question']==question and r['representative']]
            for metric in ('margin11','q11'):
                summaries.append(dict(run=ver,question=question,output=metric,n=len(rs),
                    mean_absolute={g:float(np.mean([abs(r[metric+'_'+g]) for r in rs])) for g in GN},
                    mean_signed={g:float(np.mean([r[metric+'_'+g] for r in rs])) for g in GN},
                    far_slots=sum(r['far_nonendpoint_slots'] for r in rs),slots=sum(r['nonendpoint_slots'] for r in rs),
                    all_near_targets=sum(r['all_coalitions_near'] for r in rs),
                    reference_mean=float(np.mean([r[metric+'_reference'] for r in rs])),
                    target_mean=float(np.mean([r[metric+'_target'] for r in rs]))))
    with gzip.open(out/'QUERY_CACHE.jsonl.gz','xt') as f:
        for key,ans in cache.items():f.write(json.dumps(dict(features=key,outputs=ans['y'].tolist(),near=ans['near'],nearest=ans['nearest']),separators=(',',':'),allow_nan=False)+'\n')
    for p in (selection_path,BASE/'PLAN.md',BASE/'common.py',Path(__file__),OLD_PATH,prior_path,
        old_selected_path,casesdir/'RESULTS.json',casesdir/'ATTRIBUTIONS.json'):
        inputs[str(p)]=old.sha(p)
    for mod in list(sys.modules.values()):
        p=getattr(mod,'__file__',None)
        if p and str(p).startswith(str(ROOT/'rl_agent')) and str(p).endswith('.py'):inputs[str(Path(p))]=old.sha(Path(p))
    weights=ROOT/'rl_agent/splitfusion_run5_full_live_v1/actor_export/actor_state_dict.pt'
    inputs[str(weights)]=old.sha(weights);assert inputs[str(weights)]==old.WEIGHTS_SHA256
    check_hashes(inputs);assert not torch.cuda.is_initialized()
    support=dict(prior['support'],history_masks_valid_for_all_queries=True,
        far_unique_queries=sum(not c['near'] for c in cache.values()),unique_queries=len(cache))
    result=dict(schema='splitfusion.grouped_shapley.result.v1',outputs=OUTPUTS,groups=GROUPS,
        replay=replay,summary=summaries,support=support,queries=len(cache),seeded_prior_queries=seeded_queries,
        pair_count=len(pair_cache),reused_case_pairs=reuse_pairs,
        maximum_reconstruction_error=max(errors),maximum_selected_margin_error=max_selected_err,
        weights_sha256=old.WEIGHTS_SHA256,inputs_sha256=inputs,cuda_initialized=False,
        seconds=time.monotonic()-started,
        endpoint_coverage=len(endpoint_ids),attribution_target_count=sum(len(s['target_union']) for s in sel['selection'].values()),
        reference_outputs={k:np.mean([ask(b['features'])['y'] for b in bg],axis=0).tolist() for k,bg in sel['backgrounds'].items()},
        outputs_sha256={p.name:old.sha(p) for p in out.iterdir() if p.is_file()})
    assert sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())<500*1024**2
    write(out/'RESULTS.json',result)
    print(json.dumps({k:result[k] for k in ('queries','pair_count','reused_case_pairs','maximum_reconstruction_error',
        'maximum_selected_margin_error','seconds','endpoint_coverage','attribution_target_count','cuda_initialized')},indent=1))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.selection.resolve(),a.out.resolve())
