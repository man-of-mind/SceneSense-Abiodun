"""Bounded study commands: check-freeze, train (one arm), evaluate. CPU only."""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from rl_agent.splitfusion_run5_backlog_pair_v1 import env as E
from rl_agent.splitfusion_run5_full_v1 import campaign as FCAMP, collector as FC
from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import campaign as LCAMP, evaluate as LE, registration as LREG
from rl_agent.splitfusion_run5_reward_trials_v1 import collector as C, evaluate as RE, trainer as TR
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as ORCH
from . import adapter as A

ROOT = LREG.ROOT
PACKAGE = Path(__file__).resolve().parent
OUT = ROOT / 'experiments/splitfusion_quality_history_study_v1'
REFERENCE = ROOT / 'experiments/splitfusion_run5_backlog_pair_v1/20261007_v1/runs_01/CONTROL_seed17'
AUTH = ROOT / 'experiments/splitfusion_run5_lowmcs_retrain_v1/registration_01/REGISTRATION.json'
SETTINGS = {'E1_CORRECTED': ('HELDOUT_GCV2', True), 'E2_CORRECTED': ('RETENTION_V1', True),
            'E1_OLD': ('HELDOUT_GCV2', False)}
SEEDS = (9017, 9029, 9043)
PROFILES = ('FAVORABLE_STABLE', 'MID_VARIABLE', 'ADVERSE_STABLE', 'FADE_RECOVERY')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=1, allow_nan=False)


def make_orchestrator(arm):
    shared, latency = E.components()
    factory = lambda: E.TrialBacklogCollectorV1(latency=latency, seed=17, shared_sources=shared,
                                               backlog_transition=None)
    cls = TR.TrialOrchestratorV1 if arm == 'CONTROL' else A.CandidateOrchestrator
    return cls(collector_factory=factory, seed=17, authorization_sha256=sha(AUTH))


def freeze():
    from . import checks
    assert not (OUT / 'FREEZE.json').exists(), 'study already frozen'
    LREG.verify(AUTH, seed=17)
    assert sha(AUTH) == '1a15ab019f55d479c4d3d403f77b5a7d5994437f7b1c05e4820fb39ce10e1aeb'
    result = checks.run()
    sources = set(PACKAGE.glob('*.py'))
    for mod in list(sys.modules.values()):
        name = getattr(mod, '__file__', None)
        if name and str(name).startswith(str(ROOT / 'rl_agent')) and str(name).endswith('.py'):
            sources.add(Path(name))
    paths = sources | {OUT/'STUDY_PLAN.md', AUTH, REFERENCE/'RUN_RESULT.json',
        REFERENCE/'actor_u10000.pt', REFERENCE/'checkpoints/u00000.pt',
        REFERENCE/'checkpoints/u10000.pt', ROOT/LREG.DEPLOYED['path'], LREG.CANDIDATE,
        LREG.T_V2B, LREG.BINDING_02/'OUTPUT_SHA256.json'}
    # Pin actual quality DB + corrected descriptor sidecar, not just their paths.
    from rl_agent.splitfusion_run5_reward_trials_v1 import catalog as K
    paths |= {ROOT/K.V1.PAYLOAD_AUTHORITY_RELPATH, ROOT/K.V1.PAYLOAD_AUTHORITY_MANIFEST_RELPATH,
              ROOT/K.SS.P40_SIDECAR_RELPATH}
    write_json(OUT/'CHECKS.json', result)
    doc = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        files={str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)},
        checks_sha256=sha(OUT/'CHECKS.json'), disk_free_bytes=shutil.disk_usage(OUT).free,
        torch=torch.__version__, numpy=np.__version__, inherited_registration=read_json(AUTH))
    write_json(OUT/'FREEZE.json', doc)
    print(json.dumps(result), flush=True)


def verify():
    f = read_json(OUT/'FREEZE.json')
    for p, h in f['files'].items():
        assert sha(ROOT/p) == h, 'frozen input drift: ' + p
    assert sha(OUT/'CHECKS.json') == f['checks_sha256']
    LREG.verify(AUTH, seed=17)
    assert sha(ROOT/LREG.DEPLOYED['path']) == LREG.DEPLOYED['sha256']
    assert shutil.disk_usage(OUT).free > 2*1024**3, 'disk safety floor 2 GiB'
    return f


def mechanism(diag, candidate=False):
    idx = [i+1 for i, d in enumerate(diag[:-1]) if d.get('legitimate_undefined',
           d['excluded'] and d['terminal']=='EVALUATOR_FAULT')]
    def stats(rows):
        if not rows:
            return dict(n=0)
        return dict(n=len(rows), mode_share={str(m): sum(d['mode_id']==m for d in rows)/len(rows)
                    for m in range(12)}, q_p50=float(np.median([d['q_e4']/10000 for d in rows])),
                    wire_p50=float(np.median([d['wire_bytes'] for d in rows])),
                    wire_p99=float(np.percentile([d['wire_bytes'] for d in rows],99)),
                    timely=float(np.mean([d['receipt_ms'] is not None and d['receipt_ms']<=170 for d in rows])))
    return dict(undefined=sum(d['excluded'] for d in diag), successors=len(idx),
        retained=sum(diag[i].get('candidate_state',diag[i]['state'])[19]==1 for i in idx),
        next_actions=stats([diag[i] for i in idx]),
        poor_next_actions=stats([diag[i] for i in idx if diag[i]['profile']=='ADVERSE_STABLE']),
        repeated=sum(diag[i]['excluded'] for i in idx))


def train(arm):
    verify()
    assert arm in ('CONTROL','CANDIDATE')
    torch.set_num_threads(4)
    directory = OUT/'runs_01'/f'{arm}_seed17'
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    o = make_orchestrator(arm)
    write_json(directory/'RUN_BINDING.json', dict(o.binding_document, binding_sha256=o.binding_sha256,
        freeze_sha256=sha(OUT/'FREEZE.json'), threads=4, device='cpu'))
    o.warmup_phase()
    initial = o.save(directory/'checkpoints/u00000.pt')
    if arm=='CONTROL':
        assert initial['sha256']==read_json(REFERENCE/'RUN_RESULT.json')['checkpoint_sha256']['0']
    with (directory/'metrics.jsonl').open('x', buffering=1) as metrics:
        while o.update_count < 10000:
            m = o.step_update()
            metrics.write(json.dumps(FCAMP._metrics(o.update_count,m,time.monotonic()-start))+'\n')
            if o.update_count % 1000 == 0:
                print(arm, o.update_count, round(time.monotonic()-start,1), flush=True)
    final = o.save(directory/'checkpoints/u10000.pt')
    actor_path = directory/'actor_u10000.pt'
    torch.save(o.actor.state_dict(),actor_path)
    diag = o.collector.diagnostics
    with gzip.open(directory/'decisions.jsonl.gz','wt') as f:
        for d in diag:
            f.write(json.dumps(d,separators=(',',':'))+'\n')
    result = dict(arm=arm, seed=17, updates=o.update_count, decisions=o.decision_count,
        seconds=time.monotonic()-start, actor=dict(path=str(actor_path.relative_to(ROOT)),sha256=sha(actor_path)),
        checkpoint_sha256={'0':initial['sha256'],'10000':final['sha256']},final_digest=o.state_digest(),
        mechanism=mechanism(diag,arm=='CANDIDATE'), cuda_initialized=torch.cuda.is_initialized(),
        included_replay_rows=o.replay.accepted_count, final_disk_free_bytes=shutil.disk_usage(OUT).free)
    if arm=='CONTROL':
        old=read_json(REFERENCE/'RUN_RESULT.json')
        result['sealed_actor_exact']=result['actor']['sha256']==old['actor']['sha256']
        result['sealed_final_digest_exact']=result['final_digest']==old['final_digest']
        result['sealed_final_bundle_exact']=final['sha256']==old['checkpoint_sha256']['10000']
        assert all(result[k] for k in ('sealed_actor_exact','sealed_final_digest_exact','sealed_final_bundle_exact'))
    assert not result['cuda_initialized']
    write_json(directory/'RUN_RESULT.json',result)
    print(json.dumps(result),flush=True)


def eval_job(args):
    label,spec,setting,seed=args
    torch.set_num_threads(1)
    FC.compose=LE.instrumented_compose  # inherited read-only instrumentation, worker-local
    pool,corrected=SETTINGS[setting]
    shared,latency=LCAMP.components(split='held_scene',pool=pool,corrected=corrected)
    base=C.TrialRun5CollectorV1(latency=latency,seed=seed,shared_sources=shared,arm='C0')
    col=A.HistoryCollector(base) if label=='CANDIDATE' else base
    assert sha(ROOT/spec['path'])==spec['sha256']
    if label=='CANDIDATE':
        actor=A.build_actor(A.model_config(),seed=0)
        actor.load_state_dict(torch.load(ROOT/spec['path'],map_location='cpu',weights_only=True),strict=True)
        actor.eval().requires_grad_(False)
    else:
        actor=RE.load_actor(ROOT/spec['path'])
    cat=shared['catalog']; correction=LCAMP.load_correction(); rows=[]
    previous_undefined=False
    for i in range(6000):
        state=col.current_state_features()
        mode,q=RE.act(actor,state)
        col.collect_transition(ORCH.ModeledActionRequestV1(decision_ordinal=i,mode_id=mode,q_e4=q,
            source='STOCHASTIC_ACTOR',warmup_q_bin_index=None))
        d=col.diagnostics[-1]; key=d['reward_scene_key']; flip=dict(LE.LAST)
        benign=A.legitimate_undefined(d,cat)
        r=dict(i=i,mode=mode,q=q,terminal=d['terminal'],excluded=d['excluded'],receipt=d['receipt_ms'],
            timely=d['receipt_ms'] is not None and d['receipt_ms']<=170,profile=d['profile'],snr=d['snr_db'],
            mcs=d['prior_ul_mcs'],backlog=d['backlog_bytes'],wire=d['wire_bytes'],scene=key,q_perc=d['q_perc'],
            flags=list(correction.flags(backlog_bytes=d['backlog_bytes'],wire_bytes=d['wire_bytes'],
                                       prior_ul_mcs=d['prior_ul_mcs'])),flip_gain=flip['flip_gain'],
            envelope=flip['envelope'],timely_env=flip['envelope']<=170,after_undefined=previous_undefined,
            prev_present=state[19],prev_success=state[20],quality_available=state[22] if label=='CANDIDATE' else state[20],
            unscored_latency=state[23]*170 if label=='CANDIDATE' else None,legitimate_undefined=benign)
        r.update(RE.expected(cat.components(key,mode_id=mode,q_e4=q)))
        r.update({f'ref_{k}':v for k,v in RE.expected([(1.,cat.reference_components(key))]).items()})
        rows.append(r); previous_undefined=benign
    assert not torch.cuda.is_initialized()
    return label,setting,seed,rows


def decision(results,pairs):
    checks={}
    floors={('vehicle','loc'):.95,('person','loc'):.90,('vehicle','seg'):.85,
            ('person','seg'):.80,('vehicle','precision'):.95}
    def gate(name,passed,**detail):
        checks[name]=dict(passed=bool(passed),**detail)
    def get(label,profile='all',setting='E1_CORRECTED'):
        return results[label][setting][profile]
    for profile in ('all','FAVORABLE_STABLE'):
        c=get('CANDIDATE',profile)
        for (cl,k),floor in floors.items():
            value=c['retention'][cl][k]
            gate(f'absolute_{profile}_{cl}_{k}',value is not None and value>=floor,value=value,floor=floor)
    for ref in ('CONTROL','DEPLOYED'):
        c,b=get('CANDIDATE'),get(ref)
        p=pairs[ref]['E1_CORRECTED']['all']
        gate(f'gain_{ref}',p['dP']>=.01 and p['ci95'][0]>0,**p)
        gate(f'person_precision_{ref}',c['retention']['person']['precision']>=b['retention']['person']['precision']-.005)
        c,b=get('CANDIDATE','FAVORABLE_STABLE'),get(ref,'FAVORABLE_STABLE')
        p=pairs[ref]['E1_CORRECTED']['FAVORABLE_STABLE']
        gate(f'favorable_objective_{ref}',p['dP']>=-.005 and p['ci95'][0]>=-.015,**p)
        gate(f'favorable_timeliness_{ref}',c['timely']>=b['timely']-.01)
        for cl in RE.CLASSES:
            for k in ('loc','seg','precision'):
                gate(f'favorable_{ref}_{cl}_{k}',c['retention'][cl][k]>=b['retention'][cl][k]-.02)
        for setting,bar,inclusive in (('E2_CORRECTED',0,False),('E1_OLD',-.01,True)):
            p=pairs[ref][setting]['all']
            gate(f'sensitivity_{setting}_{ref}',p['dP']>=bar if inclusive else p['dP']>bar,**p)
        c,b=get('CANDIDATE','ADVERSE_STABLE'),get(ref,'ADVERSE_STABLE')
        gate(f'poor_objective_{ref}',c['P']>=b['P'],delta=c['P']-b['P'])
        gate(f'poor_timeliness_{ref}',c['timely']>=b['timely']-.01,delta=c['timely']-b['timely'])
    return dict(verdict='QUALIFIES_FOR_FURTHER_REVIEW' if all(v['passed'] for v in checks.values())
                else 'NO_QUALIFYING_BENEFIT',checks=checks)


def evaluate(workers):
    verify(); start=time.monotonic()
    directory=OUT/'eval_01'; directory.mkdir(exist_ok=False)
    policies={a:read_json(OUT/'runs_01'/f'{a}_seed17'/'RUN_RESULT.json')['actor'] for a in ('CONTROL','CANDIDATE')}
    policies['DEPLOYED']={k:LREG.DEPLOYED[k] for k in ('path','sha256')}
    raw=collections.defaultdict(dict)
    jobs=[(label,spec,setting,s) for label,spec in policies.items() for setting in SETTINGS for s in SEEDS]
    with mp.get_context('spawn').Pool(workers) as pool:
        for label,setting,seed,rows in pool.imap_unordered(eval_job,jobs):
            raw[label,setting][seed]=rows
            print('evaluated',label,setting,seed,flush=True)
    with gzip.open(directory/'EVAL_DECISIONS.jsonl.gz','wt') as f:
        for (label,setting),per in sorted(raw.items()):
            for seed,rows in sorted(per.items()):
                for r in rows:
                    f.write(json.dumps(dict(r,policy=label,setting=setting,eval_seed=seed),separators=(',',':'))+'\n')
    for setting in SETTINGS:
        for seed in SEEDS:
            for label in policies:
                a,b=raw[label,setting][seed],raw['CONTROL',setting][seed]
                for k in ('scene','mcs','snr','profile','v_eligible','p_eligible'):
                    assert [r[k] for r in a]==[r[k] for r in b],('common schedule drift',label,setting,seed,k)
    select={'all':lambda r:True,**{p:lambda r,p=p:r['profile']==p for p in PROFILES},
            'mcs_le10':lambda r:r['mcs']<=10,'after_undefined':lambda r:r['after_undefined']}
    pooled={key:[r for s in SEEDS for r in per[s]] for key,per in raw.items()}
    results={label:{setting:{p:LE.stratum([r for r in pooled[label,setting] if fn(r)])
        for p,fn in select.items()} for setting in SETTINGS} for label in policies}
    pairs={ref:{setting:{p:LE.paired(pooled['CANDIDATE',setting],pooled[ref,setting],
             lambda a,b,fn=fn:fn(a)) for p,fn in select.items() if p!='after_undefined'}
             for setting in SETTINGS} for ref in ('CONTROL','DEPLOYED')}
    mech={label:{setting:dict(undefined=sum(r['legitimate_undefined'] for r in pooled[label,setting]),
        successors=sum(r['after_undefined'] for r in pooled[label,setting]),
        retained=sum(r['after_undefined'] and r['prev_present']==1 for r in pooled[label,setting]),
        poor_after_undefined=LE.stratum([r for r in pooled[label,setting] if r['after_undefined'] and
            r['profile']=='ADVERSE_STABLE'])) for setting in SETTINGS} for label in policies}
    result=dict(policies=policies,results=results,paired=pairs,mechanism=mech,decision=decision(results,pairs),
        identical_scene_channel_eligibility=True,decisions_per_policy_setting=18000,
        eval_seeds=SEEDS,seconds=time.monotonic()-start,evidence='previously inspected offline modeled evidence',
        freeze_sha256=sha(OUT/'FREEZE.json'),prediction_completion_latency_measured=False)
    write_json(directory/'EVALUATION.json',result)
    with (directory/'RESULTS_TABLE.csv').open('x') as f:
        fields=['policy','setting','profile','n','P','timely','wire_p50','wire_p99','loc_v','loc_p','seg_v','seg_p','precision_v','precision_p']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for label,settings in results.items():
            for setting,strata in settings.items():
                for profile,s in strata.items():
                    if not s['n']:continue
                    row=dict(policy=label,setting=setting,profile=profile,**{k:s[k] for k in ('n','P','timely','wire_p50','wire_p99')})
                    row.update({f'{k}_{cl[0]}':s['retention'][cl][k] for k in ('loc','seg','precision') for cl in RE.CLASSES})
                    w.writerow(row)
    print(json.dumps(result['decision']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('check-freeze','train','evaluate'))
    p.add_argument('--arm',choices=('CONTROL','CANDIDATE'))
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    if a.command=='check-freeze':freeze()
    elif a.command=='train':train(a.arm)
    else:evaluate(a.workers)
