"""Aggregate the ONE saved evaluation; no actor queries or training.

The frozen driver stopped after saving all rows: exact float equality of
interpolated eligible counts fails by <=1.78e-15. Establish the authoritative
integer GT count from every grid cell before canonicalizing ONLY that count.
All original raw rows, sources, checkpoints and acceptance gates are preserved.
"""
import collections
import csv
import gzip
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from rl_agent.splitfusion_quality_history_study_v1 import study as S
from rl_agent.splitfusion_run5_reward_trials_v1.catalog import TrialSceneCatalogV1


def main():
    S.verify();start=time.monotonic()
    out=S.OUT/'eval_01'
    cat=TrialSceneCatalogV1('held_scene','C0')
    eligible={}
    for key,scene in cat._scenes.items():
        eligible[key]={}
        for cl in ('vehicle','person'):
            values={v[cl]['eligible'] for v in scene['components'].values()}
            assert len(values)==1,'actual action-dependent GT eligibility: refuse aggregation'
            eligible[key][cl[0]]=next(iter(values))
    raw=collections.defaultdict(dict)
    differences=collections.Counter();max_error=0.
    original_totals=collections.defaultdict(lambda:[0.,0.])
    with gzip.open(out/'EVAL_DECISIONS.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            key=(r['policy'],r['setting']);seed=r['eval_seed']
            expected=S.RE.expected(cat.components(r['scene'],mode_id=r['mode'],q_e4=r['q']))
            # Verify the stored arithmetic exactly against the pinned catalog;
            # no broad tolerance may conceal a real eligibility discrepancy.
            for i,cl in enumerate(('v','p')):
                field=cl+'_eligible';n=eligible[r['scene']][cl]
                assert r[field]==expected[field]
                assert r['ref_'+field]==n
                original_totals[key][i]+=r[field]
                error=abs(r[field]-n);max_error=max(max_error,error)
                if error: differences[field]+=1
                r[field]=n
            raw[key].setdefault(seed,[]).append(r)
    for per in raw.values():
        assert set(per)==set(S.SEEDS)
        assert all(len(rows)==6000 and [r['i'] for r in rows]==list(range(6000)) for rows in per.values())
    assert len(raw)==9
    policies={a:S.read_json(S.OUT/'runs_01'/f'{a}_seed17'/'RUN_RESULT.json')['actor'] for a in ('CONTROL','CANDIDATE')}
    policies['DEPLOYED']={k:S.LREG.DEPLOYED[k] for k in ('path','sha256')}
    for setting in S.SETTINGS:
        for seed in S.SEEDS:
            for label in policies:
                a,b=raw[label,setting][seed],raw['CONTROL',setting][seed]
                for k in ('scene','mcs','snr','profile','v_eligible','p_eligible'):
                    assert [r[k] for r in a]==[r[k] for r in b],('real schedule/eligibility mismatch',label,setting,seed,k)
    select={'all':lambda r:True,**{p:lambda r,p=p:r['profile']==p for p in S.PROFILES},
            'mcs_le10':lambda r:r['mcs']<=10,'after_undefined':lambda r:r['after_undefined']}
    pooled={key:[r for seed in S.SEEDS for r in per[seed]] for key,per in raw.items()}
    results={label:{setting:{p:S.LE.stratum([r for r in pooled[label,setting] if fn(r)])
        for p,fn in select.items()} for setting in S.SETTINGS} for label in policies}
    pairs={ref:{setting:{p:S.LE.paired(pooled['CANDIDATE',setting],pooled[ref,setting],
             lambda a,b,fn=fn:fn(a)) for p,fn in select.items() if p!='after_undefined'}
             for setting in S.SETTINGS} for ref in ('CONTROL','DEPLOYED')}
    mech={label:{setting:dict(undefined=sum(r['legitimate_undefined'] for r in pooled[label,setting]),
        successors=sum(r['after_undefined'] for r in pooled[label,setting]),
        retained=sum(r['after_undefined'] and r['prev_present']==1 for r in pooled[label,setting]),
        poor_after_undefined=S.LE.stratum([r for r in pooled[label,setting] if r['after_undefined'] and
            r['profile']=='ADVERSE_STABLE'])) for setting in S.SETTINGS} for label in policies}
    primary_roundoff={}
    for (label,setting),rows in pooled.items():
        totals=original_totals[label,setting]
        old=sum(w*sum(r[cl+'_hits']*r['timely'] for r in rows)/den
                for cl,w,den in zip(('v','p'),(.4,.6),totals))
        primary_roundoff[label+'|'+setting]=results[label][setting]['all']['P']-old
    note=dict(reason='strict float equality after interpolation, not GT/schedule drift',
        original_error="AssertionError: ('common schedule drift', 'CANDIDATE', 'E1_CORRECTED', 9017, 'v_eligible')",
        example=dict(candidate=3.,control=2.9999999999999996),max_per_row_abs_error=max_error,
        affected_raw_values=dict(differences),p_delta_from_integer_denominators=primary_roundoff,
        proof='all 256 scene grids have action-independent integer GT counts; every raw interpolated count reproduced exactly',
        raw_rows_unchanged=True,no_new_actor_queries=True,training_or_evaluation_repeated=False,
        source_sha256=S.sha(Path(__file__)),raw_sha256=S.sha(out/'EVAL_DECISIONS.jsonl.gz'))
    S.write_json(out/'AGGREGATION_NOTE.json',note)
    result=dict(policies=policies,results=results,paired=pairs,mechanism=mech,decision=S.decision(results,pairs),
        identical_scene_channel_eligibility=True,decisions_per_policy_setting=18000,
        eval_seeds=S.SEEDS,aggregation_seconds=time.monotonic()-start,
        evidence='previously inspected offline modeled evidence',freeze_sha256=S.sha(S.OUT/'FREEZE.json'),
        prediction_completion_latency_measured=False,aggregation_note=note)
    S.write_json(out/'EVALUATION.json',result)
    with (out/'RESULTS_TABLE.csv').open('x') as f:
        fields=['policy','setting','profile','n','P','timely','wire_p50','wire_p99','loc_v','loc_p','seg_v','seg_p','precision_v','precision_p']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for label,settings in results.items():
            for setting,strata in settings.items():
                for profile,s in strata.items():
                    if not s['n']:continue
                    row=dict(policy=label,setting=setting,profile=profile,**{k:s[k] for k in ('n','P','timely','wire_p50','wire_p99')})
                    row.update({f'{k}_{cl[0]}':s['retention'][cl][k] for k in ('loc','seg','precision') for cl in S.RE.CLASSES})
                    w.writerow(row)
    print(json.dumps(result['decision']),flush=True)


if __name__=='__main__':main()
