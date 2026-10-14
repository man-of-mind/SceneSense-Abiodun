"""Render two PNGs and audit matched training schedules from this study's outputs."""
from __future__ import annotations
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rl_agent.splitfusion_quality_history_study_v1.study import verify, sha, read_json, write_json


def main():
    verify()
    out = OUT / 'presentation_01'
    out.mkdir(exist_ok=False)
    e = read_json(OUT/'eval_01/EVALUATION.json')
    labels = ('CONTROL', 'CANDIDATE', 'DEPLOYED')
    colors = ('#5379a3', '#c5682b', '#507c58')
    profiles = ('all', 'FAVORABLE_STABLE', 'MID_VARIABLE', 'ADVERSE_STABLE', 'FADE_RECOVERY')
    names = ('All', 'Favourable', 'Mid-variable', 'Adverse', 'Fade/recovery')
    eligible = {p: dict(vehicle=0., person=0.) for p in profiles}
    with gzip.open(OUT/'eval_01/EVAL_DECISIONS.jsonl.gz','rt') as f:
        for line in f:
            r = json.loads(line)
            if r['policy'] == 'CONTROL' and r['setting'] == 'E1_CORRECTED':
                for p in ('all', r['profile']):
                    eligible[p]['vehicle'] += r['v_eligible']
                    eligible[p]['person'] += r['p_eligible']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(5)
    for ax, key, title, scale in zip(axes.flat,
            ('P', 'timely', 'wire_p50', 'wire_p99'),
            ('Timely within-tolerance localisation objective P',
             'Modeled feedback receipt ≤170 ms / all decisions',
             'Payload P50 (kB; decimal)', 'Payload P99 (kB; decimal)'), (1, 100, .001, .001)):
        for j, (label, color) in enumerate(zip(labels, colors)):
            values = [e['results'][label]['E1_CORRECTED'][p][key] * scale for p in profiles]
            ax.bar(x+(j-1)*.25, values, width=.23, label=label.title(), color=color)
        ax.set_xticks(x, names, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        if key == 'timely': ax.set_ylabel('% of decision frames')
    axes[0, 0].legend(fontsize=9)
    ns = [e['results']['CONTROL']['E1_CORRECTED'][p]['n'] for p in profiles]
    fig.suptitle('One seed-17 pair vs frozen deployed actor — E1-CORRECTED\n'
                 'Previously inspected held_scene / HELDOUT_GCV2; 3 evaluation schedules', fontsize=13)
    fig.text(.5, -.035, 'n per policy: ' + ', '.join(f'{n}: {v:,}' for n,v in zip(names,ns)) +
        '\nP = .4 timely vehicle hits / ALL eligible vehicles + .6 timely person hits / ALL eligible persons; '
        'tolerances 1.0 / 1.2 m.\nMisses and late feedback remain in denominators. '
        'Feedback is not prediction-completion latency. Payload denominator: all decisions.\n'
        f"All-profile eligible GT: {eligible['all']['vehicle']:,.0f} vehicles; {eligible['all']['person']:,.0f} persons (same for all policies).",
        ha='center', va='top', fontsize=9)
    fig.savefig(out/'01_profile_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)

    metrics = (('vehicle','loc'), ('person','loc'), ('vehicle','seg'), ('person','seg'),
               ('vehicle','precision'), ('person','precision'))
    floors = (.95, .90, .85, .80, .95, None)
    ticklabels = ('Vehicle\nlocalisation', 'Person\nlocalisation', 'Vehicle\nsegmentation',
                  'Person\nsegmentation', 'Vehicle\nprecision', 'Person\nprecision')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, profile in zip(axes, ('all', 'FAVORABLE_STABLE')):
        x = np.arange(6)
        for j, (label, color) in enumerate(zip(labels, colors)):
            s = e['results'][label]['E1_CORRECTED'][profile]
            ax.bar(x+(j-1)*.24, [s['retention'][cl][k] for cl,k in metrics], width=.22,
                   color=color,label=label.title())
        for i, floor in enumerate(floors):
            if floor is not None:
                ax.plot([i-.38,i+.38],[floor,floor],color='#a62828',lw=2,
                        label='Absolute floor' if i==0 else None)
        ax.set_xticks(x,ticklabels,fontsize=8)
        ax.set_ylabel('Retention ratio (dimensionless)')
        ax.set_title(('All profiles' if profile=='all' else 'Favourable only') + f"; n={s['n']:,}/policy")
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8,ncol=2)
    fig.suptitle('All-scored quality retention — absolute floors cannot be waived', fontsize=13)
    fig.text(.5,-.045,'Denominator: same-input noAE/UINT8/q0 reference metric, from the same eligible GT. '
        'Localisation: all eligible objects, including misses.\nSegmentation: pooled intersection/union; '
        'precision: pooled TP/(TP+FP). Person-precision gate is parity −.005 vs both baselines.\n'
        'Previously inspected offline evidence, not live candidate quality.',ha='center',va='top',fontsize=9)
    fig.savefig(out/'02_quality_retention.png',dpi=160,bbox_inches='tight');plt.close(fig)

    audit = collections.Counter()
    schedule_digest = hashlib.sha256()
    streams = [gzip.open(OUT/'runs_01'/f'{arm}_seed17/decisions.jsonl.gz','rt')
               for arm in ('CONTROL','CANDIDATE')]
    try:
        import itertools
        for ca, cb in itertools.zip_longest(*streams):
            assert ca is not None and cb is not None
            a,b = json.loads(ca),json.loads(cb)
            keys=('ordinal','reward_scene_key','prior_ul_mcs','snr_db','profile','camera_si','radar_p40')
            assert all(a[k]==b[k] for k in keys), 'training exogenous schedules differ'
            schedule_digest.update(json.dumps([a[k] for k in keys]).encode())
            audit['matched_training_decisions'] += 1
            audit['candidate_' + b['terminal']] += 1
            audit['control_' + a['terminal']] += 1
            if b['legitimate_undefined']:
                assert b['excluded'] and b['reward'] is None and b['history_retained']
                assert b['candidate_next_state'][19] == 1 and b['candidate_next_state'][22] == 0
                assert abs(b['candidate_next_state'][23]*170-b['receipt_ms']) < 1e-9
                audit['valid_unscored_receipts_retained'] += 1
            if a['source']=='STRATIFIED_WARMUP':
                assert (a['mode_id'],a['q_e4'])==(b['mode_id'],b['q_e4'])
                audit['matched_warmup_actions'] += 1
    finally:
        for f in streams:f.close()
    write_json(out/'AUDIT.json',dict(counts=dict(audit),training_schedule_sha256=schedule_digest.hexdigest(),
        all_frozen_inputs_unchanged=True,script_sha256=sha(Path(__file__)),
        eligible_gt_by_profile=eligible,
        evaluation_sha256=sha(OUT/'eval_01/EVALUATION.json'),
        figures={p.name:sha(p) for p in out.glob('*.png')}))
    print(json.dumps(dict(audit),sort_keys=True))


if __name__=='__main__':main()
