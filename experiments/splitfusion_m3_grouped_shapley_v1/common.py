"""Small read-only adapter to retained actor parsing, validation and attribution code."""
import ast
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('MPLCONFIGDIR',str(BASE/'.mpl'))
OLD_PATH=ROOT/'experiments/splitfusion_decision_cases_v1/explain.py'
spec=importlib.util.spec_from_file_location('retained_case_explain',OLD_PATH)
prior_code=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior_code)
old=prior_code.old
GROUPS=old.GROUPS
GN=list(GROUPS)

def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=1,allow_nan=False)

def csvout(path,rows):
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def extract_function(name,namespace):
    """Execute the unchanged nested query/Shapley definitions with explicit local inputs."""
    tree=ast.parse(OLD_PATH.read_text())
    funcs=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(funcs)==1
    code=compile(ast.fix_missing_locations(ast.Module(body=funcs,type_ignores=[])),str(OLD_PATH),'exec')
    exec(code,namespace)
    return namespace[name]

def load_records():
    deploy=ROOT.parent/'abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1'
    runs={};inputs={};allrows=[]
    for ver,name in old.NAMES.items():
        table=ROOT/f'experiments/splitfusion_m3_route_b_v{ver}/analysis_01/DECISION_TIMELINE.csv'
        cells=list((deploy/f'20261009_m3_route_b_v{ver}'/name/'capture/cells').iterdir());assert len(cells)==1
        profile=cells[0]/'profiler/ue.profile.jsonl';spans=old.load_spans(cells[0])
        for p in (table,profile):inputs[str(p)]=old.sha(p)
        rows=list(csv.DictReader(table.open()));rr=[]
        for i,r in enumerate(rows):
            fid=int(r['frame_id']);s=spans[fid];v=s['features']
            assert old.valid(v) and len(v)==22
            assert (s['decided_mode_id'],s['decided_q_e4'])==(int(r['mode_id']),int(r['q_e4']))
            assert v[2]==int(r['prior_ul_mcs'])/28
            assert v[21]==(float(r['snr_proxy_db'])-5.5)/19
            assert i==int(r['decision_index'])
            cat=old.category(v);before='GENESIS' if i==0 else rows[i-1]['outcome_class']
            history=('INITIAL_EMPTY' if i==0 else 'CLEARED_AFTER_TIMELY_UNDEFINED' if before=='TIMELY_UNDEFINED'
                else 'EMPTY_ORIGIN_UNRESOLVED') if cat=='reset' else 'DEFINED_HISTORY' if cat=='normal' else 'TIMEOUT_HISTORY'
            rec=dict(run=ver,index=i,frame=fid,row=r,span=s,v=v,mode=int(r['mode_id']),q_e4=int(r['q_e4']),
                t_s=float(r['t_since_first_capture_s']),condition='GOOD' if float(r['snr_proxy_db'])>15 else 'BAD',
                history=history,category=cat,preceding_outcome=before)
            rr.append(rec);allrows.append(rec)
        assert len(spans)==len(rr) and len({r['frame'] for r in rr})==len(rr)
        runs[ver]=rr
    return runs,allrows,inputs

def state_id(r):return f"V{r['run']}:{r['index']}"

def record_document(r):
    return dict(id=state_id(r),run=r['run'],index=r['index'],frame=r['frame'],t_s=r['t_s'],
        mode=r['mode'],q_e4=r['q_e4'],condition=r['condition'],category=r['category'],history=r['history'],
        preceding_outcome=r['preceding_outcome'],features=r['v'],state_sha256=r['span']['state_sha256'],
        heads_sha256=r['span']['heads_sha256'],payload_sfd4_bytes=int(r['row']['payload_l2_bytes']),
        commanded_snr_db=float(r['row']['snr_proxy_db']),prior_mcs=int(r['row']['prior_ul_mcs']),
        backlog_bytes=int(r['row']['pre_action_rlc_backlog_bytes']))
