"""Alternate CPU source-producer A/B on16 retained frames with exact hashes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from . import source_geometry as O

ROOT=Path(__file__).resolve().parents[2]
REPLAY=ROOT/'experiments/splitfusion_live_parity_repair_v2/20261003_v2/gt_cost_audit/replay_source_cost.py'
REPLAY_SHA='9c31b141d1e8675ed395af077fa89d4721721034680aa86f5aabef0b9c258cfd'
DEFAULT=ROOT/'experiments/splitfusion_live_gt_optimization_v1/source_geometry_v1'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value): return A.canonical(A._json_value(value))
def tree(path):
    return {str(f.relative_to(path)):sha(f) for f in sorted(path.rglob('*')) if f.is_file()}


def load_reference():
    if sha(REPLAY)!=REPLAY_SHA: raise ValueError('sealed replay helper differs')
    spec=importlib.util.spec_from_file_location('_sealed_gt_cost_replay',REPLAY)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def make(cls,reference,bundle,evidence):
    p=evidence['dynamic_snapshot_evidence']
    return cls(static_catalogue_dir=reference.CELL/'parity_static_catalogue',
        experiment_id=bundle.inputs.frame['experiment_id'],ego_id=p['population']['ego_id'],
        dynamic_population_ids=p['registered_population_ids'],
        population_evidence_sha256=evidence['population_evidence_sha256'],max_frames=100)


def measure(fn):
    w,c=time.perf_counter_ns(),time.thread_time_ns()
    value=fn()
    return value,dict(wall_ms=(time.perf_counter_ns()-w)/1e6,thread_cpu_ms=(time.thread_time_ns()-c)/1e6)


def run(output,passes=3):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    reference=load_reference()
    source=reference.CELL/'parity_complete_targets'
    before=tree(source)
    summary=json.loads((reference.CELL/'run5_live_avo_gt_summary.json').read_text())
    cases=[reference.owned(r) for r in summary['records']]
    warm=O.prewarm()
    rows=[]
    for repeat in range(passes):
        producers={name:make(cls,reference,*cases[0][:2]) for name,cls in
                   (('control',O.BASE_PRODUCER),('treatment',O.OptimizedSourceProducer))}
        for index,(bundle,evidence,kwargs) in enumerate(cases):
            expected_input=bundle.inputs.provenance['input_content_sha256']
            expected_source=hashlib.sha256(canonical(evidence)).hexdigest()
            expected_target=bundle.prepared.content_sha256
            population=evidence['dynamic_snapshot_evidence']
            order=('control','treatment') if (index+repeat)%2==0 else ('treatment','control')
            rebuilt={}; timings={}
            for name in order:
                producer=producers[name]
                if producer.population!=frozenset(population['registered_population_ids']):
                    producer.register_population(population['registered_population_ids'],
                        population['population_sha256'],before_frame_id=kwargs['identity']['frame_id'])
                # Replay the already-recorded diagnostic observation clock;
                # this is outside numerical geometry and is necessary for
                # exact evidence hashes. Production keeps its real clock.
                with patch('rl_agent.splitfusion_run5_live_avo_v1.gt_source_sync.time.clock_gettime_ns',
                           return_value=evidence['source_synchronization']['observed_raw_ns']):
                    value,timing=measure(lambda:producer.build(**kwargs))
                if value.provenance['input_content_sha256']!=expected_input:
                    raise AssertionError(f'{name} native input mismatch frame{bundle.prepared.frame_id}')
                actual_source=hashlib.sha256(canonical(producer.last_evidence)).hexdigest()
                if actual_source!=expected_source:
                    (output/'SOURCE_MISMATCH.json').write_text(json.dumps(dict(
                        actual=producer.last_evidence,expected=evidence),indent=2)+'\n')
                    raise AssertionError(f'{name} source evidence mismatch frame{bundle.prepared.frame_id}')
                target=E.prepare_target(value)
                if target.content_sha256!=expected_target:
                    raise AssertionError(f'{name} full target mismatch frame{bundle.prepared.frame_id}')
                for key in ('segmentation_png','object_ignore_png'):
                    if target.target[key]!=bundle.prepared.target[key]:
                        raise AssertionError(f'{name} PNG bytes mismatch')
                rebuilt[name]=value;timings[name]=timing
            rows.append(dict(repeat=repeat,frame=bundle.prepared.frame_id,order=order,
                timings=timings,native_input_sha256=expected_input,
                source_sha256=expected_source,target_sha256=expected_target,
                all_hashes_exact=True))
    assert before==tree(source)
    result={'status':'EXACT_SOURCE_OPTIMIZATION_REPLAY_PASS','frames':len(cases),'passes':passes,
        'rows':rows,'source_evidence_unchanged':True,'prewarm':warm,
        'timing_quantiles':{name:{key:{label:float(np.quantile([r['timings'][name][key] for r in rows],q))
            for label,q in (('p50',.5),('p95',.95),('max',1.))} for key in ('wall_ms','thread_cpu_ms')}
            for name in ('control','treatment')},
        'paired_wall_savings_ms':{label:float(np.quantile([r['timings']['control']['wall_ms']-
            r['timings']['treatment']['wall_ms'] for r in rows],q)) for label,q in (('p50',.5),('p95',.95),('min',0.))},
        'caveats':['CPU replay only; no live or GPU evidence.','Only source geometry optimized; AVO, prepared-target transport and archives excluded.',
            'Identical source observation timestamps supplied only in offline replay to test full evidence hashes.',
            'No OpenCV operation or CPU/native thread configuration introduced.']}
    (output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'INPUT_SHA256.json').write_text(json.dumps(before,indent=2)+'\n')
    (output/'OUTPUT_SHA256.json').write_text(json.dumps(tree(output),indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('status','frames','passes','timing_quantiles','paired_wall_savings_ms')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--passes',type=int,default=3);args=parser.parse_args()
    if not 1<=args.passes<=10: raise ValueError('bounded1..10passes required')
    run(args.output,args.passes)
