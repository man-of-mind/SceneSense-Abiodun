"""CPU-only real-service proof. Retained single-ego images reused as TWO
explicit SYNTHETIC source identities; this is not a two-ego sensor experiment.
All outputs are create-only. No live launch, registration or deployment.
"""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import time
import uuid
import collections
import numpy as np

from .identity import Source, Sources, require
from .gt import SourceHandoff, SourceEvaluator, FeedbackRouter
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from rl_agent.splitfusion_exact_gt_live_v2 import contract as EC, rehearsal as R
from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
from rl_agent.splitfusion_gt_geometry_containment_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_runtime_v1.async_live_rehearsal import _owned_collector

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, sort_keys=True, indent=2, allow_nan=False)


def sources():
    return Sources([Source(ego_id=f'synthetic-ego-{i}', ue_id=i, imsi=f'00101000000000{i+1}',
        tunnel=f'oaitun_ue{i+1}', ip=f'10.0.0.{20+i}', rnti=1001+i, run_id='M4_OFFLINE_SYNTHETIC',
        cell_id='cell', stream_id=f'm4-ue{i}', session_id=f'm4-attach-fixture-{i}', feedback_port=52010+i)
        for i in range(2)])


class SourceCorpus(P.Corpus):
    def __init__(self, base, source, *, geometry_index=None):
        self.__dict__.update(base.__dict__)
        self.data = dict(base.data)
        self.predictions = {}
        self.data['predictions'] = self.predictions
        self.data['contexts'] = {}
        self.cases = []
        for index, original in enumerate(base.cases):
            capture = dict(original.kwargs['identity'], run_id=source.run_id, cell_id=source.cell_id,
                           stream_id=source.stream_id)
            case = dataclasses.replace(original, kwargs=dict(original.kwargs, identity=capture))
            if index == geometry_index:
                from rl_agent.splitfusion_gt_geometry_containment_v1.fixtures import _people
                from rl_agent.splitfusion_avo_tilt_v1.fixtures import tilt_case
                candidates = [k for k, value in _people(case).items() if not value['structural']]
                require(candidates, 'geometry fixture needs an eligible target')
                case, _ = tilt_case(case, candidates[0], 20., 0.)
            self.cases.append(case)
            context = dataclasses.replace(base.context(original), stream_id=source.stream_id)
            self.data['contexts'][case.frame_id] = dict(base.data['contexts'][case.frame_id], context=context)
            pred = dict(base.predictions[case.frame_id])
            identity = copy.deepcopy(pred['identity'])
            identity['gt_identity'] = capture
            identity['action_identity']['session_uuid'] = str(uuid.UUID(int=100 + source.ue_id))
            # One explicit actor-session rollover, preserving source GT history.
            if index >= 8:
                identity['action_identity']['session_uuid'] = str(uuid.UUID(int=200 + source.ue_id))
            pred['identity'] = identity
            self.predictions[case.frame_id] = pred
        self.data['cases'] = self.cases


def inventory():
    result = C.exact_sources(ROOT)
    result.update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(Path(__file__).parent.glob('*.py'))})
    return result


class Rig(R.Rig):
    def __init__(self, out, corpus, source, router):
        self.c, self.cases = corpus, corpus.cases
        self.ue_dir, self.evidence = out / 'ue_attempt', out / 'edge_evidence'
        self.ue_dir.mkdir(parents=True)
        self.evidence.mkdir()
        self.source = source
        limits = dict(C.limits(3900), prep_cpu=4 + source.ue_id * 2,
                      score_cpu=5 + source.ue_id * 2, service_cpu=20 + source.ue_id)
        config = dict(exact_gt_limits=limits, exact_gt_sources=inventory(),
            numeric_run_binding=corpus.binding, numeric_producer_binding=corpus.startup['binding_sha256'],
            numeric_prediction_contract=corpus.contract, exact_gt_socket_name=EC.SOCKET_NAME,
            exact_gt_capability_name=EC.CAPABILITY_NAME)
        self.h = SourceHandoff(source=source, attempt_dir=self.ue_dir, registration=config, service_config={})
        self.h.collector = _owned_collector(self.cases)
        self.h.evidence_dir = self.evidence
        engine = dict(producer_args=dict(corpus.producer_args(), max_frames=3900), run_binding=corpus.binding,
            producer_binding=corpus.startup['binding_sha256'], prediction_contract=corpus.contract,
            max_frames=3900, max_retained_jobs=limits['prep_jobs'],
            max_retained_bytes=limits['prep_jobs'] * 64 * 1024**2, offline_replay=False)
        self.h.start_service(engine)
        self.payloads, self.feedback, self.rows = queue.Queue(), {}, {}
        self.ev = SourceEvaluator(source=source, router=router, socket_path=self.evidence / EC.SOCKET_NAME,
            capability_path=self.evidence / EC.CAPABILITY_NAME, spec=corpus.spec, read_ground_truth=None,
            match_distance_m=2., gt_timeout_s=2.)
        self.ev.start()

    def close(self):
        edge = self.ev.close(timeout_s=15)
        try:
            ue = self.h.close()
        finally:
            require(not self.ev._sender.is_alive() and not self.ev._receiver.is_alive(), 'edge thread leak')
        require(not edge['failure'], edge['failure'])
        return dict(edge=edge, ue=ue)


def collect(rig, closed):
    directory = rig.h.directory / 'service'
    service = json.loads((directory / 'SERVICE_RESULT.json').read_bytes())
    frames, operations = {}, []
    for line in (directory / 'FRAMES.jsonl').read_text().splitlines():
        row = json.loads(line)
        fid = row['frame_id']
        entry = {k: row.get(k) for k in ('outcome', 'committed', 'commit', 'prep_retired', 'score_retired',
                                        'bind_refused', 'no_gt', 'cancel_reason')}
        path = directory / 'archive' / f'frame_{fid}' / 'scorer' / 'numerical_score.json'
        if path.exists():
            from rl_agent.splitfusion_fixed_action_screen_analysis_v2.analyze import rescore
            score = json.loads(path.read_bytes())
            entry['measurement'] = score['measurement']
            entry['q_perc'] = (row.get('result') or {}).get('q_perc')
            entry['exact_rescore'] = rescore(path.parent)
            require(entry['exact_rescore'], 'exact numerical archive rescoring differs')
        entry['feedback'] = {k: v for k, v in rig.feedback.get(fid, {}).items() if k != 'received'}
        frames[str(fid)] = entry
        enqueued = collections.defaultdict(collections.deque)
        for row_enq in row.get('m4_enqueue', []):
            enqueued[(row_enq['worker'], row_enq['op'])].append(row_enq['raw_ns'])
        for request in row.get('requests', []):
            timing = request.get('timing') or {}
            if not timing or not enqueued[(request['worker'], request['op'])]:
                continue
            stamp = enqueued[(request['worker'], request['op'])].popleft()
            operations.append(dict(frame=fid, worker=request['worker'], op=request['op'],
                queue_ms=(request['begin'] - stamp) / 1e6,
                cpu_ms=timing['thread_cpu_ns'] / 1e6,
                wall_ms=(timing['end_raw_ns'] - timing['begin_raw_ns']) / 1e6,
                status=request['status'], reason=request.get('reason')))
    trace = rig.h.directory / 'm4_worker_trace'
    docs = {name: json.loads((trace / name).read_bytes()) for name in
            ('prep_gc_trace.json', 'score_gc_trace.json', C.ACCOUNTING_BINDING_RECORD,
             C.ACCOUNTING_TIMING_RECORD, C.CONTAINMENT_BINDING_RECORD)}
    account = docs[C.ACCOUNTING_BINDING_RECORD]
    require(account['restored'] and account['accounting_session_closed'] and account['fast_path_available'],
            'accounting capability/cleanup failed')
    require(not service['errors'] and rig.h.process.returncode == 0, 'service failure')
    samples = json.loads((directory / 'QUEUE_SAMPLES.json').read_bytes())
    memory = json.loads((directory / 'MEMORY_SAMPLES.json').read_bytes())
    return dict(frames=frames, service=service, accounting=account, traces=docs, closed=closed,
                operations=operations, queue_samples=samples, memory_samples=memory)


def pump(rigs):
    for rig in rigs.values():
        while True:
            try:
                payload = rig.payloads.get_nowait()
            except queue.Empty:
                break
            feedback, reason = W.decode_feedback(payload)
            rig.feedback[feedback.frame_id] = dict(kind=feedback.kind, q_perc=feedback.q_perc,
                                                   reason=reason.name, received=R.raw())
            rig.terminal(feedback.frame_id)
        now = R.raw()
        for fid, row in rig.rows.items():
            if now - row['opened'] > 170_000_000:
                rig.terminal(fid)
        require(rig.ev._failure is None, str(rig.ev._failure))
        rig.h._check()


def run_group(out, corpora, roster, selected, indices, *, faults=False, cadence=None):
    out.mkdir()
    rigs, timings, staged, lifecycle = {}, [], set(), []
    def sendto(payload, destination):
        owners = [s for s in roster.sources if s.destination == destination]
        require(len(owners) == 1, 'wrong destination')
        rigs[owners[0].ue_id].payloads.put(payload)
    router = FeedbackRouter(roster, sendto)
    results = {}
    try:
        for i in selected:
            source = roster.sources[i]
            rigs[i] = Rig(out / f'ue{i}', corpora[i], source, router)
        base_time = time.monotonic()
        for n, index in enumerate(indices):
            if cadence is not None:
                while time.monotonic() < base_time + n * cadence:
                    pump(rigs)
                    time.sleep(.001)
            active = [i for i in selected if i in rigs]
            opened = time.monotonic()
            for i in active:
                cancel = faults and index == (3 if i == 0 else 4)
                if (i, index) not in staged:
                    rigs[i].admit(index, send=None if cancel else True)
            # Opposite per-source outcome trajectories; alternate which source's
            # result is consumed first while the other remains pending.
            order = active[::-1] if n % 2 == 0 else active
            if faults and index == 5:
                order = sorted(active, reverse=True)  # B must not wait for A's intentional 2-s timeout
            for i in order:
                rig, fid = rigs[i], corpora[i].cases[index].frame_id
                cancel = faults and index == (3 if i == 0 else 4)
                edge_terminal = faults and i == 0 and index == 6
                if cancel:
                    until = time.monotonic() + 10
                    while not rig.h.status(fid).get('bound'):
                        require(time.monotonic() < until, 'cancel fixture never reached bound state')
                        time.sleep(.01)
                    rig.send(fid, False)
                    rig.terminal(fid)
                    continue
                if edge_terminal:
                    until = time.monotonic() + 10
                    while not rig.h.status(fid).get('committed'):
                        require(time.monotonic() < until, 'edge-terminal fixture never reached committed state')
                        time.sleep(.01)
                    rig.ev.notify_terminal(fid, 'STALE_BEFORE_MAP', 'DECODE')
                    rig.terminal(fid)
                    continue
                if faults and i == 0 and index == 5:
                    time.sleep(2.15)  # one intentional retained horizon timeout
                rig.predict(fid)
                if cadence is None:
                    rig.await_feedback(fid, timeout=15)
                    rig.terminal(fid)
            if cadence is not None:
                pump(rigs)
            timings.append(dict(index=index, offered_monotonic=opened,
                                cycle_wall_s=time.monotonic() - opened,
                                release_lag_s=max(0., opened - (base_time + n * cadence)) if cadence else None))
            if faults and index == 10 and 0 in rigs:
                if 1 in rigs:
                    other = rigs[1]
                    other.admit(11, send=None)
                    next_frame = corpora[1].cases[11].frame_id
                    until = time.monotonic() + 10
                    while not other.h.status(next_frame).get('bound'):
                        require(time.monotonic() < until, 'other-source pending fixture did not bind')
                        time.sleep(.01)
                    staged.add((1, 11))
                rig = rigs.pop(0)
                results[0] = collect(rig, rig.close())
                if 1 in rigs:
                    state = other.h.status(next_frame)
                    require(state.get('bound') and not state.get('committed') and state.get('outcome') is None
                            and other.h.process.poll() is None, 'closing A affected B pending GT')
                    lifecycle.append(dict(closed_source=0, surviving_source=1, frame=next_frame, state=state))
                    other.send(next_frame, True)
        if cadence is not None:
            deadline = time.monotonic() + 15
            while any(len(r.feedback) < len(r.rows) for r in rigs.values()) and time.monotonic() < deadline:
                pump(rigs)
                time.sleep(.002)
            require(all(len(r.feedback) == len(r.rows) for r in rigs.values()), 'capacity feedback did not drain')
        for i in list(rigs):
            rig = rigs.pop(i)
            results[i] = collect(rig, rig.close())
    except BaseException:
        for rig in rigs.values():
            rig.h.abort('M4 offline qualification failure; preserve evidence')
        raise
    write(out / 'RESULT.json', dict(sources={str(k): v for k, v in results.items()}, timeline=timings, lifecycle=lifecycle))
    return results, timings


def semantics(result):
    # Real RAW/CPU scheduling and receipt timeliness vary. Q_perc, all numerical
    # measurements, cancellation/refusal and producer commits do not get erased.
    return result['frames']


def capacity_summary(result, timeline):
    def quantiles(values):
        return dict(n=len(values), p50=float(np.median(values)), p99=float(np.quantile(values, .99)),
                    maximum=float(max(values))) if values else dict(n=0)
    out = {}
    for source, record in result.items():
        operations = record['operations']
        groups = {}
        for worker, op in sorted({(r['worker'], r['op']) for r in operations}):
            rows = [r for r in operations if (r['worker'], r['op']) == (worker, op)]
            groups[f'{worker}/{op}'] = {field: quantiles([r[field] for r in rows])
                                       for field in ('cpu_ms', 'wall_ms', 'queue_ms')}
        accounting = record['traces'][C.ACCOUNTING_TIMING_RECORD]
        rss = {role: max((r.get(role) or 0 for r in record['memory_samples']), default=0)
               for role in ('prep', 'score', 'service', 'archive')}
        backlog = {key: max((r[key] for r in record['queue_samples']), default=0) for key in
                   ('prep_queue', 'score_queue', 'freeze_queue', 'prep_retained', 'scorer_slots', 'waiting', 'archive_outstanding')}
        span = timeline[-1]['offered_monotonic'] - timeline[0]['offered_monotonic'] + .2
        out[str(source)] = dict(operations_ms=groups, accounting_cpu_ms=quantiles([x / 1e6 for x in accounting['cpu_ns']]),
            sampled_rss_kib_max=rss, backlog_max=backlog, backlog_final=record['queue_samples'][-1],
            worker_cpu_seconds={role: sum(r['cpu_ms'] for r in operations if r['worker'] == role) / 1000
                                for role in ('prep', 'score')},
            worker_cpu_fraction_of_offer_window={role: sum(r['cpu_ms'] for r in operations if r['worker'] == role) / 1000 / span
                                                 for role in ('prep', 'score')},
            outcomes=record['service']['outcomes'], errors=record['service']['errors'],
            cleanup=record['service']['closed'], release_lag_s=quantiles([r['release_lag_s'] for r in timeline]),
            history_slices={label: {field: quantiles([r[field] for r in operations if r['op'] == 'FREEZE'
                and ((r['frame'] <= 1487) if label == 'early' else (r['frame'] >= 1607))])
                for field in ('cpu_ms', 'wall_ms', 'queue_ms')} for label in ('early', 'late')})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=('smoke', 'correctness', 'capacity'), default='correctness')
    p.add_argument('--correctness', type=Path)
    args = p.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only environment required')
    out = args.output.absolute()
    require(out.is_relative_to(ROOT / 'experiments/splitfusion_m4_two_ue_v1'), 'scoped output required')
    out.mkdir(parents=True, exist_ok=False)
    started, free = time.monotonic(), shutil.disk_usage(ROOT).free
    source_hashes = inventory()
    write(out / 'PLAN.json', dict(phase=args.phase, sources=source_hashes, disk_before=free,
        synthetic='One retained 49-frame single-ego corpus, two relabelled source identities; not new sensor evidence',
        placement={'ue0': [4, 5, 20], 'ue1': [6, 7, 21]},
        placement_scope='logical CPUs in retained edge partition; no radio CPUs, no concurrent GPU inference'))
    base = P.Corpus(49)
    roster = sources()
    corpora = [SourceCorpus(base, s, geometry_index=(9 + s.ue_id) if args.phase == 'correctness' else None)
               for s in roster.sources]
    if args.phase == 'smoke':
        result, _ = run_group(out / 'two_sources', corpora, roster, [0, 1], [0, 1])
        checks = {'both_sources_scored': all(all(v['outcome'] == 'SCORED' for v in r['frames'].values())
                                            for r in result.values())}
    elif args.phase == 'correctness':
        isolated = {}
        for i in (0, 1):
            isolated[i] = run_group(out / f'isolated_{i}', corpora, roster, [i], range(14), faults=True)[0][i]
        combined, _ = run_group(out / 'two_sources', corpora, roster, [0, 1], range(14), faults=True)
        checks = {f'ue{i}_isolated_semantic_parity': semantics(isolated[i]) == semantics(combined[i]) for i in (0, 1)}
        write(out / 'PARITY.json', dict(checks=checks, isolated={str(i): semantics(isolated[i]) for i in (0, 1)},
                                       combined={str(i): semantics(combined[i]) for i in (0, 1)}))
    else:
        require(args.correctness is not None, 'passing correctness artifact required')
        proof = json.loads(args.correctness.read_bytes())
        require(proof['status'] == 'PASS' and proof['sources'] == source_hashes, 'correctness/source drift')
        # One bounded comparison, no performance tuning/repetition.
        one, t1 = run_group(out / 'one_source', corpora, roster, [0], range(49), cadence=.2)
        two, t2 = run_group(out / 'two_sources', corpora, roster, [0, 1], range(49), cadence=.2)
        checks = {'source0_output_parity': semantics(one[0]) == semantics(two[0])}
        write(out / 'CAPACITY_RAW.json', dict(one=one, two=two, one_timeline=t1, two_timeline=t2,
            limitation='49-state early/late history only; no live inference or full-route history capacity claim'))
        write(out / 'CAPACITY.json', dict(one=capacity_summary(one, t1), two=capacity_summary(two, t2),
            cadence_ms=200, offered_frames_per_source=49, checks=checks))
    require(inventory() == source_hashes, 'sources changed during qualification')
    import torch
    require(not torch.cuda.is_initialized(), 'CUDA unexpectedly initialized')
    summary = dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks, sources=source_hashes,
                   phase=args.phase, disk_before=free, disk_after=shutil.disk_usage(ROOT).free,
                   duration_s=time.monotonic() - started, cuda_initialized=False)
    write(out / 'SUMMARY.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'sources'}, indent=2))
    return int(summary['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
