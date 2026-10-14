"""CPU-only production composition regression with labelled external doubles.

Frozen actor, native state builder/controller, wire, reassembly, GT serializer,
feedback and actual Run4 map-ingestion/journal methods execute. CARLA sensor
delivery, front/tail GPU compute, radio and GT child I/O are synthetic here.
The unchanged real GT children remain covered by correctness_04/capacity_01.
"""
import copy
import dataclasses
import hashlib
import io
import contextlib
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from . import configuration as C
from .identity import require
from .radio import resolve, EXPECTED
from .composition import independent_policy_contexts, BothStreamLifecycle
from .policy import NativePipeline, TelemetryFanout
from .datapath import DataPath
from .edge_components import SourceEvaluator, FeedbackRouter
from .map_adapter import Run4MapJournal
from .map_service import ObservationStore, service_class
from .world import TickOwner, drive_pair
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import continuous_execution_v2 as X
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W, run4_map_protocol_v2 as MP
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_ue_telemetry_provider_v2 import Harness
from rl_agent.splitfusion_run5_live_avo_v1 import snr_live as S
from phase2_map_sharing.transport import chunk_payload

OUTPUT = None
_preload_ue = None  # Explicit dependency slot for the external constructor double.


def fixture_binding():
    args = dict(expected=list(EXPECTED), run_id='M4_SYNTHETIC_COMPOSITION', cell_id='town10',
        attach_sessions={'0':'synthetic-attach-0', '1':'synthetic-attach-1'},
        uicc_text='\n'.join(f'uicc{i} = {{\n imsi = "{v["imsi"]}";\n}};' for i,v in enumerate(EXPECTED)),
        ue_log='\n'.join('UICC simulation: IMSI=' + v['imsi'] for v in EXPECTED),
        interfaces={v['tunnel']:json.dumps([dict(ifname=v['tunnel'],addr_info=[dict(family='inet',local=f'10.0.0.{41+i}')])])
                    for i,v in enumerate(EXPECTED)}, rlc_rows=[dict(ue_id=i, rnti=55228+i) for i in range(2)])
    return args


class SyntheticFront:
    """No inferred payload or model outputs; valid identity wrapping fixed bytes."""
    def prepare(self, profile, tensor, frame):
        inner = b'EXPLICIT_SYNTHETIC_GPU_BOUNDARY'
        envelope = X.ExecutionEnvelopeV3(mode_id=profile.mode_id, q_e4=profile.q_e4,
            keep_count=profile.keep_count, anchor_action_id=profile.action_id,
            execution_bundle_sha256=profile.execution_bundle_sha256,
            inner_payload_sha256=hashlib.sha256(inner).hexdigest(), inner_payload=inner,
            **dataclasses.asdict(frame))
        return types.SimpleNamespace(envelope=envelope)


class SocketDouble:
    def __init__(self, *args):
        self.sent = []
    def setsockopt(self, *args): pass
    def settimeout(self, *args): pass
    def bind(self, *args): pass
    def sendto(self, data, destination): self.sent.append((data, destination)); return len(data)
    def close(self): pass


class CollectorDouble:
    """External sensor/tick boundary; production NativePipeline is real."""
    def __init__(self, source, context):
        self.source = source
        self.harness = Harness(bind=False)
        self.harness.provider.bind_ue(rnti=source.rnti, oai_ue_id=source.ue_id)
        context['telemetry'] = self.harness.provider
        context['engine'].ue_id = self.harness.provider.ue_label
        # Engines are originally constructed with the same bound source label.
        self.provider = S.LiveSnrProvider('/tmp/not-created-m4-' + source.stream_id,
            str(uuid.UUID(int=source.ue_id + 11)), self.harness.provider.ue_label, start_receiver=False)
        self.pipeline = NativePipeline(source=source, snr_provider=self.provider, engine=context['engine'],
            continuous_ue=SyntheticFront(), provider=self.harness.provider, clock=self.harness._ticking_clock())
        self.last = None
        self.offers = []
        self.drained = False

    def wait_for_sensor_frame(self, frame, *, snapshot, timeout_s):
        require(snapshot.frame == frame, 'synthetic world drift')

    def on_world_tick(self, frame, route_tick):
        if route_tick % 2 == 0:
            return
        h = self.harness
        h.dci(rnti=self.source.rnti, mcs=20)
        for lcid, value in zip((1,2,4), (100,0,250)):
            h.rlc(frame=frame % 1024, slot=0, lcid=lcid, bytes_=value, rnti=self.source.rnti, ue=self.source.ue_id)
        h.rlc(frame=frame % 1024, slot=1, lcid=1, bytes_=0, rnti=self.source.rnti, ue=self.source.ue_id)
        now = h.host.raw()
        packet = S.SnrSnapshot(S.SCHEMA, self.provider.controller_id, len(self.offers)+1, S.CLOCK_DOMAIN,
                              'synthetic-ack', now-10000, now-5000, 23.5, True, None).encode()
        require(self.provider.accept_packet(packet, now-1000), 'synthetic SNR rejected')
        scene = U.SceneDescriptorsV2(camera_si=25., radar_p40=.5, camera_status='OK', radar_status='OK',
                                    source_raw_ns=now-500000, available_raw_ns=now-10000)
        capture = h.host.real_ns
        planned = self.pipeline.plan(frame_id=frame, capture_wall_ns=capture, scene=scene)
        self.last = self.pipeline.materialize(planned, frame_id=frame, capture_wall_ns=capture,
            ego_pose=(15.*self.source.ue_id,0.,1.,0.,0.,0.), input_7ch=lambda: None)
        self.offers.append(self.last)
        h.host.advance(100_000_000)

    def finish(self):
        self.drained = True


class ComposedChecks(unittest.TestCase):
    def setUp(self):
        self.sources = resolve(**fixture_binding())

    def test_production_constructors_and_proof_reuse(self):
        from .frontend import collector_class, runtime_class
        from .configuration import reused_proofs
        source = self.sources.sources[0]
        cls = collector_class(source=source, runtime_factory=None, ledger=None, handoff=None, pose_archive=None)
        rt = runtime_class(types.SimpleNamespace(evidence_dir=OUTPUT), source=source, context=None,
                           snr_provider=None, shared_models=None, handoff=None)
        self.assertIn('M3EndOfRouteDrainLayer', [c.__name__ for c in cls.__mro__])
        self.assertIn('Run4LiveRuntimeV2', [c.__name__ for c in rt.__mro__])
        self.assertEqual(len(reused_proofs()), 3)

    def test_ambiguous_attach_and_wrong_source_telemetry(self):
        bad = fixture_binding()
        bad['rlc_rows'].append(dict(ue_id=0,rnti=333))
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            resolve(**bad)
        bad = fixture_binding()
        bad['ue_log'] = 'UICC simulation: IMSI=001010000000001\n' * 2
        with self.assertRaises(ValueError): resolve(**bad)
        contexts = independent_policy_contexts(self.sources, '1'*64)
        fanout = TelemetryFanout(contexts)
        self.assertFalse(fanout.dispatch('NRUE_MAC_RLC_BUFFER_STATUS', dict(ue_id=0,rnti=55229), None))
        self.assertFalse(fanout.dispatch('NRUE_MAC_DCI_GRANT', dict(rnti=123), None))

    def test_two_collectors_native_policy_dispatch_feedback_maps_and_drain(self):
        contexts = independent_policy_contexts(self.sources, '1'*64)
        collectors = {s.stream_id:CollectorDouble(s, contexts[s.stream_id]) for s in self.sources.sources}
        for collector in collectors.values():
            collector.on_world_tick(4172, 1)
            self.assertEqual(len(collector.pipeline.decisions[0]['features']), 22)
            self.assertEqual(collector.pipeline.engine.counters.actor_calls, 1)
        contract = next(iter(contexts.values()))['engine']._contract
        rewards = []
        router = FeedbackRouter(self.sources, lambda data,destination: rewards.append((data,destination)))
        evaluators = {s.stream_id:SourceEvaluator(source=s, router=router, socket_path='/nonexistent',
            capability_path='/nonexistent', spec=None, read_ground_truth=None, match_distance_m=2.,gt_timeout_s=2.)
            for s in self.sources.sources}
        journal = Run4MapJournal(self.sources, contract, OUTPUT / 'observations.jsonl')
        store = ObservationStore(history_size=32)
        poses = {}
        for s in self.sources.sources:
            item = collectors[s.stream_id].last
            poses[(s.stream_id,4172)] = dict(source=s.tag,frame_id=4172,
                capture_timestamp_ns=item.envelope.capture_timestamp_ns,provenance='EXPLICIT_SYNTHETIC_SENSOR')
        services = {}
        cls = service_class(contract)
        with patch('socket.socket', SocketDouble):
            for s in self.sources.sources:
                services[s.stream_id] = cls(source=s,journal=journal,pose_lookup=lambda s,f:poses[(s.stream_id,f)],
                    bind_host='192.168.70.1', bind_port=53000+s.ue_id, install=store.install)
        def publish(source, document):
            service = services[source.stream_id]
            # Existing ingest's actual ordering/horizon checks. 250 ms is late
            # feedback, still eligible for map publication.
            stamp = document['capture_timestamp_ns']/1e9 + .25
            with patch('time.time', return_value=stamp):
                result = service.ingest(document, ingest_at=stamp)
            self.assertTrue(result['feedback']['accepted'])
            self.assertEqual(service.socket.sent[-1][1], source.destination)
        events, terminals = [], []
        records=[dict(class_name=name,internal_class=i,score=.9,world_x=1.,world_y=2.,world_z=0.,
                      size_x=2.,size_y=1.,size_z=1.,yaw_cos=1.,yaw_sin=0.) for i,name in enumerate(('vehicle','person'))]
        path = DataPath(sources=self.sources,contract=contract,evaluators=evaluators,
            compute=lambda **kw:(records,np.zeros((720,1280),np.uint8)),publish=publish,
            terminal=lambda s,v:terminals.append((s,v)),log=lambda s,k,**v:events.append((s.stream_id,k,v)))
        chunks = [chunk_payload(c.last.wire,message_id=4172,chunk_bytes=128) for c in collectors.values()]
        capture = min(c.last.envelope.capture_timestamp_ns for c in collectors.values())/1e9
        for pair in zip(*chunks):
            for source,datagram in zip(self.sources.sources,pair):
                path.receive(source,datagram,monotonic_s=1.,wall_s=capture+.01)
        self.assertEqual(path.ingress.pending.depth(),2)
        for _ in range(2): self.assertTrue(path.process_once(0,now_s=capture+.02))
        self.assertEqual(set(store.latest),set(collectors))
        for source in reversed(self.sources.sources):
            ev=evaluators[source.stream_id]
            ev._connect=lambda:None
            ev._receiver=types.SimpleNamespace(start=lambda:None)
            ev._send_service=lambda message,fd=None:None
            ev._queue.put(None)
            ev._send_loop()
            self.assertIsNone(ev._failure)
            ticket=ev._pending[4172]
            response=source.wrap(dict(op='UNSCORED',frame_id=4172,_m4_action=MP.run4_identity(ticket.envelope),
                                      reason='SYNTHETIC_REFUSAL'))
            other=self.sources.sources[1-source.ue_id]
            with self.assertRaises(ValueError): ev.accept_result(other.wrap(dict(op='UNSCORED',frame_id=4172)))
            _,ticket=ev.accept_result(response)
            ev._emit(ticket,None,W.EvaluatorReason.GROUND_TRUTH_UNAVAILABLE)
        self.assertEqual([d for _,d in rewards],[s.destination for s in reversed(self.sources.sources)])
        for data,destination in rewards:
            source=next(s for s in self.sources.sources if s.destination==destination)
            feedback,_=W.decode_feedback(data)
            engine=collectors[source.stream_id].pipeline.engine
            with self.assertRaises(ValueError): router.deliver(source,data,self.sources.sources[1-source.ue_id].destination)
            engine.on_feedback(feedback,receipt_raw_ns=collectors[source.stream_id].harness.host.raw())
        path.end(self.sources.sources[0])
        with self.assertRaises(ValueError): path.receive(self.sources.sources[0],chunks[0][0])
        self.assertNotIn(self.sources.sources[1].stream_id,path.retired)
        for service in services.values(): service.close()
        journal.close()
        observations=[json.loads(r) for r in (OUTPUT/'observations.jsonl').read_text().splitlines()]
        self.assertEqual(len(observations),2)
        self.assertTrue(all(r['wire_document']['records']==records for r in observations))
        for record in store.latest.values():
            self.assertEqual({r['type'] for r in record['objects']},{'Vehicle','Pedestrian'})

    def test_dispatch_supersession_retirement_and_horizon_terminals(self):
        from .offline_checks import ticket
        from rl_agent.splitfusion_live_dispatch_v1.dynamic_execution_contract import load_dynamic_execution_contract
        terminals=[]; notices=[]
        evaluators={s.stream_id:types.SimpleNamespace(notify_terminal=lambda *v,s=s:notices.append((s.stream_id,v)))
                    for s in self.sources.sources}
        path=DataPath(sources=self.sources,contract=load_dynamic_execution_contract(),evaluators=evaluators,
            compute=lambda **kw:self.fail('expired frame reached compute'),publish=lambda *a:self.fail('expired map'),
            terminal=lambda s,v:terminals.append((s.stream_id,v)),log=lambda *a,**kw:None)
        def offer(source,frame,late=False):
            t=ticket(source,frame)
            envelope=dataclasses.replace(t.envelope,tensor_seq=frame)
            context=dataclasses.replace(t.context,sequence_id=frame)
            wire=W.pack_sfd4(envelope,context); wall=context.capture_timestamp_ns/1e9
            for packet in chunk_payload(wire,message_id=frame,chunk_bytes=128):
                path.receive(source,packet,monotonic_s=frame,wall_s=wall+(.6 if late else .01))
            return wall
        a,b=self.sources.sources
        offer(a,4172); offer(b,4172); end=offer(a,4173)
        self.assertEqual(terminals[0][0],a.stream_id)
        self.assertEqual(terminals[0][1]['outcome'],'SUPERSEDED_PENDING')
        path.end(a)
        self.assertEqual(path.ingress.pending.depth(),2)
        self.assertTrue(path.process_once(0,now_s=end+1.))
        self.assertTrue(path.process_once(0,now_s=end+1.))
        self.assertFalse(path.process_once(0))
        offer(b,4174,late=True)
        self.assertEqual([v['outcome'] for _,v in terminals],
            ['SUPERSEDED_PENDING','STALE_BEFORE_MAP','STALE_BEFORE_MAP','STALE_BEFORE_EDGE'])
        self.assertEqual(len(notices),4)

    def test_one_tick_owner_independent_completion_and_drain(self):
        class World:
            frame=0
            def tick(self): self.frame+=1; return self.frame
            def get_snapshot(self): return types.SimpleNamespace(frame=self.frame)
        class Collector:
            def __init__(self): self.frames=[]; self.closed=False
            def wait_for_sensor_frame(self,frame,**kw): require(frame==kw['snapshot'].frame,'wrong snapshot')
            def on_world_tick(self,frame,tick): self.frames.append(frame)
            def finish(self): self.closed=True
        collectors={s.stream_id:Collector() for s in self.sources.sources}
        world=World(); owner=TickOwner(world,collectors,timeout_s=2)
        lifecycle=BothStreamLifecycle(self.sources)
        def driver(n):
            def run(world):
                for _ in range(n): world.tick()
                return dict(completed=True,abort_reason='')
            return run
        drive_pair(owner,{s.stream_id:driver(2+i) for i,s in enumerate(self.sources.sources)},lifecycle,
                   {k:v.finish for k,v in collectors.items()})
        self.assertEqual(world.frame,3)
        self.assertEqual([v.frames for v in collectors.values()],[[1,2],[1,2,3]])
        self.assertTrue(lifecycle.done)

    def test_production_runtime_constructor_with_external_gpu_socket_double(self):
        """Invoke the real constructor, not just factory/MRO inspection."""
        import threading
        import torch
        from .frontend import runtime_class
        from .host import campaign_for
        from .sender import SourceSender
        from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as B
        contexts=independent_policy_contexts(self.sources,'2'*64)
        created=[]
        def external_base_init(runtime,**kw):
            runtime.map_socket=SocketDouble(); runtime.device=torch.device('cpu')
            runtime.sender=SocketDouble()
            runtime.sender.getsockname=lambda:(kw['campaign']['runtime']['ue_bind_host'],40000)
            runtime.ue=types.SimpleNamespace(_front=lambda value:value,
                _ranker=types.SimpleNamespace(score_cells=lambda value:value),
                _ae_encoders={name:None for name in X.AE_FAMILIES},_codec=None)
        for source in self.sources.sources:
            context=contexts[source.stream_id]
            context['prewarm_report']=dict(completed=True,modes_warmed=list(range(12)))
            provider=S.LiveSnrProvider('/tmp/m4-constructor-not-created',str(uuid.UUID(int=source.ue_id+21)),
                context['telemetry'].ue_label,start_receiver=False)
            handoff=types.SimpleNamespace(on_controller_poll=lambda *a:None)
            campaign,cell,_=campaign_for(source,C.specification(),OUTPUT/'not-started')
            with patch.object(B.LivePilotCellRuntime,'__init__',external_base_init),patch('socket.if_nametoindex',return_value=10+source.ue_id):
                cls=runtime_class(types.SimpleNamespace(evidence_dir=OUTPUT),source=source,context=context,
                    snr_provider=provider,shared_models=None,handoff=handoff)
                runtime=cls(ledger=None,campaign=campaign,cell=cell,attempt_dir=OUTPUT,
                    evidence_dir=OUTPUT,ue_control_port=source.feedback_port)
            self.assertIsInstance(runtime.sender,SourceSender)
            self.assertTrue(runtime._run4_ready.is_set())
            created.append(runtime)
        self.assertIsNot(created[0].engine,created[1].engine)
        self.assertIsNot(created[0].pipeline.state_builder,created[1].pipeline.state_builder)
        self.assertNotEqual(created[0].sender.ifindex,created[1].sender.ifindex)

    def test_sender_interface_metadata_and_unchanged_datagrams(self):
        from .sender import SourceSender
        import struct
        for source in self.sources.sources:
            calls=[]; sock=SocketDouble()
            sock.getsockname=lambda s=source:(s.ip,40000)
            sock.sendmsg=lambda data,ancillary,flags,dst: calls.append((data,ancillary,flags,dst)) or len(data[0])
            with patch('socket.if_nametoindex',return_value=source.ue_id+7): sender=SourceSender(sock,source)
            sender.sendto(b'unchanged-SFD4-fragment',('192.168.70.140',52000+source.ue_id))
            self.assertEqual(calls[0][0],[b'unchanged-SFD4-fragment'])
            self.assertEqual(struct.unpack('=I4s4s',calls[0][1][0][2])[0],source.ue_id+7)

    def test_runtime_registration_rejects_rehashed_source_and_endpoint_swaps(self):
        from .edge_entry import validate_runtime
        config=C.specification()
        tags=[dict(s.tag,run_id=config['run_id'],cell_id=config['cell_id']) for s in self.sources.sources]
        runtime=dict(preparation_sha256='1'*64,sources=tags,output=str(OUTPUT),map_host='192.168.70.1',
            chunk_bytes=12500,socket_buffer_request_bytes=8388608,
            endpoints={s['stream_id']:dict(ingress_port=config['sources'][s['ue_id']]['ingress_port'],
                map_port=config['map_ports'][s['ue_id']],edge_evidence=str(OUTPUT/s['stream_id']/'edge_evidence'),
                gt_socket=str(OUTPUT/s['stream_id']/'edge_evidence/service.sock'),
                gt_capability=str(OUTPUT/s['stream_id']/'edge_evidence/cap.json')) for s in tags})
        runtime['sha256']=C.canonical(runtime)
        registration=dict(config=config,sha256='1'*64)
        self.assertEqual(len(validate_runtime(registration,runtime,dict(sources=tags)).sources),2)
        for kind in ('subscriber','endpoint','gt'):
            bad=copy.deepcopy(runtime)
            if kind=='subscriber': bad['sources'][0]['imsi']='001010000000003'
            if kind=='endpoint': bad['endpoints']['m4-ue0']['map_port']=52101
            if kind=='gt': bad['endpoints']['m4-ue0']['gt_socket']=bad['endpoints']['m4-ue1']['gt_socket']
            bad['sha256']=C.canonical({k:v for k,v in bad.items() if k!='sha256'})
            with self.assertRaises(ValueError):
                validate_runtime(registration,bad,dict(sources=bad['sources']))

    def test_cli_binding_negative_controls_and_empty_analysis(self):
        from .cli import main
        from .analysis import analyze
        directory=OUTPUT/'cli_preparation'
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['prepare','--output',str(directory)]),0)
            channel=(directory/'ue.conf').read_text()
            import re
            self.assertEqual(len(re.findall(r'model_name\s*=\s*"rfsimu_channel_ue0"',channel)),1)
            self.assertEqual(len(re.findall(r'model_name\s*=\s*"rfsimu_channel_ue1"',channel)),1)
            self.assertEqual(main(['preflight','--preparation',str(directory/'PREPARATION.json')]),0)
            self.assertEqual(main(['capture','--preparation',str(directory/'PREPARATION.json'),
                                   '--output',str(OUTPUT/'must_not_exist'),'--execute-token','WRONG']),2)
            self.assertFalse((OUTPUT/'must_not_exist').exists())
            good=json.loads((directory/'PREPARATION.json').read_bytes())
            bad=copy.deepcopy(good); bad['config']['deadline_ms']=171
            bad['sha256']=C.canonical({k:v for k,v in bad.items() if k!='sha256'})
            C.write_new(directory/'DRIFT.json',bad)
            self.assertEqual(main(['preflight','--preparation',str(directory/'DRIFT.json')]),2)
            bad=copy.deepcopy(good); bad['sources'].pop(next(iter(bad['sources'])))
            bad['sha256']=C.canonical({k:v for k,v in bad.items() if k!='sha256'})
            C.write_new(directory/'OMISSION.json',bad)
            self.assertEqual(main(['preflight','--preparation',str(directory/'OMISSION.json')]),2)
        fixture=OUTPUT/'source_fixture'; fixture.mkdir()
        C.write_new(fixture/'source.json',dict(content='deliberate isolated mismatch'))
        with self.assertRaisesRegex(ValueError,'source drift'):
            C.verify_sources({'source.json':'0'*64},fixture)
        empty=OUTPUT/'empty_capture'; empty.mkdir()
        C.write_new(empty/'PREPARATION_REFERENCE.json',dict(sha256=good['sha256']))
        report=analyze(directory/'PREPARATION.json',empty,OUTPUT/'empty_analysis')
        self.assertEqual(report['verdict'],'STOP')
        self.assertEqual(report['combined']['completed'],0)
        self.assertFalse(report['gates']['both_routes'])
        ue=json.loads((OUTPUT/'empty_analysis/ASSESSMENT.json').read_bytes())['per_ue']
        self.assertTrue(all(v['prediction_ready_latency']['p50'] is None and v['gt_scored']==0 for v in ue.values()))

    def test_route_failure_preserves_partial_results_and_both_drains(self):
        class Collector:
            def wait_for_sensor_frame(self,*a,**kw): pass
            def on_world_tick(self,*a): pass
        class World:
            def tick(self): raise RuntimeError('original synthetic world failure')
        callbacks={s.stream_id:Collector() for s in self.sources.sources}
        lifecycle=BothStreamLifecycle(self.sources); closed=[]; partial={}
        owner=TickOwner(World(),callbacks,timeout_s=1)
        def driver(world): world.tick(); return dict(completed=True)
        with self.assertRaisesRegex(RuntimeError,'original synthetic world failure'):
            drive_pair(owner,{k:driver for k in callbacks},lifecycle,
                {k:lambda name=k:closed.append(name) for k in callbacks},results=partial)
        self.assertEqual(set(closed),set(callbacks))
        self.assertIn('original synthetic world failure',owner.error)
        self.assertFalse(lifecycle.done)


def run(output):
    global OUTPUT
    OUTPUT=Path(output); OUTPUT.mkdir(parents=True,exist_ok=False)
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ComposedChecks))
    import torch
    record=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        log=stream.getvalue(),cuda_initialized=torch.cuda.is_initialized(),
        external_boundaries='SYNTHETIC CARLA sensors/radio/GPU/GT IPC; native actor, state, wire, routing and map methods',
        sources=C.source_set(),live_qualification=False)
    C.write_new(OUTPUT/'RESULT.json',record)
    print(record['log']); print(record['status'])
    return 0 if result.wasSuccessful() and not record['cuda_initialized'] else 2
