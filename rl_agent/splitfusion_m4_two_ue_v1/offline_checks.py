"""Focused boundary/ownership regressions, complementary to real-service proof."""
import argparse
import copy
import dataclasses
import hashlib
import io
import json
from pathlib import Path
import queue
import unittest
import uuid
from types import SimpleNamespace as NS

import numpy as np
from .identity import Source
from .qualification import sources, inventory, write
from .gt import SourceEvaluator, FeedbackRouter, SharedEdgeProcessor
from .composition import BothStreamLifecycle, independent_policy_contexts
from .map_adapter import Run4MapJournal
from .transport import SourceIngress
from phase2_map_sharing.transport import chunk_payload
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import continuous_execution_v2 as X
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from rl_agent.splitfusion_live_dispatch_v1.dynamic_execution_contract import load_dynamic_execution_contract
from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1

OUT = None


def ticket(source, frame=4172, session=None):
    contract = load_dynamic_execution_contract()
    profile = contract.resolve_q_e4(11, 4321)
    capture = 1_791_800_000_000_000_000 + frame * 100_000_000
    context = build_frame_context_v1(stream_id=source.stream_id, frame_id=frame, sequence_id=1,
        capture_timestamp_ns=capture, ego_world_x=15. * source.ue_id, ego_world_y=0., ego_world_z=1.,
        ego_world_pitch=0., ego_world_yaw=0., ego_world_roll=0.)
    payload = b'CPU-fixture'
    envelope = X.ExecutionEnvelopeV3(mode_id=11, q_e4=4321, keep_count=profile.keep_count,
        anchor_action_id=None, reward_requested=True, session_uuid=session or str(uuid.UUID(int=100 + source.ue_id)),
        controller_lineage_sha256='1' * 64, decision_seq=0, ticket_seq=0, frame_id=frame, tensor_seq=1,
        capture_timestamp_ns=capture, execution_bundle_sha256=profile.execution_bundle_sha256,
        inner_payload_sha256=hashlib.sha256(payload).hexdigest(), inner_payload=payload)
    return E.EvaluationTicketV2(envelope=envelope, context=context, records=(),
        predicted_mask=np.zeros((720, 1280), np.uint8),
        gt_identity=MP.gt_identity(run_id=source.run_id, cell_id=source.cell_id, stream_id=source.stream_id,
            frame_id=frame, anchor_action_id=None, anchor_profile_id=None, capture_timestamp_ns=capture), enqueued_wall_ns=1)


class Checks(unittest.TestCase):
    def setUp(self):
        self.roster = sources()
        self.sent = []
        self.router = FeedbackRouter(self.roster, lambda payload, destination: self.sent.append((payload, destination)))

    def evaluator(self, source):
        return SourceEvaluator(source=source, router=self.router, socket_path='/nonexistent/m4',
            capability_path='/nonexistent/m4-cap', spec=None, read_ground_truth=None, match_distance_m=2., gt_timeout_s=2.)

    def test_interleaved_reassembly_and_source_specific_supersession(self):
        ingress = SourceIngress(self.roster)
        chunks = []
        for source in self.roster.sources:
            item = ticket(source)
            chunks.append(chunk_payload(W.pack_sfd4(item.envelope, item.context), message_id=4172, chunk_bytes=128))
        for pair in zip(*chunks):
            for source, datagram in zip(self.roster.sources, pair):
                ingress.ingest(source, datagram, received_at_s=1.,
                               received_wall_s=item.context.capture_timestamp_ns / 1e9 + .01)
        self.assertEqual(ingress.pending.depth(), 2)
        a, b = self.roster.sources
        newer = ticket(a, 4173)
        newer = dataclasses.replace(newer, envelope=dataclasses.replace(newer.envelope, tensor_seq=2),
                                    context=dataclasses.replace(newer.context, sequence_id=2))
        result = None
        for datagram in chunk_payload(W.pack_sfd4(newer.envelope, newer.context), message_id=4173, chunk_bytes=128):
            result = ingress.ingest(a, datagram, received_at_s=1.1,
                                     received_wall_s=newer.context.capture_timestamp_ns / 1e9 + .01)
        self.assertEqual(result['displaced']['source'], a.tag)
        pending = dict(ingress.pending.take(0) for _ in range(2))
        self.assertEqual(pending[a.stream_id]['frame_id'], 4173)
        self.assertEqual(pending[b.stream_id]['frame_id'], 4172)
        with self.assertRaises(ValueError):
            for datagram in chunks[0]:
                ingress.ingest(b, datagram, received_at_s=2.,
                               received_wall_s=item.context.capture_timestamp_ns / 1e9 + .01)

    def test_frame_4172_reversed_replies_duplicates_and_wrong_sources(self):
        a, b = self.roster.sources
        evaluators = [self.evaluator(s) for s in (a, b)]
        tickets = [ticket(s) for s in (a, b)]
        for ev, item in zip(evaluators, tickets):
            ev.submit(item)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                ev.submit(item)
            with self.assertRaises(ValueError):
                ev.submit(tickets[1] if item is tickets[0] else tickets[0])
            ev._connect = lambda: None
            ev._receiver = NS(start=lambda: None)
            ev._send_service = lambda message, fd=None: None
            ev._queue.put(None)
            ev._send_loop()  # actual serializer, sealed memfd and pending insertion
            self.assertIsNone(ev._failure)
        self.assertEqual([len(e._pending) for e in evaluators], [1, 1])
        for n in (1, 0):
            ev, item = evaluators[n], tickets[n]
            response = ev.source.wrap(dict(op='UNSCORED', frame_id=4172,
                                          _m4_action=MP.run4_identity(item.envelope)))
            foreign = dict(response, _m4_source=evaluators[1-n].source.tag)
            with self.assertRaises(ValueError):
                ev.accept_result(foreign)
            wrong = copy.deepcopy(response)
            wrong['_m4_action']['session_uuid'] = str(uuid.UUID(int=999))
            with self.assertRaises(ValueError):
                ev.accept_result(wrong)
            with self.assertRaises(ValueError):
                ev.accept_result(dict(response, frame_id=99999))
            self.assertEqual(len(ev._pending), 1)
            _, accepted = ev.accept_result(response)
            ev._emit(accepted, None, W.EvaluatorReason.GROUND_TRUTH_UNAVAILABLE)
            with self.assertRaises(ValueError):
                ev.accept_result(response)
        self.assertEqual([dest for _, dest in self.sent], [b.destination, a.destination])
        with self.assertRaisesRegex(ValueError, 'destination'):
            self.router.deliver(b, self.sent[0][0], a.destination)
        rolled = ticket(a, 4173, str(uuid.UUID(int=300)))
        self.router.reserve(a, rolled)
        with self.assertRaisesRegex(ValueError, 'another source'):
            self.router.reserve(b, ticket(b, 4173, rolled.envelope.session_uuid))
        with self.assertRaises(ValueError):
            dataclasses.replace(a, session_id='foreign-attach').unwrap(a.wrap(dict(op='CLOSE')))

    def test_shared_compute_dispatch_and_terminals(self):
        calls, submitted = [], {s.stream_id: [] for s in self.roster.sources}
        targets = {s.stream_id: NS(submit=lambda t, key=s.stream_id: submitted[key].append(t),
            notify_terminal=lambda *args, key=s.stream_id: submitted[key].append(args)) for s in self.roster.sources}
        contract = load_dynamic_execution_contract()
        def compute(**kwargs):
            calls.append(kwargs['context'].stream_id)
            return [], np.zeros((720, 1280), np.uint8)
        processor = SharedEdgeProcessor(sources=self.roster, evaluators=targets, contract=contract,
            run_id=self.roster.sources[0].run_id, cell_id='cell', service_deadline_s=.17,
            ack_timeout_s=.5, compute=compute)
        for source in self.roster.sources:
            item = ticket(source)
            wire = W.pack_sfd4(item.envelope, item.context)
            result = processor.process(wire, edge_timing={})
            self.assertEqual(result.update['stream_id'], source.stream_id)
            self.assertIsNone(result.evaluation)
        self.assertEqual(calls, [s.stream_id for s in self.roster.sources])
        a = self.roster.sources[0]
        item = ticket(a)
        processor.terminal(W.pack_sfd4(item.envelope, item.context), outcome='SUPERSEDED_PENDING', stage='PENDING',
                           superseded_by_frame_id=4173)
        self.assertEqual([len(submitted[s.stream_id]) for s in self.roster.sources], [2, 1])

    def test_run4_off_anchor_map_source_order_and_late_install(self):
        contract = load_dynamic_execution_contract()
        journal = Run4MapJournal(self.roster, contract, OUT / 'map_observations.jsonl')
        try:
            for source in self.roster.sources:
                item = ticket(source)
                capture_s = item.context.capture_timestamp_ns / 1e9
                document = MP.build_run4_map_update(run_id=source.run_id, cell_id=source.cell_id,
                    stream_id=source.stream_id, frame_id=4172, sequence_id=1,
                    identity=MP.run4_identity(item.envelope), decoder_identity=contract.resolve_q_e4(11, 4321).decoder_identity,
                    capture_timestamp_ns=item.context.capture_timestamp_ns, records=[dict(class_id=0, x=1., y=2.)],
                    service_deadline_at=capture_s + .17, ack_timeout_at=capture_s + .5, edge_timing={}, segmentation={})
                original = copy.deepcopy(document)
                feedback = MP.build_run4_map_feedback(update=document, outcome='RESULT_INSTALLED', terminal=True,
                    install_timestamp=capture_s + .25, map_ingest_at=capture_s + .24, feedback_emit_at=capture_s + .251,
                    map_age_at_install_ms=250.)
                pose = dict(source=source.tag, frame_id=4172, capture_timestamp_ns=item.context.capture_timestamp_ns,
                            provenance='SYNTHETIC_CPU_FIXTURE', ego_world=list(item.context.ego_world.values()))
                with self.assertRaises(ValueError):
                    journal.record(self.roster.sources[1-source.ue_id], document, arrival_ns=1, feedback=feedback, pose=pose)
                row = journal.record(source, document, arrival_ns=item.context.capture_timestamp_ns + 240_000_000,
                                     feedback=feedback, pose=pose)
                self.assertEqual(row['wire_document'], original)
                self.assertIsNone(row['wire_document']['run4_identity']['anchor_action_id'])
                self.assertNotIn('action_id', row['wire_document'])
                with self.assertRaises(ValueError):
                    journal.record(source, document, arrival_ns=1, feedback=feedback, pose=pose)
            self.assertEqual(len(journal.latest), 2)
        finally:
            journal.close()

    def test_independent_actor_contexts_and_rollover(self):
        contexts = independent_policy_contexts(self.roster, '1' * 64)
        a, b = [contexts[s.stream_id] for s in self.roster.sources]
        self.assertIsNot(a['engine'].controller, b['engine'].controller)
        self.assertIsNot(a['telemetry'], b['telemetry'])
        self.assertIs(a['engine']._actor, b['engine']._actor)
        old = b['engine'].controller
        a['engine']._new_session()
        self.assertEqual(len(a['engine'].controllers), 2)
        self.assertEqual(len(b['engine'].controllers), 1)
        self.assertIs(b['engine'].controller, old)
        self.assertEqual(a['engine']._actor.boundary_sha256,
                         '4e28b9ff6b2173f466bc80781e3395f2b5666075d2926e3aa6b7aa848ca0039b')

    def test_both_routes_and_independent_drain_on_error(self):
        lifecycle = BothStreamLifecycle(self.roster)
        lifecycle.complete(self.roster.sources[0])
        self.assertFalse(lifecycle.done)
        lifecycle.complete(self.roster.sources[1])
        lifecycle.drain({s.stream_id: lambda: None for s in self.roster.sources})
        self.assertTrue(lifecycle.done)
        failed = BothStreamLifecycle(self.roster)
        def error():
            raise RuntimeError('fixture failure')
        with self.assertRaises(ValueError):
            failed.drain({self.roster.sources[0].stream_id: error, self.roster.sources[1].stream_id: lambda: None})
        self.assertTrue(failed.drained[self.roster.sources[1].stream_id])


def main():
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=False)
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    prefix = 'rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_ue_telemetry_provider_v2.'
    suite.addTests(unittest.defaultTestLoader.loadTestsFromNames([prefix + name for name in
        ('DciTest.test_rejected_classes_never_enter_cache', 'RlcTest.test_cross_ue_rlc_rows_ignored_and_do_not_close_ticks',
         'DecisionTest.test_cross_session_and_cross_ue')]))
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    write(OUT / 'RESULT.json', dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
                                   log=log.getvalue(), sources=inventory()))
    print(log.getvalue())
    return int(not result.wasSuccessful())


if __name__ == '__main__':
    raise SystemExit(main())
