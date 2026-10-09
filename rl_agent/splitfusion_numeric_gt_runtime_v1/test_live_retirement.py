"""CPU-only live producer terminal contracts; durable receipts are synthetic.

No CARLA/OAI route, scorer process, archive child, or radio ACK is started here.
The tests check producer/owner state transitions, not live qualification.
"""
import os
import unittest
import uuid
from unittest.mock import patch

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2.test_ownership import plan_for, advance
from rl_agent.splitfusion_numeric_gt_integration_v2.test_ownership import context_for
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import reward_hold_controller_v2 as R
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W

from . import producer as G
from .live_clock import LiveClockLedger, read_boot_id, DEADLINE_NS
from .test_producer import fixture, prepare


def live_frame(engine, kwargs):
    pending = prepare(engine, kwargs, archive_clocks=None, payload_filenames=None,
                      prescore_filename=None, replay_observed_raw_ns=None)
    engine.confirm_send(pending.token, plan_for(kwargs)[0])
    engine.publish_ready(pending.token)
    return pending


def valid_wire_frame(engine, kwargs):
    """Publish a normal fixture frame with a real wire-compatible session UUID."""
    plan = C._parse(plan_for(kwargs)[0])
    plan["action"]["session_uuid"] = str(uuid.UUID("12345678-1234-4234-8234-123456789abc"))
    raw = C.canonical(plan)
    pending = engine.prepare(raw, kwargs, frame_context=context_for(kwargs),
        population=kwargs["dynamic_snapshot_evidence"], archive_clocks=None,
        payload_filenames=None, prescore_filename=None, replay_observed_raw_ns=None)
    engine.confirm_send(pending.token, raw)
    engine.publish_ready(pending.token)
    return pending


def fault_clock_for(engine, pending, packet, fault_reason):
    row = engine._frames[pending.token.sequence]
    staged, _, avo, _ = engine._owner._archive_for_owner(row.owner_token)
    ledger = LiveClockLedger(pending.identity_bytes, pending.semantic_sha256)
    boot, pid = read_boot_id(), os.getpid()
    def add(role, name, stamp):
        ledger.record(role=role, name=name, raw_ns=stamp, pid=pid, boot_id=boot,
            ready_identity_bytes=pending.identity_bytes, semantic_sha256=pending.semantic_sha256)
    start = row.events["prepare_start_raw_ns"]
    add("collector", "owned_tick_snapshot", start - 300)
    add("producer", "source_stage_received", start - 200)
    add("ue", "action_open", start - 100)
    add("producer", "prepare_start", start)
    add("producer", "source_alignment_observed", staged.sync["observed_raw_ns"])
    add("producer", "numeric_ready", row.events["numerical_ready_raw_ns"])
    add("ue", "last_datagram_sent", row.events["numerical_ready_raw_ns"] + 1)
    published = row.events["ready_published_raw_ns"]
    add("producer", "ready_published", published)
    add("coordinator", "feedback_emitted", published + 1)
    add("ue", "feedback_received", published + 2)
    return ledger.archive_descriptor(avo_sha256=C.sha256(G._canonical(avo.avo)),
        source_synchronization_sha256=C.sha256(G._canonical(staged.sync)),
        fault_feedback_sha256=C.sha256(packet), fault_reason=fault_reason)


def clock_for(engine, pending):
    row = engine._frames[pending.token.sequence]
    staged, _, avo, _ = engine._owner._archive_for_owner(row.owner_token)
    ledger = LiveClockLedger(pending.identity_bytes, pending.semantic_sha256)
    boot, pid = read_boot_id(), os.getpid()
    def add(role, name, stamp):
        ledger.record(role=role, name=name, raw_ns=stamp, pid=pid, boot_id=boot,
            ready_identity_bytes=pending.identity_bytes, semantic_sha256=pending.semantic_sha256)
    start = row.events["prepare_start_raw_ns"]
    add("collector", "owned_tick_snapshot", start - 300)
    add("producer", "source_stage_received", start - 200)
    add("ue", "action_open", start - 100)
    add("producer", "prepare_start", start)
    add("producer", "source_alignment_observed", staged.sync["observed_raw_ns"])
    add("producer", "numeric_ready", row.events["numerical_ready_raw_ns"])
    add("ue", "last_datagram_sent", row.events["numerical_ready_raw_ns"] + 1)
    add("producer", "ready_published", row.events["ready_published_raw_ns"])
    add("ue", "feedback_timeout", start - 100 + DEADLINE_NS + 1)
    return ledger.archive_descriptor(avo_sha256=C.sha256(G._canonical(avo.avo)),
        source_synchronization_sha256=C.sha256(G._canonical(staged.sync)))


def receipt_for(pending, terminal, prefix="frame_100/terminal"):
    return dict(status="COMPLETE", durable=True, error=None, job_id=3,
        identity=C._parse(pending.identity_bytes)["capture_identity"],
        written=[dict(root=0, relative=prefix + "/" + name, sha256=C.sha256(raw),
                      bytes=len(raw), ready=False) for name, raw in terminal.files.items()])


class LiveRetirementTests(unittest.TestCase):
    def test_actual_unscored_fault_packet_retired_without_quality_or_fake_score(self):
        with fixture(offline_replay=False) as (engine, kwargs, _):
            pending = valid_wire_frame(engine, kwargs)
            identity = C._parse(pending.identity_bytes)
            action = identity["action_identity"]
            feedback = R.RewardFeedbackV2(**{name: action[name] for name in (
                "session_uuid", "controller_lineage_sha256", "decision_seq", "ticket_seq",
                "frame_id", "tensor_seq", "mode_id", "q_e4", "execution_bundle_sha256",
                "anchor_action_id", "reward_requested")},
                capture_timestamp_ns=identity["capture_identity"]["capture_timestamp_ns"],
                kind="EVALUATOR_FAULT", q_perc=None)
            packet = W.encode_feedback(feedback, W.EvaluatorReason.GROUND_TRUTH_UNAVAILABLE)
            clock = fault_clock_for(engine, pending, packet, "GROUND_TRUTH_UNAVAILABLE")
            with self.assertRaisesRegex(G.ProducerError, "UE-observed fault packet"):
                engine.prepare_live_unscored_terminal(pending.token, pending.identity_bytes,
                    pending.semantic_sha256, clock, "UNSCORED_EVALUATOR_FAULT", packet, None)
            with self.assertRaisesRegex(G.ProducerError, "exact bounded emitted packet"):
                engine.prepare_live_unscored_terminal(pending.token, pending.identity_bytes,
                    pending.semantic_sha256, clock, "UNSCORED_EVALUATOR_FAULT", packet[:-1],
                    C.sha256(packet[:-1]))
            terminal = engine.prepare_live_unscored_terminal(pending.token,
                pending.identity_bytes, pending.semantic_sha256, clock,
                "UNSCORED_EVALUATOR_FAULT", packet, C.sha256(packet))
            doc = C._parse(terminal.files["unscored_terminal.json"])
            self.assertEqual(doc["terminal"], "FEEDBACK_ON_TIME")
            self.assertEqual(doc["fault_feedback_kind"], "EVALUATOR_FAULT")
            self.assertEqual(doc["fault_feedback_reason"], "GROUND_TRUTH_UNAVAILABLE")
            self.assertTrue(doc["feedback_receipt_claim"])
            self.assertFalse(doc["quality_feedback_claim"])
            self.assertFalse(doc["score_exists"])
            result = engine.complete_live_unscored(pending.token, pending.identity_bytes,
                pending.semantic_sha256, terminal.terminal_sha256,
                receipt_for(pending, terminal), "frame_100/terminal")
            self.assertEqual(result["terminal"], "FEEDBACK_ON_TIME")
            self.assertEqual(engine.snapshot()["events"]["1"]["outcome"],
                             "LIVE_UNSCORED_FAULT_DURABLE")
            self.assertEqual(engine.snapshot()["retained_jobs"], 0)

    def test_never_scored_timeout_preserves_committed_history_and_no_feedback_claim(self):
        with fixture(offline_replay=False) as (engine, kwargs, _):
            pending = live_frame(engine, kwargs)
            history = engine._owner._producer.history_sha256()
            descriptor = clock_for(engine, pending)
            with patch.object(engine._owner, "complete_scoring", side_effect=AssertionError("fake score")), \
                 patch.object(engine._owner, "confirm_feedback", side_effect=AssertionError("fake feedback")):
                terminal = engine.prepare_live_unscored_terminal(pending.token,
                    pending.identity_bytes, pending.semantic_sha256, descriptor,
                    "NO_SCORE_BY_UE_DEADLINE")
                self.assertIs(terminal, engine.prepare_live_unscored_terminal(pending.token,
                    pending.identity_bytes, pending.semantic_sha256, descriptor,
                    "NO_SCORE_BY_UE_DEADLINE"))
                self.assertEqual(set(terminal.files), {"live_clock.json", "unscored_terminal.json"})
                document = C._parse(terminal.files["unscored_terminal.json"])
                self.assertFalse(document["score_exists"])
                self.assertFalse(document["feedback_receipt_claim"])
                self.assertEqual(document["ready_identity"], C._parse(pending.identity_bytes))
                self.assertEqual(document["deadline"]["terminal"], "FEEDBACK_TIMEOUT")
                with self.assertRaises(TypeError):
                    terminal.files["other"] = b"no"
                engine.complete_live_unscored(pending.token, pending.identity_bytes,
                    pending.semantic_sha256, terminal.terminal_sha256, receipt_for(pending, terminal),
                    "frame_100/terminal")
            self.assertEqual(engine._owner._producer.history_sha256(), history)
            self.assertEqual(engine.snapshot()["retained_jobs"], 0)
            self.assertEqual(engine.snapshot()["owner"]["released"], 1)
            self.assertTrue(engine.snapshot()["valid"])
            self.assertEqual(engine.snapshot()["events"]["1"]["outcome"],
                             "LIVE_UNSCORED_TIMEOUT_DURABLE")
            advance(kwargs)
            following = prepare(engine, kwargs, archive_clocks=None,
                payload_filenames=None, prescore_filename=None, replay_observed_raw_ns=None)
            self.assertEqual(following.token.sequence, 2)
            engine.cancel(following.token, plan_for(kwargs)[0], "next frame not sent in unit test")
            self.assertTrue(engine.snapshot()["valid"])

    def test_unscored_refuses_foreign_clock_and_nondurable_receipt(self):
        with fixture(offline_replay=False) as (engine, kwargs, _):
            pending = live_frame(engine, kwargs)
            descriptor = clock_for(engine, pending)
            changed = C._parse(descriptor); changed["semantic_sha256"] = "0" * 64
            with self.assertRaisesRegex(G.ProducerError, "descriptor differs"):
                engine.prepare_live_unscored_terminal(pending.token, pending.identity_bytes,
                    pending.semantic_sha256, C.canonical(changed), "NO_SCORE_BY_UE_DEADLINE")
            terminal = engine.prepare_live_unscored_terminal(pending.token,
                pending.identity_bytes, pending.semantic_sha256, descriptor, "NO_SCORE_BY_UE_DEADLINE")
            bad = receipt_for(pending, terminal); bad["durable"] = False
            with self.assertRaisesRegex(G.ProducerError, "durable"):
                engine.complete_live_unscored(pending.token, pending.identity_bytes,
                    pending.semantic_sha256, terminal.terminal_sha256, bad, "frame_100/terminal")
            self.assertEqual(engine.snapshot()["retained_jobs"], 1)
            bad = receipt_for(pending, terminal); bad["written"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(G.ProducerError, "bytes/hash"):
                engine.complete_live_unscored(pending.token, pending.identity_bytes,
                    pending.semantic_sha256, terminal.terminal_sha256, bad, "frame_100/terminal")
            self.assertEqual(engine.snapshot()["retained_jobs"], 1)

    def test_scored_requires_bound_scorer_durable_proof_without_feedback_call(self):
        with fixture(offline_replay=False) as (engine, kwargs, _):
            pending = live_frame(engine, kwargs)
            engine.score_completed(pending.token, pending.identity_bytes, pending.semantic_sha256)
            proof = dict(status="DURABLE", terminal="FEEDBACK_TIMEOUT",
                archive_manifest_sha256="a" * 64, sidecar_sha256="b" * 64,
                ready_identity_sha256=C.sha256(pending.identity_bytes),
                ready_semantic_sha256=pending.semantic_sha256, live_qualified=False)
            bad = dict(proof, ready_identity_sha256="0" * 64)
            with self.assertRaisesRegex(G.ProducerError, "scorer durable proof"):
                engine.complete_live_scored(pending.token, pending.identity_bytes,
                                            pending.semantic_sha256, bad)
            with patch.object(engine._owner, "confirm_feedback", side_effect=AssertionError("fake feedback")):
                engine.complete_live_scored(pending.token, pending.identity_bytes,
                                            pending.semantic_sha256, proof)
            self.assertEqual(engine.snapshot()["retained_jobs"], 0)
            self.assertEqual(engine.snapshot()["events"]["1"]["outcome"], "LIVE_SCORED_DURABLE")
            self.assertEqual(engine.snapshot()["events"]["1"]["live_terminal"], "FEEDBACK_TIMEOUT")
            self.assertTrue(engine.snapshot()["valid"])

    def test_ordinary_unsent_cancel_preserves_history_without_timeout_artifact(self):
        with fixture(offline_replay=False) as (engine, kwargs, _):
            before = engine._owner._producer.history_sha256()
            pending = prepare(engine, kwargs, archive_clocks=None, payload_filenames=None,
                prescore_filename=None, replay_observed_raw_ns=None)
            engine.cancel(pending.token, plan_for(kwargs)[0], "not transmitted")
            self.assertEqual(engine._owner._producer.history_sha256(), before)
            self.assertEqual(engine.snapshot()["retained_jobs"], 0)
            self.assertTrue(engine.snapshot()["valid"])


if __name__ == "__main__":
    unittest.main()
