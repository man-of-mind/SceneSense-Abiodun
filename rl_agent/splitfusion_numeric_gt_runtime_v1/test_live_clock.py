"""CPU-only contract checks for observed live clocks; no live service starts."""
import copy
import json
import unittest

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2.test_scoring import fixture
from rl_agent.splitfusion_minimal_quality_scorer_v1.scorer import NumericalScorer

from .live_clock import LiveClockLedger, LiveClockError, DEADLINE_NS, SCHEMA, COMPLETION_SCHEMA


BOOT_A = "11111111-1111-4111-8111-111111111111"
BOOT_B = "22222222-2222-4222-8222-222222222222"
BASE = 1_000_000_000_000
PIDS = dict(collector=101, producer=102, scorer=103, ue=104, archive_client=105,
            coordinator=106)


class LiveClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        case = fixture(NumericalScorer())
        cls.identity = case["target"].identity_bytes
        cls.semantic = case["target"].semantic_sha256
        cls.capture = C._parse(cls.identity)["capture_identity"]

    def setUp(self):
        self.ledger = LiveClockLedger(self.identity, self.semantic)

    def event(self, role, name, offset, *, boot=BOOT_A, pid=None, identity=None, semantic=None):
        return self.ledger.record(role=role, name=name, raw_ns=BASE + offset,
            pid=PIDS[role] if pid is None else pid, boot_id=boot,
            ready_identity_bytes=self.identity if identity is None else identity,
            semantic_sha256=self.semantic if semantic is None else semantic)

    def path(self, *, feedback_offset=100 + DEADLINE_NS, scorer_boot=BOOT_A,
             archive_start_offset=None):
        self.event("collector", "owned_tick_snapshot", 10)
        self.event("producer", "source_stage_received", 50)
        self.event("ue", "action_open", 100)
        self.event("producer", "prepare_start", 110)
        self.event("producer", "source_alignment_observed", 115)
        self.event("producer", "numeric_ready", 120)
        self.event("ue", "last_datagram_sent", 130)
        self.event("producer", "ready_published", 135)
        self.event("scorer", "prediction_admitted", 146, boot=scorer_boot)
        self.event("scorer", "target_admitted", 150, boot=scorer_boot)
        self.event("scorer", "score_start", 160, boot=scorer_boot)
        self.event("scorer", "score_end", 170, boot=scorer_boot)
        self.event("coordinator", "feedback_emitted", 180)
        if feedback_offset is None or feedback_offset > 100 + DEADLINE_NS:
            self.event("ue", "feedback_timeout", 100 + DEADLINE_NS + 1)
            if feedback_offset is not None:
                self.event("ue", "late_feedback_observed", feedback_offset)
        else:
            self.event("ue", "feedback_received", feedback_offset)
        end = 100 + DEADLINE_NS if feedback_offset is None else feedback_offset
        self.event("scorer", "archive_build_start",
                   end + 1 if archive_start_offset is None else archive_start_offset,
                   boot=scorer_boot)

    def test_inclusive_deadline_from_actual_ue_receipt_and_separate_archive(self):
        self.path()
        self.assertTrue(self.ledger.deadline()["within_deadline"])
        self.assertEqual(self.ledger.deadline()["elapsed_ns"], DEADLINE_NS)
        raw = self.ledger.archive_descriptor(avo_sha256="a" * 64,
                                              source_synchronization_sha256="b" * 64)
        doc = json.loads(raw)
        self.assertEqual(doc["schema"], SCHEMA)
        self.assertEqual(doc["ready_identity"], C._parse(self.identity))
        self.assertEqual(doc["semantic_sha256"], self.semantic)
        self.assertEqual(doc["avo_sha256"], "a" * 64)
        self.assertNotIn("start_raw_ns", doc)
        self.assertNotIn("write_raw_ns", doc)
        self.assertNotIn("prescore_input_ready_raw_ns", doc)
        self.assertNotIn("AVO_PRE_SCORE_INPUTS", raw.decode())
        with self.assertRaisesRegex(LiveClockError, "create-only"):
            self.ledger.archive_descriptor(avo_sha256="a" * 64,
                                           source_synchronization_sha256="b" * 64)
        self.event("scorer", "archive_build_end", 100 + DEADLINE_NS + 2)
        after = 100 + DEADLINE_NS + 4
        self.event("archive_client", "archive_submitted", after)
        receipt = dict(status="COMPLETE", durable=True, error=None, identity=self.capture,
            written=[dict(root=0, relative="frame_10/live_clock.json", sha256=C.sha256(raw),
                          bytes=len(raw), ready=False)])
        done = self.ledger.durability_record(descriptor_bytes=raw, receipt=receipt,
            raw_ns=BASE + after + 1, pid=PIDS["archive_client"], boot_id=BOOT_A,
            ready_identity_bytes=self.identity, semantic_sha256=self.semantic)
        self.assertEqual(json.loads(done)["schema"], COMPLETION_SCHEMA)
        self.assertEqual(json.loads(done)["clock_descriptor_sha256"], C.sha256(raw))
        self.assertNotIn("durable_completion_observed", raw.decode())
        with self.assertRaisesRegex(LiveClockError, "completed"):
            self.event("archive_client", "archive_submitted", after + 2)

    def test_one_nanosecond_late_is_not_timely(self):
        self.path(feedback_offset=100 + DEADLINE_NS + 1)
        result = self.ledger.deadline()
        self.assertFalse(result["within_deadline"])
        self.assertEqual(result["terminal"], "FEEDBACK_TIMEOUT")
        self.assertIsNone(result["feedback_received_raw_ns"])
        self.assertEqual(result["late_feedback_elapsed_ns"], DEADLINE_NS + 1)
        raw = self.ledger.archive_descriptor(avo_sha256="a" * 64,
            source_synchronization_sha256="b" * 64)
        late = json.loads(self.ledger.late_feedback_sidecar())
        self.assertEqual(late["clock_descriptor_sha256"], C.sha256(raw))
        self.assertEqual(late["terminal"], "FEEDBACK_TIMEOUT")

    def test_missing_feedback_uses_observed_timeout_without_invented_receipt(self):
        self.path(feedback_offset=None)
        result = self.ledger.deadline()
        self.assertEqual(result["terminal"], "FEEDBACK_TIMEOUT")
        self.assertIsNone(result["feedback_received_raw_ns"])
        self.assertIsNone(result["elapsed_ns"])
        self.assertFalse(result["within_deadline"])
        self.assertEqual(json.loads(self.ledger.archive_descriptor(avo_sha256="a" * 64,
            source_synchronization_sha256="b" * 64))["deadline"], result)

    def test_unscored_timeout_is_archivable_without_fake_scorer_events(self):
        self.event("collector", "owned_tick_snapshot", 10)
        self.event("producer", "source_stage_received", 50)
        self.event("ue", "action_open", 100)
        self.event("producer", "prepare_start", 110)
        self.event("producer", "source_alignment_observed", 115)
        self.event("producer", "numeric_ready", 120)
        self.event("ue", "last_datagram_sent", 130)
        self.event("producer", "ready_published", 135)
        self.event("ue", "feedback_timeout", 100 + DEADLINE_NS + 1)
        with self.assertRaisesRegex(LiveClockError, "timely receipt conflicts"):
            self.event("ue", "feedback_received", 100 + DEADLINE_NS + 1)
        doc = json.loads(self.ledger.archive_descriptor(avo_sha256="a" * 64,
            source_synchronization_sha256="b" * 64))
        self.assertEqual(doc["scoring_state"], "NOT_SCORED_AT_TIMEOUT")
        self.assertEqual(doc["deadline"]["terminal"], "FEEDBACK_TIMEOUT")
        raw = self.ledger._descriptor_bytes
        self.event("archive_client", "archive_submitted", 100 + DEADLINE_NS + 3)
        receipt = dict(status="COMPLETE", durable=True, error=None, identity=self.capture,
            written=[dict(root=0, relative="frame_10/live_clock.json", sha256=C.sha256(raw),
                          bytes=len(raw), ready=False)])
        done = json.loads(self.ledger.durability_record(descriptor_bytes=raw, receipt=receipt,
            raw_ns=BASE + 100 + DEADLINE_NS + 4, pid=PIDS["archive_client"], boot_id=BOOT_A,
            ready_identity_bytes=self.identity, semantic_sha256=self.semantic))
        self.assertEqual(done["archive_scope"], "UNSCORED_TERMINAL_SIDECAR_ONLY")
        self.assertIsNone(done["archive_build_end_event"])

    def test_actual_unscored_fault_receipt_is_not_a_quality_score(self):
        self.event("collector", "owned_tick_snapshot", 10)
        self.event("producer", "source_stage_received", 50)
        self.event("ue", "action_open", 100)
        self.event("producer", "prepare_start", 110)
        self.event("producer", "source_alignment_observed", 115)
        self.event("producer", "numeric_ready", 120)
        self.event("ue", "last_datagram_sent", 130)
        self.event("producer", "ready_published", 135)
        self.event("coordinator", "feedback_emitted", 140)
        self.event("ue", "feedback_received", 100 + DEADLINE_NS)
        packet_sha = "c" * 64
        raw = self.ledger.archive_descriptor(avo_sha256="a" * 64,
            source_synchronization_sha256="b" * 64,
            fault_feedback_sha256=packet_sha,
            fault_reason="GROUND_TRUTH_UNAVAILABLE")
        doc = C._parse(raw)
        self.assertEqual(doc["scoring_state"], "UNSCORED_FAULT_FEEDBACK_EMITTED")
        self.assertEqual(doc["deadline"]["terminal"], "FEEDBACK_ON_TIME")
        self.assertEqual(doc["unscored_fault"], dict(feedback_packet_sha256=packet_sha,
            reason="GROUND_TRUTH_UNAVAILABLE", score_exists=False, quality_value=None))
        self.assertNotIn("score_end", [event["name"] for event in doc["events"]])

    def test_unscored_fault_cannot_claim_receipt_without_emission_and_proof(self):
        self.event("collector", "owned_tick_snapshot", 10)
        self.event("producer", "source_stage_received", 50)
        self.event("ue", "action_open", 100)
        self.event("producer", "prepare_start", 110)
        self.event("producer", "source_alignment_observed", 115)
        self.event("producer", "numeric_ready", 120)
        self.event("ue", "last_datagram_sent", 130)
        self.event("producer", "ready_published", 135)
        self.event("ue", "feedback_received", 140)
        with self.assertRaisesRegex(LiveClockError, "actual score or fault emission"):
            self.ledger.archive_descriptor(avo_sha256="a" * 64,
                source_synchronization_sha256="b" * 64)
        self.event("coordinator", "feedback_emitted", 139)
        with self.assertRaisesRegex(LiveClockError, "fault packet digest/reason"):
            self.ledger.archive_descriptor(avo_sha256="a" * 64,
                source_synchronization_sha256="b" * 64)

    def test_event_identity_semantic_role_owner_and_order_fail_closed(self):
        wrong = C._parse(self.identity)
        wrong["run_binding"] = "9" * 64
        with self.assertRaisesRegex(LiveClockError, "foreign"):
            self.event("ue", "action_open", 100, identity=C.canonical(wrong))
        with self.assertRaisesRegex(LiveClockError, "foreign"):
            self.event("ue", "action_open", 100, semantic="f" * 64)
        with self.assertRaisesRegex(LiveClockError, "unknown"):
            self.event("ue", "score_start", 100)
        with self.assertRaisesRegex(LiveClockError, "boot"):
            self.event("ue", "action_open", 100, boot="not-a-boot")
        self.event("ue", "action_open", 100)
        with self.assertRaisesRegex(LiveClockError, "duplicate"):
            self.event("ue", "action_open", 101)
        with self.assertRaisesRegex(LiveClockError, "process owner"):
            self.event("ue", "last_datagram_sent", 101, pid=999)
        with self.assertRaisesRegex(LiveClockError, "process owner"):
            self.event("ue", "last_datagram_sent", 101, boot=BOOT_B)
        with self.assertRaisesRegex(LiveClockError, "regressed"):
            self.event("ue", "last_datagram_sent", 99)
        self.event("ue", "last_datagram_sent", 101)

    def test_stage_handoff_must_precede_action_open_and_timeout_is_strictly_late(self):
        self.event("collector", "owned_tick_snapshot", 10)
        self.event("ue", "action_open", 100)
        with self.assertRaisesRegex(LiveClockError, "causal"):
            self.event("producer", "source_stage_received", 101)
        self.event("producer", "source_stage_received", 99)
        with self.assertRaisesRegex(LiveClockError, "precedes deadline"):
            self.event("ue", "feedback_timeout", 100 + DEADLINE_NS)
        self.event("ue", "feedback_timeout", 100 + DEADLINE_NS + 1)
        self.event("ue", "late_feedback_observed", 100 + DEADLINE_NS + 1)
        self.assertEqual(self.ledger.deadline()["late_feedback_elapsed_ns"], DEADLINE_NS + 1)

    def test_same_boot_archive_build_cannot_precede_actual_ue_terminal(self):
        for feedback in (100 + DEADLINE_NS, None):
            with self.subTest(feedback=feedback):
                self.ledger = LiveClockLedger(self.identity, self.semantic)
                terminal = 100 + DEADLINE_NS if feedback is not None else 101 + DEADLINE_NS
                with self.assertRaisesRegex(LiveClockError, "causal"):
                    self.path(feedback_offset=feedback, archive_start_offset=terminal - 1)
                # Rejected future-claim did not occupy scorer's archive event.
                self.event("scorer", "archive_build_start", terminal + 1)
                document = json.loads(self.ledger.archive_descriptor(avo_sha256="a" * 64,
                    source_synchronization_sha256="b" * 64))
                self.assertEqual(document["scoring_state"], "SCORED_FEEDBACK_EMITTED")

    def test_cross_boot_archive_order_is_not_fabricated(self):
        self.path(scorer_boot=BOOT_B, archive_start_offset=DEADLINE_NS)
        document = json.loads(self.ledger.archive_descriptor(avo_sha256="a" * 64,
            source_synchronization_sha256="b" * 64))
        self.assertTrue(document["incomparable_boot_pairs"])

    def test_same_boot_causality_rejects_future_target_without_mutation(self):
        self.event("scorer", "target_admitted", 1000)
        with self.assertRaisesRegex(LiveClockError, "causal"):
            self.event("producer", "ready_published", 1001)
        # Rejection did not install an event or reserve the producer role.
        self.event("producer", "ready_published", 999)
        self.event("scorer", "score_start", 1001)
        self.event("scorer", "score_end", 1002)
        self.event("coordinator", "feedback_emitted", 1003)
        self.event("ue", "action_open", 990)
        with self.assertRaisesRegex(LiveClockError, "causal"):
            self.event("ue", "feedback_received", 1002)

    def test_different_boot_is_marked_incomparable_not_subtracted(self):
        self.path(scorer_boot=BOOT_B)
        doc = json.loads(self.ledger.archive_descriptor(avo_sha256="a" * 64,
            source_synchronization_sha256="b" * 64))
        self.assertTrue(doc["incomparable_boot_pairs"])
        self.assertEqual(self.ledger.deadline()["elapsed_ns"], DEADLINE_NS)
        # Even a scorer timestamp earlier than the UE clock can be retained as
        # separate evidence when it belongs to another boot domain.
        other = LiveClockLedger(self.identity, self.semantic)
        other.record(role="coordinator", name="feedback_emitted", raw_ns=10, pid=1, boot_id=BOOT_B,
            ready_identity_bytes=self.identity, semantic_sha256=self.semantic)
        other.record(role="ue", name="action_open", raw_ns=100, pid=2, boot_id=BOOT_A,
            ready_identity_bytes=self.identity, semantic_sha256=self.semantic)
        other.record(role="ue", name="feedback_received", raw_ns=200, pid=2, boot_id=BOOT_A,
            ready_identity_bytes=self.identity, semantic_sha256=self.semantic)
        self.assertEqual(other.deadline()["elapsed_ns"], 100)

    def test_durable_record_rejects_missing_submission_foreign_receipt_and_duplicate(self):
        self.path()
        raw = self.ledger.archive_descriptor(avo_sha256="a" * 64,
            source_synchronization_sha256="b" * 64)
        receipt = dict(status="COMPLETE", durable=True, error=None, identity=self.capture,
            written=[dict(root=0, relative="frame_10/live_clock.json", sha256=C.sha256(raw),
                          bytes=len(raw), ready=False)])
        args = dict(descriptor_bytes=raw, receipt=receipt, raw_ns=BASE + DEADLINE_NS + 120,
            pid=PIDS["archive_client"], boot_id=BOOT_A,
            ready_identity_bytes=self.identity, semantic_sha256=self.semantic)
        with self.assertRaisesRegex(LiveClockError, "build completion"):
            self.ledger.durability_record(**args)
        self.event("scorer", "archive_build_end", DEADLINE_NS + 113)
        with self.assertRaisesRegex(LiveClockError, "submission"):
            self.ledger.durability_record(**args)
        self.event("archive_client", "archive_submitted", DEADLINE_NS + 115)
        with self.assertRaisesRegex(LiveClockError, "verified archive receipt"):
            self.event("archive_client", "durable_completion_observed", DEADLINE_NS + 119)
        foreign = copy.deepcopy(receipt)
        foreign["identity"]["frame_id"] += 1
        with self.assertRaisesRegex(LiveClockError, "another capture"):
            self.ledger.durability_record(**dict(args, receipt=foreign))
        failed = dict(receipt, durable=False, status="FAILED")
        with self.assertRaisesRegex(LiveClockError, "durable COMPLETE"):
            self.ledger.durability_record(**dict(args, receipt=failed))
        altered = copy.deepcopy(receipt)
        altered["written"][0]["sha256"] = "e" * 64
        with self.assertRaisesRegex(LiveClockError, "exact live clock sidecar"):
            self.ledger.durability_record(**dict(args, receipt=altered))
        self.ledger.durability_record(**args)
        with self.assertRaisesRegex(LiveClockError, "create-only"):
            self.ledger.durability_record(**args)


if __name__ == "__main__":
    unittest.main()
