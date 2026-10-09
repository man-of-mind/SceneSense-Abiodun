"""CPU-only tests for the explicit post-send numerical source handoff V2."""
from __future__ import annotations

import copy
import unittest

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2.test_scoring import fixture
from rl_agent.splitfusion_minimal_quality_scorer_v1.scorer import NumericalScorer

from . import edge_live as E
from . import live_archive as A
from .live_clock import (ASYNC_HANDOFF_ID, ASYNC_SCHEMA, DEADLINE_NS,
                         LiveClockError, LiveClockLedger, SCHEMA)


BOOT = "11111111-1111-4111-8111-111111111111"
OTHER_BOOT = "22222222-2222-4222-8222-222222222222"
BASE = 1_000_000_000_000
PID = {"collector": 101, "ue": 101, "producer": 102,
       "scorer": 103, "coordinator": 104}


class AsyncLiveClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        case = fixture(NumericalScorer())
        cls.identity = case["target"].identity_bytes
        cls.semantic = case["target"].semantic_sha256

    def ledger(self):
        return LiveClockLedger(self.identity, self.semantic,
            handoff_id=ASYNC_HANDOFF_ID)

    def event(self, ledger, role, name, offset, *, boot=BOOT):
        return ledger.record(role=role, name=name, raw_ns=BASE + offset,
            pid=PID[role], boot_id=boot, ready_identity_bytes=self.identity,
            semantic_sha256=self.semantic)

    def source_path(self, ledger):
        self.event(ledger, "collector", "owned_tick_snapshot", 10)
        self.event(ledger, "ue", "source_owned", 20)
        self.event(ledger, "ue", "action_open", 100)
        self.event(ledger, "ue", "last_datagram_sent", 130)
        self.event(ledger, "producer", "source_stage_received", 150)
        self.event(ledger, "producer", "prepare_start", 160)
        self.event(ledger, "producer", "source_alignment_observed", 170)
        self.event(ledger, "producer", "numeric_ready", 180)
        self.event(ledger, "producer", "ready_published", 190)

    def test_v2_post_send_source_then_exact_inclusive_feedback(self):
        ledger = self.ledger()
        self.source_path(ledger)
        self.event(ledger, "scorer", "prediction_admitted", 200)
        self.event(ledger, "scorer", "target_admitted", 210)
        self.event(ledger, "scorer", "score_start", 220)
        self.event(ledger, "scorer", "score_end", 230)
        self.event(ledger, "coordinator", "feedback_emitted", 240)
        self.event(ledger, "ue", "feedback_received", 100 + DEADLINE_NS)
        self.event(ledger, "scorer", "archive_build_start", 100 + DEADLINE_NS + 1)
        raw = ledger.archive_descriptor(avo_sha256="a" * 64,
            source_synchronization_sha256="b" * 64)
        document = C._parse(raw)
        self.assertEqual(document["schema"], ASYNC_SCHEMA)
        self.assertEqual(document["handoff_id"], ASYNC_HANDOFF_ID)
        self.assertTrue(document["deadline"]["within_deadline"])
        self.assertEqual(document["deadline"]["elapsed_ns"], DEADLINE_NS)
        self.assertEqual([item["name"] for item in document["events"]].count("source_owned"), 1)
        replay = self.ledger()
        for item in document["events"]:
            self.event(replay, item["role"], item["name"], item["raw_ns"] - BASE)
        self.assertEqual(replay.archive_descriptor(avo_sha256="a" * 64,
            source_synchronization_sha256="b" * 64), raw)

    def test_v2_unscored_timeout_has_no_fabricated_score(self):
        ledger = self.ledger()
        self.source_path(ledger)
        self.event(ledger, "ue", "feedback_timeout", 100 + DEADLINE_NS + 1)
        document = C._parse(ledger.archive_descriptor(avo_sha256="a" * 64,
            source_synchronization_sha256="b" * 64))
        self.assertEqual(document["schema"], ASYNC_SCHEMA)
        self.assertEqual(document["scoring_state"], "NOT_SCORED_AT_TIMEOUT")
        self.assertIsNone(document["deadline"]["feedback_received_raw_ns"])

    def test_scored_archive_accepts_v2_descriptor_without_changing_score(self):
        # Synthetic RAW instants exercise the archive reader; this is not a
        # live timing or performance observation.
        from .test_live_archive import LiveArchiveTests
        LiveArchiveTests.setUpClass()
        _, kwargs = LiveArchiveTests().values()
        sync = kwargs["source_synchronization"]
        start = sync["observed_raw_ns"] - 2_000_000
        ledger = LiveClockLedger(kwargs["ready_target"].identity_bytes,
            kwargs["ready_target"].semantic_sha256, handoff_id=ASYNC_HANDOFF_ID)
        events = (("collector", "owned_tick_snapshot", start),
                  ("ue", "source_owned", start + 100_000),
                  ("ue", "action_open", start + 200_000),
                  ("ue", "last_datagram_sent", start + 300_000),
                  ("producer", "source_stage_received", start + 400_000),
                  ("producer", "prepare_start", start + 500_000),
                  ("producer", "source_alignment_observed", sync["observed_raw_ns"]),
                  ("producer", "numeric_ready", start + 3_000_000),
                  ("producer", "ready_published", start + 4_000_000),
                  ("scorer", "target_admitted", start + 5_000_000),
                  ("scorer", "prediction_admitted", start + 6_000_000),
                  ("scorer", "score_start", start + 7_000_000),
                  ("scorer", "score_end", start + 8_000_000),
                  ("coordinator", "feedback_emitted", start + 9_000_000),
                  ("ue", "feedback_received", start + 10_000_000),
                  ("scorer", "archive_build_start", start + 11_000_000))
        for role, name, stamp in events:
            ledger.record(role=role, name=name, raw_ns=stamp,
                pid=PID[role], boot_id=BOOT,
                ready_identity_bytes=kwargs["ready_target"].identity_bytes,
                semantic_sha256=kwargs["ready_target"].semantic_sha256)
        archive = A.build_live_archive(**dict(kwargs, clock_ledger=ledger))
        self.assertEqual(A.verify_live_archive(archive)["schema"], A.SCHEMA)
        self.assertEqual(C._parse(archive.files["live_clock.json"])["schema"], ASYNC_SCHEMA)
        self.assertEqual(C._parse(archive.files["numerical_score.json"])["q_perc"], kwargs["q_perc"])

    def test_v2_requires_real_preaction_owner_and_postsend_producer(self):
        ledger = self.ledger()
        self.event(ledger, "collector", "owned_tick_snapshot", 10)
        with self.assertRaisesRegex(LiveClockError, "actual predecessor"):
            self.event(ledger, "ue", "action_open", 100)
        self.event(ledger, "ue", "source_owned", 20)
        self.event(ledger, "ue", "action_open", 100)
        self.event(ledger, "ue", "last_datagram_sent", 130)
        with self.assertRaisesRegex(LiveClockError, "causal"):
            self.event(ledger, "producer", "source_stage_received", 129)
        self.event(ledger, "producer", "source_stage_received", 150)
        with self.assertRaisesRegex(LiveClockError, "regressed|causal"):
            self.event(ledger, "producer", "prepare_start", 149)
        self.event(ledger, "producer", "prepare_start", 160)

    def test_old_clock_does_not_silently_accept_v2_or_relax_order(self):
        old = LiveClockLedger(self.identity, self.semantic)
        self.event(old, "collector", "owned_tick_snapshot", 10)
        with self.assertRaisesRegex(LiveClockError, "only to async"):
            self.event(old, "ue", "source_owned", 20)
        self.event(old, "ue", "action_open", 100)
        with self.assertRaisesRegex(LiveClockError, "causal"):
            self.event(old, "producer", "source_stage_received", 150)
        self.assertEqual(SCHEMA, "splitfusion.numeric_gt.live_clock.v1")

    def test_v2_refuses_incomparable_ue_producer_boots(self):
        ledger = self.ledger()
        self.event(ledger, "collector", "owned_tick_snapshot", 10)
        self.event(ledger, "ue", "source_owned", 20)
        self.event(ledger, "ue", "action_open", 100)
        self.event(ledger, "ue", "last_datagram_sent", 130)
        for name, stamp in (("source_stage_received", 150), ("prepare_start", 160),
                            ("source_alignment_observed", 170), ("numeric_ready", 180),
                            ("ready_published", 190)):
            self.event(ledger, "producer", name, stamp, boot=OTHER_BOOT)
        self.event(ledger, "ue", "feedback_timeout", 100 + DEADLINE_NS + 1)
        with self.assertRaisesRegex(LiveClockError, "one observed Linux boot"):
            ledger.archive_descriptor(avo_sha256="a" * 64,
                source_synchronization_sha256="b" * 64)

    @staticmethod
    def observed():
        def item(role, name, raw):
            return dict(role=role, name=name, raw_ns=raw,
                        pid=PID[role], boot_id=BOOT)
        return dict(schema=E.ASYNC_OBSERVED_EVENTS_SCHEMA,
            handoff_id=ASYNC_HANDOFF_ID,
            events=[item("collector", "owned_tick_snapshot", BASE + 10),
                    item("ue", "source_owned", BASE + 20),
                    item("ue", "action_open", BASE + 100),
                    item("ue", "last_datagram_sent", BASE + 130),
                    item("producer", "source_stage_received", BASE + 150)])

    def test_edge_parser_selects_only_exact_v2_and_refuses_crossed_claims(self):
        observed = self.observed()
        self.assertEqual(E._observed_handoff(observed), ASYNC_HANDOFF_ID)
        for changed in ("missing_owned", "late_owned", "early_stage", "missing_raw",
                        "foreign_ue_pid", "foreign_stage_boot", "crossed_schema", "crossed_id"):
            with self.subTest(changed=changed):
                value = copy.deepcopy(observed)
                if changed == "missing_owned":
                    value["events"].pop(1)
                elif changed == "late_owned":
                    value["events"][1]["raw_ns"] = BASE + 101
                elif changed == "early_stage":
                    value["events"][-1]["raw_ns"] = BASE + 129
                elif changed == "missing_raw":
                    value["events"][1].pop("raw_ns")
                elif changed == "foreign_ue_pid":
                    value["events"][1]["pid"] = 999
                elif changed == "foreign_stage_boot":
                    value["events"][-1]["boot_id"] = OTHER_BOOT
                elif changed == "crossed_schema":
                    value["schema"] = E.OBSERVED_EVENTS_SCHEMA
                else:
                    value["handoff_id"] = "wrong"
                with self.assertRaises(E.NumericEdgeError):
                    E._observed_handoff(value)


if __name__ == "__main__":
    unittest.main()
