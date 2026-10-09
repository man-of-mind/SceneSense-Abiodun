"""Bounded read-only source export; CPU fixture, no live services or archive."""
import json
import unittest
from unittest.mock import patch

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2.test_ownership import plan_for
from rl_agent.splitfusion_numeric_gt_producer_v1 import artifacts as AR
from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T

from . import producer as G
from .test_producer import fixture, prepare


def _live_prepare(engine, kwargs):
    return prepare(engine, kwargs, archive_clocks=None, payload_filenames=None,
                   prescore_filename=None, replay_observed_raw_ns=None)


class LiveSourceExportTests(unittest.TestCase):
    def test_committed_published_frame_exports_exact_immutable_primitives(self):
        with fixture(offline_replay=False) as (engine, kwargs, _):
            pending = _live_prepare(engine, kwargs)
            engine.confirm_send(pending.token, plan_for(kwargs)[0])
            engine.publish_ready(pending.token)
            before = engine.snapshot()
            with patch.object(AR, "legacy_target", side_effect=AssertionError("legacy GT called")), \
                 patch.object(AR, "avo_documents", side_effect=AssertionError("old AVO called")), \
                 patch("cv2.imencode", side_effect=AssertionError("PNG called")):
                first = engine.export_live_source(pending.token, pending.identity_bytes,
                                                   pending.semantic_sha256)
                second = engine.export_live_source(pending.token, pending.identity_bytes,
                                                    pending.semantic_sha256)
            staged, _, avo, _ = engine._owner._archive_for_owner(
                engine._frames[pending.token.sequence].owner_token)
            self.assertEqual(first, second)
            self.assertEqual(first.schema, G.LIVE_SOURCE_EXPORT_SCHEMA)
            self.assertEqual(first.identity_bytes, pending.identity_bytes)
            self.assertEqual(first.semantic_sha256, pending.semantic_sha256)
            self.assertEqual(C._parse(first.input_metadata_bytes),
                             C._parse(G._canonical(T._input_metadata(staged.inputs))))
            self.assertEqual(C._parse(first.source_synchronization_bytes),
                             C._parse(G._canonical(staged.sync)))
            self.assertEqual(C._parse(first.numerical_avo_bytes),
                             C._parse(G._canonical(avo.avo)))
            self.assertGreater(C._parse(first.source_synchronization_bytes)["observed_raw_ns"], 0)
            event_doc = json.loads(first.producer_events_bytes)
            self.assertEqual(event_doc["ready_identity"], C._parse(pending.identity_bytes))
            self.assertEqual(event_doc["semantic_sha256"], pending.semantic_sha256)
            self.assertEqual(event_doc["role"], "producer")
            self.assertGreater(event_doc["pid"], 0)
            self.assertEqual([event["name"] for event in event_doc["events"]],
                ["prepare_start", "source_alignment_observed", "numeric_ready", "ready_published"])
            self.assertEqual(event_doc["events"][1]["raw_ns"],
                C._parse(first.source_synchronization_bytes)["observed_raw_ns"])
            self.assertEqual([event["raw_ns"] for event in event_doc["events"]],
                sorted(event["raw_ns"] for event in event_doc["events"]))
            self.assertEqual(first.payload_bytes, sum(map(len, (first.identity_bytes,
                first.input_metadata_bytes, first.source_synchronization_bytes,
                first.numerical_avo_bytes, first.producer_events_bytes))))
            self.assertLessEqual(first.payload_bytes, G.MAX_LIVE_SOURCE_EXPORT_BYTES)
            self.assertEqual(engine.snapshot(), before)
            self.assertFalse(engine._frames[pending.token.sequence].scored)
            self.assertFalse(engine._frames[pending.token.sequence].feedback)
            self.assertIsNone(engine._frames[pending.token.sequence].archive)
            with self.assertRaises(TypeError):
                first.input_metadata_bytes[0] = 0

    def test_export_refuses_before_commit_or_publication_and_foreign_identity(self):
        with fixture(offline_replay=False) as (engine, kwargs, _):
            pending = _live_prepare(engine, kwargs)
            with self.assertRaisesRegex(G.ProducerError, "notification before committed publication"):
                engine.export_live_source(pending.token, pending.identity_bytes,
                                          pending.semantic_sha256)
        with fixture(offline_replay=False) as (engine, kwargs, _):
            pending = _live_prepare(engine, kwargs)
            engine.confirm_send(pending.token, plan_for(kwargs)[0])
            with self.assertRaisesRegex(G.ProducerError, "notification before committed publication"):
                engine.export_live_source(pending.token, pending.identity_bytes,
                                          pending.semantic_sha256)
        with fixture(offline_replay=False) as (engine, kwargs, _):
            pending = _live_prepare(engine, kwargs)
            engine.confirm_send(pending.token, plan_for(kwargs)[0]); engine.publish_ready(pending.token)
            with self.assertRaisesRegex(G.ProducerError, "foreign notification"):
                engine.export_live_source(pending.token, pending.identity_bytes, "0" * 64)

    def test_export_has_explicit_aggregate_size_cap_and_no_offline_clock_claim(self):
        with fixture(offline_replay=False) as (engine, kwargs, _):
            pending = _live_prepare(engine, kwargs)
            engine.confirm_send(pending.token, plan_for(kwargs)[0]); engine.publish_ready(pending.token)
            with patch.object(G, "MAX_LIVE_SOURCE_EXPORT_BYTES", 16):
                with self.assertRaisesRegex(G.ProducerError, "bounded live source export size"):
                    engine.export_live_source(pending.token, pending.identity_bytes,
                                              pending.semantic_sha256)
            self.assertIsNotNone(engine.export_live_source(pending.token, pending.identity_bytes,
                                                           pending.semantic_sha256))
        with fixture() as (engine, kwargs, _):
            pending = prepare(engine, kwargs)
            engine.confirm_send(pending.token, plan_for(kwargs)[0]); engine.publish_ready(pending.token)
            with self.assertRaisesRegex(G.ProducerError, "observed live source clocks"):
                engine.export_live_source(pending.token, pending.identity_bytes,
                                          pending.semantic_sha256)


if __name__ == "__main__":
    unittest.main()
