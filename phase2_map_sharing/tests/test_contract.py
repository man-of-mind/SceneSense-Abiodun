from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml

from phase2_map_sharing.adapters import snapshot_stream_to_contribution
from phase2_map_sharing.engine import RecipientMapEngine
from phase2_map_sharing.evaluation import TruthTrajectory, match_warning_to_truth
from phase2_map_sharing.run_local_acceptance import evaluate_fixture
from phase2_map_sharing.run_recorded_snapshot_smoke import evaluate as evaluate_recordings
from phase2_map_sharing.selection import select_recipient_hazards
from phase2_map_sharing.transport import ChunkReassembler, chunk_payload
from phase2_map_sharing.schemas import (
    EgoState,
    MapContribution,
    MapObjectObservation,
    with_exact_payload_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "phase2_map_sharing" / "configs" / "local_acceptance_v1.yaml"


def observation(track_id: str = "track", x_m: float = 10.0, observed_at_s: float = 0.0):
    return MapObjectObservation(track_id, "pedestrian", x_m, 0.0, 0.0, 0.0, 0.9, observed_at_s)


def contribution(
    source: str = "helper",
    recipient: str = "ego",
    sequence: int = 1,
    captured_at_s: float = 0.0,
    x_m: float = 10.0,
):
    return MapContribution(
        contribution_id=f"{source}:{sequence}",
        source_ue_id=source,
        recipient_ue_id=recipient,
        sequence_number=sequence,
        captured_at_s=captured_at_s,
        published_at_s=captured_at_s,
        profile_id="test",
        payload_bytes=100,
        objects=(observation(f"{source}-track", x_m, captured_at_s),),
    )


class SchemaTests(unittest.TestCase):
    def test_round_trip_and_resource_uri(self):
        original = contribution()
        payload = original.to_dict()
        self.assertEqual(payload["resource_uri"], "/ss-sm-management/v1/spatial-maps/ego")
        restored = MapContribution.from_dict(payload)
        self.assertEqual(restored, original)

    def test_evaluation_identity_is_forbidden(self):
        payload = contribution().to_dict()
        payload["objects"][0]["carla_actor_id"] = 42
        with self.assertRaisesRegex(ValueError, "evaluation-only identity"):
            MapContribution.from_dict(payload)

    def test_resource_and_exact_byte_contracts_fail_closed(self):
        exact = with_exact_payload_bytes(contribution())
        payload = exact.to_dict()
        payload["resource_uri"] = "/ss-sm-management/v1/spatial-maps/other"
        with self.assertRaisesRegex(ValueError, "named recipient"):
            MapContribution.from_dict(payload)
        tampered = exact.to_json_bytes().replace(
            f'"payload_bytes":{exact.payload_bytes}'.encode("utf-8"),
            b'"payload_bytes":1',
        )
        with self.assertRaisesRegex(ValueError, "serialized application bytes"):
            MapContribution.from_json_bytes(tampered)

    def test_existing_snapshot_adapter_filters_source_and_does_not_leak_identity(self):
        snapshot = {
            "raw_spatial_map_objects": [
                {
                    "id": "helper:1:0",
                    "source_stream_id": "helper",
                    "type": "Vehicle",
                    "location": {"x": 2.0, "y": 3.0, "z": 0.0},
                    "velocity": {"x": 1.0, "y": -1.0},
                    "score": 0.8,
                    "carla_actor_id": 999,
                },
                {
                    "id": "other:1:0",
                    "source_stream_id": "other",
                    "type": "Pedestrian",
                    "location": {"x": 1.0, "y": 1.0, "z": 0.0},
                    "score": 0.9,
                },
            ]
        }
        result = snapshot_stream_to_contribution(
            snapshot,
            source_stream_id="helper",
            recipient_ue_id="ego",
            sequence_number=1,
            captured_at_s=1.0,
            published_at_s=1.1,
            profile_id="p",
            payload_bytes=12,
        )
        self.assertEqual(len(result.objects), 1)
        self.assertEqual(result.objects[0].vx_mps, 1.0)
        self.assertNotIn("actor", result.to_json_bytes().decode("utf-8"))

    def test_production_chunk_header_round_trip_out_of_order(self):
        exact = with_exact_payload_bytes(contribution())
        original = exact.to_json_bytes()
        chunks = chunk_payload(original, message_id=7, chunk_bytes=40)
        receiver = ChunkReassembler(timeout_s=1.0)
        result = None
        for index, datagram in enumerate(reversed(chunks)):
            result = receiver.ingest("10.0.0.3:39201", datagram, received_at_s=0.01 * index)
        self.assertIsNotNone(result)
        self.assertEqual(result.payload, original)
        self.assertEqual(MapContribution.from_json_bytes(result.payload), exact)


class EngineTests(unittest.TestCase):
    def test_recipient_sequence_stale_and_ttl_guards(self):
        engine = RecipientMapEngine("ego", track_ttl_s=1.0, max_transport_age_s=0.5)
        self.assertEqual(engine.install(contribution(recipient="other"), 0.0), "rejected_wrong_recipient")
        accepted = contribution(sequence=2, captured_at_s=0.1)
        self.assertEqual(engine.install(accepted, 0.1), "accepted")
        self.assertEqual(engine.install(accepted, 0.1), "rejected_sequence")
        self.assertEqual(
            engine.install(contribution(source="late", captured_at_s=0.0), 0.6),
            "rejected_transport_stale",
        )
        self.assertEqual(len(engine.snapshot(1.11)["tracks"]), 0)

    def test_association_and_live_provenance(self):
        engine = RecipientMapEngine("ego", association_gate_m=2.0, track_ttl_s=1.0)
        self.assertEqual(engine.install(contribution(source="helper", x_m=10.0), 0.0), "accepted")
        self.assertEqual(
            engine.install(contribution(source="ego", sequence=1, captured_at_s=0.1, x_m=10.2), 0.1),
            "accepted",
        )
        snapshot = engine.snapshot(0.1)
        self.assertEqual(len(snapshot["tracks"]), 1)
        warning = engine.warnings(EgoState("ego", 0.1, 0.5, 0.0, 5.0, 0.0))[0]
        self.assertEqual(warning.evidence_scope, "multi_source")
        self.assertEqual(warning.evidence_sources, ("ego", "helper"))

    def test_provenance_expires_independently_of_track(self):
        engine = RecipientMapEngine("ego", association_gate_m=2.0, track_ttl_s=1.0)
        engine.install(contribution(source="helper", x_m=10.0), 0.0)
        engine.install(contribution(source="ego", captured_at_s=0.8, x_m=10.0), 0.8)
        warning = engine.warnings(EgoState("ego", 1.1, 5.5, 0.0, 5.0, 0.0))[0]
        self.assertEqual(warning.evidence_sources, ("ego",))
        self.assertEqual(warning.evidence_scope, "ego_only")

    def test_truth_identity_is_joined_only_by_evaluator(self):
        engine = RecipientMapEngine("ego")
        engine.install(contribution(source="helper", x_m=10.0), 0.0)
        warning = engine.warnings(EgoState("ego", 0.0, 0.0, 0.0, 5.0, 0.0))[0]
        truth = TruthTrajectory("carla-actor-42", "pedestrian", 10.2, 0.0, safety_hazard=True)
        match = match_warning_to_truth(warning, [truth], gate_m=1.0)
        self.assertEqual(match.truth_id, "carla-actor-42")
        self.assertTrue(match.safety_hazard)
        self.assertNotIn("carla", warning.evidence_track_ids[0])


class AcceptanceTests(unittest.TestCase):
    def test_hazard_only_is_causal_and_recipient_specific(self):
        ego = EgoState("ego", 0.0, 0.0, 0.0, 5.0, 0.0)
        near_path = observation("crossing", 20.0, 0.0)
        off_path = MapObjectObservation("benign", "pedestrian", 20.0, 10.0, 0.0, 0.0, 0.9, 0.0)
        selected = select_recipient_hazards(
            [near_path, off_path],
            ego,
            capture_at_s=0.0,
            horizon_s=5.0,
            confidence_floor=0.15,
            safety_radius_m_by_class={"pedestrian": 2.5},
        )
        self.assertEqual([item.source_track_id for item in selected], ["crossing"])
        self.assertGreater(selected[0].hazard_score, 0.0)

    def test_synthetic_contract_fixture_passes(self):
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        frame, summary = evaluate_fixture(config)
        self.assertEqual(summary["verdict"], "PASS")
        self.assertFalse(summary["research_evidence"])
        self.assertAlmostEqual(summary["warning_lead_gain_vs_ego_only_s"]["hazard_only"], 1.9)
        self.assertEqual(summary["false_warning_count"], 0)
        self.assertEqual(set(frame["strategy"]), {"ego_only", "send_everything", "hazard_only"})

    def test_existing_two_stream_recordings_cross_adapter_contract(self):
        config_path = ROOT / "phase2_map_sharing" / "configs" / "recorded_snapshot_smoke_v1.yaml"
        frame, summary = evaluate_recordings(yaml.safe_load(config_path.read_text(encoding="utf-8")))
        self.assertEqual(summary["verdict"], "PASS")
        self.assertEqual(summary["eligible_two_active_snapshots"], 37)
        self.assertEqual(summary["accepted_contributions"], 26)
        self.assertTrue(frame["round_trip_exact"].all())


if __name__ == "__main__":
    unittest.main()
