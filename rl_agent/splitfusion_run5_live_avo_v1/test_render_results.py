import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import render_results as R


def joined_fixture():
    identity = dict(session_uuid="session-a", decision_seq=0, ue_id="oai-ue-test")
    wire = dict(frame_id=10, tensor_seq=0, session_uuid="session-a", decision_seq=0,
                ticket_seq=0, mode_id=11, q_e4=3000, frame_kind="POLICY_DECISION", reward_requested=True,
                controller_lineage_sha256="lineage", execution_bundle_sha256="bundle", anchor_action_id=67)
    frame = {key: str(value) for key, value in wire.items()}
    frame.update(action_open_raw_ns="1000000000", payload_bytes="1234")
    ticket = {key: frame[key] for key in ("frame_id", "decision_seq", "mode_id", "q_e4", "payload_bytes")}
    ticket.update(resolution_terminal="SUCCESS", action_open_to_feedback_ms="150.0")
    decision = dict(frame_id=10, tensor_seq=0, mode_id=11, q_e4=3000, kind="POLICY_DECISION",
        decision_session_uuid="session-a", features=[0.] * 21 + [.5],
        state_commit=dict(domain="CLOCK_MONOTONIC_RAW", ns=999999000),
        action_open=dict(domain="CLOCK_MONOTONIC_RAW", ns=1000000000))
    resolution = dict(identity=dict(identity), action_open_timestamp_ns=1000000000, terminal="SUCCESS")
    reply = dict(frame_id=10, q_perc=.45, kind="DELIVERED_SUCCESS", reason="NONE", sha256="feedback-digest",
                 receipt=dict(domain="CLOCK_MONOTONIC_RAW", ns=1150000000))
    measure = {}
    for name in ("vehicle", "person"):
        measure.update({f"seg_{name}_{suffix}_pixels": 1 for suffix in ("gt", "pred", "intersection", "union")})
        measure.update({f"loc_{name}_eligible_gt": 1, f"loc_{name}_tp": 1, f"loc_{name}_fp": 0,
                        f"loc_{name}_fn": 0, f"loc_{name}_matched_xy_errors_m": [.2]})
    evaluation = dict(frame_id=10, run4_identity=dict(wire), q_perc=.45, kind="DELIVERED_SUCCESS",
                      reason="NONE", feedback_sha256="feedback-digest", measurement=measure)
    observation = dict(frame_id=10, admitted=True, decision_identity=dict(identity), snr_scaled=.5,
        snr=dict(value_db=15.), boundary=dict(identity=dict(identity), clock_domain="CLOCK_MONOTONIC_RAW",
            state_commit_timestamp_ns=999999000, action_open_timestamp_ns=1000000000))
    return dict(ue=dict(decisions=[decision], transmitted_identities=[dict(frame_id=10, run4_identity=wire)],
                        feedback_rows=[reply], resolutions=[resolution]),
                edge=dict(evaluations=[evaluation]), snr=dict(decision_snr=[observation]),
                frames=[frame], tickets=[ticket])


class ReportingTests(unittest.TestCase):
    def test_exact_identity_join(self):
        result = R.join_records(**joined_fixture())[0]
        self.assertTrue(result["timely"])
        self.assertEqual(result["q_perc"], .45)
        self.assertEqual(result["quality_source"], "EDGE_EVALUATION")

    def test_ticket_action_mismatch_refused(self):
        fixture = joined_fixture()
        fixture["tickets"][0]["q_e4"] = "3001"
        with self.assertRaisesRegex(ValueError, "ticket/frame identity"):
            R.join_records(**fixture)

    def test_full_edge_identity_mismatch_refused(self):
        fixture = joined_fixture()
        fixture["edge"]["evaluations"][0]["run4_identity"]["execution_bundle_sha256"] = "foreign"
        with self.assertRaisesRegex(ValueError, "full identity"):
            R.join_records(**fixture)

    def test_duplicate_sources_refused(self):
        for group, key in (("ue", "feedback_rows"), ("edge", "evaluations"), ("snr", "decision_snr")):
            with self.subTest(group=group):
                fixture = joined_fixture()
                fixture[group][key].append(copy.deepcopy(fixture[group][key][0]))
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    R.join_records(**fixture)

    def test_wrong_snr_identity_or_boundary_refused(self):
        fixture = joined_fixture()
        fixture["snr"]["decision_snr"][0]["decision_identity"]["session_uuid"] = "foreign"
        with self.assertRaisesRegex(ValueError, "SNR/actual decision identity"):
            R.join_records(**fixture)
        fixture = joined_fixture()
        fixture["snr"]["decision_snr"][0]["boundary"]["state_commit_timestamp_ns"] += 1
        with self.assertRaisesRegex(ValueError, "boundary mismatch"):
            R.join_records(**fixture)

    def test_lost_reply_retains_edge_quality_without_timely_credit(self):
        fixture = joined_fixture()
        fixture["ue"]["feedback_rows"] = []
        fixture["ue"]["resolutions"][0]["terminal"] = "TIMEOUT"
        fixture["tickets"][0].update(resolution_terminal="TIMEOUT", action_open_to_feedback_ms="")
        result = R.join_records(**fixture)[0]
        self.assertIsNone(result["latency_ms"])
        self.assertFalse(result["timely"])
        self.assertFalse(result["feedback_available"])
        self.assertEqual(result["q_perc"], .45)
        self.assertEqual(result["quality_status"], "DEFINED_NO_FEEDBACK")
        self.assertEqual(result["terminal"], "TIMEOUT")

    def test_received_feedback_digest_and_quality_must_match_edge(self):
        for key, value in (("sha256", "foreign"), ("q_perc", .46)):
            with self.subTest(key=key):
                fixture = joined_fixture()
                fixture["ue"]["feedback_rows"][0][key] = value
                with self.assertRaisesRegex(ValueError, "Received feedback/edge"):
                    R.join_records(**fixture)

    def test_failed_capture_refused_before_loading_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "QUALIFICATION_RESULT.json").write_text(json.dumps(
                {"passed": False, "gates": {"G1": False}}))
            with self.assertRaisesRegex(ValueError, "no performance plots"):
                R.load(root)

    def test_missing_errors_and_empty_unions_are_not_zero(self):
        measure = {f"seg_person_{s}_pixels": 0 for s in ("gt", "pred", "intersection", "union")}
        measure.update({f"loc_person_{s}": 0 for s in ("eligible_gt", "tp", "fp", "fn")})
        measure["loc_person_matched_xy_errors_m"] = []
        row = R.components(measure, "person")
        self.assertIsNone(row["iou"])
        self.assertIsNone(row["xy_median_m"])
        self.assertIsNone(row["recall"])
        self.assertIsNone(row["precision"])

    def test_localization_error_requires_corresponding_true_positives(self):
        measure = {f"seg_vehicle_{s}_pixels": 0 for s in ("gt", "pred", "intersection", "union")}
        measure.update({f"loc_vehicle_{s}": 0 for s in ("eligible_gt", "tp", "fp", "fn")})
        measure["loc_vehicle_matched_xy_errors_m"] = [.2]
        with self.assertRaisesRegex(ValueError, "localization"):
            R.components(measure, "vehicle")

    def test_percentiles_and_missing_values(self):
        result = R.stats([None, 100., 200.])
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["p50"], 150.)
        self.assertEqual(result["p95"], 195.)
        self.assertEqual(R.stats([None]), {"n": 0})

    def test_png_render_handles_missing_feedback_and_quality(self):
        component = dict(iou=.5, xy_median_m=.4, recall=.5, precision=.6)
        records = [dict(frame_number=i + 1, frame_id=i, mode_id=i % 12,
                        q_e4=1000 * i, payload_bytes=200000,
                        latency_ms=None if i == 2 else 150. + i * 30,
                        timely=i == 0, q_perc=None if i == 2 else .45,
                        terminal="SUCCESS" if i == 0 else "TIMEOUT", snr_db=20.,
                        vehicle=component, person=component) for i in range(3)]
        frames = [{"frame_kind": "POLICY_DECISION"} for _ in records]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(R, "load", return_value=(records, frames, {})):
            output = Path(tmp) / "plots"
            result = R.render(Path(tmp) / "unused", output)
            self.assertEqual(result["missing_replies"], 1)
            self.assertEqual(len(list(output.glob("*.png"))), 7)
            self.assertFalse(list(output.glob("*.pdf")))
            manifest = R.read(output / "OUTPUT_SHA256.json")
            self.assertTrue(all(R.sha(output / name) == digest for name, digest in manifest.items()))


if __name__ == "__main__":
    unittest.main()
