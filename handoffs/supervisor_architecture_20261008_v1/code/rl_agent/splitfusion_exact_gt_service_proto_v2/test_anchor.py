"""Unit tests: anchor provenance mapping, explicit rejections, join derivation, scorer capacity release."""
import copy
import unittest

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_exact_gt_service_proto_v1 import scorer_core as V1

from .deferred import DeferralRefused, anchor_provenance, neutral_action, validate_anchor_provenance
from .scorer_core import ProtoScorer, Refused, join_identity, triple_key

ANCHORS = {17: (2, 9800, "split_noae_uint4_q9800"), 33: (5, 7000, "split_ae128_uint4_q7000"),
           35: (5, 9800, "split_ae128_uint4_q9800"), 50: (8, 5000, "split_ae64_uint4_q5000"),
           53: (8, 9800, "split_ae64_uint4_q9800"), 65: (10, 9800, "split_ae32_uint6_q9800"),
           67: (11, 3000, "split_ae32_uint4_q3000")}
TEMPLATE = dict(run_id="run", cell_id="cell", stream_id="stream", frame_id=10, capture_timestamp_ns=123456789,
                action_id=None, profile_id=None)
CONTEXT = dict(stream_id="stream", frame_id=10, sequence_id=2, capture_timestamp_ns=123456789,
               ego_world=[1.0, 2.0, 0.1, 0.0, 90.0, 0.0], camera_model_sha256="c" * 64, camera_mount_sha256="d" * 64)


def action_for(action_id, **changes):
    mode, q, _ = ANCHORS[action_id]
    value = dict(session_uuid="00000000-0000-4000-8000-000000000001", controller_lineage_sha256="a" * 64,
                 decision_seq=0, ticket_seq=0, frame_id=10, tensor_seq=2, mode_id=mode, q_e4=q,
                 execution_bundle_sha256="b" * 64, anchor_action_id=action_id, reward_requested=True,
                 frame_kind="POLICY_DECISION")
    value.update(changes)
    return value


def plan(action_id, capture_changes=None, action_changes=None, context=None):
    profile = ANCHORS[action_id][2]
    capture = dict(dict(TEMPLATE, action_id=action_id, profile_id=profile), **(capture_changes or {}))
    return C.canonical(dict(scope="TEST", capture=capture, action=action_for(action_id, **(action_changes or {})),
                            frame_context=context or CONTEXT))


def target_identity(provenance, **changes):
    value = dict(capture_identity=provenance["neutral_capture"], action_identity=provenance["neutral_action_identity"],
                 frame_context=provenance["frame_context"], plan_sha256=provenance["plan_sha256"])
    value.update(changes)
    return value


class AnchorProvenanceTest(unittest.TestCase):
    def test_all_screen_anchors_bind_with_exact_neutralisation(self):
        for action_id, (mode, q, profile) in ANCHORS.items():
            doc = anchor_provenance(TEMPLATE, plan(action_id))
            self.assertEqual(doc["neutral_capture"], TEMPLATE)
            self.assertEqual(doc["selected_capture"], dict(TEMPLATE, action_id=action_id, profile_id=profile))
            self.assertEqual(doc["catalogue"]["action_id"], action_id)
            self.assertEqual((doc["catalogue"]["mode_id"], doc["catalogue"]["q_e4"]), (mode, q))
            self.assertIsNone(doc["neutral_action_identity"]["anchor_action_id"])
            self.assertEqual(doc["neutral_action_identity"], neutral_action(doc["action_identity"]))
            self.assertEqual(doc["plan_sha256"], C.sha256(plan(action_id)))
            self.assertIs(validate_anchor_provenance(doc, target_identity=target_identity(doc))["version"], 1)

    def rejected(self, raw, reason):
        with self.assertRaises(DeferralRefused) as caught:
            anchor_provenance(TEMPLATE, raw)
        self.assertEqual(caught.exception.reason, reason)

    def test_source_identity_mismatches_are_rejected(self):
        for changes in (dict(frame_id=11), dict(capture_timestamp_ns=123456790), dict(stream_id="other"),
                        dict(run_id="other"), dict(cell_id="other")):
            self.rejected(plan(33, capture_changes=changes), "SOURCE_IDENTITY_MISMATCH")

    def test_catalogue_mismatches_are_rejected(self):
        self.rejected(plan(33, action_changes=dict(q_e4=9800)), "CATALOGUE_ANCHOR_MISMATCH")      # q of action 35
        self.rejected(plan(33, action_changes=dict(q_e4=7001)), "CATALOGUE_ANCHOR_MISMATCH")      # off-anchor q
        self.rejected(plan(33, action_changes=dict(mode_id=8)), "CATALOGUE_ANCHOR_MISMATCH")      # other mode
        self.rejected(plan(33, capture_changes=dict(profile_id="split_ae128_uint4_q9800")), "CATALOGUE_ANCHOR_MISMATCH")
        self.rejected(plan(33, action_changes=dict(anchor_action_id=35)), "CATALOGUE_ANCHOR_MISMATCH")

    def test_tampered_mapping_or_foreign_target_fails_closed(self):
        doc = anchor_provenance(TEMPLATE, plan(50))
        for key, value in (("neutral_capture", dict(TEMPLATE, frame_id=99)),
                           ("catalogue", dict(doc["catalogue"], action_id=53)),
                           ("neutral_action_identity", dict(doc["neutral_action_identity"], mode_id=7)),
                           ("differing_fields", dict(capture=["action_id"], action=[]))):
            bad = copy.deepcopy(doc)
            bad[key] = value
            with self.assertRaises(Exception):
                validate_anchor_provenance(bad, target_identity=target_identity(doc))
        for changes in (dict(plan_sha256="0" * 64), dict(capture_identity=dict(TEMPLATE, frame_id=99)),
                        dict(action_identity=dict(doc["neutral_action_identity"], session_uuid="other")),
                        dict(frame_context=dict(CONTEXT, ego_world=[9.0, 2.0, 0.1, 0.0, 90.0, 0.0])),
                        dict(frame_context=dict(CONTEXT, camera_model_sha256="e" * 64))):
            with self.assertRaises(C.ContractError):
                validate_anchor_provenance(doc, target_identity=target_identity(doc, **changes))


class JoinTest(unittest.TestCase):
    def setUp(self):
        self.doc = anchor_provenance(TEMPLATE, plan(67))
        self.received = dict(gt_identity=self.doc["selected_capture"], action_identity=self.doc["action_identity"],
                             capture_timestamp_ns=TEMPLATE["capture_timestamp_ns"], prediction_digest="f" * 64)

    def test_anchor_join_identity(self):
        identity, mode = join_identity(self.received, self.doc, TEMPLATE)
        self.assertEqual(mode, "ANCHOR_PROVENANCE_V1")
        self.assertEqual(identity["gt_identity"], TEMPLATE)
        self.assertEqual(identity["action_identity"], self.doc["neutral_action_identity"])
        self.assertEqual(identity["prediction_digest"], "f" * 64)

    def test_mismatched_received_identity_fails_closed(self):
        for changes in (dict(gt_identity=dict(self.doc["selected_capture"], capture_timestamp_ns=1)),
                        dict(gt_identity=dict(self.doc["selected_capture"], action_id=65, profile_id="split_ae32_uint6_q9800")),
                        dict(action_identity=dict(self.doc["action_identity"], session_uuid="other")),
                        dict(action_identity=dict(self.doc["action_identity"], decision_seq=5))):
            with self.assertRaises(C.ContractError):
                join_identity(dict(self.received, **changes), self.doc, TEMPLATE)
        with self.assertRaises(C.ContractError):                       # anchor prediction, no mapping
            join_identity(self.received, None, TEMPLATE)
        with self.assertRaises(C.ContractError):                       # mapping for a different target
            join_identity(self.received, self.doc, dict(TEMPLATE, frame_id=11))

    def test_template_join_unchanged(self):
        received = dict(self.received, gt_identity=TEMPLATE, action_identity=neutral_action(self.doc["action_identity"]))
        identity, mode = join_identity(received, None, TEMPLATE)
        self.assertEqual((identity, mode), (received, "IDENTITY"))

    def test_join_key_equals_v1_for_template_and_neutral_for_anchor(self):
        action = neutral_action(self.doc["action_identity"])
        self.assertEqual(triple_key(TEMPLATE, action, CONTEXT), V1.triple_key(TEMPLATE, action, CONTEXT))
        self.assertEqual(triple_key(self.doc["selected_capture"], self.doc["action_identity"], CONTEXT),
                         V1.triple_key(TEMPLATE, action, CONTEXT))


class ScorerCapacityTest(unittest.TestCase):
    def prediction(self, frame, action_id=33):
        doc = anchor_provenance(dict(TEMPLATE, frame_id=frame), plan(action_id, capture_changes=dict(frame_id=frame),
                                                                   action_changes=dict(frame_id=frame)))
        ctx = dict(CONTEXT, frame_id=frame)
        return dict(gt_identity=doc["selected_capture"], action_identity=doc["action_identity"],
                    capture_timestamp_ns=TEMPLATE["capture_timestamp_ns"], prediction_digest="f" * 64), ctx

    def test_waiting_capacity_refusal_and_release_by_plan_triple(self):
        scorer = ProtoScorer(run_binding="1" * 64, producer_binding="2" * 64, max_jobs=2, max_bytes=16 << 20,
                             reservation_bytes=8 << 20, max_waiting=1, max_frames=100)
        labels = bytes(720 * 1280)
        first, ctx1 = self.prediction(10)
        second, ctx2 = self.prediction(14)
        self.assertEqual(scorer.admit_prediction(first, ctx1, [], labels)["status"], "WAITING_FOR_TARGET")
        with self.assertRaises(Refused) as caught:
            scorer.admit_prediction(second, ctx2, [], labels)
        self.assertEqual(caught.exception.reason, "WAITING_CAPACITY")
        self.assertEqual(scorer.snapshot()["waiting_predictions"], 1)       # refusal changed nothing
        # release by the plan's (selected) identity, exactly as the service does after a BIND refusal
        released = scorer.abandon(triple_key(first["gt_identity"], first["action_identity"], ctx1), "BIND_REFUSED")
        self.assertEqual(released["status"], "WAITING_DROPPED")
        self.assertEqual(scorer.snapshot()["waiting_predictions"], 0)
        self.assertEqual(scorer.admit_prediction(second, ctx2, [], labels)["status"], "WAITING_FOR_TARGET")
        with self.assertRaises(Refused) as caught:                          # late duplicate after terminal
            scorer.admit_prediction(first, ctx1, [], labels)
        self.assertEqual(caught.exception.reason, "LATE_AFTER_TERMINAL")
        scorer.abandon(triple_key(second["gt_identity"], second["action_identity"], ctx2), "TEST")
        self.assertEqual(scorer.close()["waiting_predictions"], 0)


if __name__ == "__main__":
    unittest.main()
