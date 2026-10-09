"""CPU parity gates over real SFD4 identities; no checkpoint or service needed."""
import hashlib
import unittest

import numpy as np
import torch

from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.continuous_execution_v2 import ExecutionEnvelopeV3
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as wire
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as protocol
from .replay import Prediction, ReplayParityError, replay_pair


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.context = build_frame_context_v1(
            stream_id="saved-stream", frame_id=105, sequence_id=8,
            capture_timestamp_ns=123456789000, ego_world_x=1., ego_world_y=2.,
            ego_world_z=3., ego_world_pitch=0., ego_world_yaw=10., ego_world_roll=0.)
        self.inner = b"selected-live-inner-payload"
        self.envelope = ExecutionEnvelopeV3(
            mode_id=0, q_e4=3000, keep_count=15053, anchor_action_id=1,
            reward_requested=True, session_uuid="00000000-0000-0000-0000-000000000001",
            controller_lineage_sha256="a" * 64, decision_seq=4, ticket_seq=4,
            frame_id=105, tensor_seq=8, capture_timestamp_ns=self.context.capture_timestamp_ns,
            execution_bundle_sha256="b" * 64,
            inner_payload_sha256=hashlib.sha256(self.inner).hexdigest(), inner_payload=self.inner)
        self.input = torch.zeros((1, 7, 448, 768), dtype=torch.float32)
        self.record = {"stream_id": self.context.stream_id, "frame_id": self.context.frame_id,
                       "capture_timestamp_ns": self.context.capture_timestamp_ns,
                       "class_name": "vehicle", "score": .8, "world_x": 1., "world_y": 2.}
        self.labels = np.zeros((720, 1280), dtype=np.uint8)
        self.calls = []
        self.metadata = []

    def front(self, value):
        self.calls.append("front")
        # Deliberate callback mutation must not change the caller's saved input.
        value[0, 0, 0, 0] = 99.
        return torch.full((1, 256, 112, 192), 3., dtype=torch.float32)

    def encode(self, value, envelope):
        self.calls.append("encode")
        self.assertEqual(envelope, self.envelope)
        self.assertEqual(float(value[0, 0, 0]), 3.)
        value.fill_(99.)
        return self.inner

    def decode(self, payload, envelope):
        self.calls.append("decode")
        self.assertEqual(payload, self.inner)
        self.assertEqual(envelope, self.envelope)
        return torch.full((256, 112, 192), 2., dtype=torch.float32)

    def tail(self, value, metadata):
        marker = float(value[0, 0, 0])
        self.calls.append(("tail", marker))
        self.metadata.append(metadata)
        value.fill_(88.)
        labels = self.labels.copy()
        if marker == 3.:
            labels[0, 0] = 1
        return Prediction(labels, [dict(self.record)])

    def replay(self, **overrides):
        arguments = dict(prepared_input=self.input, selected_wire=wire.pack_sfd4(self.envelope, self.context),
                         expected_identity=protocol.run4_identity(self.envelope),
                         archived_selected=Prediction(self.labels, [self.record]),
                         front=self.front, encode_selected=self.encode,
                         decode_selected=self.decode, tail=self.tail)
        arguments.update(overrides)
        return replay_pair(**arguments)

    def test_direct_dense_branch_isolated_from_all_selected_mutations(self):
        result = self.replay()
        self.assertEqual(self.calls, ["front", "encode", "decode", ("tail", 2.), ("tail", 3.)])
        self.assertIs(self.metadata[0], self.metadata[1])
        self.assertEqual(result.metadata.frame_context, self.context)
        self.assertEqual(dict(result.metadata.run4_identity), protocol.run4_identity(self.envelope))
        self.assertEqual(float(self.input[0, 0, 0, 0]), 0.)
        self.assertEqual(int(result.selected.labels[0, 0]), 0)
        self.assertEqual(int(result.dense.labels[0, 0]), 1)
        self.assertTrue(result.evidence["selected_inner_payload_byte_identical"])
        self.assertFalse(result.evidence["live_latency_measurement"])
        self.assertFalse(result.selected.labels.flags.writeable)
        with self.assertRaises(ValueError):
            result.selected.labels.flags.writeable = True

    def test_payload_mismatch_refuses_decoder_and_both_tails(self):
        with self.assertRaisesRegex(ReplayParityError, "inner payload byte parity"):
            self.replay(encode_selected=lambda *_: b"merely-similar-payload")
        self.assertEqual(self.calls, ["front"])

    def test_selected_label_mismatch_refuses_dense_tail(self):
        archived = self.labels.copy()
        archived[4, 5] = 1
        with self.assertRaisesRegex(ReplayParityError, "native-label parity"):
            self.replay(archived_selected=Prediction(archived, [self.record]))
        self.assertEqual(self.calls[-1], ("tail", 2.))
        self.assertEqual(len(self.metadata), 1)

    def test_selected_record_mismatch_refuses_dense_tail(self):
        with self.assertRaisesRegex(ReplayParityError, "complete-record parity"):
            self.replay(archived_selected=Prediction(self.labels, [dict(self.record, world_x=9.)]))
        self.assertEqual(len(self.metadata), 1)

    def test_foreign_session_or_context_refused_before_inference(self):
        identity = protocol.run4_identity(self.envelope)
        identity["session_uuid"] = "00000000-0000-0000-0000-000000000002"
        with self.assertRaisesRegex(ReplayParityError, "action identity differs"):
            self.replay(expected_identity=identity)
        with self.assertRaisesRegex(ReplayParityError, "prediction context differs"):
            self.replay(archived_selected=Prediction(self.labels, [dict(self.record, stream_id="other")]))
        self.assertEqual(self.calls, [])

    def test_wire_tamper_refused_before_inference(self):
        data = wire.pack_sfd4(self.envelope, self.context)
        with self.assertRaisesRegex(ReplayParityError, "wire or expected identity invalid"):
            self.replay(selected_wire=data[:-1] + bytes([data[-1] ^ 1]))
        self.assertEqual(self.calls, [])

    def test_original_input_geometry_and_finiteness_are_required(self):
        self.input[0, 0, 432, 0] = 1.
        with self.assertRaisesRegex(ReplayParityError, "bottom padding"):
            self.replay()
        self.input[0, 0, 432, 0] = 0.
        self.input[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ReplayParityError, "input is non-finite"):
            self.replay()
        self.assertEqual(self.calls, [])

    def test_nonfinite_front_refuses_all_downstream_branches(self):
        with self.assertRaisesRegex(ReplayParityError, "non-finite C2"):
            self.replay(front=lambda _: torch.full((256, 112, 192), float("nan")))
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
