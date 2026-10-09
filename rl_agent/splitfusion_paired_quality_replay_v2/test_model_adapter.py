"""CPU injected-preload tests; never load a checkpoint or initialize CUDA."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import continuous_execution_v2 as X
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as wire
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as protocol
from rl_agent.splitfusion_live_dispatch_v1.dynamic_execution_contract import load_dynamic_execution_contract
from rl_agent.splitfusion_live_dispatch_v1.frame_context import StaticCameraRegistry, build_frame_context_v1
from rl_agent.splitfusion_live_dispatch_v1.registry import DispatchContractError
from rl_agent.splitfusion_live_dispatch_v1.transport import InnerIdentity
from rl_agent.splitfusion_paired_quality_replay_v1.replay import Prediction, ReplayParityError, replay_pair

from . import model_adapter as A


def inner_bytes(profile):
    return json.dumps(asdict(X.expected_inner(X.codec_view(profile))), sort_keys=True).encode()


class FakeFront:
    def __init__(self, calls):
        self._model = torch.nn.Identity().eval()
        self.calls = calls

    def __call__(self, value):
        self.calls.append(("front", tuple(value.shape), value.device.type))
        return torch.full((1, 256, 112, 192), 3., dtype=torch.float32)


class FakeRanker:
    def __init__(self):
        self._model = torch.nn.Identity().eval()

    def score_cells(self, value):
        return torch.zeros((112, 192))


class FakeAE:
    def __init__(self, mode):
        self._model = torch.nn.Identity().eval()
        self.family = mode.family
        self.family_id = mode.family_id
        self.bottleneck = mode.transported_channels
        self.routing_tag = mode.routing_tag


class FakeCodec:
    def __init__(self, calls):
        self.calls = calls

    def encode(self, view, c2, *, ranker, ae_encoder, timing):
        self.calls.append(("encode", view.family, view.q_e4, ranker, ae_encoder))
        return json.dumps(asdict(X.expected_inner(view)), sort_keys=True).encode()

    def inspect(self, payload, *, timing):
        self.calls.append(("inspect",))
        return SimpleNamespace(identity=InnerIdentity(**json.loads(payload)))

    def decode(self, inspected, *, decoder, tail_device, timing):
        self.calls.append(("decode", decoder))
        return SimpleNamespace(c2=torch.full((256, 112, 192), 2.), finite=True, device=tail_device)


def prediction_record(context, marker=2.):
    return {"stream_id": context.stream_id, "frame_id": context.frame_id,
            "capture_timestamp_ns": context.capture_timestamp_ns,
            "class_name": "vehicle", "score": .8, "world_x": marker,
            "world_y": 1., "unmodified_complete_record_field": "retained"}


class FakeTail:
    def __init__(self, calls):
        self.device = torch.device("cpu")
        self.calls = calls

    def compute_product(self, c2, metadata):
        marker = float(c2[0, 0, 0])
        self.calls.append(("tail", marker, metadata))
        labels = torch.zeros((720, 1280), dtype=torch.long)
        if marker == 3.:
            labels[0, 0] = 1
        return SimpleNamespace(semantic_labels=labels,
            records=(prediction_record(metadata.frame_context, marker),))

    def serialize_product(self, work):
        self.calls.append(("serialize",))
        return work

    def snapshot(self, serialized):
        self.calls.append(("snapshot",))
        return serialized


def preloads(contract, calls):
    aes = {family: FakeAE(next(mode for mode in contract.modes if mode.family == family))
           for family in A.AE_FAMILIES}
    ue = SimpleNamespace(device=torch.device("cpu"), _front=FakeFront(calls),
        _ranker=FakeRanker(), _ae_encoders=aes, _codec=FakeCodec(calls))
    edge = SimpleNamespace(device=torch.device("cpu"), model=torch.nn.Identity().eval(),
        perception_binding={"checkpoint_sha256": A.CHECKPOINT_SHA256["perception"]},
        camera_registry=StaticCameraRegistry.audited(), tail=FakeTail(calls),
        runtime=SimpleNamespace(_tail_device=torch.device("cpu"), _ae_decoders=dict(aes),
                                _codec=FakeCodec(calls)))
    return ue, edge


class ModelAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This verifies source/checkpoint bytes but never deserializes weights.
        with patch.object(torch, "load", side_effect=AssertionError("checkpoint load forbidden")), \
                patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
            cls.contract = load_dynamic_execution_contract()
            cls.bindings = A.source_binding_hashes()

    def setUp(self):
        self.calls = []
        self.ue, self.edge = preloads(self.contract, self.calls)
        self.context = build_frame_context_v1(stream_id="saved-stream", frame_id=105,
            sequence_id=8, capture_timestamp_ns=123456789000, ego_world_x=1., ego_world_y=2.,
            ego_world_z=3., ego_world_pitch=0., ego_world_yaw=10., ego_world_roll=0.)
        self.adapter = A.ReplayModelAdapter.from_preloaded(ue_runtime=self.ue, edge=self.edge,
            contract=self.contract, device="cpu", expected_binding_hashes=self.bindings)

    def envelope(self, mode=11, q=8123, payload=None):
        profile = self.contract.resolve_q_e4(mode, q)
        payload = inner_bytes(profile) if payload is None else payload
        return X.ExecutionEnvelopeV3(mode_id=mode, q_e4=q, keep_count=profile.keep_count,
            anchor_action_id=profile.action_id, reward_requested=True,
            session_uuid="00000000-0000-0000-0000-000000000001",
            controller_lineage_sha256="a" * 64, decision_seq=4, ticket_seq=4,
            frame_id=105, tensor_seq=8, capture_timestamp_ns=self.context.capture_timestamp_ns,
            execution_bundle_sha256=profile.execution_bundle_sha256,
            inner_payload_sha256=hashlib.sha256(payload).hexdigest(), inner_payload=payload)

    def replay(self, archived=None):
        envelope = self.envelope()
        return replay_pair(prepared_input=torch.zeros((1, 7, 448, 768)),
            selected_wire=wire.pack_sfd4(envelope, self.context),
            expected_identity=protocol.run4_identity(envelope),
            archived_selected=archived or Prediction(np.zeros((720, 1280), dtype=np.uint8),
                                                   [prediction_record(self.context)]),
            **self.adapter.replay_callbacks())

    def test_import_is_standard_library_only_and_starts_no_threads(self):
        program = (
            "import sys, threading; before=threading.active_count(); "
            "import rl_agent.splitfusion_paired_quality_replay_v2.model_adapter; "
            "assert 'torch' not in sys.modules; assert threading.active_count()==before"
        )
        result = subprocess.run([sys.executable, "-c", program],
            cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_original_callbacks_preserve_dense_c2_and_contextual_full_records(self):
        with patch.object(torch, "load", side_effect=AssertionError("no checkpoint load")), \
                patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("no CUDA")):
            result = self.replay()
        self.assertEqual([entry[0] for entry in self.calls],
                         ["front", "encode", "inspect", "decode", "tail", "serialize",
                          "snapshot", "tail", "serialize", "snapshot"])
        tails = [entry for entry in self.calls if entry[0] == "tail"]
        self.assertEqual([entry[1] for entry in tails], [2., 3.])
        self.assertIs(tails[0][2], tails[1][2])
        self.assertEqual(result.dense.records[0]["world_x"], 3.)
        self.assertEqual(result.selected.records[0]["unmodified_complete_record_field"], "retained")
        self.assertEqual(int(result.dense.labels[0, 0]), 1)
        self.assertTrue(result.evidence["selected_complete_records_identical"])

    def test_all_modes_and_off_anchor_keep_exact_original_encoder_decoder_dispatch(self):
        for mode in range(12):
            for q in (0, 4321, 9800):
                envelope = self.envelope(mode, q)
                self.calls.clear()
                payload = self.adapter.encode_selected(torch.zeros(1), envelope)
                self.adapter.decode_selected(payload, envelope)
                encoded = self.calls[0]
                family = self.contract.mode(mode).family
                self.assertEqual(payload, envelope.inner_payload)
                self.assertEqual(encoded[1:3], (family, q))
                self.assertIs(encoded[3], None if q == 0 else self.ue._ranker)
                self.assertIs(encoded[4], None if family == "noAE" else self.ue._ae_encoders[family])
                self.assertIs(self.calls[-1][1], None if family == "noAE" else self.edge.runtime._ae_decoders[family])
                self.assertEqual(envelope.anchor_action_id is None, q == 4321)

    def test_dynamic_identity_tamper_is_refused_before_codec(self):
        envelope = self.envelope()
        changes = ({"keep_count": envelope.keep_count + 1}, {"anchor_action_id": 71},
                   {"execution_bundle_sha256": "0" * 64}, {"inner_payload_sha256": "0" * 64},
                   {"q_e4": envelope.q_e4 + 1}, {"mode_id": 10})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(A.ModelAdapterError):
                self.adapter.encode_selected(torch.zeros(1), replace(envelope, **change))
        self.assertEqual(self.calls, [])

    def test_wrong_inner_identity_refuses_decoder(self):
        bad_inner = json.loads(self.envelope().inner_payload)
        bad_inner["q_e4"] -= 1
        payload = json.dumps(bad_inner).encode()
        with self.assertRaises(DispatchContractError):
            self.adapter.decode_selected(payload, self.envelope(payload=payload))
        self.assertEqual(self.calls, [("inspect",)])

    def test_decoding_foreign_payload_is_refused_before_inspection(self):
        with self.assertRaisesRegex(A.ModelAdapterError, "saved selected bytes"):
            self.adapter.decode_selected(b"other", self.envelope())
        self.assertEqual(self.calls, [])

    def test_selected_label_failure_never_runs_dense(self):
        labels = np.zeros((720, 1280), dtype=np.uint8)
        labels[0, 0] = 2
        with self.assertRaisesRegex(ReplayParityError, "native-label parity"):
            self.replay(Prediction(labels, [prediction_record(self.context)]))
        self.assertEqual([row[1] for row in self.calls if row[0] == "tail"], [2.])

    def test_selected_full_record_failure_never_runs_dense(self):
        records = [dict(prediction_record(self.context), unmodified_complete_record_field="changed")]
        with self.assertRaisesRegex(ReplayParityError, "complete-record parity"):
            self.replay(Prediction(np.zeros((720, 1280), dtype=np.uint8), records))
        self.assertEqual([row[1] for row in self.calls if row[0] == "tail"], [2.])

    def test_tail_requires_saved_envelope_and_matching_frame(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_edge_runtime_v2 import Run4TailMetadataV2
        envelope = self.envelope()
        metadata = Run4TailMetadataV2(self.context, self.context.sequence_id,
                                     self.context.capture_timestamp_ns, protocol.run4_identity(envelope))
        with self.assertRaisesRegex(A.ModelAdapterError, "decode must precede"):
            self.adapter.tail(torch.zeros(1), metadata)
        self.adapter.decode_selected(envelope.inner_payload, envelope)
        with self.assertRaisesRegex(A.ModelAdapterError, "frame context differs"):
            self.adapter.tail(torch.zeros(1), replace(metadata, sequence_id=999))
        self.assertFalse(any(entry[0] == "tail" for entry in self.calls))

    def test_invalid_padding_refuses_front(self):
        value = torch.zeros((1, 7, 448, 768))
        value[0, 6, 447, 0] = 1.
        with self.assertRaisesRegex(A.ModelAdapterError, "bottom padding"):
            self.adapter.front(value)
        self.assertEqual(self.calls, [])

    def test_sources_must_be_complete_and_match_capture(self):
        for expected in ({}, {**self.bindings, A.REQUIRED_SOURCE_PATHS[-1]: "0" * 64}):
            with self.assertRaises(A.ModelAdapterError):
                A.ReplayModelAdapter.from_preloaded(ue_runtime=self.ue, edge=self.edge,
                    contract=self.contract, device="cpu", expected_binding_hashes=expected)

    def test_preload_parameter_training_and_family_mismatch_refused(self):
        self.edge.model.train()
        with self.assertRaisesRegex(A.ModelAdapterError, "eval mode"):
            A.ReplayModelAdapter.from_preloaded(ue_runtime=self.ue, edge=self.edge,
                contract=self.contract, device="cpu", expected_binding_hashes=self.bindings)
        self.edge.model.eval()
        self.ue._ae_encoders["AE64"].routing_tag = -1
        with self.assertRaisesRegex(A.ModelAdapterError, "encoder identity differs"):
            A.ReplayModelAdapter.from_preloaded(ue_runtime=self.ue, edge=self.edge,
                contract=self.contract, device="cpu", expected_binding_hashes=self.bindings)

    def test_cold_load_requires_explicit_opt_in_device_and_source_hashes(self):
        loader = Mock(side_effect=AssertionError("must not load"))
        with self.assertRaisesRegex(A.ModelAdapterError, "allow_cold_load"):
            A.load_adapter(device="cpu", expected_binding_hashes=self.bindings, ue_loader=loader)
        with self.assertRaisesRegex(A.ModelAdapterError, "explicit replay device"):
            A.load_adapter(device=None, allow_cold_load=True, expected_binding_hashes=self.bindings,
                           ue_loader=loader, edge_loader=loader, contract_loader=loader)
        with self.assertRaisesRegex(A.ModelAdapterError, "inventory required"):
            A.load_adapter(device="cpu", allow_cold_load=True, expected_binding_hashes={},
                           ue_loader=loader, edge_loader=loader, contract_loader=loader)
        loader.assert_not_called()

    def test_actual_v3_cpu_substitution_refused_before_any_model_load(self):
        loader = Mock(side_effect=AssertionError("must not load"))
        with self.assertRaisesRegex(A.ModelAdapterError, "V3 tail requires an explicit CUDA"):
            A.load_adapter(device="cpu", allow_cold_load=True, expected_binding_hashes=self.bindings,
                           ue_loader=loader, contract_loader=loader)
        loader.assert_not_called()

    def test_injected_cold_load_reuses_authoritative_preload_shape_without_cuda(self):
        ue_loader = Mock(return_value=(self.ue, object(), []))
        edge_loader = Mock(return_value=self.edge)
        contract_loader = Mock(return_value=self.contract)
        with patch.object(torch, "load", side_effect=AssertionError("no real checkpoint load")), \
                patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("no CUDA")):
            adapter = A.load_adapter(device="cpu", allow_cold_load=True,
                expected_binding_hashes=self.bindings, ue_loader=ue_loader,
                edge_loader=edge_loader, contract_loader=contract_loader)
        ue_loader.assert_called_once_with(torch.device("cpu"))
        edge_loader.assert_called_once_with(torch.device("cpu"))
        contract_loader.assert_called_once_with()
        self.assertIs(adapter._ue, self.ue)
        self.assertIs(adapter._edge, self.edge)


if __name__ == "__main__":
    unittest.main()
