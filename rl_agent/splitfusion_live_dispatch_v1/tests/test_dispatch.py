from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from unittest import mock

import torch

from rl_agent.splitfusion_live_dispatch_v1.edge_runtime import PreloadedSplitEdgeRuntime
from rl_agent.splitfusion_live_dispatch_v1.envelope import pack_envelope, unpack_envelope
from rl_agent.splitfusion_live_dispatch_v1.registry import (
    DispatchContractError,
    SplitActionRegistry,
)
from rl_agent.splitfusion_live_dispatch_v1.transport import (
    DecodedC2,
    InnerIdentity,
    InspectedInnerPayload,
    expected_inner_identity,
)
from rl_agent.splitfusion_live_dispatch_v1.ue_runtime import PreloadedSplitUERuntime


class _FakeC2:
    def __init__(self, device: torch.device) -> None:
        self.device = device


class _FakeModule:
    constructions = 0

    def __init__(self) -> None:
        _FakeModule.constructions += 1
        self.to_calls = 0
        self.eval_calls = 0
        self.device = torch.device("cpu")

    def to(self, device: torch.device):
        self.to_calls += 1
        self.device = device
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def parameters(self):
        return ()


class _FakeFront(_FakeModule):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __call__(self, _input):
        self.calls += 1
        return _FakeC2(self.device)


class _FakeRanker(_FakeModule):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def score_cells(self, _c2):
        self.calls += 1
        return object()


class _FakeAE(_FakeModule):
    def __init__(self, profile) -> None:
        super().__init__()
        self.family_id = profile.family_id
        self.bottleneck = profile.transported_channels
        self.routing_tag = profile.routing_tag
        self.encode_calls = 0
        self.decode_calls = 0

    def encode(self, _c2):
        self.encode_calls += 1
        return object()

    def decode(self, *_args):
        self.decode_calls += 1
        return _FakeC2(self.device)


class _FakeTail(_FakeModule):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __call__(self, c2, metadata):
        self.calls += 1
        return {"profile_id": metadata.profile_id, "c2_identity": id(c2)}


class _FakeCodec:
    def __init__(self) -> None:
        self.inspect_calls = 0
        self.decode_calls = 0

    def encode(self, profile, c2, *, ranker, ae_encoder, timing):
        with timing.stage("ranker_selection"):
            if profile.q_e4 == 0:
                if ranker is not None:
                    raise AssertionError("q=0 did not bypass ranker dispatch")
            else:
                ranker.score_cells(c2)
        with timing.stage("ae_encode"):
            if profile.family == "noAE":
                if ae_encoder is not None:
                    raise AssertionError("noAE selected an encoder")
            else:
                ae_encoder.encode(c2)
        with timing.stage("quantize_pack"):
            payload = json.dumps(
                asdict(expected_inner_identity(profile)), sort_keys=True
            ).encode("utf-8")
        with timing.stage("zstd_compression"):
            return payload

    def inspect(self, payload, *, timing):
        self.inspect_calls += 1
        with timing.stage("zstd_decompression"):
            decoded = json.loads(payload.decode("utf-8"))
        identity = InnerIdentity(**decoded)
        return InspectedInnerPayload(
            identity=identity,
            compressed_bytes=len(payload),
            uncompressed_bytes=len(payload),
            kind="fake",
            sparse_bytes=payload,
            parsed=None,
        )

    def decode(self, inspected, *, decoder, tail_device, timing):
        self.decode_calls += 1
        with timing.stage("unpack_dequantize"):
            pass
        with timing.stage("ae_decode"):
            c2 = _FakeC2(tail_device) if decoder is None else decoder.decode(None, None)
        return DecodedC2(c2=c2, finite=True, device=c2.device)


def _objects(registry):
    encoders = {
        family: _FakeAE(registry.find(family, "UINT8", 0))
        for family in ("AE128", "AE64", "AE32")
    }
    decoders = {
        family: _FakeAE(registry.find(family, "UINT8", 0))
        for family in ("AE128", "AE64", "AE32")
    }
    return encoders, decoders


class PreloadedDispatchTest(unittest.TestCase):
    def test_representative_action_switches_reuse_only_preloaded_objects(self):
        registry = SplitActionRegistry.from_runtime_binding(
            verify_runtime_artifacts=False
        )
        self.assertEqual(len(registry.profiles), 72)
        front, ranker, tail = _FakeFront(), _FakeRanker(), _FakeTail()
        encoders, decoders = _objects(registry)
        codec = _FakeCodec()
        ue = PreloadedSplitUERuntime(
            registry,
            front=front,
            ranker=ranker,
            ae_encoders=encoders,
            device=torch.device("cpu"),
            codec=codec,
            startup_model_load_operations=5,
            startup_model_construction_operations=5,
        )
        edge = PreloadedSplitEdgeRuntime(
            registry,
            frozen_p025_tail=tail,
            ae_decoders=decoders,
            tail_device=torch.device("cpu"),
            codec=codec,
            output_serializer=lambda _output: b"serialized",
            startup_model_load_operations=4,
            startup_model_construction_operations=4,
        )
        module_constructions = _FakeModule.constructions
        startup_calls = {
            id(module): (module.to_calls, module.eval_calls)
            for module in (front, ranker, tail, *encoders.values(), *decoders.values())
        }
        profiles = (
            registry.find("noAE", "UINT8", 0),
            registry.find("AE128", "UINT6", 3000),
            registry.find("AE64", "UINT4", 5000),
            registry.find("AE32", "UINT8", 7000),
        )
        with mock.patch("torch.load", side_effect=AssertionError("hot-path torch.load")):
            for sequence, profile in enumerate(profiles, start=100):
                prepared = ue.prepare(
                    profile.action_id,
                    object(),
                    sequence_id=sequence,
                    capture_timestamp_ns=1_000_000 + sequence,
                )
                result = edge.process(
                    prepared.wire_bytes,
                    transmitted_action_id=profile.action_id,
                )
                self.assertEqual(result.perception["profile_id"], profile.profile_id)
                self.assertEqual(result.metadata.action_id, profile.action_id)
                self.assertEqual(prepared.outer_envelope_bytes, 36)
                self.assertEqual(
                    prepared.inner_payload_bytes + prepared.outer_envelope_bytes,
                    prepared.total_transmitted_bytes,
                )
                self.assertEqual(
                    {boundary.name for boundary in prepared.timing.boundaries},
                    set(("total_ue_preparation", "front_backbone", "ranker_selection", "ae_encode", "quantize_pack", "zstd_compression")),
                )
                self.assertEqual(
                    {boundary.name for boundary in result.timing.boundaries},
                    set(("total_edge_processing", "zstd_decompression", "unpack_dequantize", "ae_decode", "frozen_tail", "output_serialization")),
                )
        self.assertEqual(_FakeModule.constructions, module_constructions)
        self.assertEqual(
            {
                id(module): (module.to_calls, module.eval_calls)
                for module in (front, ranker, tail, *encoders.values(), *decoders.values())
            },
            startup_calls,
        )
        self.assertEqual(front.calls, 4)
        self.assertEqual(ranker.calls, 3)
        self.assertEqual([encoders[name].encode_calls for name in ("AE128", "AE64", "AE32")], [1, 1, 1])
        self.assertEqual([decoders[name].decode_calls for name in ("AE128", "AE64", "AE32")], [1, 1, 1])
        self.assertEqual(tail.calls, 4)
        self.assertEqual((ue.counters.frames_completed, edge.counters.frames_completed), (4, 4))
        self.assertEqual((ue.counters.hot_path_model_load_operations, edge.counters.hot_path_model_load_operations), (0, 0))
        self.assertEqual((ue.counters.hot_path_model_construction_operations, edge.counters.hot_path_model_construction_operations), (0, 0))
        self.assertEqual(registry.startup_audit.catalog_reads, 1)

    def test_q0_bypasses_ranker_and_mismatches_stop_before_decode_or_tail(self):
        registry = SplitActionRegistry.from_runtime_binding(
            verify_runtime_artifacts=False
        )
        front, ranker, tail = _FakeFront(), _FakeRanker(), _FakeTail()
        encoders, decoders = _objects(registry)
        codec = _FakeCodec()
        ue = PreloadedSplitUERuntime(
            registry,
            front=front,
            ranker=ranker,
            ae_encoders=encoders,
            device=torch.device("cpu"),
            codec=codec,
        )
        edge = PreloadedSplitEdgeRuntime(
            registry,
            frozen_p025_tail=tail,
            ae_decoders=decoders,
            tail_device=torch.device("cpu"),
            codec=codec,
        )
        q0 = registry.find("noAE", "UINT8", 0)
        prepared = ue.prepare(
            q0.action_id,
            object(),
            sequence_id=7,
            capture_timestamp_ns=99,
        )
        self.assertEqual(ranker.calls, 0)
        outer = unpack_envelope(prepared.wire_bytes)
        base = json.loads(outer.inner_payload.decode("utf-8"))
        mismatches = {
            "family": "AE128",
            "family_id": 1,
            "quantizer": "UINT6",
            "bit_width": 6,
            "q_e4": 3000,
            "keep_count": q0.keep_count - 1,
            "routing_tag": 1,
            "transported_channels": 128,
            "latent_width": 128,
            "wire_magic_ascii": "AE8\\0",
            "wire_codec_id": 2,
            "wire_version": 2,
        }
        for field, value in mismatches.items():
            with self.subTest(field=field):
                changed = dict(base)
                changed[field] = value
                wire = pack_envelope(
                    json.dumps(changed, sort_keys=True).encode("utf-8"),
                    action_id=q0.action_id,
                    sequence_id=outer.sequence_id,
                    capture_timestamp_ns=outer.capture_timestamp_ns,
                )
                with self.assertRaisesRegex(DispatchContractError, field):
                    edge.process(wire, transmitted_action_id=q0.action_id)
        with self.assertRaisesRegex(DispatchContractError, "control-plane action_id"):
            edge.process(prepared.wire_bytes, transmitted_action_id=q0.action_id + 1)
        unsupported = bytearray(prepared.wire_bytes)
        unsupported[4:6] = (2).to_bytes(2, "little")
        with self.assertRaisesRegex(DispatchContractError, "protocol version"):
            edge.process(bytes(unsupported), transmitted_action_id=q0.action_id)
        unregistered = pack_envelope(
            outer.inner_payload,
            action_id=72,
            sequence_id=outer.sequence_id,
            capture_timestamp_ns=outer.capture_timestamp_ns,
        )
        with self.assertRaisesRegex(DispatchContractError, "unregistered action_id"):
            edge.process(unregistered, transmitted_action_id=72)
        self.assertEqual(codec.decode_calls, 0)
        self.assertEqual(tail.calls, 0)
        self.assertEqual(sum(decoder.decode_calls for decoder in decoders.values()), 0)
        self.assertEqual(edge.counters.tail_dispatches, 0)


if __name__ == "__main__":
    unittest.main()
