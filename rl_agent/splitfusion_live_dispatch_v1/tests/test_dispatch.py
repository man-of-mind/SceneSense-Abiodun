from __future__ import annotations

import json
import struct
import tempfile
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from rl_agent.splitfusion_live_dispatch_v1 import phase13b_qualification as phase13b
from rl_agent.splitfusion_live_dispatch_v1.edge_runtime import PreloadedSplitEdgeRuntime
from rl_agent.splitfusion_live_dispatch_v1.envelope import pack_envelope, unpack_envelope
from rl_agent.splitfusion_live_dispatch_v1.frame_context import (
    STATIC_CAMERA_MODEL_SHA256,
    STATIC_CAMERA_MOUNT_SHA256,
    FrameContextV1,
    Pose6D,
    StaticCameraRegistry,
)
from rl_agent.splitfusion_live_dispatch_v1.context_tail import (
    bind_context_service_record_identity,
)
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
from rl_agent.ue_route_b_split_cell_adapter_v1 import (
    AdapterError,
    create_cell_edge_state_root,
)


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
    def test_phase15_edge_state_is_fresh_cell_scoped_and_explicitly_mounted(self):
        self.assertEqual(
            len(SplitActionRegistry.from_runtime_binding().profiles), 72
        )
        with tempfile.TemporaryDirectory() as raw_owner:
            owner = Path(raw_owner)
            state = create_cell_edge_state_root(owner)
            self.assertEqual(state.parent, owner.resolve(strict=True))
            self.assertEqual(state.name, "splitfusion_edge_state")
            self.assertTrue(state.is_dir())
            with self.assertRaisesRegex(AdapterError, "already exists"):
                create_cell_edge_state_root(owner)
        compose = (
            Path(__file__).resolve().parents[3]
            / "receiver_container/docker-compose.fusion-back.yaml"
        ).read_text(encoding="utf-8")
        base_compose = (
            Path(__file__).resolve().parents[3]
            / "receiver_container/docker-compose.yaml"
        ).read_text(encoding="utf-8")
        helper = (
            Path(__file__).resolve().parents[3]
            / "scripts/receiver_container_fusion_back_up.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "${SPLITFUSION_EDGE_STATE_ROOT:-../torch_cache}:/work/torch_cache:rw",
            compose,
        )
        self.assertIn(
            "PYTHONPATH=/work/abiodun:/work/abiodun/rl_agent/feature_ae",
            base_compose,
        )
        self.assertNotIn(
            "PYTHONPATH=/work/abiodun/pole_lraspp_multimodal_fusion:",
            base_compose,
        )
        self.assertIn(
            'SPLITFUSION_EDGE_STATE_ROOT="${SPLITFUSION_EDGE_STATE_ROOT}"',
            helper,
        )

    def test_sfd1_v2_frame_context_validates_before_decode_and_tracks_session(self):
        registry = SplitActionRegistry.from_runtime_binding(
            verify_runtime_artifacts=False
        )
        cameras = StaticCameraRegistry.audited()
        profile = registry.find("noAE", "UINT8", 0)
        context = FrameContextV1(
            stream_id="ue288/session-a",
            frame_id=214,
            sequence_id=7,
            capture_timestamp_ns=12_500_000_000,
            ego_world=Pose6D(-3.9741828441619873, 28.094629287719727,
                             -0.10292118787765503, -0.048848289996385574,
                             0.1552259624004364, 1.5732501745224),
            camera_model_sha256=STATIC_CAMERA_MODEL_SHA256,
            camera_mount_sha256=STATIC_CAMERA_MOUNT_SHA256,
        )

        def objects():
            front, ranker, tail = _FakeFront(), _FakeRanker(), _FakeTail()
            encoders, decoders = _objects(registry)
            codec = _FakeCodec()
            ue = PreloadedSplitUERuntime(
                registry, front=front, ranker=ranker, ae_encoders=encoders,
                device=torch.device("cpu"), codec=codec,
            )
            edge = PreloadedSplitEdgeRuntime(
                registry, frozen_p025_tail=tail, ae_decoders=decoders,
                tail_device=torch.device("cpu"), codec=codec,
                camera_registry=cameras, require_frame_context=True,
            )
            return ue, edge, codec, tail

        ue, edge, codec, tail = objects()
        prepared = ue.prepare(
            profile.action_id, object(), sequence_id=context.sequence_id,
            capture_timestamp_ns=context.capture_timestamp_ns,
            frame_context=context,
        )
        outer = unpack_envelope(prepared.wire_bytes)
        self.assertEqual(outer.protocol_version, 2)
        self.assertEqual(outer.frame_context, context)
        self.assertNotEqual(context.frame_id, context.sequence_id)
        service_record = bind_context_service_record_identity(
            {"sample_id": "source-frame", "frame_id": str(context.frame_id)},
            context,
        )
        self.assertEqual(service_record["frame_id"], context.frame_id)
        self.assertIs(type(service_record["frame_id"]), int)
        self.assertEqual(service_record["stream_id"], context.stream_id)
        with self.assertRaisesRegex(
            DispatchContractError, "does not equal FrameContext"
        ):
            bind_context_service_record_identity(
                {"sample_id": "source-frame", "frame_id": context.sequence_id},
                context,
            )
        self.assertEqual(prepared.outer_envelope_bytes, 180 + len(context.stream_id))
        result = edge.process(
            prepared.wire_bytes, transmitted_action_id=profile.action_id
        )
        self.assertEqual(result.metadata.frame_context, context)
        self.assertEqual((codec.inspect_calls, codec.decode_calls, tail.calls), (1, 1, 1))
        with self.assertRaisesRegex(DispatchContractError, "duplicate frame context"):
            edge.process(prepared.wire_bytes, transmitted_action_id=profile.action_id)
        self.assertEqual((codec.inspect_calls, codec.decode_calls, tail.calls), (1, 1, 1))

        def refused_before_inspect(wire, pattern):
            _ue, candidate_edge, candidate_codec, candidate_tail = objects()
            with self.assertRaisesRegex(DispatchContractError, pattern):
                candidate_edge.process(wire, transmitted_action_id=profile.action_id)
            self.assertEqual(
                (candidate_codec.inspect_calls, candidate_codec.decode_calls,
                 candidate_tail.calls),
                (0, 0, 0),
            )

        altered_sequence = bytearray(prepared.wire_bytes)
        altered_sequence[12:20] = (context.sequence_id + 1).to_bytes(8, "little")
        refused_before_inspect(bytes(altered_sequence), "sequence mismatch")
        altered_timestamp = bytearray(prepared.wire_bytes)
        altered_timestamp[20:28] = (context.capture_timestamp_ns + 1).to_bytes(8, "little")
        refused_before_inspect(bytes(altered_timestamp), "timestamp mismatch")
        nonfinite = bytearray(prepared.wire_bytes)
        nonfinite[68:76] = struct.pack("<d", float("nan"))
        refused_before_inspect(bytes(nonfinite), "pose is non-finite")

        for field, digest, pattern in (
            ("camera_model_sha256", "0" * 64, "unknown static-camera hash"),
            ("camera_mount_sha256", "1" * 64, "unknown static mount hash"),
        ):
            changed = FrameContextV1(
                **{
                    **context.__dict__,
                    field: digest,
                    "stream_id": f"ue288/{field}",
                }
            )
            changed_wire = ue.prepare(
                profile.action_id, object(), sequence_id=changed.sequence_id,
                capture_timestamp_ns=changed.capture_timestamp_ns,
                frame_context=changed,
            ).wire_bytes
            refused_before_inspect(changed_wire, pattern)
    def test_phase13b_porcelain_status_preserves_leading_space_and_refuses_unknown(self):
        exact_status = (
            " m OAI/openairinterface5g\n"
            " M pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
            "lraspp_to_splitfusion_fcos_report_v1/FULL_TECHNICAL_REPORT_AVO_V2.md\n"
        )
        completed = mock.Mock(stdout=exact_status)
        with mock.patch.object(phase13b.subprocess, "run", return_value=completed):
            porcelain = phase13b._git_output(
                "status", "--porcelain=v1", "--untracked-files=all"
            )
        self.assertEqual(porcelain.splitlines()[0][3:], "OAI/openairinterface5g")

        exact_git = (
            "workload-policy-repair-commit",
            phase13b.PHASE13B_PORCELAIN_REPAIR_COMMIT,
            phase13b.PHASE13B_IMPLEMENTATION_COMMIT,
            phase13b.PHASE13A_COMMIT,
            porcelain,
        )
        with mock.patch.object(phase13b, "_git_output", side_effect=exact_git):
            audit = phase13b._verify_git_state()
        self.assertEqual(
            audit["expected_user_owned_dirty_paths"],
            sorted(phase13b.EXPECTED_DIRTY_PATHS),
        )

        unknown_git = (*exact_git[:4], porcelain + "\n?? unknown-path")
        with mock.patch.object(phase13b, "_git_output", side_effect=unknown_git):
            with self.assertRaisesRegex(RuntimeError, "unexpected dirty paths"):
                phase13b._verify_git_state()

    def test_phase13b_gpu_workload_policy_accepts_infrastructure_and_refuses_python(self):
        baseline = (
            "2564, /usr/lib/xorg/Xorg, 31\n"
            "3086, /usr/bin/gnome-shell, 72\n"
            "3031, /usr/libexec/gnome-remote-desktop-daemon, 502\n"
            "2081845, /snap/firefox/firefox, 18\n"
            "2016454, nvidia-cuda-mps-server, 54\n"
        )
        accepted = phase13b._audit_gpu_workloads(
            baseline, mps_client_commands=[], current_pid=9000
        )
        self.assertEqual(
            {
                row["executable_basename"]
                for row in accepted["allowed_infrastructure_processes"]
            },
            phase13b.GPU_INFRASTRUCTURE_BASENAMES,
        )
        scientific_client = [
            {
                "pid": 4242,
                "executable_basename": "python3",
                "command": "/usr/bin/python3 -u train_model.py --device cuda:0",
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "scientific CUDA workload"):
            phase13b._audit_gpu_workloads(
                baseline,
                mps_client_commands=scientific_client,
                current_pid=9000,
            )

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

    def test_latest_frame_and_deadline_keep_stale_work_out_of_the_tail(self):
        """Phase-15 recovery: freshness and the deadline gate the tail.

        The retry4 audit measured an implicit FIFO backlog whose depth in
        frames grew as the payload shrank, and a 500 ms processing horizon that
        cancelled nothing. A newer frame must displace an older pending one,
        and an expired capture must never reach frozen-tail inference. The
        earlier 100 ms service target remains separately represented.
        """

        from rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime import (
            EDGE_STAGE_BEFORE_TAIL,
            DeadlineExpired,
            FastStationaryTrackAccumulator,
            LatestFramePendingSlot,
            _DeadlineGuardedTail,
            ack_timeout_s,
            check_deadline,
            deadline_at_s,
            service_deadline_s,
        )
        from pole_lraspp_multimodal_fusion.pole_lraspp_multimodal_fusion.radar_fusion import (
            StationaryTrackAccumulator,
        )

        campaign = {"cell": {"service_deadline_ms": 100, "ack_timeout_ms": 500}}
        service_s = service_deadline_s(campaign)
        deadline_s = ack_timeout_s(campaign)
        self.assertEqual((service_s, deadline_s), (0.1, 0.5))
        slot = LatestFramePendingSlot()

        # Latest-frame-first: at most one pending frame, newest wins, and the
        # displaced frame is handed back so it can be recorded exactly.
        admitted, displaced = slot.offer("stream-a", "frame-1", sequence=1)
        self.assertTrue(admitted)
        self.assertIsNone(displaced)
        admitted, displaced = slot.offer("stream-a", "frame-2", sequence=2)
        self.assertTrue(admitted)
        self.assertEqual(displaced, "frame-1")
        self.assertEqual(slot.depth(), 1)
        # An out-of-order arrival never displaces fresher pending work.
        admitted, displaced = slot.offer("stream-a", "frame-0", sequence=0)
        self.assertFalse(admitted)
        self.assertIsNone(displaced)
        self.assertEqual(slot.take(timeout=0.1), ("stream-a", "frame-2"))
        self.assertIsNone(slot.take(timeout=0.01))

        # Both absolute instants are capture-derived and remain distinct.
        capture_ns = 1_000_000_000_000
        self.assertAlmostEqual(
            deadline_at_s(capture_ns, deadline_s), capture_ns / 1e9 + deadline_s
        )
        self.assertAlmostEqual(
            deadline_at_s(capture_ns, service_s), capture_ns / 1e9 + service_s
        )

        calls = []
        guarded = _DeadlineGuardedTail(
            lambda c2, metadata: calls.append(metadata.capture_timestamp_ns),
            lambda stage, capture: check_deadline(stage, capture, deadline_s),
        )
        fresh = mock.Mock(capture_timestamp_ns=int(time.time() * 1e9))
        guarded(object(), fresh)
        self.assertEqual(len(calls), 1)

        stale = mock.Mock(
            capture_timestamp_ns=int((time.time() - 5.0) * 1e9)
        )
        with self.assertRaises(DeadlineExpired) as raised:
            guarded(object(), stale)
        self.assertEqual(raised.exception.stage, EDGE_STAGE_BEFORE_TAIL)
        self.assertGreater(raised.exception.age_ms, deadline_s * 1000.0)
        # The expired capture never reached frozen-tail inference.
        self.assertEqual(len(calls), 1)

        # The vectorized live tracker is bit-identical to the frozen reference,
        # including within-cell moving resets and stale-track eviction.
        rng = np.random.default_rng(20260906)
        reference = StationaryTrackAccumulator()
        optimized = FastStationaryTrackAccumulator()
        for step in range(6):
            points = rng.normal(size=(4096, 4)).astype(np.float32)
            points[:, :2] = np.round(points[:, :2] * 25.0, 1)
            points[:, 3] = rng.choice(
                np.asarray((-1.0, 0.0, 0.1, 0.4), dtype=np.float32),
                size=len(points),
            )
            observed_at = 100.0 + step * (0.1 if step < 5 else 3.0)
            expected = reference.update(points, observed_at)
            actual = optimized.update(points, observed_at)
            self.assertTrue(np.array_equal(actual, expected))
            self.assertEqual(optimized.tracks_snapshot(), reference._tracks)

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
        unsupported[4:6] = (3).to_bytes(2, "little")
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
