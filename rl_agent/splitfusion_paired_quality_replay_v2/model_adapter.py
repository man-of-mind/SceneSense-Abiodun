"""Opt-in adapters for the exact v5 complete-GT selected/dense model paths.

Importing this module imports only the standard library. It loads no checkpoint,
chooses no device, creates no worker, and never initializes CUDA. Real execution
requires an explicit device and either authoritative resident preloads or
``allow_cold_load=True``. The real optimized V3 tail is CUDA-only: substituting a
CPU/reference tail is not permitted as a way to pass selected prediction parity.

The caller must archive ``source_binding_hashes()`` prospectively and pass that
inventory back as ``expected_binding_hashes``. It binds the runtime and dynamic
manifests, repaired 432-content/448-padded input geometry, contextual camera
geometry, production codecs, frozen preload entry points, and optimized V3 tail
and record serialization. The dynamic contract additionally verifies its entire
checkpoint/source inventory. The five required checkpoint digests below are the
original Run5 frozen perception/ranker/AE artifacts, not learned-actor weights.
The actor and exact saved SFD4 action identity belong to the caller's archive.

This adapter intentionally does not accept reconstructed sensor inputs, weaken
the sealed v1 ``replay_pair`` byte/label/record gates, or make latency claims.
Different accelerator/library arithmetic can still fail exact selected parity;
that is an unresolved replay qualification failure, never grounds for tolerance.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping


SCHEMA = "splitfusion.paired_quality_replay.model_adapter.v2"
RUNTIME_BINDING_SHA256 = "604450c7f0f791a480ba4e5ed6b286a2b6f68497735372a7a63ed92e93585551"
CHECKPOINT_SHA256 = MappingProxyType({
    "perception": "da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f",
    "ranker": "07781c56a4c0f306f16d332f64627ce6b9458e154f40ab9fef89f89909b79cb5",
    "AE128": "0c2ba3a495684c0f8222492f554eb3de7c7a76181e0bd4b4a83529897db30f72",
    "AE64": "dd7c5124e27114584ab2083e59160a3ff2a2d040d0a37d22564ac98c838aa8e0",
    "AE32": "e2f867757e8db0620316c092264ac7eb53d12bb5ef66ed14475eb40693d1f271",
})
AE_FAMILIES = ("AE128", "AE64", "AE32")
REQUIRED_SOURCE_PATHS = (
    "rl_agent/splitfusion_live_dispatch_v1/runtime_binding.json",
    "rl_agent/splitfusion_live_dispatch_v1/dynamic_execution_source_binding.json",
    "rl_agent/splitfusion_live_dispatch_v1/dynamic_execution_contract.py",
    "rl_agent/splitfusion_live_dispatch_v1/live_pilot_runtime.py",
    "rl_agent/splitfusion_live_dispatch_v1/frame_context.py",
    "rl_agent/splitfusion_live_dispatch_v1/context_tail.py",
    "rl_agent/splitfusion_live_dispatch_v1/transport.py",
    "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/continuous_execution_v2.py",
    "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/run4_live_wire_v2.py",
    "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/run4_map_protocol_v2.py",
    "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/phase6_edge_runtime_v2.py",
    "rl_agent/splitfusion_live_parity_repair_v1/perception.py",
    "rl_agent/splitfusion_live_parity_repair_v1/runtime.py",
    "rl_agent/splitfusion_live_parity_repair_v2/prediction_archive.py",
    "rl_agent/splitfusion_live_parity_runtime_v5/entries.py",
    "rl_agent/splitfusion_edge_optimization_v1/detached_edge_preload_v3.py",
    "rl_agent/splitfusion_edge_optimization_v1/codec_v3.py",
    "rl_agent/splitfusion_edge_optimization_v1/detached_tail.py",
    "rl_agent/splitfusion_edge_optimization_v1/detached_tail_v3.py",
    "rl_agent/splitfusion_edge_optimization_v1/optimized_tail.py",
    "rl_agent/splitfusion_edge_optimization_v1/optimized_tail_v2.py",
    "rl_agent/splitfusion_edge_optimization_v1/optimized_tail_v3.py",
)


class ModelAdapterError(ValueError):
    """A source, preload, geometry, or exact execution binding disagrees."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ModelAdapterError(message)


def source_binding_hashes(repository_root: Path | None = None) -> dict[str, str]:
    """Read source bytes only; callers retain this inventory with the capture."""
    root = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[2]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in REQUIRED_SOURCE_PATHS}


def _verify_sources(expected: Mapping[str, str], root: Path | None) -> dict[str, str]:
    _require(isinstance(expected, Mapping) and set(expected) == set(REQUIRED_SOURCE_PATHS),
             "complete prospective model source binding inventory required")
    actual = source_binding_hashes(root)
    for path, digest in actual.items():
        _require(expected[path] == digest, f"archived model source hash differs: {path}")
    _require(actual[REQUIRED_SOURCE_PATHS[0]] == RUNTIME_BINDING_SHA256,
             "authoritative runtime binding hash differs")
    return actual


def _verify_contract(contract: Any) -> None:
    _require(contract.runtime_binding_sha256 == RUNTIME_BINDING_SHA256,
             "preloaded dynamic contract runtime binding differs")
    _require(len(contract.modes) == 12, "all twelve original execution modes required")
    for mode_id in range(12):
        mode = contract.mode(mode_id)
        _require(mode.perception_checkpoint.sha256 == CHECKPOINT_SHA256["perception"],
                 "frozen perception checkpoint differs")
        _require(mode.ranker_checkpoint.sha256 == CHECKPOINT_SHA256["ranker"],
                 "frozen ranker checkpoint differs")
        if mode.family == "noAE":
            _require(mode.ae_checkpoint is None, "noAE mode has an AE checkpoint")
        else:
            _require(mode.family in AE_FAMILIES and mode.ae_checkpoint is not None
                     and mode.ae_checkpoint.sha256 == CHECKPOINT_SHA256[mode.family],
                     "frozen AE checkpoint differs")


def _frozen(module: Any, role: str) -> None:
    import torch

    _require(isinstance(module, torch.nn.Module), f"{role} must expose its frozen module")
    _require(all(not child.training for child in module.modules()), f"{role} is not in eval mode")
    _require(all(not item.requires_grad for item in module.parameters()),
             f"{role} has trainable parameters")


class ReplayModelAdapter:
    """Callbacks compatible with the sealed v1 same-input ``replay_pair``.

    ``from_preloaded`` only borrows resident objects. Callers must have exclusive
    use of them while replaying; do not attach this adapter to active live work.
    Dense C2 bypasses selected coding entirely, while both branches use the very
    same optimized contextual tail and its original record serializer.
    """

    @classmethod
    def from_preloaded(cls, *, ue_runtime: Any, edge: Any, contract: Any,
                       device: Any, expected_binding_hashes: Mapping[str, str],
                       repository_root: Path | None = None) -> "ReplayModelAdapter":
        import torch
        from rl_agent.splitfusion_live_parity_repair_v1 import perception

        _require(device is not None and str(device).strip(), "explicit replay device required")
        device = torch.device(device)
        sources = _verify_sources(expected_binding_hashes, repository_root)
        _verify_contract(contract)
        _require(ue_runtime.device == device and edge.device == device
                 and edge.runtime._tail_device == device and edge.tail.device == device,
                 "explicit replay device differs from resident preloads")
        _require(set(ue_runtime._ae_encoders) == set(AE_FAMILIES)
                 and set(edge.runtime._ae_decoders) == set(AE_FAMILIES),
                 "exactly three resident AE encoders and decoders required")
        _require(edge.perception_binding["checkpoint_sha256"] == CHECKPOINT_SHA256["perception"],
                 "resident edge perception binding differs")
        for role, module in (("front", ue_runtime._front._model),
                             ("ranker", ue_runtime._ranker._model), ("tail", edge.model)):
            _frozen(module, role)
        for family in AE_FAMILIES:
            mode = next(item for item in contract.modes if item.family == family)
            for role, module in (("encoder", ue_runtime._ae_encoders[family]),
                                 ("decoder", edge.runtime._ae_decoders[family])):
                _frozen(module._model, f"{family} {role}")
                _require((module.family_id, module.bottleneck, module.routing_tag)
                         == (mode.family_id, mode.transported_channels, mode.routing_tag),
                         f"resident {family} {role} identity differs")
        for camera_hash, mount_hash in edge.camera_registry.identities:
            calibration = edge.camera_registry.resolve(camera_hash, mount_hash)
            perception.require_edge_intrinsic_parity(calibration.intrinsic_values)
        _require(bool(edge.camera_registry.identities), "audited camera registry required")
        result = cls()
        result.device = device
        result.contract = contract
        result._ue = ue_runtime
        result._edge = edge
        result.binding_hashes = MappingProxyType(sources)
        result._last_envelope = None
        return result

    def replay_callbacks(self) -> dict[str, Callable]:
        return {"front": self.front, "encode_selected": self.encode_selected,
                "decode_selected": self.decode_selected, "tail": self.tail}

    def _profile(self, envelope: Any) -> Any:
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import continuous_execution_v2 as X
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP

        profile = self.contract.resolve_q_e4(envelope.mode_id, envelope.q_e4)
        _require(envelope.execution_bundle_sha256 == profile.execution_bundle_sha256,
                 "execution-bundle digest mismatch")
        _require(envelope.keep_count == profile.keep_count, "keep_count mismatch")
        _require(envelope.anchor_action_id == profile.action_id,
                 "anchor identity mismatch; off-anchor identities cannot be fabricated")
        _require(hashlib.sha256(envelope.inner_payload).hexdigest() == envelope.inner_payload_sha256,
                 "inner payload digest mismatch")
        X.executed_identity_from_profile(profile, self.contract)
        MP.validate_run4_identity(MP.run4_identity(envelope), contract=self.contract)
        return profile

    def front(self, prepared_input: Any) -> Any:
        import torch

        self._last_envelope = None
        _require(isinstance(prepared_input, torch.Tensor)
                 and prepared_input.device.type == "cpu" and prepared_input.dtype == torch.float32
                 and tuple(prepared_input.shape) == (1, 7, 448, 768),
                 "saved prepared input must be CPU FP32 [1,7,448,768]")
        _require(bool(torch.isfinite(prepared_input).all())
                 and not bool(torch.count_nonzero(prepared_input[:, :, 432:, :])),
                 "saved prepared input has nonfinite content or nonzero bottom padding")
        with torch.inference_mode():
            return self._ue._front(prepared_input.to(device=self.device))

    def encode_selected(self, c2: Any, envelope: Any) -> bytes:
        import torch
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import continuous_execution_v2 as X
        from rl_agent.splitfusion_live_dispatch_v1.timing import StageRecorder, UE_STAGES

        profile = self._profile(envelope)
        ranker = None if profile.ranker_bypassed else self._ue._ranker
        encoder = None if profile.family == "noAE" else self._ue._ae_encoders[profile.family]
        with torch.inference_mode():
            payload = self._ue._codec.encode(X.codec_view(profile), c2.to(device=self.device),
                ranker=ranker, ae_encoder=encoder, timing=StageRecorder(UE_STAGES))
        _require(isinstance(payload, bytes) and bool(payload), "selected codec returned no bytes")
        return payload

    def decode_selected(self, payload: bytes, envelope: Any) -> Any:
        import torch
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import continuous_execution_v2 as X
        from rl_agent.splitfusion_live_dispatch_v1.timing import StageRecorder, EDGE_STAGES
        from rl_agent.splitfusion_live_dispatch_v1.transport import require_inner_agreement

        self._last_envelope = None
        profile = self._profile(envelope)
        _require(payload == envelope.inner_payload, "decoder payload differs from saved selected bytes")
        codec = self._edge.runtime._codec
        timing = StageRecorder(EDGE_STAGES)
        inspected = codec.inspect(payload, timing=timing)
        require_inner_agreement(X.codec_view(profile), inspected.identity)
        decoder = None if profile.family == "noAE" else self._edge.runtime._ae_decoders[profile.family]
        with torch.inference_mode():
            decoded = codec.decode(inspected, decoder=decoder, tail_device=self.device, timing=timing)
        _require(bool(decoded.finite), "selected decoder produced nonfinite C2")
        _require(decoded.device == self.device and decoded.c2.device == self.device,
                 "selected decoded C2 device differs")
        self._last_envelope = envelope
        return decoded.c2

    def tail(self, c2: Any, metadata: Any) -> Any:
        import torch
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
        from rl_agent.splitfusion_live_dispatch_v1.frame_context import validate_frame_context
        from rl_agent.splitfusion_paired_quality_replay_v1.replay import Prediction

        _require(self._last_envelope is not None, "selected decode must precede both tail branches")
        envelope = self._last_envelope
        _require(dict(metadata.run4_identity) == MP.run4_identity(envelope),
                 "tail action metadata differs from selected envelope")
        context = metadata.frame_context
        validate_frame_context(context)
        _require(metadata.protocol_version == 4 and context.frame_id == envelope.frame_id
                 and metadata.sequence_id == context.sequence_id == envelope.tensor_seq
                 and metadata.capture_timestamp_ns == context.capture_timestamp_ns
                 == envelope.capture_timestamp_ns, "tail frame context differs")
        self._edge.camera_registry.resolve(context.camera_model_sha256, context.camera_mount_sha256)
        with torch.inference_mode():
            # The original unguarded tail is deliberate: offline replay has no
            # wall-clock deadline. The scientific compute/serialization is exact.
            work = self._edge.tail.compute_product(c2.to(device=self.device), metadata)
            serialized = self._edge.tail.serialize_product(work)
            snapshot = self._edge.tail.snapshot(serialized)
            labels = snapshot.semantic_labels.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy()
        return Prediction(labels, tuple(snapshot.records or ()))


def load_adapter(*, device: Any, expected_binding_hashes: Mapping[str, str],
                 allow_cold_load: bool = False, repository_root: Path | None = None,
                 ue_loader: Callable | None = None, edge_loader: Callable | None = None,
                 contract_loader: Callable | None = None) -> ReplayModelAdapter:
    """Load only by explicit opt-in; loader injection supports CPU-only tests.

    Real loaders are the same authoritative UE and V3 edge preloads used live.
    No services, warmup, sockets, or live runtime scopes are started here. V3's
    own resident component worker is created only by the opted-in edge preload.
    """
    _require(allow_cold_load is True, "cold model loading requires allow_cold_load=True")
    _require(device is not None and str(device).strip(), "explicit replay device required")
    import torch

    selected_device = torch.device(device)
    _verify_sources(expected_binding_hashes, repository_root)
    if edge_loader is None:
        _require(selected_device.type == "cuda", "actual optimized V3 tail requires an explicit CUDA device")
    if contract_loader is None:
        from rl_agent.splitfusion_live_dispatch_v1.dynamic_execution_contract import load_dynamic_execution_contract
        contract_loader = load_dynamic_execution_contract
    contract = contract_loader()
    _verify_contract(contract)
    if ue_loader is None:
        from rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime import _preload_ue
        ue_loader = _preload_ue
    if edge_loader is None:
        from rl_agent.splitfusion_edge_optimization_v1.detached_edge_preload_v3 import preload_detached_optimized_edge_v3
        edge_loader = preload_detached_optimized_edge_v3
    ue, _ledger, _models = ue_loader(selected_device)
    edge = edge_loader(selected_device)
    return ReplayModelAdapter.from_preloaded(ue_runtime=ue, edge=edge, contract=contract,
        device=selected_device, expected_binding_hashes=expected_binding_hashes,
        repository_root=repository_root)
