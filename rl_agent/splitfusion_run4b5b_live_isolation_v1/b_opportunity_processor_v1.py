"""Exact one-frame Run-4B-Joint/Run-5B UE opportunity processor.

This is the narrow production composition seam.  It consumes one Route-B
prepared opportunity, selects causal UE telemetry at the action-open cutoff,
builds the registered B state, executes the hash-verified final actor at
batch one, resolves the continuous action, runs the proven front/codec, packs
SFD4 and sends the existing UDP fragments.  It never reads or computes GT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from phase2_map_sharing.transport import chunk_payload
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4LIVE
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import contract as R4B
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as SNR
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import run5b_state_contract as R5B
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
    continuous_execution_v2 as X,
    phase6_ue_runtime_v2 as P6,
    run4_live_wire_v2 as W,
    ue_telemetry_provider_v2 as T,
)
from rl_agent.splitfusion_live_dispatch_v1 import dynamic_execution_contract as DEC
from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1

from . import b_route_bridge_v4 as B
from . import b_ue_process_v1 as U
from . import final_actor_gate_v2 as GATE
from . import live_adapters_v1 as L
from . import operational_ack_v1 as A


class BOpportunityProcessorError(RuntimeError): pass


def _require(value: bool, message: str) -> None:
    if not value:
        raise BOpportunityProcessorError(message)


class EffectiveSnrProviderV1(Protocol):
    """Return the latest ACKed active target available at ``cutoff_raw_ns``."""
    def observe_db(self, *, cutoff_raw_ns: int, session_uuid: str,
                   decision_seq: int) -> float: ...


class DatagramSenderV1(Protocol):
    def sendto(self, packet: bytes, remote: tuple[str, int]) -> Any: ...


@dataclass(frozen=True, slots=True)
class LoadedActorBindingV1:
    loaded: GATE.LoadedFinalActorV2
    manifest_path: Path
    weights_path: Path


def load_final_actor_v1(*, variant: L.ActorVariant,
                        evidence_root: Path) -> LoadedActorBindingV1:
    expected = (GATE.RUN4B_VARIANT if variant is L.ActorVariant.RUN4B
                else GATE.RUN5B_VARIANT)
    manifest = GATE.bundled_manifest_path(expected)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    relative = Path(raw["source_actor_relative_path"])
    _require(not relative.is_absolute() and ".." not in relative.parts,
             "final actor path escapes evidence root")
    weights = Path(evidence_root) / relative
    loaded = GATE.verify_and_load_final_actor(
        manifest, weights, evidence_root=Path(evidence_root))
    _require(loaded.identity.variant == expected, "final actor variant differs")
    return LoadedActorBindingV1(loaded, manifest, weights)


def _prior(previous: Optional[A.OperationalOutcomeV1]) -> R4B.OperationalPriorV1:
    if previous is None:
        return R4B.OperationalPriorV1.genesis()
    _require(type(previous) is A.OperationalOutcomeV1,
             "previous outcome is foreign")
    return R4B.OperationalPriorV1.from_outcome(
        mode_id=previous.identity.mode_id, q_e4=previous.identity.q_e4,
        timely=previous.success,
        operational_latency_ns=(previous.state_latency_ns
                                if previous.success else None))


def _actor_action(module: Any, features: tuple[float, ...]) -> tuple[int, int]:
    import torch
    state = torch.tensor(features, dtype=torch.float32).reshape(1, -1)
    before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.is_initialized()
    decision = module.deterministic_execution(state)
    _require(torch.equal(before, torch.random.get_rng_state()),
             "deterministic actor changed global RNG")
    _require(torch.cuda.is_initialized() == cuda_before,
             "CPU actor initialized CUDA")
    mode = int(decision.mode_index.item())
    q_e4 = int(decision.q_e4.item())
    _require(0 <= mode < 12 and 0 <= q_e4 <= 9800,
             "actor returned an invalid action")
    return mode, q_e4


class BOpportunityProcessorV1:
    def __init__(self, *, request: U.BUEProcessRequestV1,
                 actor: LoadedActorBindingV1,
                 telemetry: T.UeTelemetryProviderV2,
                 dynamic_contract: DEC.DynamicExecutionContract,
                 continuous_ue: X.ContinuousUERuntimeV2,
                 sender: DatagramSenderV1, remote: tuple[str, int],
                 input_builder: Callable[[Any, Any], Any],
                 cell_id: str, chunk_bytes: int,
                 snr_provider: Optional[EffectiveSnrProviderV1] = None) -> None:
        _require(type(actor) is LoadedActorBindingV1, "actor binding is foreign")
        expected_variant = (GATE.RUN4B_VARIANT
                            if request.variant is L.ActorVariant.RUN4B
                            else GATE.RUN5B_VARIANT)
        _require(actor.loaded.identity.variant == expected_variant,
                 "actor/request variant mismatch")
        _require(actor.loaded.identity.feature_schema_sha256
                 == request.feature_schema_sha256,
                 "actor/request feature schema mismatch")
        _require(type(cell_id) is str and bool(cell_id), "cell id is empty")
        _require(type(chunk_bytes) is int and chunk_bytes > 0,
                 "chunk size is invalid")
        _require(type(remote) is tuple and len(remote) == 2
                 and type(remote[0]) is str and type(remote[1]) is int,
                 "remote endpoint is invalid")
        if request.variant is L.ActorVariant.RUN5B:
            _require(snr_provider is not None,
                     "Run-5B requires the causal effective-SNR provider")
        else:
            _require(snr_provider is None,
                     "Run-4B must not receive an SNR provider")
        self.request, self.actor, self.telemetry = request, actor, telemetry
        self.contract, self.continuous = dynamic_contract, continuous_ue
        self.sender, self.remote, self.input_builder = sender, remote, input_builder
        self.cell_id, self.chunk_bytes, self.snr_provider = (
            cell_id, chunk_bytes, snr_provider)
        self.scaling = R4B.ScalingV1(
            camera_si_center=117.08553307797729,
            camera_si_scale=13.336440703846854,
            backlog_log1p_scale=math.log1p(50_000_000))

    def _features(self, opportunity: B.RouteOpportunityV4,
                  previous: Optional[A.OperationalOutcomeV1]) -> tuple[float, ...]:
        kwargs = opportunity.submit_kwargs
        scene = P6.compute_scene_descriptors(
            kwargs["frame_bgr"], kwargs["b_window_meta"],
            source_raw_ns=opportunity.action_open_monotonic_raw_ns)
        _require(scene.camera_si is not None and scene.radar_p40 is not None,
                 f"scene descriptors unavailable: {scene.camera_status}/"
                 f"{scene.radar_status}")
        ue_label = self.telemetry.ue_label
        _require(ue_label is not None, "UE telemetry is not bound")
        decision_identity = R4LIVE.DecisionIdentityV1(
            self.telemetry.session_uuid, ue_label, opportunity.sequence)
        cutoff = opportunity.action_open_monotonic_raw_ns
        _require(cutoff > 0, "action-open cutoff is invalid")
        boundary = R4LIVE.DecisionBoundaryV1(
            identity=decision_identity, state_commit_timestamp_ns=cutoff - 1,
            action_open_timestamp_ns=cutoff, clock_domain=T.CLOCK_DOMAIN)
        evidence = T.assemble_radio_evidence(
            self.telemetry.snapshot(), identity=decision_identity,
            boundary=boundary, payload_enqueue_timestamp_ns=cutoff + 1)
        T.require_admitted(evidence)
        observation = R4B.ObservationV1(
            camera_si=float(scene.camera_si), radar_p40=float(scene.radar_p40),
            prior_ul_mcs=int(evidence.prior_ul_mcs.observation.value),
            pre_action_rlc_backlog_bytes=int(
                evidence.pre_action_rlc_backlog.value))
        prior = _prior(previous)
        if self.request.variant is L.ActorVariant.RUN4B:
            features = R4B.build_features(observation, prior, self.scaling)
        else:
            assert self.snr_provider is not None
            raw_db = self.snr_provider.observe_db(
                cutoff_raw_ns=cutoff,
                session_uuid=self.telemetry.session_uuid,
                decision_seq=opportunity.sequence)
            _require(type(raw_db) is float and math.isfinite(raw_db),
                     "effective SNR provider returned a non-finite value")
            scaled = float(SNR.scale_snr_db(raw_db))
            features = R5B.build_features(
                observation, prior, self.scaling, scaled)
        _require(tuple(self.actor.loaded.identity.feature_order)
                 == (R4B.FEATURE_ORDER if self.request.variant is L.ActorVariant.RUN4B
                     else R5B.FEATURE_ORDER),
                 "actor feature order differs from exact state builder")
        return tuple(features)

    def __call__(self, opportunity: B.RouteOpportunityV4,
                 previous: Optional[A.OperationalOutcomeV1]
                 ) -> U.BTransmissionV1:
        features = self._features(opportunity, previous)
        mode_id, q_e4 = _actor_action(self.actor.loaded.module, features)
        profile = self.contract.resolve_q_e4(mode_id, q_e4)
        kwargs = opportunity.submit_kwargs
        frame = X.FrameIdentityV2(
            session_uuid=self.telemetry.session_uuid,
            controller_lineage_sha256=self.request.actor_boundary_sha256,
            decision_seq=opportunity.sequence, ticket_seq=opportunity.sequence,
            frame_id=opportunity.frame_id, tensor_seq=opportunity.sequence,
            capture_timestamp_ns=opportunity.capture_timestamp_ns,
            reward_requested=True)
        input_7ch = self.input_builder(kwargs["frame_bgr"], kwargs["radar_tensor"])
        prepared = self.continuous.prepare(profile, input_7ch, frame)
        pose: Sequence[float] = kwargs["ego_pose"]
        _require(len(pose) == 6, "ego pose must contain six values")
        context = build_frame_context_v1(
            stream_id=str(kwargs["stream_id"]), frame_id=opportunity.frame_id,
            sequence_id=opportunity.sequence,
            capture_timestamp_ns=opportunity.capture_timestamp_ns,
            ego_world_x=pose[0], ego_world_y=pose[1], ego_world_z=pose[2],
            ego_world_pitch=pose[3], ego_world_yaw=pose[4],
            ego_world_roll=pose[5])
        wire = W.pack_sfd4(prepared.envelope, context)
        chunks = chunk_payload(wire, message_id=opportunity.frame_id,
                               chunk_bytes=self.chunk_bytes)
        _require(bool(chunks), "SFD4 produced no datagrams")
        for packet in chunks:
            self.sender.sendto(packet, self.remote)
        identity = L.identity_from_phase6(
            run_id=self.request.run_id, cell_id=self.cell_id,
            envelope=prepared.envelope, context=context, profile=profile,
            require_operational_ack=True)
        return U.BTransmissionV1(
            identity=identity,
            action_open_monotonic_raw_ns=
                opportunity.action_open_monotonic_raw_ns,
            payload_bytes=len(wire), decision_frame=True)


def build_live_pipeline_v1(*, request: U.BUEProcessRequestV1,
                           evidence_root: Path,
                           telemetry: T.UeTelemetryProviderV2,
                           dynamic_contract: DEC.DynamicExecutionContract,
                           continuous_ue: X.ContinuousUERuntimeV2,
                           sender: DatagramSenderV1,
                           remote: tuple[str, int],
                           input_builder: Callable[[Any, Any], Any],
                           cell_id: str, chunk_bytes: int,
                           route_kwargs: Mapping[str, Any],
                           raw_spool_root: Path,
                           snr_provider: Optional[EffectiveSnrProviderV1] = None,
                           postrun_materializer: Optional[Any] = None,
                           route_driver: Optional[Callable[[B.BRouteBridgeV4], Any]] = None,
                           ) -> B.BRouteBridgeV4:
    """Verify the selected final actor and return the production B pipeline."""
    actor = load_final_actor_v1(variant=request.variant,
                                evidence_root=Path(evidence_root))
    processor = BOpportunityProcessorV1(
        request=request, actor=actor, telemetry=telemetry,
        dynamic_contract=dynamic_contract, continuous_ue=continuous_ue,
        sender=sender, remote=remote, input_builder=input_builder,
        cell_id=cell_id, chunk_bytes=chunk_bytes, snr_provider=snr_provider)
    driver = route_driver or B.pinned_route_driver(route_kwargs)
    return B.BRouteBridgeV4(
        variant=request.variant,
        feature_schema_sha256=request.feature_schema_sha256,
        actor_boundary_sha256=request.actor_boundary_sha256,
        processor=processor, route_driver=driver,
        raw_spool_root=Path(raw_spool_root),
        postrun_materializer=postrun_materializer,
        transmitted_budget=request.transmitted_budget)


__all__ = [
    "BOpportunityProcessorError", "EffectiveSnrProviderV1",
    "DatagramSenderV1", "LoadedActorBindingV1", "load_final_actor_v1",
    "BOpportunityProcessorV1", "build_live_pipeline_v1",
]
