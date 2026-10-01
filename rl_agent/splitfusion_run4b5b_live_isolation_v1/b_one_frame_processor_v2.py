"""One-frame B opportunity processor with separate controller lineage.

The original B processor predates split-host one-frame orchestration and uses
the actor boundary as the controller lineage.  Those are different
authorities.  This additive processor keeps the actor boundary on the actor
request and requires a separately supplied execution/controller lineage for
every emitted SFD4 identity.  Equality is refused so an accidental
substitution cannot silently recur.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence

from phase2_map_sharing.transport import chunk_payload
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
    continuous_execution_v2 as X,
    run4_live_wire_v2 as W,
)
from rl_agent.splitfusion_live_dispatch_v1.frame_context import (
    build_frame_context_v1,
)

from . import b_opportunity_processor_v1 as P
from . import b_route_bridge_v4 as B
from . import b_ue_process_v1 as U
from . import live_adapters_v1 as L
from . import operational_ack_v1 as A


class ControllerLineageError(P.BOpportunityProcessorError):
    """The execution/controller identity is absent, malformed, or aliased."""


def _lineage(value: Any, actor_boundary_sha256: str) -> str:
    if not (type(value) is str and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value)):
        raise ControllerLineageError(
            "controller lineage is not a lowercase SHA-256")
    if value == actor_boundary_sha256:
        raise ControllerLineageError(
            "controller lineage must not be substituted by actor boundary")
    return value


class OneFrameOpportunityProcessorV2(P.BOpportunityProcessorV1):
    """The v1 mechanics with an authoritative, non-actor lineage."""

    def __init__(self, *, controller_lineage_sha256: str, **kwargs: Any) -> None:
        request = kwargs.get("request")
        if request is None:
            raise ControllerLineageError("request is absent")
        super().__init__(**kwargs)
        self.controller_lineage_sha256 = _lineage(
            controller_lineage_sha256, request.actor_boundary_sha256)

    def __call__(self, opportunity: B.RouteOpportunityV4,
                 previous: Optional[A.OperationalOutcomeV1]
                 ) -> U.BTransmissionV1:
        # Evidence only: raw-clock stage stamps on the action-open clock.
        stamps: dict[str, int] = {}
        opened = int(opportunity.action_open_monotonic_raw_ns)

        def mark(name: str) -> None:
            stamps[name] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - opened

        self.stage_timing = {"action_open_monotonic_raw_ns": opened,
                             "stages_ns_after_action_open": stamps}
        mark("processor_entry")
        features = self._features(opportunity, previous)
        mark("state_features")
        mode_id, q_e4 = P._actor_action(self.actor.loaded.module, features)
        mark("actor")
        profile = self.contract.resolve_q_e4(mode_id, q_e4)
        kwargs = opportunity.submit_kwargs
        frame = X.FrameIdentityV2(
            session_uuid=self.telemetry.session_uuid,
            controller_lineage_sha256=self.controller_lineage_sha256,
            decision_seq=opportunity.sequence,
            ticket_seq=opportunity.sequence,
            frame_id=opportunity.frame_id,
            tensor_seq=opportunity.sequence,
            capture_timestamp_ns=opportunity.capture_timestamp_ns,
            reward_requested=True)
        input_7ch = self.input_builder(
            kwargs["frame_bgr"], kwargs["radar_tensor"])
        mark("input_7ch")
        prepared = self.continuous.prepare(profile, input_7ch, frame)
        mark("front_ae_codec")
        pose: Sequence[float] = kwargs["ego_pose"]
        if len(pose) != 6:
            raise ControllerLineageError("ego pose must contain six values")
        context = build_frame_context_v1(
            stream_id=str(kwargs["stream_id"]),
            frame_id=opportunity.frame_id,
            sequence_id=opportunity.sequence,
            capture_timestamp_ns=opportunity.capture_timestamp_ns,
            ego_world_x=pose[0], ego_world_y=pose[1], ego_world_z=pose[2],
            ego_world_pitch=pose[3], ego_world_yaw=pose[4],
            ego_world_roll=pose[5])
        wire = W.pack_sfd4(prepared.envelope, context)
        chunks = chunk_payload(
            wire, message_id=opportunity.frame_id,
            chunk_bytes=self.chunk_bytes)
        if not chunks:
            raise ControllerLineageError("SFD4 produced no datagrams")
        mark("pack_chunk")
        for index, packet in enumerate(chunks):
            self.sender.sendto(packet, self.remote)
            if index == 0:
                mark("first_send")
        mark("last_send")
        self.stage_timing["datagrams"] = len(chunks)
        identity = L.identity_from_phase6(
            run_id=self.request.run_id, cell_id=self.cell_id,
            envelope=prepared.envelope, context=context, profile=profile,
            require_operational_ack=True)
        if identity.controller_lineage_sha256 != self.controller_lineage_sha256:
            raise ControllerLineageError("emitted identity lineage differs")
        return U.BTransmissionV1(
            identity=identity,
            action_open_monotonic_raw_ns=
                opportunity.action_open_monotonic_raw_ns,
            payload_bytes=len(wire), decision_frame=True)


__all__ = [
    "ControllerLineageError", "OneFrameOpportunityProcessorV2",
]
