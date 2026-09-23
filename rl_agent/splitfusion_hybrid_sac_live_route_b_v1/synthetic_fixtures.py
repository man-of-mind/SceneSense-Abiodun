"""Explicitly synthetic observation fixtures for CPU-only Phase-1 tests.

``SYNTHETIC_PHASE1_FIXTURE_NOT_MEASURED_EVIDENCE``

Nothing in this module measured anything.  It exists so the Phase-1 tests can
exercise the state builder's *interface* before Phase 2 supplies a real
telemetry provider, and every record it produces is stamped with a source id
that says so.  It is imported by tests only; no runtime path imports it, and
the state builder will happily reject its output if a field is wrong, because
it applies the same validation it will apply to live evidence.

The radio records use ``RadioObservationV1.for_simulator_testbed`` -- the only
attested factory the state contract currently exposes -- exactly as the Run-3
training environment does.  That is *not* a claim that a live UE can observe
these values; see the Phase-1 report's blocker on
``RadioEvidencePath.UE_VISIBLE_RUNTIME``.
"""

from __future__ import annotations

import uuid
from typing import Optional, Tuple

from rl_agent.splitfusion_hybrid_sac_v1.empirical_radio_context import (
    MCS_TABLE_ID,
)
from rl_agent.splitfusion_hybrid_sac_v1.reward_ticket_controller import (
    RewardTicketController,
)
from rl_agent.splitfusion_hybrid_sac_v1.scene_descriptors import (
    SceneDescriptorSample,
)
from rl_agent.splitfusion_hybrid_sac_v1.state_reward_transition_contract import (
    BsrReportType,
    BsrReportV1,
    BsrScope,
    BsrSource,
    EpisodeStartProofV1,
    RadioEventProvenanceV1,
    RadioObservationV1,
    RadioSourceWall,
    SceneObservationV1,
)

from . import pilot_contract as contract

__all__ = [
    "FIXTURE_LABEL",
    "SYNTHETIC_SOURCE_SHA256",
    "synthetic_episode_start",
    "synthetic_radio",
    "synthetic_scene",
    "synthetic_session_uuid",
]

FIXTURE_LABEL = "SYNTHETIC_PHASE1_FIXTURE_NOT_MEASURED_EVIDENCE"

#: A constant, obviously synthetic digest.  Real evidence carries the digest of
#: the bytes it was actually read from.
SYNTHETIC_SOURCE_SHA256 = contract.canonical_sha256(
    {"fixture": FIXTURE_LABEL, "record": "synthetic_phase1_source_v1"}
)


def synthetic_session_uuid(tag: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{FIXTURE_LABEL}:{tag}"))


def synthetic_scene(
    *,
    camera_si: float,
    radar_p40: float,
    measured_ns: int,
    carla_frame_id: int,
) -> SceneObservationV1:
    return SceneObservationV1(
        sample=SceneDescriptorSample(camera_si=camera_si, radar_p40=radar_p40),
        measured_ns=measured_ns,
        carla_frame_id=carla_frame_id,
        source_id=FIXTURE_LABEL,
        source_sha256=SYNTHETIC_SOURCE_SHA256,
    )


def _event(
    *, tag: str, index: int, timestamp_ns: int, session_uuid: str
) -> RadioEventProvenanceV1:
    return RadioEventProvenanceV1(
        source_wall=RadioSourceWall.SIMULATOR_TESTBED,
        source_event_id=f"{FIXTURE_LABEL}:{tag}:{index}",
        source_event_index=index,
        source_event_timestamp_ns=timestamp_ns,
        collector_ingest_wall_time_ns=timestamp_ns,
        collector_ingest_monotonic_ns=timestamp_ns,
        ran_epoch_id=f"{FIXTURE_LABEL}-epoch-0",
        control_session_id=session_uuid,
        raw_event_sha256=SYNTHETIC_SOURCE_SHA256,
    )


def synthetic_radio(
    *,
    achieved_snr_db: float,
    mcs_index: int,
    bsr_bytes: int,
    measured_ns: int,
    session_uuid: str,
    index: int = 0,
    snr_measured_ns: Optional[int] = None,
    mcs_measured_ns: Optional[int] = None,
    bsr_measured_ns: Optional[int] = None,
) -> RadioObservationV1:
    """One attested simulator/testbed radio record with per-source timestamps."""
    snr_ns = measured_ns if snr_measured_ns is None else snr_measured_ns
    mcs_ns = measured_ns if mcs_measured_ns is None else mcs_measured_ns
    bsr_ns = measured_ns if bsr_measured_ns is None else bsr_measured_ns
    lcg: Tuple[Optional[int], ...] = (bsr_bytes,) + (0,) * 7
    report = BsrReportV1(
        lcg_bytes=lcg,
        valid_mask=(True,) * 8,
        missing_reasons=(None,) * 8,
        scope=BsrScope.ALL_GROUPS_LATEST,
        logical_channel_group=0,
        report_type=BsrReportType.SIMULATOR_VECTOR,
        source=BsrSource.SIMULATOR_TESTBED_PRIVILEGED,
        measured_ns=bsr_ns,
        event=_event(
            tag="bsr", index=index, timestamp_ns=bsr_ns, session_uuid=session_uuid
        ),
    )
    return RadioObservationV1.for_simulator_testbed(
        achieved_snr_db=achieved_snr_db,
        snr_measured_ns=snr_ns,
        mcs_index=mcs_index,
        mcs_table_id=MCS_TABLE_ID,
        mcs_measured_ns=mcs_ns,
        bsr_bytes=bsr_bytes,
        bsr_scope=BsrScope.ALL_GROUPS_LATEST,
        bsr_logical_channel_group=0,
        bsr_measured_ns=bsr_ns,
        snr_event=_event(
            tag="snr", index=index, timestamp_ns=snr_ns, session_uuid=session_uuid
        ),
        mcs_event=_event(
            tag="mcs", index=index, timestamp_ns=mcs_ns, session_uuid=session_uuid
        ),
        bsr_report=report,
        source_id=FIXTURE_LABEL,
        source_sha256=SYNTHETIC_SOURCE_SHA256,
    )


def synthetic_episode_start(
    *,
    session_uuid: str,
    lineage_uuid: str,
    tensor_seq: int,
    carla_frame_id: int,
    observed_ns: int,
) -> EpisodeStartProofV1:
    """An episode-start proof issued by a real reward-ticket controller."""
    controller = RewardTicketController(
        session_uuid, controller_lineage_uuid=lineage_uuid
    )
    genesis = controller.authorize_episode_start(
        first_decision_seq=0,
        first_tensor_seq=tensor_seq,
        first_carla_frame_id=carla_frame_id,
        state_observed_ns=observed_ns,
    )
    return EpisodeStartProofV1.from_controller_genesis(
        genesis, source_id=FIXTURE_LABEL, source_sha256=SYNTHETIC_SOURCE_SHA256
    )
