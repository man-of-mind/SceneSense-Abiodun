"""Thread-safe multi-UE ingress and spatial-map service boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import threading
import time
from typing import Any, Mapping

from .association import (
    AssociationPolicy,
    AssociationResult,
    ConservativeTrackStore,
    MapTrack,
    associate_snapshot,
)
from .buffer import IngestResult, MultiUEFrameBuffer
from .contracts import (
    LEGACY_SPATIAL_PACKET_SCHEMA,
    SPLITFUSION_EDGE_RESULT_SCHEMA,
    MultiUEContractError,
    ObservationBatch,
    from_legacy_spatial_packet,
    from_splitfusion_edge_result,
)


INGRESS_ENVELOPE_SCHEMA = "multi_ue_object_ingress.v1"
MAP_SNAPSHOT_SCHEMA = "multi_ue_spatial_map.v1"
AGENT_BINDING_STATUS = "EXTERNAL_AGENT_POLICY_PENDING"


@dataclass(frozen=True)
class ServiceSnapshot:
    clock_domain: str
    snapshot_timestamp_ns: int
    association: AssociationResult
    tracks: tuple[MapTrack, ...]
    policy: AssociationPolicy

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": MAP_SNAPSHOT_SCHEMA,
            "clock_domain": self.clock_domain,
            "snapshot_timestamp_ns": self.snapshot_timestamp_ns,
            "agent_binding_status": AGENT_BINDING_STATUS,
            "policy": self.policy.as_dict(),
            "input_source_count": self.association.input_source_count,
            "aligned_source_count": self.association.aligned_source_count,
            "accepted_observation_count": len(self.association.accepted),
            "rejected_observations": [asdict(item) for item in self.association.rejected],
            "association_count": len(self.association.associations),
            "associations": [
                {
                    "association_id": item.association_id,
                    "class_name": item.class_name,
                    "source_count": item.source_count,
                    "selected_identity": list(item.selected.identity),
                    "member_identities": [list(member.identity) for member in item.members],
                    "position_combination": "NONE_SELECTED_SOURCE_ONLY",
                }
                for item in self.association.associations
            ],
            "tracks": [asdict(track) for track in self.tracks],
        }


class MultiUESpatialMapService:
    """Validate, buffer, associate, and expose conservative multi-UE tracks."""

    def __init__(
        self,
        policy: AssociationPolicy,
        *,
        max_frames_per_source: int = 32,
    ) -> None:
        self.policy = policy
        self.buffer = MultiUEFrameBuffer(max_frames_per_source=max_frames_per_source)
        self._track_stores: dict[str, ConservativeTrackStore] = {}
        self._latest_capture_by_domain: dict[str, int] = {}
        self._latest_snapshot_by_domain: dict[str, int] = {}
        self._stream_by_ue_session: dict[tuple[str, str], str] = {}
        self._session_by_ue: dict[str, str] = {}
        self._lock = threading.RLock()

    def ingest_batch(self, batch: ObservationBatch) -> IngestResult:
        with self._lock:
            ue_session = batch.ue_id, batch.session_id
            registered_session = self._session_by_ue.get(batch.ue_id)
            if registered_session is not None and registered_session != batch.session_id:
                raise MultiUEContractError(
                    "one UE identity attempted an unregistered session rollover"
                )
            registered_stream = self._stream_by_ue_session.get(ue_session)
            if registered_stream is not None and registered_stream != batch.stream_id:
                raise MultiUEContractError(
                    "one UE/session identity attempted to use multiple stream identities"
                )
            result = self.buffer.ingest(batch)
            if result.disposition in ("ACCEPTED", "DUPLICATE_IDENTICAL"):
                self._session_by_ue[batch.ue_id] = batch.session_id
                self._stream_by_ue_session[ue_session] = batch.stream_id
                self._latest_capture_by_domain[batch.clock_domain] = max(
                    batch.capture_timestamp_ns,
                    self._latest_capture_by_domain.get(batch.clock_domain, 0),
                )
            return result

    def ingest_splitfusion(
        self,
        payload: Mapping[str, Any],
        *,
        ue_id: str,
        session_id: str,
        received_timestamp_ns: int | None = None,
        clock_domain: str = "unix_wall_ns",
    ) -> IngestResult:
        received_ns = time.time_ns() if received_timestamp_ns is None else received_timestamp_ns
        batch = from_splitfusion_edge_result(
            payload,
            ue_id=ue_id,
            session_id=session_id,
            received_timestamp_ns=received_ns,
            clock_domain=clock_domain,
        )
        return self.ingest_batch(batch)

    def ingest_legacy(
        self,
        payload: Mapping[str, Any],
        *,
        ue_id: str,
        session_id: str,
        received_timestamp_ns: int | None = None,
    ) -> IngestResult:
        received_ns = time.time_ns() if received_timestamp_ns is None else received_timestamp_ns
        batch = from_legacy_spatial_packet(
            payload,
            ue_id=ue_id,
            session_id=session_id,
            received_timestamp_ns=received_ns,
        )
        return self.ingest_batch(batch)

    def snapshot(self, *, clock_domain: str, snapshot_timestamp_ns: int) -> ServiceSnapshot:
        with self._lock:
            previous_snapshot_ns = self._latest_snapshot_by_domain.get(clock_domain)
            if (
                previous_snapshot_ns is not None
                and snapshot_timestamp_ns < previous_snapshot_ns
            ):
                raise ValueError(
                    "stateful map snapshots must be nondecreasing within a clock domain"
                )
            raw_snapshot = self.buffer.snapshot(
                clock_domain=clock_domain,
                snapshot_timestamp_ns=snapshot_timestamp_ns,
                # Retain each source's latest causal batch here so the
                # association layer can explicitly account for age rejection.
                maximum_source_age_ns=(1 << 63) - 1,
                alignment_tolerance_ns=self.policy.alignment_tolerance_ns,
            )
            association = associate_snapshot(raw_snapshot, self.policy)
            track_store = self._track_stores.setdefault(
                clock_domain,
                ConservativeTrackStore(self.policy),
            )
            tracks = track_store.update(association)
            self._latest_snapshot_by_domain[clock_domain] = snapshot_timestamp_ns
            return ServiceSnapshot(
                clock_domain=clock_domain,
                snapshot_timestamp_ns=snapshot_timestamp_ns,
                association=association,
                tracks=tracks,
                policy=self.policy,
            )

    def snapshot_latest(self, *, clock_domain: str) -> ServiceSnapshot | None:
        with self._lock:
            timestamp_ns = self._latest_capture_by_domain.get(clock_domain)
            if timestamp_ns is None:
                return None
            return self.snapshot(
                clock_domain=clock_domain,
                snapshot_timestamp_ns=timestamp_ns,
            )


def create_flask_app(service: MultiUESpatialMapService):
    """Create a small HTTP ingress/API without starting a process on import."""

    from flask import Flask, jsonify, request

    app = Flask(__name__)

    @app.post("/api/multi_ue/v1/updates")
    def ingest_update():
        try:
            envelope = request.get_json(force=True)
            if not isinstance(envelope, Mapping):
                raise MultiUEContractError("ingress envelope must be an object")
            if envelope.get("schema") != INGRESS_ENVELOPE_SCHEMA:
                raise MultiUEContractError("unexpected ingress envelope schema")
            ue_id = str(envelope.get("ue_id") or "")
            session_id = str(envelope.get("session_id") or "")
            payload = envelope.get("payload")
            if not isinstance(payload, Mapping):
                raise MultiUEContractError("ingress payload must be an object")
            source_schema = payload.get("schema")
            if source_schema == SPLITFUSION_EDGE_RESULT_SCHEMA:
                result = service.ingest_splitfusion(
                    payload,
                    ue_id=ue_id,
                    session_id=session_id,
                    received_timestamp_ns=time.time_ns(),
                )
            elif source_schema == LEGACY_SPATIAL_PACKET_SCHEMA:
                result = service.ingest_legacy(
                    payload,
                    ue_id=ue_id,
                    session_id=session_id,
                    received_timestamp_ns=time.time_ns(),
                )
            else:
                raise MultiUEContractError("unsupported source payload schema")
            return jsonify(asdict(result)), 202
        except (MultiUEContractError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 422

    @app.get("/api/multi_ue/v1/spatial_map")
    def spatial_map():
        try:
            clock_domain = str(request.args.get("clock_domain") or "unix_wall_ns")
            raw_timestamp = request.args.get("snapshot_timestamp_ns")
            timestamp_ns = time.time_ns() if raw_timestamp is None else int(raw_timestamp)
            return jsonify(
                service.snapshot(
                    clock_domain=clock_domain,
                    snapshot_timestamp_ns=timestamp_ns,
                ).as_dict()
            )
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 422

    @app.get("/api/multi_ue/v1/healthz")
    def healthz():
        return jsonify(
            {
                "status": "ok",
                "source_count": service.buffer.source_count(),
                "agent_binding_status": AGENT_BINDING_STATUS,
            }
        )

    return app
