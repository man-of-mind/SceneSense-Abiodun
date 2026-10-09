"""Pure adapter: cooperative-demo ``scenesense.multi_ue_live_view_update.v1``.

This is the *existing* demonstration path, normalized so it can sit beside the
production direct path in one viewer without either pretending to be the other.

What this transport is, stated plainly:

* an HTTP JSON POST to a localhost Flask endpoint;
* carrying a nested ``splitfusion_edge_result.v2`` /
  ``splitfusion_object_map_update.v1``;
* with a capture clock in the **CARLA simulation** domain.

What it is not: a radio. Its POST duration is ``demo-control delivery time`` and
its model execution is a ``localhost model path``. Neither is 5G latency, and
neither may be relabelled as one. The capture clock cannot be subtracted from
the map server's wall clock, so wall AoI for this transport is ``N/A``.

Unlike the production document, this envelope *does* carry a measured ego pose
and horizontal FoV, which is what makes the 40 m ego-follow view and the FoV
wedges possible today.
"""

from __future__ import annotations

from typing import Any, Mapping

from .identity_config import BINDING_SOURCE, CoopIdentityConfig
from .normalized import (
    CLOCK_CARLA_SIM_NS,
    POSE_AVAILABLE,
    TRANSPORT_HTTP_DEMO,
    ClockStamp,
    EdgeTimingProvenance,
    MapTerminal,
    NormalizedCoopBatch,
    NormalizedPose,
    NormalizedPose as _NormalizedPose,
    _require,
    assert_finite_records,
    freeze_records,
    nonempty_str,
    sha256_of,
    strict_finite,
    strict_int,
)


LIVE_VIEW_UPDATE_SCHEMA = "scenesense.multi_ue_live_view_update.v1"
EDGE_RESULT_SCHEMA = "splitfusion_edge_result.v2"
OBJECT_MAP_UPDATE_SCHEMA = "splitfusion_object_map_update.v1"

POSE_SOURCE = "HTTP_DEMO_ENVELOPE_MEASURED_EGO_POSE"

# The demonstration envelope declares its own capture clock domain and only this
# value is accepted. A document claiming a wall clock would be a different
# contract and must not be silently absorbed.
REQUIRED_CLOCK_DOMAIN = CLOCK_CARLA_SIM_NS


def from_live_view_update(
    envelope: Mapping[str, Any],
    *,
    identity: CoopIdentityConfig,
    received_timestamp_ns: int | None = None,
    received_clock_domain: str = "unix_wall_ns",
) -> NormalizedCoopBatch:
    """Normalize one cooperative-demo HTTP live-view update."""

    _require(isinstance(envelope, Mapping), "live-view update must be an object")
    _require(
        envelope.get("schema") == LIVE_VIEW_UPDATE_SCHEMA,
        f"live-view update schema drift: {envelope.get('schema')!r}",
    )
    _require(
        envelope.get("clock_domain") == REQUIRED_CLOCK_DOMAIN,
        "live-view update must declare the carla_simulation_ns capture clock",
    )
    envelope_ue_id = nonempty_str(envelope.get("ue_id"), "ue_id")
    envelope_session_id = nonempty_str(envelope.get("session_id"), "session_id")

    payload = envelope.get("payload")
    _require(isinstance(payload, Mapping), "live-view update lacks a payload")
    _require(
        payload.get("schema") == EDGE_RESULT_SCHEMA,
        f"live-view payload schema drift: {payload.get('schema')!r}",
    )
    update = payload.get("object_map_update")
    _require(isinstance(update, Mapping), "edge result lacks object_map_update")
    _require(
        update.get("schema") == OBJECT_MAP_UPDATE_SCHEMA,
        f"object update schema drift: {update.get('schema')!r}",
    )

    stream_id = nonempty_str(payload.get("stream_id"), "stream_id")
    frame_id = strict_int(payload.get("frame_id"), "frame_id", minimum=0)
    capture_ns = strict_int(
        payload.get("capture_timestamp_ns"), "capture_timestamp_ns", minimum=0
    )
    action_id = strict_int(payload.get("action_id"), "action_id", minimum=0)

    # The nested edge/update identities must agree with the envelope.
    _require(update.get("stream_id") == stream_id, "edge/update stream identity mismatch")
    _require(
        strict_int(update.get("frame_id"), "update.frame_id") == frame_id,
        "edge/update frame identity mismatch",
    )
    _require(
        strict_int(update.get("capture_timestamp_ns"), "update.capture_timestamp_ns")
        == capture_ns,
        "edge/update capture identity mismatch",
    )
    _require(
        strict_int(update.get("action_id"), "update.action_id") == action_id,
        "edge/update action identity mismatch",
    )

    records = update.get("records")
    _require(isinstance(records, list), "object-update records must be a list")
    scalar_count = assert_finite_records(records)
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping), f"records[{index}] must be an object")
        if "stream_id" in record:
            _require(
                str(record["stream_id"]) == stream_id,
                f"records[{index}] stream identity mismatch",
            )
        if "frame_id" in record:
            _require(
                strict_int(record["frame_id"], f"records[{index}].frame_id") == frame_id,
                f"records[{index}] frame identity mismatch",
            )

    # The configured binding is authoritative; the envelope's self-declared
    # identity must agree with it rather than replace it.
    identity_row = identity.resolve(run_id=None, cell_id=None, stream_id=stream_id)
    _require(
        identity_row.ue_id == envelope_ue_id,
        f"live-view stream {stream_id!r} is configured to UE "
        f"{identity_row.ue_id!r} but the envelope claims {envelope_ue_id!r}",
    )
    _require(
        identity_row.session_id == envelope_session_id,
        "live-view session identity disagrees with the configured binding",
    )

    pose = _pose_from_envelope(
        envelope,
        stream_id=stream_id,
        frame_id=frame_id,
        capture_ns=capture_ns,
    )

    receipt = None
    if received_timestamp_ns is not None:
        receipt = ClockStamp(
            strict_int(received_timestamp_ns, "received_timestamp_ns", minimum=0),
            nonempty_str(received_clock_domain, "received_clock_domain"),
        )

    return NormalizedCoopBatch(
        transport=TRANSPORT_HTTP_DEMO,
        source_schema=LIVE_VIEW_UPDATE_SCHEMA,
        # The demo envelope is unversioned; it has no protocol_version field and
        # one is not invented for it.
        protocol_version=None,
        # The demo transport has no run or cell concept. Left absent, not "".
        run_id=None,
        cell_id=None,
        ue_id=identity_row.ue_id,
        session_id=identity_row.session_id,
        stream_id=stream_id,
        frame_id=frame_id,
        sequence_id=None,
        action_id=action_id,
        profile_id=(str(payload["profile_id"]) if payload.get("profile_id") else None),
        decoder_identity=None,
        capture=ClockStamp(capture_ns, CLOCK_CARLA_SIM_NS),
        receipt=receipt,
        records=freeze_records(records),
        record_scalar_count=scalar_count,
        records_sha256=sha256_of(list(records), "records"),
        payload_sha256=sha256_of(dict(envelope), "live-view update"),
        identity_binding_source=BINDING_SOURCE,
        # The demo path carries no edge stage stamps at all. Absent, not zero.
        edge_timing=EdgeTimingProvenance.absent("ABSENT_HTTP_DEMO_HAS_NO_EDGE_TIMING"),
        # The demo map returns an HTTP 202 acceptance, not a production terminal
        # outcome with agent credit. Reporting one here would be a fabrication.
        map_terminal=MapTerminal.absent("ABSENT_HTTP_DEMO_HAS_NO_TERMINAL_CONTRACT"),
        pose=pose,
        pose_state=POSE_AVAILABLE,
    )


def _pose_from_envelope(
    envelope: Mapping[str, Any], *, stream_id: str, frame_id: int, capture_ns: int
) -> NormalizedPose:
    ego_pose = envelope.get("ego_pose")
    _require(isinstance(ego_pose, Mapping), "live-view update lacks an ego pose")
    location = ego_pose.get("location")
    rotation = ego_pose.get("rotation")
    _require(isinstance(location, Mapping), "ego pose location is malformed")
    _require(isinstance(rotation, Mapping), "ego pose rotation is malformed")
    fov = strict_finite(envelope.get("horizontal_fov_deg"), "horizontal_fov_deg")
    _require(0.0 < fov <= 360.0, "horizontal_fov_deg must lie in (0, 360]")
    return _NormalizedPose(
        x=strict_finite(location.get("x"), "ego_pose.location.x"),
        y=strict_finite(location.get("y"), "ego_pose.location.y"),
        z=strict_finite(location.get("z"), "ego_pose.location.z"),
        pitch=strict_finite(rotation.get("pitch"), "ego_pose.rotation.pitch"),
        yaw=strict_finite(rotation.get("yaw"), "ego_pose.rotation.yaw"),
        roll=strict_finite(rotation.get("roll"), "ego_pose.rotation.roll"),
        horizontal_fov_deg=fov,
        source=POSE_SOURCE,
        # run/cell are absent for this transport; "" marks that explicitly in the
        # binding tuple rather than pretending a run identity exists.
        binding=("", "", stream_id, frame_id, capture_ns),
    )
