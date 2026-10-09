"""Pure adapter: production ``splitfusion_direct_object_map_update.v1`` document.

This adapter consumes an *already decoded* document. It never opens a socket,
never reassembles chunks and never decompresses a datagram; the transport half
belongs to the production ``DirectMapIngestService`` in the master tree.

Relationship to the frozen production validator
-----------------------------------------------
``rl_agent.splitfusion_direct_edge_map_v1.protocol.validate_object_map_update``
is the authority and is **not** re-implemented here. It cannot be imported in
this branch (the package does not exist at this commit; see ``reference_pins``).
What this adapter does instead is check exactly the fields the *normalized type*
must be able to trust — mandatory production identity, protocol version, record
count agreement, finiteness and internal identity consistency — and it refuses
anything it cannot prove. It is deliberately no more permissive than the pinned
validator on the fields it covers.

A master-based bridge worktree must call the real validator first and then this
adapter, which is why ``from_direct_object_map_update`` accepts an optional
``upstream_validator`` hook: production validation stays in production code.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .identity_config import BINDING_SOURCE, CoopIdentityConfig
from .normalized import (
    CLOCK_UNIX_WALL_NS,
    CLOCK_UNIX_WALL_S,
    POSE_UNAVAILABLE_NOT_IN_PROTOCOL,
    TRANSPORT_DIRECT_UDP,
    ClockStamp,
    CoopInputError,
    EdgeTimingProvenance,
    MapTerminal,
    NormalizedCoopBatch,
    _require,
    assert_finite_records,
    nonempty_str,
    optional_finite,
    sha256_of,
    strict_finite,
    strict_int,
    freeze_records,
)
from .reference_pins import (
    DIRECT_AGENT_CREDITS,
    DIRECT_OBJECT_MAP_UPDATE_SCHEMA,
    DIRECT_PROTOCOL_VERSION,
    DIRECT_REQUIRED_IDENTITY,
    DIRECT_TERMINAL_OUTCOMES,
)


EDGE_TIMING_SOURCE = "SPLITFUSION_DIRECT_EDGE_TIMING_V1"
MAP_FEEDBACK_SOURCE = "SPLITFUSION_DIRECT_MAP_FEEDBACK_V1"

# The production action catalog bound in protocol.validate_object_map_update.
ACTION_CATALOG_SIZE = 72


def from_direct_object_map_update(
    document: Mapping[str, Any],
    *,
    identity: CoopIdentityConfig,
    received_timestamp_ns: int | None = None,
    received_clock_domain: str = CLOCK_UNIX_WALL_NS,
    map_feedback: Mapping[str, Any] | None = None,
    upstream_validator: Callable[[Mapping[str, Any]], None] | None = None,
) -> NormalizedCoopBatch:
    """Normalize one decoded production direct object-map update.

    ``received_timestamp_ns`` is the map's own ingest stamp. It is optional and
    defaults to absent: the document itself carries no receipt time, and
    inventing one would fabricate an age.
    """

    _require(isinstance(document, Mapping), "direct update must be an object")
    if upstream_validator is not None:
        # Production validation runs first and unchanged when it is available.
        upstream_validator(document)

    _require(
        document.get("schema") == DIRECT_OBJECT_MAP_UPDATE_SCHEMA,
        f"direct object map update schema drift: {document.get('schema')!r}",
    )
    _require(
        strict_int(document.get("protocol_version"), "protocol_version")
        == DIRECT_PROTOCOL_VERSION,
        "direct object map update protocol version drift",
    )

    # ---- mandatory production identity ------------------------------------
    for name in DIRECT_REQUIRED_IDENTITY:
        _require(name in document, f"direct object map update lacks {name}")
        _require(
            document[name] not in (None, ""),
            f"direct object map update has an empty {name}",
        )

    run_id = nonempty_str(document["run_id"], "run_id")
    cell_id = nonempty_str(document["cell_id"], "cell_id")
    stream_id = nonempty_str(document["stream_id"], "stream_id")
    frame_id = strict_int(document["frame_id"], "frame_id", minimum=0)
    action_id = strict_int(document["action_id"], "action_id", minimum=0)
    _require(action_id < ACTION_CATALOG_SIZE, f"action_id {action_id} is outside the catalog")
    profile_id = nonempty_str(document["profile_id"], "profile_id")
    capture_ns = strict_int(document["capture_timestamp_ns"], "capture_timestamp_ns", minimum=1)
    sequence_id = strict_int(document.get("sequence_id"), "sequence_id", minimum=0)
    decoder_identity = nonempty_str(document.get("decoder_identity"), "decoder_identity")

    # ---- internal consistency of the redundant capture projections --------
    # build_object_map_update emits capture_timestamp = capture_timestamp_ns/1e9.
    # A disagreement means the document was edited after construction.
    capture_seconds = optional_finite(document.get("capture_timestamp"))
    if capture_seconds is not None:
        _require(
            abs(capture_seconds - capture_ns / 1_000_000_000.0) <= 1e-6,
            "capture_timestamp disagrees with capture_timestamp_ns",
        )

    # carla_timestamp is present in the production document, but the live direct
    # runtime passes 0.0 (live_pilot_runtime_direct_v1.py at the pinned digest).
    # It is therefore NOT a usable simulation clock and is never promoted to a
    # capture clock domain. It is validated for finiteness only.
    if document.get("carla_timestamp") is not None:
        strict_finite(document["carla_timestamp"], "carla_timestamp")

    service_deadline_at = strict_finite(
        document.get("service_deadline_at"), "service_deadline_at"
    )
    ack_timeout_at = strict_finite(document.get("ack_timeout_at"), "ack_timeout_at")
    _require(
        ack_timeout_at > service_deadline_at,
        "ACK timeout must exceed the service deadline",
    )

    # ---- records: validated, counted, and passed through verbatim ----------
    records = document.get("records")
    _require(isinstance(records, list), "direct object map update records must be a list")
    declared_count = strict_int(document.get("record_count"), "record_count", minimum=0)
    _require(
        declared_count == len(records),
        f"record_count {declared_count} disagrees with {len(records)} records",
    )
    scalar_count = assert_finite_records(records)
    declared_scalars = document.get("record_scalar_count")
    if declared_scalars is not None:
        _require(
            strict_int(declared_scalars, "record_scalar_count", minimum=0) == scalar_count,
            "record_scalar_count disagrees with the records subtree",
        )

    # Records that name their own stream/frame must agree with the envelope.
    for index, record in enumerate(records):
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

    identity_row = identity.resolve(run_id=run_id, cell_id=cell_id, stream_id=stream_id)

    receipt = None
    if received_timestamp_ns is not None:
        receipt = ClockStamp(
            strict_int(received_timestamp_ns, "received_timestamp_ns", minimum=0),
            nonempty_str(received_clock_domain, "received_clock_domain"),
        )

    edge_timing = EdgeTimingProvenance.from_mapping(
        document.get("edge_timing"),
        source=EDGE_TIMING_SOURCE,
        clock_domain=CLOCK_UNIX_WALL_S,
    )

    terminal = MapTerminal.absent()
    if map_feedback is not None:
        terminal = _map_terminal_from_feedback(
            map_feedback,
            run_id=run_id,
            cell_id=cell_id,
            stream_id=stream_id,
            frame_id=frame_id,
        )

    return NormalizedCoopBatch(
        transport=TRANSPORT_DIRECT_UDP,
        source_schema=DIRECT_OBJECT_MAP_UPDATE_SCHEMA,
        protocol_version=DIRECT_PROTOCOL_VERSION,
        run_id=run_id,
        cell_id=cell_id,
        ue_id=identity_row.ue_id,
        session_id=identity_row.session_id,
        stream_id=stream_id,
        frame_id=frame_id,
        sequence_id=sequence_id,
        action_id=action_id,
        profile_id=profile_id,
        decoder_identity=decoder_identity,
        # The production capture clock is a unix wall clock: map_ingest compares
        # capture_timestamp_ns directly against time.time(). It is NOT CARLA
        # simulation time, and carla_timestamp is not a substitute for one.
        capture=ClockStamp(capture_ns, CLOCK_UNIX_WALL_NS),
        receipt=receipt,
        records=freeze_records(records),
        record_scalar_count=scalar_count,
        records_sha256=sha256_of(list(records), "records"),
        payload_sha256=sha256_of(dict(document), "direct update"),
        identity_binding_source=BINDING_SOURCE,
        edge_timing=edge_timing,
        map_terminal=terminal,
        # Protocol v1 carries no ego/sensor pose or FoV. See pose_sidecar_v1.
        pose=None,
        pose_state=POSE_UNAVAILABLE_NOT_IN_PROTOCOL,
    )


def _map_terminal_from_feedback(
    feedback: Mapping[str, Any],
    *,
    run_id: str,
    cell_id: str,
    stream_id: str,
    frame_id: int,
) -> MapTerminal:
    """Normalize a ``splitfusion_direct_map_feedback.v1`` document.

    The feedback must belong to the same transmission obligation as the update;
    a mismatched terminal is refused rather than attached to the wrong frame.
    """

    _require(isinstance(feedback, Mapping), "map feedback must be an object")
    _require(
        strict_int(feedback.get("protocol_version"), "feedback.protocol_version")
        == DIRECT_PROTOCOL_VERSION,
        "map feedback protocol version drift",
    )
    _require(str(feedback.get("run_id")) == run_id, "map feedback run identity mismatch")
    _require(str(feedback.get("cell_id")) == cell_id, "map feedback cell identity mismatch")
    _require(
        str(feedback.get("stream_id")) == stream_id, "map feedback stream identity mismatch"
    )
    _require(
        strict_int(feedback.get("frame_id"), "feedback.frame_id") == frame_id,
        "map feedback frame identity mismatch",
    )
    outcome = nonempty_str(feedback.get("outcome"), "feedback.outcome")
    _require(outcome in DIRECT_TERMINAL_OUTCOMES, f"unregistered outcome {outcome!r}")
    credit = nonempty_str(feedback.get("agent_credit"), "feedback.agent_credit")
    _require(credit in DIRECT_AGENT_CREDITS, f"unregistered agent credit {credit!r}")

    install_raw = feedback.get("install_timestamp")
    install_stamp = None
    install_seconds = optional_finite(install_raw)
    if install_seconds is not None:
        install_stamp = ClockStamp(round(install_seconds * 1_000_000_000), CLOCK_UNIX_WALL_NS)
    if outcome == "RESULT_INSTALLED":
        _require(
            install_stamp is not None,
            "RESULT_INSTALLED feedback lacks a usable install_timestamp",
        )
    else:
        _require(
            install_stamp is None,
            "only RESULT_INSTALLED feedback may carry an install timestamp",
        )

    superseded = feedback.get("superseded_by_frame_id")
    superseded_frame = None
    if superseded not in (None, ""):
        superseded_frame = strict_int(superseded, "feedback.superseded_by_frame_id")

    return MapTerminal(
        source=MAP_FEEDBACK_SOURCE,
        outcome=outcome,
        agent_credit=credit,
        terminal=bool(feedback.get("terminal")),
        accepted=bool(feedback.get("accepted")),
        install_timestamp=install_stamp,
        map_age_at_install_ms=optional_finite(feedback.get("map_age_at_install_ms")),
        rejection_reason=str(feedback.get("rejection_reason") or ""),
        superseded_by_frame_id=superseded_frame,
    )
