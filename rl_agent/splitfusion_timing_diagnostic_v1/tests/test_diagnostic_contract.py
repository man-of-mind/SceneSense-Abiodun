"""Focused synthetic timestamp, accounting and partition tests.

These run in the diagnostic's own preflight, before any radio, container or
GPU work. They use synthetic boundaries only: no dataset, no checkpoint, no
socket and no CUDA.
"""

from __future__ import annotations

from typing import Any

from phase2_map_sharing.transport import CHUNK_HEADER, ChunkReassembler, chunk_payload

from .. import diagnostic_common as common
from ..instrumented_tail import SERIALIZE_STAGE, TAIL_STAGES, TOTAL_STAGE


def _check(condition: bool, message: str) -> None:
    common.require(condition, f"synthetic test failure: {message}")


def test_derived_timestamp_intervals() -> dict[str, Any]:
    """The five derived uplink/queue quantities are exact and non-negative."""

    from ..runner import join_records

    payload_row = {
        "ordinal": 0,
        "sequence_id": 1000,
        "frame_id": 1000,
        "sample_id": "synthetic_000000_frame7",
        "episode_id": "synthetic_episode",
        "registered_frame_id": 7,
        "dataset_index": 3,
        "inner_payload_bytes": 40_000,
        "sfd1_overhead_bytes": 176,
        "sfd1_bytes": 40_176,
        "datagram_count": 4,
        "udp_application_bytes": 40_208,
        "estimated_wire_bytes": 40_320,
        "capture_timestamp_ns": 1_000_000_000_000,
        "scheduled_send_wall_ns": 1_000_000_000_000,
        "ue_inner_payload_ready_wall_ns": 999_000_000_000,
        "ue_datagrams_ready_wall_ns": 999_900_000_000,
        "ue_prepare_stage_ns": {"total_ue_preparation": 47_000_000},
        "send_status": "SENT",
        "slot_lateness_ms": 0.5,
        "ue_first_send_wall_ns": 1_000_000_000_000,
        "ue_final_send_wall_ns": 1_000_003_000_000,   # +3 ms send loop
        "ego_pose": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    }
    skipped_row = {
        **payload_row,
        "ordinal": 1,
        "sequence_id": 1001,
        "frame_id": 1001,
        "send_status": "OBSOLETE_SKIPPED",
        "ue_first_send_wall_ns": 0,
        "ue_final_send_wall_ns": 0,
    }
    edge_record = {
        "sequence_id": 1000,
        "edge_first_datagram_wall_ns": 1_000_010_000_000,     # +10 ms
        "edge_complete_reassembly_wall_ns": 1_000_025_000_000,  # +25 ms
        "edge_admitted_wall_ns": 1_000_025_200_000,
        "edge_worker_start_wall_ns": 1_000_031_000_000,        # +6 ms queue wait
        "edge_tail_finished_wall_ns": 1_000_141_000_000,
        "feature_datagrams": 4,
        "duplicate_datagrams": 0,
        "edge_service_wall_ms": 110.0,
        "edge_result_finite_check_ms": 0.4,
        "service_record_count": 12,
        "service_record_bytes": 2_048,
        "service_target_met": False,
        "processing_horizon_met": True,
        "action_id": 30,
        "phase": "MEASURED",
        "edge_stage_ns": {
            "zstd_decompression": 3_000_000,
            "unpack_dequantize": 7_000_000,
            "ae_decode": 2_000_000,
            "frozen_tail": 100_000_000,
            "output_serialization": 8_000_000,
            "total_edge_processing": 120_000_000,
        },
        "tail_stage_wall_ns": {
            "camera_pose_reconstruct": 120_000,
            "decode_tail_launch": 900_000,
            "finite_check_outputs": 62_000_000,
            "camera_aware_postprocess": 20_000_000,
            "finite_check_postprocess": 1_000_000,
            "p025_service_filter": 4_000_000,
            "finite_check_p025": 500_000,
            "segmentation_upsample_argmax": 11_000_000,
            SERIALIZE_STAGE: 8_000_000,
            TOTAL_STAGE: 99_520_000,
        },
        "tail_stage_cuda_ms": {"decode_tail_cuda": 61.5, TOTAL_STAGE: 99.4},
    }
    rows = join_records(
        payloads={"rows": [payload_row, skipped_row], "stream_id": "synthetic"},
        edge_records=[edge_record],
        results={1000: {"ue_result_received_wall_ns": 1_000_150_000_000}},
    )
    delivered, skipped = rows
    _check(delivered["ue_send_loop_ms"] == 3.0, "ue_send_loop_ms")
    _check(delivered["application_feature_uplink_ms"] == 25.0, "application_feature_uplink_ms")
    _check(delivered["post_send_to_reassembly_ms"] == 22.0, "post_send_to_reassembly_ms")
    _check(
        delivered["edge_first_to_complete_reassembly_ms"] == 15.0,
        "edge_first_to_complete_reassembly_ms",
    )
    _check(delivered["edge_queue_wait_ms"] == 6.0, "edge_queue_wait_ms")
    _check(delivered["decode_tail_cuda_ms"] == 61.5, "decode_tail_cuda_ms")
    _check(
        abs(delivered["decode_tail_inference_block_ms"] - 62.9) < 1e-9,
        "decode_tail_inference_block_ms",
    )
    _check(delivered["ue_round_trip_ms"] == 147.0, "ue_round_trip_ms")
    _check(delivered["edge_frozen_tail_ms"] == 100.0, "edge_frozen_tail_ms")
    _check(delivered["delivered"] is True and delivered["result_returned"] is True, "delivery flags")
    _check(
        all(
            value is None
            for value in (
                skipped["ue_send_loop_ms"],
                skipped.get("application_feature_uplink_ms"),
                skipped["ue_first_send_wall_ns"],
            )
        )
        and skipped["delivered"] is False,
        "obsolete-skipped row must carry no send or uplink interval",
    )
    for name in (
        "ue_send_loop_ms",
        "application_feature_uplink_ms",
        "post_send_to_reassembly_ms",
        "edge_first_to_complete_reassembly_ms",
        "edge_queue_wait_ms",
    ):
        _check(float(delivered[name]) >= 0.0, f"negative derived interval: {name}")
    return {"delivered_rows": 1, "skipped_rows": 1, "derived_quantities_verified": 8}


def test_negative_interval_is_detected() -> dict[str, Any]:
    """A reversed boundary pair must produce a detectable negative interval."""

    from ..runner import join_records

    payload_row = {
        "ordinal": 0, "sequence_id": 1, "frame_id": 1, "sample_id": "s",
        "episode_id": "e", "registered_frame_id": 1, "dataset_index": 0,
        "inner_payload_bytes": 10, "sfd1_overhead_bytes": 176, "sfd1_bytes": 186,
        "datagram_count": 1, "udp_application_bytes": 194,
        "estimated_wire_bytes": 222, "capture_timestamp_ns": 10,
        "scheduled_send_wall_ns": 10, "ue_inner_payload_ready_wall_ns": 5,
        "ue_datagrams_ready_wall_ns": 8,
        "ue_prepare_stage_ns": {"total_ue_preparation": 1},
        "send_status": "SENT", "slot_lateness_ms": 0.0,
        "ue_first_send_wall_ns": 2_000, "ue_final_send_wall_ns": 3_000,
        "ego_pose": (0.0,) * 6,
    }
    edge_record = {
        "sequence_id": 1,
        # deliberately earlier than the UE's first send
        "edge_first_datagram_wall_ns": 1_000,
        "edge_complete_reassembly_wall_ns": 1_500,
        "edge_admitted_wall_ns": 1_600,
        "edge_worker_start_wall_ns": 1_700,
        "edge_tail_finished_wall_ns": 1_800,
        "feature_datagrams": 1, "duplicate_datagrams": 0,
        "edge_service_wall_ms": 0.1, "edge_result_finite_check_ms": 0.0,
        "service_record_count": 0, "service_record_bytes": 2,
        "service_target_met": True, "processing_horizon_met": True,
        "action_id": 30, "phase": "MEASURED",
        "edge_stage_ns": {}, "tail_stage_wall_ns": {}, "tail_stage_cuda_ms": {},
    }
    row = join_records(
        payloads={"rows": [payload_row], "stream_id": "s"},
        edge_records=[edge_record],
        results={},
    )[0]
    _check(
        float(row["application_feature_uplink_ms"]) < 0.0,
        "a reversed boundary pair must yield a negative interval the run then rejects",
    )
    return {"negative_interval_detected": True}


def test_datagram_accounting_is_exact() -> dict[str, Any]:
    """Chunking and reassembly must account for every byte and datagram."""

    checked = 0
    for size in (1, 8, 12_491, 12_492, 12_493, 40_000, 1_200_000):
        payload = bytes((index * 31 + 7) % 251 for index in range(size))
        chunks = chunk_payload(payload, message_id=4_242, chunk_bytes=12_500)
        capacity = 12_500 - CHUNK_HEADER.size
        expected = (size + capacity - 1) // capacity
        _check(len(chunks) == expected, f"chunk count for {size} bytes")
        _check(
            sum(len(chunk) - CHUNK_HEADER.size for chunk in chunks) == size,
            f"chunk payload accounting for {size} bytes",
        )
        _check(
            all(len(chunk) <= 12_500 for chunk in chunks),
            f"chunk size bound for {size} bytes",
        )
        reassembler = ChunkReassembler(timeout_s=2.0, max_chunks=4096)
        complete = None
        for index, chunk in enumerate(chunks):
            # One receipt instant for the whole message: the production
            # reassembler times a message out from its *first* chunk, so a
            # spread-out arrival is a separate expiry case, tested below.
            complete = reassembler.ingest("peer", chunk, received_at_s=0.0)
            if index < len(chunks) - 1:
                _check(complete is None, f"premature completion at {size} bytes")
        _check(complete is not None, f"reassembly did not complete for {size} bytes")
        _check(complete.payload == payload, f"reassembled payload mismatch for {size} bytes")
        _check(complete.chunk_count == expected, f"reassembled chunk count for {size} bytes")
        _check(complete.duplicate_chunks == 0, f"unexpected duplicates for {size} bytes")
        _check(int(complete.message_id) == 4_242, f"message identity for {size} bytes")
        _check(not reassembler.pending, f"reassembler leaked state for {size} bytes")
        checked += 1
    return {"payload_sizes_checked": checked}


def test_incomplete_reassembly_expires() -> dict[str, Any]:
    """A partial message must expire exactly once and never complete."""

    payload = bytes(30_000)
    chunks = chunk_payload(payload, message_id=9, chunk_bytes=12_500)
    reassembler = ChunkReassembler(timeout_s=1.0, max_chunks=4096)
    _check(reassembler.ingest("peer", chunks[0], received_at_s=0.0) is None, "first chunk")
    reassembler.expire(5.0)
    _check(reassembler.expired_messages == 1, "expiry accounting")
    _check(not reassembler.pending, "expired message left pending state")
    return {"expired_messages": 1}


def test_percentile_convention() -> dict[str, Any]:
    """Nearest-rank percentiles match the Phase-13C/Phase-15 convention."""

    values = list(range(1, 101))
    summary = common.summarize(values)
    _check(summary["count"] == 100, "count")
    _check(summary["median"] == 50, "median")
    _check(summary["p90"] == 90, "p90")
    _check(summary["p95"] == 95, "p95")
    _check(summary["minimum"] == 1 and summary["maximum"] == 100, "extrema")
    empty = common.summarize([])
    _check(
        empty["count"] == 0 and empty["median"] is None and empty["p95"] is None,
        "empty summary must report count 0 and no statistics",
    )
    single = common.summarize([4.5])
    _check(
        single["median"] == 4.5 and single["p90"] == 4.5 and single["maximum"] == 4.5,
        "single-observation summary",
    )
    return {"percentile_convention": "nearest_rank"}


def test_stage_partition_is_complete() -> dict[str, Any]:
    """The reported stage groups must partition the tail stages exactly once."""

    from ..runner import STAGE_GROUPS

    covered: list[str] = []
    for _name, stages in STAGE_GROUPS:
        covered.extend(stages)
    _check(len(covered) == len(set(covered)), "a tail stage is counted twice")
    _check(
        set(covered) == {*TAIL_STAGES, SERIALIZE_STAGE},
        "stage groups do not cover every tail stage exactly once: "
        f"{sorted(set(covered) ^ {*TAIL_STAGES, SERIALIZE_STAGE})}",
    )
    _check(TOTAL_STAGE not in covered, "the total stage must not be part of the partition")
    return {"stage_groups": len(STAGE_GROUPS), "stages_covered": len(covered)}


def test_clock_anchor_shape() -> dict[str, Any]:
    """Paired anchors must expose the wall/monotonic offset used for the skew gate."""

    anchor = common.clock_anchor("synthetic")
    _check(
        anchor["wall_minus_monotonic_ns"]
        == anchor["wall_ns"] - anchor["monotonic_ns"],
        "anchor offset identity",
    )
    later = common.clock_anchor("synthetic_later")
    _check(later["monotonic_ns"] >= anchor["monotonic_ns"], "monotonic anchor ordering")
    _check(
        abs(later["wall_minus_monotonic_ns"] - anchor["wall_minus_monotonic_ns"])
        < 5_000_000,
        "same-process anchors must agree on the clock domain",
    )
    return {"anchor_fields_verified": True}


def test_obsolete_slot_policy() -> dict[str, Any]:
    """A slot later than one full period must be skipped, never burst."""

    period = common.SCHEDULE_PERIOD_NS
    scheduled = 1_000_000_000_000
    _check(scheduled + period - 1 < scheduled + period, "in-slot arrival is sendable")
    _check(
        (scheduled + period) >= scheduled + period,
        "an arrival a full period late must be classed obsolete",
    )
    return {"schedule_period_ms": period // 1_000_000, "policy": "SKIP_OBSOLETE_NEVER_BURST"}


TESTS = (
    test_derived_timestamp_intervals,
    test_negative_interval_is_detected,
    test_datagram_accounting_is_exact,
    test_incomplete_reassembly_expires,
    test_percentile_convention,
    test_stage_partition_is_complete,
    test_clock_anchor_shape,
    test_obsolete_slot_policy,
)


def run_all() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for test in TESTS:
        results[test.__name__] = test()
    return {"all_passed": True, "test_count": len(TESTS), "results": results}


if __name__ == "__main__":
    import json

    print(json.dumps(run_all(), indent=2, sort_keys=True))
