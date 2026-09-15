#!/usr/bin/env python3
"""Spatial-map application with a direct edge-to-map ingest endpoint.

A narrow wrapper around the frozen, SHA-pinned
``uplink_only_spatial_map_pipeline/spatial_map_server_moving_ego_uplink_only_baseline.py``.
The baseline module is imported unmodified and supplies the authoritative map
state, its ``state_lock``, the installed-frame history, the normalisation rules,
the render loop and the whole Flask API. This wrapper only:

* binds the direct, edge-local ingest endpoint on the CN5G bridge address;
* installs validated updates through the baseline's own locked insertion;
* emits the compact feedback to the UE after installation;
* refuses to start the legacy loopback ingest listener, so the removed
  edge -> UE -> map detour has no surviving server-side endpoint.

Run with the baseline's arguments plus ``--direct-map-host``/``--direct-map-port``
and ``--ue-feedback-host``/``--ue-feedback-port``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uplink_only_spatial_map_pipeline import (  # noqa: E402
    spatial_map_server_moving_ego_uplink_only_baseline as baseline,
)

from rl_agent.splitfusion_direct_edge_map_v1 import protocol  # noqa: E402
from rl_agent.splitfusion_direct_edge_map_v1.map_ingest import (  # noqa: E402
    DirectMapIngestService,
)


SPATIAL_STREAM_SCHEMA = baseline.SPATIAL_STREAM_SCHEMA
READY_SCHEMA = "splitfusion_direct_map_ready.v1"
REPORT_SCHEMA = "splitfusion_direct_map_report.v1"

def _direct_parser() -> argparse.ArgumentParser:
    """This wrapper's own options, in one place.

    ``DIRECT_ARGUMENTS`` is derived from this parser rather than being written
    out a second time: a wrapper option missing from that list is silently
    forwarded to the baseline's parser, which rejects it and takes the map
    server down at launch.
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--direct-map-host", required=True)
    parser.add_argument("--direct-map-port", type=int, required=True)
    parser.add_argument("--ue-feedback-host", required=True)
    parser.add_argument("--ue-feedback-port", type=int, required=True)
    parser.add_argument("--direct-run-id", default="")
    parser.add_argument("--direct-cell-id", default="")
    parser.add_argument("--direct-allowed-action-ids", default="")
    parser.add_argument("--direct-processing-horizon-ms", type=float, default=500.0)
    parser.add_argument("--direct-ingest-csv", type=Path, default=None)
    parser.add_argument("--direct-ready-file", type=Path, default=None)
    parser.add_argument("--direct-report-file", type=Path, default=None)
    parser.add_argument("--direct-ingest-cpus", default="")
    parser.add_argument("--direct-receive-cpus", default="")
    parser.add_argument("--direct-ingest-queue-capacity", type=int, default=64)
    parser.add_argument(
        "--direct-render",
        choices=("on", "off"),
        default="on",
        help=(
            "Run the baseline map renderer. Measured at 97 ms median per frame "
            "on an empty state, held in one contiguous block, in the same "
            "interpreter as the receive and ingest owners. Turn it off for a "
            "measurement cell."
        ),
    )
    return parser


DIRECT_ARGUMENTS = tuple(
    option
    for action in _direct_parser()._actions
    for option in action.option_strings
)


def _split_direct_arguments(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Separate this wrapper's arguments from the baseline's own parser."""

    mine: list[str] = []
    rest: list[str] = []
    index = 0
    values = list(argv)
    while index < len(values):
        token = values[index]
        name = token.split("=", 1)[0]
        if name in DIRECT_ARGUMENTS:
            mine.append(token)
            if "=" not in token and index + 1 < len(values):
                mine.append(values[index + 1])
                index += 1
        else:
            rest.append(token)
        index += 1
    return mine, rest


def _parse_direct(argv: Sequence[str]) -> argparse.Namespace:
    return _direct_parser().parse_args(list(argv))


def install_under_state_lock(
    document: Mapping[str, Any], ingest_at: float
) -> dict[str, Any]:
    """Insert one validated update into the authoritative map state.

    The update is rendered into the baseline's own ``fusion_object_spatial_map.v1``
    shape and normalised by the baseline, so the installed record is
    byte-for-byte the same kind of object the historical pipeline installed --
    only its delivery path changed. ``install_timestamp`` is taken immediately
    before the lock and the function returns only after the insert completed.
    """

    payload = {
        "schema": SPATIAL_STREAM_SCHEMA,
        "stream_id": str(document["stream_id"]),
        "frame_id": int(document["frame_id"]),
        "capture_id": f"{document['stream_id']}:{int(document['frame_id'])}",
        "capture_timestamp": int(document["capture_timestamp_ns"]) / 1_000_000_000.0,
        "action_id": str(document["action_id"]),
        "carla_timestamp": float(document.get("carla_timestamp") or 0.0),
        "objects": list(document.get("records") or ()),
        "segmentation": dict(document.get("segmentation") or {}),
        "timing": dict(document.get("edge_timing") or {}),
        "source_script": "splitfusion_direct_edge_map_v1",
    }
    normalized = baseline._normalize_packet(payload, ingest_at)
    association_end_at = time.time()
    install_timestamp = association_end_at
    normalized["install_timestamp"] = install_timestamp
    history_key = (str(normalized["stream_id"]), int(normalized["frame_id"]))
    # Lock ownership is deliberately only the insert: normalisation, the
    # history limit read and every allocation happen outside it. The request
    # and acquisition instants are reported separately so contention on the
    # authoritative map state can be measured rather than assumed.
    lock_request_at = time.time()
    with baseline.state_lock:
        lock_acquired_at = time.time()
        baseline.latest_streams[str(normalized["stream_id"])] = normalized
        baseline.installed_frame_history[history_key] = normalized
        baseline.installed_frame_history.move_to_end(history_key)
        limit = max(1, int(baseline._config().installed_frame_history_size))
        while len(baseline.installed_frame_history) > limit:
            baseline.installed_frame_history.popitem(last=False)
    lock_released_at = time.time()
    return {
        "install_timestamp": install_timestamp,
        "object_count": int(normalized["object_count"]),
        "association_end_at": association_end_at,
        "map_lock_request_at": lock_request_at,
        "map_lock_acquired_at": lock_acquired_at,
        "map_lock_released_at": lock_released_at,
    }


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    mine, rest = _split_direct_arguments(values)
    direct = _parse_direct(mine)

    saved_argv = list(sys.argv)
    sys.argv = [saved_argv[0]] + rest
    try:
        baseline.CONFIG = baseline.parse_args()
    finally:
        sys.argv = saved_argv

    cfg = baseline.CONFIG
    # The legacy loopback ingest listener is deliberately never started: in the
    # corrected architecture the object map update arrives only from the edge.
    os.makedirs(str(cfg.output_dir), exist_ok=True)
    baseline._init_map_metrics_logger(str(cfg.ingest_metrics_csv))

    protocol.assert_direct_map_endpoint(
        direct.direct_map_host,
        direct.direct_map_port,
        ue_hosts=("10.0.0.2",),
        forbidden_ports=(51004, 51104),
    )
    allowed = tuple(
        int(value)
        for value in str(direct.direct_allowed_action_ids or "").split(",")
        if str(value).strip()
    )
    service = DirectMapIngestService(
        bind_host=str(direct.direct_map_host),
        bind_port=int(direct.direct_map_port),
        feedback_host=str(direct.ue_feedback_host),
        feedback_port=int(direct.ue_feedback_port),
        install=install_under_state_lock,
        ingest_csv=direct.direct_ingest_csv,
        expected_run_id=str(direct.direct_run_id),
        expected_cell_id=str(direct.direct_cell_id),
        allowed_action_ids=allowed,
        processing_horizon_s=float(direct.direct_processing_horizon_ms) / 1000.0,
        ingest_queue_capacity=int(direct.direct_ingest_queue_capacity),
        ingest_cpus=str(direct.direct_ingest_cpus),
        receive_cpus=str(direct.direct_receive_cpus),
    )
    service.start()

    # The renderer is a visualisation aid, not part of the authoritative map
    # state. One render measures 97 ms median on an empty state and the loop
    # sleeps at most 20 ms, so it holds the GIL in ~100 ms contiguous blocks --
    # in the same interpreter as the receive and ingest owners. That is the
    # measured shape of the map-service tail: p99 stalls of 92-238 ms at the
    # arrival stamp and at ingest validation, for work of 0.6-2.5 ms. Off for a
    # measurement cell; the install path and every recorded outcome are
    # identical either way.
    render_enabled = str(direct.direct_render) == "on"
    if render_enabled:
        render = threading.Thread(target=baseline.render_thread, daemon=True)
        render.start()

    def _write_report() -> None:
        if direct.direct_report_file is None:
            return
        document = {
            "schema": REPORT_SCHEMA,
            "status": "COMPLETE",
            "direct_ingest": service.report(),
            "legacy_loopback_listener_started": False,
            "render_thread_started": bool(render_enabled),
            "written_at_unix_s": time.time(),
        }
        temporary = Path(str(direct.direct_report_file) + ".partial")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, direct.direct_report_file)

    if direct.direct_ready_file is not None:
        ready = {
            "schema": READY_SCHEMA,
            "direct_map_host": str(direct.direct_map_host),
            "direct_map_port": int(direct.direct_map_port),
            "ue_feedback_host": str(direct.ue_feedback_host),
            "ue_feedback_port": int(direct.ue_feedback_port),
            "legacy_loopback_ingest_listener": "NOT_STARTED",
            "run_id": str(direct.direct_run_id),
            "cell_id": str(direct.direct_cell_id),
            "allowed_action_ids": list(allowed),
            "processing_horizon_ms": float(direct.direct_processing_horizon_ms),
            "installed_frame_history_size": int(cfg.installed_frame_history_size),
            "render_thread_started": bool(render_enabled),
        }
        path = Path(direct.direct_ready_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(ready, handle, sort_keys=True)

    def _shutdown(_signum: int, _frame: Any) -> None:
        """Persist the ingest report before the process goes away.

        Flask's development server does not return from ``app.run`` on a signal,
        so the ``finally`` block below is not reached on a normal supervised
        teardown. The counters are written here instead, then the process exits
        directly. The per-row ingest CSV is flushed per row and is authoritative
        regardless.
        """

        baseline.STOP_EVENT.set()
        try:
            service.close()
            _write_report()
            baseline._close_map_metrics_logger()
        finally:
            os._exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(
        f"[DIRECT] edge->map ingest on {direct.direct_map_host}:{direct.direct_map_port}; "
        f"feedback to {direct.ue_feedback_host}:{direct.ue_feedback_port}; "
        "legacy loopback ingest listener NOT started",
        flush=True,
    )
    try:
        baseline.app.run(
            host=str(cfg.api_host), port=int(cfg.api_port), threaded=True
        )
    finally:
        baseline.STOP_EVENT.set()
        service.close()
        _write_report()
        baseline._close_map_metrics_logger()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
