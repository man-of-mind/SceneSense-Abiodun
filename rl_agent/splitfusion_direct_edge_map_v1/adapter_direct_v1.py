#!/usr/bin/env python3
"""Route-B cell adapter in direct edge-to-map mode.

A narrow wrapper around the SHA-pinned
``rl_agent/ue_route_b_split_cell_adapter_v1.py``. The pinned adapter keeps
ownership of CARLA, Route B, the sensors, the preparation path, the target-SNR
actuator, the evaluation workers and every structural acceptance gate. This
module rebinds exactly four seams:

1. ``start_map_process``  -> the direct spatial-map server, bound edge-locally.
2. ``start_live_edge``    -> the direct edge service, publishing straight to the map.
3. ``LivePilotCellRuntime`` -> the direct UE runtime, which never forwards records.
4. ``InstallFeedbackLedger`` -> a compatible view over the direct terminal ledger.

Seam 4 keeps the pinned feedback worker's semantics exactly: the obligation is
closed inside ``receive_once``, between the worker's ``before`` snapshot and its
``completed`` difference, rather than asynchronously underneath it.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rl_agent.ue_route_b_split_cell_adapter_v1 as pinned  # noqa: E402
import rl_agent.ue_map_install_feedback_v1 as pinned_feedback  # noqa: E402

from rl_agent.splitfusion_direct_edge_map_v1 import protocol  # noqa: E402
from rl_agent.splitfusion_direct_edge_map_v1.endpoint import (  # noqa: E402
    resolve_direct_map_endpoint,
)
from rl_agent.splitfusion_direct_edge_map_v1.live_pilot_runtime_direct_v1 import (  # noqa: E402
    DirectLivePilotCellRuntime,
)
from rl_agent.splitfusion_direct_edge_map_v1.ue_ledger import (  # noqa: E402
    DirectTerminalLedger,
)


DIRECT_MAP_SERVER = (
    "rl_agent/splitfusion_direct_edge_map_v1/spatial_map_direct_server_v1.py"
)
DIRECT_EDGE_MODULE = (
    "-m rl_agent.splitfusion_direct_edge_map_v1.live_pilot_runtime_direct_v1"
)

# Set once per cell by the patched runtime factory and consumed by the patched
# ledger factory a few lines later inside the pinned collector's constructor.
_PENDING_LEDGER: dict[str, Any] = {}
_ENDPOINT: dict[str, Any] = {}


class CompatDirectLedger:
    """``InstallFeedbackLedger``-shaped view over :class:`DirectTerminalLedger`.

    The direct runtime's control loop only enqueues; this object performs the
    ledger write inside ``receive_once`` so the pinned feedback worker observes
    the same ``pending`` transition it was written against.
    """

    def __init__(self, ledger: DirectTerminalLedger, *, profile_id: str) -> None:
        self._ledger = ledger
        self._profile_id = str(profile_id)
        self._inbox: "queue.Queue[tuple[Mapping[str, Any], float]]" = queue.Queue()
        self.contract_errors: list[str] = []
        # A pending view owned by this object rather than delegated to the
        # ledger.
        #
        # The supervising feedback worker snapshots ``pending`` before calling
        # ``receive_once``, then calls ``record_expired``, then attributes the
        # single returned message's status to *every* capture that left
        # ``pending`` in between. If the watchdog closed other captures in that
        # window they are silently credited with this message's status. That
        # mis-credits timed-out captures as ACK_INSTALLED and then demands an
        # exact installed map record for a frame that was never installed.
        #
        # Keeping the view here makes the difference exact: only the capture
        # that ``receive_once`` actually closed leaves it.
        self._pending: dict[str, Any] = {}
        self._view_lock = threading.Lock()

    @property
    def pending(self) -> dict[str, Any]:
        with self._view_lock:
            return dict(self._pending)

    @property
    def timed_out(self) -> set[str]:
        return self._ledger.timed_out

    def register_capture(
        self,
        *,
        stream_id: str,
        capture_id: str,
        frame_id: int,
        capture_at: float,
        action_id: str,
        service_deadline_at: float,
        ack_timeout_at: float,
    ) -> None:
        self._ledger.register_capture(
            stream_id=stream_id,
            capture_id=capture_id,
            frame_id=frame_id,
            capture_at=capture_at,
            action_id=action_id,
            profile_id=self._profile_id,
            service_deadline_at=service_deadline_at,
            ack_timeout_at=ack_timeout_at,
        )
        with self._view_lock:
            self._pending[str(capture_id)] = {"frame_id": int(frame_id)}

    def enqueue(self, message: Mapping[str, Any], received_at: float) -> None:
        self._inbox.put((dict(message), float(received_at)))

    def receive_once(self) -> dict[str, Any] | None:
        try:
            message, received_at = self._inbox.get(timeout=0.05)
        except queue.Empty:
            return None
        row = self._ledger.record_message(message, received_at)
        # Exactly this capture leaves the view, so the caller's
        # before/after difference names exactly this message's capture.
        with self._view_lock:
            self._pending.pop(str(message.get("capture_id") or ""), None)
        return self._compat_view(message, row)

    @staticmethod
    def _compat_view(
        message: Mapping[str, Any], row: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Translate a direct terminal into the pinned worker's vocabulary."""

        outcome = str(message.get("outcome") or "")
        status = (
            "ACK_INSTALLED"
            if outcome == protocol.OUTCOME_RESULT_INSTALLED
            else "NACK_REJECTED"
        )
        return {
            "schema": str(message.get("schema") or ""),
            "status": status,
            "outcome": outcome,
            "agent_credit": str(message.get("agent_credit") or ""),
            "terminal": bool(row.get("terminal")),
            "frame_id": message.get("frame_id", ""),
            "capture_id": message.get("capture_id", ""),
            "action_id": message.get("action_id", ""),
            "install_timestamp": message.get("install_timestamp", ""),
            "feedback_emit_at": message.get(
                "feedback_emit_at", message.get("emit_at", "")
            ),
            "map_age_at_install_ms": message.get("map_age_at_install_ms", ""),
            "rejection_reason": str(message.get("rejection_reason") or ""),
            "result_status": (
                "DECODED_RESULT_ACCEPTED_AND_INSTALLED"
                if status == "ACK_INSTALLED"
                else "RESULT_REJECTED"
            ),
        }

    def record_expired(self, now: float | None = None) -> int:
        """Write UE-local timeout terminals without disturbing the pending view.

        The timed-out capture stays in the view until a message for it is
        received or the cell finishes, so it can never be mistaken for the
        capture that ``receive_once`` just closed.
        """

        return self._ledger.record_expired(now)

    def drain_view(self) -> int:
        """Drop view entries the ledger has already closed. Used at teardown."""

        with self._view_lock:
            closed = set(self._ledger.closed)
            removed = [key for key in self._pending if key in closed]
            for key in removed:
                self._pending.pop(key, None)
            return len(removed)

    def record_reassembly_failure(self, *, capture_id: str, reason: str) -> None:
        # The direct architecture has no UE-side reassembly of object records,
        # so this pinned hook can never fire; it is kept for interface parity.
        raise pinned_feedback.FeedbackContractError(
            "UE-side result reassembly does not exist in the direct architecture"
        )

    def summary(self) -> dict[str, Any]:
        return self._ledger.summary()

    def close(self) -> None:
        self.drain_view()
        self._ledger.close()


def direct_runtime_factory(
    *,
    campaign: Mapping[str, Any],
    cell: Mapping[str, Any],
    attempt_dir: Path,
    map_host: str,
    map_port: int,
    evidence_dir: Path,
) -> DirectLivePilotCellRuntime:
    """Replacement for ``LivePilotCellRuntime`` inside the pinned collector."""

    del map_host, map_port  # the UE never publishes to the map any more
    ledger = DirectTerminalLedger(
        output_csv=Path(attempt_dir) / "map_feedback.csv",
        experiment_id=str(campaign["campaign_id"]),
        cell_id=str(cell["cell_id"]),
    )
    compat = CompatDirectLedger(ledger, profile_id=str(cell["profile_id"]))
    _PENDING_LEDGER["ledger"] = compat
    return DirectLivePilotCellRuntime(
        campaign=campaign,
        cell=cell,
        attempt_dir=Path(attempt_dir),
        evidence_dir=Path(evidence_dir),
        ue_control_port=int(campaign["runtime"]["ue_control_port"]),
        ledger=compat,
    )


def direct_ledger_factory(
    *,
    output_csv: Path,
    experiment_id: str,
    cell_id: str,
    bind_host: str,
    bind_port: int,
) -> CompatDirectLedger:
    """Replacement for ``InstallFeedbackLedger`` inside the pinned collector."""

    del output_csv, experiment_id, cell_id, bind_host, bind_port
    compat = _PENDING_LEDGER.pop("ledger", None)
    if compat is None:
        raise pinned.AdapterError(
            "direct terminal ledger was not created by the runtime factory"
        )
    return compat


def _reservation(campaign: Mapping[str, Any], name: str, default: Any = "") -> Any:
    """Read one CPU-reservation setting; absent means no placement request."""

    block = (campaign.get("direct_edge_map") or {}).get("cpu_reservation") or {}
    value = block.get(name, default)
    return value if isinstance(value, int) else str(value or "")


def start_direct_map_process(
    campaign: Mapping[str, Any],
    *,
    temporary_dir: Path,
    action_id: str,
    carla_host: str,
    carla_port: int,
    api_port: int,
    udp_port: int,
    feedback_port: int,
) -> "subprocess.Popen[bytes]":
    """Start the spatial map with an edge-local direct ingest endpoint."""

    del udp_port, feedback_port  # the loopback detour endpoints are not used
    runtime = campaign["runtime"]
    endpoint = _ENDPOINT["endpoint"]
    runtime_path = pinned.repo_path(DIRECT_MAP_SERVER)
    evidence = _direct_evidence_dir()
    ingest_csv = evidence / "direct_map_ingest.csv"
    ready_file = evidence / "direct_map_ready.json"
    report_file = evidence / "direct_map_report.json"
    argv = [
        sys.executable,
        str(runtime_path),
        "--api-host", "127.0.0.1",
        "--api-port", str(api_port),
        "--default-action-id", str(action_id),
        "--carla-host", str(carla_host),
        "--carla-port", str(carla_port),
        "--output-dir", str(Path(temporary_dir) / "map"),
        "--focus-follow-stream-id", "unused",
        "--installed-frame-history-size",
        str(int(campaign["measurement_contract"]["installed_frame_history_size"])),
        "--direct-map-host", str(endpoint.host),
        "--direct-map-port", str(int(runtime["direct_map_ingest_port"])),
        "--ue-feedback-host", str(runtime["ue_bind_host"]),
        "--ue-feedback-port", str(int(runtime["ue_control_port"])),
        "--direct-run-id", str(campaign["campaign_id"]),
        "--direct-cell-id", str(_ENDPOINT.get("cell_id", "")),
        "--direct-allowed-action-ids", str(action_id),
        "--direct-processing-horizon-ms",
        str(float(campaign["cell"]["ack_timeout_ms"])),
        "--direct-ingest-csv", str(ingest_csv),
        "--direct-ready-file", str(ready_file),
        "--direct-report-file", str(report_file),
        "--direct-receive-cpus", _reservation(campaign, "map_receive_cpus"),
        "--direct-ingest-cpus", _reservation(campaign, "map_ingest_cpus"),
        "--direct-ingest-queue-capacity",
        str(int(_reservation(campaign, "map_ingest_queue_capacity", 64))),
        "--direct-render",
        str(_reservation(campaign, "map_render", "on") or "on"),
    ]
    process = subprocess.Popen(argv, cwd=str(pinned.ROOT), stdin=subprocess.DEVNULL)
    try:
        import urllib.error
        import urllib.request

        deadline = time.monotonic() + 60.0
        url = f"http://127.0.0.1:{api_port}/healthz"
        while time.monotonic() < deadline:
            pinned.require(
                process.poll() is None, "direct map process exited during startup"
            )
            if ready_file.is_file():
                try:
                    with urllib.request.urlopen(url, timeout=1.0) as response:
                        if response.status == 200:
                            ready = json.loads(ready_file.read_text(encoding="utf-8"))
                            pinned.require(
                                ready.get("legacy_loopback_ingest_listener")
                                == "NOT_STARTED",
                                "direct map still started the legacy loopback listener",
                            )
                            pinned.require(
                                str(ready.get("direct_map_host")) == str(endpoint.host),
                                "direct map bound an unexpected host",
                            )
                            return process
                except (OSError, urllib.error.URLError):
                    pass
            time.sleep(0.5)
        raise pinned.AdapterError("direct map process did not become ready")
    except BaseException:
        pinned.stop_process(process)
        raise


def start_direct_live_edge(
    campaign: Mapping[str, Any], cell: Mapping[str, Any], temporary_dir: Path
) -> Path:
    """Start the direct edge service in the OAI-network GPU container."""

    runtime = campaign["runtime"]
    endpoint = _ENDPOINT["endpoint"]
    pinned.require(
        not pinned.tail_running(),
        "a previous phase-owned edge container is still running",
    )
    edge_scratch = pinned.create_cell_edge_state_root(temporary_dir)
    pinned.seed_cell_edge_state(campaign, edge_scratch)
    evidence_host = edge_scratch / pinned.EDGE_EVIDENCE_LEAF
    evidence_host.mkdir(parents=False, exist_ok=False, mode=0o777)
    os.chmod(evidence_host, 0o777)
    evidence_container = Path("/work/torch_cache") / pinned.EDGE_EVIDENCE_LEAF
    ready_host = edge_scratch / "ready.json"
    ready_container = Path("/work/torch_cache/ready.json")
    config_container = Path("/work/abiodun") / str(_ENDPOINT["config_relpath"])
    allowed = ",".join(
        str(value)
        for value in campaign.get("_qualification", {}).get(
            "action_ids", [cell["action_id"]]
        )
    )
    env = os.environ.copy()
    env.update(
        {
            "FUSION_BACK_DUAL": "0",
            "FUSION_BACK_BIND_HOST": "0.0.0.0",
            # Retained only because the shared launcher template interpolates
            # them; the direct edge never uses a UE result destination.
            "FUSION_BACK_REMOTE_HOST": str(runtime["ue_bind_host"]),
            "FUSION_BACK_REMOTE_HOST_1": str(runtime["ue_bind_host"]),
            "FUSION_BACK_DEVICE": "cuda",
            "SPLITFUSION_EDGE_STATE_ROOT": str(edge_scratch),
            "SPLITFUSION_FCOS_WEIGHT_PATH": str(
                pinned.repo_path(
                    str(campaign["deployment"]["fcos_constructor_weights"]["path"])
                )
            ),
            "FUSION_BACK_SCRIPT": DIRECT_EDGE_MODULE,
            "FUSION_REMOTE_PORT_1": str(runtime["edge_receive_port"]),
            "FUSION_REMOTE_SOURCE_PORT_1": str(runtime["edge_source_port"]),
            "FUSION_CAMERA_RESULT_PORT_1": str(runtime["camera_result_port"]),
            "FUSION_BACK_EXTRA_ARGS": " ".join(
                (
                    "--edge",
                    "--config", str(config_container),
                    "--action-id", str(cell["action_id"]),
                    "--allowed-action-ids", allowed,
                    "--ready-file", str(ready_container),
                    "--edge-port", str(runtime["edge_receive_port"]),
                    "--direct-map-host", str(endpoint.host),
                    "--direct-map-port", str(int(runtime["direct_map_ingest_port"])),
                    "--ue-control-host", str(runtime["ue_bind_host"]),
                    "--ue-control-port", str(int(runtime["ue_control_port"])),
                    pinned.EDGE_SEGMENTATION_EVIDENCE_FLAG, str(evidence_container),
                    "--run-id", str(campaign["campaign_id"]),
                    "--cell-id", str(cell["cell_id"]),
                    "--edge-compute-cpus",
                    _reservation(campaign, "edge_compute_cpus"),
                    "--edge-receive-cpus",
                    _reservation(campaign, "edge_receive_cpus"),
                )
            ),
        }
    )
    launcher_log = Path(temporary_dir) / "edge_launcher.log"
    try:
        with launcher_log.open("xb") as stream:
            completed = subprocess.run(
                [str(pinned.ROOT / "scripts/receiver_container_fusion_back_up.sh")],
                cwd=str(pinned.ROOT), env=env, check=False,
                stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, timeout=180.0,
            )
        pinned.require(
            completed.returncode == 0,
            f"direct edge container startup failed rc={completed.returncode}; "
            f"launcher_tail={pinned._bounded_log_tail(launcher_log)!r}",
        )
        deadline = time.monotonic() + 240.0
        while time.monotonic() < deadline:
            pinned.require(
                pinned.tail_running(),
                "direct edge container exited before preload completed",
            )
            if ready_host.is_file():
                ready = json.loads(ready_host.read_text(encoding="utf-8"))
                pinned.require(
                    ready.get("schema") == "splitfusion_direct_live_edge_ready.v1"
                    and ready.get("architecture") == "DIRECT_EDGE_TO_MAP_V1"
                    and ready.get("tail_device") == "cuda:0"
                    and ready.get("dense_label_map_on_radio") is False
                    and ready.get("object_records_on_radio") is False
                    and str(ready.get("direct_map_host")) == str(endpoint.host)
                    and int(ready.get("direct_map_port"))
                    == int(runtime["direct_map_ingest_port"])
                    and str(ready.get("evaluation_evidence_dir"))
                    == str(evidence_container),
                    "direct edge ready record identity/endpoint drift",
                )
                _ENDPOINT["edge_scratch"] = str(edge_scratch)
                return edge_scratch
            time.sleep(0.25)
        raise pinned.AdapterError("direct edge did not complete preload readiness")
    except Exception:
        _ENDPOINT["edge_scratch"] = str(edge_scratch)
        subprocess.run(
            ["sudo", "docker", "logs", "--tail", "120", "oai-perception-rx"],
            cwd=str(pinned.ROOT), check=False, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=30.0,
        )
        raise


def _direct_evidence_dir() -> Path:
    """Durable per-cell home for direct-architecture evidence."""

    root = Path(_ENDPOINT["attempt_dir"]) / "direct_edge_map"
    root.mkdir(parents=True, exist_ok=True)
    return root


_PINNED_STOP_TAIL = pinned.stop_tail


def stop_tail_preserving_edge_evidence() -> bool:
    """Rescue the edge's durable counters before the container goes away.

    The edge can only write inside its own mount, which lives in the cell's
    temporary directory and is deleted at teardown. The counters are the only
    record that separates uplink loss from edge-side refusal, and the
    publication ledger is the only record of the edge-side send instants, so
    both are copied into the attempt directory first. A copy failure is
    recorded, never fatal.
    """

    scratch = _ENDPOINT.get("edge_scratch")
    if scratch:
        evidence = _direct_evidence_dir()
        for name in (
            "direct_edge_counters.json",
            "ready.json",
            "direct_edge_publication.csv",
        ):
            source = Path(scratch) / name
            if not source.is_file():
                continue
            try:
                destination = evidence / (
                    "direct_edge_ready.json" if name == "ready.json" else name
                )
                destination.write_bytes(source.read_bytes())
            except OSError as exc:
                (evidence / "edge_evidence_copy_error.txt").write_text(
                    f"{name}: {exc}\n", encoding="utf-8"
                )
    return _PINNED_STOP_TAIL()


def install_direct_seams(campaign: Mapping[str, Any]) -> dict[str, Any]:
    """Rebind the four seams and resolve/audit the direct endpoint."""

    runtime = campaign["runtime"]
    endpoint = resolve_direct_map_endpoint(
        port=int(runtime["direct_map_ingest_port"])
    )
    _ENDPOINT["endpoint"] = endpoint
    pinned.start_map_process = start_direct_map_process
    pinned.start_live_edge = start_direct_live_edge
    pinned.stop_tail = stop_tail_preserving_edge_evidence
    pinned.LivePilotCellRuntime = direct_runtime_factory
    pinned_feedback.InstallFeedbackLedger = direct_ledger_factory
    pinned_feedback.FIELDS = tuple(
        __import__(
            "rl_agent.splitfusion_direct_edge_map_v1.ue_ledger",
            fromlist=["FIELDS"],
        ).FIELDS
    )
    return endpoint.as_dict()


def main(argv: list[str] | None = None) -> int:
    """Install the direct seams, then hand the whole cell to the pinned adapter."""

    values = list(sys.argv[1:] if argv is None else argv)
    args = pinned.build_parser().parse_args(values)
    if args.contract_check:
        return pinned.main(values)
    pinned.require(
        args.resolved_config is not None and args.attempt_dir is not None,
        "direct live run requires --resolved-config and --attempt-dir",
    )
    resolved = pinned.load_yaml(args.resolved_config.resolve())
    campaign = resolved["campaign"]
    cell = resolved["cell"]
    _ENDPOINT["cell_id"] = str(cell["cell_id"])
    _ENDPOINT["attempt_dir"] = str(args.attempt_dir.resolve())
    _ENDPOINT["config_relpath"] = str(campaign["runtime"]["direct_edge_config_relpath"])
    report = install_direct_seams(campaign)
    print(
        "[DIRECT] edge->map endpoint "
        f"{report['host']}:{report['port']} on {report['network_name']} "
        f"({report['bridge_interface']}, {report['subnet']}); "
        "object records never address the UE",
        flush=True,
    )
    return pinned.main(values)


if __name__ == "__main__":
    raise SystemExit(main())
