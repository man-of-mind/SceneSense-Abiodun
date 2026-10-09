#!/usr/bin/env python3
"""Add privileged exact-quality feedback to the direct live cell adapter.

This is an additive diagnostic wrapper.  The qualified direct edge/map path,
its terminal ledger, and all production outputs remain authoritative and are
not modified by a quality progress event.
"""

from __future__ import annotations

import json
import csv
import hashlib
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rl_agent.ue_route_b_split_cell_adapter_v1 as pinned  # noqa: E402
import rl_agent.ue_map_install_feedback_v1 as pinned_feedback  # noqa: E402
from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as direct  # noqa: E402
from rl_agent.splitfusion_direct_edge_map_v1 import protocol as map_protocol  # noqa: E402
from rl_agent.splitfusion_direct_edge_map_v1.live_pilot_runtime_direct_v1 import (  # noqa: E402
    DirectLivePilotCellRuntime,
)

from . import protocol as quality_protocol  # noqa: E402
from .gt_evidence import (  # noqa: E402
    write_object_ground_truth,
    write_semantic_ground_truth,
)
from .ledger import QualityFeedbackLedger  # noqa: E402


QUALITY_EDGE_MODULE = (
    "-m rl_agent.splitfusion_quality_feedback_probe_v1.edge_runtime_quality_v1"
)
QUALITY_CONFIG_SCHEMA = "splitfusion_privileged_quality_probe_config.v1"
QUALITY_REPORT_SCHEMA = "splitfusion_privileged_quality_edge_report.v1"
_CONTEXT: dict[str, Any] = {}
_ORIGINAL_SEED = pinned.seed_cell_edge_state
_DIRECT_STOP: Callable[[], bool] | None = None


class QualityAdapterError(RuntimeError):
    pass


class _UEPrepareProxy:
    def __init__(self, delegate: Any, runtime: "QualityDirectLiveRuntime") -> None:
        self._delegate = delegate
        self._runtime = runtime

    def prepare(self, *args: Any, **kwargs: Any) -> Any:
        self._runtime._mark_model_action_start()
        return self._delegate.prepare(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _TimestampingSender:
    def __init__(self, delegate: Any, runtime: "QualityDirectLiveRuntime") -> None:
        self._delegate = delegate
        self._runtime = runtime

    def sendto(self, *args: Any, **kwargs: Any) -> Any:
        self._runtime._mark_first_feature_send()
        return self._delegate.sendto(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class QualityDirectLiveRuntime(DirectLivePilotCellRuntime):
    """Dispatch compact quality messages to an independent nonterminal ledger."""

    def __init__(self, *, quality_ledger: QualityFeedbackLedger, **kwargs: Any) -> None:
        self.quality_ledger = quality_ledger
        self._quality_identity: dict[int, dict[str, Any]] = {}
        self._terminal_details: dict[int, dict[str, Any]] = {}
        self._sensor_ready_wall_ns: dict[int, int] = {}
        self._model_action_start_wall_ns: dict[int, int] = {}
        self._model_prepare_start_wall_ns: dict[int, int] = {}
        self._active_frame_id: int | None = None
        super().__init__(**kwargs)
        self.ue = _UEPrepareProxy(self.ue, self)
        self.sender = _TimestampingSender(self.sender, self)

    def register_sensor_ready(self, frame_id: int, *, observed_wall_ns: int) -> None:
        self._sensor_ready_wall_ns[int(frame_id)] = int(observed_wall_ns)

    def _mark_model_action_start(self) -> None:
        frame_id = self._active_frame_id
        if frame_id is None:
            raise QualityAdapterError("model action start lacks active frame")
        self._model_prepare_start_wall_ns.setdefault(frame_id, time.time_ns())

    def _mark_first_feature_send(self) -> None:
        frame_id = self._active_frame_id
        if frame_id is None:
            raise QualityAdapterError("feature send lacks active frame")
        with self.lock:
            metric = self.metrics.get(frame_id)
            if metric is None:
                raise QualityAdapterError("feature send preceded metric registration")
            metric.setdefault("first_feature_datagram_send_wall_ns", time.time_ns())

    def identity_for_frame(self, frame_id: int) -> dict[str, Any]:
        try:
            return dict(self._quality_identity[int(frame_id)])
        except KeyError as exc:
            raise QualityAdapterError(f"quality identity absent for frame {frame_id}") from exc

    def submit(self, **kwargs: Any) -> dict[str, Any]:
        frame_id = int(kwargs["frame_id"])
        selected = self.profile.action_id if kwargs.get("action_id") is None else int(kwargs["action_id"])
        profile = self.allowed_profiles[selected]
        identity = {
            "run_id": str(self.campaign["campaign_id"]),
            "cell_id": str(self.cell["cell_id"]),
            "stream_id": str(kwargs["stream_id"]),
            "frame_id": frame_id,
            "action_id": int(profile.action_id),
            "profile_id": str(profile.profile_id),
            "capture_timestamp_ns": int(kwargs["capture_timestamp_ns"]),
        }
        original_commit = kwargs.get("on_commit")

        def commit() -> None:
            if original_commit is not None:
                original_commit()
            self.quality_ledger.register(identity, registered_wall_ns=time.time_ns())
            self._quality_identity[frame_id] = dict(identity)

        kwargs["on_commit"] = commit
        self._active_frame_id = frame_id
        # This is the start of the selected action path: immediately before
        # _prepare_live_input constructs the seven-channel tensor.  The model
        # prepare boundary is stamped separately by _UEPrepareProxy.
        self._model_action_start_wall_ns.setdefault(frame_id, time.time_ns())
        try:
            return super().submit(**kwargs)
        finally:
            self._active_frame_id = None

    def _result_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                datagram, address = self.receiver.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            received_at = time.time()
            received_wall_ns = time.time_ns()
            self.control_counters.bump("control_datagrams_received")
            self.control_counters.bump("control_bytes_received", len(datagram))
            try:
                raw = json.loads(datagram.decode("utf-8"))
                if not isinstance(raw, dict):
                    raise QualityAdapterError("control datagram is not an object")
                if str(raw.get("s") or "") in {
                    quality_protocol.QUALITY_EVALUATED_ACK_SCHEMA,
                    quality_protocol.QUALITY_EVALUATION_FAILED_ACK_SCHEMA,
                }:
                    quality_protocol.validate(raw)
                    identity = quality_protocol.identity_dict(raw)
                    frame_id = int(identity["frame_id"])
                    metric = self.metrics.get(frame_id)
                    pinned.require(metric is not None, "quality ACK has no transmitted frame")
                    row = self.quality_ledger.record(
                        raw,
                        received_wall_ns=received_wall_ns,
                        message_bytes=len(datagram),
                        source_address=f"{address[0]}:{address[1]}",
                    )
                    quality = quality_protocol.quality_dict(raw)
                    timing = quality_protocol.timing_dict(raw)
                    with self.lock:
                        sensor_ready = self._sensor_ready_wall_ns.get(frame_id)
                        action_start = self._model_action_start_wall_ns.get(frame_id)
                        model_prepare_start = self._model_prepare_start_wall_ns.get(frame_id)
                        first_send = metric.get("first_feature_datagram_send_wall_ns")
                        metric.update(
                            {
                                "quality_feedback_event": row["event"],
                                "quality_ack_received_wall_ns": received_wall_ns,
                                "quality_ack_edge_to_ue_ms": row["ack_edge_to_ue_ms"],
                                "capture_to_quality_ack_ms": row["capture_to_quality_ack_ms"],
                                "quality_detail_sha256": row["detail_sha256"],
                                "quality_feedback": quality,
                                "quality_feedback_timing": timing,
                                "sensor_ready_to_quality_ack_ms": (
                                    (received_wall_ns - sensor_ready) / 1e6
                                    if sensor_ready is not None else None
                                ),
                                "model_action_start_to_quality_ack_ms": (
                                    (received_wall_ns - int(action_start)) / 1e6
                                    if action_start is not None else None
                                ),
                                "model_prepare_start_to_quality_ack_ms": (
                                    (received_wall_ns - int(model_prepare_start)) / 1e6
                                    if model_prepare_start is not None else None
                                ),
                                "first_feature_send_to_quality_ack_ms": (
                                    (received_wall_ns - int(first_send)) / 1e6
                                    if first_send is not None else None
                                ),
                            }
                        )
                    self.control_counters.bump("quality_progress_messages")
                    self.control_counters.bump("quality_progress_bytes", len(datagram))
                    continue

                message = map_protocol.decode(datagram)
                map_protocol.assert_no_object_records(message)
                message["_feedback_bytes"] = len(datagram)
                message["_source_address"] = f"{address[0]}:{address[1]}"
                schema = str(message.get("schema") or "")
                if schema == map_protocol.DIRECT_MAP_FEEDBACK_SCHEMA:
                    self.control_counters.bump("map_feedback_messages")
                elif schema == map_protocol.EDGE_TERMINAL_CONTROL_SCHEMA:
                    self.control_counters.bump("edge_terminal_messages")
                else:
                    raise QualityAdapterError(f"unknown control schema {schema!r}")
                frame_id = int(message["frame_id"])
                metric = self.metrics.get(frame_id)
                pinned.require(metric is not None, "control message has no transmitted frame")
                pinned.require(int(message["action_id"]) == int(metric["action_id"]), "control action drift")
                pinned.require(str(message.get("stream_id") or "") == str(metric["stream_id"]), "control stream drift")
                self.ledger.enqueue(message, received_at)
                terminal = bool(message.get("terminal"))
                with self.lock:
                    if terminal:
                        self._terminal_details[frame_id] = dict(message)
                        if frame_id in self._terminal_frames:
                            self.control_counters.bump("duplicate_terminal_suppressed")
                        else:
                            self._terminal_frames[frame_id] = str(message.get("outcome") or "")
                            self._published_frames.add(frame_id)
                            self.completed += 1
                    else:
                        self.control_counters.bump("late_nonterminal_messages")
                    metric.update(self._control_fields(message, {}, received_at))
            except Exception as exc:
                with self.lock:
                    self.errors.append(f"{type(exc).__name__}: {exc}")
                return

    def _reconcile_quality_eligibility(self) -> list[int]:
        missing_eligible: list[int] = []
        for identity in self.quality_ledger.pending_records():
            frame_id = int(identity["frame_id"])
            terminal = self._terminal_details.get(frame_id, {})
            outcome = str(terminal.get("outcome") or "")
            stage = str(terminal.get("stage") or "")
            if outcome in {
                map_protocol.OUTCOME_SUPERSEDED_PENDING,
                map_protocol.OUTCOME_STALE_BEFORE_EDGE,
                map_protocol.OUTCOME_TRANSPORT_INCOMPLETE,
                map_protocol.OUTCOME_FEEDBACK_TIMEOUT,
            }:
                self.quality_ledger.mark_not_eligible(identity, reason=outcome)
            elif outcome == map_protocol.OUTCOME_STALE_BEFORE_MAP and stage != "EDGE_BEFORE_PUBLICATION":
                self.quality_ledger.mark_not_eligible(
                    identity, reason=f"{outcome}:{stage or 'UNKNOWN_STAGE'}"
                )
            elif outcome:
                missing_eligible.append(frame_id)
            else:
                # No terminal is not silently called loss here.  The later
                # edge-report join must prove whether a final prediction existed.
                missing_eligible.append(frame_id)
        return missing_eligible

    def close(self) -> dict[str, Any]:
        deadline = time.monotonic() + 5.0
        while self.quality_ledger.summary()["pending"] and time.monotonic() < deadline:
            time.sleep(0.02)
        missing_eligible = self._reconcile_quality_eligibility()
        if missing_eligible:
            self.errors.append(
                "quality outcomes missing for possibly eligible frames: "
                + ",".join(str(value) for value in missing_eligible[:32])
            )
        report = super().close()
        self._write_policy_timing()
        report["quality_feedback"] = self.quality_ledger.summary()
        report["quality_missing_possibly_eligible_frames"] = missing_eligible
        self.quality_ledger.close()
        return report

    def _write_policy_timing(self) -> None:
        fields = (
            "run_id", "cell_id", "stream_id", "frame_id", "action_id",
            "profile_id", "capture_timestamp_ns", "sensor_ready_wall_ns",
            "model_action_start_wall_ns", "model_prepare_start_wall_ns",
            "first_feature_datagram_send_wall_ns",
            "model_ready_wall_ns", "final_prediction_ready_wall_ns",
            "gt_ready_wall_ns", "evaluation_enqueued_wall_ns",
            "evaluation_started_wall_ns", "evaluation_completed_wall_ns",
            "ack_emit_start_wall_ns", "quality_ack_received_wall_ns",
            "capture_to_quality_ack_ms", "sensor_ready_to_quality_ack_ms",
            "model_action_start_to_quality_ack_ms",
            "model_prepare_start_to_quality_ack_ms",
            "first_feature_send_to_quality_ack_ms", "quality_feedback_event",
            "quality_detail_sha256",
        )
        path = Path(self.attempt_dir) / "quality_policy_timing.csv"
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields))
            writer.writeheader()
            with self.lock:
                metrics = {key: dict(value) for key, value in self.metrics.items()}
            for frame_id, identity in sorted(self._quality_identity.items()):
                metric = metrics.get(frame_id, {})
                timing = dict(metric.get("quality_feedback_timing") or {})
                writer.writerow(
                    {
                        **identity,
                        "sensor_ready_wall_ns": self._sensor_ready_wall_ns.get(frame_id, ""),
                        "model_action_start_wall_ns": self._model_action_start_wall_ns.get(frame_id, ""),
                        "model_prepare_start_wall_ns": self._model_prepare_start_wall_ns.get(frame_id, ""),
                        "first_feature_datagram_send_wall_ns": metric.get("first_feature_datagram_send_wall_ns", ""),
                        **{name: timing.get(name, "") for name in quality_protocol.TIMING_FIELDS},
                        "quality_ack_received_wall_ns": metric.get("quality_ack_received_wall_ns", ""),
                        "capture_to_quality_ack_ms": metric.get("capture_to_quality_ack_ms", ""),
                        "sensor_ready_to_quality_ack_ms": metric.get("sensor_ready_to_quality_ack_ms", ""),
                        "model_action_start_to_quality_ack_ms": metric.get("model_action_start_to_quality_ack_ms", ""),
                        "model_prepare_start_to_quality_ack_ms": metric.get("model_prepare_start_to_quality_ack_ms", ""),
                        "first_feature_send_to_quality_ack_ms": metric.get("first_feature_send_to_quality_ack_ms", ""),
                        "quality_feedback_event": metric.get("quality_feedback_event", ""),
                        "quality_detail_sha256": metric.get("quality_detail_sha256", ""),
                    }
                )


def quality_runtime_factory(
    *, campaign: Mapping[str, Any], cell: Mapping[str, Any], attempt_dir: Path,
    map_host: str, map_port: int, evidence_dir: Path,
) -> QualityDirectLiveRuntime:
    del map_host, map_port
    map_ledger = direct.DirectTerminalLedger(
        output_csv=Path(attempt_dir) / "map_feedback.csv",
        experiment_id=str(campaign["campaign_id"]),
        cell_id=str(cell["cell_id"]),
    )
    compat = direct.CompatDirectLedger(map_ledger, profile_id=str(cell["profile_id"]))
    direct._PENDING_LEDGER["ledger"] = compat
    quality_ledger = QualityFeedbackLedger(Path(attempt_dir) / "quality_feedback.csv")
    return QualityDirectLiveRuntime(
        campaign=campaign,
        cell=cell,
        attempt_dir=Path(attempt_dir),
        evidence_dir=Path(evidence_dir),
        ue_control_port=int(campaign["runtime"]["ue_control_port"]),
        ledger=compat,
        quality_ledger=quality_ledger,
    )


class QualityFrozenWorld(pinned.FrozenWorld):
    __slots__ = ("frozen_frame_id",)

    def __init__(self, actors: Any, *, frozen_frame_id: int) -> None:
        super().__init__(actors)
        self.frozen_frame_id = int(frozen_frame_id)


class QualitySceneSnapshotSource(pinned.SceneSnapshotSource):
    def capture(self, world_snapshot: Any) -> QualityFrozenWorld:
        frozen = super().capture(world_snapshot)
        return QualityFrozenWorld(
            frozen._actors, frozen_frame_id=int(world_snapshot.frame)
        )


class QualityPassiveSplitCollector(pinned.PassiveSplitCollector):
    def __init__(self, **kwargs: Any) -> None:
        self._quality_scene_lock = threading.Lock()
        self._quality_scenes: dict[int, QualityFrozenWorld] = {}
        super().__init__(**kwargs)

    def on_world_tick(self, frame_id: int, route_tick: int) -> None:
        """Freeze actor state at the exact CARLA tick before work is queued.

        CARLA runs asynchronously during this bounded probe.  Asking for the
        current snapshot later in the preparation worker can therefore return
        frame N+1 (or newer).  Capture N here, at the tick boundary owned by
        the route runner, and let the evaluation thread consume that immutable
        scene later.
        """

        if (
            (self.qualification_capture_limit is None
             or self.sent < self.qualification_capture_limit)
            and (int(route_tick) - 1) % 2 == 0
        ):
            snapshot = self.world.get_snapshot()
            pinned.require(
                int(snapshot.frame) == int(frame_id),
                "owned world tick/snapshot frame drift",
            )
            pinned.require(
                self.scene_source is not None,
                "exact scene source is unavailable at the owned world tick",
            )
            frozen = self.scene_source.capture(snapshot)
            pinned.require(
                isinstance(frozen, QualityFrozenWorld)
                and int(frozen.frozen_frame_id) == int(frame_id),
                "exact scene snapshot identity drift",
            )
            with self._quality_scene_lock:
                self._quality_scenes[int(frame_id)] = frozen
                # This is only a defensive ceiling; normal evaluation pops
                # each entry and never approaches it in a 300-frame cell.
                while len(self._quality_scenes) > 512:
                    self._quality_scenes.pop(min(self._quality_scenes))
        super().on_world_tick(frame_id, route_tick)

    def _records_for(self, frame_id: int, timeout_s: float = 0.25) -> Any:
        records = super()._records_for(frame_id, timeout_s=timeout_s)
        if records is not None:
            self.live.register_sensor_ready(frame_id, observed_wall_ns=time.time_ns())
        return records

    def _semantic_for(self, frame_id: int, timeout_s: float = 0.25) -> Any | None:
        image = super()._semantic_for(frame_id, timeout_s=timeout_s)
        if image is None:
            return None
        pinned.require(int(image.frame) == int(frame_id), "semantic sensor/frame drift")
        identity = self.live.identity_for_frame(frame_id)
        write_semantic_ground_truth(
            self.edge_evidence_dir,
            identity=identity,
            frozen_carla_frame_id=int(image.frame),
            mask=pinned.semantic_gt_3class(image),
        )
        return image

    def _ground_truth(self, **kwargs: Any) -> list[dict[str, Any]]:
        frame_id = int(kwargs["frame_id"])
        # The qualified adapter's separate aligned/current diagnostic is
        # identified by its dedicated tracker. A missing source-scene ticket
        # must *not* fall through to live/current GT: source reward evidence
        # remains fail-closed and always consumes the exact owned-tick cache.
        aligned_current = (
            kwargs.get("world") is None
            and kwargs.get("stationary_tracker") is self.aligned_actor_tracker
        )
        if aligned_current:
            return super()._ground_truth(**kwargs)
        with self._quality_scene_lock:
            world = self._quality_scenes.pop(frame_id, None)
        if world is None:
            raise QualityAdapterError(
                f"object GT lacks exact owned-tick snapshot for frame {frame_id}"
            )
        if int(world.frozen_frame_id) != frame_id:
            raise QualityAdapterError("object GT snapshot/frame drift")
        exact_kwargs = dict(kwargs)
        exact_kwargs["world"] = world
        rows = super()._ground_truth(**exact_kwargs)
        write_object_ground_truth(
            self.edge_evidence_dir,
            identity=self.live.identity_for_frame(frame_id),
            frozen_carla_frame_id=world.frozen_frame_id,
            rows=rows,
        )
        return rows


def _seed_quality_config(campaign: Mapping[str, Any], edge_scratch: Path) -> None:
    _ORIGINAL_SEED(campaign, edge_scratch)
    cell = _CONTEXT["cell"]
    evidence_container = Path("/work/torch_cache") / pinned.EDGE_EVIDENCE_LEAF
    config = {
        "schema": QUALITY_CONFIG_SCHEMA,
        "run_id": str(campaign["campaign_id"]),
        "cell_id": str(cell["cell_id"]),
        "evidence_dir": str(evidence_container),
        "report_path": str(evidence_container / "quality_edge_report.json"),
        "ue_host": str(campaign["runtime"]["ue_bind_host"]),
        "ue_port": int(campaign["runtime"]["ue_control_port"]),
        "match_distance_m": float(campaign["measurement_contract"]["match_distance_m"]),
        "queue_depth": 64,
        "gt_timeout_s": 2.0,
        "parity_sample_limit": 8,
        "early_segmentation": True,
        "privileged_carla_ground_truth": True,
        "deployable_feedback": False,
    }
    path = Path(edge_scratch) / "quality_probe_config.json"
    path.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _CONTEXT["edge_scratch"] = str(edge_scratch)


def _quality_destination() -> Path:
    destination = Path(_CONTEXT["attempt_dir"]) / "direct_edge_map"
    destination.mkdir(parents=True, exist_ok=True)
    return destination / "quality_edge_report.json"


def _copy_and_validate_quality_report() -> bool:
    scratch = _CONTEXT.get("edge_scratch")
    if not scratch:
        return False
    source = Path(str(scratch)) / pinned.EDGE_EVIDENCE_LEAF / "quality_edge_report.json"
    if not source.is_file():
        return False
    destination = _quality_destination()
    shutil.copyfile(source, destination)
    report = json.loads(destination.read_text(encoding="utf-8"))
    if report.get("schema") != QUALITY_REPORT_SCHEMA:
        raise QualityAdapterError("quality edge report schema drift")
    if (
        str(report.get("run_id") or "") != str(_CONTEXT.get("campaign_id") or "")
        or str(report.get("cell_id") or "")
        != str((_CONTEXT.get("cell") or {}).get("cell_id") or "")
    ):
        raise QualityAdapterError("quality edge report run/cell identity drift")
    coordinator = dict(report.get("coordinator") or {})
    submitted = int(coordinator.get("submitted", -1))
    completed = int(coordinator.get("completed", -1))
    failed = int(coordinator.get("failed", -1))
    if submitted < 0 or completed < 0 or failed < 0:
        raise QualityAdapterError("quality coordinator counters are absent")
    if submitted != completed + failed:
        raise QualityAdapterError("quality coordinator accounting does not reconcile")
    if failed or coordinator.get("failures"):
        raise QualityAdapterError("one or more eligible quality evaluations failed")
    if any(
        int(coordinator.get(name, 0)) != 0
        for name in (
            "queue_overflow",
            "early_without_final",
            "unfinished_at_deadline",
            "parity_cancelled_at_deadline",
        )
    ) or bool(coordinator.get("worker_alive")):
        raise QualityAdapterError("quality coordinator close/drain gate failed")
    expected_parity = min(8, completed)
    if (
        int(coordinator.get("parity_submitted", -1)) != expected_parity
        or int(coordinator.get("parity_checked", -1)) != expected_parity
    ):
        raise QualityAdapterError("bounded production/candidate parity sample is incomplete")
    if report.get("tap_failures"):
        raise QualityAdapterError("early semantic tap reported a failure")
    expected_sources = {
        name: hashlib.sha256(
            Path(__file__).with_name(name).read_bytes()
        ).hexdigest()
        for name in ("scoring.py", "gt_evidence.py", "protocol.py", "coordinator.py")
    }
    if report.get("source_sha256") != expected_sources:
        raise QualityAdapterError("edge scorer/GT/protocol source binding drift")
    details = {str(row.get("sha256")): row for row in report.get("details", [])}
    if len(details) != completed or len(report.get("messages", [])) != completed:
        raise QualityAdapterError("quality message/detail population drift")
    for digest, row in details.items():
        unhashed = {key: value for key, value in row.items() if key != "sha256"}
        if quality_protocol.detail_digest(unhashed) != digest:
            raise QualityAdapterError("quality detail digest mismatch")
    message_identities: set[tuple[Any, ...]] = set()
    for message in report.get("messages", []):
        digest = str(message.get("detail_sha256") or "")
        if digest not in details:
            raise QualityAdapterError("quality ACK lacks durable detail hash join")
        if int(message.get("bytes", 0)) > quality_protocol.MAX_WIRE_BYTES:
            raise QualityAdapterError("quality ACK exceeded the one-datagram wire budget")
        identity = tuple(message.get("identity") or ())
        if len(identity) != len(quality_protocol.IDENTITY_FIELDS):
            raise QualityAdapterError("quality ACK durable identity layout drift")
        if identity in message_identities:
            raise QualityAdapterError("duplicate quality ACK in edge report")
        message_identities.add(identity)
    validations = list(report.get("serial_validations") or ())
    if len(validations) != expected_parity or any(
        row.get("exact_parity") is not True
        or row.get("early_production_mask_exact_equal") is not True
        for row in validations
    ):
        raise QualityAdapterError("production-mask/serial parity gate failed")
    return True


def preserve_quality_edge_evidence() -> bool:
    """Drain the probe, persist its report, but leave compose cleanup separate."""

    stopped = True
    if pinned.tail_running():
        completed = subprocess.run(
            ["sudo", "docker", "stop", "--time", "30", "oai-perception-rx"],
            cwd=str(pinned.ROOT), check=False, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=40.0,
        )
        stopped = completed.returncode == 0 and not pinned.tail_running()
    # The wrapper writes atomically in its signal/finally path.  Bound the
    # visibility wait before the host mount is later removed by compose-down.
    deadline = time.monotonic() + 5.0
    copied = False
    while time.monotonic() < deadline:
        copied = _copy_and_validate_quality_report()
        if copied:
            break
        time.sleep(0.05)
    return stopped and copied


def quality_stop_tail() -> bool:
    if _DIRECT_STOP is None:
        raise QualityAdapterError("quality stop called before seam installation")
    preserved = preserve_quality_edge_evidence()
    cleaned = bool(_DIRECT_STOP())
    return preserved and cleaned


def install_quality_seams(
    campaign: Mapping[str, Any], *, cell: Mapping[str, Any], attempt_dir: Path
) -> dict[str, Any]:
    global _DIRECT_STOP
    _CONTEXT.clear()
    _CONTEXT.update(
        {
            "campaign_id": str(campaign["campaign_id"]),
            "cell": dict(cell),
            "attempt_dir": str(Path(attempt_dir).resolve()),
        }
    )
    direct._ENDPOINT["cell_id"] = str(cell["cell_id"])
    direct._ENDPOINT["attempt_dir"] = str(Path(attempt_dir).resolve())
    direct._ENDPOINT["config_relpath"] = str(
        campaign["runtime"]["direct_edge_config_relpath"]
    )
    endpoint = direct.install_direct_seams(campaign)
    direct.DIRECT_EDGE_MODULE = QUALITY_EDGE_MODULE
    _DIRECT_STOP = pinned.stop_tail
    pinned.seed_cell_edge_state = _seed_quality_config
    pinned.stop_tail = quality_stop_tail
    pinned.LivePilotCellRuntime = quality_runtime_factory
    pinned.SceneSnapshotSource = QualitySceneSnapshotSource
    pinned.PassiveSplitCollector = QualityPassiveSplitCollector
    # Map-terminal compatibility remains exactly the direct adapter's factory.
    pinned_feedback.InstallFeedbackLedger = direct.direct_ledger_factory
    return endpoint


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    args = pinned.build_parser().parse_args(values)
    if args.contract_check:
        return pinned.main(values)
    pinned.require(
        args.resolved_config is not None and args.attempt_dir is not None,
        "quality live run requires --resolved-config and --attempt-dir",
    )
    resolved = pinned.load_yaml(args.resolved_config.resolve())
    campaign, cell = resolved["campaign"], resolved["cell"]
    direct._ENDPOINT["cell_id"] = str(cell["cell_id"])
    direct._ENDPOINT["attempt_dir"] = str(args.attempt_dir.resolve())
    direct._ENDPOINT["config_relpath"] = str(
        campaign["runtime"]["direct_edge_config_relpath"]
    )
    install_quality_seams(campaign, cell=cell, attempt_dir=args.attempt_dir)
    return pinned.main(values)


if __name__ == "__main__":
    raise SystemExit(main())
