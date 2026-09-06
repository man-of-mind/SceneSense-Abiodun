"""Qualified SFD1-v2 live-pilot UE bridge and isolated edge service.

The UE owns only the frozen front/ranker/encoders. The separately started
``oai-perception-rx`` service owns only the frozen tail/decoders. Both
directions use the existing production ``!IHH`` fragmentation header; neither
direction adds feature compression beyond the mandatory inner zstd level 1.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import socket
import threading
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch

from phase2_map_sharing.transport import ChunkReassembler, chunk_payload
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import (
    ae_phase11b_gpu_qualification as phase11b,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    guards,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import (
    load_frozen_perception,
)

from .context_tail import ContextualFrozenP025TailAdapter
from .edge_runtime import PreloadedSplitEdgeRuntime
from .envelope import unpack_envelope
from .frame_context import StaticCameraRegistry, build_frame_context_v1
from .registry import SplitActionRegistry
from .transport import ProductionSplitCodec
from .ue_runtime import PreloadedSplitUERuntime


class LivePilotRuntimeError(RuntimeError):
    """The live dispatch, result identity, or map handoff was invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LivePilotRuntimeError(message)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), f"configuration is not an object: {path}")
    return value


def _finite_tree(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        _require(bool(torch.isfinite(value).all()), "non-finite frozen-tail output")
    elif isinstance(value, Mapping):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _finite_tree(child)


def _trace_ns(trace: Any) -> dict[str, int]:
    return {
        str(boundary.name): int(boundary.finished_monotonic_ns - boundary.started_monotonic_ns)
        for boundary in trace.boundaries
    }


class _Ledger:
    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    def bump(self, name: str) -> None:
        with self._lock:
            self._counts[str(name)] += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


class _Front:
    def __init__(self, model: torch.nn.Module, ledger: _Ledger) -> None:
        self._model, self._ledger = model, ledger

    def __call__(self, input_7ch: torch.Tensor) -> torch.Tensor:
        self._ledger.bump("front")
        return self._model.encode_front(input_7ch)


class _Ranker:
    def __init__(self, model: torch.nn.Module, ledger: _Ledger) -> None:
        self._model, self._ledger = model, ledger

    def score_cells(self, c2: torch.Tensor) -> torch.Tensor:
        self._ledger.bump("ranker")
        return self._model.score_cells(c2)


class _AE:
    def __init__(self, family: str, model: Any, ledger: _Ledger) -> None:
        self.family = str(family)
        self.family_id = int(model.family_id)
        self.bottleneck = int(model.bottleneck)
        self.routing_tag = int(model.routing_tag)
        self._model, self._ledger = model, ledger

    def encode(self, c2: torch.Tensor) -> torch.Tensor:
        self._ledger.bump(f"ae_encoder_{self.family}")
        return self._model.encode(c2)

    def decode(self, latent: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        self._ledger.bump(f"ae_decoder_{self.family}")
        return self._model.decode(latent, keep_mask)


def _preload_ue(device: torch.device) -> tuple[PreloadedSplitUERuntime, _Ledger, list[Any]]:
    registry = SplitActionRegistry.from_runtime_binding()
    model, _base, _binding = load_frozen_perception(device)
    phase11b.common.freeze(model)
    ranker = phase11b._load_ranker(device)
    autoencoders = {}
    for family, _family_id, bottleneck in phase11b.FAMILIES:
        if bottleneck is None:
            continue
        item = phase11b.FROZEN_INPUTS[family]
        payload = torch.load(
            phase11b._repository_path(item["path"]),
            map_location="cpu", weights_only=False,
        )
        autoencoders[family] = phase11b._load_selected_autoencoder(
            family, bottleneck, item, payload, device,
        )
        del payload
    guards.require_frozen_perception([model, ranker, *autoencoders.values()])
    guards.require_eval_mode([model, ranker, *autoencoders.values()])
    ledger = _Ledger()
    wrapped = {family: _AE(family, ae, ledger) for family, ae in autoencoders.items()}
    runtime = PreloadedSplitUERuntime(
        registry, front=_Front(model, ledger), ranker=_Ranker(ranker, ledger),
        ae_encoders=wrapped, device=device, codec=ProductionSplitCodec(),
        prepare_modules=False, startup_model_load_operations=5,
        startup_model_construction_operations=5,
    )
    return runtime, ledger, [model, ranker, *autoencoders.values()]


def _preload_edge(device: torch.device) -> tuple[PreloadedSplitEdgeRuntime, ContextualFrozenP025TailAdapter, _Ledger, list[Any]]:
    registry = SplitActionRegistry.from_runtime_binding()
    model, base, _binding = load_frozen_perception(device)
    phase11b.common.freeze(model)
    autoencoders = {}
    for family, _family_id, bottleneck in phase11b.FAMILIES:
        if bottleneck is None:
            continue
        item = phase11b.FROZEN_INPUTS[family]
        payload = torch.load(
            phase11b._repository_path(item["path"]),
            map_location="cpu", weights_only=False,
        )
        autoencoders[family] = phase11b._load_selected_autoencoder(
            family, bottleneck, item, payload, device,
        )
        del payload
    guards.require_frozen_perception([model, *autoencoders.values()])
    guards.require_eval_mode([model, *autoencoders.values()])
    ledger = _Ledger()
    wrapped = {family: _AE(family, ae, ledger) for family, ae in autoencoders.items()}
    tail = ContextualFrozenP025TailAdapter(
        model=model, base=base, camera_registry=StaticCameraRegistry.audited(),
        device=device, ledger=ledger,
    )
    runtime = PreloadedSplitEdgeRuntime(
        registry, frozen_p025_tail=tail, ae_decoders=wrapped, tail_device=device,
        codec=ProductionSplitCodec(), output_serializer=tail.serialize,
        prepare_modules=False, startup_model_load_operations=4,
        startup_model_construction_operations=4, camera_registry=StaticCameraRegistry.audited(),
        require_frame_context=True,
    )
    return runtime, tail, ledger, [model, *autoencoders.values()]


def _prepare_live_input(frame_bgr: np.ndarray, radar_tensor: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (768, 448), interpolation=cv2.INTER_LINEAR)
    rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
    rgb_tensor = rgb_tensor.to(device=device, dtype=torch.float32).div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    radar = torch.from_numpy(np.ascontiguousarray(np.stack([
        cv2.resize(channel, (768, 448), interpolation=(cv2.INTER_NEAREST if index == 0 else cv2.INTER_LINEAR))
        for index, channel in enumerate(radar_tensor)
    ], axis=0))).unsqueeze(0)
    return torch.cat(((rgb_tensor - mean) / std, radar.to(device=device, dtype=torch.float32)), dim=1)


class LivePilotCellRuntime:
    """The UE half of one cell; its peer is the separately preloaded edge service."""

    def __init__(self, *, campaign: Mapping[str, Any], cell: Mapping[str, Any], attempt_dir: Path,
                 map_host: str, map_port: int, evidence_dir: Path) -> None:
        runtime = campaign["runtime"]
        _require(int(runtime["sfd1_protocol_version"]) == 2 and bool(runtime["frame_context_required"]), "SFD1 v2 frame context is required")
        _require(runtime["udp_fragment_header"] == "!IHH" and not bool(runtime["retransmission"]), "fragment contract drift")
        _require(bool(runtime["no_secondary_feature_compression"]), "secondary feature compression forbidden")
        _require(torch.cuda.is_available() and torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5090", "expected CUDA device unavailable")
        self.campaign, self.cell, self.device = campaign, cell, torch.device("cuda:0")
        self.registry = SplitActionRegistry.from_runtime_binding()
        self.profile = self.registry.resolve(int(cell["action_id"]))
        _require(self.profile.profile_id == str(cell["profile_id"]), "cell/catalog identity mismatch")
        qualification = campaign.get("_qualification")
        self.allowed_action_ids = (
            tuple(int(value) for value in qualification["action_ids"])
            if isinstance(qualification, Mapping)
            else (self.profile.action_id,)
        )
        _require(
            len(self.allowed_action_ids) == len(set(self.allowed_action_ids))
            and all(0 <= value < 72 for value in self.allowed_action_ids),
            "live action allowlist is invalid",
        )
        self.allowed_profiles = {
            action_id: self.registry.resolve(action_id)
            for action_id in self.allowed_action_ids
        }
        self.ue, self._ledger, self._models = _preload_ue(self.device)
        self.attempt_dir, self.evidence_dir = Path(attempt_dir), Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        request = int(runtime["socket_buffer_request_bytes"])
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, request)
        self.sender.bind((str(runtime["ue_bind_host"]), 0))
        self.receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, request)
        self.receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.receiver.bind((str(runtime["ue_bind_host"]), int(runtime["camera_result_port"])))
        self.receiver.settimeout(0.1)
        self.map_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.remote = (str(runtime["edge_remote_host"]), int(runtime["edge_receive_port"]))
        self.map_remote, self.chunk_bytes = (str(map_host), int(map_port)), int(runtime["udp_chunk_bytes"])
        self.reassembler = ChunkReassembler(timeout_s=2.0, max_chunks=4096)
        self.stop_event, self.lock = threading.Event(), threading.Lock()
        self.metrics: dict[int, dict[str, Any]] = {}
        self.errors: list[str] = []
        self.sent = self.completed = 0
        self.thread = threading.Thread(target=self._result_loop, name="splitfusion-sfd1-result", daemon=True)
        self.thread.start()

    def socket_buffer_report(self) -> dict[str, int]:
        return {"requested_bytes": int(self.campaign["runtime"]["socket_buffer_request_bytes"]),
                "ue_send_reported_bytes": int(self.sender.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)),
                "ue_result_receive_reported_bytes": int(self.receiver.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))}

    def submit(self, *, frame_bgr: np.ndarray, radar_tensor: np.ndarray, frame_id: int,
               capture_timestamp_ns: int, ego_pose: tuple[float, float, float, float, float, float],
               stream_id: str, carla_timestamp: float, capture_id: str,
               action_id: int | None = None) -> dict[str, Any]:
        _require(not self.errors, self.errors[0] if self.errors else "edge service failed")
        _require(self.thread.is_alive(), "result service exited")
        selected_action = self.profile.action_id if action_id is None else int(action_id)
        _require(selected_action in self.allowed_profiles, "action is outside the live allowlist")
        profile = self.allowed_profiles[selected_action]
        started = time.perf_counter_ns()
        input_7ch = _prepare_live_input(frame_bgr, radar_tensor, self.device)
        context = build_frame_context_v1(
            stream_id=stream_id, frame_id=int(frame_id), sequence_id=int(frame_id),
            capture_timestamp_ns=int(capture_timestamp_ns), ego_world_x=ego_pose[0], ego_world_y=ego_pose[1],
            ego_world_z=ego_pose[2], ego_world_pitch=ego_pose[3], ego_world_yaw=ego_pose[4], ego_world_roll=ego_pose[5],
        )
        with torch.inference_mode():
            prepared = self.ue.prepare(profile.action_id, input_7ch, sequence_id=context.sequence_id,
                                       capture_timestamp_ns=context.capture_timestamp_ns, frame_context=context)
        chunks = chunk_payload(prepared.wire_bytes, message_id=int(frame_id), chunk_bytes=self.chunk_bytes)
        sent_started = time.perf_counter_ns()
        with self.lock:
            self.metrics[int(frame_id)] = {
                "capture_id": str(capture_id), "frame_id": int(frame_id), "stream_id": str(stream_id),
                "action_id": profile.action_id, "profile_id": profile.profile_id,
                "model_family": profile.family, "quantizer": profile.quantizer,
                "q_e4": profile.q_e4, "routing_tag": profile.routing_tag,
                "carla_timestamp": float(carla_timestamp), "capture_started_ns": started,
                "ue_prepare_finished_ns": sent_started,
                "scientific_inner_bytes": int(prepared.inner_payload_bytes),
                "sfd1_overhead_bytes": int(prepared.outer_envelope_bytes), "sfd1_bytes": int(prepared.total_transmitted_bytes),
                "datagrams": len(chunks), "udp_application_bytes": sum(map(len, chunks)),
                "estimated_wire_bytes": sum(len(chunk) + 28 for chunk in chunks), "front_timing_ns": _trace_ns(prepared.timing),
            }
        for chunk in chunks:
            self.sender.sendto(chunk, self.remote)
        sent_finished = time.perf_counter_ns()
        with self.lock:
            self.metrics[int(frame_id)]["send_finished_ns"] = sent_finished
            self.sent += 1
        return {"front_ms": (sent_started - started) / 1e6, "payload_bytes": len(prepared.wire_bytes),
                "payload_bytes_uncompressed": prepared.inner_payload_bytes, "payload_chunks": len(chunks)}

    def _result_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                datagram, address = self.receiver.recvfrom(65535)
            except socket.timeout:
                self.reassembler.expire(time.monotonic())
                continue
            except OSError:
                return
            received_ns = time.perf_counter_ns()
            try:
                complete = self.reassembler.ingest(str(address), datagram, received_at_s=time.monotonic())
                if complete is None:
                    continue
                value = json.loads(complete.payload.decode("utf-8"))
                _require(value.get("schema") == "splitfusion_edge_result.v1", "edge result schema drift")
                _require(int(value["frame_id"]) == complete.message_id, "result chunk/frame identity drift")
                metric = self.metrics.get(int(value["frame_id"]))
                _require(metric is not None, "edge result has no transmitted UE frame")
                _require(int(value["action_id"]) == int(metric["action_id"]), "edge action identity drift")
                _require(str(value["profile_id"]) == str(metric["profile_id"]), "edge profile identity drift")
                _require(str(value["stream_id"]) == str(metric["stream_id"]), "edge stream identity drift")
                labels = np.frombuffer(base64.b64decode(value["semantic_labels_b64"], validate=True), dtype=np.uint8)
                shape = tuple(int(x) for x in value["semantic_labels_shape"])
                _require(shape == (720, 1280) and labels.size == 720 * 1280, "edge segmentation shape drift")
                evidence = self.evidence_dir / f"{hashlib.sha256(str(value['stream_id']).encode()).hexdigest()[:16]}_{int(value['frame_id'])}.npy"
                with evidence.open("xb") as handle:
                    np.save(handle, labels.reshape(shape), allow_pickle=False)
                published = {
                    "schema": "fusion_object_spatial_map.v1", "stream_id": value["stream_id"],
                    "frame_id": int(value["frame_id"]), "capture_id": metric["capture_id"],
                    "capture_timestamp": int(value["capture_timestamp_ns"]) / 1_000_000_000.0,
                    "action_id": str(value["action_id"]), "carla_timestamp": metric["carla_timestamp"],
                    "objects": value["records"], "segmentation": {"available": True},
                    "timing": {"t_edge_recv_perf": float(value["edge_received_ns"]) / 1e9,
                               "t_tail_done_perf": float(value["tail_finished_ns"]) / 1e9,
                               "t_map_publish_perf": time.perf_counter()},
                }
                self.map_socket.sendto(zlib.compress(json.dumps(published, allow_nan=False, separators=(",", ":")).encode("utf-8"), level=1), self.map_remote)
                with self.lock:
                    metric.update({"edge_result_received_ns": received_ns, "edge_timing_ns": value["edge_timing_ns"],
                                   "duplicate_datagrams": int(complete.duplicate_chunks), "decoded": True, "finite": bool(value["finite"]),
                                   "decoder_identity": str(value["decoder_identity"]), "edge_result_datagrams": int(complete.chunk_count),
                                   "feature_received_datagrams": int(value["feature_received_datagrams"]),
                                   "feature_duplicate_datagrams": int(value["feature_duplicate_datagrams"]),
                                   "reconstructed_device": str(value["reconstructed_device"]),
                                   "frame_context_valid": bool(value["frame_context_valid"]),
                                   "camera_pose_reconstruct_ns": int(value["camera_pose_reconstruct_ns"]),
                                   "finite_output_tensor_count": int(value["finite_output_tensor_count"]),
                                   "service_record_count": int(value["service_record_count"]),
                                   "edge_call_ledger": dict(value["edge_call_ledger"]),
                                   "edge_counters": dict(value["edge_counters"])})
                    self.completed += 1
            except Exception as exc:
                with self.lock:
                    self.errors.append(f"{type(exc).__name__}: {exc}")
                return

    def take_metric(self, frame_id: int) -> dict[str, Any] | None:
        with self.lock:
            value = self.metrics.get(int(frame_id))
            return dict(value) if value else None

    def close(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=10.0)
        buffers = self.socket_buffer_report()
        for item in (self.sender, self.receiver, self.map_socket):
            try:
                item.close()
            except OSError:
                pass
        return {"sent": self.sent, "edge_completed": self.completed, "result_thread_alive": self.thread.is_alive(),
                "errors": list(self.errors), "socket_buffers": buffers, "call_ledger": self._ledger.snapshot(),
                "ue_counters": self.ue.counters.__dict__}


def run_edge_service(*, config_path: Path, action_id: int, allowed_action_ids: tuple[int, ...],
                     ready_file: Path, edge_port: int, result_host: str, result_port: int) -> int:
    campaign = _load_json(config_path)
    runtime = campaign["runtime"]
    _require(int(runtime["sfd1_protocol_version"]) == 2 and runtime["udp_fragment_header"] == "!IHH", "edge protocol binding drift")
    _require(torch.cuda.is_available() and torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5090", "edge CUDA device unavailable")
    device = torch.device("cuda:0")
    registry = SplitActionRegistry.from_runtime_binding()
    _require(
        allowed_action_ids and len(allowed_action_ids) == len(set(allowed_action_ids)),
        "edge action allowlist is empty or duplicated",
    )
    profiles = {value: registry.resolve(value) for value in allowed_action_ids}
    _require(int(action_id) in profiles, "edge fixed action is outside its allowlist")
    edge, tail, ledger, models = _preload_edge(device)
    request = int(runtime["socket_buffer_request_bytes"])
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, request)
    receiver.bind(("0.0.0.0", int(edge_port)))
    receiver.settimeout(0.25)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, request)
    reassembler = ChunkReassembler(timeout_s=2.0, max_chunks=4096)
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    with ready_file.open("x", encoding="utf-8") as handle:
        json.dump({"schema": "splitfusion_live_edge_ready.v1", "action_id": int(action_id),
                   "allowed_action_ids": list(allowed_action_ids),
                   "profiles": {str(key): value.profile_id for key, value in profiles.items()},
                   "tail_device": str(edge.tail_device),
                   "state_root": str(ready_file.parent), "state_root_writable": True,
                   "edge_receive_reported_bytes": receiver.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF),
                   "edge_send_reported_bytes": sender.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)}, handle, sort_keys=True)
    try:
        while True:
            try:
                datagram, address = receiver.recvfrom(65535)
            except socket.timeout:
                reassembler.expire(time.monotonic())
                continue
            complete = reassembler.ingest(str(address), datagram, received_at_s=time.monotonic())
            if complete is None:
                continue
            edge_received_ns = time.perf_counter_ns()
            outer = unpack_envelope(complete.payload)
            _require(outer.action_id in profiles, "received action is outside the edge allowlist")
            profile = profiles[outer.action_id]
            result = edge.process(complete.payload, transmitted_action_id=outer.action_id)
            _finite_tree(result.perception)
            snapshot = tail.take_snapshot()
            context = result.metadata.frame_context
            _require(context is not None and context.frame_id == complete.message_id, "SFD1/chunk frame identity drift")
            labels = snapshot.semantic_labels.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy()
            value = {"schema": "splitfusion_edge_result.v1", "action_id": profile.action_id,
                     "profile_id": profile.profile_id, "decoder_identity": profile.decoder_identity,
                     "stream_id": context.stream_id, "frame_id": context.frame_id,
                     "capture_timestamp_ns": context.capture_timestamp_ns, "finite": True,
                     "frame_context_valid": True, "reconstructed_device": str(edge.tail_device),
                     "camera_pose_reconstruct_ns": int(snapshot.camera_pose_reconstruct_ns),
                     "finite_output_tensor_count": int(snapshot.output_tensor_count),
                     "service_record_count": len(snapshot.records or ()),
                     "feature_received_datagrams": int(complete.chunk_count),
                     "feature_duplicate_datagrams": int(complete.duplicate_chunks),
                     "edge_call_ledger": ledger.snapshot(), "edge_counters": edge.counters.__dict__,
                     "edge_received_ns": edge_received_ns, "tail_finished_ns": time.perf_counter_ns(),
                     "edge_timing_ns": _trace_ns(result.timing), "records": list(snapshot.records or ()),
                     "semantic_labels_shape": list(labels.shape),
                     "semantic_labels_b64": base64.b64encode(labels.tobytes()).decode("ascii")}
            payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
            for chunk in chunk_payload(payload, message_id=context.frame_id, chunk_bytes=int(runtime["udp_chunk_bytes"])):
                sender.sendto(chunk, (str(result_host), int(result_port)))
    finally:
        receiver.close()
        sender.close()
        del models, ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="qualified SFD1-v2 live edge")
    parser.add_argument("--edge", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--action-id", type=int)
    parser.add_argument("--allowed-action-ids")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--edge-port", type=int, default=51002)
    parser.add_argument("--result-host", default="10.0.0.2")
    parser.add_argument("--result-port", type=int, default=51004)
    args, _ignored = parser.parse_known_args(argv)
    _require(args.edge and args.config and args.action_id is not None and args.ready_file, "edge mode and all qualified bindings are required")
    allowed_action_ids = tuple(
        int(value)
        for value in str(args.allowed_action_ids or args.action_id).split(",")
    )
    return run_edge_service(config_path=args.config.resolve(strict=True), action_id=int(args.action_id),
                            allowed_action_ids=allowed_action_ids,
                            ready_file=args.ready_file, edge_port=int(args.edge_port),
                            result_host=str(args.result_host), result_port=int(args.result_port))


if __name__ == "__main__":
    raise SystemExit(main())
