"""One-shot four-action CUDA plus localhost-UDP Phase-13B qualification.

This runner is deliberately not the 36-profile latency campaign. It loads all
frozen objects once, admits one Phase-11B-bound fit frame, exercises four
catalog action IDs through the committed Phase-13A UE/edge API and a raw UDP
chunk/reassembly path, then compares each result with the corresponding
existing direct codec and frozen-tail path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import queue
import socket
import subprocess
import threading
import time
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping
from unittest import mock

import torch
import torch.nn.functional as F

from phase2_map_sharing.transport import CHUNK_HEADER, ChunkReassembler, chunk_payload
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import (
    ae_phase11b_gpu_qualification as phase11b,
    ae_uint8_transport,
    lowbit_dispatch,
    lowbit_transport,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_ae_v1.ae_family_dispatch import (
    PreloadedAeDecoders,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_ae_v1.ae_gpu_qualification import (
    require_tree_finite,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    continuous_q,
    guards,
    uint8_codec,
    uint8_zstd_transport,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import (
    load_frozen_perception,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.zstd_transport import (
    ZstdWireCodec,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1.runtime import (
    apply_p025_service_policy,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.runtime import (
    combined_records,
)

from .edge_runtime import PreloadedSplitEdgeRuntime
from .envelope import HEADER_BYTES, PROTOCOL_VERSION, unpack_envelope
from .registry import ActionProfile, SplitActionRegistry, sha256_file
from .timing import EDGE_STAGES, UE_STAGES, TimingTrace
from .transport import DecodedC2, InspectedInnerPayload, ProductionSplitCodec
from .ue_runtime import DispatchMetadata, PreloadedSplitUERuntime


EXECUTE_TOKEN = "SPLITFUSION_LIVE_DISPATCH_PHASE13B_QUALIFICATION"
SCHEMA = "scenesense.splitfusion_live_dispatch_phase13b_qualification.v1"
TERMINAL = "SPLITFUSION_LIVE_DISPATCH_FOUR_ACTION_GPU_LOOPBACK_QUALIFIED"
OUTPUT_RELPATH = (
    "experiments/splitfusion_live_dispatch_v1/"
    "20260904_phase13b_four_action_gpu_loopback_qualification"
)
ACTION_IDS = (0, 20, 46, 71)
DEVICE_NAME = "NVIDIA GeForce RTX 5090"
CHUNK_BYTES = 12_500
SOCKET_BUFFER_REQUEST_BYTES = 8 * 1024 * 1024
EXPECTED_DIRTY_PATHS = frozenset(
    {
        "OAI/openairinterface5g",
        (
            "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
            "lraspp_to_splitfusion_fcos_report_v1/FULL_TECHNICAL_REPORT_AVO_V2.md"
        ),
    }
)
PHASE13A_TERMINAL = "SPLITFUSION_LIVE_DISPATCH_IMPLEMENTATION_READY_FOR_REVIEW"
PHASE13A_COMMIT = "219bd004ef74550128228062768dac7c6fa6525b"
PHASE13B_IMPLEMENTATION_COMMIT = "71cfc1951c3bd5baca897eaf2483029f5ca9f0c2"
PHASE11B_EVIDENCE = {
    "path": (
        "experiments/splitfusion_fcos_ae_v1/"
        "20260903_phase11b_lowbit_gpu_qualification/"
        "phase11b_lowbit_gpu_qualification.json"
    ),
    "sha256": "379aa07148e3e47384cfbebbe0ede5990c07f11b8a4bdef056d6a533cee5fc01",
}
PHASE11B_TERMINAL = {
    "path": (
        "experiments/splitfusion_fcos_ae_v1/"
        "20260903_phase11b_lowbit_gpu_qualification/"
        "SPLITFUSION_LOWBIT_PHASE11B_GPU_QUALIFIED"
    ),
    "sha256": "83f41560a3327c4207834f5725e5e313ceb6b3e0f9e22ea1f8c37b6dcf0b56e2",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_path(relative: str) -> Path:
    root = _root().resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"path escapes repository: {relative}") from exc
    return path


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    prefix = f"{str(tensor.dtype)}:{list(tensor.shape)}:".encode("ascii")
    return _digest_bytes(prefix + tensor.numpy().tobytes(order="C"))


def _mapping_digest(value: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(value):
        tensor = value[name]
        _require(isinstance(tensor, torch.Tensor), f"non-tensor output field: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_digest(tensor).encode("ascii"))
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> str:
    staging = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with staging.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()
    return sha256_file(path)


def _atomic_json(path: Path, value: Any) -> str:
    return _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=_root(),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.rstrip("\r\n")


def _verify_git_state() -> dict[str, Any]:
    head = _git_output("rev-parse", "HEAD")
    parent = _git_output("rev-parse", "HEAD^")
    _require(
        parent == PHASE13B_IMPLEMENTATION_COMMIT,
        "Phase-13B repair parent is "
        f"{parent}, expected {PHASE13B_IMPLEMENTATION_COMMIT}",
    )
    phase13a = _git_output("rev-parse", "HEAD^^")
    _require(
        phase13a == PHASE13A_COMMIT,
        f"Phase-13B implementation base is {phase13a}, expected {PHASE13A_COMMIT}",
    )
    lines = _git_output(
        "status", "--porcelain=v1", "--untracked-files=all"
    ).splitlines()
    paths = []
    for line in lines:
        _require(len(line) >= 4, f"unparseable git status line: {line!r}")
        path = line[3:]
        _require(" -> " not in path, "renamed dirty paths are not authorized")
        paths.append(path)
    _require(
        frozenset(paths) == EXPECTED_DIRTY_PATHS and len(paths) == len(EXPECTED_DIRTY_PATHS),
        f"unexpected dirty paths: observed={sorted(paths)} expected={sorted(EXPECTED_DIRTY_PATHS)}",
    )
    source = _repo_path(
        "rl_agent/splitfusion_live_dispatch_v1/phase13b_qualification.py"
    )
    return {
        "head": head,
        "phase13b_implementation_commit": parent,
        "phase13a_commit": phase13a,
        "implementation_source": str(source.relative_to(_root())),
        "implementation_source_sha256": sha256_file(source),
        "expected_user_owned_dirty_paths": sorted(paths),
    }


def _verify_phase13a_terminal() -> dict[str, str]:
    manifest = _repo_path("rl_agent/splitfusion_live_dispatch_v1/runtime_binding.json")
    terminal = _repo_path(
        "rl_agent/splitfusion_live_dispatch_v1/"
        "SPLITFUSION_LIVE_DISPATCH_IMPLEMENTATION_READY_FOR_REVIEW"
    )
    manifest_hash = sha256_file(manifest)
    _require(
        terminal.read_text(encoding="utf-8").strip()
        == f"{PHASE13A_TERMINAL} {manifest_hash}",
        "Phase-13A runtime terminal does not bind the live manifest",
    )
    document = json.loads(manifest.read_text(encoding="utf-8"))
    _require(
        document.get("status") == "IMPLEMENTATION_READY_NOT_GPU_OR_LIVE_QUALIFIED",
        "Phase-13A runtime binding status drift",
    )
    return {
        "runtime_binding_path": str(manifest.relative_to(_root())),
        "runtime_binding_sha256": manifest_hash,
        "terminal_path": str(terminal.relative_to(_root())),
        "terminal_sha256": sha256_file(terminal),
    }


def _phase11b_sample_binding() -> tuple[dict[str, Any], dict[str, str]]:
    evidence_path = _repo_path(PHASE11B_EVIDENCE["path"])
    terminal_path = _repo_path(PHASE11B_TERMINAL["path"])
    _require(
        sha256_file(evidence_path) == PHASE11B_EVIDENCE["sha256"],
        "Phase-11B qualification evidence hash drift",
    )
    _require(
        sha256_file(terminal_path) == PHASE11B_TERMINAL["sha256"],
        "Phase-11B qualification terminal hash drift",
    )
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    _require(
        document.get("schema") == "splitfusion_fcos_phase11b_lowbit_gpu_qualification_v1"
        and document.get("terminal") == "SPLITFUSION_LOWBIT_PHASE11B_GPU_QUALIFIED",
        "Phase-11B evidence schema or terminal drift",
    )
    sample = document.get("fit_frame")
    scope = document.get("scope")
    _require(isinstance(sample, dict) and isinstance(scope, dict), "Phase-11B sample binding missing")
    _require(sample.get("registered_split") == "fit", "Phase-11B sample is not fit")
    _require(
        int(sample.get("train_holdout_frames_registered_but_unread", -1)) > 0,
        "Phase-11B holdout exclusion evidence missing",
    )
    for field in ("train_holdout_frames_read", "validation_frames_read", "test_frames_read"):
        _require(int(scope.get(field, -1)) == 0, f"Phase-11B evidence reports {field}")
    terminal_text = terminal_path.read_text(encoding="utf-8").strip()
    _require(
        terminal_text
        == f"SPLITFUSION_LOWBIT_PHASE11B_GPU_QUALIFIED {PHASE11B_EVIDENCE['sha256']}",
        "Phase-11B terminal content drift",
    )
    return dict(sample), {
        "evidence_path": PHASE11B_EVIDENCE["path"],
        "evidence_sha256": PHASE11B_EVIDENCE["sha256"],
        "terminal_path": PHASE11B_TERMINAL["path"],
        "terminal_sha256": PHASE11B_TERMINAL["sha256"],
    }


def _gpu_preflight() -> dict[str, Any]:
    _require(torch.cuda.is_available(), "CUDA is unavailable to /usr/bin/python3")
    count = int(torch.cuda.device_count())
    _require(count >= 1, "CUDA device count is zero")
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    _require(name == DEVICE_NAME, f"cuda:0 is {name!r}, expected {DEVICE_NAME!r}")

    processes = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    unrelated = []
    for line in processes.splitlines():
        if not line.strip():
            continue
        pid_text = line.split(",", 1)[0].strip()
        if pid_text.isdigit() and int(pid_text) == os.getpid():
            continue
        unrelated.append(line.strip())
    _require(not unrelated, f"unrelated CUDA workload present: {unrelated}")

    gpu_line = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,name,driver_version",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.splitlines()[0].strip()
    probe = torch.ones(1, dtype=torch.float32, device=device)
    _require(float(probe.item()) == 1.0, "tiny CUDA allocation returned wrong value")
    torch.cuda.synchronize(device)
    del probe
    return {
        "executable": "/usr/bin/python3",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_count": count,
        "device": str(device),
        "device_name": name,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "nvidia_smi_gpu": gpu_line,
        "unrelated_compute_processes": [],
        "tiny_allocation_and_synchronization": True,
    }


class CallLedger:
    def __init__(self) -> None:
        self._section = "startup"
        self._counts: dict[str, Counter[str]] = {
            "live": Counter(),
            "direct": Counter(),
        }

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        _require(name in self._counts, f"unknown audit section {name}")
        _require(self._section == "idle", f"nested audit section from {self._section}")
        self._section = name
        try:
            yield
        finally:
            self._section = "idle"

    def arm(self) -> None:
        _require(self._section == "startup", "call ledger was already armed")
        self._section = "idle"

    def bump(self, name: str) -> None:
        _require(self._section in self._counts, f"{name} executed outside an audited path")
        self._counts[self._section][name] += 1

    def snapshot(self, section: str) -> Counter[str]:
        return Counter(self._counts[section])


class CountingRanker:
    def __init__(self, ranker: torch.nn.Module, ledger: CallLedger) -> None:
        self._ranker = ranker
        self._ledger = ledger

    def score_cells(self, c2: torch.Tensor) -> torch.Tensor:
        self._ledger.bump("ranker")
        return self._ranker.score_cells(c2)


class FrozenFrontAdapter:
    def __init__(self, model: torch.nn.Module, ledger: CallLedger) -> None:
        self._model = model
        self._ledger = ledger
        self._last_c2: torch.Tensor | None = None

    def __call__(self, input_7ch: torch.Tensor) -> torch.Tensor:
        self._ledger.bump("front")
        c2 = self._model.encode_front(input_7ch)
        self._last_c2 = c2[0]
        return c2

    def take_c2(self) -> torch.Tensor:
        _require(self._last_c2 is not None, "front did not capture C2")
        value = self._last_c2
        self._last_c2 = None
        return value


@dataclass
class TailSnapshot:
    perception: Mapping[str, torch.Tensor]
    original_indices: torch.Tensor
    semantic_logits: torch.Tensor
    semantic_labels: torch.Tensor
    output_tensor_count: int
    records: tuple[dict[str, Any], ...] | None = None
    serialized_records: bytes | None = None


class FrozenP025TailAdapter:
    """Resident calibration around only the existing frozen tail/service calls."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        base: Any,
        row: Mapping[str, str],
        calibration: Mapping[str, torch.Tensor],
        ledger: CallLedger,
    ) -> None:
        self._model = model
        self._base = base
        self._row = MappingProxyType(dict(row))
        self._calibration = MappingProxyType(dict(calibration))
        self._ledger = ledger
        self._last: TailSnapshot | None = None
        fields = tuple(base.infer.FIELDS)
        forbidden = ("gt", "ground_truth", "target", "label_path")
        _require(
            not any(any(token in field.lower() for token in forbidden) for field in fields),
            "existing service-record schema requires evaluator-only GT fields",
        )
        self._record_fields = fields

    def __call__(self, c2: torch.Tensor, _metadata: DispatchMetadata) -> Mapping[str, torch.Tensor]:
        self._ledger.bump("tail")
        _require(self._last is None, "previous tail snapshot was not consumed")
        batch = c2.unsqueeze(0) if c2.ndim == 3 else c2
        outputs = self._model.decode_tail(batch, dense=False)
        tensor_count = require_tree_finite(outputs, "frozen tail output")
        postprocessed = self._model.postprocess(outputs, [self._calibration])[0]
        tensor_count += require_tree_finite(postprocessed, "camera-aware postprocess output")
        perception, original_indices = apply_p025_service_policy(
            {"semantic_logits": outputs["semantic_logits"]}, postprocessed
        )
        tensor_count += require_tree_finite(perception, "p025 service output")
        source_hw = (int(self._row["camera_height"]), int(self._row["camera_width"]))
        semantic_logits = outputs["semantic_logits"]
        semantic_labels = F.interpolate(
            semantic_logits.float(), size=source_hw, mode="bilinear", align_corners=False
        ).argmax(1)[0]
        self._last = TailSnapshot(
            perception=perception,
            original_indices=original_indices,
            semantic_logits=semantic_logits,
            semantic_labels=semantic_labels,
            output_tensor_count=tensor_count,
        )
        return perception

    def serialize(self, perception: Mapping[str, torch.Tensor]) -> bytes:
        self._ledger.bump("service_record_serialization")
        snapshot = self._last
        _require(snapshot is not None and perception is snapshot.perception, "tail/serializer handoff drift")
        records = tuple(
            combined_records(
                self._base,
                dict(self._row),
                snapshot.perception,
                snapshot.original_indices,
            )
        )
        for record in records:
            _require(tuple(record) == self._record_fields, "service-record schema drift")
            _require(record["sample_id"] == self._row["sample_id"], "service-record sample drift")
            for value in record.values():
                if isinstance(value, (int, float)):
                    _require(math.isfinite(float(value)), "non-finite service-record scalar")
        serialized = _canonical_bytes(records)
        snapshot.records = records
        snapshot.serialized_records = serialized
        return serialized

    def take_snapshot(self) -> TailSnapshot:
        _require(
            self._last is not None
            and self._last.records is not None
            and self._last.serialized_records is not None,
            "tail snapshot was not serialized",
        )
        value = self._last
        self._last = None
        return value


class AuditedWireCodec(ZstdWireCodec):
    def __init__(self, ledger: CallLedger, label: str) -> None:
        super().__init__()
        self._ledger = ledger
        self._label = label

    def compress(self, payload: Any):
        self._ledger.bump(f"{self._label}_zstd_compressions")
        return super().compress(payload)

    def decompress(self, frame: bytes, *, expected_bytes: int | None = None) -> bytes:
        self._ledger.bump(f"{self._label}_zstd_decompressions")
        return super().decompress(frame, expected_bytes=expected_bytes)

    def decompress_bytes(self, frame: bytes) -> bytes:
        self._ledger.bump(f"{self._label}_zstd_decompressions")
        return super().decompress_bytes(frame)


class AuditedUECodec(ProductionSplitCodec):
    def __init__(self, wire: AuditedWireCodec, ledger: CallLedger) -> None:
        super().__init__(wire)
        self._ledger = ledger

    def encode(self, *args: Any, **kwargs: Any) -> bytes:
        self._ledger.bump("phase13a_codec_encode")
        return super().encode(*args, **kwargs)


class AuditedEdgeCodec(ProductionSplitCodec):
    def __init__(self, wire: AuditedWireCodec, ledger: CallLedger) -> None:
        super().__init__(wire)
        self._ledger = ledger
        self._inspected: InspectedInnerPayload | None = None
        self._decoded: DecodedC2 | None = None

    def inspect(self, *args: Any, **kwargs: Any) -> InspectedInnerPayload:
        self._ledger.bump("phase13a_codec_inspect")
        _require(self._inspected is None, "prior inspected payload was not consumed")
        self._inspected = super().inspect(*args, **kwargs)
        return self._inspected

    def decode(self, *args: Any, **kwargs: Any) -> DecodedC2:
        self._ledger.bump("phase13a_codec_decode")
        _require(self._decoded is None, "prior decoded C2 was not consumed")
        self._decoded = super().decode(*args, **kwargs)
        return self._decoded

    def take(self) -> tuple[InspectedInnerPayload, DecodedC2]:
        _require(self._inspected is not None and self._decoded is not None, "edge codec capture missing")
        inspected, decoded = self._inspected, self._decoded
        self._inspected = None
        self._decoded = None
        return inspected, decoded


@dataclass(frozen=True)
class UdpDelivery:
    payload: bytes
    message_id: int
    datagrams: int
    duplicate_datagrams: int
    sfd1_application_bytes: int
    chunk_header_bytes: int
    udp_application_bytes: int
    estimated_ip_udp_bytes: int
    estimated_on_wire_bytes: int


class RawUdpLoopback:
    """One sender/receiver pair around the existing raw `!IHH` primitives."""

    def __init__(self, expected_messages: int) -> None:
        self._expected = int(expected_messages)
        self._delivered: queue.Queue[UdpDelivery | BaseException] = queue.Queue()
        self._receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUFFER_REQUEST_BYTES)
        self._sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SOCKET_BUFFER_REQUEST_BYTES)
        self._receiver.bind(("127.0.0.1", 0))
        self._receiver.settimeout(120.0)
        self._destination = self._receiver.getsockname()
        self._ready = threading.Event()
        self._closed = False
        self._thread = threading.Thread(
            target=self._receive,
            name="phase13b-raw-udp-receiver",
            daemon=True,
        )
        self.requested_buffer_bytes = SOCKET_BUFFER_REQUEST_BYTES
        self.reported_receive_buffer_bytes = int(
            self._receiver.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        )
        self.reported_send_buffer_bytes = int(
            self._sender.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        )
        try:
            self._thread.start()
            _require(self._ready.wait(timeout=5.0), "UDP receiver did not become ready")
        except BaseException:
            self._closed = True
            self._receiver.close()
            self._sender.close()
            raise

    def _receive(self) -> None:
        reassembler = ChunkReassembler(timeout_s=120.0, max_chunks=4096)
        completed = 0
        self._ready.set()
        try:
            while completed < self._expected:
                datagram, source = self._receiver.recvfrom(65_535)
                result = reassembler.ingest(
                    f"{source[0]}:{source[1]}",
                    datagram,
                    received_at_s=time.monotonic(),
                )
                if result is None:
                    continue
                datagrams = int(result.chunk_count)
                application = len(result.payload)
                chunk_headers = datagrams * CHUNK_HEADER.size
                udp_application = application + chunk_headers
                ip_udp = datagrams * 28
                self._delivered.put(
                    UdpDelivery(
                        payload=result.payload,
                        message_id=int(result.message_id),
                        datagrams=datagrams,
                        duplicate_datagrams=int(result.duplicate_chunks),
                        sfd1_application_bytes=application,
                        chunk_header_bytes=chunk_headers,
                        udp_application_bytes=udp_application,
                        estimated_ip_udp_bytes=ip_udp,
                        estimated_on_wire_bytes=udp_application + ip_udp,
                    )
                )
                completed += 1
        except BaseException as exc:
            if not self._closed:
                self._delivered.put(exc)

    def roundtrip(self, payload: bytes, *, message_id: int) -> UdpDelivery:
        _require(not self._closed, "UDP loopback is closed")
        chunks = chunk_payload(payload, message_id=message_id, chunk_bytes=CHUNK_BYTES)
        for datagram in chunks:
            sent = self._sender.sendto(datagram, self._destination)
            _require(sent == len(datagram), "localhost UDP datagram was truncated on send")
        delivered = self._delivered.get(timeout=125.0)
        if isinstance(delivered, BaseException):
            raise RuntimeError("localhost UDP receiver failed") from delivered
        _require(delivered.message_id == message_id, "reassembled UDP message ID drift")
        _require(delivered.datagrams == len(chunks), "reassembled UDP datagram count drift")
        _require(delivered.duplicate_datagrams == 0, "duplicate UDP datagram observed")
        _require(delivered.payload == payload, "reassembled SFD1 bytes differ from transmitted bytes")
        return delivered

    def close(self) -> None:
        self._closed = True
        self._receiver.close()
        self._sender.close()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


@dataclass
class DirectResult:
    packet_bytes: bytes
    reconstructed_c2: torch.Tensor
    keep_indices: torch.Tensor
    keep_count: int
    q_e4: int
    tail: TailSnapshot
    serialized: bytes


def _direct_path(
    *,
    profile: ActionProfile,
    c2: torch.Tensor,
    ranker: CountingRanker,
    autoencoders: Mapping[str, Any],
    uint8_decoders: PreloadedAeDecoders,
    lowbit_decoders: lowbit_dispatch.PreloadedLowBitDecoders,
    wire: AuditedWireCodec,
    tail: FrozenP025TailAdapter,
    metadata: DispatchMetadata,
    device: torch.device,
) -> DirectResult:
    plan = continuous_q.quantize_q(profile.q)
    selected_ranker = None if plan.is_bypass else ranker
    if profile.quantizer == "UINT8" and profile.family == "noAE":
        prepared = uint8_zstd_transport.prepare_frame(c2)
        transport = uint8_zstd_transport.encode(
            prepared, selected_ranker, plan.wire_q, wire_codec=wire
        )
        sparse = wire.decompress_bytes(transport.packet.data)
        parsed = uint8_codec.inspect(sparse)
        reconstructed, q = uint8_codec.decode(sparse)
        reconstructed = reconstructed.to(device)
        packet_bytes = transport.packet.data
        keep_indices = parsed.keep_indices
        keep_count = int(parsed.header.keep_count)
        q_e4 = int(parsed.header.q_e4)
    elif profile.quantizer == "UINT8":
        autoencoder = autoencoders[profile.family]
        transport = ae_uint8_transport.encode_frame(
            c2,
            autoencoder,
            selected_ranker,
            plan.wire_q,
            wire_codec=wire,
        )
        received = uint8_decoders.receive(
            transport.packet.data, wire_codec=wire, diagnostics=True
        )
        _require(received.diagnostics is not None, "direct UINT8 diagnostics missing")
        parsed = received.diagnostics.parsed
        reconstructed, q = received.c2, received.q
        packet_bytes = transport.packet.data
        keep_indices = parsed.keep_indices
        keep_count = int(parsed.header.keep_count)
        q_e4 = int(parsed.header.q_e4)
    else:
        if profile.family == "noAE":
            transport = lowbit_transport.encode_noae_frame(
                c2,
                selected_ranker,
                plan.wire_q,
                profile.bit_width,
                wire_codec=wire,
            )
        else:
            transport = lowbit_transport.encode_ae_frame(
                c2,
                autoencoders[profile.family],
                selected_ranker,
                plan.wire_q,
                profile.bit_width,
                wire_codec=wire,
            )
        received = lowbit_decoders.receive(
            transport.packet.data, wire_codec=wire, diagnostics=True
        )
        _require(received.diagnostics is not None, "direct low-bit diagnostics missing")
        parsed = received.diagnostics.parsed
        reconstructed, q = received.c2, received.q
        packet_bytes = transport.packet.data
        keep_indices = parsed.keep_indices
        keep_count = int(parsed.header.keep_count)
        q_e4 = int(parsed.header.q_e4)
    _require(continuous_q.quantize_q(q).q_e4 == profile.q_e4, "direct q drift")
    guards.require_frozen_c2(reconstructed, what="direct reconstructed C2")
    _require(reconstructed.device == device, "direct reconstructed C2 device drift")
    perception = tail(reconstructed, metadata)
    serialized = tail.serialize(perception)
    snapshot = tail.take_snapshot()
    return DirectResult(
        packet_bytes=packet_bytes,
        reconstructed_c2=reconstructed,
        keep_indices=keep_indices,
        keep_count=keep_count,
        q_e4=q_e4,
        tail=snapshot,
        serialized=serialized,
    )


def _require_timing(trace: TimingTrace, stages: tuple[str, ...]) -> None:
    expected = (*stages[1:], stages[0])
    names = tuple(boundary.name for boundary in trace.boundaries)
    _require(names == expected, f"timing boundary order drift: {names} != {expected}")
    total = trace.boundaries[-1]
    for boundary in trace.boundaries:
        _require(
            boundary.finished_monotonic_ns >= boundary.started_monotonic_ns,
            f"negative timing boundary: {boundary.name}",
        )
        _require(
            total.started_monotonic_ns <= boundary.started_monotonic_ns
            and boundary.finished_monotonic_ns <= total.finished_monotonic_ns,
            f"timing boundary escapes total interval: {boundary.name}",
        )
    _require(trace.clock == "time.monotonic_ns" and not trace.latency_published, "timing clock/status drift")


def _counter_delta(after: Counter[str], before: Counter[str]) -> dict[str, int]:
    keys = sorted(set(after) | set(before))
    return {name: int(after[name] - before[name]) for name in keys if after[name] != before[name]}


def _operation_delta(after: Any, before: Any) -> dict[str, int]:
    after_values = asdict(after)
    before_values = asdict(before)
    return {
        name: int(after_values[name] - before_values[name])
        for name in after_values
        if isinstance(after_values[name], int) and after_values[name] != before_values[name]
    }


def _require_exact_outputs(live: TailSnapshot, direct: TailSnapshot) -> dict[str, Any]:
    _require(set(live.perception) == set(direct.perception), "p025 output field set drift")
    for name in live.perception:
        _require(
            torch.equal(live.perception[name], direct.perception[name]),
            f"p025 output mismatch: {name}",
        )
    _require(
        torch.equal(live.original_indices, direct.original_indices),
        "p025 original-index ordering mismatch",
    )
    _require(
        torch.equal(live.semantic_logits, direct.semantic_logits),
        "segmentation logit mismatch",
    )
    _require(
        torch.equal(live.semantic_labels, direct.semantic_labels),
        "segmentation label mismatch",
    )
    _require(live.records == direct.records, "service-record value/order mismatch")
    _require(
        live.serialized_records == direct.serialized_records,
        "serialized service-record bytes mismatch",
    )
    return {
        "exact_detection_fields": len(live.perception),
        "detection_count": int(live.perception["scores"].numel()),
        "detection_mapping_sha256": _mapping_digest(live.perception),
        "segmentation_logits_sha256": _tensor_digest(live.semantic_logits),
        "segmentation_labels_sha256": _tensor_digest(live.semantic_labels),
        "serialized_service_records_sha256": _digest_bytes(live.serialized_records or b""),
        "service_record_count": len(live.records or ()),
    }


@contextmanager
def _hot_path_guard() -> Iterator[None]:
    def forbidden(name: str):
        def fail(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(f"hot path attempted forbidden operation: {name}")

        return fail

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(torch, "load", forbidden("torch.load")))
        stack.enter_context(mock.patch.object(torch.nn.Module, "__init__", forbidden("model construction")))
        stack.enter_context(mock.patch.object(torch.nn.Module, "to", forbidden("module.to")))
        stack.enter_context(mock.patch.object(torch.nn.Module, "eval", forbidden("module.eval")))
        stack.enter_context(mock.patch.object(torch.nn.Module, "load_state_dict", forbidden("load_state_dict")))
        stack.enter_context(
            mock.patch.object(torch.nn.Parameter, "requires_grad_", forbidden("parameter mutation"))
        )
        stack.enter_context(
            mock.patch.object(
                SplitActionRegistry,
                "from_runtime_binding",
                forbidden("registry reconstruction"),
            )
        )
        yield


def _state(module: torch.nn.Module) -> dict[str, Any]:
    return phase11b._state_record(module)


def _prepare_output_leaf() -> Path:
    root = (_root() / "experiments").resolve(strict=True)
    candidate = (_root() / OUTPUT_RELPATH).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("Phase-13B output escapes experiments root") from exc
    _require(not candidate.exists(), f"create-only output already exists: {candidate}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.mkdir(parents=False, exist_ok=False)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("created Phase-13B output escapes experiments root") from exc
    return resolved


def _startup() -> dict[str, Any]:
    git = _verify_git_state()
    phase13a = _verify_phase13a_terminal()
    registry = SplitActionRegistry.from_runtime_binding(verify_runtime_artifacts=True)
    sample_evidence, sample_binding = _phase11b_sample_binding()
    gpu = _gpu_preflight()
    device = torch.device("cuda:0")

    historical = phase11b.phase11b_preflight()
    model, base, perception_binding = load_frozen_perception(device)
    phase11b.common.freeze(model)
    ranker_model = phase11b._load_ranker(device)
    autoencoders = {
        family_name: phase11b._load_selected_autoencoder(
            family_name,
            bottleneck,
            phase11b.FROZEN_INPUTS[family_name],
            historical["checkpoint_payloads"][family_name],
            device,
        )
        for family_name, _family_id, bottleneck in phase11b.FAMILIES
        if bottleneck is not None
    }
    historical["checkpoint_payloads"].clear()
    guards.require_frozen_perception([model, ranker_model, *autoencoders.values()])
    guards.require_eval_mode([model, ranker_model, *autoencoders.values()])

    train_dataset, train_index, fit_proof = phase11b._select_fit_frame(base)
    _require(
        fit_proof["sample_id"] == sample_evidence["sample_id"]
        and train_index == int(sample_evidence["dataset_index"]),
        "live fit selection disagrees with bound Phase-11B evidence",
    )
    train_row = dict(train_dataset.rows[train_index])
    inference = base.data.InferenceDataset(train_dataset.root, "train")
    inference_matches = [
        index
        for index, row in enumerate(inference.rows)
        if row["sample_id"] == sample_evidence["sample_id"]
    ]
    _require(inference_matches == [train_index], "deployable train sample ordering drift")
    fused, row, calibration_cpu = inference[train_index]
    _require(row["sample_id"] == fit_proof["sample_id"], "loaded fit sample identity drift")
    _require(tuple(fused.shape)[0] == 7 and fused.dtype is torch.float32, "fit input is not FP32 7-channel")
    calibration_binding = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "sha256": _tensor_digest(value),
        }
        for name, value in sorted(calibration_cpu.items())
    }
    calibration = {name: value.to(device) for name, value in calibration_cpu.items()}
    input_7ch = fused.unsqueeze(0).to(device)
    sequence_id = int(row["frame_id"])
    capture_timestamp_ns = int(round(float(train_row["timestamp"]) * 1_000_000_000))
    _require(sequence_id >= 0 and capture_timestamp_ns >= 0, "sample sequence/timestamp invalid")
    del inference, train_dataset, fused, calibration_cpu

    ledger = CallLedger()
    ranker = CountingRanker(ranker_model, ledger)
    front = FrozenFrontAdapter(model, ledger)
    tail = FrozenP025TailAdapter(
        model=model,
        base=base,
        row=row,
        calibration=calibration,
        ledger=ledger,
    )
    hooks = []
    for family, autoencoder in autoencoders.items():
        hooks.append(
            autoencoder.project.register_forward_hook(
                lambda _module, _inputs, _output, family=family: ledger.bump(
                    f"ae_encoder_{family}"
                )
            )
        )
        hooks.append(
            autoencoder.expand.register_forward_hook(
                lambda _module, _inputs, _output, family=family: ledger.bump(
                    f"ae_decoder_{family}"
                )
            )
        )

    ue_wire = AuditedWireCodec(ledger, "live_ue")
    edge_wire = AuditedWireCodec(ledger, "live_edge")
    direct_wire = AuditedWireCodec(ledger, "direct")
    ue_codec = AuditedUECodec(ue_wire, ledger)
    edge_codec = AuditedEdgeCodec(edge_wire, ledger)
    ue = PreloadedSplitUERuntime(
        registry,
        front=front,
        ranker=ranker,
        ae_encoders=autoencoders,
        device=device,
        codec=ue_codec,
        prepare_modules=False,
        startup_model_load_operations=5,
        startup_model_construction_operations=5,
    )
    edge = PreloadedSplitEdgeRuntime(
        registry,
        frozen_p025_tail=tail,
        ae_decoders=autoencoders,
        tail_device=device,
        codec=edge_codec,
        output_serializer=tail.serialize,
        prepare_modules=False,
        startup_model_load_operations=4,
        startup_model_construction_operations=4,
    )
    uint8_decoders = PreloadedAeDecoders(autoencoders.values())
    lowbit_decoders = lowbit_dispatch.PreloadedLowBitDecoders(
        autoencoders.values(), tail_device=device
    )
    state_before = {
        "perception": _state(model),
        "ranker": _state(ranker_model),
        **{family: _state(autoencoder) for family, autoencoder in autoencoders.items()},
    }
    loopback = RawUdpLoopback(expected_messages=len(ACTION_IDS))
    ledger.arm()
    return {
        "git": git,
        "phase13a": phase13a,
        "registry": registry,
        "registry_audit": asdict(registry.startup_audit),
        "phase11b": {
            **sample_binding,
            "historical_provenance_checks_passed": True,
            "families": sorted(historical["historical_source_bindings"]),
            "device_repair_transition": historical[
                "phase11b_device_repair_source_transition"
            ],
        },
        "gpu": gpu,
        "device": device,
        "model": model,
        "base": base,
        "perception_binding": perception_binding,
        "ranker_model": ranker_model,
        "ranker": ranker,
        "autoencoders": autoencoders,
        "front": front,
        "tail": tail,
        "ue": ue,
        "edge": edge,
        "edge_codec": edge_codec,
        "direct_wire": direct_wire,
        "uint8_decoders": uint8_decoders,
        "lowbit_decoders": lowbit_decoders,
        "ledger": ledger,
        "hooks": hooks,
        "state_before": state_before,
        "loopback": loopback,
        "input_7ch": input_7ch,
        "row": row,
        "sequence_id": sequence_id,
        "capture_timestamp_ns": capture_timestamp_ns,
        "fit_frame": {
            **fit_proof,
            "phase11b_evidence_bound": True,
            "input_view": "InferenceDataset(train): RGB+radar+camera calibration only",
            "rgb_frames_read": 1,
            "radar_frames_read": 1,
            "semantic_or_depth_labels_read": 0,
            "calibration": calibration_binding,
            "camera_matrix_json_sha256": _digest_bytes(
                row["camera_matrix_json"].encode("utf-8")
            ),
        },
    }


def _qualify_action(runtime: Mapping[str, Any], action_id: int, ordinal: int) -> dict[str, Any]:
    registry: SplitActionRegistry = runtime["registry"]
    profile = registry.resolve(action_id)
    ledger: CallLedger = runtime["ledger"]
    live_before = ledger.snapshot("live")
    direct_before = ledger.snapshot("direct")
    ue_before = runtime["ue"].counters
    edge_before = runtime["edge"].counters

    with _hot_path_guard():
        with ledger.section("live"):
            prepared = runtime["ue"].prepare(
                action_id,
                runtime["input_7ch"],
                sequence_id=runtime["sequence_id"],
                capture_timestamp_ns=runtime["capture_timestamp_ns"],
            )
            c2 = runtime["front"].take_c2()
            delivery = runtime["loopback"].roundtrip(
                prepared.wire_bytes, message_id=ordinal
            )
            received_hash = _digest_bytes(delivery.payload)
            transmitted_hash = _digest_bytes(prepared.wire_bytes)
            _require(received_hash == transmitted_hash, "UDP SFD1 digest mismatch")
            live_result = runtime["edge"].process(
                delivery.payload, transmitted_action_id=action_id
            )
            live_inspected, live_decoded = runtime["edge_codec"].take()
            live_tail = runtime["tail"].take_snapshot()

        with ledger.section("direct"):
            direct = _direct_path(
                profile=profile,
                c2=c2,
                ranker=runtime["ranker"],
                autoencoders=runtime["autoencoders"],
                uint8_decoders=runtime["uint8_decoders"],
                lowbit_decoders=runtime["lowbit_decoders"],
                wire=runtime["direct_wire"],
                tail=runtime["tail"],
                metadata=live_result.metadata,
                device=runtime["device"],
            )

    outer = unpack_envelope(prepared.wire_bytes)
    _require(outer.protocol_version == PROTOCOL_VERSION, "SFD1 version drift")
    _require(outer.action_id == action_id == live_result.metadata.action_id, "action identity drift")
    _require(
        outer.sequence_id
        == prepared.metadata.sequence_id
        == live_result.metadata.sequence_id
        == runtime["sequence_id"],
        "SFD1 sequence identity drift",
    )
    _require(
        outer.capture_timestamp_ns
        == prepared.metadata.capture_timestamp_ns
        == live_result.metadata.capture_timestamp_ns
        == runtime["capture_timestamp_ns"],
        "SFD1 capture timestamp drift",
    )
    _require(outer.inner_payload_length == prepared.inner_payload_bytes, "SFD1 inner length drift")
    _require(
        outer.control_overhead_bytes == prepared.outer_envelope_bytes == HEADER_BYTES,
        "SFD1 control overhead drift",
    )
    _require(
        outer.total_transmitted_bytes
        == prepared.total_transmitted_bytes
        == delivery.sfd1_application_bytes,
        "SFD1 total byte accounting drift",
    )
    _require(outer.inner_payload == direct.packet_bytes, "live/direct inner codec bytes differ")
    _require(live_inspected.identity.q_e4 == profile.q_e4, "live inner q drift")
    _require(live_inspected.identity.keep_count == profile.keep_count, "live inner keep-count drift")
    _require(direct.q_e4 == profile.q_e4 and direct.keep_count == profile.keep_count, "direct identity drift")
    _require(
        torch.equal(live_inspected.parsed.keep_indices, direct.keep_indices),
        "live/direct selected indices differ",
    )
    guards.require_frozen_c2(live_decoded.c2, what="live reconstructed C2")
    _require(
        live_decoded.c2.dtype is torch.float32
        and tuple(live_decoded.c2.shape) == (256, 112, 192)
        and live_decoded.c2.device == runtime["device"]
        and bool(torch.isfinite(live_decoded.c2).all()),
        "live reconstructed C2 contract drift",
    )
    _require(
        torch.equal(live_decoded.c2, direct.reconstructed_c2),
        "live/direct reconstructed C2 mismatch",
    )
    parity = _require_exact_outputs(live_tail, direct.tail)
    _require(live_result.perception is live_tail.perception, "edge changed p025 representation")
    _require(live_result.serialized_output == direct.serialized, "edge/direct serialized output drift")
    _require_timing(prepared.timing, UE_STAGES)
    _require_timing(live_result.timing, EDGE_STAGES)

    live_counts = _counter_delta(ledger.snapshot("live"), live_before)
    direct_counts = _counter_delta(ledger.snapshot("direct"), direct_before)
    expected_ranker = 0 if profile.q_e4 == 0 else 1
    expected_ae = 0 if profile.family == "noAE" else 1
    _require(live_counts.get("ranker", 0) == expected_ranker, "live ranker call count drift")
    _require(live_counts.get(f"ae_encoder_{profile.family}", 0) == expected_ae, "live AE encoder call count drift")
    _require(live_counts.get(f"ae_decoder_{profile.family}", 0) == expected_ae, "live AE decoder call count drift")
    _require(sum(value for key, value in live_counts.items() if key.startswith("ae_encoder_")) == expected_ae, "wrong live AE encoder selected")
    _require(sum(value for key, value in live_counts.items() if key.startswith("ae_decoder_")) == expected_ae, "wrong live AE decoder selected")
    for name in (
        "live_ue_zstd_compressions",
        "live_edge_zstd_decompressions",
        "phase13a_codec_encode",
        "phase13a_codec_inspect",
        "phase13a_codec_decode",
        "front",
        "tail",
        "service_record_serialization",
    ):
        _require(live_counts.get(name, 0) == 1, f"live call count drift: {name}")
    _require(direct_counts.get("direct_zstd_compressions", 0) == 1, "direct compression count drift")
    _require(direct_counts.get("direct_zstd_decompressions", 0) == 1, "direct decompression count drift")
    _require(direct_counts.get("tail", 0) == 1, "direct tail count drift")
    _require(
        direct_counts.get("service_record_serialization", 0) == 1,
        "direct service serialization count drift",
    )
    _require(direct_counts.get("ranker", 0) == expected_ranker, "direct ranker count drift")
    _require(
        direct_counts.get(f"ae_encoder_{profile.family}", 0) == expected_ae,
        "direct AE encoder call count drift",
    )
    _require(
        direct_counts.get(f"ae_decoder_{profile.family}", 0) == expected_ae,
        "direct AE decoder call count drift",
    )
    _require(
        sum(value for key, value in direct_counts.items() if key.startswith("ae_encoder_"))
        == expected_ae,
        "wrong direct AE encoder selected",
    )
    _require(
        sum(value for key, value in direct_counts.items() if key.startswith("ae_decoder_"))
        == expected_ae,
        "wrong direct AE decoder selected",
    )

    ue_delta = _operation_delta(runtime["ue"].counters, ue_before)
    edge_delta = _operation_delta(runtime["edge"].counters, edge_before)
    _require(ue_delta.get("frames_attempted") == 1 and ue_delta.get("frames_completed") == 1, "UE frame counters drift")
    _require(edge_delta.get("frames_attempted") == 1 and edge_delta.get("frames_completed") == 1, "edge frame counters drift")
    _require(edge_delta.get("tail_dispatches") == 1, "edge tail dispatch count drift")
    _require(runtime["ue"].counters.hot_path_model_load_operations == 0, "UE hot load count nonzero")
    _require(runtime["edge"].counters.hot_path_model_load_operations == 0, "edge hot load count nonzero")

    row = {
        "action_id": profile.action_id,
        "profile_id": profile.profile_id,
        "resolved_catalog_identity": {
            "family": profile.family,
            "family_id": profile.family_id,
            "quantizer": profile.quantizer,
            "bit_width": profile.bit_width,
            "q_e4": profile.q_e4,
            "keep_count": profile.keep_count,
            "latent_width": profile.latent_width,
            "routing_tag": profile.routing_tag,
            "wire_magic_ascii": profile.wire.magic_ascii,
            "wire_codec_id": profile.wire.codec_id,
            "wire_version": profile.wire.version,
            "segmentation_installable": profile.segmentation_installable,
        },
        "sfd1": {
            "protocol_version": outer.protocol_version,
            "sequence_id": outer.sequence_id,
            "capture_timestamp_ns": outer.capture_timestamp_ns,
            "scientific_inner_payload_bytes": prepared.inner_payload_bytes,
            "framing_control_overhead_bytes": prepared.outer_envelope_bytes,
            "total_application_bytes": prepared.total_transmitted_bytes,
            "transmitted_sha256": transmitted_hash,
            "reassembled_sha256": received_hash,
            "reassembled_exact": True,
        },
        "udp": {
            "diagnostic_chunk_bytes_including_header": CHUNK_BYTES,
            "sfd1_bytes_per_full_chunk": CHUNK_BYTES - CHUNK_HEADER.size,
            "datagrams": delivery.datagrams,
            "duplicate_datagrams": delivery.duplicate_datagrams,
            "chunk_header_bytes": delivery.chunk_header_bytes,
            "udp_application_bytes": delivery.udp_application_bytes,
            "estimated_ip_udp_bytes": delivery.estimated_ip_udp_bytes,
            "estimated_on_wire_bytes": delivery.estimated_on_wire_bytes,
        },
        "live_calls": live_counts,
        "direct_diagnostic_calls": direct_counts,
        "phase13a_ue_counter_delta": ue_delta,
        "phase13a_edge_counter_delta": edge_delta,
        "reconstructed_c2": {
            "shape": list(live_decoded.c2.shape),
            "dtype": str(live_decoded.c2.dtype).replace("torch.", ""),
            "device": str(live_decoded.c2.device),
            "finite": True,
            "sha256": _tensor_digest(live_decoded.c2),
        },
        "parity": {
            "selected_indices_exact": True,
            "selected_indices_count": int(live_inspected.parsed.keep_indices.numel()),
            "selected_indices_sha256": _tensor_digest(
                live_inspected.parsed.keep_indices
            ),
            "keep_count_exact": True,
            "inner_codec_bytes_exact": True,
            "reconstructed_c2_exact": True,
            "detections_all_fields_and_order_exact": True,
            "segmentation_logits_exact": True,
            "segmentation_labels_exact": True,
            "serialized_service_records_exact": True,
            **parity,
        },
        "timing_instrumentation": {
            "clock": "time.monotonic_ns",
            "ue_boundaries_present": True,
            "edge_boundaries_present": True,
            "ordering_valid": True,
            "durations_nonnegative": True,
            "latency_measured_or_published": False,
        },
    }
    del direct, live_tail, live_result, live_inspected, live_decoded, c2, prepared, outer, delivery
    return row


def _report(document: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 13B four-action GPU/localhost qualification",
        "",
        f"Status: `{document['terminal']}`.",
        "",
        "This is a bounded functional qualification, not a latency or 36-profile campaign.",
        "",
        "| action | profile | family | quantizer | q_e4 | keep | SFD1 bytes | UDP datagrams | detections | parity |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in document["actions"]:
        identity = row["resolved_catalog_identity"]
        lines.append(
            f"| {row['action_id']} | {row['profile_id']} | {identity['family']} | "
            f"{identity['quantizer']} | {identity['q_e4']} | {identity['keep_count']} | "
            f"{row['sfd1']['total_application_bytes']} | {row['udp']['datagrams']} | "
            f"{row['parity']['detection_count']} | exact |"
        )
    lines.extend(
        [
            "",
            "All four live messages used the committed SFD1 and inner codec bytes, exactly one live zstd decompression and one live tail call. Direct diagnostic calls were counted separately.",
            "",
            "The same Phase-11B-bound fit frame and one resident camera calibration were used throughout. No holdout, validation, or test frame was opened. No tensor, SFD1 blob, datagram, prediction payload, or serialized service payload is retained.",
            "",
            "Monotonic stage boundaries were checked only for presence, order, and non-negative duration. No latency, FPS, throughput, Pi/OAI, or deployment-performance result is claimed.",
            "",
            "The 288-cell campaign remains blocked pending the later localhost measurement, RFsim calibration, and 16-cell OAI pilot.",
        ]
    )
    return "\n".join(lines) + "\n"


def _close_runtime(runtime: Mapping[str, Any] | None) -> None:
    if runtime is None:
        return
    loopback = runtime.get("loopback")
    if loopback is not None:
        loopback.close()
    for hook in runtime.get("hooks", ()):
        hook.remove()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase-13B four-action real-model CUDA/localhost qualification"
    )
    parser.add_argument("--execute", required=True, choices=(EXECUTE_TOKEN,))
    args = parser.parse_args()
    del args

    runtime: dict[str, Any] | None = None
    output: Path | None = None
    completed: list[dict[str, Any]] = []
    completed_files: list[dict[str, str]] = []
    current_action: int | None = None
    operation = "preflight"
    started = time.perf_counter()
    try:
        runtime = _startup()
        operation = "create_output"
        output = _prepare_output_leaf()
        (output / "actions").mkdir(parents=False, exist_ok=False)
        torch.cuda.reset_peak_memory_stats(runtime["device"])
        operation = "four_action_qualification"
        for ordinal, action_id in enumerate(ACTION_IDS, start=1):
            current_action = action_id
            row = _qualify_action(runtime, action_id, ordinal)
            relative = Path("actions") / f"action_{action_id:02d}.json"
            digest = _atomic_json(output / relative, row)
            completed.append(row)
            completed_files.append({"path": str(relative), "sha256": digest})

        operation = "frozen_state_check"
        state_after = {
            "perception": _state(runtime["model"]),
            "ranker": _state(runtime["ranker_model"]),
            **{
                family: _state(autoencoder)
                for family, autoencoder in runtime["autoencoders"].items()
            },
        }
        frozen_equal = {
            name: runtime["state_before"][name] == state_after[name]
            for name in runtime["state_before"]
        }
        _require(all(frozen_equal.values()), "frozen model/ranker/AE state changed")
        _require(
            all(
                parameter.grad is None
                for module in (
                    runtime["model"],
                    runtime["ranker_model"],
                    *runtime["autoencoders"].values(),
                )
                for parameter in module.parameters()
            ),
            "a frozen parameter received a gradient",
        )
        torch.cuda.synchronize(runtime["device"])
        wall_seconds = time.perf_counter() - started
        udp_totals = {
            "messages": len(completed),
            "datagrams": sum(row["udp"]["datagrams"] for row in completed),
            "sfd1_application_bytes": sum(
                row["sfd1"]["total_application_bytes"] for row in completed
            ),
            "scientific_inner_payload_bytes": sum(
                row["sfd1"]["scientific_inner_payload_bytes"] for row in completed
            ),
            "sfd1_control_overhead_bytes": len(completed) * HEADER_BYTES,
            "chunk_header_bytes": sum(row["udp"]["chunk_header_bytes"] for row in completed),
            "udp_application_bytes": sum(
                row["udp"]["udp_application_bytes"] for row in completed
            ),
            "estimated_ip_udp_bytes": sum(
                row["udp"]["estimated_ip_udp_bytes"] for row in completed
            ),
            "estimated_on_wire_bytes": sum(
                row["udp"]["estimated_on_wire_bytes"] for row in completed
            ),
        }
        _require(udp_totals["messages"] == 4, "localhost delivered-message count drift")
        document = {
            "schema": SCHEMA,
            "terminal": TERMINAL,
            "status": "FOUR_ACTION_GPU_LOOPBACK_FUNCTIONALLY_QUALIFIED_NOT_LATENCY_MEASURED",
            "implementation_commit": runtime["git"]["head"],
            "implementation_source": {
                "path": runtime["git"]["implementation_source"],
                "sha256": runtime["git"]["implementation_source_sha256"],
                "phase13b_implementation_commit": runtime["git"][
                    "phase13b_implementation_commit"
                ],
                "phase13a_commit": runtime["git"]["phase13a_commit"],
            },
            "phase13a_binding": runtime["phase13a"],
            "phase11b_sample_binding": runtime["phase11b"],
            "registry_startup_audit": runtime["registry_audit"],
            "perception_binding": runtime["perception_binding"],
            "environment": runtime["gpu"],
            "fit_frame": runtime["fit_frame"],
            "startup_objects": {
                "ue": {
                    "device": str(runtime["device"]),
                    "objects": ["frozen_front", "stable_epoch4_ranker", "AE128_encoder", "AE64_encoder", "AE32_encoder"],
                    "phase13a_counters": asdict(runtime["ue"].counters),
                },
                "edge": {
                    "device": str(runtime["device"]),
                    "objects": ["frozen_tail_p025_adapter_with_resident_calibration", "AE128_decoder", "AE64_decoder", "AE32_decoder"],
                    "phase13a_counters": asdict(runtime["edge"].counters),
                },
                "checkpoint_deserializations": {
                    "perception": 1,
                    "ranker": 1,
                    "AE128": 1,
                    "AE64": 1,
                    "AE32": 1,
                },
                "transient_cpu_ae_checkpoint_contract_validators": 3,
                "resident_modules_prepared_before_hot_path": True,
                "hot_path_load_construct_move_eval_mutate_registry_rebuild": 0,
            },
            "udp": {
                "transport": "real localhost UDP",
                "fragmentation_header": "!IHH",
                "chunk_bytes_including_header": CHUNK_BYTES,
                "diagnostic_only_not_oai_binding": True,
                "sender_sockets_created": 1,
                "receiver_sockets_created": 1,
                "sockets_reused_for_all_actions": True,
                "receiver_ready_before_first_transmission": True,
                "concurrent_receive_and_reassembly": True,
                "retransmission": False,
                "secondary_compression_or_pickle": False,
                "requested_socket_buffer_bytes": runtime["loopback"].requested_buffer_bytes,
                "reported_receive_buffer_bytes": runtime["loopback"].reported_receive_buffer_bytes,
                "reported_send_buffer_bytes": runtime["loopback"].reported_send_buffer_bytes,
                "totals": udp_totals,
            },
            "actions": completed,
            "per_action_records": completed_files,
            "integrity": {
                "requested_actions_resolved_from_catalog": list(ACTION_IDS),
                "sfd1_messages_delivered": 4,
                "live_zstd_decompressions": sum(
                    row["live_calls"]["live_edge_zstd_decompressions"] for row in completed
                ),
                "live_tail_calls": sum(row["live_calls"]["tail"] for row in completed),
                "direct_diagnostic_zstd_decompressions": sum(
                    row["direct_diagnostic_calls"]["direct_zstd_decompressions"]
                    for row in completed
                ),
                "direct_diagnostic_tail_calls": sum(
                    row["direct_diagnostic_calls"]["tail"] for row in completed
                ),
                "all_direct_path_parity_exact": True,
                "frozen_state_equal": frozen_equal,
                "all_gradients_absent": True,
                "calibration_resident_not_transmitted": True,
                "no_30m_runtime_filter": True,
                "all_p025_detections_emitted_at_every_distance": True,
            },
            "scope": {
                "fit_training_frames_read": 1,
                "train_holdout_frames_read": 0,
                "validation_frames_read": 0,
                "test_frames_read": 0,
                "semantic_gt_frames_read": 0,
                "depth_gt_frames_read": 0,
                "scoring_used": False,
                "training_or_tuning": False,
                "latency_measured_or_published": False,
                "fps_or_throughput_claimed": False,
                "oai_or_rfsim_used": False,
                "campaign_launched": False,
                "payload_or_tensor_artifacts_retained": False,
                "campaign_288_status": "BLOCKED_PENDING_LATER_LOCALHOST_MEASUREMENT_RFSIM_CALIBRATION_AND_16_CELL_OAI_PILOT",
            },
            "resources": {
                "wall_seconds": wall_seconds,
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(runtime["device"])
                ),
                "peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(runtime["device"])
                ),
            },
        }
        operation = "write_compact_evidence"
        qualification_hash = _atomic_json(output / "qualification.json", document)
        report_hash = _atomic_text(output / "REPORT.md", _report(document))
        terminal_hash = _atomic_text(
            output / TERMINAL, f"{TERMINAL} {qualification_hash}\n"
        )
        print(
            json.dumps(
                {
                    "terminal": TERMINAL,
                    "output": str(output.relative_to(_root())),
                    "qualification_sha256": qualification_hash,
                    "report_sha256": report_hash,
                    "terminal_sha256": terminal_hash,
                    "actions": [row["action_id"] for row in completed],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        if output is not None and output.exists():
            failure = {
                "schema": "scenesense.splitfusion_live_dispatch_phase13b_failure.v1",
                "status": "FAILED_NO_RETRY_AUTHORIZED",
                "operation": operation,
                "action_id": current_action,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_action_ids": [row["action_id"] for row in completed],
                "completed_records": completed_files,
            }
            try:
                _atomic_json(output / "FAILURE.json", failure)
            except Exception:
                pass
        raise
    finally:
        _close_runtime(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
