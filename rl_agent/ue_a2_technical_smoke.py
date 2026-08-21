"""Bounded offline preflight for the UE-A2 72-action technical smoke.

This runner deliberately separates what can be proved without CARLA, OAI, or
model inference from the later CUDA model smoke.  It exercises the production
per-channel quantizers, pickle/zstd envelope, deterministic UDP chunk format,
fail-closed profile identity, and the v2 map-packet construction path for all
72 registered actions.  It never emits a UE-A2 pass: every row remains
``REGISTERED_PENDING_SMOKE`` until strict front/edge model execution and real
localhost UDP have run.

The module imports PyTorch and the production codec lazily so registry and
error reporting remain usable when those runtime dependencies are missing.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import math
import pickle
import platform
import random
import socket
import struct
import sys
import tempfile
import time
import zlib
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent.ue_split_wire_contract import (  # noqa: E402
    A1_REGISTRY_CSV_SHA256,
    RegisteredSplitProfile,
    SplitWireContractError,
    build_launch_binding,
    load_registered_profiles,
    registry_row_fingerprint,
    resolve_registered_profile,
    sha256_file,
    validate_declared_feature_shapes,
    validate_feature_payload,
    validate_runtime_binding,
    validate_serialized_feature_headers,
    validate_wire_identity,
)


CONFIG_SCHEMA = "scenesense.ue_a2_technical_smoke_config.v1"
PREFLIGHT_SCHEMA = "scenesense.ue_a2_technical_smoke_preflight.v1"
MANIFEST_SCHEMA = "scenesense.ue_a2_technical_smoke_manifest.v1"
DEFAULT_CONFIG = ROOT / "rl_agent/configs/ue_a2_technical_smoke_v1.json"
UDP_HEADER = struct.Struct("!IHH")
MESSAGE_ID = 0x00A20001

PROFILE_FIELDS = (
    "action_index",
    "profile_id",
    "model_family",
    "quantization_mode",
    "quantization_bits",
    "roi_drop_fraction",
    "checkpoint_sha256",
    "action_contract_sha256",
    "registry_binding_status",
    "fixture_status",
    "production_codec_status",
    "in_memory_wire_status",
    "wire_shape_status",
    "map_schema_status",
    "model_front_status",
    "model_edge_status",
    "roi_execution_status",
    "ae_execution_status",
    "tail_execution_status",
    "actual_udp_status",
    "technical_validity_status",
    "blocking_codes",
    "pickle_bytes",
    "zstd_bytes",
    "udp_chunks",
    "payload_bytes_uncompressed",
    "max_quantization_abs_error",
    "compressed_sha256",
)


class A2TechnicalSmokeError(RuntimeError):
    """A classified UE-A2 preflight failure."""

    def __init__(self, code: str, detail: str, classification: str = "CONTRACT") -> None:
        self.code = str(code)
        self.detail = str(detail)
        self.classification = str(classification)
        super().__init__(f"{self.code}: {self.detail}")


def _fail(code: str, detail: str, classification: str = "CONTRACT") -> None:
    raise A2TechnicalSmokeError(code, detail, classification)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail("CONFIG_READ_FAILED", f"{path}: {exc}")
    if not isinstance(value, dict):
        _fail("CONFIG_INVALID", "top-level JSON must be an object")
    return value


def _import_pinned_source(
    path: Path,
    *,
    expected_sha256: str,
    module_prefix: str,
) -> Any:
    """Import one exact source file without trusting a same-named module.

    The repository's parent directory contains older modules with overlapping
    basenames.  A normal ``import carla_split_inference_udp_data_collect`` can
    therefore silently exercise the wrong codec depending on test order and
    ``sys.path`` state.  Use a digest-derived private module name, verify both
    file identity and the resulting ``__file__``, and restore ``sys.path``
    after module execution.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        _fail("PINNED_SOURCE_MISSING", str(source), "INFRASTRUCTURE")
    actual_sha256 = sha256_file(source)
    if actual_sha256 != str(expected_sha256):
        _fail(
            "PINNED_SOURCE_HASH_MISMATCH",
            f"expected={expected_sha256} actual={actual_sha256} path={source}",
        )
    module_name = f"_{module_prefix}_{actual_sha256[:16]}"
    cached = sys.modules.get(module_name)
    if cached is not None:
        cached_path = Path(str(getattr(cached, "__file__", ""))).resolve()
        if cached_path != source:
            _fail(
                "PINNED_SOURCE_CACHE_COLLISION",
                f"module={module_name} expected={source} actual={cached_path}",
            )
        return cached

    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        _fail("PINNED_SOURCE_IMPORT_SPEC_FAILED", str(source), "INFRASTRUCTURE")
    module = importlib.util.module_from_spec(spec)
    original_sys_path = list(sys.path)
    sys.modules[module_name] = module
    try:
        # Local dependencies must resolve beside the pinned source even when a
        # previously imported runtime placed neu_collab ahead of abiodun.
        sys.path[:] = [str(source.parent)] + [
            entry for entry in original_sys_path if Path(entry or ".").resolve() != source.parent
        ]
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        _fail(
            "PINNED_SOURCE_IMPORT_FAILED",
            f"{source}: {type(exc).__name__}: {exc}",
            "INFRASTRUCTURE",
        )
    finally:
        sys.path[:] = original_sys_path
    imported_path = Path(str(getattr(module, "__file__", ""))).resolve()
    if imported_path != source:
        sys.modules.pop(module_name, None)
        _fail(
            "PINNED_SOURCE_PATH_MISMATCH",
            f"expected={source} actual={imported_path}",
        )
    return module


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the frozen config and resolve all repository-relative paths."""

    config_path = Path(path).expanduser().resolve()
    config = _read_json(config_path)
    if config.get("schema") != CONFIG_SCHEMA:
        _fail("CONFIG_SCHEMA_MISMATCH", repr(config.get("schema")))
    repository_root = (config_path.parent / str(config.get("repository_root", ""))).resolve()
    if repository_root != ROOT:
        _fail(
            "REPOSITORY_ROOT_MISMATCH",
            f"expected={ROOT} resolved={repository_root}",
        )
    resolved = copy.deepcopy(config)
    resolved["config_path"] = str(config_path)
    resolved["config_sha256"] = sha256_file(config_path)
    resolved["repository_root"] = str(repository_root)
    for section, keys in (
        ("registry", ("path",)),
        ("fixture", ("path",)),
        ("sources", ("wire_contract_path", "runtime_path", "launcher_path", "codec_path")),
        ("output", ("root",)),
    ):
        payload = resolved.get(section)
        if not isinstance(payload, dict):
            _fail("CONFIG_SECTION_MISSING", section)
        for key in keys:
            raw = str(payload.get(key, ""))
            if not raw:
                _fail("CONFIG_FIELD_MISSING", f"{section}.{key}")
            payload[key] = str((repository_root / raw).resolve())
    codec_sha256 = str(resolved["sources"].get("codec_sha256", ""))
    if len(codec_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in codec_sha256
    ):
        _fail("CONFIG_CODEC_SHA256_INVALID", repr(codec_sha256))
    families = resolved["registry"].get("families", [])
    quantizers = resolved["registry"].get("quantizers", [])
    q_values = resolved["registry"].get("roi_drop_fractions", [])
    expected_counts = {
        "strict_front_loads": len(families),
        "strict_edge_loads": len(families),
        "native_backbone_encodes": len(families),
        "roi_ae_paths": len(families) * len(q_values),
        "tail_decodes": len(families) * len(q_values) * len(quantizers),
        "actual_udp_roundtrips": len(families) * len(q_values) * len(quantizers),
    }
    smoke = resolved.get("model_smoke")
    if not isinstance(smoke, Mapping):
        _fail("CONFIG_SECTION_MISSING", "model_smoke")
    for field, expected in expected_counts.items():
        actual = int(smoke.get(field, -1))
        if actual != expected:
            _fail(
                "CONFIG_MODEL_SMOKE_COUNT_MISMATCH",
                f"field={field} expected={expected} actual={actual}",
            )
    return resolved


def inspect_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    """Hash and validate the retained UE input without running a model."""

    fixture = config["fixture"]
    path = Path(str(fixture["path"]))
    if not path.is_file():
        _fail("FIXTURE_MISSING", str(path))
    actual_sha256 = sha256_file(path)
    expected_sha256 = str(fixture["sha256"])
    if actual_sha256 != expected_sha256:
        _fail(
            "FIXTURE_HASH_MISMATCH",
            f"expected={expected_sha256} actual={actual_sha256} path={path}",
        )
    actual_bytes = path.stat().st_size
    expected_bytes = int(fixture["bytes"])
    if actual_bytes != expected_bytes:
        _fail(
            "FIXTURE_SIZE_MISMATCH",
            f"expected={expected_bytes} actual={actual_bytes} path={path}",
        )
    try:
        import numpy as np
    except ImportError as exc:
        _fail("NUMPY_UNAVAILABLE", str(exc), "INFRASTRUCTURE")

    required = fixture.get("required_arrays")
    if not isinstance(required, Mapping):
        _fail("FIXTURE_CONTRACT_INVALID", "required_arrays must be a mapping")
    arrays: dict[str, Any] = {}
    try:
        with np.load(path, allow_pickle=False) as loaded:
            missing = sorted(set(required) - set(loaded.files))
            if missing:
                _fail("FIXTURE_ARRAYS_MISSING", ",".join(missing))
            for name, expected in required.items():
                array = loaded[str(name)]
                shape = tuple(int(item) for item in array.shape)
                wanted_shape = tuple(int(item) for item in expected["shape"])
                dtype = str(array.dtype)
                wanted_dtype = str(expected["dtype"])
                if shape != wanted_shape or dtype != wanted_dtype:
                    _fail(
                        "FIXTURE_ARRAY_CONTRACT_MISMATCH",
                        f"array={name} expected={wanted_shape}/{wanted_dtype} "
                        f"actual={shape}/{dtype}",
                    )
                if array.dtype.kind in "fc" and not bool(np.isfinite(array).all()):
                    _fail("FIXTURE_NONFINITE", str(name))
                arrays[str(name)] = {
                    "shape": list(shape),
                    "dtype": dtype,
                    "bytes": int(array.nbytes),
                }
            extra_arrays = sorted(set(loaded.files) - set(required))
            frame_id = int(loaded["frame_id"].reshape(-1)[0])
            carla_timestamp = float(loaded["carla_timestamp"].reshape(-1)[0])
    except A2TechnicalSmokeError:
        raise
    except Exception as exc:
        _fail("FIXTURE_LOAD_FAILED", f"{path}: {type(exc).__name__}: {exc}")
    return {
        "schema": "scenesense.ue_a2_fixture_manifest.v1",
        "status": "PASS",
        "path": str(path),
        "sha256": actual_sha256,
        "bytes": actual_bytes,
        "frame_id": frame_id,
        "carla_timestamp": carla_timestamp,
        "required_arrays": arrays,
        "extra_arrays_ignored": extra_arrays,
        "ground_truth_consumed": False,
        "phase2_logic_consumed": False,
    }


def verify_registry_matrix(
    config: Mapping[str, Any],
) -> tuple[tuple[RegisteredSplitProfile, ...], dict[str, Any]]:
    registry = config["registry"]
    profiles = load_registered_profiles(
        Path(str(registry["path"])),
        expected_registry_sha256=str(registry["sha256"]),
    )
    if len(profiles) != int(registry["expected_profiles"]):
        _fail(
            "PROFILE_COUNT_MISMATCH",
            f"expected={registry['expected_profiles']} actual={len(profiles)}",
        )
    observed_families = {profile.row["model_family"] for profile in profiles}
    observed_quantizers = {profile.row["quantization_mode"] for profile in profiles}
    observed_q = {profile.row["roi_drop_fraction"] for profile in profiles}
    for label, observed, expected in (
        ("families", observed_families, set(registry["families"])),
        ("quantizers", observed_quantizers, set(registry["quantizers"])),
        ("roi_drop_fractions", observed_q, set(registry["roi_drop_fractions"])),
    ):
        if observed != expected:
            _fail("PROFILE_FACTOR_SET_MISMATCH", f"{label}: expected={expected} actual={observed}")

    expected_grid = {
        (family, quantizer, q)
        for family in registry["families"]
        for quantizer in registry["quantizers"]
        for q in registry["roi_drop_fractions"]
    }
    observed_grid = {
        (
            profile.row["model_family"],
            profile.row["quantization_mode"],
            profile.row["roi_drop_fraction"],
        )
        for profile in profiles
    }
    if observed_grid != expected_grid:
        _fail(
            "PROFILE_GRID_MISMATCH",
            f"missing={sorted(expected_grid - observed_grid)} "
            f"extra={sorted(observed_grid - expected_grid)}",
        )

    checkpoint_audits: dict[str, dict[str, Any]] = {}
    launch_digests: dict[str, str] = {}
    frozen_transport = config["transport"]
    for profile in profiles:
        row = profile.row
        if row.get("quality_mask_applied") != "False":
            _fail("QUALITY_MASK_PRESENT", profile.profile_id)
        if row.get("technical_validity_status") != "REGISTERED_PENDING_SMOKE":
            _fail(
                "PREMATURE_TECHNICAL_STATUS",
                f"profile={profile.profile_id} status={row.get('technical_validity_status')}",
            )
        for field, expected in (
            ("entropy_coder", str(frozen_transport["entropy_coder"])),
            ("entropy_level", str(int(frozen_transport["entropy_level"]))),
            ("udp_chunk_bytes", str(int(frozen_transport["chunk_bytes"]))),
            ("udp_chunk_header_struct", str(frozen_transport["udp_chunk_header_struct"])),
        ):
            if row.get(field) != expected:
                _fail(
                    "PROFILE_TRANSPORT_CONFIG_MISMATCH",
                    f"profile={profile.profile_id} field={field} "
                    f"expected={expected!r} actual={row.get(field)!r}",
                )
        checkpoint_sha = row["checkpoint_sha256"]
        if checkpoint_sha not in checkpoint_audits:
            checkpoint_path = (ROOT / row["checkpoint_path"]).resolve()
            actual_sha = sha256_file(checkpoint_path)
            actual_bytes = checkpoint_path.stat().st_size
            if actual_sha != checkpoint_sha or actual_bytes != int(row["checkpoint_bytes"]):
                _fail(
                    "CHECKPOINT_SEAL_MISMATCH",
                    f"profile={profile.profile_id} path={checkpoint_path}",
                )
            checkpoint_audits[checkpoint_sha] = {
                "path": str(checkpoint_path),
                "sha256": actual_sha,
                "bytes": actual_bytes,
                "family": row["model_family"],
            }
        binding = build_launch_binding(profile)
        launch_digests[profile.profile_id] = hashlib.sha256(
            _canonical_json_bytes(binding)
        ).hexdigest()

    if len(checkpoint_audits) != 4:
        _fail("CHECKPOINT_FAMILY_COUNT_MISMATCH", str(len(checkpoint_audits)))
    return profiles, {
        "status": "PASS",
        "profiles": len(profiles),
        "quality_mask_applied": False,
        "unique_action_contracts": len(
            {profile.action_contract_sha256 for profile in profiles}
        ),
        "checkpoint_audits": list(checkpoint_audits.values()),
        "launch_binding_sha256": launch_digests,
    }


def _packed_data_bytes(total_values: int, bits: int) -> int:
    if bits == 8:
        return total_values
    if bits == 6:
        return ((total_values + 3) // 4) * 3
    if bits == 4:
        return (total_values + 1) // 2
    _fail("QUANTIZATION_BITS_INVALID", str(bits))


def structural_serialized_features(
    profile: RegisteredSplitProfile,
) -> dict[str, dict[str, bytes]]:
    """Build a header-valid payload for rejection-order and schema tests."""

    bits = int(profile.row["quantization_bits"])
    result: dict[str, dict[str, bytes]] = {}
    for level, shape in profile.expected_wire_shapes.items():
        _, channels, height, width = shape
        result[level] = {
            "header": struct.pack("!IIIB", channels, height, width, bits),
            "ranges": bytes(channels * 2 * 4),
            "data": bytes(_packed_data_bytes(channels * height * width, bits)),
        }
    return result


def build_structural_payload(profile: RegisteredSplitProfile) -> dict[str, Any]:
    return {
        "frame_id": 1,
        "batch_size": 1,
        "model_input_size": [
            int(profile.row["input_width"]),
            int(profile.row["input_height"]),
        ],
        "display_size": [1280, 720],
        "profile_identity": dict(profile.wire_identity),
        "feature_shapes": {
            level: list(shape) for level, shape in profile.expected_wire_shapes.items()
        },
        "features": structural_serialized_features(profile),
    }


def chunk_message(data: bytes, chunk_bytes: int, message_id: int = MESSAGE_ID) -> list[bytes]:
    if int(chunk_bytes) <= UDP_HEADER.size:
        _fail("CHUNK_BYTES_INVALID", str(chunk_bytes))
    max_payload = int(chunk_bytes) - UDP_HEADER.size
    total = max(1, math.ceil(len(data) / max_payload))
    if total > 0xFFFF:
        _fail("CHUNK_COUNT_OVERFLOW", str(total))
    return [
        UDP_HEADER.pack(int(message_id), index, total)
        + data[index * max_payload : (index + 1) * max_payload]
        for index in range(total)
    ]


def reassemble_chunks(chunks: Iterable[bytes]) -> bytes:
    observed: dict[int, bytes] = {}
    expected_message_id: int | None = None
    expected_total: int | None = None
    for packet in chunks:
        if len(packet) < UDP_HEADER.size:
            _fail("CHUNK_HEADER_TRUNCATED", str(len(packet)))
        message_id, index, total = UDP_HEADER.unpack(packet[: UDP_HEADER.size])
        if total <= 0 or index >= total:
            _fail("CHUNK_HEADER_INVALID", f"index={index} total={total}")
        if expected_message_id is None:
            expected_message_id, expected_total = int(message_id), int(total)
        elif message_id != expected_message_id or total != expected_total:
            _fail("CHUNK_MESSAGE_MISMATCH", f"message={message_id} total={total}")
        if int(index) in observed:
            _fail("CHUNK_DUPLICATE", str(index))
        observed[int(index)] = packet[UDP_HEADER.size :]
    if expected_total is None:
        _fail("CHUNK_SET_EMPTY", "no chunks supplied")
    missing = sorted(set(range(expected_total)) - set(observed))
    if missing:
        _fail("CHUNK_SET_INCOMPLETE", f"missing={missing}")
    return b"".join(observed[index] for index in range(expected_total))


def _decode_outer_payload(coder: Any, compressed: bytes) -> Any:
    try:
        return pickle.loads(coder.decompress(compressed))
    except Exception as exc:
        _fail(
            "OUTER_DECOMPRESSION_REJECTED",
            f"{type(exc).__name__}: {exc}",
        )


def outer_wire_roundtrip(
    payload: Mapping[str, Any],
    *,
    coder: Any,
    chunk_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = pickle.dumps(dict(payload), protocol=pickle.HIGHEST_PROTOCOL)
    compressed = coder.compress(raw)
    chunks = chunk_message(compressed, int(chunk_bytes))
    # Deliberately reverse the datagrams: the production assembler is indexed,
    # not arrival-order dependent.
    reassembled = reassemble_chunks(reversed(chunks))
    decoded = _decode_outer_payload(coder, reassembled)
    if not isinstance(decoded, dict):
        _fail("OUTER_PAYLOAD_TYPE_MISMATCH", type(decoded).__name__)
    return decoded, {
        "pickle_bytes": len(raw),
        "zstd_bytes": len(compressed),
        "udp_chunks": len(chunks),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
    }


def _synthetic_tensor(shape: tuple[int, ...], torch_module: Any) -> Any:
    _, channels, height, width = shape
    row = torch_module.linspace(-1.0, 1.0, steps=height, dtype=torch_module.float32)
    col = torch_module.linspace(-0.5, 0.5, steps=width, dtype=torch_module.float32)
    spatial = row.view(1, 1, height, 1) + col.view(1, 1, 1, width)
    offsets = (
        torch_module.arange(channels, dtype=torch_module.float32)
        .remainder(19)
        .view(1, channels, 1, 1)
        / 19.0
    )
    tensor = (spatial + offsets).contiguous()
    # Exercise the zero-span per-channel branch as part of every level.
    tensor[:, -1, :, :] = 0.125
    return tensor


def _quantization_error(
    original: Any, decoded: Any, bits: int, torch_module: Any
) -> float:
    flat = original.reshape(original.shape[1], -1)
    span = flat.max(dim=1).values - flat.min(dim=1).values
    error = (original - decoded).abs().reshape(original.shape[1], -1).max(dim=1).values
    bound = span / (2.0 * float((1 << int(bits)) - 1)) + 2e-6
    if bool((error > bound).any().item()):
        index = int(torch_module.argmax(error - bound).item())
        _fail(
            "QUANTIZATION_ERROR_BOUND_EXCEEDED",
            f"channel={index} error={float(error[index])} bound={float(bound[index])}",
        )
    return float(error.max().item())


def run_production_codec_matrix(
    profiles: Sequence[RegisteredSplitProfile],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run all 72 production codec paths on deterministic CPU features."""

    try:
        import torch
    except Exception as exc:
        _fail(
            "PRODUCTION_CODEC_IMPORT_FAILED",
            f"{type(exc).__name__}: {exc}",
            "INFRASTRUCTURE",
        )
    sources = config["sources"]
    codec = _import_pinned_source(
        Path(str(sources["codec_path"])),
        expected_sha256=str(sources["codec_sha256"]),
        module_prefix="ue_a2_production_codec",
    )
    for required_name in (
        "TransportConfig",
        "serialize_feature_maps",
        "deserialize_feature_maps",
        "QUANT_MODE_PER_CHANNEL_UINT6",
    ):
        if not hasattr(codec, required_name):
            _fail(
                "PRODUCTION_CODEC_API_MISSING",
                f"path={sources['codec_path']} symbol={required_name}",
                "IMPLEMENTATION",
            )

    feature_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], OrderedDict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    max_message = 0
    total_message = 0
    started = time.perf_counter()
    for profile in profiles:
        row = profile.row
        shape_key = (
            profile.expected_wire_shapes["low"],
            profile.expected_wire_shapes["high"],
        )
        features = feature_cache.get(shape_key)
        if features is None:
            features = OrderedDict(
                (
                    (level, _synthetic_tensor(shape, torch))
                    for level, shape in profile.expected_wire_shapes.items()
                )
            )
            feature_cache[shape_key] = features

        transport = codec.TransportConfig(
            quantization_mode=str(row["quantization_mode"]),
            entropy_coder_name=str(row["entropy_coder"]),
            zstd_level=int(row["entropy_level"]),
            roi_objectness_threshold=0.0,
            bypass_rcnn_transform=False,
        )
        front_codecs: MutableMapping[str, object] = OrderedDict()
        serialized, payload_uncompressed, _, _ = codec.serialize_feature_maps(
            features,
            front_codecs,
            quantization_mode=transport.quantization_mode,
            per_level_compress_probe=False,
            entropy_coder=transport.make_entropy_coder(),
        )
        validate_serialized_feature_headers(profile, serialized)
        payload = {
            "frame_id": 1,
            "batch_size": 1,
            "model_input_size": [int(row["input_width"]), int(row["input_height"])],
            "display_size": [1280, 720],
            "profile_identity": dict(profile.wire_identity),
            "feature_shapes": {
                level: list(tensor.shape) for level, tensor in features.items()
            },
            "features": serialized,
            "payload_bytes_uncompressed": int(payload_uncompressed),
        }
        decoded_payload, wire = outer_wire_roundtrip(
            payload,
            coder=transport.make_entropy_coder(),
            chunk_bytes=int(row["udp_chunk_bytes"]),
        )
        validate_feature_payload(profile, decoded_payload)
        edge_codecs: MutableMapping[str, object] = OrderedDict()
        decoded_features = codec.deserialize_feature_maps(
            decoded_payload["features"],
            torch.device("cpu"),
            batch_size=1,
            feature_codecs=edge_codecs,
            quantization_mode=transport.quantization_mode,
        )
        observed_shapes = {
            level: tuple(int(value) for value in tensor.shape)
            for level, tensor in decoded_features.items()
        }
        validate_declared_feature_shapes(profile, observed_shapes, stage="wire")
        max_error = 0.0
        for level in profile.expected_wire_shapes:
            decoded_tensor = decoded_features[level]
            if decoded_tensor.dtype != torch.float32 or not bool(
                torch.isfinite(decoded_tensor).all().item()
            ):
                _fail("DEQUANTIZED_FEATURE_INVALID", f"{profile.profile_id}:{level}")
            max_error = max(
                max_error,
                _quantization_error(
                    features[level],
                    decoded_tensor,
                    int(row["quantization_bits"]),
                    torch,
                ),
            )
        max_message = max(max_message, int(wire["zstd_bytes"]))
        total_message += int(wire["zstd_bytes"])
        rows.append(
            {
                "action_index": int(row["action_index"]),
                "profile_id": profile.profile_id,
                "model_family": row["model_family"],
                "quantization_mode": row["quantization_mode"],
                "quantization_bits": int(row["quantization_bits"]),
                "roi_drop_fraction": row["roi_drop_fraction"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "action_contract_sha256": profile.action_contract_sha256,
                "registry_binding_status": "PASS",
                "fixture_status": "PASS",
                "production_codec_status": "PASS_CPU_SYNTHETIC_FEATURES",
                "in_memory_wire_status": "PASS",
                "wire_shape_status": "PASS",
                "map_schema_status": "PENDING_RUNTIME_PROBE",
                "model_front_status": "NOT_EXECUTED",
                "model_edge_status": "NOT_EXECUTED",
                "roi_execution_status": "NOT_EXECUTED",
                "ae_execution_status": "NOT_EXECUTED",
                "tail_execution_status": "NOT_EXECUTED",
                "actual_udp_status": "NOT_EXECUTED",
                "technical_validity_status": "REGISTERED_PENDING_SMOKE",
                "blocking_codes": "MODEL_AND_ACTUAL_UDP_PENDING",
                "payload_bytes_uncompressed": int(payload_uncompressed),
                "max_quantization_abs_error": max_error,
                **wire,
            }
        )
    if len(rows) != 72:
        _fail("CODEC_MATRIX_ROW_COUNT_MISMATCH", str(len(rows)))
    return rows, {
        "status": "PASS",
        "profiles": len(rows),
        "device": "cpu",
        "model_inference_executed": False,
        "feature_source": "deterministic_synthetic_registered_wire_shapes",
        "max_zstd_message_bytes": max_message,
        "total_zstd_message_bytes": total_message,
        "elapsed_s": time.perf_counter() - started,
        "codec_path": str(Path(str(sources["codec_path"])).resolve()),
        "codec_sha256": str(sources["codec_sha256"]),
    }


def _runtime_args(profile: RegisteredSplitProfile, role: str = "back") -> dict[str, Any]:
    row = profile.row
    args: dict[str, Any] = {
        "role": role,
        "ue_profile_id": profile.profile_id,
        "quantization_mode": row["quantization_mode"],
        "roi_threshold": float(row["roi_drop_fraction"]),
        "entropy_coder": row["entropy_coder"],
        "zstd_level": int(row["entropy_level"]),
        "chunk_bytes": int(row["udp_chunk_bytes"]),
        "model_input_width": int(row["input_width"]),
        "model_input_height": int(row["input_height"]),
        "ae_checkpoint": "",
        "object_score_threshold": float(row["object_score_threshold"]),
        "object_nms_radius_px": int(row["object_nms_radius_px"]),
        "topk_objects": int(row["topk_objects"]),
        "max_objects_drawn": int(row["max_objects_published"]),
    }
    return args


def _model_runtime_args(
    profile: RegisteredSplitProfile,
    *,
    role: str,
) -> SimpleNamespace:
    """Build the minimal strict runtime namespace for one model load."""

    checkpoint = (ROOT / profile.row["checkpoint_path"]).resolve()
    args = _runtime_args(profile, role=role)
    args.update(
        {
            "fusion_checkpoint": str(checkpoint),
            "fusion_experiment_dir": "",
            "num_classes": 3,
            "object_hidden_channels": 128,
            "ue_profile_registry_csv": str(profile.registry_path),
            "require_ue_profile_binding": True,
        }
    )
    return SimpleNamespace(**args)


def _load_fixture_arrays(config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import numpy as np

        with np.load(Path(str(config["fixture"]["path"])), allow_pickle=False) as loaded:
            return {
                "frame_bgr": loaded["frame_bgr"].copy(),
                "radar_tensor": loaded["radar_tensor"].copy(),
                "camera_matrix": loaded["camera_matrix"].copy(),
                "camera_intrinsics_input": loaded["camera_intrinsics_input"].copy(),
                "frame_id": int(loaded["frame_id"].reshape(-1)[0]),
                "carla_timestamp": float(loaded["carla_timestamp"].reshape(-1)[0]),
            }
    except Exception as exc:
        _fail("MODEL_FIXTURE_LOAD_FAILED", f"{type(exc).__name__}: {exc}")


def _assert_runtime_codec_binding(
    runtime: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_path = Path(str(config["sources"]["codec_path"])).resolve()
    expected_sha256 = str(config["sources"]["codec_sha256"])
    observed_path = Path(str(getattr(runtime.od_collect, "__file__", ""))).resolve()
    if observed_path != expected_path:
        _fail(
            "RUNTIME_CODEC_PATH_MISMATCH",
            f"expected={expected_path} actual={observed_path}",
            "IMPLEMENTATION",
        )
    observed_sha256 = sha256_file(observed_path)
    if observed_sha256 != expected_sha256:
        _fail(
            "RUNTIME_CODEC_HASH_MISMATCH",
            f"expected={expected_sha256} actual={observed_sha256}",
            "IMPLEMENTATION",
        )
    return {"path": str(observed_path), "sha256": observed_sha256}


def run_negative_contract_tests(
    profiles: Sequence[RegisteredSplitProfile],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Inject bounded contract mutations without invoking feature decode."""

    resolved_config = config if config is not None else load_config(DEFAULT_CONFIG)
    profile = next(
        item
        for item in profiles
        if item.row["model_family"] == "ae32"
        and item.row["quantization_mode"] == "per_channel_uint4"
        and item.row["roi_drop_fraction"] == "0.9"
    )
    cross_profile = next(item for item in profiles if item.profile_id != profile.profile_id)
    base = build_structural_payload(profile)
    records: list[dict[str, Any]] = []

    def expect(name: str, expected: str, callback: Callable[[], Any]) -> None:
        observed = "NO_REJECTION"
        detail = ""
        try:
            callback()
        except (SplitWireContractError, A2TechnicalSmokeError) as exc:
            observed = exc.code
            detail = str(exc)
        except Exception as exc:  # An unexpected exception is evidence, never a pass.
            observed = f"UNEXPECTED_{type(exc).__name__}"
            detail = str(exc)
        records.append(
            {
                "name": name,
                "expected_code": expected,
                "observed_code": observed,
                "status": "PASS" if observed == expected else "FAIL",
                "detail": detail,
            }
        )

    missing_identity = copy.deepcopy(base)
    missing_identity.pop("profile_identity")
    expect(
        "missing_identity",
        "WIRE_IDENTITY_MISSING",
        lambda: validate_feature_payload(profile, missing_identity),
    )
    identity_before_features = copy.deepcopy(base)
    identity_before_features["profile_identity"]["profile_id"] = "wrong"
    identity_before_features["features"] = "must-not-be-inspected"
    expect(
        "identity_rejected_before_feature_inspection",
        "WIRE_IDENTITY_VALUE_MISMATCH",
        lambda: validate_feature_payload(profile, identity_before_features),
    )
    for field, value in profile.wire_identity.items():
        changed = copy.deepcopy(base)
        changed["profile_identity"][field] = (
            value + 1 if isinstance(value, int) else f"{value}-bad"
        )
        expect(
            f"identity_{field}",
            "WIRE_IDENTITY_VALUE_MISMATCH",
            lambda changed=changed: validate_feature_payload(profile, changed),
        )
    cross = copy.deepcopy(base)
    cross["profile_identity"] = dict(cross_profile.wire_identity)
    expect(
        "cross_profile_payload",
        "WIRE_IDENTITY_VALUE_MISMATCH",
        lambda: validate_feature_payload(profile, cross),
    )
    wrong_feature_names = copy.deepcopy(base)
    wrong_feature_names["feature_shapes"]["wrong_high"] = (
        wrong_feature_names["feature_shapes"].pop("high")
    )
    wrong_feature_names["features"]["wrong_high"] = wrong_feature_names[
        "features"
    ].pop("high")
    expect(
        "wrong_feature_names",
        "FEATURE_LEVELS_MISMATCH",
        lambda: validate_feature_payload(profile, wrong_feature_names),
    )
    bad_shape = copy.deepcopy(base)
    bad_shape["feature_shapes"]["high"][1] += 1
    expect(
        "declared_high_shape",
        "FEATURE_SHAPE_MISMATCH",
        lambda: validate_feature_payload(profile, bad_shape),
    )
    bad_header = copy.deepcopy(base)
    _, channels, height, width = profile.expected_wire_shapes["high"]
    bad_header["features"]["high"]["header"] = struct.pack(
        "!IIIB", channels + 1, height, width, int(profile.row["quantization_bits"])
    )
    expect(
        "serialized_high_header",
        "FEATURE_WIRE_HEADER_MISMATCH",
        lambda: validate_feature_payload(profile, bad_header),
    )
    expect(
        "unknown_profile_resolution",
        "PROFILE_NOT_FOUND",
        lambda: resolve_registered_profile(
            "unknown-profile",
            profile.registry_path,
            expected_registry_sha256=A1_REGISTRY_CSV_SHA256,
        ),
    )

    # Exercise the production runtime's strict-binding entry point.  An empty
    # profile ID must fail before checkpoint resolution or model construction.
    runtime = _import_runtime(Path(str(resolved_config["sources"]["runtime_path"])))
    incomplete_binding = SimpleNamespace(
        require_ue_profile_binding=True,
        ue_profile_registry_csv=str(profile.registry_path),
        ue_profile_id="",
    )
    expect(
        "missing_strict_binding_arguments",
        "REGISTERED_PROFILE_ARGUMENTS_INCOMPLETE",
        lambda: runtime._resolve_registered_ue_profile(incomplete_binding),
    )

    # Registry corruption probes use disposable 72-row copies.  The duplicate
    # row receives a valid new row fingerprint so the duplicate-ID gate itself
    # is reached; the fingerprint probe deliberately retains an invalid seal.
    source_registry_sha256_before = sha256_file(profile.registry_path)
    try:
        with profile.registry_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            registry_fieldnames = list(reader.fieldnames or ())
            registry_rows = [dict(row) for row in reader]
    except (OSError, csv.Error) as exc:
        _fail("NEGATIVE_REGISTRY_COPY_READ_FAILED", str(exc), "INFRASTRUCTURE")
    if len(registry_rows) != 72 or not registry_fieldnames:
        _fail(
            "NEGATIVE_REGISTRY_COPY_INVALID",
            f"rows={len(registry_rows)} fields={len(registry_fieldnames)}",
        )

    def write_registry_copy(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        try:
            with path.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=registry_fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except (OSError, csv.Error) as exc:
            _fail("NEGATIVE_REGISTRY_COPY_WRITE_FAILED", str(exc), "INFRASTRUCTURE")

    with tempfile.TemporaryDirectory(prefix="ue-a2-negative-registry-") as directory:
        temporary_root = Path(directory)

        duplicate_rows = copy.deepcopy(registry_rows)
        duplicate_rows[1]["profile_id"] = duplicate_rows[0]["profile_id"]
        duplicate_rows[1]["row_fingerprint_sha256"] = registry_row_fingerprint(
            duplicate_rows[1]
        )
        duplicate_path = temporary_root / "duplicate_profile_id.csv"
        write_registry_copy(duplicate_path, duplicate_rows)
        expect(
            "duplicate_profile_id_registry",
            "PROFILE_ID_DUPLICATE",
            lambda: load_registered_profiles(
                duplicate_path,
                expected_registry_sha256=sha256_file(duplicate_path),
            ),
        )

        corrupt_rows = copy.deepcopy(registry_rows)
        corrupt_rows[0]["row_fingerprint_sha256"] = "0" * 64
        corrupt_path = temporary_root / "corrupt_row_fingerprint.csv"
        write_registry_copy(corrupt_path, corrupt_rows)
        expect(
            "corrupt_registry_row_fingerprint",
            "REGISTRY_ROW_FINGERPRINT_MISMATCH",
            lambda: load_registered_profiles(
                corrupt_path,
                expected_registry_sha256=sha256_file(corrupt_path),
            ),
        )

    source_registry_sha256_after = sha256_file(profile.registry_path)
    if source_registry_sha256_after != source_registry_sha256_before:
        _fail(
            "SOURCE_REGISTRY_MUTATED_DURING_NEGATIVE_TESTS",
            f"before={source_registry_sha256_before} after={source_registry_sha256_after}",
        )

    checkpoint = ROOT / profile.row["checkpoint_path"]
    runtime_mutations = {
        "quantization_mode": "per_channel_uint8",
        "roi_threshold": 0.7,
        "entropy_coder": "zlib",
        "zstd_level": 2,
        "chunk_bytes": 59999,
        "object_score_threshold": 0.05,
        "object_nms_radius_px": 4,
        "topk_objects": 80,
        "max_objects_drawn": 30,
    }
    for field, value in runtime_mutations.items():
        args = _runtime_args(profile)
        args[field] = value
        expect(
            f"runtime_{field}",
            "RUNTIME_BINDING_MISMATCH",
            lambda args=args: validate_runtime_binding(
                profile, args, checkpoint_path=checkpoint
            ),
        )
    external_ae = _runtime_args(profile)
    external_ae["ae_checkpoint"] = "/tmp/forbidden-ae.pt"
    expect(
        "external_ae_override",
        "EXTERNAL_AE_OVERRIDE_FORBIDDEN",
        lambda: validate_runtime_binding(profile, external_ae, checkpoint_path=checkpoint),
    )

    chunks = chunk_message(b"x" * 120000, 60000)
    expect(
        "missing_udp_chunk",
        "CHUNK_SET_INCOMPLETE",
        lambda: reassemble_chunks(chunks[:-1]),
    )
    expect(
        "duplicate_udp_chunk",
        "CHUNK_DUPLICATE",
        lambda: reassemble_chunks([chunks[0], chunks[0], *chunks[1:]]),
    )

    failed = [record["name"] for record in records if record["status"] != "PASS"]
    if failed:
        _fail("NEGATIVE_TESTS_FAILED", ",".join(failed))
    return {
        "schema": "scenesense.ue_a2_negative_contract_tests.v1",
        "status": "PASS",
        "tests": len(records),
        "decode_or_map_calls_after_rejection": 0,
        "temporary_registry_copies_only": True,
        "source_registry_sha256_before": source_registry_sha256_before,
        "source_registry_sha256_after": source_registry_sha256_after,
        "records": records,
    }


def _import_runtime(path: Path) -> Any:
    name = f"ue_a2_runtime_probe_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _fail("RUNTIME_IMPORT_SPEC_FAILED", str(path), "IMPLEMENTATION")
    module = importlib.util.module_from_spec(spec)
    original_sys_path = list(sys.path)
    modules_before = set(sys.modules)
    # The runtime conditionally prepends neu_collab when absent.  Make both
    # roots present with abiodun first, then restore the caller's exact path.
    ordered_roots = (str(ROOT), str(ROOT.parent))
    sys.path[:] = list(ordered_roots) + [
        entry
        for entry in original_sys_path
        if str(Path(entry or ".").resolve()) not in ordered_roots
    ]
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        _fail(
            "RUNTIME_IMPORT_FAILED",
            f"{path}: {type(exc).__name__}: {exc}",
            "IMPLEMENTATION",
        )
    finally:
        sys.path[:] = original_sys_path
        fusion_root = (ROOT / "pole_lraspp_multimodal_fusion").resolve()
        for imported_name in set(sys.modules) - modules_before:
            imported = sys.modules.get(imported_name)
            imported_file = str(getattr(imported, "__file__", "") or "")
            if not imported_file:
                continue
            try:
                belongs_to_fusion_tree = Path(imported_file).resolve().is_relative_to(
                    fusion_root
                )
            except (OSError, ValueError):
                belongs_to_fusion_tree = False
            if belongs_to_fusion_tree:
                sys.modules.pop(imported_name, None)
    return module


def _validate_map_packet(
    packet: Any,
    profile: RegisteredSplitProfile,
    config: Mapping[str, Any],
    *,
    require_nonempty_object: bool = True,
) -> dict[str, Any]:
    if not isinstance(packet, Mapping):
        _fail("MAP_PACKET_INVALID", type(packet).__name__, "IMPLEMENTATION")
    expected_schema = str(config["map_contract"]["schema"])
    if packet.get("schema") != expected_schema:
        _fail(
            "MAP_SCHEMA_MISMATCH",
            f"expected={expected_schema} actual={packet.get('schema')}",
            "IMPLEMENTATION",
        )
    validate_wire_identity(packet.get("profile_identity"), profile.wire_identity)
    required = {
        "stream_id",
        "node_id",
        "frame_id",
        "timestamp",
        "carla_timestamp",
        "camera",
        "segmentation",
        "objects",
        "latency",
    }
    missing = sorted(required - set(packet))
    if missing:
        _fail("MAP_FIELDS_MISSING", ",".join(missing), "IMPLEMENTATION")
    objects = packet["objects"]
    if not isinstance(objects, list):
        _fail("MAP_OBJECTS_INVALID", type(objects).__name__, "IMPLEMENTATION")
    if require_nonempty_object and len(objects) != 1:
        _fail("MAP_SYNTHETIC_OBJECT_MISSING", repr(objects), "IMPLEMENTATION")
    for obj in objects:
        for field in ("id", "type", "score", "location", "dimensions", "center_px"):
            if field not in obj:
                _fail("MAP_OBJECT_FIELD_MISSING", field, "IMPLEMENTATION")
    encoded = json.dumps(
        packet, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    compressed = zlib.compress(encoded, level=1)
    maximum = int(config["map_contract"]["max_udp_datagram_bytes"])
    if len(compressed) > maximum:
        _fail(
            "MAP_PACKET_OVERSIZED",
            f"bytes={len(compressed)} max={maximum}",
            "IMPLEMENTATION",
        )
    return {
        "json_bytes": len(encoded),
        "zlib_bytes": len(compressed),
        "object_count": len(objects),
        "schema": expected_schema,
    }


def run_runtime_map_probes(
    profiles: Sequence[RegisteredSplitProfile],
    config: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Invoke the real v2 publisher method without sockets or background threads."""

    runtime_path = Path(str(config["sources"]["runtime_path"]))
    runtime = _import_runtime(runtime_path)
    fixture_path = Path(str(config["fixture"]["path"]))
    try:
        import numpy as np

        with np.load(fixture_path, allow_pickle=False) as loaded:
            camera_matrix = loaded["camera_matrix"].copy()
            frame_id = int(loaded["frame_id"].reshape(-1)[0])
            carla_timestamp = float(loaded["carla_timestamp"].reshape(-1)[0])
    except Exception as exc:
        _fail("MAP_FIXTURE_LOAD_FAILED", f"{type(exc).__name__}: {exc}")

    statuses: dict[str, str] = {}
    details: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for profile in profiles:
        captured: list[dict[str, Any]] = []
        publisher = object.__new__(runtime.SpatialMapResultPublisher)
        publisher.stream_id = "ue_a2_fixture"
        publisher.traffic_light_id = ""
        publisher.traffic_light_actor_id = -1
        publisher.traffic_light_opendrive_id = ""
        publisher.camera_width = 1280
        publisher.camera_height = 720
        publisher.camera_fov = 120.0
        publisher._enqueue = lambda payload, _frame_id: captured.append(payload)
        source_payload = {
            "frame_id": frame_id,
            "stream_id": "ue_a2_fixture",
            "carla_timestamp": carla_timestamp,
            "camera_matrix": camera_matrix,
            "camera_transform": {},
            "profile_identity": dict(profile.wire_identity),
            "payload_bytes": 1234,
            "payload_bytes_uncompressed": 5678,
            "payload_chunks": 1,
        }
        result = {
            "frame_id": frame_id,
            "server_ms": 1.0,
            "mask": np.zeros((432, 768), dtype=np.uint8),
            "objects": [
                {
                    "class_name": "vehicle",
                    "score": 0.9,
                    "center_x_px": 384.0,
                    "center_y_px": 216.0,
                    "world_x": 1.0,
                    "world_y": 2.0,
                    "world_z": 0.5,
                    "size_x": 4.2,
                    "size_y": 1.8,
                    "size_z": 1.5,
                    "yaw_deg": 5.0,
                    "parked_score": 0.1,
                    "radar_support_score": 0.8,
                    "bbox_xyxy": [300.0, 150.0, 460.0, 320.0],
                }
            ],
        }
        try:
            publisher.publish_from_payload(
                source_payload=source_payload,
                result=result,
                timing={"front_compute_ms": 1.0, "t_capture_perf": 1.0, "t_tail_done_perf": 1.01},
            )
            if len(captured) != 1:
                _fail(
                    "MAP_PACKET_CAPTURE_COUNT_MISMATCH",
                    f"profile={profile.profile_id} count={len(captured)}",
                    "IMPLEMENTATION",
                )
            details[profile.profile_id] = _validate_map_packet(captured[0], profile, config)
            statuses[profile.profile_id] = "PASS"
        except (A2TechnicalSmokeError, SplitWireContractError) as exc:
            statuses[profile.profile_id] = "FAIL"
            failures[profile.profile_id] = getattr(exc, "code", type(exc).__name__)
        except Exception as exc:
            statuses[profile.profile_id] = "FAIL"
            failures[profile.profile_id] = f"UNEXPECTED_{type(exc).__name__}"
    return statuses, {
        "status": "PASS" if not failures else "BLOCKED_IMPLEMENTATION",
        "profiles_passed": len(profiles) - len(failures),
        "profiles_failed": len(failures),
        "failures": failures,
        "details": details,
        "runtime_path": str(runtime_path),
        "runtime_sha256": sha256_file(runtime_path),
        "carla_connected": False,
        "sockets_created": False,
    }


def inspect_runtime_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_required_for_final_smoke": bool(
            config["scope"]["cuda_required_for_final_smoke"]
        ),
        "model_inference_executed": False,
        "actual_udp_executed": False,
        "carla_run": False,
        "oai_run": False,
        "blockers": [],
    }
    try:
        import torch

        result["torch_version"] = str(torch.__version__)
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            result["cuda_device_count"] = int(torch.cuda.device_count())
            result["cuda_device_name"] = str(torch.cuda.get_device_name(0))
        elif result["cuda_required_for_final_smoke"]:
            result["blockers"].append("CUDA_UNAVAILABLE")
    except Exception as exc:
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
        result["cuda_available"] = False
        result["blockers"].append("TORCH_UNAVAILABLE")
    try:
        import zstandard

        result["zstandard_version"] = str(zstandard.__version__)
    except Exception as exc:
        result["zstandard_error"] = f"{type(exc).__name__}: {exc}"
        result["blockers"].append("ZSTD_UNAVAILABLE")
    return result


def inspect_socket_buffers(
    config: Mapping[str, Any], *, observed_max_message_bytes: int
) -> dict[str, Any]:
    transport = config["transport"]
    historical = int(transport["historical_max_message_bytes"])
    design_message = max(int(observed_max_message_bytes), historical)
    margin = float(transport["buffer_margin_ratio"])
    required = int(math.ceil(design_message * margin))
    requested = int(transport["requested_socket_buffer_bytes"])
    result: dict[str, Any] = {
        "requested_socket_buffer_bytes": requested,
        "observed_synthetic_max_message_bytes": int(observed_max_message_bytes),
        "historical_max_message_bytes": historical,
        "design_max_message_bytes": design_message,
        "buffer_margin_ratio": margin,
        "required_reported_receive_buffer_bytes": required,
        "actual_udp_executed": False,
        "status": "PENDING",
        "blockers": [],
    }
    sock: socket.socket | None = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, requested)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, requested)
        result["reported_receive_buffer_bytes"] = int(
            sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        )
        result["reported_send_buffer_bytes"] = int(
            sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        )
        if result["reported_receive_buffer_bytes"] < required:
            result["status"] = "BLOCKED_INFRASTRUCTURE_BUFFER"
            result["blockers"].append("RECEIVE_BUFFER_BELOW_FROZEN_MARGIN")
        else:
            result["status"] = "READY_FOR_ACTUAL_UDP_SMOKE"
    except PermissionError as exc:
        result["status"] = "BLOCKED_INFRASTRUCTURE_SOCKET_PERMISSION"
        result["socket_error"] = f"{type(exc).__name__}: {exc}"
        result["blockers"].append("SOCKET_PERMISSION_DENIED")
    except OSError as exc:
        result["status"] = "BLOCKED_INFRASTRUCTURE_SOCKET"
        result["socket_error"] = f"{type(exc).__name__}: {exc}"
        result["blockers"].append("SOCKET_PREFLIGHT_FAILED")
    finally:
        if sock is not None:
            sock.close()
    return result


class ActualUDPLoopback:
    """One bounded localhost UDP pair using the pinned production transport."""

    def __init__(self, codec: Any, config: Mapping[str, Any]) -> None:
        transport_cfg = config["transport"]
        smoke_cfg = config["model_smoke"]
        self.codec = codec
        self.config = config
        self.receiver: Any = None
        self.sender: Any = None
        self.messages_sent = 0
        self.messages_received = 0
        self.total_compressed_bytes = 0
        self.max_compressed_bytes = 0
        self.total_chunks = 0
        host = str(smoke_cfg["socket_host"])
        chunk_bytes = int(transport_cfg["chunk_bytes"])
        timeout_s = float(smoke_cfg["socket_timeout_s"])
        requested = int(transport_cfg["requested_socket_buffer_bytes"])
        coder_transport = codec.TransportConfig(
            quantization_mode="per_channel_uint8",
            entropy_coder_name=str(transport_cfg["entropy_coder"]),
            zstd_level=int(transport_cfg["entropy_level"]),
            roi_objectness_threshold=0.0,
            bypass_rcnn_transform=False,
        )
        try:
            self.receiver = codec.UDPMessageSocket(
                bind_port=0,
                remote_port=None,
                chunk_bytes=chunk_bytes,
                socket_timeout=timeout_s,
                host=host,
                entropy_coder=coder_transport.make_entropy_coder(),
            )
            receiver_port = int(self.receiver.socket.getsockname()[1])
            self.sender = codec.UDPMessageSocket(
                bind_port=0,
                remote_port=receiver_port,
                chunk_bytes=chunk_bytes,
                socket_timeout=timeout_s,
                host=host,
                remote_host=host,
                entropy_coder=coder_transport.make_entropy_coder(),
            )
            self.receiver.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, requested)
            self.sender.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, requested)
        except PermissionError as exc:
            self.close()
            _fail(
                "ACTUAL_UDP_PERMISSION_DENIED",
                str(exc),
                "INFRASTRUCTURE",
            )
        except OSError as exc:
            self.close()
            _fail("ACTUAL_UDP_SOCKET_FAILED", str(exc), "INFRASTRUCTURE")

        self.requested_socket_buffer_bytes = requested
        self.reported_receive_buffer_bytes = int(
            self.receiver.socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        )
        self.reported_send_buffer_bytes = int(
            self.sender.socket.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        )
        historical = int(transport_cfg["historical_max_message_bytes"])
        margin = float(transport_cfg["buffer_margin_ratio"])
        required = int(math.ceil(historical * margin))
        if self.reported_receive_buffer_bytes < required:
            self.close()
            _fail(
                "ACTUAL_UDP_BUFFER_BELOW_MARGIN",
                f"required={required} actual={self.reported_receive_buffer_bytes}",
                "INFRASTRUCTURE",
            )

    def roundtrip(
        self,
        payload: Mapping[str, Any],
        *,
        expected_compressed_bytes: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        margin = float(self.config["transport"]["buffer_margin_ratio"])
        required = int(math.ceil(int(expected_compressed_bytes) * margin))
        if self.reported_receive_buffer_bytes < required:
            _fail(
                "ACTUAL_UDP_MESSAGE_EXCEEDS_BUFFER_MARGIN",
                f"message={expected_compressed_bytes} required={required} "
                f"actual={self.reported_receive_buffer_bytes}",
                "INFRASTRUCTURE",
            )
        started = time.perf_counter()
        try:
            compressed_bytes, chunks = self.sender.send(dict(payload))
            received = self.receiver.receive()
        except PermissionError as exc:
            _fail("ACTUAL_UDP_PERMISSION_DENIED", str(exc), "INFRASTRUCTURE")
        except OSError as exc:
            _fail("ACTUAL_UDP_IO_FAILED", str(exc), "INFRASTRUCTURE")
        if received is None:
            _fail(
                "ACTUAL_UDP_LOSS_OR_TIMEOUT",
                f"profile={payload.get('profile_identity', {}).get('profile_id', '')} "
                f"message={expected_compressed_bytes} required={required} "
                f"requested={self.requested_socket_buffer_bytes} "
                f"reported_receive={self.reported_receive_buffer_bytes}",
                "INFRASTRUCTURE",
            )
        if not isinstance(received, dict):
            _fail("ACTUAL_UDP_PAYLOAD_TYPE_MISMATCH", type(received).__name__)
        if int(compressed_bytes) != int(expected_compressed_bytes):
            _fail(
                "ACTUAL_UDP_COMPRESSED_SIZE_MISMATCH",
                f"expected={expected_compressed_bytes} actual={compressed_bytes}",
            )
        self.messages_sent += 1
        self.messages_received += 1
        self.total_compressed_bytes += int(compressed_bytes)
        self.max_compressed_bytes = max(self.max_compressed_bytes, int(compressed_bytes))
        self.total_chunks += int(chunks)
        return received, {
            "compressed_bytes": int(compressed_bytes),
            "chunks": int(chunks),
            "roundtrip_ms": float((time.perf_counter() - started) * 1000.0),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "status": "PASS" if self.messages_sent == self.messages_received else "FAIL",
            "actual_udp_executed": self.messages_sent > 0,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "total_compressed_bytes": self.total_compressed_bytes,
            "max_compressed_bytes": self.max_compressed_bytes,
            "total_chunks": self.total_chunks,
            "requested_socket_buffer_bytes": self.requested_socket_buffer_bytes,
            "reported_receive_buffer_bytes": self.reported_receive_buffer_bytes,
            "reported_send_buffer_bytes": self.reported_send_buffer_bytes,
            "host": str(self.config["model_smoke"]["socket_host"]),
        }

    def close(self) -> None:
        for endpoint in (self.sender, self.receiver):
            if endpoint is not None:
                try:
                    endpoint.close()
                except OSError:
                    pass
        self.sender = None
        self.receiver = None

    def __enter__(self) -> "ActualUDPLoopback":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


def _finite_tensor_shapes(outputs: Any, torch_module: Any) -> dict[str, list[int]]:
    if not isinstance(outputs, Mapping):
        _fail("TAIL_OUTPUT_INVALID", type(outputs).__name__)
    required = {"out", "object"}
    if not required <= set(outputs):
        _fail("TAIL_OUTPUT_HEAD_MISSING", repr(sorted(required - set(outputs))))
    shapes: dict[str, list[int]] = {}
    for name, tensor in outputs.items():
        if not isinstance(tensor, torch_module.Tensor):
            _fail("TAIL_OUTPUT_TENSOR_INVALID", f"head={name} type={type(tensor).__name__}")
        if not bool(torch_module.isfinite(tensor).all().item()):
            _fail("TAIL_OUTPUT_NONFINITE", str(name))
        shapes[str(name)] = [int(value) for value in tensor.shape]
    return shapes


def _validate_finite_tree(value: Any, *, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            _fail("TAIL_RESULT_NONFINITE", path)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_tree(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_tree(item, path=f"{path}[{index}]")
        return
    _fail("TAIL_RESULT_VALUE_TYPE_INVALID", f"path={path} type={type(value).__name__}")


def _audit_model_ae_binding(
    front_model: Any,
    edge_model: Any,
    profile: RegisteredSplitProfile,
) -> dict[str, Any]:
    expected = int(profile.row["ae_bottleneck_channels"])
    result: dict[str, Any] = {"expected_bottleneck_channels": expected}
    for role, model in (("front", front_model), ("edge", edge_model)):
        ae = getattr(model, "_ae", None)
        if expected == 0:
            if ae is not None:
                _fail(
                    "UNEXPECTED_INTEGRATED_AE",
                    f"role={role} family={profile.row['model_family']}",
                )
            result[role] = {"present": False}
            continue
        if ae is None:
            _fail("INTEGRATED_AE_MISSING", f"role={role} family={profile.row['model_family']}")
        actual = int(getattr(ae, "bottleneck", -1))
        if actual != expected:
            _fail(
                "INTEGRATED_AE_BOTTLENECK_MISMATCH",
                f"role={role} expected={expected} actual={actual}",
            )
        result[role] = {
            "present": True,
            "class": type(ae).__name__,
            "bottleneck_channels": actual,
        }
    return result


def _audit_front_compression(
    *,
    model: Any,
    native_features: Mapping[str, Any],
    wire_features: Mapping[str, Any],
    out_hw: tuple[int, int],
    q: float,
    profile: RegisteredSplitProfile,
    torch_module: Any,
) -> dict[str, Any]:
    """Independently verify rank-drop cardinality and integrated-AE output."""

    import torch.nn.functional as functional

    expected_gated: OrderedDict[str, Any] = OrderedDict()
    dropped: dict[str, int] = {}
    if q > 0.0:
        with torch_module.inference_mode():
            object_maps = model.decode_object_maps(native_features, out_hw)
            objectness = torch_module.sigmoid(
                object_maps[:, : int(model._n_heat)]
            ).amax(dim=1, keepdim=True)
        for name, feature in native_features.items():
            pooled = functional.adaptive_max_pool2d(
                objectness, feature.shape[-2:]
            ).reshape(-1).float()
            count = int(round(float(q) * int(pooled.numel())))
            keep = torch_module.ones_like(pooled)
            if count > 0:
                keep[pooled.argsort()[:count]] = 0.0
            expected_gated[str(name)] = feature * keep.reshape(
                1, 1, feature.shape[-2], feature.shape[-1]
            ).to(feature.dtype)
            dropped[str(name)] = count
    else:
        expected_gated.update((str(name), tensor) for name, tensor in native_features.items())
        dropped.update((str(name), 0) for name in native_features)

    ae = getattr(model, "_ae", None)
    with torch_module.inference_mode():
        expected_wire = OrderedDict(
            (
                name,
                ae.encode(tensor) if name == "high" and ae is not None else tensor,
            )
            for name, tensor in expected_gated.items()
        )
    if set(expected_wire) != set(wire_features):
        _fail("FRONT_COMPRESSION_LEVEL_MISMATCH", profile.profile_id)
    for name, expected_tensor in expected_wire.items():
        observed = wire_features[name]
        if tuple(observed.shape) != tuple(expected_tensor.shape):
            _fail(
                "FRONT_COMPRESSION_SHAPE_MISMATCH",
                f"profile={profile.profile_id} level={name}",
            )
        if not bool(
            torch_module.allclose(
                observed,
                expected_tensor,
                rtol=1e-5,
                atol=1e-6,
                equal_nan=False,
            )
        ):
            _fail(
                "FRONT_COMPRESSION_EFFECT_MISMATCH",
                f"profile={profile.profile_id} level={name} q={q}",
            )
    expected_high_channels = int(profile.expected_wire_shapes["high"][1])
    actual_high_channels = int(wire_features["high"].shape[1])
    if actual_high_channels != expected_high_channels:
        _fail(
            "INTEGRATED_AE_WIRE_CHANNEL_MISMATCH",
            f"expected={expected_high_channels} actual={actual_high_channels}",
        )
    return {
        "q": float(q),
        "dropped_spatial_cells": dropped,
        "effect_matches_independent_reconstruction": True,
        "wire_high_channels": actual_high_channels,
    }


def _run_registered_tail(
    *,
    runtime: Any,
    edge_model: Any,
    profile: RegisteredSplitProfile,
    transport: Any,
    payload: dict[str, Any],
    device: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Invoke the v2 tail implementation while auditing its raw heads."""

    try:
        import numpy as np
        import torch
    except Exception as exc:
        _fail("TAIL_DEPENDENCY_IMPORT_FAILED", f"{type(exc).__name__}: {exc}")
    worker = object.__new__(runtime.FusionRemoteInferenceWorker)
    worker.model = edge_model
    worker.device = device
    worker.transport = transport
    worker.feature_codecs = OrderedDict()
    worker.registered_profile = profile
    worker.score_threshold = float(profile.row["object_score_threshold"])
    worker.nms_radius_px = int(profile.row["object_nms_radius_px"])
    worker.topk = int(profile.row["topk_objects"])
    worker.max_objects_drawn = int(profile.row["max_objects_published"])
    worker.draw_projected_obb_box = False

    raw_shapes: dict[str, list[int]] = {}
    original_decode = edge_model.decode_outputs

    def audited_decode(features: Any, output_size: tuple[int, int]) -> Any:
        outputs = original_decode(features, output_size)
        raw_shapes.update(_finite_tensor_shapes(outputs, torch))
        return outputs

    edge_model.decode_outputs = audited_decode
    try:
        result = worker._run_back_half(payload)
    finally:
        edge_model.decode_outputs = original_decode
    if not raw_shapes:
        _fail("TAIL_OUTPUT_AUDIT_NOT_EXECUTED", profile.profile_id)
    expected_raw_shapes = {
        "out": [
            1,
            3,
            int(profile.row["input_height"]),
            int(profile.row["input_width"]),
        ],
        "object": [
            1,
            int(profile.row["object_channels"]),
            int(profile.row["input_height"]),
            int(profile.row["input_width"]),
        ],
    }
    for head, expected_shape in expected_raw_shapes.items():
        if raw_shapes.get(head) != expected_shape:
            _fail(
                "TAIL_OUTPUT_SHAPE_MISMATCH",
                f"head={head} expected={expected_shape} actual={raw_shapes.get(head)}",
            )
    validate_wire_identity(result.get("profile_identity"), profile.wire_identity)
    mask = result.get("mask")
    expected_mask_shape = (int(payload["display_size"][1]), int(payload["display_size"][0]))
    if not isinstance(mask, np.ndarray) or tuple(mask.shape) != expected_mask_shape:
        _fail(
            "TAIL_MASK_SHAPE_MISMATCH",
            f"expected={expected_mask_shape} actual={getattr(mask, 'shape', None)}",
        )
    if not bool(np.isfinite(mask).all()):
        _fail("TAIL_MASK_NONFINITE", profile.profile_id)
    objects = result.get("objects")
    if not isinstance(objects, list):
        _fail("TAIL_OBJECTS_INVALID", type(objects).__name__)
    _validate_finite_tree(objects, path="objects")
    return result, {
        "raw_output_shapes": raw_shapes,
        "mask_shape": [int(value) for value in mask.shape],
        "mask_dtype": str(mask.dtype),
        "object_count": len(objects),
    }


def _capture_model_map_packet(
    *,
    runtime: Any,
    profile: RegisteredSplitProfile,
    config: Mapping[str, Any],
    source_payload: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    captured: list[dict[str, Any]] = []
    display_w, display_h = (int(value) for value in source_payload["display_size"])
    publisher = object.__new__(runtime.SpatialMapResultPublisher)
    publisher.stream_id = str(source_payload.get("stream_id") or "ue_a2_fixture")
    publisher.traffic_light_id = ""
    publisher.traffic_light_actor_id = -1
    publisher.traffic_light_opendrive_id = ""
    publisher.camera_width = display_w
    publisher.camera_height = display_h
    publisher.camera_fov = 120.0
    publisher._enqueue = lambda payload, _frame_id: captured.append(payload)
    timing = {
        "front_compute_ms": 0.0,
        "t_capture_perf": float(source_payload["timing"]["t_capture_perf"]),
        "t_tail_done_perf": float(result["tail_done_perf"]),
    }
    publisher.publish_from_payload(
        source_payload=source_payload,
        result=result,
        timing=timing,
    )
    if len(captured) != 1:
        _fail("MODEL_MAP_CAPTURE_COUNT_MISMATCH", str(len(captured)))
    return _validate_map_packet(
        captured[0],
        profile,
        config,
        require_nonempty_object=False,
    )


def _exception_record(exc: BaseException, fallback: str) -> tuple[str, str, str]:
    code = str(getattr(exc, "code", f"{fallback}_{type(exc).__name__}"))
    detail = str(getattr(exc, "detail", str(exc)))
    classification = str(getattr(exc, "classification", "TECHNICAL"))
    lowered = f"{type(exc).__name__}: {exc}".lower()
    if "cuda" in lowered and "out of memory" in lowered:
        code = "CUDA_OUT_OF_MEMORY"
        classification = "INFRASTRUCTURE"
    return code, detail, classification


def _mark_rows(
    rows_by_profile: Mapping[str, MutableMapping[str, Any]],
    profiles: Sequence[RegisteredSplitProfile],
    *,
    status: str,
    code: str,
    stage_field: str,
) -> None:
    for profile in profiles:
        row = rows_by_profile[profile.profile_id]
        row[stage_field] = f"FAIL:{code}"
        row["technical_validity_status"] = status
        row["blocking_codes"] = code


def _model_source_seals(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    paths = {
        "runtime": Path(str(config["sources"]["runtime_path"])),
        "codec": Path(str(config["sources"]["codec_path"])),
        "model": ROOT
        / "pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/model.py",
        "split_runtime": ROOT
        / "pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/split_runtime.py",
        "feature_ae": ROOT / "rl_agent/feature_ae/ae_model.py",
        "wire_contract": Path(str(config["sources"]["wire_contract_path"])),
    }
    seals: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        resolved = path.resolve()
        if not resolved.is_file():
            _fail("MODEL_SOURCE_MISSING", f"label={label} path={resolved}")
        seals[label] = {
            "path": str(resolved),
            "bytes": int(resolved.stat().st_size),
            "sha256": sha256_file(resolved),
        }
    return seals


def _model_row_passes(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("technical_validity_status") == "TECHNICALLY_VALID"
        and row.get("registry_binding_status") == "PASS"
        and row.get("fixture_status") == "PASS"
        and row.get("production_codec_status") == "PASS_MODEL_FEATURES"
        and row.get("in_memory_wire_status") == "PASS"
        and row.get("wire_shape_status") == "PASS"
        and row.get("map_schema_status") == "PASS"
        and row.get("model_front_status") == "PASS_STRICT"
        and row.get("model_edge_status") == "PASS_STRICT"
        and str(row.get("roi_execution_status", "")).startswith("PASS_")
        and str(row.get("ae_execution_status", "")).startswith("PASS_")
        and row.get("tail_execution_status") == "PASS_FINITE"
        and row.get("actual_udp_status") == "PASS"
        and not str(row.get("blocking_codes", ""))
    )


def run_cuda_model_matrix(
    profiles: Sequence[RegisteredSplitProfile],
    config: Mapping[str, Any],
    base_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Run the factored deployment-representative 72-action CUDA smoke."""

    try:
        import numpy as np
        import torch
    except Exception as exc:
        _fail("MODEL_DEPENDENCY_IMPORT_FAILED", f"{type(exc).__name__}: {exc}", "INFRASTRUCTURE")
    device_name = str(config["model_smoke"]["device"])
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        _fail("CUDA_UNAVAILABLE", f"requested={device_name}", "INFRASTRUCTURE")
    device = torch.device(device_name)
    free_vram, total_vram = torch.cuda.mem_get_info(device)
    minimum_free = int(config["model_smoke"]["min_free_vram_bytes"])
    if int(free_vram) < minimum_free:
        _fail(
            "CUDA_FREE_VRAM_BELOW_MINIMUM",
            f"required={minimum_free} free={int(free_vram)} total={int(total_vram)}",
            "INFRASTRUCTURE",
        )
    seed = int(config["model_smoke"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    sources = config["sources"]
    codec = _import_pinned_source(
        Path(str(sources["codec_path"])),
        expected_sha256=str(sources["codec_sha256"]),
        module_prefix="ue_a2_production_codec",
    )
    runtime = _import_runtime(Path(str(sources["runtime_path"])))
    runtime_codec = _assert_runtime_codec_binding(runtime, config)
    fixture = _load_fixture_arrays(config)
    source_seals_start = _model_source_seals(config)

    rows = [dict(row) for row in base_rows]
    rows_by_profile = {str(row["profile_id"]): row for row in rows}
    for row in rows:
        row.update(
            {
                "production_codec_status": "PENDING_MODEL_FEATURES",
                "in_memory_wire_status": "PENDING_MODEL_FEATURES",
                "wire_shape_status": "PENDING_MODEL_FEATURES",
                "map_schema_status": "PENDING_MODEL_RESULT",
                "model_front_status": "PENDING_STRICT_LOAD",
                "model_edge_status": "PENDING_STRICT_LOAD",
                "roi_execution_status": "PENDING",
                "ae_execution_status": "PENDING",
                "tail_execution_status": "PENDING",
                "actual_udp_status": "PENDING",
                "technical_validity_status": "REGISTERED_PENDING_SMOKE",
                "blocking_codes": "MODEL_SMOKE_IN_PROGRESS",
            }
        )

    by_family: dict[str, list[RegisteredSplitProfile]] = {}
    for profile in profiles:
        by_family.setdefault(profile.row["model_family"], []).append(profile)
    family_order = [str(value) for value in config["registry"]["families"]]
    family_records: dict[str, Any] = {}
    strict_front_loads = 0
    strict_edge_loads = 0
    native_encodes = 0
    roi_ae_paths = 0
    tail_decodes = 0
    started = time.perf_counter()
    udp_summary: dict[str, Any] = {"status": "NOT_STARTED", "actual_udp_executed": False}

    with ActualUDPLoopback(codec, config) as loopback:
        for family in family_order:
            family_profiles = sorted(
                by_family.get(family, []), key=lambda item: int(item.row["action_index"])
            )
            if len(family_profiles) != 18:
                _fail(
                    "MODEL_FAMILY_PROFILE_COUNT_MISMATCH",
                    f"family={family} count={len(family_profiles)}",
                )
            representative = family_profiles[0]
            family_record: dict[str, Any] = {
                "profiles": len(family_profiles),
                "checkpoint_sha256": representative.row["checkpoint_sha256"],
                "errors": [],
            }
            family_records[family] = family_record
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            family_free_start, family_total = torch.cuda.mem_get_info(device)
            family_record["cuda_memory_start"] = {
                "free_bytes": int(family_free_start),
                "total_bytes": int(family_total),
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            }
            front_model = None
            edge_model = None
            fused = None
            native_features = None
            try:
                front_args = _model_runtime_args(representative, role="front")
                front_model, front_size = runtime.load_fusion_model(front_args, device)
                if not isinstance(getattr(front_args, "_ue_profile_binding_audit", None), dict):
                    _fail("STRICT_FRONT_BINDING_AUDIT_MISSING", family)
                strict_front_loads += 1
                for profile in family_profiles:
                    rows_by_profile[profile.profile_id]["model_front_status"] = "PASS_STRICT"
            except Exception as exc:
                code, detail, classification = _exception_record(exc, "STRICT_FRONT_LOAD")
                family_record["errors"].append(
                    {"stage": "strict_front_load", "code": code, "detail": detail}
                )
                if classification == "INFRASTRUCTURE":
                    raise A2TechnicalSmokeError(code, detail, classification) from exc
                _mark_rows(
                    rows_by_profile,
                    family_profiles,
                    status="TECHNICALLY_INVALID",
                    code=code,
                    stage_field="model_front_status",
                )
                front_model = None
                gc.collect()
                torch.cuda.empty_cache()
                continue

            try:
                edge_args = _model_runtime_args(representative, role="back")
                edge_model, edge_size = runtime.load_fusion_model(edge_args, device)
                if not isinstance(getattr(edge_args, "_ue_profile_binding_audit", None), dict):
                    _fail("STRICT_EDGE_BINDING_AUDIT_MISSING", family)
                if front_model is edge_model or front_model.model is edge_model.model:
                    _fail("FRONT_EDGE_MODEL_NOT_SEPARATE", family)
                strict_edge_loads += 1
                for profile in family_profiles:
                    rows_by_profile[profile.profile_id]["model_edge_status"] = "PASS_STRICT"
            except Exception as exc:
                code, detail, classification = _exception_record(exc, "STRICT_EDGE_LOAD")
                family_record["errors"].append(
                    {"stage": "strict_edge_load", "code": code, "detail": detail}
                )
                if classification == "INFRASTRUCTURE":
                    raise A2TechnicalSmokeError(code, detail, classification) from exc
                _mark_rows(
                    rows_by_profile,
                    family_profiles,
                    status="TECHNICALLY_INVALID",
                    code=code,
                    stage_field="model_edge_status",
                )
                front_model = None
                edge_model = None
                gc.collect()
                torch.cuda.empty_cache()
                continue

            try:
                expected_size = (
                    int(representative.row["input_width"]),
                    int(representative.row["input_height"]),
                )
                if tuple(front_size) != expected_size or tuple(edge_size) != expected_size:
                    _fail(
                        "MODEL_INPUT_SIZE_LOAD_MISMATCH",
                        f"family={family} expected={expected_size} "
                        f"front={front_size} edge={edge_size}",
                    )
                family_record["integrated_ae_binding"] = _audit_model_ae_binding(
                    front_model,
                    edge_model,
                    representative,
                )
            except Exception as exc:
                code, detail, classification = _exception_record(exc, "MODEL_BINDING_AUDIT")
                family_record["errors"].append(
                    {"stage": "model_binding_audit", "code": code, "detail": detail}
                )
                if classification == "INFRASTRUCTURE":
                    raise A2TechnicalSmokeError(code, detail, classification) from exc
                _mark_rows(
                    rows_by_profile,
                    family_profiles,
                    status="TECHNICALLY_INVALID",
                    code=code,
                    stage_field="ae_execution_status",
                )
                front_model = None
                edge_model = None
                gc.collect()
                torch.cuda.empty_cache()
                continue

            try:
                rgb_mean = torch.tensor(
                    [0.485, 0.456, 0.406], device=device
                ).view(1, 3, 1, 1)
                rgb_std = torch.tensor(
                    [0.229, 0.224, 0.225], device=device
                ).view(1, 3, 1, 1)
                with torch.inference_mode():
                    fused = runtime.prepare_fusion_input(
                        frame_bgr=fixture["frame_bgr"],
                        radar_tensor_chw=fixture["radar_tensor"],
                        model_size=expected_size,
                        device=device,
                        rgb_mean=rgb_mean,
                        rgb_std=rgb_std,
                    )
                    native_features = front_model.encode(fused)
                validate_declared_feature_shapes(
                    representative,
                    {
                        name: tuple(int(value) for value in tensor.shape)
                        for name, tensor in native_features.items()
                    },
                    stage="native",
                )
                for name, tensor in native_features.items():
                    if not bool(torch.isfinite(tensor).all().item()):
                        _fail("NATIVE_FEATURE_NONFINITE", f"family={family} level={name}")
                native_encodes += 1
            except Exception as exc:
                code, detail, classification = _exception_record(exc, "NATIVE_ENCODE")
                family_record["errors"].append(
                    {"stage": "native_encode", "code": code, "detail": detail}
                )
                if classification == "INFRASTRUCTURE":
                    raise A2TechnicalSmokeError(code, detail, classification) from exc
                _mark_rows(
                    rows_by_profile,
                    family_profiles,
                    status="TECHNICALLY_INVALID",
                    code=code,
                    stage_field="model_front_status",
                )
                native_features = None
                fused = None
                front_model = None
                edge_model = None
                gc.collect()
                torch.cuda.empty_cache()
                continue

            try:
                for q_text in config["registry"]["roi_drop_fractions"]:
                    q_profiles = [
                        profile
                        for profile in family_profiles
                        if profile.row["roi_drop_fraction"] == str(q_text)
                    ]
                    if len(q_profiles) != 3:
                        _fail(
                            "MODEL_Q_PROFILE_COUNT_MISMATCH",
                            f"family={family} q={q_text} count={len(q_profiles)}",
                        )
                    front_model._roi_threshold = float(q_text)
                    with torch.inference_mode():
                        wire_features = runtime._front_compress(
                            front_model,
                            native_features,
                            tuple(int(value) for value in fused.shape[-2:]),
                        )
                    validate_declared_feature_shapes(
                        q_profiles[0],
                        {
                            name: tuple(int(value) for value in tensor.shape)
                            for name, tensor in wire_features.items()
                        },
                        stage="wire",
                    )
                    for name, tensor in wire_features.items():
                        if not bool(torch.isfinite(tensor).all().item()):
                            _fail(
                                "WIRE_FEATURE_NONFINITE",
                                f"family={family} q={q_text} level={name}",
                            )
                    front_audit = _audit_front_compression(
                        model=front_model,
                        native_features=native_features,
                        wire_features=wire_features,
                        out_hw=tuple(int(value) for value in fused.shape[-2:]),
                        q=float(q_text),
                        profile=q_profiles[0],
                        torch_module=torch,
                    )
                    family_record.setdefault("front_paths", {})[str(q_text)] = front_audit
                    roi_ae_paths += 1

                    for profile in sorted(
                        q_profiles, key=lambda item: int(item.row["action_index"])
                    ):
                        row = rows_by_profile[profile.profile_id]
                        try:
                            profile_transport = codec.TransportConfig(
                                quantization_mode=str(profile.row["quantization_mode"]),
                                entropy_coder_name=str(profile.row["entropy_coder"]),
                                zstd_level=int(profile.row["entropy_level"]),
                                roi_objectness_threshold=0.0,
                                bypass_rcnn_transform=False,
                            )
                            serialized, payload_uncompressed, _, _ = codec.serialize_feature_maps(
                                wire_features,
                                OrderedDict(),
                                quantization_mode=profile_transport.quantization_mode,
                                per_level_compress_probe=False,
                                entropy_coder=profile_transport.make_entropy_coder(),
                            )
                            validate_serialized_feature_headers(profile, serialized)
                            frame_id = int(fixture["frame_id"]) + int(profile.row["action_index"])
                            capture_perf = time.perf_counter()
                            payload = {
                                "frame_id": frame_id,
                                "batch_size": 1,
                                "model_input_size": list(expected_size),
                                "display_size": [
                                    int(fixture["frame_bgr"].shape[1]),
                                    int(fixture["frame_bgr"].shape[0]),
                                ],
                                "profile_identity": dict(profile.wire_identity),
                                "feature_shapes": {
                                    name: [int(value) for value in tensor.shape]
                                    for name, tensor in wire_features.items()
                                },
                                "features": serialized,
                                "payload_bytes_uncompressed": int(payload_uncompressed),
                                "camera_matrix": fixture["camera_matrix"],
                                "camera_intrinsics_input": fixture["camera_intrinsics_input"],
                                "camera_transform": {},
                                "stream_id": "ue_a2_fixture",
                                "carla_timestamp": float(fixture["carla_timestamp"]),
                                "camera_sent_perf": capture_perf,
                                "timing": {
                                    "t_capture_perf": capture_perf,
                                    "t_front_send_perf": capture_perf,
                                },
                            }
                            in_memory_payload, wire = outer_wire_roundtrip(
                                payload,
                                coder=profile_transport.make_entropy_coder(),
                                chunk_bytes=int(profile.row["udp_chunk_bytes"]),
                            )
                            validate_feature_payload(profile, in_memory_payload)
                            received_payload, udp_record = loopback.roundtrip(
                                payload,
                                expected_compressed_bytes=int(wire["zstd_bytes"]),
                            )
                            validate_feature_payload(profile, received_payload)
                            result, tail_audit = _run_registered_tail(
                                runtime=runtime,
                                edge_model=edge_model,
                                profile=profile,
                                transport=profile_transport,
                                payload=received_payload,
                                device=device,
                            )
                            map_audit = _capture_model_map_packet(
                                runtime=runtime,
                                profile=profile,
                                config=config,
                                source_payload=received_payload,
                                result=result,
                            )
                            tail_decodes += 1
                            row.update(
                                {
                                    "production_codec_status": "PASS_MODEL_FEATURES",
                                    "in_memory_wire_status": "PASS",
                                    "wire_shape_status": "PASS",
                                    "map_schema_status": "PASS",
                                    "roi_execution_status": (
                                        "PASS_NO_DROP_VERIFIED"
                                        if float(q_text) == 0.0
                                        else "PASS_RANK_DROP_VERIFIED"
                                    ),
                                    "ae_execution_status": (
                                        "PASS_NO_AE"
                                        if int(profile.row["ae_bottleneck_channels"]) == 0
                                        else "PASS_INTEGRATED_AE"
                                    ),
                                    "tail_execution_status": "PASS_FINITE",
                                    "actual_udp_status": "PASS",
                                    "technical_validity_status": "TECHNICALLY_VALID",
                                    "blocking_codes": "",
                                    "payload_bytes_uncompressed": int(payload_uncompressed),
                                    "pickle_bytes": int(wire["pickle_bytes"]),
                                    "zstd_bytes": int(wire["zstd_bytes"]),
                                    "udp_chunks": int(udp_record["chunks"]),
                                    "max_quantization_abs_error": "",
                                    "compressed_sha256": str(wire["compressed_sha256"]),
                                }
                            )
                            family_record.setdefault("actions", {})[profile.profile_id] = {
                                "tail": tail_audit,
                                "map": map_audit,
                                "udp": udp_record,
                            }
                        except Exception as exc:
                            code, detail, classification = _exception_record(
                                exc, "MODEL_ACTION"
                            )
                            family_record["errors"].append(
                                {
                                    "stage": "action",
                                    "profile_id": profile.profile_id,
                                    "code": code,
                                    "detail": detail,
                                }
                            )
                            if classification == "INFRASTRUCTURE":
                                raise A2TechnicalSmokeError(code, detail, classification) from exc
                            row["technical_validity_status"] = "TECHNICALLY_INVALID"
                            row["blocking_codes"] = code
                            for field in (
                                "production_codec_status",
                                "in_memory_wire_status",
                                "wire_shape_status",
                                "map_schema_status",
                                "roi_execution_status",
                                "ae_execution_status",
                                "tail_execution_status",
                                "actual_udp_status",
                            ):
                                if str(row[field]).startswith("PENDING"):
                                    row[field] = f"FAIL:{code}"
            except Exception as exc:
                code, detail, classification = _exception_record(exc, "MODEL_Q_PATH")
                family_record["errors"].append(
                    {"stage": "q_path", "code": code, "detail": detail}
                )
                pending_profiles = [
                    profile
                    for profile in family_profiles
                    if rows_by_profile[profile.profile_id]["technical_validity_status"]
                    == "REGISTERED_PENDING_SMOKE"
                ]
                if classification == "INFRASTRUCTURE":
                    raise A2TechnicalSmokeError(code, detail, classification) from exc
                _mark_rows(
                    rows_by_profile,
                    pending_profiles,
                    status="TECHNICALLY_INVALID",
                    code=code,
                    stage_field="roi_execution_status",
                )
            finally:
                del native_features
                del fused
                del edge_model
                del front_model
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                family_free_end, _ = torch.cuda.mem_get_info(device)
                family_record["cuda_memory_end"] = {
                    "free_bytes": int(family_free_end),
                    "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
        udp_summary = loopback.summary()

    source_seals_end = _model_source_seals(config)
    source_drift = {
        label: {
            "start": source_seals_start[label],
            "end": source_seals_end[label],
        }
        for label in source_seals_start
        if source_seals_start[label] != source_seals_end[label]
    }
    if source_drift:
        for row in rows:
            row["technical_validity_status"] = "TECHNICALLY_INVALID"
            row["blocking_codes"] = "SOURCE_CHANGED_DURING_SMOKE"
    for row in rows:
        if row["technical_validity_status"] == "TECHNICALLY_VALID" and not _model_row_passes(row):
            row["technical_validity_status"] = "TECHNICALLY_INVALID"
            row["blocking_codes"] = "ROW_ACCEPTANCE_GATE_INCOMPLETE"
    valid = sum(_model_row_passes(row) for row in rows)
    invalid = sum(row["technical_validity_status"] == "TECHNICALLY_INVALID" for row in rows)
    summary = {
        "schema": "scenesense.ue_a2_cuda_model_smoke.v1",
        "status": "PASS" if valid == 72 and invalid == 0 else "FAIL",
        "device": str(device),
        "cuda_device_name": str(torch.cuda.get_device_name(device)),
        "cuda_total_vram_bytes": int(total_vram),
        "cuda_free_vram_bytes_at_start": int(free_vram),
        "minimum_free_vram_bytes": minimum_free,
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version() or 0),
        "seed": seed,
        "strict_front_loads": strict_front_loads,
        "strict_edge_loads": strict_edge_loads,
        "native_backbone_encodes": native_encodes,
        "roi_ae_paths": roi_ae_paths,
        "tail_decodes": tail_decodes,
        "technical_valid_profiles": valid,
        "technical_invalid_profiles": invalid,
        "model_inference_executed": native_encodes > 0,
        "actual_udp_executed": bool(udp_summary.get("actual_udp_executed")),
        "runtime_codec": runtime_codec,
        "source_seals_start": source_seals_start,
        "source_seals_end": source_seals_end,
        "source_drift": source_drift,
        "families": family_records,
        "elapsed_s": float(time.perf_counter() - started),
    }
    expected = config["model_smoke"]
    exact_counts = {
        "strict_front_loads": int(expected["strict_front_loads"]),
        "strict_edge_loads": int(expected["strict_edge_loads"]),
        "native_backbone_encodes": int(expected["native_backbone_encodes"]),
        "roi_ae_paths": int(expected["roi_ae_paths"]),
        "tail_decodes": int(expected["tail_decodes"]),
    }
    for field, wanted in exact_counts.items():
        if int(summary[field]) != wanted:
            summary["status"] = "FAIL"
            summary.setdefault("count_mismatches", {})[field] = {
                "expected": wanted,
                "actual": int(summary[field]),
            }
    if int(udp_summary.get("messages_received", 0)) != int(expected["actual_udp_roundtrips"]):
        summary["status"] = "FAIL"
        summary.setdefault("count_mismatches", {})["actual_udp_roundtrips"] = {
            "expected": int(expected["actual_udp_roundtrips"]),
            "actual": int(udp_summary.get("messages_received", 0)),
        }
    return rows, summary, udp_summary


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_profile_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PROFILE_FIELDS), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in PROFILE_FIELDS})


def _source_manifest(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    inputs = []
    for label, path_value in (
        ("config", config["config_path"]),
        ("registry", config["registry"]["path"]),
        ("fixture", config["fixture"]["path"]),
        ("wire_contract", config["sources"]["wire_contract_path"]),
        ("runtime", config["sources"]["runtime_path"]),
        ("launcher", config["sources"]["launcher_path"]),
        ("codec", config["sources"]["codec_path"]),
        ("runner", Path(__file__).resolve()),
    ):
        path = Path(str(path_value))
        inputs.append(
            {
                "label": label,
                "path": str(path),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else 0,
                "sha256": sha256_file(path) if path.is_file() else "",
            }
        )
    return inputs


def _report(
    *,
    terminal_status: str,
    rows: Sequence[Mapping[str, Any]],
    negatives: Mapping[str, Any],
    environment: Mapping[str, Any],
    transport: Mapping[str, Any],
    map_probe: Mapping[str, Any],
) -> str:
    blockers = list(environment.get("blockers", [])) + list(transport.get("blockers", []))
    if map_probe.get("status") != "PASS":
        blockers.append("RUNTIME_MAP_PROFILE_IDENTITY_NOT_READY")
    codec_passed = sum(
        row["production_codec_status"].startswith("PASS") for row in rows
    )
    wire_passed = sum(row["in_memory_wire_status"] == "PASS" for row in rows)
    return f"""# UE-A2 offline technical-smoke preflight

**Status:** {terminal_status}

This bundle is a preflight, not UE-A2 completion. It ran no CARLA, OAI, model
inference, or actual UDP traffic and did not mark any profile technically valid.

## Deterministic results

- registry actions resolved: {len(rows)} / 72
- production CPU quantizer/dequantizer paths: {codec_passed} / {len(rows)}
- in-memory pickle/zstd/chunk round trips: {wire_passed} / {len(rows)}
- negative contract mutations rejected: {negatives.get('tests', 0)} / {negatives.get('tests', 0)}
- v2 runtime map-packet probes: {map_probe.get('profiles_passed', 0)} / 72

## Remaining final UE-A2 gates

- strict front/back checkpoint execution on CUDA;
- real ROI and integrated-AE encode/decode for every action;
- finite tail segmentation/object decoding for every action;
- actual localhost UDP after socket-buffer preflight; and
- one final 72-row successor technical registry.

Readiness blockers: {', '.join(blockers) if blockers else 'none'}.
"""


def _model_report(
    *,
    terminal_status: str,
    rows: Sequence[Mapping[str, Any]],
    model_summary: Mapping[str, Any],
    transport: Mapping[str, Any],
    blockers: Sequence[str],
) -> str:
    valid = sum(row["technical_validity_status"] == "TECHNICALLY_VALID" for row in rows)
    invalid = sum(row["technical_validity_status"] == "TECHNICALLY_INVALID" for row in rows)
    blocked = sum(row["technical_validity_status"] == "BLOCKED_INFRASTRUCTURE" for row in rows)
    return f"""# UE-A2 local CUDA technical smoke

**Status:** {terminal_status}

This create-only bundle ran no CARLA or OAI and introduced no quality-derived
profile filter. A passing bundle proves only technical executability of the 72
registered split actions on the pinned fixture; it is not a perception-quality
approval and it does not create the later UE-A4 successor registry.

## Required counts

- technically valid profiles: {valid} / 72
- technically invalid profiles: {invalid} / 72
- infrastructure-blocked profiles: {blocked} / 72
- strict front loads: {model_summary.get('strict_front_loads', 0)} / 4
- strict edge loads: {model_summary.get('strict_edge_loads', 0)} / 4
- native backbone encodes: {model_summary.get('native_backbone_encodes', 0)} / 4
- q plus integrated-AE paths: {model_summary.get('roi_ae_paths', 0)} / 24
- finite tail decodes: {model_summary.get('tail_decodes', 0)} / 72
- actual localhost UDP receives: {transport.get('actual_udp', {}).get('messages_received', 0)} / 72

Infrastructure blockers: {', '.join(blockers) if blockers else 'none'}.
"""


def _write_model_bundle(
    *,
    config: Mapping[str, Any],
    output_dir: Path | None,
    fixture_manifest: Mapping[str, Any],
    registry_audit: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    negatives: Mapping[str, Any],
    environment: Mapping[str, Any],
    transport_preflight: Mapping[str, Any],
    actual_udp: Mapping[str, Any],
    model_summary: Mapping[str, Any],
    terminal_status: str,
    blockers: Sequence[str],
    started: float,
) -> dict[str, Any]:
    terminal_names = {
        "PASSED": "UE_A2_PASSED.json",
        "FAILED": "UE_A2_FAILED.json",
        "BLOCKED_INFRASTRUCTURE": "UE_A2_BLOCKED_INFRASTRUCTURE.json",
    }
    if terminal_status not in terminal_names:
        _fail("MODEL_TERMINAL_STATUS_INVALID", terminal_status)
    terminal_name = terminal_names[terminal_status]
    configured_root = Path(str(config["output"]["root"]))
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        final_dir = configured_root / stamp
    else:
        final_dir = Path(output_dir).expanduser().resolve()
    if final_dir.exists():
        _fail("OUTPUT_DIRECTORY_EXISTS", str(final_dir))
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{final_dir.name}.building-", dir=str(final_dir.parent))
    )
    output_cfg = config["output"]
    combined_transport = {
        "schema": "scenesense.ue_a2_transport_evidence.v1",
        "preflight": dict(transport_preflight),
        "actual_udp": dict(actual_udp),
    }
    _write_json(staging / str(output_cfg["resolved_config_json"]), config)
    _write_json(staging / str(output_cfg["fixture_json"]), fixture_manifest)
    _write_profile_csv(staging / str(output_cfg["profile_csv"]), rows)
    _write_json(staging / str(output_cfg["negative_tests_json"]), negatives)
    _write_json(staging / str(output_cfg["transport_json"]), combined_transport)
    _write_json(staging / str(output_cfg["model_smoke_json"]), model_summary)
    report = _model_report(
        terminal_status=terminal_status,
        rows=rows,
        model_summary=model_summary,
        transport=combined_transport,
        blockers=blockers,
    )
    (staging / str(output_cfg["report_md"])).write_text(report, encoding="utf-8")
    valid = sum(row["technical_validity_status"] == "TECHNICALLY_VALID" for row in rows)
    invalid = sum(row["technical_validity_status"] == "TECHNICALLY_INVALID" for row in rows)
    blocked = sum(row["technical_validity_status"] == "BLOCKED_INFRASTRUCTURE" for row in rows)
    terminal = {
        "schema": "scenesense.ue_a2_technical_smoke_terminal.v1",
        "status": terminal_status,
        "created_at": _utc_now(),
        "claim_scope": "LOCAL_CUDA_TECHNICAL_EXECUTABILITY_ONLY",
        "profile_rows": len(rows),
        "technical_valid_profiles": valid,
        "technical_invalid_profiles": invalid,
        "infrastructure_blocked_profiles": blocked,
        "model_inference_executed": bool(model_summary.get("model_inference_executed")),
        "actual_udp_executed": bool(actual_udp.get("actual_udp_executed")),
        "carla_run": False,
        "oai_run": False,
        "quality_gate_applied": False,
        "successor_registry_emitted": False,
        "blockers": list(blockers),
        "elapsed_s": float(time.perf_counter() - started),
    }
    _write_json(staging / terminal_name, terminal)
    manifest_path = staging / str(output_cfg["manifest_json"])
    outputs = []
    for path in sorted(staging.iterdir(), key=lambda item: item.name):
        if path == manifest_path or not path.is_file():
            continue
        entry = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.name == str(output_cfg["profile_csv"]):
            entry["rows"] = len(rows)
        outputs.append(entry)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "experiment_id": config["experiment_id"],
        "created_at": _utc_now(),
        "claim_scope": "LOCAL_CUDA_TECHNICAL_EXECUTABILITY_ONLY",
        "terminal_path": terminal_name,
        "terminal_status": terminal_status,
        "inputs": _source_manifest(config),
        "outputs": outputs,
        "registry_audit": registry_audit,
        "environment": environment,
        "model_summary": model_summary,
        "transport": combined_transport,
        "successor_registry_emitted": False,
    }
    _write_json(manifest_path, manifest)
    staging.rename(final_dir)
    return {
        "status": terminal_status,
        "output_dir": str(final_dir),
        "terminal": str(final_dir / terminal_name),
        "profiles": len(rows),
        "technical_valid_profiles": valid,
        "technical_invalid_profiles": invalid,
        "infrastructure_blocked_profiles": blocked,
        "blockers": list(blockers),
    }


def run_model_smoke(
    config_path: Path = DEFAULT_CONFIG,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run or truthfully block the complete no-CARLA/no-OAI UE-A2 smoke."""

    started = time.perf_counter()
    config = load_config(config_path)
    fixture_manifest = inspect_fixture(config)
    profiles, registry_audit = verify_registry_matrix(config)
    rows, codec_summary = run_production_codec_matrix(profiles, config)
    negatives = run_negative_contract_tests(profiles, config)
    map_statuses, map_probe = run_runtime_map_probes(profiles, config)
    for row in rows:
        map_status = map_statuses.get(str(row["profile_id"]), "FAIL")
        row["map_schema_status"] = (
            "PASS_SYNTHETIC_NONEMPTY" if map_status == "PASS" else "FAIL_SYNTHETIC"
        )
    environment = inspect_runtime_environment(config)
    transport_preflight = inspect_socket_buffers(
        config,
        observed_max_message_bytes=int(codec_summary["max_zstd_message_bytes"]),
    )
    infrastructure_blockers = list(environment.get("blockers", [])) + list(
        transport_preflight.get("blockers", [])
    )
    implementation_blockers = []
    if map_probe.get("status") != "PASS":
        implementation_blockers.append("RUNTIME_MAP_SCHEMA_PROBE_FAILED")

    actual_udp: dict[str, Any] = {
        "status": "NOT_EXECUTED",
        "actual_udp_executed": False,
        "messages_sent": 0,
        "messages_received": 0,
    }
    model_summary: dict[str, Any] = {
        "schema": "scenesense.ue_a2_cuda_model_smoke.v1",
        "status": "NOT_EXECUTED",
        "strict_front_loads": 0,
        "strict_edge_loads": 0,
        "native_backbone_encodes": 0,
        "roi_ae_paths": 0,
        "tail_decodes": 0,
        "model_inference_executed": False,
        "actual_udp_executed": False,
    }

    if infrastructure_blockers:
        code = "|".join(infrastructure_blockers)
        for row in rows:
            row.update(
                {
                    "model_front_status": "BLOCKED_INFRASTRUCTURE",
                    "model_edge_status": "BLOCKED_INFRASTRUCTURE",
                    "roi_execution_status": "BLOCKED_INFRASTRUCTURE",
                    "ae_execution_status": "BLOCKED_INFRASTRUCTURE",
                    "tail_execution_status": "BLOCKED_INFRASTRUCTURE",
                    "actual_udp_status": "BLOCKED_INFRASTRUCTURE",
                    "technical_validity_status": "BLOCKED_INFRASTRUCTURE",
                    "blocking_codes": code,
                }
            )
        model_summary["status"] = "BLOCKED_INFRASTRUCTURE"
        model_summary["blockers"] = infrastructure_blockers
        terminal_status = "BLOCKED_INFRASTRUCTURE"
        blockers = infrastructure_blockers
    elif implementation_blockers:
        code = "|".join(implementation_blockers)
        for row in rows:
            row.update(
                {
                    "model_front_status": "NOT_EXECUTED",
                    "model_edge_status": "NOT_EXECUTED",
                    "tail_execution_status": "NOT_EXECUTED",
                    "actual_udp_status": "NOT_EXECUTED",
                    "technical_validity_status": "TECHNICALLY_INVALID",
                    "blocking_codes": code,
                }
            )
        model_summary["status"] = "FAILED_PRE_MODEL_GATE"
        model_summary["blockers"] = implementation_blockers
        terminal_status = "FAILED"
        blockers = implementation_blockers
    else:
        try:
            rows, model_summary, actual_udp = run_cuda_model_matrix(
                profiles,
                config,
                rows,
            )
        except Exception as exc:
            code, detail, classification = _exception_record(exc, "MODEL_SMOKE_FATAL")
            model_summary.update(
                {
                    "status": (
                        "BLOCKED_INFRASTRUCTURE"
                        if classification == "INFRASTRUCTURE"
                        else "FAILED"
                    ),
                    "fatal_error": {"code": code, "detail": detail},
                }
            )
            row_status = (
                "BLOCKED_INFRASTRUCTURE"
                if classification == "INFRASTRUCTURE"
                else "TECHNICALLY_INVALID"
            )
            for row in rows:
                row.update(
                    {
                        "model_front_status": f"NOT_COMPLETED:{code}",
                        "model_edge_status": f"NOT_COMPLETED:{code}",
                        "tail_execution_status": f"NOT_COMPLETED:{code}",
                        "actual_udp_status": f"NOT_COMPLETED:{code}",
                        "technical_validity_status": row_status,
                        "blocking_codes": code,
                    }
                )
            terminal_status = (
                "BLOCKED_INFRASTRUCTURE"
                if classification == "INFRASTRUCTURE"
                else "FAILED"
            )
            blockers = [code]
        else:
            valid = sum(
                _model_row_passes(row) for row in rows
            )
            passed = (
                model_summary.get("status") == "PASS"
                and valid == 72
                and all(_model_row_passes(row) for row in rows)
                and int(actual_udp.get("messages_received", 0)) == 72
            )
            terminal_status = "PASSED" if passed else "FAILED"
            blockers = [] if passed else ["MODEL_MATRIX_ACCEPTANCE_FAILED"]

    return _write_model_bundle(
        config=config,
        output_dir=output_dir,
        fixture_manifest=fixture_manifest,
        registry_audit=registry_audit,
        rows=rows,
        negatives=negatives,
        environment=environment,
        transport_preflight=transport_preflight,
        actual_udp=actual_udp,
        model_summary=model_summary,
        terminal_status=terminal_status,
        blockers=blockers,
        started=started,
    )


def run_preflight(
    config_path: Path = DEFAULT_CONFIG,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the bounded no-model/no-socket preflight and write a create-only bundle."""

    started = time.perf_counter()
    config = load_config(config_path)
    fixture_manifest = inspect_fixture(config)
    profiles, registry_audit = verify_registry_matrix(config)
    rows, codec_summary = run_production_codec_matrix(profiles, config)
    negatives = run_negative_contract_tests(profiles, config)
    map_statuses, map_probe = run_runtime_map_probes(profiles, config)
    for row in rows:
        status = map_statuses.get(str(row["profile_id"]), "FAIL")
        row["map_schema_status"] = status
        blocking = ["MODEL_AND_ACTUAL_UDP_PENDING"]
        if status != "PASS":
            blocking.append("RUNTIME_MAP_PROFILE_IDENTITY_NOT_READY")
        row["blocking_codes"] = "|".join(blocking)

    environment = inspect_runtime_environment(config)
    transport = inspect_socket_buffers(
        config,
        observed_max_message_bytes=int(codec_summary["max_zstd_message_bytes"]),
    )
    environment_blockers = list(environment.get("blockers", []))
    transport_blockers = list(transport.get("blockers", []))
    implementation_blockers = []
    if map_probe.get("status") != "PASS":
        implementation_blockers.append("RUNTIME_MAP_PROFILE_IDENTITY_NOT_READY")
    all_blockers = environment_blockers + transport_blockers + implementation_blockers
    terminal_status = "PREFLIGHT_BLOCKED" if all_blockers else "PREFLIGHT_READY"
    terminal_name = (
        "UE_A2_PREFLIGHT_BLOCKED.json"
        if all_blockers
        else "UE_A2_PREFLIGHT_READY.json"
    )

    configured_root = Path(str(config["output"]["root"]))
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        final_dir = configured_root / stamp
    else:
        final_dir = Path(output_dir).expanduser().resolve()
    if final_dir.exists():
        _fail("OUTPUT_DIRECTORY_EXISTS", str(final_dir))
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{final_dir.name}.building-", dir=str(final_dir.parent))
    )

    output_cfg = config["output"]
    _write_json(staging / str(output_cfg["resolved_config_json"]), config)
    _write_json(staging / str(output_cfg["fixture_json"]), fixture_manifest)
    _write_profile_csv(staging / str(output_cfg["profile_csv"]), rows)
    _write_json(staging / str(output_cfg["negative_tests_json"]), negatives)
    _write_json(staging / str(output_cfg["transport_json"]), transport)
    report = _report(
        terminal_status=terminal_status,
        rows=rows,
        negatives=negatives,
        environment=environment,
        transport=transport,
        map_probe=map_probe,
    )
    (staging / str(output_cfg["report_md"])).write_text(report, encoding="utf-8")
    terminal = {
        "schema": PREFLIGHT_SCHEMA,
        "status": terminal_status,
        "created_at": _utc_now(),
        "claim_scope": "PREFLIGHT_ONLY_NO_TECHNICAL_VALIDITY",
        "profile_rows": len(rows),
        "technical_valid_profiles": 0,
        "technical_pending_profiles": len(rows),
        "model_inference_executed": False,
        "actual_udp_executed": False,
        "carla_run": False,
        "oai_run": False,
        "readiness_blockers": all_blockers,
        "environment": environment,
        "map_probe": map_probe,
        "elapsed_s": time.perf_counter() - started,
    }
    _write_json(staging / terminal_name, terminal)

    manifest_path = staging / str(output_cfg["manifest_json"])
    outputs = []
    for path in sorted(staging.iterdir(), key=lambda item: item.name):
        if path == manifest_path or not path.is_file():
            continue
        entry = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.name == str(output_cfg["profile_csv"]):
            entry["rows"] = len(rows)
        outputs.append(entry)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "experiment_id": config["experiment_id"],
        "created_at": _utc_now(),
        "claim_scope": "PREFLIGHT_ONLY_NO_TECHNICAL_VALIDITY",
        "terminal_path": terminal_name,
        "terminal_status": terminal_status,
        "inputs": _source_manifest(config),
        "outputs": outputs,
        "registry_audit": registry_audit,
        "codec_summary": codec_summary,
        "map_probe_summary": map_probe,
        "environment": environment,
    }
    _write_json(manifest_path, manifest)
    staging.rename(final_dir)
    return {
        "status": terminal_status,
        "output_dir": str(final_dir),
        "terminal": str(final_dir / terminal_name),
        "profiles": len(rows),
        "technical_valid_profiles": 0,
        "readiness_blockers": all_blockers,
    }


def _parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the no-CARLA/no-OAI UE-A2 deterministic technical-smoke preflight."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    preflight.add_argument("--output-dir", type=Path)
    model_smoke = subparsers.add_parser(
        "model-smoke",
        help="Run the complete local CUDA plus localhost-UDP UE-A2 smoke.",
    )
    model_smoke.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    model_smoke.add_argument("--output-dir", type=Path)
    full = subparsers.add_parser("full")
    full.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help=argparse.SUPPRESS
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli(argv)
    if args.command == "full":
        print(
            json.dumps(
                {
                    "error": "DEPRECATED_FULL_COMMAND",
                    "detail": (
                        "Use the explicit model-smoke command for the strict CUDA, "
                        "AE/tail, map, and localhost-UDP gates."
                    ),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        if args.command == "model-smoke":
            result = run_model_smoke(args.config, output_dir=args.output_dir)
        else:
            result = run_preflight(args.config, output_dir=args.output_dir)
    except (A2TechnicalSmokeError, SplitWireContractError) as exc:
        print(
            json.dumps(
                {
                    "error": getattr(exc, "code", type(exc).__name__),
                    "detail": getattr(exc, "detail", str(exc)),
                    "classification": getattr(exc, "classification", "CONTRACT"),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    if result["status"] in {"PREFLIGHT_READY", "PASSED"}:
        return 0
    if result["status"] in {"PREFLIGHT_BLOCKED", "BLOCKED_INFRASTRUCTURE"}:
        return 3
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
