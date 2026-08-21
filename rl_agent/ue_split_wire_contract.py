"""Pure UE-A2 split-profile registry and wire-contract validation.

This module deliberately has no CARLA, OAI, NumPy, PyTorch, socket, or model
dependencies.  It binds one fixed UE split action to the immutable UE-A1
registry, constructs the identity carried by every feature payload, and
validates that identity and the self-describing per-channel feature headers
before an edge decoder sees them.

The contract is a consistency/provenance guard for the trusted testbed.  It is
not authentication and does not make the existing pickle transport safe for
untrusted peers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_CSV = (
    ROOT
    / "rl_agent/registries/ue_split_profile_registry_v1/ue_split_profile_registry.csv"
)
A1_REGISTRY_CSV_SHA256 = (
    "9542adc8e014960bf8876e87cdbd9783f8911140fbc37820605ce7dd69e23722"
)
A1_REGISTRY_SCHEMA = "scenesense.ue_split_profile_registry.v1"
A1_REGISTRY_ID = "ue_split_profile_registry_v1"
EXPECTED_A1_PROFILE_COUNT = 72

ACTION_CONTRACT_SCHEMA = "scenesense.ue_split_action_contract.v1"
WIRE_IDENTITY_SCHEMA = "scenesense.ue_split_wire_identity.v1"
LAUNCH_BINDING_SCHEMA = "scenesense.ue_split_profile_launch_binding.v1"
PER_CHANNEL_HEADER = struct.Struct("!IIIB")


class SplitWireContractError(ValueError):
    """A fail-closed registry, runtime-binding, or feature-wire violation."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


@dataclass(frozen=True)
class RegisteredSplitProfile:
    """One verified, fixed action resolved from the UE-A1 registry."""

    registry_path: Path
    registry_sha256: str
    row: Mapping[str, str]
    action_contract_sha256: str

    @property
    def profile_id(self) -> str:
        return self.row["profile_id"]

    @property
    def action_contract(self) -> dict[str, Any]:
        return build_action_contract(self.row)

    @property
    def wire_identity(self) -> dict[str, Any]:
        return build_wire_identity(self.row, self.action_contract_sha256)

    @property
    def expected_wire_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "low": _parse_shape(self.row["expected_low_shape"], "expected_low_shape"),
            "high": _parse_shape(
                self.row["expected_high_wire_shape"], "expected_high_wire_shape"
            ),
        }

    @property
    def expected_native_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "low": _parse_shape(self.row["expected_low_shape"], "expected_low_shape"),
            "high": _parse_shape(
                self.row["expected_high_native_shape"], "expected_high_native_shape"
            ),
        }

    @property
    def expected_edge_decoded_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "low": _parse_shape(self.row["expected_low_shape"], "expected_low_shape"),
            "high": _parse_shape(
                self.row["expected_high_after_edge_decode_shape"],
                "expected_high_after_edge_decode_shape",
            ),
        }


def _fail(code: str, detail: str) -> None:
    raise SplitWireContractError(code, detail)


def sha256_file(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        _fail("FILE_READ_FAILED", f"{path}: {exc}")
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_decimal(value: Any, label: str) -> str:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        _fail("INVALID_DECIMAL", f"{label}={value!r}: {exc}")
    if not parsed.is_finite():
        _fail("INVALID_DECIMAL", f"{label} must be finite, got {value!r}")
    if parsed == 0:
        return "0"
    rendered = format(parsed.normalize(), "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        _fail("INVALID_INTEGER", f"{label} must not be boolean")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        _fail("INVALID_INTEGER", f"{label}={value!r}: {exc}")
    if str(value).strip() not in {str(parsed), f"+{parsed}"}:
        _fail("INVALID_INTEGER", f"{label} is not a canonical integer: {value!r}")
    return parsed


def _strict_bool(value: Any, label: str) -> bool:
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    _fail("INVALID_BOOLEAN", f"{label} must be True or False, got {value!r}")


def _split_pipe(value: str, label: str) -> list[str]:
    parts = str(value).split("|")
    if not parts or any(not part for part in parts):
        _fail("INVALID_LIST", f"{label} is malformed: {value!r}")
    return parts


def _parse_shape(value: Any, label: str) -> tuple[int, ...]:
    try:
        parts = tuple(int(item) for item in str(value).split("x"))
    except ValueError as exc:
        _fail("INVALID_FEATURE_SHAPE", f"{label}={value!r}: {exc}")
    if len(parts) != 4 or any(item <= 0 for item in parts):
        _fail("INVALID_FEATURE_SHAPE", f"{label} must be positive NCHW, got {value!r}")
    return parts


def _normalize_runtime_shape(value: Any, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail("FEATURE_SHAPE_INVALID", f"{label} must be an integer sequence")
    values: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            _fail("FEATURE_SHAPE_INVALID", f"{label} contains non-integer {item!r}")
        values.append(int(item))
    result = tuple(values)
    if len(result) != 4 or any(item <= 0 for item in result):
        _fail("FEATURE_SHAPE_INVALID", f"{label} must be positive NCHW, got {result}")
    return result


def registry_row_fingerprint(row: Mapping[str, Any]) -> str:
    """Reproduce the UE-A1 fingerprint from exact CSV scalar values."""

    source = {
        str(key): "" if value is None else str(value)
        for key, value in row.items()
        if key != "row_fingerprint_sha256"
    }
    return hashlib.sha256(canonical_json_bytes(source)).hexdigest()


def verify_registry_row_fingerprint(row: Mapping[str, Any]) -> str:
    expected = str(row.get("row_fingerprint_sha256") or "")
    actual = registry_row_fingerprint(row)
    if len(expected) != 64 or expected != actual:
        _fail(
            "REGISTRY_ROW_FINGERPRINT_MISMATCH",
            f"profile={row.get('profile_id', '<unknown>')} expected={expected} actual={actual}",
        )
    return actual


def _required_row_fields() -> set[str]:
    return {
        "registry_schema",
        "registry_id",
        "action_index",
        "profile_id",
        "model_family",
        "ae_bottleneck_channels",
        "ae_source",
        "ae_arch",
        "external_ae_override_allowed",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_bytes",
        "input_schema_id",
        "input_width",
        "input_height",
        "rgb_channels",
        "radar_channels",
        "feature_schema_id",
        "feature_levels",
        "expected_low_shape",
        "expected_high_native_shape",
        "expected_high_wire_shape",
        "expected_high_after_edge_decode_shape",
        "quantization_mode",
        "quantization_bits",
        "roi_drop_fraction",
        "roi_semantics",
        "entropy_coder",
        "entropy_level",
        "feature_wire_schema_id",
        "udp_chunk_bytes",
        "udp_chunk_header_struct",
        "object_classes",
        "object_channels",
        "object_score_threshold",
        "object_nms_radius_px",
        "topk_objects",
        "max_objects_published",
        "map_output_schema_id",
        "edge_container_checkpoint_path",
        "row_fingerprint_sha256",
    }


def load_registered_profiles(
    registry_csv: Path = DEFAULT_REGISTRY_CSV,
    *,
    expected_registry_sha256: str | None = A1_REGISTRY_CSV_SHA256,
) -> tuple[RegisteredSplitProfile, ...]:
    """Load and verify the complete immutable UE-A1 action registry."""

    path = Path(registry_csv).expanduser().resolve()
    if not path.is_file():
        _fail("REGISTRY_FILE_MISSING", str(path))
    actual_registry_sha256 = sha256_file(path)
    if expected_registry_sha256 is not None and actual_registry_sha256 != str(
        expected_registry_sha256
    ):
        _fail(
            "REGISTRY_FILE_HASH_MISMATCH",
            f"expected={expected_registry_sha256} actual={actual_registry_sha256} path={path}",
        )

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                _fail("REGISTRY_HEADER_MISSING", str(path))
            missing = sorted(_required_row_fields() - set(reader.fieldnames))
            if missing:
                _fail("REGISTRY_FIELDS_MISSING", ",".join(missing))
            rows = [dict(row) for row in reader]
    except csv.Error as exc:
        _fail("REGISTRY_CSV_INVALID", f"{path}: {exc}")

    if len(rows) != EXPECTED_A1_PROFILE_COUNT:
        _fail(
            "REGISTRY_PROFILE_COUNT_MISMATCH",
            f"expected={EXPECTED_A1_PROFILE_COUNT} actual={len(rows)}",
        )

    profile_ids: set[str] = set()
    action_indexes: set[int] = set()
    profiles: list[RegisteredSplitProfile] = []
    for row in rows:
        if None in row:
            _fail("REGISTRY_CSV_INVALID", "row contains values beyond the declared header")
        if row.get("registry_schema") != A1_REGISTRY_SCHEMA:
            _fail("REGISTRY_SCHEMA_MISMATCH", str(row.get("registry_schema")))
        if row.get("registry_id") != A1_REGISTRY_ID:
            _fail("REGISTRY_ID_MISMATCH", str(row.get("registry_id")))
        verify_registry_row_fingerprint(row)

        profile_id = str(row.get("profile_id") or "")
        if not profile_id:
            _fail("PROFILE_ID_MISSING", "registry row has an empty profile_id")
        if profile_id in profile_ids:
            _fail("PROFILE_ID_DUPLICATE", profile_id)
        profile_ids.add(profile_id)

        action_index = _strict_int(row.get("action_index"), "action_index")
        if action_index in action_indexes:
            _fail("ACTION_INDEX_DUPLICATE", str(action_index))
        action_indexes.add(action_index)

        action_contract = build_action_contract(row)
        contract_sha256 = hashlib.sha256(canonical_json_bytes(action_contract)).hexdigest()
        profiles.append(
            RegisteredSplitProfile(
                registry_path=path,
                registry_sha256=actual_registry_sha256,
                row=MappingProxyType(dict(row)),
                action_contract_sha256=contract_sha256,
            )
        )

    if action_indexes != set(range(EXPECTED_A1_PROFILE_COUNT)):
        _fail(
            "ACTION_INDEX_SET_MISMATCH",
            f"expected=0..{EXPECTED_A1_PROFILE_COUNT - 1} actual={sorted(action_indexes)}",
        )
    return tuple(sorted(profiles, key=lambda profile: int(profile.row["action_index"])))


def resolve_registered_profile(
    profile_id: str,
    registry_csv: Path = DEFAULT_REGISTRY_CSV,
    *,
    expected_registry_sha256: str | None = A1_REGISTRY_CSV_SHA256,
) -> RegisteredSplitProfile:
    requested = str(profile_id)
    if not requested:
        _fail("PROFILE_ID_MISSING", "requested profile_id is empty")
    matches = [
        profile
        for profile in load_registered_profiles(
            registry_csv, expected_registry_sha256=expected_registry_sha256
        )
        if profile.profile_id == requested
    ]
    if not matches:
        _fail("PROFILE_NOT_FOUND", requested)
    if len(matches) != 1:  # Defensive; registry loading already rejects duplicates.
        _fail("PROFILE_ID_DUPLICATE", requested)
    return matches[0]


def build_action_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build a stable digest input from immutable operational fields only.

    Paths, source-code hashes, evidence references, quality values, launch
    argument strings, and mutable smoke/validity statuses are intentionally
    excluded.  Consequently UE-A2/UE-A3 annotations do not change the action's
    semantic identity.
    """

    missing = sorted(_required_row_fields() - set(row))
    if missing:
        _fail("REGISTRY_FIELDS_MISSING", ",".join(missing))

    low = _parse_shape(row["expected_low_shape"], "expected_low_shape")
    native_high = _parse_shape(
        row["expected_high_native_shape"], "expected_high_native_shape"
    )
    wire_high = _parse_shape(row["expected_high_wire_shape"], "expected_high_wire_shape")
    edge_high = _parse_shape(
        row["expected_high_after_edge_decode_shape"],
        "expected_high_after_edge_decode_shape",
    )
    levels = _split_pipe(str(row["feature_levels"]), "feature_levels")
    if levels != ["low", "high"]:
        _fail("FEATURE_LEVELS_MISMATCH", repr(levels))

    quantization_bits = _strict_int(row["quantization_bits"], "quantization_bits")
    quantizer_to_bits = {
        "per_channel_uint8": 8,
        "per_channel_uint6": 6,
        "per_channel_uint4": 4,
    }
    quantization_mode = str(row["quantization_mode"])
    if quantizer_to_bits.get(quantization_mode) != quantization_bits:
        _fail(
            "QUANTIZATION_BINDING_MISMATCH",
            f"mode={quantization_mode} bits={quantization_bits}",
        )

    checkpoint_sha256 = str(row["checkpoint_sha256"])
    if len(checkpoint_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_sha256
    ):
        _fail("CHECKPOINT_SHA256_INVALID", checkpoint_sha256)

    return {
        "schema": ACTION_CONTRACT_SCHEMA,
        "registry": {
            "registry_schema": str(row["registry_schema"]),
            "registry_id": str(row["registry_id"]),
            "action_index": _strict_int(row["action_index"], "action_index"),
            "profile_id": str(row["profile_id"]),
        },
        "model": {
            "family": str(row["model_family"]),
            "ae_bottleneck_channels": _strict_int(
                row["ae_bottleneck_channels"], "ae_bottleneck_channels"
            ),
            "ae_source": str(row["ae_source"]),
            "ae_arch": str(row["ae_arch"]),
            "external_ae_override_allowed": _strict_bool(
                row["external_ae_override_allowed"], "external_ae_override_allowed"
            ),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_bytes": _strict_int(row["checkpoint_bytes"], "checkpoint_bytes"),
        },
        "input": {
            "schema_id": str(row["input_schema_id"]),
            "width": _strict_int(row["input_width"], "input_width"),
            "height": _strict_int(row["input_height"], "input_height"),
            "rgb_channels": _strict_int(row["rgb_channels"], "rgb_channels"),
            "radar_channels": _strict_int(row["radar_channels"], "radar_channels"),
        },
        "features": {
            "schema_id": str(row["feature_schema_id"]),
            "levels": levels,
            "low_shape": list(low),
            "high_native_shape": list(native_high),
            "high_wire_shape": list(wire_high),
            "high_after_edge_decode_shape": list(edge_high),
        },
        "action": {
            "quantization_mode": quantization_mode,
            "quantization_bits": quantization_bits,
            "roi_drop_fraction": _canonical_decimal(
                row["roi_drop_fraction"], "roi_drop_fraction"
            ),
            "roi_semantics": str(row["roi_semantics"]),
        },
        "transport": {
            "entropy_coder": str(row["entropy_coder"]),
            "entropy_level": _strict_int(row["entropy_level"], "entropy_level"),
            "feature_wire_schema_id": str(row["feature_wire_schema_id"]),
            "udp_chunk_bytes": _strict_int(row["udp_chunk_bytes"], "udp_chunk_bytes"),
            "udp_chunk_header_struct": str(row["udp_chunk_header_struct"]),
        },
        "decoder": {
            "object_classes": _split_pipe(str(row["object_classes"]), "object_classes"),
            "object_channels": _strict_int(row["object_channels"], "object_channels"),
            "score_threshold": _canonical_decimal(
                row["object_score_threshold"], "object_score_threshold"
            ),
            "nms_radius_px": _strict_int(
                row["object_nms_radius_px"], "object_nms_radius_px"
            ),
            "topk_objects": _strict_int(row["topk_objects"], "topk_objects"),
            "max_objects_published": _strict_int(
                row["max_objects_published"], "max_objects_published"
            ),
        },
        "map_output_schema_id": str(row["map_output_schema_id"]),
    }


def action_contract_sha256(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(build_action_contract(row))).hexdigest()


def build_wire_identity(
    row: Mapping[str, Any], contract_sha256: str | None = None
) -> dict[str, Any]:
    expected_digest = action_contract_sha256(row)
    digest = contract_sha256 or expected_digest
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        _fail("ACTION_CONTRACT_SHA256_INVALID", str(digest))
    if digest != expected_digest:
        _fail(
            "ACTION_CONTRACT_SHA256_MISMATCH",
            f"expected={expected_digest} actual={digest}",
        )
    return {
        "schema": WIRE_IDENTITY_SCHEMA,
        "registry_id": str(row["registry_id"]),
        "profile_id": str(row["profile_id"]),
        "action_contract_sha256": digest,
        "checkpoint_sha256": str(row["checkpoint_sha256"]),
        "feature_schema_id": str(row["feature_schema_id"]),
        "feature_wire_schema_id": str(row["feature_wire_schema_id"]),
        "quantization_mode": str(row["quantization_mode"]),
        "roi_drop_fraction": _canonical_decimal(
            row["roi_drop_fraction"], "roi_drop_fraction"
        ),
        "entropy_coder": str(row["entropy_coder"]),
        "entropy_level": _strict_int(row["entropy_level"], "entropy_level"),
        "udp_chunk_bytes": _strict_int(row["udp_chunk_bytes"], "udp_chunk_bytes"),
    }


def validate_wire_identity(
    observed: Any, expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Require an exact, typed identity mapping; no missing or extra fields."""

    if not isinstance(observed, Mapping):
        _fail("WIRE_IDENTITY_MISSING", "profile_identity must be a mapping")
    expected_keys = set(expected)
    observed_keys = set(observed)
    if observed_keys != expected_keys:
        _fail(
            "WIRE_IDENTITY_KEYS_MISMATCH",
            f"missing={sorted(expected_keys - observed_keys)} extra={sorted(observed_keys - expected_keys)}",
        )
    for key in sorted(expected_keys):
        actual_value = observed[key]
        expected_value = expected[key]
        if type(actual_value) is not type(expected_value) or actual_value != expected_value:
            _fail(
                "WIRE_IDENTITY_VALUE_MISMATCH",
                f"field={key} expected={expected_value!r} actual={actual_value!r}",
            )
    return dict(observed)


def _arg(args: Any, name: str) -> Any:
    if isinstance(args, Mapping):
        if name not in args:
            _fail("RUNTIME_ARGUMENT_MISSING", name)
        return args[name]
    if not hasattr(args, name):
        _fail("RUNTIME_ARGUMENT_MISSING", name)
    return getattr(args, name)


def _expect_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        _fail("RUNTIME_BINDING_MISMATCH", f"{label}: expected={expected!r} actual={actual!r}")


def validate_runtime_binding(
    profile: RegisteredSplitProfile,
    args: Any,
    *,
    checkpoint_path: Path,
    role: str | None = None,
) -> dict[str, Any]:
    """Validate one process' actual fixed-profile arguments and checkpoint.

    ``args`` may be an ``argparse.Namespace`` or mapping.  Back/loopback roles
    must also prove the evidence-compatible object decoder settings.
    """

    row = profile.row
    actual_role = str(role if role is not None else _arg(args, "role"))
    if actual_role not in {"front", "back", "loopback"}:
        _fail("RUNTIME_ROLE_INVALID", actual_role)

    if isinstance(args, Mapping):
        supplied_profile_id = args.get("ue_profile_id", profile.profile_id)
    else:
        supplied_profile_id = getattr(args, "ue_profile_id", profile.profile_id)
    _expect_equal("ue_profile_id", str(supplied_profile_id), profile.profile_id)

    _expect_equal(
        "quantization_mode",
        str(_arg(args, "quantization_mode")),
        row["quantization_mode"],
    )
    _expect_equal(
        "roi_drop_fraction",
        _canonical_decimal(_arg(args, "roi_threshold"), "roi_threshold"),
        _canonical_decimal(row["roi_drop_fraction"], "roi_drop_fraction"),
    )
    _expect_equal("entropy_coder", str(_arg(args, "entropy_coder")), row["entropy_coder"])
    _expect_equal(
        "entropy_level",
        _strict_int(_arg(args, "zstd_level"), "zstd_level"),
        _strict_int(row["entropy_level"], "entropy_level"),
    )
    _expect_equal(
        "chunk_bytes",
        _strict_int(_arg(args, "chunk_bytes"), "chunk_bytes"),
        _strict_int(row["udp_chunk_bytes"], "udp_chunk_bytes"),
    )
    _expect_equal(
        "model_input_width",
        _strict_int(_arg(args, "model_input_width"), "model_input_width"),
        _strict_int(row["input_width"], "input_width"),
    )
    _expect_equal(
        "model_input_height",
        _strict_int(_arg(args, "model_input_height"), "model_input_height"),
        _strict_int(row["input_height"], "input_height"),
    )

    external_ae = str(_arg(args, "ae_checkpoint") or "")
    if external_ae and not _strict_bool(
        row["external_ae_override_allowed"], "external_ae_override_allowed"
    ):
        _fail(
            "EXTERNAL_AE_OVERRIDE_FORBIDDEN",
            f"profile={profile.profile_id} ae_checkpoint={external_ae}",
        )

    if actual_role in {"back", "loopback"}:
        _expect_equal(
            "object_score_threshold",
            _canonical_decimal(
                _arg(args, "object_score_threshold"), "object_score_threshold"
            ),
            _canonical_decimal(row["object_score_threshold"], "object_score_threshold"),
        )
        _expect_equal(
            "object_nms_radius_px",
            _strict_int(_arg(args, "object_nms_radius_px"), "object_nms_radius_px"),
            _strict_int(row["object_nms_radius_px"], "object_nms_radius_px"),
        )
        _expect_equal(
            "topk_objects",
            _strict_int(_arg(args, "topk_objects"), "topk_objects"),
            _strict_int(row["topk_objects"], "topk_objects"),
        )
        _expect_equal(
            "max_objects_published",
            _strict_int(_arg(args, "max_objects_drawn"), "max_objects_drawn"),
            _strict_int(row["max_objects_published"], "max_objects_published"),
        )

    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    actual_checkpoint_sha256 = sha256_file(resolved_checkpoint)
    expected_checkpoint_sha256 = row["checkpoint_sha256"]
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        _fail(
            "CHECKPOINT_HASH_MISMATCH",
            f"profile={profile.profile_id} expected={expected_checkpoint_sha256} "
            f"actual={actual_checkpoint_sha256} path={resolved_checkpoint}",
        )
    expected_bytes = _strict_int(row["checkpoint_bytes"], "checkpoint_bytes")
    actual_bytes = resolved_checkpoint.stat().st_size
    if actual_bytes != expected_bytes:
        _fail(
            "CHECKPOINT_SIZE_MISMATCH",
            f"expected={expected_bytes} actual={actual_bytes} path={resolved_checkpoint}",
        )

    return {
        "profile_id": profile.profile_id,
        "role": actual_role,
        "checkpoint_path": str(resolved_checkpoint),
        "checkpoint_sha256": actual_checkpoint_sha256,
        "checkpoint_bytes": actual_bytes,
        "action_contract_sha256": profile.action_contract_sha256,
        "profile_identity": dict(profile.wire_identity),
    }


def _expected_shapes(
    profile: RegisteredSplitProfile, stage: str
) -> dict[str, tuple[int, ...]]:
    if stage == "wire":
        return profile.expected_wire_shapes
    if stage == "native":
        return profile.expected_native_shapes
    if stage == "edge_decoded":
        return profile.expected_edge_decoded_shapes
    _fail("FEATURE_SHAPE_STAGE_INVALID", stage)


def validate_declared_feature_shapes(
    profile: RegisteredSplitProfile,
    observed: Any,
    *,
    stage: str = "wire",
) -> dict[str, tuple[int, ...]]:
    if not isinstance(observed, Mapping):
        _fail("FEATURE_SHAPES_MISSING", "feature_shapes must be a mapping")
    expected = _expected_shapes(profile, stage)
    if set(observed) != set(expected):
        _fail(
            "FEATURE_LEVELS_MISMATCH",
            f"stage={stage} missing={sorted(set(expected) - set(observed))} "
            f"extra={sorted(set(observed) - set(expected))}",
        )
    normalized: dict[str, tuple[int, ...]] = {}
    for level in sorted(expected):
        shape = _normalize_runtime_shape(observed[level], f"feature_shapes.{level}")
        if shape != expected[level]:
            _fail(
                "FEATURE_SHAPE_MISMATCH",
                f"stage={stage} level={level} expected={expected[level]} actual={shape}",
            )
        normalized[level] = shape
    return normalized


def _expected_packed_data_bytes(total_values: int, bitdepth: int) -> int:
    if bitdepth == 8:
        return total_values
    if bitdepth == 6:
        return ((total_values + 3) // 4) * 3
    if bitdepth == 4:
        return (total_values + 1) // 2
    _fail("QUANTIZATION_BITDEPTH_INVALID", str(bitdepth))


def validate_serialized_feature_headers(
    profile: RegisteredSplitProfile,
    serialized: Any,
) -> dict[str, tuple[int, int, int, int]]:
    """Validate codec field sets, headers, and byte lengths before decoding."""

    if not isinstance(serialized, Mapping):
        _fail("SERIALIZED_FEATURES_MISSING", "features must be a mapping")
    expected_shapes = profile.expected_wire_shapes
    if set(serialized) != set(expected_shapes):
        _fail(
            "FEATURE_LEVELS_MISMATCH",
            f"serialized missing={sorted(set(expected_shapes) - set(serialized))} "
            f"extra={sorted(set(serialized) - set(expected_shapes))}",
        )
    expected_bits = _strict_int(profile.row["quantization_bits"], "quantization_bits")
    observed_headers: dict[str, tuple[int, int, int, int]] = {}
    for level in sorted(expected_shapes):
        wire = serialized[level]
        if not isinstance(wire, Mapping):
            _fail("FEATURE_WIRE_INVALID", f"level={level} must be a mapping")
        expected_fields = {"header", "ranges", "data"}
        if set(wire) != expected_fields:
            _fail(
                "FEATURE_WIRE_FIELDS_MISMATCH",
                f"level={level} missing={sorted(expected_fields - set(wire))} "
                f"extra={sorted(set(wire) - expected_fields)}",
            )
        for field in sorted(expected_fields):
            if not isinstance(wire[field], (bytes, bytearray, memoryview)):
                _fail(
                    "FEATURE_WIRE_FIELD_TYPE_INVALID",
                    f"level={level} field={field} type={type(wire[field]).__name__}",
                )
        header = bytes(wire["header"])
        if len(header) != PER_CHANNEL_HEADER.size:
            _fail(
                "FEATURE_WIRE_HEADER_INVALID",
                f"level={level} expected_bytes={PER_CHANNEL_HEADER.size} actual={len(header)}",
            )
        channels, height, width, bitdepth = PER_CHANNEL_HEADER.unpack(header)
        expected_shape = expected_shapes[level]
        actual_shape = (1, int(channels), int(height), int(width))
        if actual_shape != expected_shape or int(bitdepth) != expected_bits:
            _fail(
                "FEATURE_WIRE_HEADER_MISMATCH",
                f"level={level} expected_shape={expected_shape} actual_shape={actual_shape} "
                f"expected_bits={expected_bits} actual_bits={bitdepth}",
            )
        ranges_bytes = len(wire["ranges"])
        expected_ranges_bytes = int(channels) * 2 * 4
        if ranges_bytes != expected_ranges_bytes:
            _fail(
                "FEATURE_WIRE_LENGTH_MISMATCH",
                f"level={level} field=ranges expected={expected_ranges_bytes} actual={ranges_bytes}",
            )
        total = int(channels) * int(height) * int(width)
        expected_data_bytes = _expected_packed_data_bytes(total, int(bitdepth))
        data_bytes = len(wire["data"])
        if data_bytes != expected_data_bytes:
            _fail(
                "FEATURE_WIRE_LENGTH_MISMATCH",
                f"level={level} field=data expected={expected_data_bytes} actual={data_bytes}",
            )
        observed_headers[level] = (
            int(channels),
            int(height),
            int(width),
            int(bitdepth),
        )
    return observed_headers


def validate_feature_payload(
    profile: RegisteredSplitProfile,
    payload: Any,
) -> dict[str, Any]:
    """Run all checks that must precede edge queueing and feature decode."""

    if not isinstance(payload, Mapping):
        _fail("FEATURE_PAYLOAD_INVALID", "payload must be a mapping")

    # Deliberate order: identity is rejected before inspecting tensor metadata.
    identity = validate_wire_identity(payload.get("profile_identity"), profile.wire_identity)
    shapes = validate_declared_feature_shapes(
        profile, payload.get("feature_shapes"), stage="wire"
    )
    headers = validate_serialized_feature_headers(profile, payload.get("features"))
    if "batch_size" in payload:
        batch_size = _strict_int(payload["batch_size"], "batch_size")
        if batch_size != 1:
            _fail("FEATURE_BATCH_SIZE_MISMATCH", f"expected=1 actual={batch_size}")
    if "model_input_size" in payload:
        value = payload["model_input_size"]
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            _fail("MODEL_INPUT_SIZE_MISMATCH", repr(value))
        observed_input = tuple(value)
        expected_input = (
            _strict_int(profile.row["input_width"], "input_width"),
            _strict_int(profile.row["input_height"], "input_height"),
        )
        if observed_input != expected_input:
            _fail(
                "MODEL_INPUT_SIZE_MISMATCH",
                f"expected={expected_input} actual={observed_input}",
            )
    return {
        "profile_identity": identity,
        "feature_shapes": shapes,
        "feature_headers": headers,
    }


def build_launch_binding(profile: RegisteredSplitProfile) -> dict[str, Any]:
    """Return deterministic host/edge paths and argv arrays for one profile.

    Consumers should extract the JSON arrays as arrays (for example with
    ``jq``) and must not evaluate them as shell source.
    """

    row = profile.row
    host_checkpoint = (ROOT / row["checkpoint_path"]).resolve()
    actual_checkpoint_sha256 = sha256_file(host_checkpoint)
    if actual_checkpoint_sha256 != row["checkpoint_sha256"]:
        _fail(
            "CHECKPOINT_HASH_MISMATCH",
            f"profile={profile.profile_id} expected={row['checkpoint_sha256']} "
            f"actual={actual_checkpoint_sha256} path={host_checkpoint}",
        )
    actual_checkpoint_bytes = host_checkpoint.stat().st_size
    expected_checkpoint_bytes = _strict_int(row["checkpoint_bytes"], "checkpoint_bytes")
    if actual_checkpoint_bytes != expected_checkpoint_bytes:
        _fail(
            "CHECKPOINT_SIZE_MISMATCH",
            f"expected={expected_checkpoint_bytes} actual={actual_checkpoint_bytes} "
            f"path={host_checkpoint}",
        )

    try:
        registry_relative = profile.registry_path.relative_to(ROOT)
    except ValueError:
        _fail(
            "REGISTRY_CONTAINER_PATH_UNRESOLVED",
            f"registry must be within repository root {ROOT}: {profile.registry_path}",
        )

    relative_checkpoint = str(row["checkpoint_path"]).lstrip("/")
    edge_checkpoint = str(row.get("edge_container_checkpoint_path") or "")
    suffix = f"/{relative_checkpoint}"
    if not edge_checkpoint.endswith(suffix):
        _fail(
            "EDGE_CHECKPOINT_PATH_INVALID",
            f"profile={profile.profile_id} path={edge_checkpoint!r}",
        )
    container_root = edge_checkpoint[: -len(suffix)]
    if not container_root.startswith("/"):
        _fail("EDGE_CHECKPOINT_PATH_INVALID", edge_checkpoint)
    edge_registry = f"{container_root}/{registry_relative.as_posix()}"

    common = [
        "--quantization-mode",
        row["quantization_mode"],
        "--entropy-coder",
        row["entropy_coder"],
        "--zstd-level",
        str(_strict_int(row["entropy_level"], "entropy_level")),
        "--roi-threshold",
        _canonical_decimal(row["roi_drop_fraction"], "roi_drop_fraction"),
        "--chunk-bytes",
        str(_strict_int(row["udp_chunk_bytes"], "udp_chunk_bytes")),
        "--model-input-width",
        str(_strict_int(row["input_width"], "input_width")),
        "--model-input-height",
        str(_strict_int(row["input_height"], "input_height")),
    ]
    front_binding = [
        "--ue-profile-registry-csv",
        str(profile.registry_path),
        "--ue-profile-id",
        profile.profile_id,
        "--require-ue-profile-binding",
    ]
    edge_binding = [
        "--ue-profile-registry-csv",
        edge_registry,
        "--ue-profile-id",
        profile.profile_id,
        "--require-ue-profile-binding",
    ]
    front_args = ["--fusion-checkpoint", str(host_checkpoint), *common, *front_binding]
    edge_args = [
        "--fusion-checkpoint",
        edge_checkpoint,
        *common,
        "--object-score-threshold",
        _canonical_decimal(row["object_score_threshold"], "object_score_threshold"),
        "--object-nms-radius-px",
        str(_strict_int(row["object_nms_radius_px"], "object_nms_radius_px")),
        "--topk-objects",
        str(_strict_int(row["topk_objects"], "topk_objects")),
        "--max-objects-drawn",
        str(_strict_int(row["max_objects_published"], "max_objects_published")),
        *edge_binding,
    ]
    return {
        "schema": LAUNCH_BINDING_SCHEMA,
        "registry_id": row["registry_id"],
        "registry_sha256": profile.registry_sha256,
        "profile_id": profile.profile_id,
        "action_index": _strict_int(row["action_index"], "action_index"),
        "action_contract_sha256": profile.action_contract_sha256,
        "profile_identity": dict(profile.wire_identity),
        "registry_paths": {
            "host": str(profile.registry_path),
            "container": edge_registry,
        },
        "checkpoint_paths": {
            "host": str(host_checkpoint),
            "container": edge_checkpoint,
        },
        "front_args": front_args,
        "edge_args": edge_args,
    }


def _parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve one immutable UE split profile as safe JSON argv arrays."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    resolve = subparsers.add_parser("resolve", help="Resolve exactly one registered profile.")
    resolve.add_argument("--profile-id", required=True)
    resolve.add_argument("--registry-csv", type=Path, default=DEFAULT_REGISTRY_CSV)
    resolve.add_argument(
        "--registry-sha256",
        default=A1_REGISTRY_CSV_SHA256,
        help="Expected immutable registry CSV SHA-256.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli(argv)
    try:
        if args.command != "resolve":  # argparse currently makes this unreachable.
            _fail("CLI_COMMAND_INVALID", str(args.command))
        profile = resolve_registered_profile(
            args.profile_id,
            args.registry_csv,
            expected_registry_sha256=args.registry_sha256,
        )
        output = build_launch_binding(profile)
    except SplitWireContractError as exc:
        import sys

        print(json.dumps({"error": exc.code, "detail": exc.detail}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "A1_REGISTRY_CSV_SHA256",
    "ACTION_CONTRACT_SCHEMA",
    "DEFAULT_REGISTRY_CSV",
    "LAUNCH_BINDING_SCHEMA",
    "RegisteredSplitProfile",
    "SplitWireContractError",
    "WIRE_IDENTITY_SCHEMA",
    "action_contract_sha256",
    "build_action_contract",
    "build_launch_binding",
    "build_wire_identity",
    "canonical_json_bytes",
    "load_registered_profiles",
    "registry_row_fingerprint",
    "resolve_registered_profile",
    "sha256_file",
    "validate_declared_feature_shapes",
    "validate_feature_payload",
    "validate_runtime_binding",
    "validate_serialized_feature_headers",
    "validate_wire_identity",
    "verify_registry_row_fingerprint",
]


if __name__ == "__main__":
    raise SystemExit(main())
