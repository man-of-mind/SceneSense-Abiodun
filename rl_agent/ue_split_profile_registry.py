"""Build the static 72-action UE split-profile registry.

UE-A1 is deliberately static.  It proves that every measured profile has a
complete, hash-bound model/codec/decoder declaration.  It does not run model
inference, CARLA, OAI, or the UE-to-edge wire smoke; those remain UE-A2.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "rl_agent/configs/ue_split_profile_registry_v1.json"
CONFIG_SCHEMA = "scenesense.ue_split_profile_registry_config.v1"
MANIFEST_SCHEMA = "scenesense.ue_split_profile_registry_manifest.v1"
TERMINAL_SCHEMA = "scenesense.ue_split_profile_registry_decision.v1"
REGISTRY_SCHEMA = "scenesense.ue_split_profile_registry.v1"


REGISTRY_FIELDS = (
    "registry_schema",
    "registry_id",
    "action_index",
    "profile_id",
    "display_profile_id",
    "model_family",
    "ae_bottleneck_channels",
    "ae_source",
    "ae_arch",
    "external_ae_override_allowed",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "checkpoint_trial_name",
    "checkpoint_epoch",
    "checkpoint_strict_structure_status",
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
    "feature_shape_status",
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
    "decoder_override_required",
    "map_output_schema_id",
    "runtime_path",
    "runtime_sha256",
    "edge_container_checkpoint_path",
    "edge_decoder_checkpoint_sha256",
    "edge_binding_mode",
    "edge_decoder_binding_status",
    "edge_launcher_propagation_status",
    "wire_profile_identity_present",
    "wire_mismatch_rejection_present",
    "quality_source_profile_id",
    "quality_source_sha256",
    "offline_payload_semantics",
    "quality_mask_applied",
    "static_binding_status",
    "wire_smoke_status",
    "technical_validity_status",
    "technical_invalid_reason",
    "front_profile_launch_args_json",
    "edge_profile_launch_args_json",
    "row_fingerprint_sha256",
)


class ProfileRegistryError(RuntimeError):
    """Raised when a profile binding cannot be proven exactly."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _csv_scalar(value: Any) -> str:
    """Return the exact scalar representation written by ``csv.DictWriter``."""

    return "" if value is None else str(value)


def registry_row_fingerprint(row: Mapping[str, Any]) -> str:
    """Hash a registry row exactly as a ``csv.DictReader`` consumer sees it."""

    source = {
        field: _csv_scalar(row[field])
        for field in REGISTRY_FIELDS
        if field != "row_fingerprint_sha256"
    }
    return hashlib.sha256(canonical_json_bytes(source)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileRegistryError(f"{label} must be a mapping")
    return value


def _repo_path(root: Path, relative: str) -> Path:
    path = (root / str(relative)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ProfileRegistryError(f"path escapes repository root: {relative}") from exc
    return path


def _pinned_path(root: Path, relative: str, expected_sha256: str, label: str) -> Path:
    path = _repo_path(root, relative)
    if not path.is_file():
        raise ProfileRegistryError(f"missing {label}: {relative}")
    actual = sha256_file(path)
    if actual != str(expected_sha256):
        raise ProfileRegistryError(
            f"{label} hash drift: expected={expected_sha256} actual={actual} path={relative}"
        )
    return path


def _format_q(value: float) -> str:
    return f"{float(value):.12g}"


def _quant_short(mode: str) -> str:
    mapping = {
        "per_channel_uint8": "u8",
        "per_channel_uint6": "u6",
        "per_channel_uint4": "u4",
    }
    try:
        return mapping[str(mode)]
    except KeyError as exc:
        raise ProfileRegistryError(f"unsupported registered quantizer: {mode}") from exc


def canonical_profile_id(
    family: str,
    quantization_mode: str,
    q: float,
    entropy_level: int,
    checkpoint_sha256: str,
) -> str:
    return (
        f"{family}__{_quant_short(quantization_mode)}__q{_format_q(q)}"
        f"__zstd{int(entropy_level)}__ckpt{str(checkpoint_sha256)[:12]}"
    )


def load_config(path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ProfileRegistryError("invalid registry config schema")
    root = (path.parent / str(config.get("repository_root", ""))).resolve()
    if root != ROOT:
        raise ProfileRegistryError(f"registry repository root mismatch: {root} != {ROOT}")

    authority = _require_mapping(config.get("authority"), "authority")
    expected_authority = {
        "static_checkpoint_inspection": True,
        "model_inference": False,
        "carla_run": False,
        "oai_run": False,
        "profile_quality_filter": False,
        "wire_smoke": False,
        "policy_training": False,
    }
    if dict(authority) != expected_authority:
        raise ProfileRegistryError("UE-A1 authority must be static-only with no run/filter authority")

    factors = _require_mapping(config.get("factor_contract"), "factor_contract")
    models = _require_mapping(factors.get("models"), "factor_contract.models")
    quantizers = _require_mapping(
        factors.get("quantizers"), "factor_contract.quantizers"
    )
    q_values = list(factors.get("roi_drop_fractions", []))
    if len(models) != 4 or len(quantizers) != 3 or len(q_values) != 6:
        raise ProfileRegistryError("registry factors must be exactly 4 models x 3 quantizers x 6 q")
    if len({float(value) for value in q_values}) != 6 or any(
        float(value) < 0.0 or float(value) >= 1.0 for value in q_values
    ):
        raise ProfileRegistryError("registered q values must be six unique values in [0,1)")
    expected_count = len(models) * len(quantizers) * len(q_values)
    if int(factors.get("expected_profiles", -1)) != expected_count or expected_count != 72:
        raise ProfileRegistryError("expected profile count must be exactly 72")
    if factors.get("entropy_coder") != "zstd" or int(factors.get("entropy_level", -1)) != 3:
        raise ProfileRegistryError("the measured registry must use zstd level 3")

    runtime = _require_mapping(config.get("runtime_contract"), "runtime_contract")
    if runtime.get("binding_mode") != "fixed_profile_per_process_launch":
        raise ProfileRegistryError("UE-A1 supports only the fixed-profile launch binding")
    if runtime.get("container_repository_root") != "/work/abiodun":
        raise ProfileRegistryError("edge container repository root must be /work/abiodun")
    if runtime.get("edge_profile_overrides_propagated_by_current_launcher") is not False:
        raise ProfileRegistryError(
            "current edge launcher override propagation must be recorded truthfully as absent"
        )
    if runtime.get("integrated_ae_external_override_allowed") is not False:
        raise ProfileRegistryError("external AE override must be forbidden")
    if runtime.get("wire_profile_identity_present") is not False:
        raise ProfileRegistryError("current payload identity must be recorded truthfully as absent")
    if runtime.get("wire_mismatch_rejection_present") is not False:
        raise ProfileRegistryError("current wire mismatch rejection must be recorded as absent")

    decoder = _require_mapping(config.get("decoder_contract"), "decoder_contract")
    if decoder.get("launch_override_required") is not True:
        raise ProfileRegistryError("evidence-compatible decoder overrides must be required")
    if decoder.get("live_defaults_match_registered_evidence") is not False:
        raise ProfileRegistryError("live decoder defaults must not be claimed evidence-compatible")
    if int(decoder.get("max_objects_published", -1)) < int(decoder.get("topk_objects", -1)):
        raise ProfileRegistryError("published-object cap must be at least the registered top-k")
    return config, root


def _argparse_defaults(source: str) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        try:
            option = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            continue
        if not isinstance(option, str) or not option.startswith("--"):
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                try:
                    defaults[option] = ast.literal_eval(keyword.value)
                except (ValueError, TypeError):
                    pass
    return defaults


def _front_payload_keys(source: str) -> set[str]:
    tree = ast.parse(source)
    for class_node in tree.body:
        if not isinstance(class_node, ast.ClassDef) or class_node.name != "CameraSideFusionInference":
            continue
        for function in class_node.body:
            if not isinstance(function, ast.FunctionDef) or function.name != "process":
                continue
            for node in ast.walk(function):
                if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
                    continue
                if not any(
                    isinstance(target, ast.Name) and target.id == "payload"
                    for target in node.targets
                ):
                    continue
                keys = {
                    str(ast.literal_eval(key))
                    for key in node.value.keys
                    if key is not None and isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
                if "features" in keys:
                    return keys
    raise ProfileRegistryError("could not locate CameraSideFusionInference front payload")


def _validate_runtime(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    runtime = _require_mapping(config["runtime_contract"], "runtime_contract")
    pinned_specs = (
        ("active_runtime_path", "active_runtime_sha256"),
        ("feature_codec_path", "feature_codec_sha256"),
        ("split_runtime_path", "split_runtime_sha256"),
        ("model_source_path", "model_source_sha256"),
        ("object_decoder_path", "object_decoder_sha256"),
        ("integrated_ae_source_path", "integrated_ae_source_sha256"),
        ("oai_launcher_path", "oai_launcher_sha256"),
        ("receiver_compose_path", "receiver_compose_sha256"),
        ("receiver_fusion_overlay_path", "receiver_fusion_overlay_sha256"),
    )
    paths: dict[str, Path] = {}
    for path_key, hash_key in pinned_specs:
        paths[path_key] = _pinned_path(
            root, str(runtime[path_key]), str(runtime[hash_key]), path_key
        )

    decoder = _require_mapping(config["decoder_contract"], "decoder_contract")
    eval_settings_path = _pinned_path(
        root,
        str(decoder["source_eval_settings_path"]),
        str(decoder["source_eval_settings_sha256"]),
        "decoder evidence settings",
    )
    source_evidence = _require_mapping(config["source_evidence"], "source_evidence")
    evaluator_path = _pinned_path(
        root,
        str(source_evidence["profile_evaluator_path"]),
        str(source_evidence["profile_evaluator_sha256"]),
        "profile evaluator",
    )
    eval_settings = json.loads(eval_settings_path.read_text(encoding="utf-8"))
    eval_knobs = _require_mapping(eval_settings.get("eval_knobs"), "eval_settings.eval_knobs")
    registered_evidence_knobs = {
        "object_score_threshold": float(decoder["object_score_threshold"]),
        "object_nms_radius_px": int(decoder["object_nms_radius_px"]),
        "topk_objects": int(decoder["topk_objects"]),
    }
    observed_evidence_knobs = {
        "object_score_threshold": float(eval_knobs.get("object_score_threshold", -1)),
        "object_nms_radius_px": int(eval_knobs.get("object_nms_radius_px", -1)),
        "topk_objects": int(eval_knobs.get("topk_objects", -1)),
    }
    if observed_evidence_knobs != registered_evidence_knobs:
        raise ProfileRegistryError(
            "registered decoder does not match the pinned profile-evidence settings: "
            f"{registered_evidence_knobs} != {observed_evidence_knobs}"
        )
    factors = _require_mapping(config["factor_contract"], "factor_contract")
    if (
        str(eval_settings.get("entropy_coder")) != str(factors["entropy_coder"])
        or int(eval_settings.get("zstd_level", -1)) != int(factors["entropy_level"])
    ):
        raise ProfileRegistryError("profile-evidence entropy settings do not match registry")
    evaluator_source = evaluator_path.read_text(encoding="utf-8")
    for token in (
        "def payload_of",
        'ENTROPY_CODER, ZSTD_LEVEL = "zstd", 3',
        "ZstdCompressor(level=ZSTD_LEVEL)",
    ):
        if token not in evaluator_source:
            raise ProfileRegistryError(f"profile evaluator lost payload contract token: {token}")

    active_source = paths["active_runtime_path"].read_text(encoding="utf-8")
    payload_keys = _front_payload_keys(active_source)
    required = set(str(value) for value in runtime["required_front_payload_keys"])
    if not required.issubset(payload_keys):
        raise ProfileRegistryError(
            f"front payload lost required keys: {sorted(required - payload_keys)}"
        )
    identity_keys = set(str(value) for value in runtime["missing_identity_keys"])
    present_identity = sorted(identity_keys & payload_keys)
    if present_identity:
        raise ProfileRegistryError(
            "runtime gained identity fields; update the registry contract instead of "
            f"silently retaining the old gap: {present_identity}"
        )

    defaults = _argparse_defaults(active_source)
    observed = {
        "object_score_threshold": defaults.get("--object-score-threshold"),
        "object_nms_radius_px": defaults.get("--object-nms-radius-px"),
        "topk_objects": defaults.get("--topk-objects"),
        "max_objects_published": defaults.get("--max-objects-drawn"),
    }
    registered = {
        key: decoder[key]
        for key in (
            "object_score_threshold",
            "object_nms_radius_px",
            "topk_objects",
            "max_objects_published",
        )
    }
    if observed == registered:
        raise ProfileRegistryError(
            "config says decoder override is required, but runtime defaults now match; re-lock it"
        )
    if int(defaults.get("--chunk-bytes", -1)) != int(runtime["chunk_bytes"]):
        raise ProfileRegistryError("registered UDP chunk size differs from active runtime default")

    codec_source = paths["feature_codec_path"].read_text(encoding="utf-8")
    required_codec_tokens = (
        'QUANT_MODE_PER_CHANNEL_UINT8 = "per_channel_uint8"',
        'QUANT_MODE_PER_CHANNEL_UINT6 = "per_channel_uint6"',
        'QUANT_MODE_PER_CHANNEL_UINT4 = "per_channel_uint4"',
        "class PerChannelFeatureCodec",
        "class _ZstdCoder",
        'HEADER_STRUCT = struct.Struct("!IHH")',
    )
    missing_tokens = [token for token in required_codec_tokens if token not in codec_source]
    if missing_tokens:
        raise ProfileRegistryError(f"feature codec source lost contract tokens: {missing_tokens}")

    split_source = paths["split_runtime_path"].read_text(encoding="utf-8")
    for token in (
        "def serialize_backbone_features",
        "def deserialize_backbone_features",
        "def decode_outputs",
    ):
        if token not in split_source:
            raise ProfileRegistryError(f"split runtime lost contract token: {token}")
    for token in ("def _front_compress", "def _back_decompress", "def load_fusion_model"):
        if token not in active_source:
            raise ProfileRegistryError(f"active runtime lost contract token: {token}")

    launcher_source = paths["oai_launcher_path"].read_text(encoding="utf-8")
    for token in (
        'CHECKPOINT_CONTAINER="${CHECKPOINT_CONTAINER:-/work/abiodun/',
        'FUSION_BACK_CHECKPOINT="${CHECKPOINT_CONTAINER}"',
        'FUSION_BACK_EXTRA_ARGS="${extra_args[*]}"',
    ):
        if token not in launcher_source:
            raise ProfileRegistryError(f"OAI launcher lost edge-binding token: {token}")
    missing_edge_overrides = [
        option
        for option in (
            "--object-score-threshold",
            "--object-nms-radius-px",
            "--topk-objects",
            "--max-objects-drawn",
        )
        if option not in launcher_source
    ]
    if not missing_edge_overrides:
        raise ProfileRegistryError(
            "config records missing edge override propagation, but the launcher now contains "
            "all registered decoder options; re-lock the contract"
        )
    receiver_compose = paths["receiver_compose_path"].read_text(encoding="utf-8")
    if "../../abiodun:/work/abiodun:ro" not in receiver_compose:
        raise ProfileRegistryError("receiver container no longer mounts the repository at /work/abiodun")
    fusion_overlay = paths["receiver_fusion_overlay_path"].read_text(encoding="utf-8")
    for token in ("${FUSION_BACK_CHECKPOINT", "${FUSION_BACK_EXTRA_ARGS"):
        if token not in fusion_overlay:
            raise ProfileRegistryError(f"receiver overlay lost launch-binding token: {token}")

    return {
        "payload_keys": sorted(payload_keys),
        "missing_identity_keys": sorted(identity_keys),
        "live_decoder_defaults": observed,
        "registered_decoder_binding": registered,
        "profile_evidence_decoder_binding": observed_evidence_knobs,
        "profile_evidence_entropy_binding": {
            "entropy_coder": factors["entropy_coder"],
            "entropy_level": factors["entropy_level"],
        },
        "container_repository_root": runtime["container_repository_root"],
        "edge_profile_override_propagation": {
            "status": "PENDING_UE_A2",
            "missing_options": missing_edge_overrides,
        },
        "pinned_sources": [
            {
                "path": str(runtime[path_key]),
                "sha256": str(runtime[hash_key]),
                "bytes": paths[path_key].stat().st_size,
            }
            for path_key, hash_key in pinned_specs
        ]
        + [
            {
                "path": str(decoder["source_eval_settings_path"]),
                "sha256": str(decoder["source_eval_settings_sha256"]),
                "bytes": eval_settings_path.stat().st_size,
            },
            {
                "path": str(source_evidence["profile_evaluator_path"]),
                "sha256": str(source_evidence["profile_evaluator_sha256"]),
                "bytes": evaluator_path.stat().st_size,
            },
        ],
    }


def _strict_checkpoint_metadata(
    family: str,
    spec: Mapping[str, Any],
    service: Mapping[str, Any],
    checkpoint_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    import torch

    from pole_lraspp_multimodal_fusion.pole_lraspp_multimodal_fusion.model import (
        build_multitask_fusion_lraspp,
    )
    from rl_agent.feature_ae.ae_model import build_ae

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("model"), Mapping):
        raise ProfileRegistryError(f"{family} checkpoint lacks a model state mapping")
    trial = _require_mapping(checkpoint.get("trial"), f"{family}.checkpoint.trial")
    bottleneck = int(trial.get("ae_bottleneck", 0) or 0)
    expected_bottleneck = int(spec["ae_bottleneck_channels"])
    if bottleneck != expected_bottleneck:
        raise ProfileRegistryError(
            f"{family} AE bottleneck mismatch: {bottleneck} != {expected_bottleneck}"
        )
    expected_input = [int(service["input_width"]), int(service["input_height"])]
    if list(checkpoint.get("input_size", [])) != expected_input:
        raise ProfileRegistryError(f"{family} input size mismatch")
    if int(checkpoint.get("radar_channels", -1)) != int(service["radar_channels"]):
        raise ProfileRegistryError(f"{family} radar-channel mismatch")
    if int(checkpoint.get("object_channels", -1)) != int(service["object_channels"]):
        raise ProfileRegistryError(f"{family} object-channel mismatch")
    if list(checkpoint.get("object_class_names", [])) != list(service["object_classes"]):
        raise ProfileRegistryError(f"{family} object-class mismatch")
    if not bool(checkpoint.get("fuse_low_into_object_head")):
        raise ProfileRegistryError(f"{family} does not bind the required low-feature object fusion")
    if str(checkpoint.get("object_head_arch")) not in {"shared", "split_class_heatmaps"}:
        raise ProfileRegistryError(f"{family} object-head architecture mismatch")
    if int(checkpoint.get("object_head_depth", -1)) != 3:
        raise ProfileRegistryError(f"{family} object-head depth mismatch")
    if not bool(checkpoint.get("object_predict_bbox2d")):
        raise ProfileRegistryError(f"{family} does not expose the registered 2-D box head")

    model = build_multitask_fusion_lraspp(
        num_classes=len(service["segmentation_classes"]),
        radar_channels=int(checkpoint["radar_channels"]),
        pretrained=False,
        object_channels=int(checkpoint["object_channels"]),
        object_hidden_channels=128,
        fuse_low_into_object_head=bool(checkpoint["fuse_low_into_object_head"]),
        head_arch=str(checkpoint["object_head_arch"]),
        use_coordconv=bool(checkpoint.get("object_use_coordconv", False)),
        head_depth=int(checkpoint["object_head_depth"]),
        predict_bbox2d=bool(checkpoint["object_predict_bbox2d"]),
        use_groundplane_prior=bool(checkpoint.get("object_use_groundplane_prior", False)),
        groundplane_params=dict(checkpoint.get("object_groundplane_params") or {}),
        device=torch.device("cpu"),
    )
    ae_arch = "none"
    if bottleneck > 0:
        ae_arch = str(trial.get("ae_arch", ""))
        if not ae_arch:
            raise ProfileRegistryError(f"{family} integrated AE architecture is missing")
        high_channels = int(model.classifier.cbr[0].in_channels)
        model.feature_ae = build_ae(ae_arch, high_channels, bottleneck)
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as exc:
        raise ProfileRegistryError(f"{family} strict checkpoint reconstruction failed: {exc}") from exc

    low_channels = int(model.classifier.low_classifier.weight.shape[1])
    high_channels = int(model.classifier.cbr[0].weight.shape[1])
    native_shapes = _require_mapping(service["native_feature_shapes"], "native_feature_shapes")
    if low_channels != int(native_shapes["low"][1]) or high_channels != int(native_shapes["high"][1]):
        raise ProfileRegistryError(f"{family} native feature-channel mismatch")
    state = checkpoint["model"]
    feature_ae_keys = [str(key) for key in state if str(key).startswith("feature_ae.")]
    if (bottleneck > 0) != bool(feature_ae_keys):
        raise ProfileRegistryError(f"{family} integrated-AE state presence mismatch")
    if bottleneck > 0:
        encoded_channels = int(state["feature_ae.encoder.4.weight"].shape[0])
        decoded_channels = int(state["feature_ae.decoder.4.weight"].shape[0])
        if encoded_channels != bottleneck or decoded_channels != high_channels:
            raise ProfileRegistryError(f"{family} integrated-AE channel binding mismatch")
    else:
        encoded_channels = high_channels
        decoded_channels = high_channels

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_trial = _require_mapping(summary.get("trial"), f"{family}.trial_summary.trial")
    if int(summary_trial.get("ae_bottleneck", 0) or 0) != bottleneck:
        raise ProfileRegistryError(f"{family} trial summary bottleneck mismatch")
    if list(summary_trial.get("input_size", [])) != expected_input:
        raise ProfileRegistryError(f"{family} trial summary input mismatch")

    result = {
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "trial_name": str(trial.get("name", "")),
        "epoch": int(checkpoint.get("epoch", -1)),
        "ae_arch": ae_arch,
        "native_low_channels": low_channels,
        "native_high_channels": high_channels,
        "wire_high_channels": encoded_channels,
        "edge_decoded_high_channels": decoded_channels,
        "state_key_count": len(state),
        "integrated_ae_state_key_count": len(feature_ae_keys),
        "strict_structure_status": "PASS",
    }
    del model
    del checkpoint
    return result


def _read_evidence_rows(
    config: Mapping[str, Any], root: Path
) -> tuple[list[dict[str, str]], Path]:
    source = _require_mapping(config["source_evidence"], "source_evidence")
    path = _pinned_path(root, str(source["path"]), str(source["sha256"]), "evidence pool")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(source["expected_rows"]):
        raise ProfileRegistryError(f"evidence pool has {len(rows)} rows, expected 72")
    if len({row.get("profile_id") for row in rows}) != len(rows):
        raise ProfileRegistryError("evidence pool profile IDs are not unique")
    if set(row.get("evidence_pool_version") for row in rows) != {
        str(source["evidence_pool_version"])
    }:
        raise ProfileRegistryError("evidence pool version mismatch")
    return rows, path


def _expected_profile_keys(config: Mapping[str, Any]) -> list[tuple[str, str, float]]:
    factors = _require_mapping(config["factor_contract"], "factor_contract")
    return [
        (str(family), str(quantizer), float(q))
        for family in factors["models"]
        for quantizer in factors["quantizers"]
        for q in factors["roi_drop_fractions"]
    ]


def _build_rows(
    config: Mapping[str, Any],
    root: Path,
    evidence_rows: Sequence[Mapping[str, str]],
    evidence_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    factors = _require_mapping(config["factor_contract"], "factor_contract")
    service = _require_mapping(config["service_contract"], "service_contract")
    decoder = _require_mapping(config["decoder_contract"], "decoder_contract")
    runtime = _require_mapping(config["runtime_contract"], "runtime_contract")
    models = _require_mapping(factors["models"], "factor_contract.models")
    quantizers = _require_mapping(factors["quantizers"], "factor_contract.quantizers")
    entropy_level = int(factors["entropy_level"])

    evidence_by_key: dict[tuple[str, str, float], Mapping[str, str]] = {}
    for row in evidence_rows:
        key = (
            str(row["model_family"]),
            str(row["quantization_mode"]),
            float(row["roi_drop_fraction"]),
        )
        if key in evidence_by_key:
            raise ProfileRegistryError(f"duplicate evidence factor row: {key}")
        evidence_by_key[key] = row
    expected_keys = _expected_profile_keys(config)
    if set(evidence_by_key) != set(expected_keys):
        missing = sorted(set(expected_keys) - set(evidence_by_key))
        extra = sorted(set(evidence_by_key) - set(expected_keys))
        raise ProfileRegistryError(f"evidence factor grid mismatch missing={missing} extra={extra}")

    checkpoint_metadata: dict[str, Any] = {}
    checkpoint_inputs: list[dict[str, Any]] = []
    for family, raw_spec in models.items():
        spec = _require_mapping(raw_spec, f"models.{family}")
        checkpoint_path = _pinned_path(
            root,
            str(spec["checkpoint_path"]),
            str(spec["checkpoint_sha256"]),
            f"{family} checkpoint",
        )
        summary_path = _pinned_path(
            root,
            str(spec["trial_summary_path"]),
            str(spec["trial_summary_sha256"]),
            f"{family} trial summary",
        )
        metadata = _strict_checkpoint_metadata(
            str(family), spec, service, checkpoint_path, summary_path
        )
        checkpoint_metadata[str(family)] = metadata
        checkpoint_inputs.extend(
            [
                {
                    "kind": "checkpoint",
                    "family": str(family),
                    "path": str(spec["checkpoint_path"]),
                    "sha256": str(spec["checkpoint_sha256"]),
                    "bytes": checkpoint_path.stat().st_size,
                },
                {
                    "kind": "trial_summary",
                    "family": str(family),
                    "path": str(spec["trial_summary_path"]),
                    "sha256": str(spec["trial_summary_sha256"]),
                    "bytes": summary_path.stat().st_size,
                },
            ]
        )

    native_low = list(service["native_feature_shapes"]["low"])
    native_high = list(service["native_feature_shapes"]["high"])
    output_rows: list[dict[str, Any]] = []
    for action_index, key in enumerate(expected_keys):
        family, quantizer, q = key
        evidence = evidence_by_key[key]
        spec = _require_mapping(models[family], f"models.{family}")
        metadata = checkpoint_metadata[family]
        expected_id = canonical_profile_id(
            family, quantizer, q, entropy_level, str(spec["checkpoint_sha256"])
        )
        if evidence["profile_id"] != expected_id:
            raise ProfileRegistryError(
                f"canonical profile ID mismatch: {evidence['profile_id']} != {expected_id}"
            )
        if evidence["checkpoint_sha256"] != str(spec["checkpoint_sha256"]):
            raise ProfileRegistryError(f"{expected_id} checkpoint evidence mismatch")
        if evidence["entropy_coder"] != factors["entropy_coder"] or int(
            evidence["entropy_level"]
        ) != entropy_level:
            raise ProfileRegistryError(f"{expected_id} entropy binding mismatch")

        high_wire = list(native_high)
        high_wire[1] = int(metadata["wire_high_channels"])
        shared_profile_args = [
            "--quantization-mode",
            quantizer,
            "--entropy-coder",
            str(factors["entropy_coder"]),
            "--zstd-level",
            str(entropy_level),
            "--roi-threshold",
            _format_q(q),
            "--chunk-bytes",
            str(runtime["chunk_bytes"]),
        ]
        front_profile_args = [
            "--fusion-checkpoint",
            str(spec["checkpoint_path"]),
            *shared_profile_args,
        ]
        container_checkpoint_path = (
            f"{str(runtime['container_repository_root']).rstrip('/')}"
            f"/{str(spec['checkpoint_path']).lstrip('/')}"
        )
        edge_profile_args = [
            "--fusion-checkpoint",
            container_checkpoint_path,
            *shared_profile_args,
            "--object-score-threshold",
            str(decoder["object_score_threshold"]),
            "--object-nms-radius-px",
            str(decoder["object_nms_radius_px"]),
            "--topk-objects",
            str(decoder["topk_objects"]),
            "--max-objects-drawn",
            str(decoder["max_objects_published"]),
        ]
        row: dict[str, Any] = {
            "registry_schema": REGISTRY_SCHEMA,
            "registry_id": config["registry_id"],
            "action_index": action_index,
            "profile_id": expected_id,
            "display_profile_id": evidence["display_profile_id"],
            "model_family": family,
            "ae_bottleneck_channels": int(spec["ae_bottleneck_channels"]),
            "ae_source": "none" if family == "noae" else "integrated_checkpoint",
            "ae_arch": metadata["ae_arch"],
            "external_ae_override_allowed": False,
            "checkpoint_path": spec["checkpoint_path"],
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "checkpoint_bytes": metadata["checkpoint_bytes"],
            "checkpoint_trial_name": metadata["trial_name"],
            "checkpoint_epoch": metadata["epoch"],
            "checkpoint_strict_structure_status": metadata["strict_structure_status"],
            "input_schema_id": service["aligned_input_schema_id"],
            "input_width": service["input_width"],
            "input_height": service["input_height"],
            "rgb_channels": 3,
            "radar_channels": service["radar_channels"],
            "feature_schema_id": service["feature_schema_id"],
            "feature_levels": "low|high",
            "expected_low_shape": "x".join(str(value) for value in native_low),
            "expected_high_native_shape": "x".join(str(value) for value in native_high),
            "expected_high_wire_shape": "x".join(str(value) for value in high_wire),
            "expected_high_after_edge_decode_shape": "x".join(
                str(value) for value in native_high
            ),
            "feature_shape_status": "REGISTERED_EXPECTED_PENDING_UE_A2_OBSERVATION",
            "quantization_mode": quantizer,
            "quantization_bits": int(quantizers[quantizer]),
            "roi_drop_fraction": _format_q(q),
            "roi_semantics": factors["roi_semantics"],
            "entropy_coder": factors["entropy_coder"],
            "entropy_level": entropy_level,
            "feature_wire_schema_id": runtime["feature_wire_schema_id"],
            "udp_chunk_bytes": runtime["chunk_bytes"],
            "udp_chunk_header_struct": runtime["udp_chunk_header_struct"],
            "object_classes": "|".join(str(value) for value in service["object_classes"]),
            "object_channels": service["object_channels"],
            "object_score_threshold": decoder["object_score_threshold"],
            "object_nms_radius_px": decoder["object_nms_radius_px"],
            "topk_objects": decoder["topk_objects"],
            "max_objects_published": decoder["max_objects_published"],
            "decoder_override_required": True,
            "map_output_schema_id": service["map_output_schema_id"],
            "runtime_path": runtime["active_runtime_path"],
            "runtime_sha256": runtime["active_runtime_sha256"],
            "edge_container_checkpoint_path": container_checkpoint_path,
            "edge_decoder_checkpoint_sha256": spec["checkpoint_sha256"],
            "edge_binding_mode": runtime["binding_mode"],
            "edge_decoder_binding_status": "STATIC_ARGUMENT_VECTOR_DECLARED_PENDING_UE_A2_WIRE_SMOKE",
            "edge_launcher_propagation_status": "PENDING_UE_A2_DECODER_OVERRIDE_INTEGRATION",
            "wire_profile_identity_present": False,
            "wire_mismatch_rejection_present": False,
            "quality_source_profile_id": evidence["profile_id"],
            "quality_source_sha256": sha256_file(evidence_path),
            "offline_payload_semantics": factors["offline_payload_semantics"],
            "quality_mask_applied": False,
            "static_binding_status": "VERIFIED",
            "wire_smoke_status": "PENDING_UE_A2",
            "technical_validity_status": "REGISTERED_PENDING_SMOKE",
            "technical_invalid_reason": "",
            "front_profile_launch_args_json": json.dumps(
                front_profile_args, separators=(",", ":")
            ),
            "edge_profile_launch_args_json": json.dumps(
                edge_profile_args, separators=(",", ":")
            ),
        }
        row["row_fingerprint_sha256"] = registry_row_fingerprint(row)
        output_rows.append(row)
    return output_rows, {
        "checkpoint_metadata": checkpoint_metadata,
        "checkpoint_inputs": checkpoint_inputs,
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in REGISTRY_FIELDS})
            count += 1
    return count


def validate_registry_bundle(output_dir: Path) -> dict[str, Any]:
    """Validate file seals, row fingerprints, and the terminal decision chain."""

    output_dir = Path(output_dir).expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ProfileRegistryError(f"missing registry manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ProfileRegistryError("invalid registry manifest schema")
    for output in manifest.get("outputs", []):
        path = output_dir / str(output["path"])
        if not path.is_file() or sha256_file(path) != str(output["sha256"]):
            raise ProfileRegistryError(f"registry output seal mismatch: {path.name}")
        if "rows" in output:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != int(output["rows"]):
                raise ProfileRegistryError(f"registry row-count mismatch: {path.name}")
            for row in rows:
                if row.get("row_fingerprint_sha256") != registry_row_fingerprint(row):
                    raise ProfileRegistryError(
                        f"registry row fingerprint mismatch: {row.get('profile_id', '<unknown>')}"
                    )

    terminal_path = output_dir / str(manifest["terminal_decision_path"])
    if not terminal_path.is_file():
        raise ProfileRegistryError(f"missing terminal decision: {terminal_path.name}")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal_payload = dict(terminal)
    manifest_back_reference = terminal_payload.pop("manifest_sha256", None)
    if manifest_back_reference != sha256_file(manifest_path):
        raise ProfileRegistryError("terminal manifest back-reference mismatch")
    if terminal_payload != manifest.get("terminal_decision_payload"):
        raise ProfileRegistryError("terminal decision differs from manifest-committed payload")
    payload_sha = hashlib.sha256(canonical_json_bytes(terminal_payload)).hexdigest()
    if payload_sha != manifest.get("terminal_decision_payload_sha256"):
        raise ProfileRegistryError("terminal decision payload seal mismatch")
    return manifest


def _report(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    families = sorted({str(row["model_family"]) for row in rows})
    return f"""# UE split-profile registry v1 — UE-A1

**Status:** STATIC BINDINGS VERIFIED; UE-A2 WIRE SMOKE REQUIRED

This registry contains all {len(rows)} measured actions across {len(families)} model
families, three per-channel quantizers, and six rank-drop fractions. No action
was filtered using perception quality, payload size, or a preferred knob.

## What UE-A1 proves

- every checkpoint and trial summary is present and hash-bound;
- every checkpoint reconstructs with an exact strict state-dictionary match;
- integrated AE weights come from the selected checkpoint; external AE
  override is forbidden;
- model input, expected low/high feature schema, quantizer, q semantics,
  zstd-3 codec, evidence-compatible decoder settings, and distinct host/edge
  checkpoint paths are explicit for every profile;
- the edge profile argument vector is declared, but the current OAI launcher
  does not yet propagate all decoder overrides; and
- all rows remain `REGISTERED_PENDING_SMOKE`.

## Known current-runtime gaps

1. The feature payload does not contain profile/checkpoint/schema/codec
   identity, so the edge cannot reject a mismatched launch. Fixed-profile
   characterization must resolve both sides from the same registry row.
2. Live decoder defaults differ from the retained evidence. Every launch must
   override them with score={config['decoder_contract']['object_score_threshold']},
   NMS={config['decoder_contract']['object_nms_radius_px']},
   top-k={config['decoder_contract']['topk_objects']}, and published-object
   cap={config['decoder_contract']['max_objects_published']}.
3. Expected feature shapes are statically registered but remain unobserved on
   the live wire until UE-A2.
4. The current OAI launcher still needs UE-A2 integration for the registered
   edge decoder overrides before any profile is technically valid.

No CARLA run, OAI run, model inference, or policy training was performed.
"""


def assemble(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path | None = None,
    *,
    now: str | None = None,
) -> Path:
    config_path = Path(config_path).expanduser().resolve()
    config, root = load_config(config_path)
    if output_dir is None:
        output_dir = _repo_path(root, str(config["output"]["directory"]))
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise ProfileRegistryError(f"refusing to overwrite existing registry: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent))
    )
    created_at = now or utc_now()
    try:
        runtime_audit = _validate_runtime(config, root)
        evidence_rows, evidence_path = _read_evidence_rows(config, root)
        rows, row_audit = _build_rows(config, root, evidence_rows, evidence_path)
        if len(rows) != 72 or any(row["quality_mask_applied"] for row in rows):
            raise ProfileRegistryError("registry must retain all 72 actions without quality masks")

        output = _require_mapping(config["output"], "output")
        registry_name = str(output["registry_csv"])
        report_name = str(output["report_md"])
        resolved_name = "resolved_config.json"
        registry_path = temporary / registry_name
        if _write_csv(registry_path, rows) != 72:
            raise ProfileRegistryError("registry CSV row count changed while writing")
        (temporary / report_name).write_text(_report(config, rows), encoding="utf-8")
        _write_json(temporary / resolved_name, config)

        source = _require_mapping(config["source_evidence"], "source_evidence")
        terminal_payload = {
            "schema": TERMINAL_SCHEMA,
            "registry_id": config["registry_id"],
            "created_at": created_at,
            "status": "STATIC_BINDINGS_VERIFIED_WIRE_SMOKE_REQUIRED",
            "profile_count": 72,
            "quality_mask_count": 0,
            "technical_validity_frozen": False,
            "next_checklist_item": "UE-A2",
        }
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "registry_id": config["registry_id"],
            "created_at": created_at,
            "status": "STATIC_BINDINGS_VERIFIED_WIRE_SMOKE_REQUIRED",
            "authority": dict(config["authority"]),
            "counts": {
                "profiles": 72,
                "static_bindings_verified": 72,
                "registered_pending_smoke": 72,
                "technically_valid": 0,
                "technically_invalid": 0,
                "quality_masked": 0,
            },
            "gates": {
                "exact_factor_grid": "PASS",
                "checkpoint_hashes": "PASS",
                "checkpoint_strict_structure": "PASS",
                "integrated_ae_binding": "PASS",
                "quantizer_codec_source_binding": "PASS",
                "profile_evidence_settings_binding": "PASS",
                "offline_payload_semantics_binding": "PASS",
                "evidence_decoder_override_binding": "PASS",
                "no_quality_filter": "PASS",
                "fixed_profile_front_argument_binding": "PASS",
                "fixed_profile_edge_argument_declaration": "PASS",
                "current_edge_launcher_profile_override_propagation": "PENDING_UE_A2",
                "wire_profile_identity": "PENDING_UE_A2_GAP_RECORDED",
                "wire_smoke": "PENDING_UE_A2",
            },
            "known_gaps": [
                "WIRE_PROFILE_IDENTITY_ABSENT",
                "WIRE_MISMATCH_REJECTION_ABSENT",
                "LIVE_DECODER_DEFAULTS_REQUIRE_EXPLICIT_OVERRIDE",
                "EDGE_LAUNCHER_DECODER_OVERRIDES_NOT_PROPAGATED",
                "EXPECTED_FEATURE_SHAPES_UNOBSERVED_UNTIL_UE_A2",
            ],
            "terminal_decision_payload": terminal_payload,
            "terminal_decision_path": str(output["terminal_json"]),
            "terminal_decision_payload_sha256": hashlib.sha256(
                canonical_json_bytes(terminal_payload)
            ).hexdigest(),
            "runtime_audit": runtime_audit,
            "checkpoint_metadata": row_audit["checkpoint_metadata"],
            "inputs": [
                {
                    "kind": "config",
                    "path": str(config_path.relative_to(root)),
                    "sha256": sha256_file(config_path),
                    "bytes": config_path.stat().st_size,
                },
                {
                    "kind": "evidence_pool",
                    "path": str(source["path"]),
                    "sha256": str(source["sha256"]),
                    "bytes": evidence_path.stat().st_size,
                    "rows": 72,
                },
                *row_audit["checkpoint_inputs"],
                *runtime_audit["pinned_sources"],
            ],
            "repository": {
                "assembler_path": str(Path(__file__).resolve().relative_to(root)),
                "assembler_sha256": sha256_file(Path(__file__).resolve()),
            },
            "outputs": [
                {
                    "path": registry_name,
                    "sha256": sha256_file(registry_path),
                    "bytes": registry_path.stat().st_size,
                    "rows": 72,
                },
                {
                    "path": report_name,
                    "sha256": sha256_file(temporary / report_name),
                    "bytes": (temporary / report_name).stat().st_size,
                },
                {
                    "path": resolved_name,
                    "sha256": sha256_file(temporary / resolved_name),
                    "bytes": (temporary / resolved_name).stat().st_size,
                },
            ],
        }
        manifest_name = str(output["manifest_json"])
        _write_json(temporary / manifest_name, manifest)
        terminal = {
            **terminal_payload,
            "manifest_sha256": sha256_file(temporary / manifest_name),
        }
        _write_json(temporary / str(output["terminal_json"]), terminal)
        validate_registry_bundle(temporary)
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the static, unfiltered 72-action UE split-profile registry."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-config", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_config:
        config, _root = load_config(args.config)
        print(
            json.dumps(
                {
                    "status": "VALID_CONFIG_STATIC_ONLY",
                    "registry_id": config["registry_id"],
                    "profile_count": config["factor_contract"]["expected_profiles"],
                },
                sort_keys=True,
            )
        )
        return 0
    output = assemble(args.config, args.output_dir)
    print(f"UE-A1 static registry: {output}")
    print("Status: STATIC_BINDINGS_VERIFIED_WIRE_SMOKE_REQUIRED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
