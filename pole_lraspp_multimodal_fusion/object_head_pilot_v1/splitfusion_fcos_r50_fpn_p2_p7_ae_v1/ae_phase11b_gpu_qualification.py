"""One-frame GPU qualification of the shared UINT6/UINT4 transport.

This is deliberately a bounded execution check, not a validation or accuracy
run.  It loads one registered fit-training frame, computes frozen FP32 C2 once,
and exercises the public low-bit encode/receive paths for all 16 registered
family/quantizer/q combinations.  No validation or test dataset is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    contract,
    continuous_q,
    guards,
    teacher_cache,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import (
    build_train_dataset,
    collate_batch,
    encode_front,
    load_frozen_perception,
    sha256_file,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.ranker import build_ranker
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.zstd_transport import (
    implementation_report,
)
from . import ae_contract, ae_phase10_common as phase10, ae_training_common as common
from . import lowbit_dispatch, lowbit_transport
from .ae_gpu_qualification import CountingWireCodec, require_tree_finite, shape_signature
from .ae_model import SplitFeatureAE, build_split_feature_ae


EXECUTE_TOKEN = "SPLITFUSION_LOWBIT_PHASE11B_GPU_QUALIFICATION"
TERMINAL = "SPLITFUSION_LOWBIT_PHASE11B_GPU_QUALIFIED"
SCHEMA = "splitfusion_fcos_phase11b_lowbit_gpu_qualification_v1"

OUTPUT_RELPATH = (
    "experiments/splitfusion_fcos_ae_v1/"
    "20260903_phase11b_lowbit_gpu_qualification"
)
TARGET_SAMPLE_ID = "extra_v3_13_train_30_30_s805_tm1805_000761_frame3271"
Q_VALUES = (0.00, 0.50)
BIT_WIDTHS = (6, 4)


# These are deliberately restated here.  The runner verifies every byte before
# torch.load, rather than trusting an experiment-directory name or a manifest.
FROZEN_INPUTS: dict[str, dict[str, str]] = {
    "perception": {
        "path": (
            "experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_v1_"
            "numerical_recovery_v1/20260830_recovered_epoch10_gate_v1/"
            "checkpoints/epoch_026.pt"
        ),
        "sha256": "da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f",
    },
    "ranker": {
        "path": (
            "experiments/splitfusion_fcos_hybrid_q_v1/"
            "20260901_185725_phase5_ranker_training/checkpoints/ranker_epoch_04.pt"
        ),
        "sha256": "07781c56a4c0f306f16d332f64627ce6b9458e154f40ab9fef89f89909b79cb5",
    },
    "AE128": {
        "path": (
            "experiments/splitfusion_fcos_ae_v1/"
            "20260902_220623_phase9c_ae128_training/checkpoints/ae128_epoch_08.pt"
        ),
        "sha256": "0c2ba3a495684c0f8222492f554eb3de7c7a76181e0bd4b4a83529897db30f72",
        "selection_path": (
            "experiments/splitfusion_fcos_ae_v1/"
            "20260902_220623_phase9c_ae128_training/holdout_selection/"
            "holdout_selection.json"
        ),
        "selection_sha256": "69e49deac302fc46c1eec56036e3ab3d769b3aac10b76541cfb4abb80f878194",
    },
    "AE64": {
        "path": (
            "experiments/splitfusion_fcos_ae_v1/"
            "20260903_phase10_ae64_training/checkpoints/ae64_epoch_12.pt"
        ),
        "sha256": "dd7c5124e27114584ab2083e59160a3ff2a2d040d0a37d22564ac98c838aa8e0",
        "selection_path": (
            "experiments/splitfusion_fcos_ae_v1/"
            "20260903_phase10_ae64_training/holdout_selection_ae64/"
            "ae64_holdout_selection.json"
        ),
        "selection_sha256": "0d2fe444574d3fdc9aee287448084bf2cfc1efa2d0ec6944ac07355d9ff7c87e",
    },
    "AE32": {
        "path": (
            "experiments/splitfusion_fcos_ae_v1/"
            "20260903_phase10_ae32_training/checkpoints/ae32_epoch_08.pt"
        ),
        "sha256": "e2f867757e8db0620316c092264ac7eb53d12bb5ef66ed14475eb40693d1f271",
        "selection_path": (
            "experiments/splitfusion_fcos_ae_v1/"
            "20260903_phase10_ae32_training/holdout_selection_ae32/"
            "ae32_holdout_selection.json"
        ),
        "selection_sha256": "e3dfbfb736bb8847ad11d92b1573f88058e1c4319ac4a0180284db2171afac34",
    },
}

FAMILIES = (
    ("noAE", ae_contract.AE_FAMILY_NOAE, None),
    ("AE128", ae_contract.AE_FAMILY_AE128, 128),
    ("AE64", ae_contract.AE_FAMILY_AE64, 64),
    ("AE32", ae_contract.AE_FAMILY_AE32, 32),
)

# A selected checkpoint's package map predates Phase 11.  Historic files are
# immutable.  The five additions below are the complete, explicit delta; this
# local comparison intentionally does not relax any existing binding helper.
PHASE11_ADDED_SOURCES = frozenset(
    {
        "AE_PHASE11A_UINT6_UINT4_IMPLEMENTATION_REPORT.md",
        "lowbit_dispatch.py",
        "lowbit_transport.py",
        "tests/test_lowbit_transport.py",
        "ae_phase11b_gpu_qualification.py",
    }
)


class _CountingRanker:
    """Count calls and prove the public paths score the original C2 object."""

    def __init__(self, ranker: torch.nn.Module) -> None:
        self._ranker = ranker
        self.invocations = 0
        self.scored_pointers: list[int] = []

    def score_cells(self, c2: torch.Tensor) -> torch.Tensor:
        self.invocations += 1
        self.scored_pointers.append(int(c2.data_ptr()))
        return self._ranker.score_cells(c2)


class _EncodeCapture:
    """Observe the latent emitted by the real AE encode call without rerunning it."""

    def __init__(self, autoencoder: SplitFeatureAE) -> None:
        self._project: torch.Tensor | None = None
        self._context: torch.Tensor | None = None
        self._project_calls = 0
        self._context_calls = 0
        self._project_handle = autoencoder.project.register_forward_hook(
            self._capture_project
        )
        self._context_handle = autoencoder.latent_context.register_forward_hook(
            self._capture_context
        )

    def _capture_project(self, _module, _inputs, output) -> None:
        self._project_calls += 1
        self._project = output.detach()

    def _capture_context(self, _module, _inputs, output) -> None:
        self._context_calls += 1
        self._context = output.detach()

    def reset(self) -> None:
        self._project = None
        self._context = None
        self._project_calls = 0
        self._context_calls = 0

    def latent(self) -> torch.Tensor:
        if (
            self._project is None
            or self._context is None
            or self._project_calls != 1
            or self._context_calls != 1
        ):
            raise guards.HybridQPayloadError(
                "the public AE encode path did not emit exactly one latent"
            )
        return (self._project + self._context).detach()

    def close(self) -> None:
        self._project_handle.remove()
        self._context_handle.remove()
        self.reset()


def _repository_path(relative: str) -> Path:
    return (contract.repository_root() / relative).resolve(strict=True)


def _verify_frozen_input_hashes() -> dict[str, Any]:
    """Hash every named artifact before any one of them is loaded."""
    observed: dict[str, Any] = {}
    for name, item in FROZEN_INPUTS.items():
        path = _repository_path(item["path"])
        digest = sha256_file(path)
        if digest != item["sha256"]:
            raise guards.HybridQConfigError(f"{name} frozen artifact sha256 drift")
        row: dict[str, Any] = {"path": item["path"], "sha256": digest}
        if "selection_path" in item:
            selection_path = _repository_path(item["selection_path"])
            selection_digest = sha256_file(selection_path)
            if selection_digest != item["selection_sha256"]:
                raise guards.HybridQConfigError(
                    f"{name} selection artifact sha256 drift"
                )
            row["selection_path"] = item["selection_path"]
            row["selection_sha256"] = selection_digest
        observed[name] = row
    return observed


def _live_ae_source_map() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def _bind_historical_source_map(
    family_name: str, recorded: Mapping[str, Any], live: Mapping[str, str]
) -> dict[str, Any]:
    """Require byte identity for history and exactly the authorized additions."""
    if not isinstance(recorded, Mapping) or not recorded:
        raise guards.HybridQConfigError(
            f"{family_name} checkpoint does not carry a historical source map"
        )
    historical = {str(name): str(digest) for name, digest in recorded.items()}
    changed = sorted(
        name for name in historical if name in live and live[name] != historical[name]
    )
    removed = sorted(set(historical) - set(live))
    added = sorted(set(live) - set(historical))
    if changed:
        raise guards.HybridQConfigError(
            f"{family_name} historical AE source changed: {changed}"
        )
    if removed:
        raise guards.HybridQConfigError(
            f"{family_name} historical AE source removed: {removed}"
        )
    if set(added) != PHASE11_ADDED_SOURCES:
        raise guards.HybridQConfigError(
            f"{family_name} source additions must be exactly "
            f"{sorted(PHASE11_ADDED_SOURCES)}, found {added}"
        )
    return {
        "historical_files_byte_identical": True,
        "historical_file_count": len(historical),
        "live_file_count": len(live),
        "changed": [],
        "removed": [],
        "allowed_new_files": sorted(PHASE11_ADDED_SOURCES),
        "added": [
            {"path": name, "sha256": live[name]} for name in sorted(PHASE11_ADDED_SOURCES)
        ],
    }


def _verify_selection_document(
    family_name: str, item: Mapping[str, str], bottleneck: int
) -> dict[str, Any]:
    """Bind each frozen selection record to the selected checkpoint it names."""
    path = _repository_path(item["selection_path"])
    document = json.loads(path.read_text(encoding="utf-8"))
    expected_epoch = {128: 8, 64: 12, 32: 8}[bottleneck]
    if int(document["selection"]["selected_epoch"]) != expected_epoch:
        raise guards.HybridQConfigError(f"{family_name} selection epoch drift")
    if bottleneck == 128:
        if document.get("schema") != common.AE_HOLDOUT_SCHEMA:
            raise guards.HybridQConfigError("AE128 selection schema drift")
        if document.get("terminal") != "SPLITFUSION_AE128_HOLDOUT_CHECKPOINT_SELECTED":
            raise guards.HybridQConfigError("AE128 selection terminal drift")
        expected_name = "ae128_epoch_08.pt"
    else:
        if document.get("schema") != phase10.holdout_schema(bottleneck):
            raise guards.HybridQConfigError(f"{family_name} selection schema drift")
        if document.get("terminal") != phase10.holdout_terminal(bottleneck):
            raise guards.HybridQConfigError(f"{family_name} selection terminal drift")
        expected_name = phase10.candidate_filename(bottleneck, expected_epoch)
    candidates = dict(document["training_run"]["candidate_checkpoints"])
    if candidates.get(expected_name) != item["sha256"]:
        raise guards.HybridQConfigError(
            f"{family_name} selection does not bind its requested checkpoint"
        )
    scope = document["scope"]
    if bool(scope["validation_or_test_accessed"]):
        raise guards.HybridQConfigError(
            f"{family_name} selection record reports validation/test access"
        )
    return {
        "schema": str(document["schema"]),
        "terminal": str(document["terminal"]),
        "selected_epoch": expected_epoch,
        "selected_checkpoint": expected_name,
        "selected_checkpoint_sha256": item["sha256"],
        "selection_reports_validation_or_test_access": False,
    }


def _load_selected_autoencoder(
    family_name: str,
    bottleneck: int,
    item: Mapping[str, str],
    payload: Mapping[str, Any],
    device: torch.device,
) -> SplitFeatureAE:
    """Build the registered family, load the hash-bound state, then freeze it."""
    expected_family = ae_contract.family_for_bottleneck(bottleneck)
    expected_epoch = {128: 8, 64: 12, 32: 8}[bottleneck]
    if int(payload["epoch"]) != expected_epoch:
        raise guards.HybridQConfigError(f"{family_name} checkpoint epoch drift")
    if int(payload["bottleneck"]) != bottleneck:
        raise guards.HybridQConfigError(f"{family_name} checkpoint bottleneck drift")
    if int(payload["family_id"]) != expected_family:
        raise guards.HybridQConfigError(f"{family_name} checkpoint family drift")
    expected_configuration = (
        common.training_configuration()
        if bottleneck == 128
        else phase10.training_configuration(bottleneck)
    )
    if payload["configuration"] != expected_configuration:
        raise guards.HybridQConfigError(
            f"{family_name} checkpoint locked configuration drift"
        )
    autoencoder = build_split_feature_ae(bottleneck)
    autoencoder.load_state_dict(payload["autoencoder"], strict=True)
    if autoencoder.parameter_count() != int(payload["parameter_count"]):
        raise guards.HybridQConfigError(f"{family_name} parameter-count drift")
    autoencoder = autoencoder.to(device)
    common.freeze(autoencoder)
    guards.require_module_parameters_finite(autoencoder, f"selected {family_name}")
    autoencoder.bind_routing_tag(ae_contract.routing_tag_from_sha256(item["sha256"]))
    if not autoencoder.is_bound:
        raise guards.HybridQConfigError(f"{family_name} routing tag is unbound")
    return autoencoder


def _load_ranker(device: torch.device) -> torch.nn.Module:
    payload = torch.load(_repository_path(FROZEN_INPUTS["ranker"]["path"]), map_location="cpu", weights_only=False)
    if int(payload["epoch"]) != contract.VALIDATION_RANKER_EPOCH:
        raise guards.HybridQConfigError("stable ranker epoch drift")
    if int(payload["parameter_count"]) != contract.RANKER_PARAMETER_COUNT:
        raise guards.HybridQConfigError("stable ranker parameter-count drift")
    ranker = build_ranker()
    ranker.load_state_dict(payload["ranker"], strict=True)
    del payload
    ranker = ranker.to(device)
    common.freeze(ranker)
    return ranker


def _state_record(module: torch.nn.Module) -> dict[str, Any]:
    from .ae_gpu_qualification import state_hashes

    per_tensor, aggregate = state_hashes(module)
    return {
        "aggregate_sha256": aggregate,
        "tensor_count": len(per_tensor),
    }


def _select_fit_frame(base: Any) -> tuple[Any, int, dict[str, Any]]:
    """Admit exactly the named row and prove it belongs to the fit partition."""
    dataset = build_train_dataset(base)
    matches = [
        index
        for index, row in enumerate(dataset.rows)
        if str(row["sample_id"]) == TARGET_SAMPLE_ID
    ]
    if len(matches) != 1:
        raise guards.HybridQConfigError(
            f"expected one registered fit frame {TARGET_SAMPLE_ID}, found {len(matches)}"
        )
    partition = teacher_cache.build_split_partition(dataset)
    index = matches[0]
    if index not in set(int(value) for value in partition.fit_indices):
        raise guards.HybridQOwnershipError(
            f"{TARGET_SAMPLE_ID} is not in the registered fit split"
        )
    if TARGET_SAMPLE_ID not in set(str(value) for value in partition.fit_sample_ids):
        raise guards.HybridQOwnershipError(
            f"{TARGET_SAMPLE_ID} is absent from registered fit sample IDs"
        )
    if TARGET_SAMPLE_ID in set(str(value) for value in partition.holdout_sample_ids):
        raise guards.HybridQOwnershipError(
            f"{TARGET_SAMPLE_ID} is also registered as a train-holdout frame"
        )
    row = dict(dataset.rows[index])
    return dataset, index, {
        "sample_id": TARGET_SAMPLE_ID,
        "dataset_index": index,
        "registered_split": "fit",
        "fit_frames_registered": len(partition.fit_indices),
        "train_holdout_frames_registered_but_unread": len(partition.holdout_indices),
        "row_keys": sorted(str(name) for name in row),
    }


def _digest_tensor(tensor: torch.Tensor) -> str:
    raw = tensor.detach().to(device="cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _retained_affine_error(
    source: torch.Tensor, diagnostics: lowbit_dispatch.LowBitReceiveDiagnostics
) -> dict[str, float | bool]:
    """Check the quantizer's declared per-channel half-step error bound."""
    parsed = diagnostics.parsed
    channels = int(parsed.channels)
    source_flat = source.detach().to(device="cpu", dtype=torch.float32).reshape(
        channels, contract.SPLIT_CELLS
    )
    decoded_flat = diagnostics.decoded_feature.detach().to(
        device="cpu", dtype=torch.float32
    ).reshape(channels, contract.SPLIT_CELLS)
    indices = parsed.keep_indices.to(device="cpu", dtype=torch.int64)
    source_retained = source_flat.index_select(1, indices)
    decoded_retained = decoded_flat.index_select(1, indices)
    ranges = parsed.channel_ranges.to(device="cpu", dtype=torch.float32)
    spans = ranges[:, 1] - ranges[:, 0]
    magnitude = torch.maximum(ranges[:, 0].abs(), ranges[:, 1].abs()).clamp_min(1.0)
    allowance = 8.0 * torch.finfo(torch.float32).eps * magnitude
    bound = spans / (2.0 * ((1 << parsed.bit_width) - 1)) + allowance
    per_channel_error = (decoded_retained - source_retained).abs().amax(dim=1)
    if bool((per_channel_error > bound).any()):
        raise guards.HybridQNumericalError(
            "retained low-bit values exceed the affine quantization error bound"
        )
    return {
        "satisfies_per_channel_affine_bound": True,
        "maximum_error": float(per_channel_error.max()),
        "maximum_bound": float(bound.max()),
        "fp32_numerical_allowance": "8 * eps(float32) * max(abs(min), abs(max), 1)",
    }


def _require_header(
    parsed: lowbit_transport.InspectedLowBitPayload,
    *,
    plan: continuous_q.ContinuousQ,
    family_id: int,
    routing_tag: int,
    bit_width: int,
) -> lowbit_transport.LowBitAnalyticalSize:
    expected = lowbit_transport.analytical_size(plan.wire_q, family_id, bit_width)
    header = parsed.header
    observed = (
        header.bit_width,
        header.family_id,
        header.routing_tag,
        header.channels,
        header.height,
        header.width,
        header.q_e4,
        header.keep_count,
        header.mask_bytes,
        header.range_bytes,
        header.value_bytes,
    )
    wanted = (
        bit_width,
        family_id,
        routing_tag,
        expected.channels,
        contract.SPLIT_HEIGHT,
        contract.SPLIT_WIDTH,
        plan.q_e4,
        expected.keep_count,
        expected.mask_bytes,
        expected.range_bytes,
        expected.value_bytes,
    )
    if observed != wanted:
        raise guards.HybridQPayloadError(
            f"low-bit header drift: observed {observed}, expected {wanted}"
        )
    return expected


def _tail_structure(
    outputs: Mapping[str, Any], expected: Mapping[str, list[int]] | None
) -> tuple[dict[str, list[int]], int]:
    required_top_level = {
        "c2",
        "features",
        "resnet_features",
        "detection",
        "geometry",
        "semantic_logits_stride4",
        "semantic_logits",
        "anchors",
        "dense_depth_log1p_stride4",
        "dense_depth_log1p",
    }
    if set(outputs) != required_top_level:
        raise guards.HybridQPayloadError("frozen tail output key structure drift")
    signature = shape_signature(outputs)
    finite_tensors = require_tree_finite(outputs)
    if finite_tensors <= 0:
        raise guards.HybridQPayloadError("frozen tail produced no tensor outputs")
    if expected is not None and signature != expected:
        raise guards.HybridQPayloadError("frozen tail tensor structure drift")
    return signature, finite_tensors


def _qualify_setting(
    *,
    model: torch.nn.Module,
    c2: torch.Tensor,
    ranker: _CountingRanker,
    decoders: lowbit_dispatch.PreloadedLowBitDecoders,
    autoencoders: Mapping[str, SplitFeatureAE],
    captures: Mapping[str, _EncodeCapture],
    family_name: str,
    family_id: int,
    bottleneck: int | None,
    bit_width: int,
    q: float,
    wire: CountingWireCodec,
    expected_tail_signature: Mapping[str, list[int]] | None,
) -> tuple[
    dict[str, Any],
    dict[str, list[int]],
    tuple[torch.Tensor, torch.Tensor] | None,
]:
    """Exercise one public encode/receive/tail path and audit its wire facts."""
    plan = continuous_q.quantize_q(q)
    expected_ranker_calls = 0 if plan.is_bypass else 1
    ranker_before = ranker.invocations
    routing_tag = ae_contract.AE_UNBOUND_ROUTING_TAG
    source: torch.Tensor
    transport: lowbit_transport.LowBitTransport

    if family_id == ae_contract.AE_FAMILY_NOAE:
        source = c2
        transport = lowbit_transport.encode_noae_frame(
            c2, ranker, plan.wire_q, bit_width, wire_codec=wire
        )
    else:
        if bottleneck is None:
            raise guards.HybridQConfigError("AE family has no bottleneck")
        autoencoder = autoencoders[family_name]
        capture = captures[family_name]
        capture.reset()
        routing_tag = int(autoencoder.routing_tag)
        transport = lowbit_transport.encode_ae_frame(
            c2, autoencoder, ranker, plan.wire_q, bit_width, wire_codec=wire
        )
        source = capture.latent()

    ranker_calls = ranker.invocations - ranker_before
    if ranker_calls != expected_ranker_calls:
        raise guards.HybridQPayloadError(
            f"{family_name} UINT{bit_width} q={plan.wire_q} invoked the ranker "
            f"{ranker_calls} times, expected {expected_ranker_calls}"
        )
    if ranker_calls and any(
        pointer != int(c2.data_ptr()) for pointer in ranker.scored_pointers[-ranker_calls:]
    ):
        raise guards.HybridQPayloadError("ranker did not receive original FP32 C2")
    if plan.is_bypass:
        if transport.selection is not None or int(transport.keep_mask.sum()) != contract.SPLIT_CELLS:
            raise guards.HybridQPayloadError("q=0 did not use the dense ranker bypass")
    elif transport.selection is None:
        raise guards.HybridQPayloadError("q=0.50 omitted its sparse selection")

    if transport.plan.q_e4 != plan.q_e4:
        raise guards.HybridQPayloadError("requested q differs from transmitted wire q")
    if int(transport.family_id) != family_id or int(transport.bit_width) != bit_width:
        raise guards.HybridQPayloadError("transmitted low-bit family or width drift")
    if transport.packet.compressed_bytes <= 0:
        raise guards.HybridQPayloadError("low-bit transport emitted an empty zstd frame")

    # Raw compressed bytes only: do not supply the local packet metadata to the
    # receiver, so header parsing is its sole decoder-selection authority.
    wire.decompressions = 0
    received = decoders.receive(transport.packet.data, wire_codec=wire, diagnostics=True)
    if wire.decompressions != 1:
        raise guards.HybridQPayloadError(
            f"{family_name} UINT{bit_width} receiver decompressed {wire.decompressions} times"
        )
    diagnostics = received.diagnostics
    if diagnostics is None:
        raise guards.HybridQPayloadError("low-bit receiver did not return diagnostics")
    parsed = diagnostics.parsed
    analytical = _require_header(
        parsed,
        plan=plan,
        family_id=family_id,
        routing_tag=routing_tag,
        bit_width=bit_width,
    )
    if received.q != plan.wire_q or parsed.q != plan.wire_q:
        raise guards.HybridQPayloadError("received q differs from requested wire q")
    if transport.packet.uncompressed_bytes != analytical.total_bytes:
        raise guards.HybridQPayloadError("measured pre-zstd bytes differ from analytical size")
    if received.uncompressed_bytes != analytical.total_bytes:
        raise guards.HybridQPayloadError("received raw payload size differs from analytical size")
    if int(received.keep_count) != plan.keep_count:
        raise guards.HybridQPayloadError("received keep count drift")
    if int(diagnostics.keep_mask.sum()) != plan.keep_count:
        raise guards.HybridQPayloadError("received keep-mask cardinality drift")

    if plan.is_bypass:
        if parsed.header.mask_bytes != 0 or transport.selection is not None:
            raise guards.HybridQPayloadError("q=0 unexpectedly carried a bitmask")
    else:
        if parsed.header.mask_bytes != contract.mask_byte_count():
            raise guards.HybridQPayloadError("q=0.50 did not carry the 2,688-byte mask")
        if int(received.keep_count) != 10752:
            raise guards.HybridQPayloadError("q=0.50 did not retain exactly 10,752 cells")

    if family_id == ae_contract.AE_FAMILY_NOAE:
        if diagnostics.decoder is not None or received.c2 is not diagnostics.decoded_feature:
            raise guards.HybridQPayloadError("noAE frame did not route directly to C2")
    else:
        autoencoder = autoencoders[family_name]
        if diagnostics.decoder is not autoencoder:
            raise guards.HybridQPayloadError("decoder was not selected from received header")
        if received.family.routing_tag != autoencoder.routing_tag:
            raise guards.HybridQPayloadError("received AE routing tag drift")

    affine = _retained_affine_error(source, diagnostics)
    dropped = ~diagnostics.keep_mask.reshape(-1)
    decoded_flat = diagnostics.decoded_feature.reshape(int(parsed.channels), -1)
    if bool((decoded_flat[:, dropped] != 0.0).any()):
        raise guards.HybridQNumericalError(
            "a dropped cell was not exact zero before the decoder or frozen tail"
        )

    reconstructed = received.c2
    guards.require_frozen_c2(reconstructed, what="reconstructed low-bit C2")
    if reconstructed.device != c2.device or reconstructed.device.type != "cuda" or reconstructed.device.index != 0:
        raise guards.HybridQPayloadError("reconstructed C2 is not on cuda:0")
    outputs = model.decode_tail(reconstructed.unsqueeze(0), dense=True)
    signature, tail_tensors = _tail_structure(outputs, expected_tail_signature)

    selection_digest: dict[str, str] | None = None
    selection_evidence: tuple[torch.Tensor, torch.Tensor] | None = None
    if not plan.is_bypass:
        selection_digest = {
            "mask_sha256": _digest_tensor(diagnostics.keep_mask.to(torch.uint8)),
            "keep_indices_sha256": _digest_tensor(parsed.keep_indices.to(torch.int64)),
        }
        selection_evidence = (
            diagnostics.keep_mask.detach().to(device="cpu").clone(),
            parsed.keep_indices.detach().to(device="cpu", dtype=torch.int64).clone(),
        )

    row = {
        "family": family_name,
        "family_id": family_id,
        "quantizer": f"UINT{bit_width}",
        "bit_width": bit_width,
        "q_requested": q,
        "q_wire": received.q,
        "q_e4": plan.q_e4,
        "keep_count": int(received.keep_count),
        "ranker_invocations": ranker_calls,
        "zstd_decompressions": wire.decompressions,
        "header": {
            "bit_width": int(parsed.header.bit_width),
            "family_id": int(parsed.header.family_id),
            "transported_channels": int(parsed.header.channels),
            "routing_tag": int(parsed.header.routing_tag),
            "q_e4": int(parsed.header.q_e4),
            "keep_count": int(parsed.header.keep_count),
            "mask_bytes": int(parsed.header.mask_bytes),
            "range_bytes": int(parsed.header.range_bytes),
            "packed_value_bytes": int(parsed.header.value_bytes),
        },
        "analytical_pre_zstd_bytes": int(analytical.total_bytes),
        "measured_pre_zstd_bytes": int(transport.packet.uncompressed_bytes),
        "compressed_bytes_diagnostic_only": int(transport.packet.compressed_bytes),
        "decoder_selected_from_received_inner_header": True,
        "noae_direct_to_c2_without_ae_decoder": family_id == ae_contract.AE_FAMILY_NOAE,
        "matching_preloaded_ae_decoder": family_id != ae_contract.AE_FAMILY_NOAE,
        "dropped_cells_exact_zero_before_decoder_or_tail": True,
        "reconstructed_c2": {
            "shape": list(reconstructed.shape),
            "dtype": str(reconstructed.dtype).replace("torch.", ""),
            "device": str(reconstructed.device),
            "finite": True,
        },
        "retained_affine_error": affine,
        "q50_selection_digest": selection_digest,
        "tail_output_tensors": tail_tensors,
        "tail_outputs_finite": True,
    }
    del outputs, reconstructed, decoded_flat, source, received, diagnostics, transport
    return row, signature, selection_evidence


def _atomic_write(path: Path, data: str) -> str:
    staging = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with staging.open("w", encoding="utf-8", newline="") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if staging.exists():
            staging.unlink()
    return sha256_file(path)


def _atomic_json(path: Path, document: Mapping[str, Any]) -> str:
    return _atomic_write(path, json.dumps(document, indent=2, sort_keys=True) + "\n")


def _report_text(document: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 11B — shared UINT6/UINT4 GPU qualification",
        "",
        f"Terminal: `{TERMINAL}`",
        "",
        "One registered fit-training frame was run through the public low-bit "
        "encode/receive paths. This is a structural GPU qualification only: no "
        "validation, test, accuracy, scoring, calibration, NMS, training, tuning "
        "or CARLA activity occurred.",
        "",
        "| family | quantizer | q | keep | pre-zstd analytical/measured B | zstd B (diagnostic) | ranker | decomp | tail finite |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in document["settings"]:
        lines.append(
            f"| {row['family']} | {row['quantizer']} | {row['q_wire']:.2f} | "
            f"{row['keep_count']:,} | {row['analytical_pre_zstd_bytes']:,}/"
            f"{row['measured_pre_zstd_bytes']:,} | "
            f"{row['compressed_bytes_diagnostic_only']:,} | "
            f"{row['ranker_invocations']} | {row['zstd_decompressions']} | yes |"
        )
    integrity = document["integrity"]
    lines += [
        "",
        "## Integrity",
        "",
        f"- settings: {len(document['settings'])}/16",
        f"- ranker invocations: {integrity['ranker_invocations']} (required 8)",
        f"- zstd decompressions: {integrity['zstd_decompressions']} (required 16)",
        f"- q=0 ranker bypass: {integrity['q0_ranker_bypassed']}",
        f"- q=0.50 masks/indices identical: {integrity['q50_masks_and_indices_identical']}",
        f"- all frozen states unchanged: {integrity['all_frozen_states_unchanged']}",
        f"- peak allocated/reserved VRAM: {document['resources']['peak_allocated_bytes']:,}/"
        f"{document['resources']['peak_reserved_bytes']:,} bytes",
        f"- wall time: {document['resources']['wall_seconds']:.3f} s",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase-11B one-frame UINT6/UINT4 GPU qualification"
    )
    parser.add_argument("--execute", required=True, choices=(EXECUTE_TOKEN,))
    args = parser.parse_args()
    del args

    output = contract.repository_root() / OUTPUT_RELPATH
    if output.exists():
        raise guards.HybridQConfigError(f"create-only output already exists: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 11B requires CUDA on cuda:0")
    device = torch.device("cuda:0")
    started = time.perf_counter()

    frozen_hashes = _verify_frozen_input_hashes()
    live_sources = _live_ae_source_map()
    checkpoint_payloads: dict[str, Mapping[str, Any]] = {}
    source_bindings: dict[str, Any] = {}
    selection_bindings: dict[str, Any] = {}
    for family_name, _family_id, bottleneck in FAMILIES:
        if bottleneck is None:
            continue
        item = FROZEN_INPUTS[family_name]
        payload = torch.load(_repository_path(item["path"]), map_location="cpu", weights_only=False)
        checkpoint_payloads[family_name] = payload
        source_bindings[family_name] = _bind_historical_source_map(
            family_name, payload.get("ae_package_source_sha256", {}), live_sources
        )
        selection_bindings[family_name] = _verify_selection_document(
            family_name, item, bottleneck
        )

    model, base, perception_binding = load_frozen_perception(device)
    common.freeze(model)
    ranker_model = _load_ranker(device)
    autoencoders = {
        family_name: _load_selected_autoencoder(
            family_name,
            bottleneck,
            FROZEN_INPUTS[family_name],
            checkpoint_payloads[family_name],
            device,
        )
        for family_name, _family_id, bottleneck in FAMILIES
        if bottleneck is not None
    }
    checkpoint_payloads.clear()
    guards.require_frozen_perception([model, ranker_model, *autoencoders.values()])
    guards.require_eval_mode([model, ranker_model, *autoencoders.values()])
    frozen_before = {
        "perception": _state_record(model),
        "ranker": _state_record(ranker_model),
        **{name: _state_record(autoencoder) for name, autoencoder in autoencoders.items()},
    }

    dataset, frame_index, fit_frame = _select_fit_frame(base)
    batch = collate_batch(base, dataset, [frame_index])
    if list(batch["sample_ids"]) != [TARGET_SAMPLE_ID]:
        raise guards.HybridQConfigError("fit-frame collation sample-id drift")

    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        c2_batch = encode_front(model, batch, device)
        if int(c2_batch.shape[0]) != 1:
            raise guards.HybridQPayloadError("qualification front emitted more than one C2")
        c2 = c2_batch[0].detach()
        del c2_batch, batch
        guards.require_frozen_c2(c2, what="single frozen qualification C2")
        if c2.device != device:
            raise guards.HybridQPayloadError("frozen qualification C2 is not on cuda:0")

        ranker = _CountingRanker(ranker_model)
        decoders = lowbit_dispatch.PreloadedLowBitDecoders(autoencoders.values())
        expected_decoder_families = tuple(
            ae_contract.family_for_bottleneck(size) for size in (128, 64, 32)
        )
        if decoders.families != expected_decoder_families:
            raise guards.HybridQConfigError("preloaded decoder family registry drift")
        captures = {name: _EncodeCapture(autoencoder) for name, autoencoder in autoencoders.items()}
        wire = CountingWireCodec()
        rows: list[dict[str, Any]] = []
        expected_tail_signature: dict[str, list[int]] | None = None
        expected_q50_indices: torch.Tensor | None = None
        expected_q50_mask: torch.Tensor | None = None
        q50_identical = True
        try:
            for family_name, family_id, bottleneck in FAMILIES:
                for bit_width in BIT_WIDTHS:
                    for q in Q_VALUES:
                        row, signature, selection_evidence = _qualify_setting(
                            model=model,
                            c2=c2,
                            ranker=ranker,
                            decoders=decoders,
                            autoencoders=autoencoders,
                            captures=captures,
                            family_name=family_name,
                            family_id=family_id,
                            bottleneck=bottleneck,
                            bit_width=bit_width,
                            q=q,
                            wire=wire,
                            expected_tail_signature=expected_tail_signature,
                        )
                        if expected_tail_signature is None:
                            expected_tail_signature = signature
                        if q == 0.50:
                            if selection_evidence is None:
                                raise guards.HybridQPayloadError("q=0.50 has no selection digest")
                            mask, indices = selection_evidence
                            if expected_q50_mask is None or expected_q50_indices is None:
                                expected_q50_mask = mask
                                expected_q50_indices = indices
                            elif not (
                                torch.equal(mask, expected_q50_mask)
                                and torch.equal(indices, expected_q50_indices)
                            ):
                                q50_identical = False
                            del mask, indices
                        rows.append(row)
        finally:
            for capture in captures.values():
                capture.close()

    del expected_q50_mask, expected_q50_indices, c2, dataset, base

    # The per-setting helper records the digest but deliberately drops tensors.
    # A digest collision is not an equality proof, so retrieve equality from the
    # independent q=0.50 ranking property encoded in the records: all public
    # selections must have the same exact byte digest *and* the same cardinality.
    q50_rows = [row for row in rows if row["q_e4"] == 5000]
    if len(q50_rows) != 8:
        raise guards.HybridQPayloadError("qualification did not produce eight q=0.50 settings")
    q50_masks = {row["q50_selection_digest"]["mask_sha256"] for row in q50_rows}
    q50_indices = {row["q50_selection_digest"]["keep_indices_sha256"] for row in q50_rows}
    q50_identical = q50_identical and len(q50_masks) == 1 and len(q50_indices) == 1
    if not q50_identical:
        raise guards.HybridQPayloadError("q=0.50 masks or indices differ across low-bit settings")

    expected_settings = len(FAMILIES) * len(BIT_WIDTHS) * len(Q_VALUES)
    if len(rows) != expected_settings:
        raise guards.HybridQPayloadError("qualification matrix cardinality drift")
    q0_rows = [row for row in rows if row["q_e4"] == 0]
    if len(q0_rows) != 8 or any(row["ranker_invocations"] != 0 for row in q0_rows):
        raise guards.HybridQPayloadError("a q=0 setting invoked the ranker")
    if any(row["keep_count"] != contract.SPLIT_CELLS for row in q0_rows):
        raise guards.HybridQPayloadError("a q=0 setting did not retain all cells")
    if ranker.invocations != 8:
        raise guards.HybridQPayloadError(
            f"ranker invocation count {ranker.invocations} != required 8"
        )
    if any(row["zstd_decompressions"] != 1 for row in rows):
        raise guards.HybridQPayloadError("a setting did not perform exactly one zstd decompression")
    if any(
        row["analytical_pre_zstd_bytes"] != row["measured_pre_zstd_bytes"]
        for row in rows
    ):
        raise guards.HybridQPayloadError("analytical and measured pre-zstd bytes diverged")

    frozen_after = {
        "perception": _state_record(model),
        "ranker": _state_record(ranker_model),
        **{name: _state_record(autoencoder) for name, autoencoder in autoencoders.items()},
    }
    frozen_equal = {
        name: frozen_before[name] == frozen_after[name] for name in frozen_before
    }
    if not all(frozen_equal.values()):
        raise guards.HybridQOwnershipError("a frozen model, ranker or AE state changed")
    if any(
        parameter.grad is not None
        for module in (model, ranker_model, *autoencoders.values())
        for parameter in module.parameters()
    ):
        raise guards.HybridQOwnershipError("a frozen module received a gradient")

    torch.cuda.synchronize(device)
    resources = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "wall_seconds": time.perf_counter() - started,
    }
    document = {
        "schema": SCHEMA,
        "terminal": TERMINAL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_inputs": frozen_hashes,
        "perception_binding": perception_binding,
        "historical_checkpoint_source_bindings": source_bindings,
        "selection_bindings": selection_bindings,
        "fit_frame": fit_frame,
        "scope": {
            "qualification_only": True,
            "frozen_fp32_c2_computed_exactly_once": True,
            "front_forwards": 1,
            "fit_training_frames_read": 1,
            "train_holdout_frames_read": 0,
            "validation_frames_read": 0,
            "test_frames_read": 0,
            "optimizer_created": False,
            "gradients_used": False,
            "training_or_tuning": False,
            "threshold_change": False,
            "scoring_used": False,
            "calibration_used": False,
            "nms_used": False,
            "carla_used": False,
            "warmup_loops": 0,
            "latency_distribution_claimed": False,
            "payload_blobs_retained": False,
            "reconstructed_c2_retained": False,
            "tail_outputs_retained": False,
        },
        "environment": {
            "python": platform.python_version(),
            "executable": "/usr/bin/python3",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        },
        "transport": {
            "public_encode_noae": "lowbit_transport.encode_noae_frame",
            "public_encode_ae": "lowbit_transport.encode_ae_frame",
            "public_receive": "PreloadedLowBitDecoders.receive",
            "zstd": implementation_report(),
            "q_values": list(Q_VALUES),
            "bit_widths": list(BIT_WIDTHS),
            "families": [name for name, _family, _size in FAMILIES],
            "tail_signature": expected_tail_signature,
        },
        "settings": rows,
        "integrity": {
            "settings_completed": len(rows),
            "settings_required": 16,
            "ranker_invocations": ranker.invocations,
            "required_ranker_invocations": 8,
            "q0_ranker_bypassed": True,
            "zstd_decompressions": sum(int(row["zstd_decompressions"]) for row in rows),
            "required_zstd_decompressions": 16,
            "q50_masks_and_indices_identical": q50_identical,
            "all_header_checks_passed": True,
            "all_analytical_sizes_match_measured_pre_zstd": True,
            "all_retained_affine_bounds_passed": True,
            "all_dropped_cells_exact_zero": True,
            "all_reconstructed_c2_finite_fp32_cuda0": True,
            "all_tail_outputs_finite_and_structurally_identical": True,
            "all_frozen_states_unchanged": all(frozen_equal.values()),
            "frozen_state_equal": frozen_equal,
        },
        "frozen_state_before": frozen_before,
        "frozen_state_after": frozen_after,
        "resources": resources,
    }

    output.mkdir(parents=True, exist_ok=False)
    report_hash = _atomic_json(output / "phase11b_lowbit_gpu_qualification.json", document)
    _atomic_write(output / "PHASE11B_LOWBIT_GPU_QUALIFICATION_REPORT.md", _report_text(document))
    _atomic_write(output / TERMINAL, f"{TERMINAL} {report_hash}\n")
    print(json.dumps({"output": str(output), "report_sha256": report_hash}))
    print(TERMINAL)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
