"""Phase 9D: deployment-path UINT8 + mandatory zstd validation of the selected AE128.

One measurement phase. It trains nothing, tunes nothing, changes no threshold,
no NMS, no calibration and no scorer, and it never opens test data. It measures
the six registered q anchors exactly once each on the registered 3,345
validation frames, through the **real deployment path** and nothing shorter:

    original FP32 C2
      -> selected AE128 encoder (always, on the complete frame)
      -> per-channel UINT8 latent quantization, ranges from the complete latent
      -> sparse AE wire
      -> mandatory zstd level 1
      -> received raw bytes
      -> exactly one decompression
      -> header-driven preloaded AE128 decoder selection
      -> dequantization / zero scatter
      -> AE128 decoder
      -> unchanged frozen perception tail

q=0 bypasses the ranker but never bypasses AE128, so even the q=0 row is a
lossy channel reconstruction rather than an identity. At q>0 the ranker scores
the *original FP32 C2*, independently per frame, before any cell is dropped, and
the per-channel ranges are computed from the complete latent before dropping, so
a retained cell quantizes to the same code at every q of that frame.

Every codec, dispatch and scoring step is the existing implementation: this
module frames no bytes, selects no decoder and computes no metric of its own. It
composes `ae_uint8_transport.encode_frame`, `PreloadedAeDecoders.receive`, the
p025 service policy, the AVO person view, the frozen segmentation scorer and the
registered gate definitions, and adds only the per-frame integrity audit, the
payload accounting and the preregistered interpretation.

Each measured q is compared against the frozen noAE UINT8+zstd validation result
at the **same q**, so the reported degradation isolates the AE128 latent
transport instead of re-measuring the ROI drop the noAE path already pays.

Per q the durability order is fixed: the setting JSON is the scientific
completion record and is fsynced into place *first*, that q's scratch
predictions are removed only afterwards, and the cleanup marker is written last.
An interruption can therefore lose at most a scratch prediction directory, never
a completed measurement, and a resume finishes the unfinished cleanup rather
than remeasuring the q.

Acceptance is preregistered here, before any Phase-9D number exists, and is
evaluated verbatim; q=0.90 and q=0.98 are stress/emergency profiles whatever
they measure, and no setting is tuned or removed after observing a result. The
recorded component latency is current-host diagnostic evidence only: no
Raspberry Pi and no OAI latency is claimed anywhere.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1.runtime import (
    apply_p025_service_policy,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.runtime import (
    combined_records,
)

from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import contract, continuous_q, guards
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import (
    load_frozen_perception,
    sha256_file,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.phase5_common import (
    load_frozen_scorers,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.phase6_validation import (
    _collate,
    _person_only,
    load_validation_person_truth,
    score_validation_pass,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.phase7_zstd_measurement import (
    TORCH_CPU_THREADS,
    _latency_stats,
    _sync,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.phase8b_uint8_validation import (
    _atomic_json,
    _atomic_write,
    _identity_digest,
    _q_slug,
    _require_tree_finite,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.zstd_transport import (
    implementation_report,
)
from . import ae_contract, ae_family_dispatch, ae_uint8_transport
from . import ae_training_common as common
from .ae_gpu_qualification import CountingWireCodec, bind_inputs as bind_frozen_inputs
from .ae_gpu_qualification import state_hashes
from .ae_model import build_split_feature_ae


EXECUTE_TOKEN = "SPLITFUSION_AE128_PHASE9D_UINT8_VALIDATION"
TERMINAL = "SPLITFUSION_AE128_UINT8_VALIDATION_COMPLETE"
SETTING_TERMINAL = "SPLITFUSION_AE128_UINT8_Q_SETTING_COMPLETE"
CLEANUP_TERMINAL = "SPLITFUSION_AE128_UINT8_Q_PREDICTIONS_REMOVED"
SCHEMA = "splitfusion_fcos_ae128_phase9d_uint8_validation_v1"
SETTING_SCHEMA = "splitfusion_fcos_ae128_phase9d_uint8_setting_v1"
CLEANUP_SCHEMA = "splitfusion_fcos_ae128_phase9d_uint8_cleanup_v1"

# Durability order per q, and the reason for it. The per-q setting JSON is the
# scientific completion record, so it is written and fsynced into place *first*.
# Only then are that q's scratch predictions removed, and the cleanup marker is
# written last. An interruption can therefore lose at most a scratch prediction
# directory -- never a completed measurement -- and resume finishes the cleanup
# instead of remeasuring the q.
DURABILITY_ORDER = (
    "atomically write settings/<q>.json",
    "remove working_predictions/<q>",
    "atomically write cleanup/<q>.json",
)

DATALOADER_WORKERS = 8
INFERENCE_BATCH = 8

# Exactly the six registered anchors, one inference/evaluation pass each.
Q_VALUES = tuple(contract.REGISTERED_Q_VALUES)  # 0, .30, .50, .70, .90, .98
STRESS_Q_VALUES = tuple(contract.EVALUATION_STRESS_Q_VALUES)  # .90, .98
ACCEPTANCE_PRIMARY_Q = (0.30, 0.50, 0.70)
GATE_COUNT = common.SAME_Q_GATE_COUNT  # 12
SERVICE_GATE_COUNT = len(contract.ABSOLUTE_SERVICE_TARGETS)  # 9

AE_BOTTLENECK = common.AE_TRAINING_BOTTLENECK  # 128
AE_FAMILY_ID = ae_contract.AE_FAMILY_AE128


# ---------------------------------------------------------------------------
# The three Phase-9D artifacts, bound by exact SHA-256
# ---------------------------------------------------------------------------

PHASE9C_TRAINING_RELPATH = (
    "experiments/splitfusion_fcos_ae_v1/20260902_220623_phase9c_ae128_training"
)
SELECTED_CHECKPOINT_EPOCH = 8
SELECTED_CHECKPOINT_RELPATH = (
    f"{PHASE9C_TRAINING_RELPATH}/checkpoints/"
    f"{common.candidate_filename(SELECTED_CHECKPOINT_EPOCH)}"
)
SELECTED_CHECKPOINT_SHA256 = (
    "0c2ba3a495684c0f8222492f554eb3de7c7a76181e0bd4b4a83529897db30f72"
)
HOLDOUT_DECISION_RELPATH = (
    f"{PHASE9C_TRAINING_RELPATH}/holdout_selection/holdout_selection.json"
)
HOLDOUT_DECISION_SHA256 = (
    "69e49deac302fc46c1eec56036e3ab3d769b3aac10b76541cfb4abb80f878194"
)
NOAE_UINT8_VALIDATION_RELPATH = (
    "experiments/splitfusion_fcos_hybrid_q_v1/"
    "20260902_223610_phase8b_uint8_validation/phase8b_uint8_validation.json"
)
NOAE_UINT8_VALIDATION_SHA256 = (
    "a2779f5fb0a585b1c317dc755b5ab577fa7c34963ab7945cb704e0d4146bb029"
)
HOLDOUT_DECISION_TERMINAL = "SPLITFUSION_AE128_HOLDOUT_CHECKPOINT_SELECTED"
NOAE_UINT8_VALIDATION_SCHEMA = "splitfusion_fcos_hybrid_q_phase8b_uint8_validation_v1"
NOAE_UINT8_VALIDATION_TERMINAL = "HYBRID_Q_UINT8_VALIDATION_COMPLETE"

# The two AE modules that decide what the saved AE128 tensors *mean* -- the
# architecture and shapes, and the family/bottleneck/latent-geometry registry
# they are built against. They must be byte-identical to what the selected
# checkpoint recorded, or the loaded weights are being reinterpreted.
AE_CHECKPOINT_SEMANTICS_SOURCES = ("ae_contract.py", "ae_model.py")
# The AE modules that define the wire and the composition order this phase
# measures. Phase 9D is a measurement of exactly the committed transport, so
# these must also be unchanged since the checkpoint was written.
AE_TRANSPORT_SEMANTICS_SOURCES = (
    "ae_composition.py",
    "ae_loss.py",
    "ae_uint8_transport.py",
    "__init__.py",
)
# Phase 9D *is* an addition to the AE package, so its own source map cannot be
# bit-identical to the one the checkpoint recorded. Exactly these pre-existing
# files may differ, and every difference is reported per file either way.
AE_PHASE9D_MODIFIED_SOURCES = (
    "ae_family_dispatch.py",
    "ae_holdout_selection.py",
    "ae_training_common.py",
)


# ---------------------------------------------------------------------------
# Preregistered interpretation. Registered here, before any Phase-9D number
# exists, and applied verbatim by `evaluate_acceptance`.
# ---------------------------------------------------------------------------

BASELINE_SERVICE_PASS_COUNT = contract.FROZEN_Q0_SERVICE_PASS_COUNT  # 7 of 9

ACCEPTANCE_RULE = (
    "AE128 UINT8+zstd deployment is accepted if and only if both hold: "
    "(1) q=0 passes all 12 same-q preservation gates against the frozen noAE "
    "UINT8+zstd validation result and retains at least the baseline "
    f"{BASELINE_SERVICE_PASS_COUNT}/{SERVICE_GATE_COUNT} absolute service gates; "
    "and (2) at least one of q in {0.30, 0.50, 0.70} passes all 12 same-q "
    "preservation gates without reducing the absolute service-gate count below "
    "the frozen noAE UINT8+zstd count at that same q. "
    "q=0.90 and q=0.98 are stress/emergency profiles regardless of their "
    "results and cannot make or break acceptance. Every q is reported "
    "independently, and no setting is tuned or removed after observing a result."
)
ACCEPTED_TERMINAL = "AE128_UINT8_ZSTD_DEPLOYMENT_ACCEPTED"
NOT_ACCEPTED_TERMINAL = "AE128_UINT8_ZSTD_DEPLOYMENT_NOT_ACCEPTED"

SAME_Q_BASELINE_LABEL = "frozen noAE UINT8+zstd validation result at the same q"


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------


def routing_tag() -> int:
    """The deterministic nonzero routing tag of the selected checkpoint.

    Derived from the **full** selected-checkpoint SHA-256 by
    `ae_contract.routing_tag_from_sha256`. The result is a 32-bit decoder-routing
    discriminator and is deliberately not the checkpoint's identity: 32 bits
    cannot authenticate a checkpoint, and the authoritative identity stays the
    full digest recorded beside it.
    """
    return ae_contract.routing_tag_from_sha256(SELECTED_CHECKPOINT_SHA256)


def routing_record() -> dict[str, Any]:
    """Both facts, side by side, with the tag's role stated."""
    tag = routing_tag()
    return {
        "selected_checkpoint_path": SELECTED_CHECKPOINT_RELPATH,
        "selected_checkpoint_sha256": SELECTED_CHECKPOINT_SHA256,
        "routing_tag": tag,
        "routing_tag_hex": f"0x{tag:08x}",
        "routing_tag_bytes": ae_contract.AE_ROUTING_TAG_BYTES,
        "routing_tag_derivation": (
            "leading 32 bits of the full selected-checkpoint SHA-256, via "
            "ae_contract.routing_tag_from_sha256"
        ),
        "routing_tag_is_checkpoint_identity": False,
        "routing_tag_role": (
            "per-frame decoder-routing discriminator, so a frame delayed or "
            "reordered across a profile switch is refused instead of decoded by "
            "a different AE"
        ),
        "checkpoint_identity_authority": (
            "the full SHA-256 recorded here; the truncated 32-bit tag "
            "authenticates nothing"
        ),
    }


def _bind_artifact(relative: str, expected: str) -> dict[str, str]:
    path = (contract.repository_root() / relative).resolve(strict=True)
    observed = sha256_file(path)
    if observed != expected:
        raise guards.HybridQConfigError(f"{relative} sha256 drift")
    return {"path": relative, "sha256": observed}


def bind_inputs() -> dict[str, Any]:
    """The frozen stack plus the three Phase-9D artifacts, all hash-bound.

    The frozen perception checkpoint, the stable epoch-4 ranker, the p025 forward
    lock and the hybrid-q locked configuration are bound by the existing
    authorized AE binding, unchanged.
    """
    binding = bind_frozen_inputs()
    return {
        **binding,
        "selected_ae128_checkpoint": _bind_artifact(
            SELECTED_CHECKPOINT_RELPATH, SELECTED_CHECKPOINT_SHA256
        ),
        "ae128_holdout_selection_decision": _bind_artifact(
            HOLDOUT_DECISION_RELPATH, HOLDOUT_DECISION_SHA256
        ),
        "noae_uint8_zstd_validation_reference": _bind_artifact(
            NOAE_UINT8_VALIDATION_RELPATH, NOAE_UINT8_VALIDATION_SHA256
        ),
        "framed_fp32_noae_q0_payload_bytes": contract.FRAMED_Q0_PAYLOAD_BYTES,
        "routing": routing_record(),
    }


def load_holdout_decision(binding: Mapping[str, Any]) -> dict[str, Any]:
    """The bound Phase-9C decision, and the chain from it to this checkpoint."""
    root = contract.repository_root()
    document = json.loads(
        (root / HOLDOUT_DECISION_RELPATH).read_text(encoding="utf-8")
    )
    if document.get("schema") != common.AE_HOLDOUT_SCHEMA:
        raise guards.HybridQConfigError("AE128 holdout-selection schema drift")
    if document.get("terminal") != HOLDOUT_DECISION_TERMINAL:
        raise guards.HybridQConfigError("AE128 holdout selection did not complete")
    scope = document["scope"]
    if bool(scope["validation_or_test_accessed"]):
        raise guards.HybridQConfigError(
            "the holdout decision reports validation/test access"
        )
    if str(scope["transport"]) != common.AE_HOLDOUT_QUANTIZER:
        raise guards.HybridQConfigError("holdout decision transport drift")
    selection = document["selection"]
    if int(selection["selected_epoch"]) != SELECTED_CHECKPOINT_EPOCH:
        raise guards.HybridQConfigError(
            f"the bound decision selected epoch {selection['selected_epoch']}, "
            f"Phase 9D binds epoch {SELECTED_CHECKPOINT_EPOCH}"
        )
    candidates = dict(document["training_run"]["candidate_checkpoints"])
    name = common.candidate_filename(SELECTED_CHECKPOINT_EPOCH)
    if candidates.get(name) != SELECTED_CHECKPOINT_SHA256:
        raise guards.HybridQConfigError(
            "the decision's recorded candidate hash is not the bound checkpoint"
        )
    decided = document["binding"]
    for name_, ours in (
        ("frozen_checkpoint", binding["frozen_checkpoint"]["sha256"]),
        ("stable_epoch4_ranker", binding["stable_epoch4_ranker"]["sha256"]),
        ("perception_forward_lock", binding["perception_forward_lock"]["sha256"]),
        ("hybrid_q_locked_config", binding["hybrid_q_locked_config"]["sha256"]),
    ):
        if decided[name_]["sha256"] != ours:
            raise guards.HybridQConfigError(
                f"the holdout decision bound a different {name_}"
            )
    return {
        "path": HOLDOUT_DECISION_RELPATH,
        "sha256": HOLDOUT_DECISION_SHA256,
        "selected_epoch": int(selection["selected_epoch"]),
        "decided_at_criterion": selection["decided_at_criterion"],
        "rule": selection["rule"],
        "selection_is_a_service_ready_claim": bool(
            selection["selection_is_a_service_ready_claim"]
        ),
        "selected_checkpoint": name,
        "selected_checkpoint_sha256": candidates[name],
        "candidate_checkpoints": candidates,
        "holdout_frames": int(scope["holdout_frames"]),
        "holdout_transport": str(scope["transport"]),
    }


def load_noae_reference() -> dict[int, dict[str, Any]]:
    """The frozen noAE UINT8+zstd validation rows, keyed by wire q_e4.

    Nothing is recomputed: these are exactly the rows the completed Phase-8B
    measurement published on the same 3,345 validation frames, the same frozen
    perception tail, the same stable epoch-4 ranker and the same scorers.
    """
    root = contract.repository_root()
    path = (root / NOAE_UINT8_VALIDATION_RELPATH).resolve(strict=True)
    if sha256_file(path) != NOAE_UINT8_VALIDATION_SHA256:
        raise guards.HybridQConfigError("frozen noAE UINT8+zstd reference sha256 drift")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != NOAE_UINT8_VALIDATION_SCHEMA:
        raise guards.HybridQConfigError("noAE UINT8+zstd reference schema drift")
    if document.get("terminal") != NOAE_UINT8_VALIDATION_TERMINAL:
        raise guards.HybridQConfigError("noAE UINT8+zstd reference is incomplete")
    scope = document["scope"]
    if int(scope["validation_frames_per_q"]) != contract.VALIDATION_FRAMES:
        raise guards.HybridQConfigError("noAE reference validation frame-count drift")
    if int(scope["inference_passes_per_q"]) != 1:
        raise guards.HybridQConfigError("noAE reference did not run one pass per q")
    if bool(scope["test_accessed"]) or bool(scope["carla_launched"]):
        raise guards.HybridQConfigError("noAE reference reports test/CARLA access")
    if bool(scope["training_or_tuning"]) or bool(scope["threshold_or_gate_change"]):
        raise guards.HybridQConfigError("noAE reference reports training/gate change")
    if not bool(document["transport"]["zstd_mandatory"]):
        raise guards.HybridQConfigError("noAE reference did not mandate zstd")

    rows: dict[int, dict[str, Any]] = {}
    for row in document["curve"]:
        plan = continuous_q.quantize_q(float(row["q"]))
        if plan.q_e4 in rows:
            raise guards.HybridQConfigError("duplicate q in the noAE reference curve")
        if int(row["retained_cells"]) != plan.keep_count:
            raise guards.HybridQConfigError("noAE reference keep-count drift")
        if int(row["frames"]) != contract.VALIDATION_FRAMES:
            raise guards.HybridQConfigError("noAE reference row frame-count drift")
        if set(row["metrics"]) != set(contract.PROTECTED_METRICS):
            raise guards.HybridQConfigError("noAE reference protected metric drift")
        gates = row["absolute_service_gates"]
        if len(gates["targets"]) != SERVICE_GATE_COUNT:
            raise guards.HybridQConfigError("noAE reference service-gate count drift")
        rows[plan.q_e4] = {
            "q": plan.wire_q,
            "q_e4": plan.q_e4,
            "retained_cells": plan.keep_count,
            "metrics": dict(row["metrics"]),
            "canonical_person_metrics": dict(row["canonical_person_metrics"]),
            "absolute_service_pass_count": int(gates["pass_count"]),
            "failed_absolute_service_gates": list(gates["failed"]),
            "pre_zstd_bytes": int(row["measured_uint8_sparse_bytes"]),
            "zstd_bytes": dict(row["compressed_zstd_bytes"]),
        }
    expected = {continuous_q.quantize_q(q).q_e4 for q in Q_VALUES}
    if set(rows) != expected:
        raise guards.HybridQConfigError("noAE UINT8+zstd reference q ladder drift")
    baseline = rows[continuous_q.quantize_q(0.0).q_e4]
    if baseline["absolute_service_pass_count"] != BASELINE_SERVICE_PASS_COUNT:
        raise guards.HybridQConfigError(
            f"the noAE UINT8+zstd q=0 row passes "
            f"{baseline['absolute_service_pass_count']} absolute service gates, "
            f"the registered baseline is {BASELINE_SERVICE_PASS_COUNT}"
        )
    return rows


def require_selected_bindings(
    payload: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Enforce every binding the selected checkpoint saved.

    All but one field are required to be bit-identical, iterated from
    `common.binding_fields` so no field can be silently forgotten.

    The single exception is `ae_package_source_sha256`, the AE package's own
    source map: Phase 9D *is* an addition to that package, so requiring equality
    would be requiring this phase not to exist. It is enforced as a declared
    delta instead -- every module that defines what the saved tensors mean, and
    every module that defines the transport being measured, must be byte
    identical; only the explicitly allowlisted pre-existing files may differ; and
    every added, removed and changed file is reported with its before/after
    hashes either way.
    """
    expected = common.binding_fields(binding)
    delta: dict[str, Any] | None = None
    for name, value in expected.items():
        if name not in payload:
            raise guards.HybridQConfigError(f"selected checkpoint does not carry {name}")
        if name == "ae_package_source_sha256":
            delta = ae_package_source_delta(dict(payload[name]), dict(value))
            continue
        if payload[name] != value:
            raise guards.HybridQConfigError(f"selected checkpoint {name} drift")
    if delta is None:
        raise guards.HybridQConfigError(
            "the saved binding no longer carries an AE package source map"
        )
    return {"enforced_exactly": sorted(set(expected) - {"ae_package_source_sha256"}),
            "ae_package_source_delta": delta}


def ae_package_source_delta(
    recorded: Mapping[str, str], live: Mapping[str, str]
) -> dict[str, Any]:
    """Which AE package files moved since the selected checkpoint was written."""
    changed = sorted(
        name for name in recorded if name in live and live[name] != recorded[name]
    )
    added = sorted(set(live) - set(recorded))
    removed = sorted(set(recorded) - set(live))
    frozen = tuple(AE_CHECKPOINT_SEMANTICS_SOURCES) + tuple(
        AE_TRANSPORT_SEMANTICS_SOURCES
    )
    violated = [name for name in frozen if live.get(name) != recorded.get(name)]
    if violated:
        raise guards.HybridQConfigError(
            "AE module(s) defining the checkpoint or the measured transport "
            f"changed since the selected checkpoint was written: {violated}"
        )
    if removed:
        raise guards.HybridQConfigError(
            f"AE package file(s) recorded by the checkpoint are gone: {removed}"
        )
    unexpected = [name for name in changed if name not in AE_PHASE9D_MODIFIED_SOURCES]
    if unexpected:
        raise guards.HybridQConfigError(
            f"unallowlisted AE package file(s) changed since the checkpoint: {unexpected}"
        )
    return {
        "semantics_modules_required_unchanged": list(frozen),
        "semantics_modules_unchanged": True,
        "allowlisted_modifiable": list(AE_PHASE9D_MODIFIED_SOURCES),
        "changed": [
            {"path": name, "checkpoint_sha256": recorded[name], "live_sha256": live[name]}
            for name in changed
        ],
        "added": [{"path": name, "live_sha256": live[name]} for name in added],
        "removed": removed,
        "rationale": (
            "Phase 9D adds a runner, a test and a report to the AE package and "
            "extends the receive adapter with opt-in diagnostics, so its source "
            "map differs from the one the checkpoint recorded by construction"
        ),
    }


def load_selected_ae(
    device: torch.device, binding: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Load exactly the bound AE128 checkpoint, frozen, eval, routing-tag bound."""
    path = (contract.repository_root() / SELECTED_CHECKPOINT_RELPATH).resolve(strict=True)
    digest = sha256_file(path)
    if digest != SELECTED_CHECKPOINT_SHA256:
        raise guards.HybridQConfigError("selected AE128 checkpoint sha256 drift")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema"] != common.AE_CANDIDATE_SCHEMA:
        raise guards.HybridQConfigError("selected checkpoint schema drift")
    if int(payload["epoch"]) != SELECTED_CHECKPOINT_EPOCH:
        raise guards.HybridQConfigError("selected checkpoint epoch drift")
    if int(payload["bottleneck"]) != AE_BOTTLENECK:
        raise guards.HybridQConfigError("selected checkpoint bottleneck drift")
    if int(payload["family_id"]) != AE_FAMILY_ID:
        raise guards.HybridQConfigError("selected checkpoint family drift")
    if payload["configuration"] != common.training_configuration():
        raise guards.HybridQConfigError(
            "the selected checkpoint was trained under a different locked configuration"
        )
    enforced = require_selected_bindings(payload, binding)

    autoencoder = build_split_feature_ae(AE_BOTTLENECK)
    autoencoder.load_state_dict(payload["autoencoder"])
    if autoencoder.parameter_count() != int(payload["parameter_count"]):
        raise guards.HybridQConfigError("selected checkpoint parameter-count drift")
    autoencoder = autoencoder.to(device)
    common.freeze(autoencoder)
    guards.require_module_parameters_finite(autoencoder, "selected AE128")
    autoencoder.bind_routing_tag(routing_tag())
    if autoencoder.routing_tag != routing_tag() or not autoencoder.is_bound:
        raise guards.HybridQConfigError("selected AE128 routing tag was not bound")
    per_tensor, aggregate = state_hashes(autoencoder)
    provenance = {
        **routing_record(),
        "epoch": int(payload["epoch"]),
        "stage": str(payload["stage"]),
        "bottleneck": AE_BOTTLENECK,
        "family_id": AE_FAMILY_ID,
        "family_name": ae_contract.family_name(AE_FAMILY_ID),
        "parameter_count": autoencoder.parameter_count(),
        "global_update_index": int(payload["global_update_index"]),
        "stage_b_cycle_position": int(payload["stage_b_cycle_position"]),
        "wire_identity": autoencoder.wire_identity(),
        "state_sha256": aggregate,
        "state_sha256_per_tensor": per_tensor,
        "bindings": enforced,
        "trained_in_this_phase": False,
    }
    del payload
    return autoencoder, provenance


# ---------------------------------------------------------------------------
# One frame through the deployment path
# ---------------------------------------------------------------------------


class _CountingRanker:
    """Instrumentation-only proxy over the one frozen stable epoch-4 ranker.

    It owns no parameters, changes no score and adds no tensor op: it forwards
    to the frozen `score_cells` and records how often it was called and the
    identity of the tensor it was handed, so the pass can prove that q=0 called
    the ranker zero times and that every q>0 call read the original FP32 C2
    object rather than a latent, a copy or a quantized tensor.
    """

    def __init__(self, ranker: torch.nn.Module) -> None:
        self._ranker = ranker
        self.invocations = 0
        self.scored_pointers: list[int] = []

    def reset(self) -> None:
        self.invocations = 0
        self.scored_pointers = []

    def score_cells(self, c2: torch.Tensor) -> torch.Tensor:
        self.invocations += 1
        self.scored_pointers.append(int(c2.data_ptr()))
        return self._ranker.score_cells(c2)


def _transport_one(
    *,
    frame: torch.Tensor,
    autoencoder: Any,
    ranker: _CountingRanker,
    decoders: ae_family_dispatch.PreloadedAeDecoders,
    plan: continuous_q.ContinuousQ,
    wire: CountingWireCodec,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Transmit and receive one frame, then audit every registered invariant.

    Returns the reconstructed FP32 C2 the frozen tail will consume, plus the
    per-frame integrity and cost row. Nothing here reimplements a codec step:
    the transmit side is `ae_uint8_transport.encode_frame` and the receive side
    is `PreloadedAeDecoders.receive` over the raw wire bytes.
    """
    guards.require_frozen_c2(frame, what="original validation FP32 C2")
    original_pointer = int(frame.data_ptr())
    ranker.reset()

    _sync(device)
    started = time.perf_counter_ns()
    transport = ae_uint8_transport.encode_frame(
        frame, autoencoder, ranker, plan.wire_q, wire_codec=wire
    )
    _sync(device)
    transmit_ns = time.perf_counter_ns() - started
    packet = transport.packet

    expected_ranker_calls = 0 if plan.is_bypass else 1
    if ranker.invocations != expected_ranker_calls:
        raise guards.HybridQPayloadError(
            f"q={plan.wire_q} invoked the ranker {ranker.invocations} times, "
            f"expected {expected_ranker_calls}"
        )
    if any(pointer != original_pointer for pointer in ranker.scored_pointers):
        raise guards.HybridQPayloadError(
            "the ranker did not score the original FP32 C2 tensor"
        )
    if plan.is_bypass and transport.selection is not None:
        raise guards.HybridQPayloadError("q=0 emitted a sparse selection")
    if not plan.is_bypass and transport.selection is None:
        raise guards.HybridQPayloadError("q>0 omitted its selection")

    analytical = ae_uint8_transport.analytical_size(plan.wire_q, AE_BOTTLENECK)
    if packet.uncompressed_bytes != analytical.total_bytes:
        raise guards.HybridQPayloadError("pre-zstd AE payload size drift")
    if packet.compressed_bytes <= 0:
        raise guards.HybridQPayloadError("empty zstd packet")

    # Received raw bytes only: the local packet object is never handed to the
    # receive path, so the decoder can only be discovered from the header.
    wire.decompressions = 0
    _sync(device)
    started = time.perf_counter_ns()
    received = decoders.receive(packet.data, wire_codec=wire, diagnostics=True)
    _sync(device)
    receive_ns = time.perf_counter_ns() - started
    decompressions = wire.decompressions
    if decompressions != 1:
        raise guards.HybridQPayloadError(
            f"the receive path performed {decompressions} zstd decompressions"
        )

    diagnostics = received.diagnostics
    if diagnostics is None:
        raise guards.HybridQPayloadError("the receive path returned no diagnostics")
    parsed = diagnostics.parsed
    if diagnostics.decoder is not autoencoder:
        raise guards.HybridQPayloadError(
            "the header-selected decoder is not the preloaded selected AE128"
        )
    if received.family.family_id != AE_FAMILY_ID:
        raise guards.HybridQPayloadError("received family is not AE128")
    if received.family.transported_channels != AE_BOTTLENECK:
        raise guards.HybridQPayloadError("received latent width is not 128 channels")
    if received.family.routing_tag != routing_tag():
        raise guards.HybridQPayloadError("received routing tag is not the bound tag")
    if received.family.codec != "ae_latent_uint8":
        raise guards.HybridQPayloadError("received codec identity drift")
    if int(parsed.header.q_e4) != plan.q_e4:
        raise guards.HybridQPayloadError("received header q drift")
    if continuous_q.quantize_q(received.q).q_e4 != plan.q_e4:
        raise guards.HybridQPayloadError("decoded AE q drift")
    guards.require_keep_cardinality(int(received.keep_count), plan.keep_count)
    guards.require_keep_cardinality(int(diagnostics.keep_mask.sum()), plan.keep_count)
    if int(parsed.values.shape[0]) != plan.keep_count or int(
        parsed.values.shape[1]
    ) != AE_BOTTLENECK:
        raise guards.HybridQPayloadError("retained UINT8 value block shape drift")

    # The retained UINT8 cells are exactly the cells selection chose.
    if plan.is_bypass:
        expected_indices = torch.arange(
            ae_contract.AE_LATENT_CELLS, dtype=torch.int64
        )
    else:
        expected_indices = (
            transport.selection.keep_indices.detach().to(device="cpu", dtype=torch.int64)
        )
    if not torch.equal(parsed.keep_indices, expected_indices):
        raise guards.HybridQPayloadError(
            "the retained UINT8 cells are not the selected cells"
        )

    # Dropped cells scatter to exact zero across all 128 latent channels before
    # the decoder ever runs.
    flat = diagnostics.latent.reshape(AE_BOTTLENECK, ae_contract.AE_LATENT_CELLS)
    occupied = (flat != 0).any(dim=0)
    dropped = ~diagnostics.keep_mask.reshape(-1)
    if int(dropped.sum()) != plan.drop_count:
        raise guards.HybridQPayloadError("reconstructed drop cardinality drift")
    if bool(occupied[dropped].any()):
        raise guards.HybridQNumericalError(
            "a dropped latent cell did not scatter to exact zero"
        )

    reconstructed = received.c2
    guards.require_frozen_c2(reconstructed, what="reconstructed AE128 validation C2")
    # q=0 bypasses the ranker but not AE128, so no q is ever an identity.
    if torch.equal(reconstructed, frame):
        raise guards.HybridQNumericalError(
            "the AE128 reconstruction is bit-identical to the original C2"
        )

    row = {
        "pre_zstd_bytes": int(packet.uncompressed_bytes),
        "zstd_bytes": int(packet.compressed_bytes),
        "keep_count": int(received.keep_count),
        "ranker_invocations": int(ranker.invocations),
        "zstd_decompressions": int(decompressions),
        "transmit_ns": int(transmit_ns),
        "receive_ns": int(receive_ns),
    }
    return reconstructed, row


# ---------------------------------------------------------------------------
# One complete validation pass at one q
# ---------------------------------------------------------------------------


def _byte_stats(values: Sequence[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "minimum": int(array.min()),
        "maximum": int(array.max()),
        "samples": int(array.size),
    }


def _load_runtime(device: torch.device) -> dict[str, Any]:
    """The one frozen model/ranker/AE and the registered validation ordering."""
    model, base, perception = load_frozen_perception(device)
    common.freeze(model)
    ranker = common.load_stable_ranker(device)
    guards.require_frozen_perception([model, ranker])
    guards.require_eval_mode([model, ranker])

    root = contract.repository_root()
    config = json.loads(
        (
            root
            / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
            "splitfusion_fcos_r50_fpn_p2_p7_v1/config.json"
        ).read_text(encoding="utf-8")
    )
    dataset_root = (root / config["dataset_root"]).resolve(strict=True)
    truth = load_validation_person_truth()
    frame_ids = list(truth["frame_ids"])
    inference = base.data.InferenceDataset(dataset_root, "val")
    position_by_id = {
        str(row["sample_id"]): index for index, row in enumerate(inference.rows)
    }
    if len(position_by_id) != contract.VALIDATION_FRAMES:
        raise guards.HybridQConfigError("validation inference row count drift")
    positions = [position_by_id[sample_id] for sample_id in frame_ids]
    if len(set(positions)) != contract.VALIDATION_FRAMES:
        raise guards.HybridQConfigError("validation position mapping is not one-to-one")
    return {
        "model": model,
        "base": base,
        "perception": perception,
        "ranker": ranker,
        "device": device,
        "dataset_root": dataset_root,
        "truth": truth,
        "frame_ids": frame_ids,
        "inference": inference,
        "positions": positions,
    }


def _require_state_unchanged(runtime: Mapping[str, Any]) -> None:
    guards.require_module_state_unchanged(runtime["model"], runtime["model_snapshot"])
    guards.require_module_state_unchanged(runtime["ranker"], runtime["ranker_snapshot"])
    guards.require_module_state_unchanged(
        runtime["autoencoder"], runtime["autoencoder_snapshot"]
    )


def run_validation_pass(
    *,
    runtime: Mapping[str, Any],
    q: float,
    output: Path,
    workers: int,
    wire: CountingWireCodec,
) -> dict[str, Any]:
    """The one and only complete AE128 UINT8+zstd inference pass for one q."""
    plan = continuous_q.quantize_q(q)
    if not plan.is_registered:
        raise guards.HybridQConfigError("Phase 9D accepts registered q only")
    output.mkdir(parents=True, exist_ok=False)
    (output / "segmentation").mkdir()
    detections_path = output / "detections.csv"
    segmentation_manifest = output / "segmentation_manifest.csv"

    loader = DataLoader(
        Subset(runtime["inference"], list(runtime["positions"])),
        batch_size=INFERENCE_BATCH,
        shuffle=False,
        num_workers=workers,
        collate_fn=_collate,
        drop_last=False,
        pin_memory=False,
    )
    ranker = _CountingRanker(runtime["ranker"])
    decoders = runtime["decoders"]
    autoencoder = runtime["autoencoder"]
    device = runtime["device"]

    pre_zstd_sizes: set[int] = set()
    pre_zstd_bytes: list[int] = []
    zstd_sizes: list[int] = []
    transmit_samples: list[int] = []
    receive_samples: list[int] = []
    tail_samples: list[int] = []
    observed_ids: list[str] = []
    segmentation_rows: list[dict[str, Any]] = []
    keep_counts: set[int] = set()
    ranker_invocations = 0
    decompressions = 0
    detection_count = person_count = vehicle_count = 0
    output_tensors_checked = 0
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device=device)

    with detections_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=runtime["base"].infer.FIELDS)
        writer.writeheader()
        with torch.inference_mode():
            for fused, rows, calibrations in loader:
                inputs = fused.to(device, non_blocking=True)
                c2 = runtime["model"].encode_front(inputs).float()
                guards.require_frozen_batched_c2(c2, what="frozen validation C2")

                transported: list[torch.Tensor] = []
                for index in range(c2.shape[0]):
                    reconstructed, frame_row = _transport_one(
                        frame=c2[index],
                        autoencoder=autoencoder,
                        ranker=ranker,
                        decoders=decoders,
                        plan=plan,
                        wire=wire,
                        device=device,
                    )
                    pre_zstd_sizes.add(frame_row["pre_zstd_bytes"])
                    pre_zstd_bytes.append(frame_row["pre_zstd_bytes"])
                    zstd_sizes.append(frame_row["zstd_bytes"])
                    keep_counts.add(frame_row["keep_count"])
                    ranker_invocations += frame_row["ranker_invocations"]
                    decompressions += frame_row["zstd_decompressions"]
                    transmit_samples.append(frame_row["transmit_ns"])
                    receive_samples.append(frame_row["receive_ns"])
                    transported.append(reconstructed.to(device))

                hybrid = torch.stack(transported)
                _sync(device)
                tail_started = time.perf_counter_ns()
                outputs = runtime["model"].decode_tail(hybrid, dense=False)
                _sync(device)
                tail_samples.append(time.perf_counter_ns() - tail_started)
                output_tensors_checked += _require_tree_finite(
                    outputs, "frozen model output"
                )
                calibration_gpu = [
                    {name: tensor.to(device) for name, tensor in calibration.items()}
                    for calibration in calibrations
                ]
                detections = runtime["model"].postprocess(outputs, calibration_gpu)
                output_tensors_checked += _require_tree_finite(
                    detections, "frozen postprocess output"
                )
                for index, row in enumerate(rows):
                    frame_view = {
                        "semantic_logits": outputs["semantic_logits"][index : index + 1]
                    }
                    served, original_indices = apply_p025_service_policy(
                        frame_view, detections[index]
                    )
                    output_tensors_checked += _require_tree_finite(
                        served, "p025 service output"
                    )
                    records = combined_records(
                        runtime["base"], row, served, original_indices
                    )
                    for record in records:
                        writer.writerow(record)
                        if record["class_name"] == "person":
                            person_count += 1
                        else:
                            vehicle_count += 1
                    detection_count += len(records)
                    observed_ids.append(str(row["sample_id"]))

                    source_hw = (int(row["camera_height"]), int(row["camera_width"]))
                    labels = F.interpolate(
                        outputs["semantic_logits"][index : index + 1].float(),
                        size=source_hw,
                        mode="bilinear",
                        align_corners=False,
                    ).argmax(1)[0]
                    array = labels.cpu().numpy().astype(np.uint8)
                    relative = Path("segmentation") / f"{row['sample_id']}.png"
                    if not cv2.imwrite(str(output / relative), array):
                        raise RuntimeError(f"failed segmentation write {relative}")
                    segmentation_rows.append(
                        {
                            "sample_id": row["sample_id"],
                            "prediction_path": str(relative),
                            "width": array.shape[1],
                            "height": array.shape[0],
                        }
                    )
                del c2, hybrid, outputs, detections, transported

    with segmentation_manifest.open("x", encoding="utf-8", newline="") as stream:
        manifest_writer = csv.DictWriter(
            stream, fieldnames=("sample_id", "prediction_path", "width", "height")
        )
        manifest_writer.writeheader()
        manifest_writer.writerows(segmentation_rows)
    del loader

    if observed_ids != list(runtime["frame_ids"]):
        raise guards.HybridQConfigError("validation inference order/coverage drift")
    if len(set(observed_ids)) != contract.VALIDATION_FRAMES:
        raise guards.HybridQConfigError("validation frame uniqueness drift")
    analytical = ae_uint8_transport.analytical_size(plan.wire_q, AE_BOTTLENECK)
    if pre_zstd_sizes != {analytical.total_bytes}:
        raise guards.HybridQPayloadError(f"pre-zstd payload drift: {pre_zstd_sizes}")
    if len(zstd_sizes) != contract.VALIDATION_FRAMES:
        raise guards.HybridQPayloadError("zstd payload count drift")
    if len(pre_zstd_bytes) != contract.VALIDATION_FRAMES:
        raise guards.HybridQPayloadError("pre-zstd payload count drift")
    if keep_counts != {plan.keep_count}:
        raise guards.HybridQPayloadError(f"keep-count drift: {sorted(keep_counts)}")
    expected_ranker_calls = 0 if plan.is_bypass else contract.VALIDATION_FRAMES
    if ranker_invocations != expected_ranker_calls:
        raise guards.HybridQPayloadError("ranker invocation count drift")
    if decompressions != contract.VALIDATION_FRAMES:
        raise guards.HybridQPayloadError(
            "the pass did not perform exactly one zstd decompression per frame"
        )

    return {
        "q": plan.wire_q,
        "q_e4": plan.q_e4,
        "frames": len(observed_ids),
        "prediction_root": str(output),
        "detections_csv_sha256": sha256_file(detections_path),
        "segmentation_manifest_sha256": sha256_file(segmentation_manifest),
        "detections": detection_count,
        "person_service_outputs": person_count,
        "vehicle_service_outputs": vehicle_count,
        "retained_cells": plan.keep_count,
        "dropped_cells": plan.drop_count,
        "payload": {
            "family": ae_contract.family_name(AE_FAMILY_ID),
            "transported_latent_channels": AE_BOTTLENECK,
            "analytical_pre_zstd_bytes": analytical.total_bytes,
            "analytical_breakdown": {
                "header_bytes": analytical.header_bytes,
                "mask_bytes": analytical.mask_bytes,
                "range_bytes": analytical.range_bytes,
                "value_bytes": analytical.value_bytes,
            },
            "pre_zstd_bytes": _byte_stats(pre_zstd_bytes),
            "zstd_bytes": _byte_stats(zstd_sizes),
            "zstd_mandatory": True,
            "zstd_level_tuned_here": False,
            "zstd": implementation_report(),
        },
        "component_latency": {
            "transmit": _latency_stats(transmit_samples),
            "receive": _latency_stats(receive_samples),
            "frozen_tail_per_batch": _latency_stats(tail_samples),
            "transmit_includes": [
                "stable epoch-4 ranker and continuous-q selection (q>0 only)",
                "AE128 encoder on the complete frame",
                "per-channel range computation from the complete latent",
                "UINT8 quantization and sparse AE framing",
                "mandatory zstd level-1 compression",
            ],
            "receive_includes": [
                "exactly one zstd decompression",
                "AE header inspection and preloaded-decoder selection",
                "UINT8 dequantization and zero scatter",
                "AE128 decoder",
            ],
            "excludes": [
                "RGB/radar loading and collation",
                "frozen backbone front inference",
                "postprocessing, p025 service policy and scoring",
                "segmentation write-out",
            ],
            "evidence_scope": (
                "current-host diagnostic only; no Raspberry Pi and no OAI "
                "latency is claimed or implied"
            ),
            "batch_size_for_tail": INFERENCE_BATCH,
        },
        "integrity": {
            "ranker_invocations": ranker_invocations,
            "q0_ranker_bypassed": plan.is_bypass,
            "ae128_encoder_bypassed": False,
            "ranked_original_fp32_c2_per_frame": not plan.is_bypass,
            "selection_independent_per_frame": True,
            "batched_or_cross_frame_selection_used": False,
            "ranges_from_complete_latent_before_dropping": True,
            "ranges_evidence": (
                "structurally enforced by the hash-bound, unmodified "
                "ae_uint8_transport.encode_frame, which prepares the ranges from "
                "the complete pre-drop latent and is covered by the committed AE "
                "transport test; this pass does not re-derive them, because doing "
                "so would need a second encode of every frame"
            ),
            "retained_uint8_cells_equal_selected_indices": True,
            "dropped_cells_scattered_to_exact_zero": True,
            "zstd_decompressions": decompressions,
            "decoder_selected_from_received_header_bytes": True,
            "local_packet_metadata_used_for_selection": False,
            "reconstruction_is_identity_at_any_q": False,
            "all_outputs_finite": True,
            "output_tensors_checked": output_tensors_checked,
        },
        "wall_seconds": time.time() - started,
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device) / 2 ** 20,
        "peak_reserved_vram_mib": torch.cuda.max_memory_reserved(device) / 2 ** 20,
    }


# ---------------------------------------------------------------------------
# Preregistered interpretation
# ---------------------------------------------------------------------------


def acceptance_inputs(row: Mapping[str, Any]) -> dict[str, Any]:
    """Exactly the fields the preregistered rule reads, and nothing else."""
    preservation = row["same_q_preservation_vs_noae_uint8_zstd"]
    return {
        "q": float(row["q"]),
        "q_e4": int(row["q_e4"]),
        "preservation_gates_passed": int(preservation["gates_passed"]),
        "preservation_all_passed": bool(preservation["all_passed"]),
        "failed_preservation_gates": list(preservation["failed"]),
        "absolute_service_pass_count": int(row["absolute_service_gates"]["pass_count"]),
        "failed_absolute_service_gates": list(row["absolute_service_gates"]["failed"]),
        "noae_same_q_absolute_service_pass_count": int(
            row["noae_same_q_reference"]["absolute_service_pass_count"]
        ),
    }


def evaluate_acceptance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply `ACCEPTANCE_RULE` verbatim to the six `acceptance_inputs` rows."""
    by_q_e4: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        q_e4 = int(row["q_e4"])
        if q_e4 in by_q_e4:
            raise guards.HybridQConfigError(f"duplicate q_e4 {q_e4} in acceptance input")
        by_q_e4[q_e4] = row
    expected = {continuous_q.quantize_q(q).q_e4 for q in Q_VALUES}
    if set(by_q_e4) != expected:
        raise guards.HybridQConfigError("acceptance input does not cover every q exactly")

    zero = by_q_e4[continuous_q.quantize_q(0.0).q_e4]
    if int(zero["noae_same_q_absolute_service_pass_count"]) != BASELINE_SERVICE_PASS_COUNT:
        raise guards.HybridQConfigError(
            "the frozen noAE q=0 reference does not report the registered "
            f"{BASELINE_SERVICE_PASS_COUNT}/{SERVICE_GATE_COUNT} baseline"
        )
    zero_service = int(zero["absolute_service_pass_count"])
    zero_condition = {
        "q": 0.0,
        "preservation_gates_passed": int(zero["preservation_gates_passed"]),
        "preservation_gate_count": GATE_COUNT,
        "preservation_all_passed": bool(zero["preservation_all_passed"]),
        "failed_preservation_gates": list(zero["failed_preservation_gates"]),
        "absolute_service_pass_count": zero_service,
        "baseline_absolute_service_pass_count": BASELINE_SERVICE_PASS_COUNT,
        "retains_baseline_absolute_service_gates": (
            zero_service >= BASELINE_SERVICE_PASS_COUNT
        ),
        "failed_absolute_service_gates": list(zero["failed_absolute_service_gates"]),
        "passed": bool(zero["preservation_all_passed"])
        and zero_service >= BASELINE_SERVICE_PASS_COUNT,
    }

    primary: list[dict[str, Any]] = []
    for q in ACCEPTANCE_PRIMARY_Q:
        row = by_q_e4[continuous_q.quantize_q(float(q)).q_e4]
        service = int(row["absolute_service_pass_count"])
        noae = int(row["noae_same_q_absolute_service_pass_count"])
        primary.append(
            {
                "q": float(q),
                "preservation_gates_passed": int(row["preservation_gates_passed"]),
                "preservation_all_passed": bool(row["preservation_all_passed"]),
                "failed_preservation_gates": list(row["failed_preservation_gates"]),
                "absolute_service_pass_count": service,
                "noae_same_q_absolute_service_pass_count": noae,
                "reduces_absolute_service_gate_count": service < noae,
                "failed_absolute_service_gates": list(
                    row["failed_absolute_service_gates"]
                ),
                "qualifies": bool(row["preservation_all_passed"]) and service >= noae,
            }
        )
    qualifying = [entry["q"] for entry in primary if entry["qualifies"]]

    stress = []
    for q in STRESS_Q_VALUES:
        row = by_q_e4[continuous_q.quantize_q(float(q)).q_e4]
        stress.append(
            {
                "q": float(q),
                "preservation_gates_passed": int(row["preservation_gates_passed"]),
                "preservation_all_passed": bool(row["preservation_all_passed"]),
                "absolute_service_pass_count": int(row["absolute_service_pass_count"]),
                "noae_same_q_absolute_service_pass_count": int(
                    row["noae_same_q_absolute_service_pass_count"]
                ),
                "designation": "stress/emergency profile regardless of result",
                "influences_acceptance": False,
            }
        )

    accepted = bool(zero_condition["passed"]) and bool(qualifying)
    return {
        "rule": ACCEPTANCE_RULE,
        "registered_before_measurement": True,
        "q0_condition": zero_condition,
        "primary_q_conditions": primary,
        "qualifying_primary_q": qualifying,
        "primary_condition_satisfied": bool(qualifying),
        "stress_q_status": stress,
        "accepted": accepted,
        "decision": ACCEPTED_TERMINAL if accepted else NOT_ACCEPTED_TERMINAL,
        "every_q_reported_independently": True,
        "setting_tuned_after_observing_validation": False,
        "setting_removed_after_observing_validation": False,
    }


# ---------------------------------------------------------------------------
# Per-q setting artifact
# ---------------------------------------------------------------------------


def _setting_document(
    *,
    raw: Mapping[str, Any],
    scored: Mapping[str, Any],
    reference: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    q = float(raw["q"])
    preservation = common.evaluate_same_q_gates(
        reference["metrics"], scored["metrics"], baseline=SAME_Q_BASELINE_LABEL
    )
    deltas = {
        name: {
            "noae_uint8_zstd_same_q": float(reference["metrics"][name]),
            "ae128_uint8_zstd_same_q": float(scored["metrics"][name]),
            "delta_ae_minus_noae": (
                float(scored["metrics"][name]) - float(reference["metrics"][name])
            ),
            "degradation": preservation["gates"][name]["degradation"],
            "bound": preservation["gates"][name]["bound"],
            "normalized_degradation": preservation["gates"][name][
                "normalized_degradation"
            ],
            "passed": preservation["gates"][name]["passed"],
        }
        for name in contract.PROTECTED_METRICS
    }
    finite = all(
        math.isfinite(float(value))
        for value in (
            list(scored["metrics"].values())
            + list(scored["canonical_person_metrics"].values())
        )
    )
    if not finite:
        raise guards.HybridQNumericalError("a scored AE128 metric is non-finite")
    stress = q in contract.EVALUATION_STRESS_Q_VALUES
    return {
        "schema": SETTING_SCHEMA,
        "terminal": SETTING_TERMINAL,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "run_identity_sha256": identity["sha256"],
        **dict(raw),
        "metrics": dict(scored["metrics"]),
        "canonical_person_metrics": dict(scored["canonical_person_metrics"]),
        "absolute_service_gates": dict(scored["absolute_service_gates"]),
        "same_q_preservation_vs_noae_uint8_zstd": preservation,
        "protected_metric_deltas_vs_noae_uint8_zstd_same_q": deltas,
        "noae_same_q_reference": dict(reference),
        "profile_status": {
            "is_stress_anchor": stress,
            "designation": (
                "stress/emergency profile" if stress else "primary profile"
            ),
            "influences_acceptance": not stress,
            "executable": True,
            "removed_for_gate_miss": False,
        },
        "prediction_artifacts": {
            "root": str(raw["prediction_root"]),
            "removed_before_this_record": False,
            "removal_marker": f"cleanup/{_q_slug(q)}.json",
            "durability_order": list(DURABILITY_ORDER),
            "rule": (
                "this record is the durable scientific completion record for "
                "this q and is fsynced into place before its predictions are "
                "removed, so an interruption can only lose scratch predictions"
            ),
        },
        "all_outputs_and_metrics_finite": True,
        "frozen_perception_state_unchanged": True,
        "stable_ranker_state_unchanged": True,
        "selected_ae128_state_unchanged": True,
        "inference_passes_for_this_q": 1,
        "training_or_tuning": False,
        "threshold_nms_or_gate_change": False,
        "test_or_carla_access": False,
    }


def cleanup_marker_path(output: Path, q: float) -> Path:
    return output / "cleanup" / f"{_q_slug(q)}.json"


def load_durable_setting(
    path: Path, q: float, identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Fully validate one durable per-q setting record before it may be reused.

    A setting JSON is the only thing that lets Phase 9D skip a q, so it is
    validated in full rather than spot-checked: identity and q, the registered
    frame count and single pass, the registered keep/drop cardinality and exact
    payload size, the finite-result flags, the expected ranker and zstd
    decompression counts, the frozen-state flags, and the complete metric and
    gate structure. Anything short of a complete, self-consistent record raises
    instead of being reused or silently remeasured.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    plan = continuous_q.quantize_q(q)

    def fail(reason: str) -> None:
        raise guards.HybridQConfigError(f"{path}: {reason}")

    if (
        document.get("schema") != SETTING_SCHEMA
        or document.get("terminal") != SETTING_TERMINAL
    ):
        fail("incomplete or foreign setting artifact")
    if document.get("run_identity_sha256") != identity["sha256"]:
        fail("run identity mismatch")
    if int(document.get("q_e4", -1)) != plan.q_e4:
        fail("q_e4 mismatch")
    if continuous_q.quantize_q(float(document.get("q", -1.0))).q_e4 != plan.q_e4:
        fail("q mismatch")
    if int(document.get("frames", -1)) != contract.VALIDATION_FRAMES:
        fail("frame count mismatch")
    if int(document.get("inference_passes_for_this_q", -1)) != 1:
        fail("inference pass count mismatch")
    if int(document.get("retained_cells", -1)) != plan.keep_count:
        fail("retained-cell count mismatch")
    if int(document.get("dropped_cells", -1)) != plan.drop_count:
        fail("dropped-cell count mismatch")

    # Payload: the pre-zstd size is exact and analytical, so every statistic of
    # it must be that one value, and the wire must have one sample per frame.
    analytical = ae_uint8_transport.analytical_size(plan.wire_q, AE_BOTTLENECK)
    payload = document.get("payload")
    if not isinstance(payload, Mapping):
        fail("no payload block")
    if int(payload.get("transported_latent_channels", -1)) != AE_BOTTLENECK:
        fail("transported latent width mismatch")
    if int(payload.get("analytical_pre_zstd_bytes", -1)) != analytical.total_bytes:
        fail("analytical pre-zstd payload size mismatch")
    pre_zstd = payload.get("pre_zstd_bytes")
    zstd = payload.get("zstd_bytes")
    if not isinstance(pre_zstd, Mapping) or not isinstance(zstd, Mapping):
        fail("no measured payload statistics")
    for name in ("mean", "median", "p95", "minimum", "maximum"):
        if float(pre_zstd.get(name, -1.0)) != float(analytical.total_bytes):
            fail(f"measured pre-zstd {name} is not the analytical payload size")
    for block, label in ((pre_zstd, "pre-zstd"), (zstd, "zstd")):
        if int(block.get("samples", -1)) != contract.VALIDATION_FRAMES:
            fail(f"{label} payload sample count mismatch")
        if not all(
            math.isfinite(float(block[name]))
            for name in ("mean", "median", "p95", "minimum", "maximum")
        ):
            fail(f"non-finite {label} payload statistic")
    if not bool(payload.get("zstd_mandatory")):
        fail("the record does not report zstd as mandatory")

    # Integrity: exactly the invocation and decompression counts one pass owes.
    integrity = document.get("integrity")
    if not isinstance(integrity, Mapping):
        fail("no integrity block")
    expected_ranker_calls = 0 if plan.is_bypass else contract.VALIDATION_FRAMES
    if int(integrity.get("ranker_invocations", -1)) != expected_ranker_calls:
        fail("ranker invocation count mismatch")
    if bool(integrity.get("q0_ranker_bypassed")) != plan.is_bypass:
        fail("q=0 ranker-bypass flag mismatch")
    if int(integrity.get("zstd_decompressions", -1)) != contract.VALIDATION_FRAMES:
        fail("zstd decompression count mismatch")
    for flag in (
        "all_outputs_finite",
        "retained_uint8_cells_equal_selected_indices",
        "dropped_cells_scattered_to_exact_zero",
        "decoder_selected_from_received_header_bytes",
        "ranges_from_complete_latent_before_dropping",
        "selection_independent_per_frame",
    ):
        if not bool(integrity.get(flag)):
            fail(f"integrity flag {flag} is not set")
    for flag in (
        "ae128_encoder_bypassed",
        "local_packet_metadata_used_for_selection",
        "reconstruction_is_identity_at_any_q",
        "batched_or_cross_frame_selection_used",
    ):
        if bool(integrity.get(flag, True)):
            fail(f"integrity flag {flag} must be false")

    # Finite-result and frozen-state declarations.
    for flag in (
        "all_outputs_and_metrics_finite",
        "frozen_perception_state_unchanged",
        "stable_ranker_state_unchanged",
        "selected_ae128_state_unchanged",
    ):
        if not bool(document.get(flag)):
            fail(f"{flag} is not set")
    for flag in ("training_or_tuning", "threshold_nms_or_gate_change", "test_or_carla_access"):
        if bool(document.get(flag, True)):
            fail(f"{flag} must be false")

    # Complete metric and gate structure.
    metrics = document.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(contract.PROTECTED_METRICS):
        fail("protected metric set is incomplete")
    canonical = document.get("canonical_person_metrics")
    if not isinstance(canonical, Mapping) or set(canonical) != set(_CSV_CANONICAL):
        fail("canonical person metric set is incomplete")
    if not all(
        math.isfinite(float(value))
        for value in list(metrics.values()) + list(canonical.values())
    ):
        fail("a recorded metric is non-finite")

    service = document.get("absolute_service_gates")
    if not isinstance(service, Mapping) or len(service.get("targets", ())) != SERVICE_GATE_COUNT:
        fail("absolute service-gate set is incomplete")
    targets = service["targets"]
    if {name for name, _target, _direction in contract.ABSOLUTE_SERVICE_TARGETS} != set(targets):
        fail("absolute service-gate names drift")
    passed = sum(1 for row in targets.values() if bool(row.get("passed")))
    if int(service.get("pass_count", -1)) != passed:
        fail("absolute service-gate pass count is inconsistent")
    if sorted(service.get("failed", ())) != sorted(
        name for name, row in targets.items() if not bool(row.get("passed"))
    ):
        fail("absolute service-gate failure list is inconsistent")

    preservation = document.get("same_q_preservation_vs_noae_uint8_zstd")
    if not isinstance(preservation, Mapping):
        fail("no same-q preservation block")
    gates = preservation.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != set(contract.PROTECTED_METRICS):
        fail("same-q preservation gate set is incomplete")
    if int(preservation.get("gates_total", -1)) != GATE_COUNT:
        fail("same-q preservation gate count drift")
    gates_passed = sum(1 for row in gates.values() if bool(row.get("passed")))
    if int(preservation.get("gates_passed", -1)) != gates_passed:
        fail("same-q preservation pass count is inconsistent")
    if bool(preservation.get("all_passed")) != (gates_passed == GATE_COUNT):
        fail("same-q preservation all-passed flag is inconsistent")
    if preservation.get("baseline") != SAME_Q_BASELINE_LABEL:
        fail("same-q preservation baseline label drift")
    deltas = document.get("protected_metric_deltas_vs_noae_uint8_zstd_same_q")
    if not isinstance(deltas, Mapping) or set(deltas) != set(contract.PROTECTED_METRICS):
        fail("protected metric delta set is incomplete")

    reference = document.get("noae_same_q_reference")
    if not isinstance(reference, Mapping):
        fail("no frozen noAE same-q reference")
    if int(reference.get("q_e4", -1)) != plan.q_e4:
        fail("the recorded noAE reference is for a different q")
    if int(reference.get("retained_cells", -1)) != plan.keep_count:
        fail("the recorded noAE reference keep count drifts")
    if not 0 <= int(reference.get("absolute_service_pass_count", -1)) <= SERVICE_GATE_COUNT:
        fail("the recorded noAE reference service-gate count is unusable")

    artifacts = document.get("prediction_artifacts")
    if not isinstance(artifacts, Mapping):
        fail("no prediction-artifact block")
    if bool(artifacts.get("removed_before_this_record", True)):
        fail("the record claims its predictions were removed before it was written")
    return document


def complete_cleanup(
    *,
    output: Path,
    q: float,
    identity: Mapping[str, Any],
    setting_sha256: str,
    prediction_root: Path,
) -> dict[str, Any]:
    """Remove one q's scratch predictions, then mark the cleanup durable.

    Called only after that q's setting JSON is durably on disk. Idempotent: a
    resume that finds the directory already gone still writes the marker, and a
    resume that finds the marker complete does not come here at all.
    """
    plan = continuous_q.quantize_q(q)
    if prediction_root.exists():
        shutil.rmtree(prediction_root)
    document = {
        "schema": CLEANUP_SCHEMA,
        "terminal": CLEANUP_TERMINAL,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "run_identity_sha256": identity["sha256"],
        "q": plan.wire_q,
        "q_e4": plan.q_e4,
        "setting_path": f"settings/{_q_slug(q)}.json",
        "setting_sha256": str(setting_sha256),
        "prediction_root": str(prediction_root),
        "prediction_artifacts_removed_after_scoring": True,
        "durability_order": list(DURABILITY_ORDER),
    }
    _atomic_json(document_path := cleanup_marker_path(output, q), document)
    return {**document, "path": str(document_path)}


def cleanup_is_complete(
    output: Path, q: float, identity: Mapping[str, Any], setting_sha256: str
) -> bool:
    """True only for a marker that belongs to exactly this durable setting."""
    path = cleanup_marker_path(output, q)
    if not path.is_file():
        return False
    document = json.loads(path.read_text(encoding="utf-8"))
    plan = continuous_q.quantize_q(q)
    return bool(
        document.get("schema") == CLEANUP_SCHEMA
        and document.get("terminal") == CLEANUP_TERMINAL
        and document.get("run_identity_sha256") == identity["sha256"]
        and int(document.get("q_e4", -1)) == plan.q_e4
        and document.get("setting_sha256") == str(setting_sha256)
        and bool(document.get("prediction_artifacts_removed_after_scoring"))
    )


def reuse_or_complete(
    *, output: Path, q: float, identity: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Resume step for one q. Measures nothing, ever.

    Returns the fully validated durable setting when one exists, having finished
    that q's cleanup if an interruption left it unfinished. Returns None only
    when no durable record exists, which is the single case in which the caller
    is allowed to run the inference/evaluation pass.
    """
    setting_path = output / "settings" / f"{_q_slug(q)}.json"
    if not setting_path.is_file():
        return None
    document = load_durable_setting(setting_path, q, identity)
    digest = sha256_file(setting_path)
    if not cleanup_is_complete(output, q, identity, digest):
        marker = complete_cleanup(
            output=output,
            q=q,
            identity=identity,
            setting_sha256=digest,
            prediction_root=output / "working_predictions" / _q_slug(q),
        )
        print(
            json.dumps(
                {
                    "completed_interrupted_cleanup_for_q": q,
                    "reran_inference": False,
                    "marker": marker["path"],
                }
            ),
            flush=True,
        )
    return document


# ---------------------------------------------------------------------------
# Payload ratios and reporting
# ---------------------------------------------------------------------------


def _payload_ratios(
    row: Mapping[str, Any], q0: Mapping[str, Any]
) -> dict[str, Any]:
    """Ratios against the three registered payload references."""
    payload = row["payload"]
    pre = payload["pre_zstd_bytes"]
    zstd = payload["zstd_bytes"]
    reference = row["noae_same_q_reference"]
    q0_payload = q0["payload"]
    framed = contract.FRAMED_Q0_PAYLOAD_BYTES
    if framed != 22020140:
        raise guards.HybridQConfigError("framed FP32 noAE q=0 reference byte drift")
    return {
        "vs_framed_fp32_noae_q0": {
            "reference_bytes": framed,
            "reference_description": (
                "framed FP32 noAE q=0 payload: 256 C2 channels, no quantization, "
                "no AE, no zstd"
            ),
            "pre_zstd_mean": pre["mean"] / framed,
            "pre_zstd_median": pre["median"] / framed,
            "zstd_mean": zstd["mean"] / framed,
            "zstd_median": zstd["median"] / framed,
            "zstd_p95": zstd["p95"] / framed,
        },
        "vs_noae_uint8_zstd_same_q": {
            "reference_pre_zstd_bytes": int(reference["pre_zstd_bytes"]),
            "reference_zstd_median_bytes": float(reference["zstd_bytes"]["median"]),
            "reference_zstd_p95_bytes": float(reference["zstd_bytes"]["p95"]),
            "reference_publishes_mean_zstd_bytes": False,
            "pre_zstd_mean": pre["mean"] / float(reference["pre_zstd_bytes"]),
            "pre_zstd_median": pre["median"] / float(reference["pre_zstd_bytes"]),
            "zstd_median": zstd["median"] / float(reference["zstd_bytes"]["median"]),
            "zstd_p95": zstd["p95"] / float(reference["zstd_bytes"]["p95"]),
        },
        "vs_ae128_uint8_zstd_q0": {
            "reference_pre_zstd_bytes": float(q0_payload["pre_zstd_bytes"]["mean"]),
            "reference_zstd_mean_bytes": float(q0_payload["zstd_bytes"]["mean"]),
            "reference_zstd_median_bytes": float(q0_payload["zstd_bytes"]["median"]),
            "pre_zstd_mean": pre["mean"] / float(q0_payload["pre_zstd_bytes"]["mean"]),
            "zstd_mean": zstd["mean"] / float(q0_payload["zstd_bytes"]["mean"]),
            "zstd_median": zstd["median"] / float(q0_payload["zstd_bytes"]["median"]),
        },
    }


_CSV_METRICS = tuple(contract.PROTECTED_METRICS)
_CSV_CANONICAL = ("person_precision", "person_recall", "person_f1", "person_xy_mae_m")


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = [
        "q", "q_e4", "retained_cells", "pre_zstd_bytes_mean", "pre_zstd_bytes_median",
        "pre_zstd_bytes_p95", "zstd_bytes_mean", "zstd_bytes_median", "zstd_bytes_p95",
        "zstd_bytes_min", "zstd_bytes_max",
        "zstd_ratio_vs_framed_fp32_noae_q0_median",
        "zstd_ratio_vs_noae_uint8_zstd_same_q_median",
        "zstd_ratio_vs_ae128_uint8_zstd_q0_median",
        *_CSV_METRICS, *_CSV_CANONICAL,
        "absolute_service_pass_count", "failed_absolute_service_gates",
        "same_q_preservation_pass_count", "same_q_preservation_all_passed",
        "failed_same_q_preservation_gates", "worst_normalized_degradation",
        "profile_designation", "influences_acceptance", "all_outputs_finite",
    ]
    columns += [f"degradation_{name}" for name in _CSV_METRICS]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        payload = row["payload"]
        ratios = row["payload_ratios"]
        preservation = row["same_q_preservation_vs_noae_uint8_zstd"]
        deltas = row["protected_metric_deltas_vs_noae_uint8_zstd_same_q"]
        writer.writerow(
            {
                "q": f"{row['q']:.2f}",
                "q_e4": row["q_e4"],
                "retained_cells": row["retained_cells"],
                "pre_zstd_bytes_mean": payload["pre_zstd_bytes"]["mean"],
                "pre_zstd_bytes_median": payload["pre_zstd_bytes"]["median"],
                "pre_zstd_bytes_p95": payload["pre_zstd_bytes"]["p95"],
                "zstd_bytes_mean": payload["zstd_bytes"]["mean"],
                "zstd_bytes_median": payload["zstd_bytes"]["median"],
                "zstd_bytes_p95": payload["zstd_bytes"]["p95"],
                "zstd_bytes_min": payload["zstd_bytes"]["minimum"],
                "zstd_bytes_max": payload["zstd_bytes"]["maximum"],
                "zstd_ratio_vs_framed_fp32_noae_q0_median": ratios[
                    "vs_framed_fp32_noae_q0"
                ]["zstd_median"],
                "zstd_ratio_vs_noae_uint8_zstd_same_q_median": ratios[
                    "vs_noae_uint8_zstd_same_q"
                ]["zstd_median"],
                "zstd_ratio_vs_ae128_uint8_zstd_q0_median": ratios[
                    "vs_ae128_uint8_zstd_q0"
                ]["zstd_median"],
                **{name: row["metrics"][name] for name in _CSV_METRICS},
                **{
                    name: row["canonical_person_metrics"][name]
                    for name in _CSV_CANONICAL
                },
                "absolute_service_pass_count": row["absolute_service_gates"][
                    "pass_count"
                ],
                "failed_absolute_service_gates": ";".join(
                    row["absolute_service_gates"]["failed"]
                ),
                "same_q_preservation_pass_count": preservation["gates_passed"],
                "same_q_preservation_all_passed": preservation["all_passed"],
                "failed_same_q_preservation_gates": ";".join(preservation["failed"]),
                "worst_normalized_degradation": preservation[
                    "worst_normalized_degradation"
                ],
                "profile_designation": row["profile_status"]["designation"],
                "influences_acceptance": row["profile_status"]["influences_acceptance"],
                "all_outputs_finite": row["all_outputs_and_metrics_finite"],
                **{
                    f"degradation_{name}": deltas[name]["degradation"]
                    for name in _CSV_METRICS
                },
            }
        )
    return stream.getvalue()


def _report_text(document: Mapping[str, Any]) -> str:
    rows = document["curve"]
    acceptance = document["preregistered_interpretation"]
    lines = [
        "# Phase 9D — selected AE128 UINT8 + mandatory-zstd validation",
        "",
        f"Generated {document['generated_utc']} · terminal `{TERMINAL}`",
        "",
        "One frozen measurement of the six registered q anchors on the "
        f"{contract.VALIDATION_FRAMES:,} registered validation frames, one "
        "inference/evaluation pass per q. Nothing was trained, tuned, "
        "recalibrated or removed; no threshold, NMS setting, scorer or geometry "
        "evaluator changed; test data and CARLA were never opened. Component "
        "latency below is current-host diagnostic evidence only — no Raspberry "
        "Pi and no OAI latency is claimed.",
        "",
        "## Deployment path measured",
        "",
        "```text",
        "original FP32 C2 -> AE128 encoder (complete frame) -> per-channel UINT8",
        "  -> sparse AE wire -> mandatory zstd-1 -> received raw bytes",
        "  -> exactly one decompression -> header-driven AE128 decoder selection",
        "  -> dequantize / zero scatter -> AE128 decoder -> frozen perception tail",
        "```",
        "",
        f"Selected checkpoint `{document['selected_ae128']['selected_checkpoint_path']}`",
        f"(sha256 `{document['selected_ae128']['selected_checkpoint_sha256']}`), "
        f"routing tag `{document['selected_ae128']['routing_tag_hex']}` derived from "
        "that full digest. The 32-bit tag routes a frame to the decoder that "
        "produced it; it is not the checkpoint's identity.",
        "",
        "## Payload",
        "",
        "| q | keep | pre-zstd mean B | median | p95 | zstd mean B | median | p95 | "
        "vs framed FP32 noAE q0 | vs noAE UINT8+zstd same q | vs AE128 UINT8+zstd q0 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        payload = row["payload"]
        ratios = row["payload_ratios"]
        lines.append(
            f"| {row['q']:.2f} | {row['retained_cells']:,} | "
            f"{payload['pre_zstd_bytes']['mean']:,.0f} | "
            f"{payload['pre_zstd_bytes']['median']:,.0f} | "
            f"{payload['pre_zstd_bytes']['p95']:,.0f} | "
            f"{payload['zstd_bytes']['mean']:,.0f} | "
            f"{payload['zstd_bytes']['median']:,.0f} | "
            f"{payload['zstd_bytes']['p95']:,.0f} | "
            f"{ratios['vs_framed_fp32_noae_q0']['zstd_median']:.6f} | "
            f"{ratios['vs_noae_uint8_zstd_same_q']['zstd_median']:.6f} | "
            f"{ratios['vs_ae128_uint8_zstd_q0']['zstd_median']:.6f} |"
        )
    lines += [
        "",
        "Ratios use median bytes on both sides. The frozen noAE UINT8+zstd "
        "reference publishes no mean compressed size, so no mean-vs-mean ratio "
        "against it is reported.",
        "",
        "## Accuracy",
        "",
        "| q | vehicle P/R/F1/XY | canonical-p025 person P/R/F1/XY | "
        "AVO>=0.65 person P/R/F1/XY | person 20–40 m recall | vehicle IoU | "
        "person box-mask IoU | foreground mIoU | service gates | same-q gates | "
        "profile |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        m = row["metrics"]
        c = row["canonical_person_metrics"]
        preservation = row["same_q_preservation_vs_noae_uint8_zstd"]
        lines.append(
            f"| {row['q']:.2f} | {m['vehicle_precision']:.6f}/"
            f"{m['vehicle_recall']:.6f}/{m['vehicle_f1']:.6f}/"
            f"{m['vehicle_xy_mae_m']:.6f} | "
            f"{c['person_precision']:.6f}/{c['person_recall']:.6f}/"
            f"{c['person_f1']:.6f}/{c['person_xy_mae_m']:.6f} | "
            f"{m['person_avo_precision']:.6f}/{m['person_avo_recall']:.6f}/"
            f"{m['person_avo_f1']:.6f}/{m['person_avo_xy_mae_m']:.6f} | "
            f"{m['person_avo_recall_20_40m']:.6f} | {m['vehicle_iou']:.6f} | "
            f"{m['person_box_mask_iou']:.6f} | {m['foreground_miou']:.6f} | "
            f"{row['absolute_service_gates']['pass_count']}/{SERVICE_GATE_COUNT} | "
            f"{preservation['gates_passed']}/{GATE_COUNT} | "
            f"{row['profile_status']['designation']} |"
        )
    lines += [
        "",
        "## Failed gates and exact degradations",
        "",
        "| q | failed same-q preservation gates | degradation / bound | "
        "failed absolute service gates |",
        "| ---: | --- | --- | --- |",
    ]
    for row in rows:
        preservation = row["same_q_preservation_vs_noae_uint8_zstd"]
        failed = preservation["failed"]
        detail = (
            "; ".join(
                f"{name} {preservation['gates'][name]['degradation']:+.6f} / "
                f"{preservation['gates'][name]['bound']}"
                for name in failed
            )
            or "—"
        )
        lines.append(
            f"| {row['q']:.2f} | {', '.join(failed) or '—'} | {detail} | "
            f"{', '.join(row['absolute_service_gates']['failed']) or '—'} |"
        )
    lines += [
        "",
        "## Preregistered interpretation",
        "",
        acceptance["rule"],
        "",
        f"- q=0 condition: **{'met' if acceptance['q0_condition']['passed'] else 'not met'}** "
        f"({acceptance['q0_condition']['preservation_gates_passed']}/{GATE_COUNT} "
        "same-q gates, "
        f"{acceptance['q0_condition']['absolute_service_pass_count']}/"
        f"{SERVICE_GATE_COUNT} absolute service gates against a "
        f"{BASELINE_SERVICE_PASS_COUNT}/{SERVICE_GATE_COUNT} baseline)",
        f"- qualifying primary q: "
        f"{acceptance['qualifying_primary_q'] or 'none'}",
        f"- **decision: {acceptance['decision']}**",
        "- q=0.90 and q=0.98 are stress/emergency profiles regardless of their "
        "results and did not enter the decision",
        "",
        "## Integrity",
        "",
        f"- validation frames per q: {contract.VALIDATION_FRAMES:,}",
        f"- q settings completed exactly once: {len(rows)}/{len(Q_VALUES)}",
        "- every frame carried the AE128 family id, a 128-channel latent and the "
        "bound routing tag in its own header",
        "- every frame was decompressed exactly once, and the decoder was "
        "discovered from the received header bytes alone",
        "- retained UINT8 cells were exactly the selected cells; dropped cells "
        "scattered to exact zero before reconstruction",
        "- q=0 invoked the ranker zero times and AE128 every time; no q produced "
        "an identity reconstruction",
        "- frozen perception, stable ranker and selected AE128 parameters and "
        "buffers were unchanged",
        "- per q the setting JSON was fsynced into place first, its predictions "
        "were removed only afterwards, and the cleanup marker was written last, "
        "so an interruption could only lose scratch predictions",
        "- only compact evidence is retained; no prediction directory survives",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Run identity and finalization
# ---------------------------------------------------------------------------


def _run_identity(binding: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        "schema": SCHEMA,
        "q_e4": [continuous_q.quantize_q(q).q_e4 for q in Q_VALUES],
        "validation_frames": contract.VALIDATION_FRAMES,
        "selected_checkpoint_sha256": SELECTED_CHECKPOINT_SHA256,
        "holdout_decision_sha256": HOLDOUT_DECISION_SHA256,
        "noae_reference_sha256": NOAE_UINT8_VALIDATION_SHA256,
        "routing_tag": routing_tag(),
        "acceptance_rule": ACCEPTANCE_RULE,
        "binding": dict(binding),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    return {**identity, "sha256": _identity_digest(identity)}


def finalize(
    *,
    output: Path,
    rows: list[dict[str, Any]],
    binding: Mapping[str, Any],
    identity: Mapping[str, Any],
    runtime: Mapping[str, Any],
    decision: Mapping[str, Any],
    scorers: Any,
    default_cpu_threads: int,
    started: float,
) -> dict[str, Any]:
    q0 = rows[0]
    if continuous_q.quantize_q(float(q0["q"])).q_e4 != 0:
        raise guards.HybridQConfigError("the first completed setting is not q=0")
    for row in rows:
        row["payload_ratios"] = _payload_ratios(row, q0)
    acceptance = evaluate_acceptance([acceptance_inputs(row) for row in rows])
    _require_state_unchanged(runtime)

    document = {
        "schema": SCHEMA,
        "terminal": TERMINAL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "measure the selected AE128 on the real UINT8 + mandatory-zstd "
            "deployment path at the six registered q anchors, and compare each "
            "row with the frozen noAE UINT8+zstd validation result at the same q "
            "so the reported degradation isolates the AE128 latent transport"
        ),
        "scope": {
            "validation_frames_per_q": contract.VALIDATION_FRAMES,
            "validation_episodes": list(contract.VALIDATION_EPISODES),
            "q_values": list(Q_VALUES),
            "stress_q_values": list(STRESS_Q_VALUES),
            "completed_settings": len(rows),
            "inference_passes_per_q": 1,
            "family": ae_contract.family_name(AE_FAMILY_ID),
            "transported_latent_channels": AE_BOTTLENECK,
            "training_or_tuning": False,
            "threshold_nms_calibration_or_geometry_change": False,
            "scorer_change": False,
            "test_accessed": False,
            "carla_launched": False,
            "prediction_directories_retained": False,
        },
        "run_identity": dict(identity),
        "binding": dict(binding),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(runtime["device"]),
            "torch_cpu_threads": TORCH_CPU_THREADS,
            "torch_default_cpu_threads": default_cpu_threads,
            "inference_precision": "fp32 inference_mode, no autocast",
        },
        "perception_binding": runtime["perception"],
        "selected_ae128": runtime["autoencoder_provenance"],
        "holdout_selection_decision": dict(decision),
        "transport": {
            "pipeline": [
                "original FP32 C2",
                "selected AE128 encoder on the complete frame",
                "per-channel UINT8 latent quantization, ranges from the complete "
                "latent before q dropping",
                "sparse AE latent wire",
                "mandatory zstd level 1",
                "received raw bytes",
                "exactly one zstd decompression",
                "header-driven preloaded AE128 decoder selection",
                "UINT8 dequantization and zero scatter",
                "AE128 decoder",
                "unchanged frozen perception tail",
            ],
            "q0_bypasses_ranker": True,
            "q0_bypasses_ae128": False,
            "ranking_input": "original FP32 C2, independently per frame",
            "ranges": "per frame/channel from the complete AE latent before dropping",
            "zstd": implementation_report(),
            "zstd_mandatory": True,
            "zstd_level_tuned_here": False,
            "codec_logic_duplicated_here": False,
            "snap_continuous_q_called": False,
        },
        "same_q_reference": {
            "path": NOAE_UINT8_VALIDATION_RELPATH,
            "sha256": NOAE_UINT8_VALIDATION_SHA256,
            "description": SAME_Q_BASELINE_LABEL,
            "preservation_gates": [
                {"metric": name, "direction": direction, "bound": bound}
                for name, direction, bound in contract.HOLDOUT_PRESERVATION_GATES
            ],
            "absolute_service_targets": [
                {"metric": name, "target": target, "direction": direction}
                for name, target, direction in contract.ABSOLUTE_SERVICE_TARGETS
            ],
        },
        "service_pipeline": {
            "p025_service_policy": "unchanged",
            "vehicle_score_threshold": contract.VEHICLE_SCORE_THRESHOLD,
            "person_service_score_threshold": contract.PERSON_SERVICE_SCORE_THRESHOLD,
            "person_avo_threshold": contract.PERSON_AVO_THRESHOLD,
            "scorer_sha256": scorers.sha256,
            "unchanged": True,
        },
        "payload_references": {
            "framed_fp32_noae_q0_bytes": contract.FRAMED_Q0_PAYLOAD_BYTES,
            "noae_uint8_zstd_same_q": NOAE_UINT8_VALIDATION_RELPATH,
            "ae128_uint8_zstd_q0": "this run, q=0 row",
        },
        "preregistered_interpretation": acceptance,
        "curve": rows,
        "settings": {
            _q_slug(row["q"]): {
                "path": f"settings/{_q_slug(row['q'])}.json",
                "sha256": sha256_file(output / "settings" / f"{_q_slug(row['q'])}.json"),
            }
            for row in rows
        },
        "durability": {
            "order_per_q": list(DURABILITY_ORDER),
            "setting_json_is_the_completion_record": True,
            "predictions_removed_only_after_the_setting_is_durable": True,
            "cleanup_markers": {
                _q_slug(row["q"]): {
                    "path": f"cleanup/{_q_slug(row['q'])}.json",
                    "terminal": CLEANUP_TERMINAL,
                    "sha256": sha256_file(cleanup_marker_path(output, float(row["q"]))),
                }
                for row in rows
            },
        },
        "integrity": {
            "zstd_decompressions": sum(
                int(row["integrity"]["zstd_decompressions"]) for row in rows
            ),
            "required_zstd_decompressions": len(rows) * contract.VALIDATION_FRAMES,
            "all_outputs_and_metrics_finite": all(
                row["all_outputs_and_metrics_finite"] for row in rows
            ),
            "q0_ranker_bypassed": rows[0]["integrity"]["ranker_invocations"] == 0,
            "ae128_never_bypassed": all(
                not row["integrity"]["ae128_encoder_bypassed"] for row in rows
            ),
            "decoder_always_selected_from_header": all(
                row["integrity"]["decoder_selected_from_received_header_bytes"]
                for row in rows
            ),
            "frozen_perception_state_unchanged": True,
            "stable_ranker_state_unchanged": True,
            "selected_ae128_state_unchanged": True,
            "every_q_has_a_durable_setting_and_cleanup_marker": all(
                cleanup_is_complete(
                    output,
                    float(row["q"]),
                    identity,
                    sha256_file(output / "settings" / f"{_q_slug(row['q'])}.json"),
                )
                for row in rows
            ),
        },
        "wall_seconds_this_invocation": time.time() - started,
    }
    integrity = document["integrity"]
    if integrity["zstd_decompressions"] != integrity["required_zstd_decompressions"]:
        raise guards.HybridQPayloadError("final zstd decompression count drift")
    if not integrity["all_outputs_and_metrics_finite"]:
        raise guards.HybridQNumericalError("final result contains a non-finite row")
    if not integrity["every_q_has_a_durable_setting_and_cleanup_marker"]:
        raise guards.HybridQConfigError(
            "a completed q is missing its durable setting or its cleanup marker"
        )

    _atomic_json(output / "phase9d_ae128_uint8_validation.json", document)
    _atomic_write(output / "phase9d_ae128_uint8_validation.csv", _csv_text(rows))
    _atomic_write(
        output / "AE_PHASE9D_UINT8_VALIDATION_REPORT.md", _report_text(document)
    )
    _atomic_write(output / TERMINAL, f"{TERMINAL} {document['generated_utc']}\n")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase-9D selected AE128 UINT8 + mandatory-zstd validation"
    )
    parser.add_argument("--execute", required=True, choices=(EXECUTE_TOKEN,))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=DATALOADER_WORKERS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Phase 9D requires the qualified CUDA runtime")
    device = torch.device("cuda:0")
    torch.manual_seed(contract.RANKER_INIT_SEED)
    default_cpu_threads = torch.get_num_threads()
    torch.set_num_threads(TORCH_CPU_THREADS)
    started = time.time()

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    binding = bind_inputs()
    decision = load_holdout_decision(binding)
    references = load_noae_reference()
    identity = _run_identity(binding)

    manifest_path = output / "run_manifest.json"
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_identity": identity,
        "q_values": list(Q_VALUES),
        "acceptance_rule": ACCEPTANCE_RULE,
        "resume_rule": "skip an exactly complete setting; rerun only an unfinished q",
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("run_identity") != identity:
            raise guards.HybridQConfigError("existing run manifest identity drift")
    else:
        _atomic_json(manifest_path, manifest)

    settings_dir = output / "settings"
    work_dir = output / "working_predictions"
    settings_dir.mkdir(exist_ok=True)
    work_dir.mkdir(exist_ok=True)

    runtime = dict(_load_runtime(device))
    autoencoder, provenance = load_selected_ae(device, binding)
    decoders = ae_family_dispatch.PreloadedAeDecoders([autoencoder])
    if decoders.families != (AE_FAMILY_ID,):
        raise guards.HybridQConfigError("exactly one preloaded AE128 decoder is expected")
    runtime.update(
        {
            "autoencoder": autoencoder,
            "autoencoder_provenance": provenance,
            "decoders": decoders,
            "model_snapshot": guards.snapshot_module_state(runtime["model"]),
            "ranker_snapshot": guards.snapshot_module_state(runtime["ranker"]),
            "autoencoder_snapshot": guards.snapshot_module_state(autoencoder),
        }
    )

    scorers = load_frozen_scorers()
    gt, _gt_states = scorers.load_gt(runtime["dataset_root"], contract.PRIMARY_CONTRACT)
    validation_gt = {
        sample_id: gt.get(sample_id, []) for sample_id in runtime["frame_ids"]
    }
    person_gt = _person_only(validation_gt)
    ignore_cache: dict[str, Any] = {}
    wire = CountingWireCodec()
    completed_rows: list[dict[str, Any]] = []

    for q in Q_VALUES:
        slug = _q_slug(q)
        setting_path = settings_dir / f"{slug}.json"
        # A q with a valid durable record is never remeasured; at most its
        # interrupted cleanup is finished.
        reused = reuse_or_complete(output=output, q=q, identity=identity)
        if reused is not None:
            completed_rows.append(reused)
            print(
                json.dumps({"reused_completed_q": q, "setting": str(setting_path)}),
                flush=True,
            )
            continue

        prediction_root = work_dir / slug
        if prediction_root.exists():
            shutil.rmtree(prediction_root)
        raw = run_validation_pass(
            runtime=runtime,
            q=q,
            output=prediction_root,
            workers=int(args.workers),
            wire=wire,
        )
        _require_state_unchanged(runtime)
        scored = score_validation_pass(
            result=raw,
            scorers=scorers,
            truth=runtime["truth"],
            experiment=runtime["dataset_root"],
            frame_ids=runtime["frame_ids"],
            gt=validation_gt,
            person_gt=person_gt,
            ignore_cache=ignore_cache,
        )
        reference = references[continuous_q.quantize_q(q).q_e4]
        setting = _setting_document(
            raw=raw, scored=scored, reference=reference, identity=identity
        )
        # The scientific completion record goes down first, fsynced into place,
        # and is then re-read through the same validator the resume path uses,
        # so the in-memory row is exactly the durable bytes.
        digest = _atomic_json(setting_path, setting)
        setting = load_durable_setting(setting_path, q, identity)
        # Only now: drop the scratch predictions, then mark the cleanup durable.
        complete_cleanup(
            output=output,
            q=q,
            identity=identity,
            setting_sha256=digest,
            prediction_root=prediction_root,
        )
        completed_rows.append(setting)
        preservation = setting["same_q_preservation_vs_noae_uint8_zstd"]
        print(
            json.dumps(
                {
                    "completed_q": q,
                    "frames": setting["frames"],
                    "zstd_bytes_median": setting["payload"]["zstd_bytes"]["median"],
                    "absolute_service_gates": setting["absolute_service_gates"][
                        "pass_count"
                    ],
                    "same_q_preservation_gates": preservation["gates_passed"],
                    "failed_same_q_preservation_gates": preservation["failed"],
                    "setting": str(setting_path),
                    "sha256": digest,
                }
            ),
            flush=True,
        )

    if [row["q_e4"] for row in completed_rows] != [
        continuous_q.quantize_q(q).q_e4 for q in Q_VALUES
    ]:
        raise guards.HybridQConfigError("completed q order drift")
    _require_state_unchanged(runtime)
    document = finalize(
        output=output,
        rows=completed_rows,
        binding=binding,
        identity=identity,
        runtime=runtime,
        decision=decision,
        scorers=scorers,
        default_cpu_threads=default_cpu_threads,
        started=started,
    )
    if work_dir.exists() and not any(work_dir.iterdir()):
        work_dir.rmdir()
    acceptance = document["preregistered_interpretation"]
    print(
        json.dumps(
            {
                "terminal": TERMINAL,
                "output": str(output),
                "settings": len(document["curve"]),
                "decision": acceptance["decision"],
                "accepted": acceptance["accepted"],
                "qualifying_primary_q": acceptance["qualifying_primary_q"],
                "all_finite": document["integrity"]["all_outputs_and_metrics_finite"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(TERMINAL)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
