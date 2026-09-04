"""Phase 11D: one resumable UINT6/UINT4 deployment validation catalog.

This runner is deliberately inert unless its exact ``--execute`` token is
provided.  Its fixed catalog is the 48 settings produced by four wire families,
two low-bit widths and the six registered q anchors.  A future execution runs
each setting once over the registered 3,345 validation frames through the
public encode -> zstd-1 -> raw-byte receive -> frozen-tail path.  It neither
trains nor changes a threshold, scorer, checkpoint, wire format, zstd
implementation or action set.

The source deliberately composes the completed Phase-8B/9D/10B scorers and
classification helpers.  It does not reimplement selection, low-bit packing,
quantization, p025, matching, AVO, segmentation, preservation gates or service
gates.  ``PreloadedLowBitDecoders`` owns the receiver-configured tail device;
no packet byte is ever an output-device selector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import time
from dataclasses import dataclass
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
    sha256_file,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.phase5_common import (
    load_frozen_scorers,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.phase6_validation import (
    _collate,
    _person_only,
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
from . import ae_contract, ae_phase10b_uint8_validation as phase10b
from . import ae_training_common as common
from . import ae_uint8_validation as phase9d
from . import lowbit_dispatch, lowbit_transport
from . import ae_phase11b_gpu_qualification as phase11b
from .ae_gpu_qualification import CountingWireCodec
from .ae_model import SplitFeatureAE


EXECUTE_TOKEN = "SPLITFUSION_LOWBIT_PHASE11D_VALIDATION"
TERMINAL = "SPLITFUSION_LOWBIT_PHASE11D_VALIDATION_COMPLETE"
SETTING_TERMINAL = "SPLITFUSION_LOWBIT_PHASE11D_SETTING_COMPLETE"
CLEANUP_TERMINAL = "SPLITFUSION_LOWBIT_PHASE11D_SETTING_CLEANUP_COMPLETE"
SCHEMA = "splitfusion_fcos_phase11d_lowbit_validation_v1"
SETTING_SCHEMA = "splitfusion_fcos_phase11d_lowbit_setting_v1"
CLEANUP_SCHEMA = "splitfusion_fcos_phase11d_lowbit_cleanup_v1"

SETTINGS_DIRNAME = "settings"
CLEANUP_DIRNAME = "cleanup"
WORKING_DIRNAME = "working_predictions"
DATALOADER_WORKERS = 8
INFERENCE_BATCH = 8
ZSTD_LEVEL = 1
BIT_WIDTHS = (6, 4)
Q_VALUES = tuple(float(q) for q in contract.REGISTERED_Q_VALUES)
STRESS_Q_VALUES = tuple(float(q) for q in contract.EVALUATION_STRESS_Q_VALUES)

DURABILITY_ORDER = (
    "atomically write settings/<family>_<bits>_<q>.json",
    "remove working_predictions/<family>_<bits>_<q> only after that record",
    "atomically write cleanup/<family>_<bits>_<q>.json",
)


@dataclass(frozen=True)
class Family:
    name: str
    family_id: int
    bottleneck: int | None

    @property
    def channels(self) -> int:
        return contract.SPLIT_CHANNELS if self.bottleneck is None else self.bottleneck


@dataclass(frozen=True)
class Setting:
    family: Family
    bit_width: int
    q: float

    @property
    def q_e4(self) -> int:
        return continuous_q.quantize_q(self.q).q_e4

    @property
    def key(self) -> str:
        return f"{self.family.name.lower()}_uint{self.bit_width}_{_q_slug(self.q)}"


FAMILIES = (
    Family("noAE", ae_contract.AE_FAMILY_NOAE, None),
    Family("AE128", ae_contract.AE_FAMILY_AE128, 128),
    Family("AE64", ae_contract.AE_FAMILY_AE64, 64),
    Family("AE32", ae_contract.AE_FAMILY_AE32, 32),
)
FAMILY_BY_NAME = {family.name: family for family in FAMILIES}


def catalog() -> tuple[Setting, ...]:
    """The sole registered matrix; there are no CLI matrix overrides."""
    rows = tuple(
        Setting(family, bits, q)
        for family in FAMILIES
        for bits in BIT_WIDTHS
        for q in Q_VALUES
    )
    if len(rows) != 48 or len({row.key for row in rows}) != 48:
        raise guards.HybridQConfigError("Phase-11D catalog is not exactly 48 unique settings")
    return rows


CATALOG = catalog()


@dataclass(frozen=True)
class Uint8Reference:
    family: str
    relative: str
    sha256: str
    schema: str
    terminal: str
    checkpoint_sha256: str | None
    selection_sha256: str | None


# Every reference has completed independently.  The exact result digest, and
# the selected AE digest where relevant, are the admission rule--not a filename.
UINT8_REFERENCES = {
    "noAE": Uint8Reference(
        "noAE",
        "experiments/splitfusion_fcos_hybrid_q_v1/"
        "20260902_223610_phase8b_uint8_validation/phase8b_uint8_validation.json",
        "a2779f5fb0a585b1c317dc755b5ab577fa7c34963ab7945cb704e0d4146bb029",
        "splitfusion_fcos_hybrid_q_phase8b_uint8_validation_v1",
        "HYBRID_Q_UINT8_VALIDATION_COMPLETE",
        None,
        None,
    ),
    "AE128": Uint8Reference(
        "AE128",
        "experiments/splitfusion_fcos_ae_v1/"
        "20260903_phase9d_ae128_uint8_validation/phase9d_ae128_uint8_validation.json",
        "89cc7c706fc3383106a5680d3d54d5fb514dcd5d808c13d9eaf1a2c380785963",
        "splitfusion_fcos_ae128_phase9d_uint8_validation_v1",
        "SPLITFUSION_AE128_UINT8_VALIDATION_COMPLETE",
        phase11b.FROZEN_INPUTS["AE128"]["sha256"],
        phase11b.FROZEN_INPUTS["AE128"]["selection_sha256"],
    ),
    "AE64": Uint8Reference(
        "AE64",
        "experiments/splitfusion_fcos_ae_v1/"
        "20260903_phase10b_ae64_uint8_validation/phase10b_ae64_uint8_validation.json",
        "eead1786a5d12294b9d61d9271431049aac28540a20f2c4608db33ab66de3aad",
        "splitfusion_fcos_ae64_phase10b_uint8_validation_v1",
        "SPLITFUSION_AE64_PHASE10B_UINT8_VALIDATION_COMPLETE",
        phase11b.FROZEN_INPUTS["AE64"]["sha256"],
        phase11b.FROZEN_INPUTS["AE64"]["selection_sha256"],
    ),
    "AE32": Uint8Reference(
        "AE32",
        "experiments/splitfusion_fcos_ae_v1/"
        "20260903_phase10b_ae32_uint8_validation/phase10b_ae32_uint8_validation.json",
        "db4b944eb5992a82e9fe6b0befd2e3bcf583629a6f27ad2ad6cb4075f03a90ec",
        "splitfusion_fcos_ae32_phase10b_uint8_validation_v1",
        "SPLITFUSION_AE32_PHASE10B_UINT8_VALIDATION_COMPLETE",
        phase11b.FROZEN_INPUTS["AE32"]["sha256"],
        phase11b.FROZEN_INPUTS["AE32"]["selection_sha256"],
    ),
}

PHASE11B_ARTIFACTS = {
    "report": (
        "experiments/splitfusion_fcos_ae_v1/"
        "20260903_phase11b_lowbit_gpu_qualification/phase11b_lowbit_gpu_qualification.json",
        "379aa07148e3e47384cfbebbe0ede5990c07f11b8a4bdef056d6a533cee5fc01",
    ),
    "terminal": (
        "experiments/splitfusion_fcos_ae_v1/"
        "20260903_phase11b_lowbit_gpu_qualification/"
        "SPLITFUSION_LOWBIT_PHASE11B_GPU_QUALIFIED",
        "83f41560a3327c4207834f5725e5e313ceb6b3e0f9e22ea1f8c37b6dcf0b56e2",
    ),
}
PHASE11C_ARTIFACTS = {
    "report": (
        "experiments/splitfusion_fcos_ae_v1/"
        "20260903_phase11c_zstd_level_sweep/phase11c_zstd_level_sweep.json",
        "4bcd2eddff502cc55d799bfdf5af920ccc1378ae87bd2c0d6017d3d2586c7b2d",
    ),
    "terminal": (
        "experiments/splitfusion_fcos_ae_v1/"
        "20260903_phase11c_zstd_level_sweep/"
        "HYBRID_Q_PHASE11C_ZSTD_LEVEL_SWEEP_COMPLETE",
        "198585a7384382b6c858a4f17595b132b008e7ef8f9a58d2c595032acdb1b5c2",
    ),
}
FP32_Q0_REFERENCE = {
    "path": (
        "experiments/splitfusion_fcos_hybrid_q_v1/"
        "20260902_182401_phase6_validation_curve/validation_curve.json"
    ),
    "sha256": "54987920a7430564425664e82511d1121e77935beabfbd4cf2f34bee5cadfc74",
}
P025_FORWARD_LOCK_SHA256 = "86d6f13ae9168b33b697df5b785c5f7c320afc52cfdcded5b632d94a6d943fe1"

# The frozen historical phase validates the full checkpoint source maps.  These
# are the exact current files this new deployment execution imports for the
# low-bit path.  They are byte locks, never a filename-only allowlist.
LIVE_LOWBIT_EXECUTION_SOURCES = {
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_ae_v1/lowbit_transport.py": (
        "c708389982b10968978002d2d8423984d857229021149d51ec0d135619d69f12"
    ),
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_ae_v1/lowbit_dispatch.py": (
        "8e25fa64feeb22d4315957faaa216e0ea672170f09250cdc2930d705a621b0ce"
    ),
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1/zstd_transport.py": (
        "57d1846b3fdc4084266e5a8adcc7abf99556ed2b187befec729003fcdb77edec"
    ),
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_ae_v1/ae_composition.py": (
        "3e54ea2dc8599c0579349e4ce430a34a2d2343764402aee5113e1c2d54dfa608"
    ),
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_ae_v1/ae_model.py": (
        "4f2ec0a69cf127ebac44f6147124705d54785ab400ed033a83ad1e339735b542"
    ),
}


def _repository_path(relative: str) -> Path:
    return (contract.repository_root() / relative).resolve(strict=True)


def _require_hash(relative: str, expected: str) -> dict[str, str]:
    digest = sha256_file(_repository_path(relative))
    if digest != expected:
        raise guards.HybridQConfigError(f"Phase-11D provenance hash drift: {relative}")
    return {"path": relative, "sha256": digest}


def _selected_checkpoint(document: Mapping[str, Any], family: str) -> str | None:
    if family == "noAE":
        return None
    block = document.get("selected_ae128") if family == "AE128" else document.get("selected_ae")
    if not isinstance(block, Mapping):
        raise guards.HybridQConfigError(f"{family} UINT8 reference lacks selected AE")
    return str(block.get("selected_checkpoint_sha256", ""))


def _extract_reference_rows(
    spec: Uint8Reference, document: Mapping[str, Any]
) -> dict[int, dict[str, Any]]:
    """Normalize one completed UINT8 result without recalculating it."""
    if document.get("schema") != spec.schema or document.get("terminal") != spec.terminal:
        raise guards.HybridQConfigError(f"{spec.family} UINT8 reference is incomplete")
    scope = document.get("scope")
    rows = document.get("curve")
    if not isinstance(scope, Mapping) or not isinstance(rows, list):
        raise guards.HybridQConfigError(f"{spec.family} UINT8 reference structure drift")
    if int(scope.get("validation_frames_per_q", -1)) != contract.VALIDATION_FRAMES:
        raise guards.HybridQConfigError(f"{spec.family} UINT8 reference frame count drift")
    if int(scope.get("inference_passes_per_q", -1)) != 1:
        raise guards.HybridQConfigError(f"{spec.family} UINT8 reference one-pass drift")
    if _selected_checkpoint(document, spec.family) != spec.checkpoint_sha256:
        raise guards.HybridQConfigError(f"{spec.family} UINT8 reference checkpoint drift")
    binding = document.get("binding")
    if not isinstance(binding, Mapping):
        raise guards.HybridQConfigError(f"{spec.family} UINT8 reference lacks binding")
    perception = binding.get("frozen_perception_checkpoint", binding.get("frozen_checkpoint"))
    ranker = binding.get("stable_epoch4_ranker", binding.get("stable_epoch4_ranker"))
    forward = binding.get("perception_forward_lock", binding.get("p025_forward_lock"))
    if not all(isinstance(item, Mapping) for item in (perception, ranker, forward)):
        raise guards.HybridQConfigError(f"{spec.family} UINT8 frozen scoring binding drift")
    if (
        perception.get("sha256") != phase11b.FROZEN_INPUTS["perception"]["sha256"]
        or ranker.get("sha256") != phase11b.FROZEN_INPUTS["ranker"]["sha256"]
    ):
        raise guards.HybridQConfigError(f"{spec.family} UINT8 perception/ranker drift")
    # The noAE record calls it a perception forward lock and the AE records
    # retain it under the same name; either way it is part of the exact result
    # digest and must point to the frozen p025 forward contract.
    if forward.get("sha256") != P025_FORWARD_LOCK_SHA256:
        raise guards.HybridQConfigError(f"{spec.family} UINT8 p025 forward lock drift")
    if spec.selection_sha256 is not None:
        selected_binding = binding.get("ae128_holdout_selection_decision", binding.get("ae_holdout_selection_decision"))
        if not isinstance(selected_binding, Mapping) or selected_binding.get("sha256") != spec.selection_sha256:
            raise guards.HybridQConfigError(f"{spec.family} UINT8 selection provenance drift")
    by_q: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise guards.HybridQConfigError("UINT8 reference row is not an object")
        plan = continuous_q.quantize_q(float(row.get("q", -1.0)))
        if int(row.get("q_e4", -1)) != plan.q_e4 or plan.q_e4 in by_q:
            raise guards.HybridQConfigError(f"{spec.family} UINT8 reference q drift")
        if int(row.get("frames", -1)) != contract.VALIDATION_FRAMES:
            raise guards.HybridQConfigError(f"{spec.family} UINT8 reference row frames drift")
        metrics = row.get("metrics")
        canonical = row.get("canonical_person_metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != set(contract.PROTECTED_METRICS):
            raise guards.HybridQConfigError(f"{spec.family} UINT8 protected metrics drift")
        if not isinstance(canonical, Mapping) or set(canonical) != set(phase9d._CSV_CANONICAL):
            raise guards.HybridQConfigError(f"{spec.family} UINT8 canonical metrics drift")
        if spec.family == "noAE":
            payload = {
                "pre_zstd_bytes": int(row["measured_uint8_sparse_bytes"]),
                "zstd_bytes": dict(row["compressed_zstd_bytes"]),
            }
        else:
            payload_block = row.get("payload")
            if not isinstance(payload_block, Mapping):
                raise guards.HybridQConfigError(f"{spec.family} UINT8 payload drift")
            payload = {
                "pre_zstd_bytes": dict(payload_block["pre_zstd_bytes"]),
                "zstd_bytes": dict(payload_block["zstd_bytes"]),
            }
        by_q[plan.q_e4] = {
            "family": spec.family,
            "q": plan.wire_q,
            "q_e4": plan.q_e4,
            "retained_cells": int(row["retained_cells"]),
            "metrics": dict(metrics),
            "canonical_person_metrics": dict(canonical),
            "absolute_service_gates": dict(row["absolute_service_gates"]),
            "payload": payload,
            "source_path": spec.relative,
            "source_sha256": spec.sha256,
            "checkpoint_sha256": spec.checkpoint_sha256,
            "selection_sha256": spec.selection_sha256,
        }
    expected = {continuous_q.quantize_q(q).q_e4 for q in Q_VALUES}
    if set(by_q) != expected:
        raise guards.HybridQConfigError(f"{spec.family} UINT8 q inventory drift")
    return by_q


def load_uint8_references() -> dict[str, dict[int, dict[str, Any]]]:
    """Hash-bind and normalize the four same-family completed UINT8 curves."""
    references: dict[str, dict[int, dict[str, Any]]] = {}
    for family, spec in UINT8_REFERENCES.items():
        _require_hash(spec.relative, spec.sha256)
        document = json.loads(_repository_path(spec.relative).read_text(encoding="utf-8"))
        references[family] = _extract_reference_rows(spec, document)
    return references


def resolve_same_family_uint8_reference(
    setting: Setting, references: Mapping[str, Mapping[int, Mapping[str, Any]]]
) -> dict[str, Any]:
    """Return only the completed UINT8 row for this exact family and wire q."""
    rows = references.get(setting.family.name)
    if not isinstance(rows, Mapping):
        raise guards.HybridQConfigError(f"no UINT8 reference for {setting.family.name}")
    row = rows.get(setting.q_e4)
    if not isinstance(row, Mapping):
        raise guards.HybridQConfigError(
            f"no same-q UINT8 reference for {setting.family.name} q_e4={setting.q_e4}"
        )
    if (
        row.get("family") != setting.family.name
        or int(row.get("q_e4", -1)) != setting.q_e4
        or continuous_q.quantize_q(float(row.get("q", -1.0))).q_e4 != setting.q_e4
    ):
        raise guards.HybridQConfigError("same-family/same-q UINT8 reference mismatch")
    spec = UINT8_REFERENCES[setting.family.name]
    if row.get("checkpoint_sha256") != spec.checkpoint_sha256:
        raise guards.HybridQConfigError("same-family UINT8 checkpoint provenance mismatch")
    return dict(row)


def _verify_completed_phase_artifacts() -> dict[str, Any]:
    bound = {
        "phase11b": {name: _require_hash(*item) for name, item in PHASE11B_ARTIFACTS.items()},
        "phase11c": {name: _require_hash(*item) for name, item in PHASE11C_ARTIFACTS.items()},
    }
    p11b = json.loads(_repository_path(PHASE11B_ARTIFACTS["report"][0]).read_text(encoding="utf-8"))
    p11c = json.loads(_repository_path(PHASE11C_ARTIFACTS["report"][0]).read_text(encoding="utf-8"))
    if p11b.get("terminal") != phase11b.TERMINAL:
        raise guards.HybridQConfigError("Phase-11B report terminal drift")
    if _repository_path(PHASE11B_ARTIFACTS["terminal"][0]).read_text(encoding="utf-8") != (
        f"{phase11b.TERMINAL} {PHASE11B_ARTIFACTS['report'][1]}\n"
    ):
        raise guards.HybridQConfigError("Phase-11B terminal does not bind report")
    if p11c.get("terminal") != "HYBRID_Q_PHASE11C_ZSTD_LEVEL_SWEEP_COMPLETE":
        raise guards.HybridQConfigError("Phase-11C report terminal drift")
    if _repository_path(PHASE11C_ARTIFACTS["terminal"][0]).read_text(encoding="utf-8") != (
        f"HYBRID_Q_PHASE11C_ZSTD_LEVEL_SWEEP_COMPLETE {PHASE11C_ARTIFACTS['report'][1]}\n"
    ):
        raise guards.HybridQConfigError("Phase-11C terminal does not bind report")
    comparisons = p11c.get("aggregate_comparisons", {}).get("comparisons", [])
    by_level = {int(row["candidate_level"]): row for row in comparisons if row.get("baseline_level") == 1}
    for level in (3, 5):
        row = by_level.get(level)
        if not isinstance(row, Mapping) or float(row.get("incremental_size_saving_bytes", 0)) >= 0 or float(row.get("incremental_codec_ms", 0)) <= 0:
            raise guards.HybridQConfigError("Phase-11C does not support fixed zstd L1")
    return {
        **bound,
        "zstd_campaign_decision": {
            "level": ZSTD_LEVEL,
            "fixed_for_phase11d": True,
            "basis": "completed Phase-11C aggregate comparisons: L3/L5 were larger and higher host cost",
            "perception_decision": False,
            "raspberry_pi_oai_latency_confirmation_pending": True,
            "zstd_level_is_not_an_rl_action": True,
        },
    }


def _verify_live_lowbit_sources() -> dict[str, str]:
    return {relative: _require_hash(relative, expected)["sha256"] for relative, expected in LIVE_LOWBIT_EXECUTION_SOURCES.items()}


def _load_fp32_q0_reference() -> dict[str, Any]:
    _require_hash(FP32_Q0_REFERENCE["path"], FP32_Q0_REFERENCE["sha256"])
    document = json.loads(_repository_path(FP32_Q0_REFERENCE["path"]).read_text(encoding="utf-8"))
    rows = document.get("curve")
    if not isinstance(rows, list):
        raise guards.HybridQConfigError("dense FP32 validation curve drift")
    q0 = next((row for row in rows if continuous_q.quantize_q(float(row.get("q", -1.0))).q_e4 == 0), None)
    if not isinstance(q0, Mapping) or set(q0.get("metrics", ())) != set(contract.PROTECTED_METRICS):
        raise guards.HybridQConfigError("dense FP32 q=0 metric reference drift")
    return {"path": FP32_Q0_REFERENCE["path"], "sha256": FP32_Q0_REFERENCE["sha256"], "metrics": dict(q0["metrics"])}


def phase11d_preflight() -> dict[str, Any]:
    """All historic and live-input checks, before a CUDA query or model build."""
    historical = phase11b.phase11b_preflight()
    references = load_uint8_references()
    completed = _verify_completed_phase_artifacts()
    live_sources = _verify_live_lowbit_sources()
    # Historical maps prove checkpoint compatibility; this separately records
    # the exact live Python files imported by this execution so a resume cannot
    # silently mix an edited runner/scorer composition with earlier records.
    live_execution_sources = phase11b._loaded_repository_source_hashes()
    if not set(LIVE_LOWBIT_EXECUTION_SOURCES).issubset(live_execution_sources):
        raise guards.HybridQConfigError("live Phase-11D source map omits a low-bit dependency")
    fp32_q0 = _load_fp32_q0_reference()
    zstd = implementation_report()
    if int(zstd.get("level", -1)) != ZSTD_LEVEL:
        raise guards.HybridQConfigError("mandatory zstd implementation is not level 1")
    return {
        "historical_checkpoint_and_source_provenance": historical,
        "same_family_uint8_references": references,
        "completed_phase_evidence": completed,
        "live_lowbit_execution_source_sha256": live_sources,
        "live_execution_source_sha256": live_execution_sources,
        "dense_fp32_q0_reference": fp32_q0,
        "zstd_implementation": zstd,
    }


def _routing_tag(setting: Setting, autoencoders: Mapping[str, SplitFeatureAE]) -> int:
    if setting.family.bottleneck is None:
        return ae_contract.AE_UNBOUND_ROUTING_TAG
    return int(autoencoders[setting.family.name].routing_tag)


def _expected_routing_tag(family: Family) -> int:
    if family.bottleneck is None:
        return ae_contract.AE_UNBOUND_ROUTING_TAG
    return ae_contract.routing_tag_from_sha256(phase11b.FROZEN_INPUTS[family.name]["sha256"])


def _byte_stats(values: Sequence[int]) -> dict[str, float | int]:
    return phase9d._byte_stats(values)


def _finite_metrics(scored: Mapping[str, Any]) -> bool:
    values = list(scored["metrics"].values()) + list(scored["canonical_person_metrics"].values())
    return all(math.isfinite(float(value)) for value in values)


def _transport_one(
    *,
    setting: Setting,
    frame: torch.Tensor,
    ranker: Any,
    decoders: lowbit_dispatch.PreloadedLowBitDecoders,
    autoencoders: Mapping[str, SplitFeatureAE],
    captures: Mapping[str, Any],
    wire: CountingWireCodec,
    tail_device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """One actual public low-bit packet, raw receive and receiver-side audit."""
    plan = continuous_q.quantize_q(setting.q)
    guards.require_frozen_c2(frame, what="original validation FP32 C2")
    ranker_before = ranker.invocations
    expected_tag = _routing_tag(setting, autoencoders)
    if setting.family.bottleneck is None:
        source = frame
        transport = lowbit_transport.encode_noae_frame(
            frame, ranker, plan.wire_q, setting.bit_width, wire_codec=wire
        )
    else:
        autoencoder = autoencoders[setting.family.name]
        capture = captures[setting.family.name]
        capture.reset()
        transport = lowbit_transport.encode_ae_frame(
            frame, autoencoder, ranker, plan.wire_q, setting.bit_width, wire_codec=wire
        )
        source = capture.latent()
    ranker_calls = ranker.invocations - ranker_before
    expected_calls = 0 if plan.is_bypass else 1
    if ranker_calls != expected_calls:
        raise guards.HybridQPayloadError("ranker invocation count drift")
    if ranker_calls and any(pointer != int(frame.data_ptr()) for pointer in ranker.scored_pointers[-ranker_calls:]):
        raise guards.HybridQPayloadError("ranker did not score original FP32 C2")
    if plan.is_bypass != (transport.selection is None):
        raise guards.HybridQPayloadError("q=0 ranker-bypass selection drift")

    wire.decompressions = 0
    received = decoders.receive(transport.packet.data, wire_codec=wire, diagnostics=True)
    if wire.decompressions != 1:
        raise guards.HybridQPayloadError("low-bit receive did not decompress exactly once")
    diagnostics = received.diagnostics
    if diagnostics is None:
        raise guards.HybridQPayloadError("low-bit receive omitted diagnostics")
    analytical = phase11b._require_header(
        diagnostics.parsed,
        plan=plan,
        family_id=setting.family.family_id,
        routing_tag=expected_tag,
        bit_width=setting.bit_width,
    )
    if transport.packet.uncompressed_bytes != analytical.total_bytes or received.uncompressed_bytes != analytical.total_bytes:
        raise guards.HybridQPayloadError("low-bit analytical payload size drift")
    if received.family.family_id != setting.family.family_id or received.family.bit_width != setting.bit_width:
        raise guards.HybridQPayloadError("received family/width drift")
    if received.q != plan.wire_q or int(received.keep_count) != plan.keep_count:
        raise guards.HybridQPayloadError("received q or keep-count drift")
    expected_indices = (
        torch.arange(contract.SPLIT_CELLS, dtype=torch.int64)
        if plan.is_bypass
        else transport.selection.keep_indices.detach().to(device="cpu", dtype=torch.int64)
    )
    if not torch.equal(diagnostics.parsed.keep_indices, expected_indices):
        raise guards.HybridQPayloadError("transmitted mask does not equal selected indices")
    dropped = ~diagnostics.keep_mask.reshape(-1)
    decoded = diagnostics.decoded_feature.reshape(int(diagnostics.parsed.channels), -1)
    if bool((decoded[:, dropped] != 0.0).any()):
        raise guards.HybridQNumericalError("dropped low-bit cells were not exact zero")
    affine = phase11b._retained_affine_error(source, diagnostics)
    if received.c2.device != tail_device:
        raise guards.HybridQPayloadError("reconstructed C2 is not on configured tail device")
    guards.require_frozen_c2(received.c2, what="reconstructed low-bit C2")
    if setting.family.bottleneck is None:
        if diagnostics.decoder is not None:
            raise guards.HybridQPayloadError("noAE packet selected an AE decoder")
    elif diagnostics.decoder is not autoencoders[setting.family.name]:
        raise guards.HybridQPayloadError("AE decoder was not selected from raw packet header")
    return received.c2, {
        "pre_zstd_bytes": int(transport.packet.uncompressed_bytes),
        "zstd_bytes": int(transport.packet.compressed_bytes),
        "ranker_invocations": ranker_calls,
        "zstd_decompressions": int(wire.decompressions),
        "keep_count": int(received.keep_count),
        "affine": affine,
        "output_device_receiver_configured_only": True,
        "mask_equals_selected_indices": True,
        "dropped_cells_exact_zero_before_ae_or_tail": True,
        "decoder_selected_from_raw_packet_metadata": True,
    }


def _load_runtime(device: torch.device) -> dict[str, Any]:
    """Reuse Phase-10B's fixed validation ordering and frozen perception load."""
    runtime = dict(phase10b._load_runtime(64, device))
    runtime.pop("bottleneck", None)
    return runtime


def _require_state_unchanged(runtime: Mapping[str, Any]) -> None:
    guards.require_module_state_unchanged(runtime["model"], runtime["model_snapshot"])
    guards.require_module_state_unchanged(runtime["ranker"], runtime["ranker_snapshot"])
    for name, autoencoder in runtime["autoencoders"].items():
        guards.require_module_state_unchanged(autoencoder, runtime["ae_snapshots"][name])


def run_validation_pass(
    *, runtime: Mapping[str, Any], setting: Setting, output: Path, workers: int, wire: CountingWireCodec
) -> dict[str, Any]:
    """One full registered validation pass for exactly one low-bit setting."""
    plan = continuous_q.quantize_q(setting.q)
    output.mkdir(parents=True, exist_ok=False)
    (output / "segmentation").mkdir()
    detections_path = output / "detections.csv"
    segmentation_manifest = output / "segmentation_manifest.csv"
    loader = DataLoader(
        Subset(runtime["inference"], list(runtime["positions"])), batch_size=INFERENCE_BATCH,
        shuffle=False, num_workers=workers, collate_fn=_collate, drop_last=False, pin_memory=False,
    )
    ranker = phase11b._CountingRanker(runtime["ranker"])
    pre_zstd, zstd, transmit_ns, tail_ns, affine_errors, affine_bounds = [], [], [], [], [], []
    observed_ids: list[str] = []
    segmentation_rows: list[dict[str, Any]] = []
    detections = person = vehicle = output_tensors = decompressions = ranker_calls = 0
    started = time.time()
    device = runtime["device"]
    torch.cuda.reset_peak_memory_stats(device=device)
    with detections_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=runtime["base"].infer.FIELDS)
        writer.writeheader()
        with torch.inference_mode():
            for fused, rows, calibrations in loader:
                c2 = runtime["model"].encode_front(fused.to(device, non_blocking=True)).float()
                guards.require_frozen_batched_c2(c2, what="frozen validation C2")
                transported: list[torch.Tensor] = []
                for index in range(c2.shape[0]):
                    _sync(device)
                    began = time.perf_counter_ns()
                    reconstructed, audit = _transport_one(
                        setting=setting, frame=c2[index], ranker=ranker,
                        decoders=runtime["decoders"], autoencoders=runtime["autoencoders"],
                        captures=runtime["captures"], wire=wire, tail_device=device,
                    )
                    _sync(device)
                    transmit_ns.append(time.perf_counter_ns() - began)
                    pre_zstd.append(audit["pre_zstd_bytes"])
                    zstd.append(audit["zstd_bytes"])
                    affine_errors.append(float(audit["affine"]["maximum_error"]))
                    affine_bounds.append(float(audit["affine"]["maximum_bound"]))
                    ranker_calls += audit["ranker_invocations"]
                    decompressions += audit["zstd_decompressions"]
                    transported.append(reconstructed)
                hybrid = torch.stack(transported)
                _sync(device)
                began = time.perf_counter_ns()
                outputs = runtime["model"].decode_tail(hybrid, dense=False)
                _sync(device)
                tail_ns.append(time.perf_counter_ns() - began)
                output_tensors += _require_tree_finite(outputs, "frozen tail output")
                calibration_gpu = [{name: tensor.to(device) for name, tensor in item.items()} for item in calibrations]
                postprocessed = runtime["model"].postprocess(outputs, calibration_gpu)
                output_tensors += _require_tree_finite(postprocessed, "frozen postprocess output")
                for index, row in enumerate(rows):
                    served, original_indices = apply_p025_service_policy(
                        {"semantic_logits": outputs["semantic_logits"][index:index + 1]}, postprocessed[index]
                    )
                    output_tensors += _require_tree_finite(served, "p025 service output")
                    records = combined_records(runtime["base"], row, served, original_indices)
                    for record in records:
                        writer.writerow(record)
                        person += int(record["class_name"] == "person")
                        vehicle += int(record["class_name"] != "person")
                    detections += len(records)
                    observed_ids.append(str(row["sample_id"]))
                    source_hw = (int(row["camera_height"]), int(row["camera_width"]))
                    labels = F.interpolate(outputs["semantic_logits"][index:index + 1].float(), size=source_hw, mode="bilinear", align_corners=False).argmax(1)[0]
                    image = labels.cpu().numpy().astype(np.uint8)
                    relative = Path("segmentation") / f"{row['sample_id']}.png"
                    if not cv2.imwrite(str(output / relative), image):
                        raise RuntimeError(f"failed segmentation write {relative}")
                    segmentation_rows.append({"sample_id": row["sample_id"], "prediction_path": str(relative), "width": image.shape[1], "height": image.shape[0]})
                del c2, hybrid, outputs, postprocessed, transported
    with segmentation_manifest.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "prediction_path", "width", "height"))
        writer.writeheader()
        writer.writerows(segmentation_rows)
    del loader
    if observed_ids != list(runtime["frame_ids"]) or len(set(observed_ids)) != contract.VALIDATION_FRAMES:
        raise guards.HybridQConfigError("registered validation ordering/coverage drift")
    analytical = lowbit_transport.analytical_size(plan.wire_q, setting.family.family_id, setting.bit_width)
    if len(pre_zstd) != contract.VALIDATION_FRAMES or set(pre_zstd) != {analytical.total_bytes}:
        raise guards.HybridQPayloadError("pre-zstd low-bit size drift")
    if decompressions != contract.VALIDATION_FRAMES:
        raise guards.HybridQPayloadError("low-bit decompression count drift")
    expected_ranker = 0 if plan.is_bypass else contract.VALIDATION_FRAMES
    if ranker_calls != expected_ranker:
        raise guards.HybridQPayloadError("low-bit ranker count drift")
    return {
        "family": setting.family.name, "family_id": setting.family.family_id,
        "latent_channels": setting.family.channels, "bit_width": setting.bit_width,
        "quantizer": f"UINT{setting.bit_width}", "q": plan.wire_q, "q_e4": plan.q_e4,
        "frames": len(observed_ids), "one_pass_declaration": True,
        "inference_passes_for_this_setting": 1, "prediction_root": str(output),
        "detections_csv_sha256": sha256_file(detections_path),
        "segmentation_manifest_sha256": sha256_file(segmentation_manifest),
        "detections": detections, "person_service_outputs": person, "vehicle_service_outputs": vehicle,
        "retained_cells": plan.keep_count, "dropped_cells": plan.drop_count,
        "payload": {
            "family_id": setting.family.family_id, "transported_channels": setting.family.channels,
            "routing_tag": _routing_tag(setting, runtime["autoencoders"]),
            "analytical_pre_zstd_bytes": analytical.total_bytes,
            "analytical_breakdown": {"header_bytes": analytical.header_bytes, "mask_bytes": analytical.mask_bytes, "range_bytes": analytical.range_bytes, "value_bytes": analytical.value_bytes},
            "pre_zstd_bytes": _byte_stats(pre_zstd), "zstd_bytes": _byte_stats(zstd),
            "zstd_mandatory": True, "zstd_level": ZSTD_LEVEL, "zstd": implementation_report(),
        },
        "integrity": {
            "ranker_invocations": ranker_calls, "q0_ranker_bypassed": plan.is_bypass,
            "zstd_decompressions": decompressions, "exactly_one_decompression_per_frame": True,
            "transmitted_mask_equals_selected_indices": True,
            "retained_lowbit_values_within_registered_error_bound": True,
            "maximum_retained_affine_error": max(affine_errors),
            "maximum_registered_affine_error_bound": max(affine_bounds),
            "dropped_cells_reconstruct_to_exact_zero": True,
            "decoder_selected_from_raw_packet_metadata": True,
            "reconstructed_c2_on_configured_tail_device": True,
            "packet_fields_control_output_device": False,
            "all_outputs_finite": True, "output_tensors_checked": output_tensors,
        },
        "component_latency": {"transport_receive": _latency_stats(transmit_ns), "frozen_tail_per_batch": _latency_stats(tail_ns), "evidence_scope": "current-host diagnostic only; Raspberry Pi/OAI latency remains pending"},
        "wall_seconds": time.time() - started,
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device) / 2 ** 20,
        "peak_reserved_vram_mib": torch.cuda.max_memory_reserved(device) / 2 ** 20,
    }


def _metric_vs_fp32(scored: Mapping[str, Any], fp32_q0: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in contract.PROTECTED_METRICS:
        reference, actual = float(fp32_q0["metrics"][name]), float(scored["metrics"][name])
        out[name] = {"dense_fp32_q0": reference, "lowbit": actual, "delta_lowbit_minus_fp32_q0": actual - reference, "ratio_lowbit_over_fp32_q0": None if reference == 0 else actual / reference}
    return out


def _payload_reference_value(value: Any) -> float:
    return float(value["median"] if isinstance(value, Mapping) else value)


def _classify(setting: Setting, row: Mapping[str, Any]) -> dict[str, Any]:
    """Compose the corrected Phase-10B classification contract unchanged."""
    integrity = row["integrity"]
    valid = bool(row["all_outputs_and_metrics_finite"]) and all(bool(integrity[name]) for name in (
        "exactly_one_decompression_per_frame", "transmitted_mask_equals_selected_indices",
        "retained_lowbit_values_within_registered_error_bound", "dropped_cells_reconstruct_to_exact_zero",
        "decoder_selected_from_raw_packet_metadata", "reconstructed_c2_on_configured_tail_device",
    )) and not bool(integrity["packet_fields_control_output_device"])
    range_stratified = phase10b.person_range_stratification(row["person_avo_detail"]["distance_bins"], row["metrics"])
    localization = phase10b.localization_requirements(row["metrics"], range_stratified=range_stratified)
    segmentation = phase10b.segmentation_installability(row["metrics"])
    service = phase10b.service_readiness(row)
    stress = setting.q in STRESS_Q_VALUES
    if not valid:
        tier, reason = "INVALID", "transport, routing, numerical or execution integrity failed"
    elif stress:
        tier, reason = "EMERGENCY_ONLY", "registered q=0.90/q=0.98 stress anchor"
    elif bool(row["same_family_same_q_uint8_preservation"]["all_passed"]):
        tier, reason = "FULL_PRESERVATION", "all 12 same-family/same-q UINT8 preservation gates pass"
    elif bool(localization["all_passed"]):
        tier, reason = "LOCALIZATION_PRIORITY", "all corrected Phase-10B AVO/object requirements pass"
    else:
        tier, reason = "EMERGENCY_ONLY", "valid execution but object-priority requirements fail"
    return {
        "tier": tier, "tier_reason": reason, "full_preservation": bool(row["same_family_same_q_uint8_preservation"]["all_passed"]),
        "service_readiness": service, "segmentation": segmentation,
        "localization_priority": localization, "person_range_stratified": range_stratified,
        "stress_anchor_forced_emergency_only": stress,
        "state_infeasible": {"tier": "STATE_INFEASIBLE", "assigned_by_offline_validation": False, "reserved_for": "runtime resource/network infeasibility only"},
        "perception_degradation_masks_action": False,
    }


def _setting_document(*, setting: Setting, raw: Mapping[str, Any], scored: Mapping[str, Any], reference: Mapping[str, Any], family_q0: Mapping[str, Any], fp32_q0: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    preservation = common.evaluate_same_q_gates(reference["metrics"], scored["metrics"], baseline="completed same-family/same-q UINT8+zstd result")
    if not _finite_metrics(scored):
        raise guards.HybridQNumericalError("non-finite scored low-bit metric")
    row = {
        "schema": SETTING_SCHEMA, "terminal": SETTING_TERMINAL,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "run_identity_sha256": identity["sha256"],
        **dict(raw), "metrics": dict(scored["metrics"]), "canonical_person_metrics": dict(scored["canonical_person_metrics"]),
        "absolute_service_gates": dict(scored["absolute_service_gates"]), "person_avo_detail": dict(scored["person_avo_detail"]),
        "same_family_same_q_uint8_reference": dict(reference),
        "same_family_same_q_uint8_preservation": preservation,
        "absolute_metrics_and_deltas_vs_dense_fp32_q0": _metric_vs_fp32(scored, fp32_q0),
        "dense_fp32_q0_reference": {"path": fp32_q0["path"], "sha256": fp32_q0["sha256"]},
        "payload_ratios": {
            "vs_framed_fp32_noae_q0": float(raw["payload"]["zstd_bytes"]["median"]) / contract.FRAMED_Q0_PAYLOAD_BYTES,
            "vs_same_family_same_q_uint8_zstd": float(raw["payload"]["zstd_bytes"]["median"]) / _payload_reference_value(reference["payload"]["zstd_bytes"]),
            "vs_same_family_uint8_zstd_q0": float(raw["payload"]["zstd_bytes"]["median"]) / _payload_reference_value(family_q0["payload"]["zstd_bytes"]),
        },
        "all_outputs_and_metrics_finite": True,
        "frozen_perception_state_unchanged": True, "stable_ranker_state_unchanged": True,
        "all_selected_ae_states_unchanged": True,
        "prediction_artifacts": {"root": str(raw["prediction_root"]), "removed_before_this_record": False, "durability_order": list(DURABILITY_ORDER)},
    }
    row["classification"] = _classify(setting, row)
    return row


def setting_path(output: Path, setting: Setting) -> Path:
    return Path(output) / SETTINGS_DIRNAME / f"{setting.key}.json"


def cleanup_path(output: Path, setting: Setting) -> Path:
    return Path(output) / CLEANUP_DIRNAME / f"{setting.key}.json"


def prediction_path(output: Path, setting: Setting) -> Path:
    return Path(output) / WORKING_DIRNAME / setting.key


def _validate_durable_setting(path: Path, setting: Setting, identity: Mapping[str, Any]) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    def fail(reason: str) -> None:
        raise guards.HybridQConfigError(f"{path}: {reason}")
    if document.get("schema") != SETTING_SCHEMA or document.get("terminal") != SETTING_TERMINAL:
        fail("incomplete or foreign setting record")
    if document.get("run_identity_sha256") != identity["sha256"]:
        fail("run identity mismatch")
    for key, expected in (("family", setting.family.name), ("family_id", setting.family.family_id), ("bit_width", setting.bit_width), ("q_e4", setting.q_e4)):
        if document.get(key) != expected:
            fail(f"{key} mismatch")
    if continuous_q.quantize_q(float(document.get("q", -1))).q_e4 != setting.q_e4:
        fail("q mismatch")
    if int(document.get("frames", -1)) != contract.VALIDATION_FRAMES or int(document.get("inference_passes_for_this_setting", -1)) != 1 or not bool(document.get("one_pass_declaration")):
        fail("frame or one-pass declaration mismatch")
    plan = continuous_q.quantize_q(setting.q)
    if int(document.get("retained_cells", -1)) != plan.keep_count or int(document.get("dropped_cells", -1)) != plan.drop_count:
        fail("keep/drop count mismatch")
    payload = document.get("payload")
    analytical = lowbit_transport.analytical_size(setting.q, setting.family.family_id, setting.bit_width)
    if not isinstance(payload, Mapping) or int(payload.get("family_id", -1)) != setting.family.family_id or int(payload.get("transported_channels", -1)) != setting.family.channels or int(payload.get("routing_tag", -1)) != _expected_routing_tag(setting.family) or int(payload.get("analytical_pre_zstd_bytes", -1)) != analytical.total_bytes or int(payload.get("zstd_level", -1)) != ZSTD_LEVEL or not bool(payload.get("zstd_mandatory")):
        fail("payload binding mismatch")
    for block in (payload.get("pre_zstd_bytes"), payload.get("zstd_bytes")):
        if not isinstance(block, Mapping) or int(block.get("samples", -1)) != contract.VALIDATION_FRAMES:
            fail("payload sample structure mismatch")
        if not all(math.isfinite(float(block.get(name, float("nan")))) for name in ("mean", "median", "p95", "minimum", "maximum")):
            fail("non-finite payload statistic")
    if any(float(payload["pre_zstd_bytes"].get(name, float("nan"))) != float(analytical.total_bytes) for name in ("mean", "median", "p95", "minimum", "maximum")):
        fail("pre-zstd payload is not the analytical size")
    integrity = document.get("integrity")
    required = ("exactly_one_decompression_per_frame", "transmitted_mask_equals_selected_indices", "retained_lowbit_values_within_registered_error_bound", "dropped_cells_reconstruct_to_exact_zero", "decoder_selected_from_raw_packet_metadata", "reconstructed_c2_on_configured_tail_device")
    expected_ranker = 0 if plan.is_bypass else contract.VALIDATION_FRAMES
    if not isinstance(integrity, Mapping) or int(integrity.get("ranker_invocations", -1)) != expected_ranker or int(integrity.get("zstd_decompressions", -1)) != contract.VALIDATION_FRAMES or any(not bool(integrity.get(name)) for name in required) or bool(integrity.get("packet_fields_control_output_device", True)):
        fail("integrity binding mismatch")
    if not all(bool(document.get(name)) for name in ("all_outputs_and_metrics_finite", "frozen_perception_state_unchanged", "stable_ranker_state_unchanged", "all_selected_ae_states_unchanged")):
        fail("finite/frozen-state binding mismatch")
    metrics = document.get("metrics")
    preservation = document.get("same_family_same_q_uint8_preservation")
    reference = document.get("same_family_same_q_uint8_reference")
    if not isinstance(metrics, Mapping) or set(metrics) != set(contract.PROTECTED_METRICS) or not all(math.isfinite(float(value)) for value in metrics.values()):
        fail("protected metric binding mismatch")
    if not isinstance(preservation, Mapping) or int(preservation.get("gates_total", -1)) != len(contract.HOLDOUT_PRESERVATION_GATES):
        fail("preservation gate binding mismatch")
    spec = UINT8_REFERENCES[setting.family.name]
    if not isinstance(reference, Mapping) or reference.get("family") != setting.family.name or int(reference.get("q_e4", -1)) != setting.q_e4 or reference.get("source_sha256") != spec.sha256 or reference.get("checkpoint_sha256") != spec.checkpoint_sha256 or reference.get("selection_sha256") != spec.selection_sha256:
        fail("same-family reference binding mismatch")
    classification = document.get("classification")
    if not isinstance(classification, Mapping) or classification.get("tier") not in {"FULL_PRESERVATION", "LOCALIZATION_PRIORITY", "EMERGENCY_ONLY", "INVALID"}:
        fail("classification binding mismatch")
    return document


def _complete_cleanup(*, output: Path, setting: Setting, identity: Mapping[str, Any], record_sha256: str, keep_segmentation: bool) -> None:
    scratch = prediction_path(output, setting)
    if scratch.exists() and not keep_segmentation:
        shutil.rmtree(scratch)
    _atomic_json(cleanup_path(output, setting), {
        "schema": CLEANUP_SCHEMA, "terminal": CLEANUP_TERMINAL,
        "run_identity_sha256": identity["sha256"], "setting_key": setting.key,
        "setting_sha256": record_sha256, "prediction_root": str(scratch),
        "prediction_artifacts_removed_after_durable_record": not keep_segmentation,
        "retained_by_keep_segmentation": bool(keep_segmentation),
    })


def reuse_or_refuse(*, output: Path, setting: Setting, identity: Mapping[str, Any], keep_segmentation: bool) -> dict[str, Any] | None:
    path = setting_path(output, setting)
    if not path.is_file():
        return None
    document = _validate_durable_setting(path, setting, identity)
    digest = sha256_file(path)
    marker = cleanup_path(output, setting)
    if marker.is_file():
        cleanup = json.loads(marker.read_text(encoding="utf-8"))
        if cleanup.get("schema") != CLEANUP_SCHEMA or cleanup.get("run_identity_sha256") != identity["sha256"] or cleanup.get("setting_sha256") != digest:
            raise guards.HybridQConfigError(f"{marker}: invalid cleanup marker")
    else:
        _complete_cleanup(output=output, setting=setting, identity=identity, record_sha256=digest, keep_segmentation=keep_segmentation)
    return document


def run_identity(preflight: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema": SCHEMA, "execute": EXECUTE_TOKEN,
        "catalog": [{"family": row.family.name, "family_id": row.family.family_id, "bits": row.bit_width, "q": row.q, "q_e4": row.q_e4} for row in CATALOG],
        "validation_frames": contract.VALIDATION_FRAMES, "one_pass_per_setting": True,
        "zstd_campaign": preflight["completed_phase_evidence"]["zstd_campaign_decision"],
        "historical_provenance": preflight["historical_checkpoint_and_source_provenance"]["frozen_hashes"],
        "uint8_reference_sha256": {name: spec.sha256 for name, spec in UINT8_REFERENCES.items()},
        "live_lowbit_execution_source_sha256": preflight["live_lowbit_execution_source_sha256"],
        "live_execution_source_sha256": preflight["live_execution_source_sha256"],
        "runner_sha256": sha256_file(Path(__file__)),
    }
    return {**body, "sha256": _identity_digest(body)}


def manifest_document(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "terminal_when_complete": TERMINAL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_identity_sha256": identity["sha256"], "run_identity": dict(identity),
        "expected_settings": [setting.key for setting in CATALOG], "expected_setting_count": 48,
        "durability_order_per_setting": list(DURABILITY_ORDER),
        "resume_rule": "--resume validates and reuses only a complete exact record; invalid records are refused and only missing settings run",
    }


def _load_manifest(output: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    path = output / "run_manifest.json"
    if not path.is_file():
        raise guards.HybridQConfigError("--resume requires a run manifest")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA or document.get("run_identity_sha256") != identity["sha256"] or dict(document.get("run_identity", {})) != dict(identity) or document.get("expected_settings") != [setting.key for setting in CATALOG]:
        raise guards.HybridQConfigError("run manifest identity or inventory drift")
    return document


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = ("family", "quantizer", "q", "q_e4", "retained_cells", "zstd_median_bytes", "same_family_uint8_gates", "tier", "service_ready", "segmentation_installable", "person_avo_recall_0_30m")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        classification = row["classification"]
        writer.writerow({
            "family": row["family"], "quantizer": row["quantizer"], "q": f"{float(row['q']):.2f}", "q_e4": row["q_e4"], "retained_cells": row["retained_cells"],
            "zstd_median_bytes": row["payload"]["zstd_bytes"]["median"], "same_family_uint8_gates": row["same_family_same_q_uint8_preservation"]["gates_passed"],
            "tier": classification["tier"], "service_ready": classification["service_readiness"]["service_ready"], "segmentation_installable": classification["segmentation"]["segmentation_installable"],
            "person_avo_recall_0_30m": classification["person_range_stratified"][phase10b.PERSON_PRIMARY_RANGE_RECALL_METRIC],
        })
    return stream.getvalue()


def _report_text(document: Mapping[str, Any]) -> str:
    return "\n".join((
        "# Phase 11D low-bit deployment validation", "",
        f"Terminal: `{TERMINAL}`", "",
        "Fixed catalog: 4 families × UINT6/UINT4 × six registered q anchors = 48 settings. Each row is one full registered 3,345-frame validation pass through the public low-bit/zstd-1/raw-byte-dispatch/frozen-tail path.", "",
        "Every row compares to the completed same-family, same-q UINT8 record and reports absolute metrics plus deltas to dense FP32 q=0. Phase-10B's corrected 0–30 m AVO person-recall classification contract is used without changing detection emission at other ranges.", "",
        "Zstd L1 is fixed campaign configuration, bound to Phase-11C aggregate evidence: L3/L5 were larger and higher host cost. This is not a perception decision or RL action; Raspberry Pi/OAI latency remains pending.", "",
        "Durability is per setting: atomically fsync its record, then remove scratch (unless retained explicitly), then atomically write cleanup. Resume reuses only fully valid exact records and refuses invalid ones.", "",
        f"Completed settings: {len(document['curve'])}/48.", "",
    ))


def finalize(*, output: Path, rows: list[dict[str, Any]], identity: Mapping[str, Any], preflight: Mapping[str, Any], runtime: Mapping[str, Any], manifest: Mapping[str, Any], started: float) -> dict[str, Any]:
    if len(rows) != 48 or [row["setting_key"] for row in rows] != [setting.key for setting in CATALOG]:
        raise guards.HybridQConfigError("final report requires exactly the registered 48 records")
    _require_state_unchanged(runtime)
    document = {
        "schema": SCHEMA, "terminal": TERMINAL, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "full UINT6/UINT4 deployment-path validation; no training, tuning, checkpoint selection or action deletion",
        "scope": {"families": [family.name for family in FAMILIES], "bit_widths": list(BIT_WIDTHS), "q_values": list(Q_VALUES), "settings": 48, "validation_frames_per_setting": contract.VALIDATION_FRAMES, "one_pass_per_setting": True, "test_accessed": False, "carla_launched": False},
        "run_identity": dict(identity), "manifest": {"path": "run_manifest.json", "sha256": sha256_file(output / "run_manifest.json"), "identity_sha256": manifest["run_identity_sha256"]},
        "provenance": {key: value for key, value in preflight.items() if key != "same_family_uint8_references"},
        "tail_device": str(runtime["device"]), "receiver_tail_device_configured": True,
        "curve": rows,
        "integrity": {"settings_completed": len(rows), "required_settings": 48, "zstd_decompressions": sum(int(row["integrity"]["zstd_decompressions"]) for row in rows), "required_zstd_decompressions": 48 * contract.VALIDATION_FRAMES, "all_frozen_state_equal": True, "packet_fields_control_output_device": False},
        "wall_seconds_this_invocation": time.time() - started,
    }
    if document["integrity"]["zstd_decompressions"] != document["integrity"]["required_zstd_decompressions"]:
        raise guards.HybridQPayloadError("final decompression total drift")
    _atomic_json(output / "phase11d_lowbit_validation.json", document)
    _atomic_write(output / "phase11d_lowbit_validation.csv", _csv_text(rows))
    _atomic_write(output / "PHASE11D_LOWBIT_VALIDATION_REPORT.md", _report_text(document))
    _atomic_write(output / TERMINAL, f"{TERMINAL} {sha256_file(output / 'phase11d_lowbit_validation.json')}\n")
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase-11D fixed 48-setting UINT6/UINT4 deployment validation")
    parser.add_argument("--execute", required=True, choices=(EXECUTE_TOKEN,))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=DATALOADER_WORKERS)
    parser.add_argument("--keep-segmentation", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output)
    if args.resume:
        if not output.is_dir() or (output / TERMINAL).exists():
            raise guards.HybridQConfigError("--resume requires an incomplete existing Phase-11D output")
    elif output.exists() and any(output.iterdir()):
        raise guards.HybridQConfigError("Phase-11D output is create-only without --resume")
    # This preflight hashes artifacts, validates historical checkpoint source
    # maps and verifies live low-bit source bytes before CUDA is queried.
    preflight = phase11d_preflight()
    identity = run_identity(preflight)
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-11D requires the qualified CUDA runtime")
    device = torch.device("cuda:0")
    output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        manifest = _load_manifest(output, identity)
    else:
        _atomic_json(output / "run_manifest.json", manifest_document(identity))
        manifest = _load_manifest(output, identity)
    for directory in (SETTINGS_DIRNAME, CLEANUP_DIRNAME, WORKING_DIRNAME):
        (output / directory).mkdir(exist_ok=True)
    torch.manual_seed(contract.RANKER_INIT_SEED)
    default_threads = torch.get_num_threads()
    torch.set_num_threads(TORCH_CPU_THREADS)
    started = time.time()
    runtime = _load_runtime(device)
    autoencoders: dict[str, SplitFeatureAE] = {}
    for family in FAMILIES:
        if family.bottleneck is None:
            continue
        autoencoders[family.name] = phase11b._load_selected_autoencoder(
            family.name, family.bottleneck, phase11b.FROZEN_INPUTS[family.name],
            preflight["historical_checkpoint_and_source_provenance"]["checkpoint_payloads"][family.name], device,
        )
    decoders = lowbit_dispatch.PreloadedLowBitDecoders(autoencoders.values(), tail_device=device)
    if decoders.tail_device != device or decoders.families != tuple(f.family_id for f in FAMILIES if f.bottleneck is not None):
        raise guards.HybridQConfigError("preloaded decoder/tail-device configuration drift")
    runtime.update({
        "autoencoders": autoencoders, "decoders": decoders,
        "captures": {name: phase11b._EncodeCapture(autoencoder) for name, autoencoder in autoencoders.items()},
        "model_snapshot": guards.snapshot_module_state(runtime["model"]),
        "ranker_snapshot": guards.snapshot_module_state(runtime["ranker"]),
        "ae_snapshots": {name: guards.snapshot_module_state(autoencoder) for name, autoencoder in autoencoders.items()},
    })
    scorers = load_frozen_scorers()
    gt, _ = scorers.load_gt(runtime["dataset_root"], contract.PRIMARY_CONTRACT)
    validation_gt = {sample_id: gt.get(sample_id, []) for sample_id in runtime["frame_ids"]}
    person_gt, ignore_cache, wire = _person_only(validation_gt), {}, CountingWireCodec()
    completed: list[dict[str, Any]] = []
    references = preflight["same_family_uint8_references"]
    for setting in CATALOG:
        durable = reuse_or_refuse(output=output, setting=setting, identity=identity, keep_segmentation=bool(args.keep_segmentation))
        if durable is not None:
            completed.append(durable)
            continue
        scratch = prediction_path(output, setting)
        if scratch.exists():
            raise guards.HybridQConfigError(f"{scratch} exists without a durable record; refusing to overwrite")
        raw = run_validation_pass(runtime=runtime, setting=setting, output=scratch, workers=int(args.workers), wire=wire)
        _require_state_unchanged(runtime)
        scored = score_validation_pass(result=raw, scorers=scorers, truth=runtime["truth"], experiment=runtime["dataset_root"], frame_ids=runtime["frame_ids"], gt=validation_gt, person_gt=person_gt, ignore_cache=ignore_cache)
        reference = resolve_same_family_uint8_reference(setting, references)
        family_q0 = resolve_same_family_uint8_reference(Setting(setting.family, setting.bit_width, 0.0), references)
        record = _setting_document(setting=setting, raw=raw, scored=scored, reference=reference, family_q0=family_q0, fp32_q0=preflight["dense_fp32_q0_reference"], identity=identity)
        record["setting_key"] = setting.key
        path = setting_path(output, setting)
        digest = _atomic_json(path, record)
        durable = _validate_durable_setting(path, setting, identity)
        _complete_cleanup(output=output, setting=setting, identity=identity, record_sha256=digest, keep_segmentation=bool(args.keep_segmentation))
        completed.append(durable)
    try:
        document = finalize(output=output, rows=completed, identity=identity, preflight=preflight, runtime=runtime, manifest=manifest, started=started)
        print(json.dumps({"terminal": TERMINAL, "settings": len(document["curve"]), "output": str(output)}), flush=True)
    finally:
        for capture in runtime["captures"].values():
            capture.close()
        torch.set_num_threads(default_threads)
    return 0


def _minimal_durable_record_for_test(setting: Setting, identity: Mapping[str, Any]) -> dict[str, Any]:
    """Synthetic-only helper for the two focused CPU durability checks."""
    plan = continuous_q.quantize_q(setting.q)
    analytical = lowbit_transport.analytical_size(setting.q, setting.family.family_id, setting.bit_width)
    metrics = {name: 1.0 for name in contract.PROTECTED_METRICS}
    stats = {"samples": contract.VALIDATION_FRAMES, "mean": 1.0, "median": 1.0, "p95": 1.0, "minimum": 1.0, "maximum": 1.0}
    pre_stats = {**stats, **{name: float(analytical.total_bytes) for name in ("mean", "median", "p95", "minimum", "maximum")}}
    return {
        "schema": SETTING_SCHEMA, "terminal": SETTING_TERMINAL, "run_identity_sha256": identity["sha256"], "setting_key": setting.key,
        "family": setting.family.name, "family_id": setting.family.family_id, "bit_width": setting.bit_width, "q": plan.wire_q, "q_e4": plan.q_e4,
        "frames": contract.VALIDATION_FRAMES, "one_pass_declaration": True, "inference_passes_for_this_setting": 1, "retained_cells": plan.keep_count, "dropped_cells": plan.drop_count,
        "payload": {"family_id": setting.family.family_id, "transported_channels": setting.family.channels, "routing_tag": _expected_routing_tag(setting.family), "analytical_pre_zstd_bytes": analytical.total_bytes, "zstd_level": ZSTD_LEVEL, "zstd_mandatory": True, "pre_zstd_bytes": pre_stats, "zstd_bytes": dict(stats)},
        "integrity": {"ranker_invocations": 0 if plan.is_bypass else contract.VALIDATION_FRAMES, "zstd_decompressions": contract.VALIDATION_FRAMES, "exactly_one_decompression_per_frame": True, "transmitted_mask_equals_selected_indices": True, "retained_lowbit_values_within_registered_error_bound": True, "dropped_cells_reconstruct_to_exact_zero": True, "decoder_selected_from_raw_packet_metadata": True, "reconstructed_c2_on_configured_tail_device": True, "packet_fields_control_output_device": False},
        "all_outputs_and_metrics_finite": True, "frozen_perception_state_unchanged": True, "stable_ranker_state_unchanged": True, "all_selected_ae_states_unchanged": True,
        "metrics": metrics, "same_family_same_q_uint8_preservation": {"gates_total": len(contract.HOLDOUT_PRESERVATION_GATES)},
        "same_family_same_q_uint8_reference": {"family": setting.family.name, "q_e4": plan.q_e4, "source_sha256": UINT8_REFERENCES[setting.family.name].sha256, "checkpoint_sha256": UINT8_REFERENCES[setting.family.name].checkpoint_sha256, "selection_sha256": UINT8_REFERENCES[setting.family.name].selection_sha256},
        "classification": {"tier": "FULL_PRESERVATION"},
    }


if __name__ == "__main__":  # pragma: no cover - explicit future execution only
    raise SystemExit(main())
