"""Phase 10B: deployment-path UINT8 + mandatory-zstd validation of AE64 and AE32.

One runner, two families, one family per process. Every family-dependent
quantity -- execute token, terminal, schema, family id, latent width, analytical
payload size, range bytes, routing tag, artifact paths, output filenames and
report labels -- is derived from the single ``--bottleneck`` argument, so AE64
and AE32 cannot drift apart and neither can borrow the other's artifact:

    python3 -m ...ae_v1.ae_phase10b_uint8_validation \\
      --execute SPLITFUSION_AE64_PHASE10B_UINT8_VALIDATION \\
      --bottleneck 64 --output <run>

    python3 -m ...ae_v1.ae_phase10b_uint8_validation \\
      --execute SPLITFUSION_AE32_PHASE10B_UINT8_VALIDATION \\
      --bottleneck 32 --output <run>

AE128 is deliberately not reachable here: it is complete, it keeps its own
Phase-9D runner and artifacts, and ``--bottleneck 128`` is refused by argparse
and again by ``require_phase10_bottleneck``. A token that names a different
family than ``--bottleneck`` is refused before CUDA is touched and before any
output directory is created.

This is a measurement phase. It trains nothing, tunes nothing, changes no
threshold, no NMS setting, no calibration and no scorer, and it never opens test
data. Per family it measures the six registered q anchors exactly once each on
the registered 3,345 validation frames, through the **real deployment path** and
nothing shorter:

    original FP32 C2
      -> selected family encoder (always, on the complete frame)
      -> per-channel UINT8 latent quantization, ranges from the complete latent
      -> stable per-frame top-K selection for q>0
      -> family-labelled sparse AE wire
      -> mandatory zstd level 1
      -> received raw bytes
      -> exactly one decompression
      -> decoder selected from the received header family/bottleneck/routing tag
      -> dequantization / zero scatter
      -> selected family decoder
      -> unchanged frozen perception tail and p025 service policy

q=0 bypasses the ranker but never bypasses the AE, so even the q=0 row is a
lossy channel reconstruction rather than an identity. At q>0 the ranker scores
the *original FP32 C2*, independently per frame, before any cell is dropped.

Nothing scientific is restated here. The frozen noAE UINT8+zstd reference
loader, the six registered q, the twelve preservation gates, the nine absolute
service gates, the registered service baseline, the per-frame ranker counter,
the byte statistics and the preregistered acceptance rule are the completed
Phase-9D objects, imported and reused rather than re-expressed; only the family
identity and the family-labelled terminals differ. Reusing the objects is what
makes "the same gates and the same rule as AE128" checkable rather than merely
claimed.

Per q the durability order is the proven Phase-9D one: the setting JSON is the
scientific completion record and is fsynced into place *first*, that q's scratch
predictions are removed only afterwards, and the cleanup marker is written last.
``--resume`` reuses only fully validated durable records, never remeasures a
valid q, and refuses rather than overwrites an invalid one.

Two independent verdicts are recorded per run, and the second cannot move the
first:

* the **primary** result -- the original twelve-gate relative preservation study
  and the family-level preregistered acceptance rule, unchanged; and
* a **secondary prospective classification** registered here, before any
  Phase-10B number exists, which labels each measured profile
  FULL_PRESERVATION / LOCALIZATION_PRIORITY / EMERGENCY_ONLY / INVALID against
  absolute AVO/object requirements, and reports segmentation installability and
  9/9 absolute service readiness as separate results.

Neither verdict selects a checkpoint, moves a threshold, an NMS setting, a model
or a scorer, and a failed family acceptance suppresses no measured q row.

The recorded component latency is current-host diagnostic evidence only: no
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
from . import ae_phase10_common as family
from . import ae_training_common as common
from .ae_gpu_qualification import CountingWireCodec, bind_inputs as bind_frozen_inputs
from .ae_gpu_qualification import state_hashes
from .ae_model import build_split_feature_ae

# The completed Phase-9D objects this phase measures *with*, never re-expresses.
# Importing them is what pins "the same six q, the same twelve preservation
# gates, the same nine service gates, the same registered baseline, the same
# per-frame ranker counter and the same acceptance rule as AE128".
from .ae_uint8_validation import (
    ACCEPTANCE_PRIMARY_Q,
    BASELINE_SERVICE_PASS_COUNT,
    GATE_COUNT,
    NOAE_UINT8_VALIDATION_RELPATH,
    NOAE_UINT8_VALIDATION_SHA256,
    Q_VALUES,
    SAME_Q_BASELINE_LABEL,
    SERVICE_GATE_COUNT,
    STRESS_Q_VALUES,
    _byte_stats,
    _CountingRanker,
    acceptance_inputs,
    load_noae_reference,
)
from . import ae_uint8_validation as phase9d


PHASE = "phase10b"

DATALOADER_WORKERS = 8
INFERENCE_BATCH = 8

SETTINGS_DIRNAME = "settings"
CLEANUP_DIRNAME = "cleanup"
WORKING_DIRNAME = "working_predictions"

# Durability order per q, and the reason for it. The per-q setting JSON is the
# scientific completion record, so it is written and fsynced into place *first*.
# Only then are that q's scratch predictions removed, and the cleanup marker is
# written last. An interruption can therefore lose at most a scratch prediction
# directory -- never a completed measurement -- and a resume finishes the cleanup
# instead of remeasuring the q.
DURABILITY_ORDER = (
    f"atomically write {SETTINGS_DIRNAME}/<q>.json",
    f"remove {WORKING_DIRNAME}/<q>",
    f"atomically write {CLEANUP_DIRNAME}/<q>.json",
)

# The AE package modules whose identity this runner is pinned to by name. The
# whole package source map is bound as well; these are called out so a reviewer
# can see which files define the runner itself.
RUNNER_SOURCES = (
    "ae_phase10b_uint8_validation.py",
    "ae_phase10_common.py",
    "ae_uint8_validation.py",
    "ae_uint8_transport.py",
    "ae_family_dispatch.py",
    "ae_composition.py",
    "ae_model.py",
    "ae_contract.py",
    "ae_training_common.py",
)


# ---------------------------------------------------------------------------
# Family identity carried by every Phase-10B artifact
# ---------------------------------------------------------------------------

# `ae_phase10_common.family_fields` stamps `phase: "phase10a"`, which is correct
# for the training and selection artifacts this phase *consumes*. Phase-10B's own
# artifacts carry the same family identity under their own phase label, so the
# two are never confused for one another on disk.
FAMILY_IDENTITY_KEYS = ("family", "family_id", "bottleneck", "init_seed")


def family_fields(bottleneck: int) -> dict[str, Any]:
    """The family identity every Phase-10B artifact carries, in one place."""
    size = family.require_phase10_bottleneck(bottleneck)
    fields = dict(family.family_fields(size))
    if set(FAMILY_IDENTITY_KEYS) - set(fields):
        raise guards.HybridQConfigError(
            "the registered family fields no longer cover "
            f"{sorted(set(FAMILY_IDENTITY_KEYS) - set(fields))}"
        )
    fields["phase"] = PHASE
    return fields


def require_family_identity(
    payload: Mapping[str, Any], bottleneck: int, *, what: str
) -> None:
    """Fail closed unless a Phase-10B artifact declares exactly this family."""
    expected = family_fields(bottleneck)
    for name in tuple(FAMILY_IDENTITY_KEYS) + ("phase",):
        if name not in payload:
            raise guards.HybridQConfigError(f"{what} does not carry {name}")
        if payload[name] != expected[name]:
            raise guards.HybridQConfigError(
                f"{what} {name} is {payload[name]!r}, this run is "
                f"{expected[name]!r}"
            )


# ---------------------------------------------------------------------------
# Family-derived identity: tokens, terminals, schemas, filenames
# ---------------------------------------------------------------------------


def execute_token(bottleneck: int) -> str:
    size = family.require_phase10_bottleneck(bottleneck)
    return family.require_family_labelled(
        f"SPLITFUSION_AE{size}_PHASE10B_UINT8_VALIDATION",
        size,
        what="Phase-10B execute token",
    )


EXECUTE_TOKENS = tuple(
    execute_token(size) for size in family.AE_PHASE10_BOTTLENECKS
)


def require_token_agrees_with_bottleneck(token: str, bottleneck: int) -> int:
    """The execute token and ``--bottleneck`` must name the same family, exactly.

    Called first in ``main``, before CUDA is touched and before any output
    directory is created, so a mismatch or an out-of-scope family costs nothing
    and leaves nothing behind.
    """
    expected = {execute_token(size): size for size in family.AE_PHASE10_BOTTLENECKS}
    if token not in expected:
        raise guards.HybridQConfigError(
            f"{token!r} is not a registered Phase-10B execute token "
            f"{sorted(expected)}"
        )
    size = family.require_phase10_bottleneck(bottleneck)
    if expected[token] != size:
        raise guards.HybridQConfigError(
            f"execute token {token} names {family.family_label(expected[token])} "
            f"but --bottleneck {size} names {family.family_label(size)}; they "
            "must agree"
        )
    return size


def _schema(bottleneck: int, kind: str) -> str:
    size = family.require_phase10_bottleneck(bottleneck)
    return family.require_family_labelled(
        f"splitfusion_fcos_ae{size}_{PHASE}_{kind}_v1", size, what=f"{kind} schema"
    )


def schema(bottleneck: int) -> str:
    return _schema(bottleneck, "uint8_validation")


def setting_schema(bottleneck: int) -> str:
    return _schema(bottleneck, "uint8_setting")


def cleanup_schema(bottleneck: int) -> str:
    return _schema(bottleneck, "uint8_cleanup")


def terminal(bottleneck: int) -> str:
    size = family.require_phase10_bottleneck(bottleneck)
    return family.require_family_labelled(
        f"SPLITFUSION_AE{size}_PHASE10B_UINT8_VALIDATION_COMPLETE",
        size,
        what="Phase-10B terminal",
    )


def setting_terminal(bottleneck: int) -> str:
    size = family.require_phase10_bottleneck(bottleneck)
    return family.require_family_labelled(
        f"SPLITFUSION_AE{size}_PHASE10B_UINT8_Q_SETTING_COMPLETE",
        size,
        what="Phase-10B setting terminal",
    )


def cleanup_terminal(bottleneck: int) -> str:
    size = family.require_phase10_bottleneck(bottleneck)
    return family.require_family_labelled(
        f"SPLITFUSION_AE{size}_PHASE10B_UINT8_Q_PREDICTIONS_REMOVED",
        size,
        what="Phase-10B cleanup terminal",
    )


def accepted_terminal(bottleneck: int) -> str:
    size = family.require_phase10_bottleneck(bottleneck)
    return family.require_family_labelled(
        f"AE{size}_UINT8_ZSTD_DEPLOYMENT_ACCEPTED", size, what="acceptance terminal"
    )


def not_accepted_terminal(bottleneck: int) -> str:
    size = family.require_phase10_bottleneck(bottleneck)
    return family.require_family_labelled(
        f"AE{size}_UINT8_ZSTD_DEPLOYMENT_NOT_ACCEPTED",
        size,
        what="non-acceptance terminal",
    )


def result_json_filename(bottleneck: int) -> str:
    size = family.require_phase10_bottleneck(bottleneck)
    return family.require_family_labelled(
        f"{PHASE}_ae{size}_uint8_validation.json", size, what="result JSON filename"
    )


def result_csv_filename(bottleneck: int) -> str:
    size = family.require_phase10_bottleneck(bottleneck)
    return family.require_family_labelled(
        f"{PHASE}_ae{size}_uint8_validation.csv", size, what="result CSV filename"
    )


def report_filename(bottleneck: int) -> str:
    size = family.require_phase10_bottleneck(bottleneck)
    return family.require_family_labelled(
        f"AE_PHASE10B_AE{size}_UINT8_VALIDATION_REPORT.md",
        size,
        what="report filename",
    )


def manifest_filename(bottleneck: int) -> str:
    size = family.require_phase10_bottleneck(bottleneck)
    return family.require_family_labelled(
        f"ae{size}_{PHASE}_run_manifest.json", size, what="run manifest filename"
    )


# ---------------------------------------------------------------------------
# Family-derived latent geometry and payload accounting
# ---------------------------------------------------------------------------


def latent_width(bottleneck: int) -> int:
    """The transported latent channel count: the bottleneck itself."""
    return family.require_phase10_bottleneck(bottleneck)


def analytical_size(bottleneck: int, q: float) -> ae_uint8_transport.AeAnalyticalSize:
    """Exact pre-zstd byte accounting for one family at one q.

    Derived, never tabulated: the committed transport's own accounting is asked
    for the family's bottleneck, so header, mask, range and value bytes all
    follow from ``--bottleneck``.
    """
    return ae_uint8_transport.analytical_size(
        float(q), family.require_phase10_bottleneck(bottleneck)
    )


def range_bytes(bottleneck: int) -> int:
    """One little-endian FP32 min/max pair per transported latent channel."""
    return ae_uint8_transport.range_byte_count(
        family.require_phase10_bottleneck(bottleneck)
    )


# ---------------------------------------------------------------------------
# The selected Phase-10A artifacts, bound by exact SHA-256
# ---------------------------------------------------------------------------

# One row per family. Nothing else in this module names an epoch or a digest,
# and every path below is *derived* from the family's own registered filename
# helpers rather than spelled out twice.
SELECTED_ARTIFACTS: dict[int, dict[str, Any]] = {
    64: {
        "training_relpath": (
            "experiments/splitfusion_fcos_ae_v1/20260903_phase10_ae64_training"
        ),
        "epoch": 12,
        "checkpoint_sha256": (
            "dd7c5124e27114584ab2083e59160a3ff2a2d040d0a37d22564ac98c838aa8e0"
        ),
        "decision_sha256": (
            "0d2fe444574d3fdc9aee287448084bf2cfc1efa2d0ec6944ac07355d9ff7c87e"
        ),
    },
    32: {
        "training_relpath": (
            "experiments/splitfusion_fcos_ae_v1/20260903_phase10_ae32_training"
        ),
        "epoch": 8,
        "checkpoint_sha256": (
            "e2f867757e8db0620316c092264ac7eb53d12bb5ef66ed14475eb40693d1f271"
        ),
        "decision_sha256": (
            "e3dfbfb736bb8847ad11d92b1573f88058e1c4319ac4a0180284db2171afac34"
        ),
    },
}
if set(SELECTED_ARTIFACTS) != set(family.AE_PHASE10_BOTTLENECKS):
    raise guards.HybridQConfigError(
        "the Phase-10B artifact registry does not cover exactly the Phase-10A "
        f"families {family.AE_PHASE10_BOTTLENECKS}"
    )


def selected(bottleneck: int) -> dict[str, Any]:
    """The one selected checkpoint and decision for one family, fully derived."""
    size = family.require_phase10_bottleneck(bottleneck)
    row = SELECTED_ARTIFACTS[size]
    epoch = int(row["epoch"])
    if epoch not in common.AE_CANDIDATE_EPOCHS:
        raise guards.HybridQConfigError(
            f"selected epoch {epoch} is not a registered candidate epoch "
            f"{common.AE_CANDIDATE_EPOCHS}"
        )
    training = str(row["training_relpath"])
    checkpoint_name = family.candidate_filename(size, epoch)
    decision_name = family.holdout_report_filename(size)
    selection_dirname = family.holdout_selection_dirname(size)
    return {
        **family_fields(size),
        "phase10a_training_relpath": training,
        "selected_epoch": epoch,
        "selected_checkpoint_name": checkpoint_name,
        "selected_checkpoint_relpath": f"{training}/checkpoints/{checkpoint_name}",
        "selected_checkpoint_sha256": str(row["checkpoint_sha256"]),
        "holdout_decision_relpath": f"{training}/{selection_dirname}/{decision_name}",
        "holdout_decision_sha256": str(row["decision_sha256"]),
        "holdout_decision_terminal": family.holdout_terminal(size),
        "holdout_decision_terminal_relpath": (
            f"{training}/{selection_dirname}/{family.holdout_terminal(size)}"
        ),
        "training_terminal_relpath": (
            f"{training}/{family.training_terminal(size)}"
        ),
    }


def routing_tag(bottleneck: int) -> int:
    """The deterministic nonzero routing tag of this family's selected checkpoint.

    Derived from the **full** selected-checkpoint SHA-256 by
    ``ae_contract.routing_tag_from_sha256``. The result is a 32-bit
    decoder-routing discriminator and is deliberately not the checkpoint's
    identity: 32 bits cannot authenticate a checkpoint, and the authoritative
    identity stays the full digest recorded beside it. AE64 and AE32 therefore
    carry different tags as well as different family ids.
    """
    return ae_contract.routing_tag_from_sha256(
        selected(bottleneck)["selected_checkpoint_sha256"]
    )


def routing_record(bottleneck: int) -> dict[str, Any]:
    """Both facts, side by side, with the tag's role stated."""
    size = family.require_phase10_bottleneck(bottleneck)
    row = selected(size)
    tag = routing_tag(size)
    return {
        **family_fields(size),
        "selected_checkpoint_path": row["selected_checkpoint_relpath"],
        "selected_checkpoint_sha256": row["selected_checkpoint_sha256"],
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


# The AE modules that decide what the saved family tensors *mean* -- the
# architecture and shapes, and the family/bottleneck/latent-geometry registry
# they are built against -- plus the modules that define the wire and the
# composition order this phase measures. All must be byte-identical to what the
# selected checkpoint recorded, or the loaded weights are being reinterpreted or
# a different transport is being measured.
AE_CHECKPOINT_SEMANTICS_SOURCES = ("ae_contract.py", "ae_model.py", "ae_phase10_common.py")
AE_TRANSPORT_SEMANTICS_SOURCES = (
    "ae_composition.py",
    "ae_loss.py",
    "ae_uint8_transport.py",
    "ae_family_dispatch.py",
    "__init__.py",
)

# Phase 10B *is* an addition to the AE package, so its own source map cannot be
# bit-identical to the one the selected checkpoint recorded. Exactly these three
# new files may appear as additions; unlike Phase 9D, **no** previously recorded
# file may change and none may disappear.
AE_PHASE10B_ADDED_SOURCES = (
    "AE_PHASE10B_UINT8_VALIDATION_IMPLEMENTATION_REPORT.md",
    "ae_phase10b_uint8_validation.py",
    "tests/test_ae_phase10b_validation.py",
)


def _bind_artifact(relative: str, expected: str) -> dict[str, str]:
    path = (contract.repository_root() / relative).resolve(strict=True)
    observed = sha256_file(path)
    if observed != expected:
        raise guards.HybridQConfigError(f"{relative} sha256 drift")
    return {"path": relative, "sha256": observed}


def bind_inputs(bottleneck: int) -> dict[str, Any]:
    """The frozen stack plus this family's three Phase-10B artifacts, hash-bound.

    The frozen perception checkpoint, the stable epoch-4 ranker, the p025 forward
    lock and the hybrid-q locked configuration come from the existing authorized
    AE binding, unchanged. The frozen noAE Phase-8B UINT8+zstd reference is the
    same file, at the same digest, that AE128 Phase 9D was scored against.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    row = selected(size)
    binding = bind_frozen_inputs()
    return {
        **binding,
        "selected_ae_checkpoint": _bind_artifact(
            row["selected_checkpoint_relpath"], row["selected_checkpoint_sha256"]
        ),
        "ae_holdout_selection_decision": _bind_artifact(
            row["holdout_decision_relpath"], row["holdout_decision_sha256"]
        ),
        "noae_uint8_zstd_validation_reference": _bind_artifact(
            NOAE_UINT8_VALIDATION_RELPATH, NOAE_UINT8_VALIDATION_SHA256
        ),
        "framed_fp32_noae_q0_payload_bytes": contract.FRAMED_Q0_PAYLOAD_BYTES,
        "routing": routing_record(size),
        **family_fields(size),
    }


def load_holdout_decision(
    bottleneck: int, binding: Mapping[str, Any]
) -> dict[str, Any]:
    """The bound Phase-10A decision, and the chain from it to this checkpoint.

    The decision must be this family's own, must have completed, must not have
    opened validation or test data, must have selected exactly the bound epoch,
    and must have recorded exactly the bound checkpoint digest for that epoch.
    Its own completion marker holds the sha256 of the report the selector had
    just written, so that digest is the selector's own statement of what it
    produced and must still equal the bound document's hash.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    row = selected(size)
    root = contract.repository_root()
    path = (root / row["holdout_decision_relpath"]).resolve(strict=True)

    marker = (root / row["holdout_decision_terminal_relpath"]).resolve(strict=True)
    recorded = marker.read_text(encoding="utf-8").strip()
    if recorded != row["holdout_decision_sha256"]:
        raise guards.HybridQConfigError(
            f"{row['holdout_decision_terminal']} records report sha256 "
            f"{recorded!r}, Phase 10B binds "
            f"{row['holdout_decision_sha256']!r}"
        )

    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != family.holdout_schema(size):
        raise guards.HybridQConfigError(
            f"{family.family_label(size)} holdout-selection schema drift"
        )
    if document.get("terminal") != family.holdout_terminal(size):
        raise guards.HybridQConfigError(
            f"the {family.family_label(size)} holdout selection did not complete"
        )
    scope = document["scope"]
    family.require_family_fields(scope, size, what="holdout decision scope")
    if bool(scope["validation_or_test_accessed"]):
        raise guards.HybridQConfigError(
            "the holdout decision reports validation/test access"
        )
    if bool(scope["training_run_here"]):
        raise guards.HybridQConfigError("the holdout decision reports training")
    if bool(scope["deployment_validation_performed_here"]):
        raise guards.HybridQConfigError(
            "the holdout decision claims to have performed deployment validation"
        )
    if str(scope["transport"]) != common.AE_HOLDOUT_QUANTIZER:
        raise guards.HybridQConfigError("holdout decision transport drift")
    if int(scope["holdout_frames"]) != contract.TRAIN_HOLDOUT_FRAMES:
        raise guards.HybridQConfigError("holdout decision frame-count drift")

    selection = document["selection"]
    family.require_family_fields(selection, size, what="holdout decision selection")
    if int(selection["selected_epoch"]) != int(row["selected_epoch"]):
        raise guards.HybridQConfigError(
            f"the bound decision selected epoch {selection['selected_epoch']}, "
            f"Phase 10B binds epoch {row['selected_epoch']}"
        )
    if bool(selection["selection_is_a_service_ready_claim"]):
        raise guards.HybridQConfigError(
            "the holdout decision claims to be a service-ready decision"
        )

    candidates = dict(document["training_run"]["candidate_checkpoints"])
    name = row["selected_checkpoint_name"]
    if candidates.get(name) != row["selected_checkpoint_sha256"]:
        raise guards.HybridQConfigError(
            "the decision's recorded candidate hash is not the bound checkpoint"
        )
    if str(document["training_run"]["path"]).rstrip("/").split("/")[-1] != (
        row["phase10a_training_relpath"].rstrip("/").split("/")[-1]
    ):
        raise guards.HybridQConfigError(
            "the decision was made against a different training run directory"
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
        **family_fields(size),
        "path": row["holdout_decision_relpath"],
        "sha256": row["holdout_decision_sha256"],
        "terminal_records_this_digest": True,
        "selected_epoch": int(selection["selected_epoch"]),
        "decided_at_criterion": selection["decided_at_criterion"],
        "rule": selection["rule"],
        "rule_source": selection.get("rule_source"),
        "selection_is_a_service_ready_claim": False,
        "selected_checkpoint": name,
        "selected_checkpoint_sha256": candidates[name],
        "candidate_checkpoints": candidates,
        "holdout_frames": int(scope["holdout_frames"]),
        "holdout_transport": str(scope["transport"]),
        "holdout_q_values": [float(q) for q in scope["evaluated_q_values"]],
        "holdout_epochs": [int(e) for e in scope["evaluated_epochs"]],
    }


def ae_package_source_delta(
    recorded: Mapping[str, str], live: Mapping[str, str]
) -> dict[str, Any]:
    """Which AE package files moved since the selected checkpoint was written.

    Phase 10B is stricter than Phase 9D: no previously recorded file may change
    and none may disappear. Exactly the three new Phase-10B files may appear as
    additions, and every added, changed and removed path is reported either way.
    """
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
    if changed:
        raise guards.HybridQConfigError(
            "AE package file(s) recorded by the selected checkpoint have "
            f"changed since it was written: {changed}"
        )
    if removed:
        raise guards.HybridQConfigError(
            f"AE package file(s) recorded by the checkpoint are gone: {removed}"
        )
    unexpected = [name for name in added if name not in AE_PHASE10B_ADDED_SOURCES]
    if unexpected:
        raise guards.HybridQConfigError(
            "only the registered Phase-10B files may be added to the AE package "
            f"source map; found {unexpected}"
        )
    return {
        "semantics_modules_required_unchanged": list(frozen),
        "semantics_modules_unchanged": True,
        "recorded_files": len(recorded),
        "live_files": len(live),
        "changed_files_allowed": 0,
        "changed": [],
        "removed": [],
        "allowlisted_additions": list(AE_PHASE10B_ADDED_SOURCES),
        "added": [{"path": name, "live_sha256": live[name]} for name in added],
        "rationale": (
            "Phase 10B adds exactly one runner, one test module and one "
            "implementation report to the AE package, so its source map differs "
            "from the one the selected checkpoint recorded by exactly those "
            "additions and by nothing else"
        ),
    }


def require_selected_bindings(
    payload: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Enforce every binding the selected checkpoint saved.

    All but one field are required to be bit-identical, iterated from
    ``common.binding_fields`` so no field can be silently forgotten. The single
    exception is ``ae_package_source_sha256``, the AE package's own source map:
    Phase 10B *is* an addition to that package, so requiring equality would be
    requiring this phase not to exist. It is enforced as a declared delta with a
    zero-change allowance instead.
    """
    expected = common.binding_fields(binding)
    delta: dict[str, Any] | None = None
    for name, value in expected.items():
        if name not in payload:
            raise guards.HybridQConfigError(
                f"selected checkpoint does not carry {name}"
            )
        if name == "ae_package_source_sha256":
            delta = ae_package_source_delta(dict(payload[name]), dict(value))
            continue
        if payload[name] != value:
            raise guards.HybridQConfigError(f"selected checkpoint {name} drift")
    if delta is None:
        raise guards.HybridQConfigError(
            "the saved binding no longer carries an AE package source map"
        )
    return {
        "enforced_exactly": sorted(set(expected) - {"ae_package_source_sha256"}),
        "ae_package_source_delta": delta,
    }


def load_selected_ae(
    bottleneck: int, device: torch.device, binding: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Load exactly this family's bound checkpoint, frozen, eval, tag-bound."""
    size = family.require_phase10_bottleneck(bottleneck)
    row = selected(size)
    path = (contract.repository_root() / row["selected_checkpoint_relpath"]).resolve(
        strict=True
    )
    digest = sha256_file(path)
    if digest != row["selected_checkpoint_sha256"]:
        raise guards.HybridQConfigError(
            f"selected {family.family_label(size)} checkpoint sha256 drift"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema"] != family.candidate_schema(size):
        raise guards.HybridQConfigError("selected checkpoint schema drift")
    family.require_family_fields(payload, size, what=path.name)
    if int(payload["epoch"]) != int(row["selected_epoch"]):
        raise guards.HybridQConfigError("selected checkpoint epoch drift")
    if payload["configuration"] != family.training_configuration(size):
        raise guards.HybridQConfigError(
            "the selected checkpoint was trained under a different locked "
            "configuration"
        )
    enforced = require_selected_bindings(payload, binding)

    autoencoder = build_split_feature_ae(size)
    autoencoder.load_state_dict(payload["autoencoder"])
    if autoencoder.parameter_count() != int(payload["parameter_count"]):
        raise guards.HybridQConfigError("selected checkpoint parameter-count drift")
    if autoencoder.bottleneck != latent_width(size):
        raise guards.HybridQConfigError("selected checkpoint latent width drift")
    if autoencoder.family_id != family.family_id(size):
        raise guards.HybridQConfigError("selected checkpoint family id drift")
    autoencoder = autoencoder.to(device)
    common.freeze(autoencoder)
    guards.require_module_parameters_finite(
        autoencoder, f"selected {family.family_label(size)}"
    )
    tag = routing_tag(size)
    autoencoder.bind_routing_tag(tag)
    if autoencoder.routing_tag != tag or not autoencoder.is_bound:
        raise guards.HybridQConfigError(
            f"selected {family.family_label(size)} routing tag was not bound"
        )
    per_tensor, aggregate = state_hashes(autoencoder)
    provenance = {
        **routing_record(size),
        "epoch": int(payload["epoch"]),
        "stage": str(payload["stage"]),
        "latent_channels": latent_width(size),
        "family_name": ae_contract.family_name(family.family_id(size)),
        "parameter_count": autoencoder.parameter_count(),
        "global_update_index": int(payload["global_update_index"]),
        "stage_b_cycle_position": int(payload["stage_b_cycle_position"]),
        "wire_identity": autoencoder.wire_identity(),
        "state_sha256": aggregate,
        "state_sha256_per_tensor": per_tensor,
        "bindings": enforced,
        "trained_in_this_phase": False,
        "selected_in_this_phase": False,
    }
    del payload
    return autoencoder, provenance


# ---------------------------------------------------------------------------
# Preregistered primary interpretation: the Phase-9D rule, family-relabelled
# ---------------------------------------------------------------------------


def acceptance_rule(bottleneck: int) -> str:
    """The completed Phase-9D acceptance rule with only the family name changed.

    Built by substitution on ``ae_uint8_validation.ACCEPTANCE_RULE`` rather than
    restated, and checked to carry this family's label and no AE128 label, so a
    reviewer can see the rule text cannot have been reworded here.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    label = family.family_label(size)
    rule = phase9d.ACCEPTANCE_RULE.replace("AE128", label)
    if "AE128" in rule or label not in rule:
        raise guards.HybridQConfigError(
            "the Phase-9D acceptance rule did not relabel cleanly onto "
            f"{label}"
        )
    if rule == phase9d.ACCEPTANCE_RULE:
        raise guards.HybridQConfigError(
            "the relabelled acceptance rule is byte-identical to the AE128 rule"
        )
    return rule


ACCEPTANCE_RULE_SOURCE = "phase9d_preregistered_acceptance_rule_reused_unchanged"


def evaluate_acceptance(
    rows: Sequence[Mapping[str, Any]], bottleneck: int
) -> dict[str, Any]:
    """Apply the Phase-9D rule verbatim, then relabel only the family terminal.

    ``ae_uint8_validation.evaluate_acceptance`` is the completed object, so the
    q=0 condition, the primary-q condition, the stress-q exclusion, the
    registered service baseline and the accept/not-accept logic cannot drift
    here. Only the decision terminal and the family identity are this phase's
    own.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    result = dict(phase9d.evaluate_acceptance(rows))
    accepted = bool(result["accepted"])
    if result["decision"] not in (
        phase9d.ACCEPTED_TERMINAL,
        phase9d.NOT_ACCEPTED_TERMINAL,
    ):
        raise guards.HybridQConfigError("the reused acceptance rule changed shape")
    result["rule"] = acceptance_rule(size)
    result["rule_source"] = ACCEPTANCE_RULE_SOURCE
    result["rule_evaluator"] = "ae_uint8_validation.evaluate_acceptance"
    result["decision"] = (
        accepted_terminal(size) if accepted else not_accepted_terminal(size)
    )
    result["preservation_is_relative_to_frozen_noae_uint8_zstd_same_q"] = True
    result["preservation_is_not_an_absolute_service_claim"] = True
    result["failed_acceptance_suppresses_no_measured_q_row"] = True
    result.update(family_fields(size))
    return result


# ---------------------------------------------------------------------------
# Secondary prospective classification
#
# Registered here, in source, before any Phase-10B number exists, and applied
# verbatim. It is deliberately *secondary*: it cannot change checkpoint
# selection (which is already complete and hash-bound), cannot change the
# primary acceptance terminal, cannot alter a threshold, an NMS setting, a model
# or a scorer, and cannot erase or reinterpret an original preservation failure.
# The primary twelve-gate relative-preservation result above is untouched by
# everything below.
# ---------------------------------------------------------------------------

# Absolute AVO/object requirements. Absolute, not relative: these are compared
# against fixed numbers, not against the frozen noAE same-q row.
LOCALIZATION_OBJECT_REQUIREMENTS = (
    ("vehicle_precision", 0.80, "higher"),
    ("vehicle_recall", 0.85, "higher"),
    ("vehicle_xy_mae_m", 1.00, "lower"),
    ("person_avo_precision", 0.70, "higher"),
    ("person_avo_recall", 0.70, "higher"),
    ("person_avo_f1", 0.70, "higher"),
    ("person_avo_xy_mae_m", 1.20, "lower"),
    ("person_avo_recall_0_30m", 0.70, "higher"),
)

# ---------------------------------------------------------------------------
# Frozen pedestrian operating ranges (contract correction)
#
# The completed range-aware person support feasibility study did not corroborate
# on train-holdout episode 04, so no range-aware runtime policy is implemented
# and the frozen p025 perception path is unchanged. What changes here is only
# which range the absolute tier gate is *read on*: the primary operating range
# is 0-30 m, and 30-40 m is retained as reported extended-range stress.
#
# This boundary is EVALUATION-ONLY. It is expressed on ground-truth distance,
# which exists only in the evaluator, so it cannot be a runtime action even in
# principle. Deployment continues to emit every detection the frozen p025
# pipeline accepts, throughout its existing range: nothing here filters,
# suppresses, relabels, rescores, reorders or truncates a runtime detection, and
# no range test is applied to model output. (The rejected feasibility policies
# A/B/C were different in kind -- they gated on *predicted* radial distance,
# which is runtime-computable -- and they are deliberately not implemented.)
# ---------------------------------------------------------------------------

PEDESTRIAN_PRIMARY_RANGE_BINS = ("00_10m", "10_20m", "20_30m")
PEDESTRIAN_EXTENDED_RANGE_BINS = ("30_40m",)
PEDESTRIAN_BOUNDARY_BIN = "20_30m"
PEDESTRIAN_PRIMARY_RANGE = "0 <= gt_distance_m < 30"
PEDESTRIAN_EXTENDED_DIAGNOSTIC_RANGE = "30 <= gt_distance_m <= 40"
PERSON_PRIMARY_RANGE_RECALL_METRIC = "person_avo_recall_0_30m"
PERSON_HISTORICAL_LONG_RANGE_RECALL_METRIC = "person_avo_recall_20_40m"
RANGE_STRATIFIED_KEY = "person_range_stratified"

PEDESTRIAN_RANGE_PROVENANCE = (
    "The 0-30 m primary operating range was selected from frozen noAE "
    "range-stratified analysis and literature context before Phase-10B AE64/AE32 "
    "validation. The 30-40 m results remain reported as extended-range stress. "
    "Independent test-set confirmation has not been performed."
)

EVALUATION_ONLY_BOUNDARY_RULE = (
    "The 30 m boundary is evaluation-only. It does not filter, suppress, "
    "relabel, rescore or otherwise change any runtime detection. Deployment "
    "continues to emit every detection accepted by the frozen p025 pipeline "
    "throughout its existing range. Ground-truth distance is available only to "
    "the evaluator, so the boundary is not runtime-computable and is never "
    "applied to model output."
)

# Recorded verbatim beside every classification so the declaration travels with
# the number it qualifies.
EVALUATION_ONLY_BOUNDARY_DECLARATIONS = {
    "boundary_is_evaluation_only": True,
    "boundary_quantity": "ground-truth distance, evaluator-only",
    "boundary_runtime_computable": False,
    "runtime_detections_filtered_by_range": False,
    "runtime_detections_suppressed_by_range": False,
    "runtime_detections_relabelled_by_range": False,
    "runtime_detections_rescored_by_range": False,
    "deployment_emits_all_p025_detections_at_every_range": True,
    "frozen_p025_perception_path_changed": False,
    "range_aware_runtime_policy_implemented": False,
    "rejected_feasibility_policies_abc_implemented": False,
}

PER_BAND_PRECISION_UNAVAILABLE_REASON = (
    "The frozen AVO scorer publishes each distance bin as a recall slice "
    "(eligible_gt / tp / fn) only: a false positive is not attributed to a "
    "range, because doing so would require binning predictions by predicted "
    "distance, which is new matching logic this correction does not introduce. "
    "Per-band precision is therefore not derivable from the frozen artifacts, "
    "and aggregate AVO precision remains the precision gate."
)

RANGE_METRIC_UNAVAILABLE_REASON = (
    "The frozen noAE UINT8+zstd reference document publishes the twelve "
    "protected metrics without per-bin distance slices, so "
    f"{PERSON_PRIMARY_RANGE_RECALL_METRIC} cannot be derived for those rows. It "
    "is recorded as not evaluable rather than as failing."
)

# The three segmentation outputs. They are measured and reported, and they
# decide segmentation *installability*, but they deliberately do not enter the
# localization-priority classification.
SEGMENTATION_INSTALL_REQUIREMENTS = (
    ("vehicle_iou", 0.85, "higher"),
    ("person_box_mask_iou", 0.50, "higher"),
    ("foreground_miou", 0.675, "higher"),
)

SEGMENTATION_EXCLUDED_FROM_LOCALIZATION_CLASSIFICATION = tuple(
    name for name, _target, _direction in SEGMENTATION_INSTALL_REQUIREMENTS
)

LOCALIZATION_THRESHOLD_PROVENANCE = (
    "Holdout-informed thresholds frozen before AE64/AE32 held-out deployment "
    "validation. The validation frames were not used for AE training or "
    "checkpoint selection."
)

SEGMENTATION_INSTALL_RULE = (
    "segmentation_installable = vehicle_iou >= 0.85 and person_box_mask_iou >= "
    "0.50 and foreground_miou >= 0.675. A 12/12 relative-preservation result "
    "does not by itself authorize replacing the spatial-map segmentation layer. "
    "Install new segmentation only when segmentation_installable is true; "
    "otherwise retain the previous segmentation layer with its original "
    "timestamp."
)

SEGMENTATION_INSTALL_ACTION_INSTALL = "install_new_segmentation"
SEGMENTATION_INSTALL_ACTION_RETAIN = (
    "retain_previous_segmentation_layer_with_original_timestamp"
)

SERVICE_READY_RULE = (
    "SERVICE_READY is a separate absolute result: all nine registered absolute "
    "service gates pass at this profile. It is never derived from, implied by or "
    "substituted for a 12/12 relative preservation result."
)
SERVICE_READY_TERMINAL = "SERVICE_READY"

TIER_FULL_PRESERVATION = "FULL_PRESERVATION"
TIER_LOCALIZATION_PRIORITY = "LOCALIZATION_PRIORITY"
TIER_EMERGENCY_ONLY = "EMERGENCY_ONLY"
TIER_INVALID = "INVALID"
TIER_STATE_INFEASIBLE = "STATE_INFEASIBLE"

CLASSIFICATION_TIERS = (
    (
        TIER_FULL_PRESERVATION,
        "the existing primary rule passes at this profile: all twelve same-q "
        "preservation gates hold against the frozen noAE UINT8+zstd row at the "
        "same q, and the absolute service-gate count is not reduced below the "
        "profile's registered comparison count",
    ),
    (
        TIER_LOCALIZATION_PRIORITY,
        "the absolute AVO/object requirements all pass, whatever the relative "
        "preservation gates did",
    ),
    (
        TIER_EMERGENCY_ONLY,
        "transport and output integrity pass but the absolute object "
        "requirements fail",
    ),
    (
        TIER_INVALID,
        "transport, routing, numerical or execution failure; a measured "
        "quality shortfall is never INVALID",
    ),
)

STATE_INFEASIBLE_DEFINITION = (
    "reserved for a runtime action that a hard state-dependent resource "
    "constraint makes unavailable in the current state -- for example a payload "
    "that does not fit the instantaneous transport budget. It is a runtime "
    "availability verdict about a state, not a measurement outcome about a "
    "profile, so this offline validation never assigns it: every registered q "
    "is measured and reported here."
)

MASKING_POLICY = (
    "perception degradation changes a profile's tier and therefore its reward; "
    "it never permanently masks the action. Only technical invalidity (INVALID) "
    "or a hard state-dependent resource constraint (STATE_INFEASIBLE) may mask "
    "an action."
)

# Exactly the recorded per-frame integrity declarations a valid row owes. A row
# failing any of them is INVALID rather than merely degraded.
INTEGRITY_REQUIRED_TRUE = (
    "all_outputs_finite",
    "retained_uint8_cells_equal_selected_indices",
    "dropped_cells_scattered_to_exact_zero",
    "decoder_selected_from_received_header_bytes",
    "received_family_matches_selected_family",
    "received_latent_width_matches_bottleneck",
    "received_routing_tag_matches_bound_tag",
    "ranges_from_complete_latent_before_dropping",
    "selection_independent_per_frame",
)
INTEGRITY_REQUIRED_FALSE = (
    "ae_encoder_bypassed",
    "local_packet_metadata_used_for_selection",
    "reconstruction_is_identity_at_any_q",
    "batched_or_cross_frame_selection_used",
)


def _requirement_rows(
    requirements: Sequence[tuple[str, float, str]], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate a set of absolute requirements, treating non-finite as failure."""
    rows: dict[str, Any] = {}
    for name, target, direction in requirements:
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError):
            value = float("nan")
        finite = math.isfinite(value)
        passed = finite and (
            value >= float(target) if direction == "higher" else value <= float(target)
        )
        rows[name] = {
            "value": value if finite else None,
            "target": float(target),
            "direction": direction,
            "finite": finite,
            "passed": bool(passed),
        }
    return {
        "requirements": rows,
        "total": len(rows),
        "passed_count": sum(1 for row in rows.values() if row["passed"]),
        "failed": sorted(name for name, row in rows.items() if not row["passed"]),
        "all_passed": all(row["passed"] for row in rows.values()),
    }


def person_range_stratification(
    distance_bins: Mapping[str, Any], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Range-stratified person reporting, derived from the frozen recall slices.

    Nothing is matched, scored or inferred here. Every number is a sum or a ratio
    of the per-bin ``eligible_gt`` / ``tp`` / ``fn`` counts the frozen AVO scorer
    already produced, so the 0-30 m recall is a strict partition of exactly the
    same accounting as the aggregate. Only the primary-range recall is a tier
    gate; every other row is report-only.
    """
    bins: dict[str, Any] = {}
    for name in PEDESTRIAN_PRIMARY_RANGE_BINS + PEDESTRIAN_EXTENDED_RANGE_BINS:
        bucket = distance_bins.get(name)
        if not isinstance(bucket, Mapping):
            raise guards.HybridQConfigError(f"missing frozen distance bin: {name}")
        eligible = int(bucket["eligible_gt"])
        tp, fn = int(bucket["tp"]), int(bucket["fn"])
        if tp + fn != eligible:
            raise guards.HybridQConfigError(
                f"distance-bin TP+FN denominator failure: {name}"
            )
        bins[name] = {
            "eligible_gt": eligible,
            "tp": tp,
            "fn": fn,
            "recall": (tp / eligible) if eligible else None,
            "xy_mae_m": bucket.get("xy_mae_m"),
            "precision": None,
            "precision_available": False,
            "is_tier_gate": False,
        }

    def cumulative(names: Sequence[str]) -> dict[str, Any]:
        eligible = sum(int(bins[name]["eligible_gt"]) for name in names)
        tp = sum(int(bins[name]["tp"]) for name in names)
        fn = sum(int(bins[name]["fn"]) for name in names)
        if tp + fn != eligible:
            raise guards.HybridQConfigError(
                f"cumulative TP+FN denominator failure: {tuple(names)}"
            )
        return {
            "bins": list(names),
            "eligible_gt": eligible,
            "tp": tp,
            "fn": fn,
            "recall": (tp / eligible) if eligible else None,
            "precision": None,
            "precision_available": False,
        }

    primary = cumulative(PEDESTRIAN_PRIMARY_RANGE_BINS)
    extended = cumulative(PEDESTRIAN_EXTENDED_RANGE_BINS)
    historical = cumulative(contract.PERSON_LONG_RANGE_BINS)

    # The historical 20-40 m recall is kept for comparison, and it must reproduce
    # the protected metric exactly: same counts, same formula.
    recorded = float(metrics[PERSON_HISTORICAL_LONG_RANGE_RECALL_METRIC])
    derived = float(historical["recall"]) if historical["recall"] is not None else 0.0
    if derived != recorded:
        raise guards.HybridQConfigError(
            "derived 20-40 m person recall does not reproduce the protected metric"
        )

    primary_recall = primary["recall"] if primary["recall"] is not None else 0.0
    return {
        PERSON_PRIMARY_RANGE_RECALL_METRIC: float(primary_recall),
        "primary_operating_range": PEDESTRIAN_PRIMARY_RANGE,
        "extended_diagnostic_range": PEDESTRIAN_EXTENDED_DIAGNOSTIC_RANGE,
        "primary_operating_range_detail": {**primary, "is_tier_gate": True},
        "extended_range_stress": {
            **extended,
            "is_tier_gate": False,
            "role": "extended-range stress, reported and never gated",
        },
        "boundary_band": {
            **bins[PEDESTRIAN_BOUNDARY_BIN],
            "band": PEDESTRIAN_BOUNDARY_BIN,
            "role": (
                "reported separately so the cumulative 0-30 m result cannot hide "
                "boundary-band behaviour"
            ),
        },
        "historical_20_40m": {
            **historical,
            "is_tier_gate": False,
            "role": "historical comparison against the superseded 20-40 m gate",
            "reproduces_protected_metric": True,
        },
        "bins": bins,
        "precision_by_range": {
            "available": False,
            "reason": PER_BAND_PRECISION_UNAVAILABLE_REASON,
            "precision_gate": "person_avo_precision, aggregate AVO view",
        },
        "range_provenance": PEDESTRIAN_RANGE_PROVENANCE,
        "derived_from": (
            "frozen AVO scorer per-bin eligible_gt/tp/fn; no new matching, "
            "scoring or inference logic"
        ),
        "evaluation_only_boundary_rule": EVALUATION_ONLY_BOUNDARY_RULE,
        **dict(EVALUATION_ONLY_BOUNDARY_DECLARATIONS),
    }


def localization_requirements(
    metrics: Mapping[str, Any],
    *,
    range_stratified: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The eight absolute AVO/object requirements, evaluated as registered.

    The primary-range person recall is not one of the twelve protected metrics,
    so it is supplied from the range stratification rather than read out of
    ``metrics``. Where that stratification does not exist -- the frozen noAE
    reference document publishes no per-bin slices -- the requirement is recorded
    as *not evaluable* and the result fails closed instead of reporting a
    fabricated miss.
    """
    if range_stratified is None:
        evaluated = tuple(
            row
            for row in LOCALIZATION_OBJECT_REQUIREMENTS
            if row[0] != PERSON_PRIMARY_RANGE_RECALL_METRIC
        )
        values: Mapping[str, Any] = dict(metrics)
        not_evaluable = [PERSON_PRIMARY_RANGE_RECALL_METRIC]
    else:
        evaluated = LOCALIZATION_OBJECT_REQUIREMENTS
        values = {
            **dict(metrics),
            PERSON_PRIMARY_RANGE_RECALL_METRIC: float(
                range_stratified[PERSON_PRIMARY_RANGE_RECALL_METRIC]
            ),
        }
        not_evaluable = []

    result = _requirement_rows(evaluated, values)
    result["basis"] = "absolute AVO>=0.65 person view and vehicle object metrics"
    result["segmentation_excluded"] = list(
        SEGMENTATION_EXCLUDED_FROM_LOCALIZATION_CLASSIFICATION
    )
    result["threshold_provenance"] = LOCALIZATION_THRESHOLD_PROVENANCE
    result["independent_test_set_confirmation"] = False
    result["untouched_test_set_confirmation"] = False
    result["test_split_accessed"] = False
    result["registered_total"] = len(LOCALIZATION_OBJECT_REQUIREMENTS)
    result["not_evaluable"] = list(not_evaluable)
    result["all_registered_requirements_evaluated"] = not not_evaluable
    if not_evaluable:
        result["not_evaluable_reason"] = RANGE_METRIC_UNAVAILABLE_REASON
        # Fail closed: an unevaluated requirement is never a pass.
        result["all_passed"] = False
    result["primary_operating_range"] = PEDESTRIAN_PRIMARY_RANGE
    result["extended_diagnostic_range"] = PEDESTRIAN_EXTENDED_DIAGNOSTIC_RANGE
    result["extended_range_is_a_tier_gate"] = False
    result["superseded_requirement"] = PERSON_HISTORICAL_LONG_RANGE_RECALL_METRIC
    result["range_provenance"] = PEDESTRIAN_RANGE_PROVENANCE
    result["evaluation_only_boundary_rule"] = EVALUATION_ONLY_BOUNDARY_RULE
    result.update(dict(EVALUATION_ONLY_BOUNDARY_DECLARATIONS))
    return result


def segmentation_installability(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Whether this profile authorizes replacing the spatial-map segmentation."""
    result = _requirement_rows(SEGMENTATION_INSTALL_REQUIREMENTS, metrics)
    installable = bool(result["all_passed"])
    result["rule"] = SEGMENTATION_INSTALL_RULE
    result["segmentation_installable"] = installable
    result["action"] = (
        SEGMENTATION_INSTALL_ACTION_INSTALL
        if installable
        else SEGMENTATION_INSTALL_ACTION_RETAIN
    )
    result["twelve_of_twelve_preservation_authorizes_install"] = False
    result["segmentation_layer_installed_here"] = False
    return result


def service_readiness(row: Mapping[str, Any]) -> dict[str, Any]:
    """The separate 9/9 absolute service result. Never inferred from 12/12."""
    gates = row["absolute_service_gates"]
    passed = int(gates["pass_count"])
    ready = passed == SERVICE_GATE_COUNT
    return {
        "rule": SERVICE_READY_RULE,
        "absolute_service_pass_count": passed,
        "absolute_service_gate_count": SERVICE_GATE_COUNT,
        "failed_absolute_service_gates": list(gates["failed"]),
        "service_ready": ready,
        "terminal": SERVICE_READY_TERMINAL if ready else None,
        "derived_from_relative_preservation": False,
        "registered_baseline_absolute_service_pass_count": (
            BASELINE_SERVICE_PASS_COUNT
        ),
    }


def integrity_verdict(row: Mapping[str, Any]) -> dict[str, Any]:
    """Transport/routing/numerical integrity of one recorded row.

    The pass itself already refuses to record a row that violates any of these,
    so this is a second, record-level reading of the same declarations: the tier
    is then a function of the durable record rather than of trust in the writer.
    """
    integrity = dict(row["integrity"])
    missing_true = [
        name for name in INTEGRITY_REQUIRED_TRUE if not bool(integrity.get(name))
    ]
    wrong_false = [
        name for name in INTEGRITY_REQUIRED_FALSE if bool(integrity.get(name, True))
    ]
    finite = bool(row.get("all_outputs_and_metrics_finite"))
    decompressions = int(integrity.get("zstd_decompressions", -1))
    one_per_frame = decompressions == contract.VALIDATION_FRAMES
    valid = not missing_true and not wrong_false and finite and one_per_frame
    return {
        "valid": bool(valid),
        "unset_required_true": missing_true,
        "set_required_false": wrong_false,
        "all_outputs_and_metrics_finite": finite,
        "zstd_decompressions": decompressions,
        "exactly_one_decompression_per_frame": one_per_frame,
    }


def classify_profile(
    *,
    bottleneck: int,
    row: Mapping[str, Any],
    full_preservation_passed: bool,
    full_preservation_basis: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify one measured profile independently, by the registered cascade.

    ``full_preservation_passed`` is read out of the already-computed primary
    acceptance result, never recomputed, so this classification cannot disagree
    with the primary rule about whether the primary rule passed.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    q = float(row["q"])
    stress = q in contract.EVALUATION_STRESS_Q_VALUES
    metrics = dict(row["metrics"])
    canonical = dict(row["canonical_person_metrics"])

    stratified = row.get(RANGE_STRATIFIED_KEY)
    if not isinstance(stratified, Mapping):
        raise guards.HybridQConfigError(
            "the measured row carries no person range stratification"
        )

    integrity = integrity_verdict(row)
    localization = localization_requirements(metrics, range_stratified=stratified)
    segmentation = segmentation_installability(metrics)
    service = service_readiness(row)

    if not integrity["valid"]:
        tier = TIER_INVALID
        reason = "transport, routing, numerical or execution integrity failed"
    elif stress:
        tier = TIER_EMERGENCY_ONLY
        reason = (
            "registered stress profile: q=0.90 and q=0.98 are EMERGENCY_ONLY "
            "regardless of their measured metrics"
        )
    elif full_preservation_passed:
        tier = TIER_FULL_PRESERVATION
        reason = "the existing primary twelve-gate relative rule passes here"
    elif localization["all_passed"]:
        tier = TIER_LOCALIZATION_PRIORITY
        reason = "every absolute AVO/object requirement passes"
    else:
        tier = TIER_EMERGENCY_ONLY
        reason = (
            "integrity passes but the absolute object requirement(s) "
            f"{localization['failed']} fail"
        )

    return {
        **family_fields(size),
        "q": q,
        "q_e4": int(row["q_e4"]),
        "tier": tier,
        "tier_reason": reason,
        "registered_tiers": [
            {"tier": name, "definition": definition}
            for name, definition in CLASSIFICATION_TIERS
        ],
        "is_registered_stress_profile": stress,
        "stress_profiles_are_emergency_only_regardless_of_metrics": True,
        "integrity": integrity,
        "full_preservation": {
            "passed": bool(full_preservation_passed),
            "source": "primary preregistered acceptance result, read not recomputed",
            "basis": dict(full_preservation_basis),
            "authorizes_segmentation_install": False,
            "implies_service_ready": False,
        },
        "localization_priority": localization,
        "person_range_stratified": dict(stratified),
        "segmentation": segmentation,
        "service_readiness": service,
        "canonical_person_diagnostics": {
            "metrics": canonical,
            "role": "diagnostic only",
            "used_for_localization_priority_classification": False,
            "classification_person_view": "AVO>=0.65 visible-object person view",
        },
        "availability": {
            "policy": MASKING_POLICY,
            "masked": False,
            "mask_reason": None,
            "masked_for_perception_degradation": False,
            "state_infeasible_definition": STATE_INFEASIBLE_DEFINITION,
            "state_infeasible_assignable_by_this_validation": False,
        },
        "changed_checkpoint_selection": False,
        "changed_primary_acceptance_terminal": False,
        "changed_threshold_nms_model_or_scorer": False,
        "reinterpreted_original_preservation_failures": False,
    }


def classify_profiles(
    *,
    bottleneck: int,
    rows: Sequence[Mapping[str, Any]],
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    """The whole secondary classification, one independent verdict per profile."""
    size = family.require_phase10_bottleneck(bottleneck)

    # Per-q the primary rule's own verdict, read straight out of the primary
    # result. Stress q are excluded there by construction and are
    # EMERGENCY_ONLY here regardless, so they carry no primary verdict.
    zero_e4 = continuous_q.quantize_q(0.0).q_e4
    basis: dict[int, dict[str, Any]] = {
        zero_e4: dict(acceptance["q0_condition"])
    }
    for entry in acceptance["primary_q_conditions"]:
        basis[continuous_q.quantize_q(float(entry["q"])).q_e4] = dict(entry)

    classified: list[dict[str, Any]] = []
    for row in rows:
        q_e4 = int(row["q_e4"])
        entry = basis.get(q_e4)
        if entry is None:
            passed = False
            detail: Mapping[str, Any] = {
                "primary_verdict_available": False,
                "reason": (
                    "registered stress profile; it cannot influence the primary "
                    "rule and carries no primary verdict"
                ),
            }
        else:
            passed = bool(entry.get("passed", entry.get("qualifies", False)))
            detail = {"primary_verdict_available": True, **entry}
        classified.append(
            classify_profile(
                bottleneck=size,
                row=row,
                full_preservation_passed=passed,
                full_preservation_basis=detail,
            )
        )

    tiers = {name for name, _definition in CLASSIFICATION_TIERS}
    tiers.add(TIER_STATE_INFEASIBLE)
    counts = {name: 0 for name in sorted(tiers)}
    for entry in classified:
        counts[entry["tier"]] += 1
    return {
        **family_fields(size),
        "purpose": (
            "a secondary, prospective, localization-priority classification of "
            "each measured profile against absolute AVO/object requirements, "
            "registered before any Phase-10B number existed"
        ),
        "registered_before_measurement": True,
        "object_requirements": [
            {"metric": name, "target": target, "direction": direction}
            for name, target, direction in LOCALIZATION_OBJECT_REQUIREMENTS
        ],
        "threshold_provenance": LOCALIZATION_THRESHOLD_PROVENANCE,
        "independent_test_set_confirmation": False,
        "untouched_test_set_confirmation": False,
        "segmentation_install_requirements": [
            {"metric": name, "target": target, "direction": direction}
            for name, target, direction in SEGMENTATION_INSTALL_REQUIREMENTS
        ],
        "segmentation_install_rule": SEGMENTATION_INSTALL_RULE,
        "segmentation_excluded_from_localization_classification": list(
            SEGMENTATION_EXCLUDED_FROM_LOCALIZATION_CLASSIFICATION
        ),
        "segmentation_measured_and_reported": True,
        "service_ready_rule": SERVICE_READY_RULE,
        "masking_policy": MASKING_POLICY,
        "state_infeasible": {
            "tier": TIER_STATE_INFEASIBLE,
            "definition": STATE_INFEASIBLE_DEFINITION,
            "assignable_by_this_validation": False,
        },
        "profiles": classified,
        "tier_counts": counts,
        "profiles_classified": len(classified),
        "every_measured_profile_classified_independently": (
            len(classified) == len(rows)
        ),
        "segmentation_installable_profiles": sorted(
            entry["q"]
            for entry in classified
            if entry["segmentation"]["segmentation_installable"]
        ),
        "service_ready_profiles": sorted(
            entry["q"]
            for entry in classified
            if entry["service_readiness"]["service_ready"]
        ),
        "changed_checkpoint_selection": False,
        "changed_primary_acceptance_terminal": False,
        "changed_any_threshold_nms_model_or_scorer": False,
        "erased_or_reinterpreted_original_preservation_failures": False,
        "is_secondary_to": "the primary twelve-gate preservation acceptance rule",
    }


def reference_feasibility(bottleneck: int) -> dict[str, Any]:
    """The registered classification applied to the *frozen noAE* reference rows.

    Evaluated at registration time from an already-published frozen document, so
    it contains no Phase-10B measurement. It exists so a reviewer can see which
    absolute requirements the frozen noAE UINT8+zstd path itself already misses,
    and therefore that these bars were registered with their difficulty visible
    rather than fitted afterwards to an AE result.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    rows = load_noae_reference()
    out: list[dict[str, Any]] = []
    for q in Q_VALUES:
        reference = rows[continuous_q.quantize_q(float(q)).q_e4]
        metrics = dict(reference["metrics"])
        localization = localization_requirements(metrics)
        segmentation = segmentation_installability(metrics)
        out.append(
            {
                "q": float(reference["q"]),
                "q_e4": int(reference["q_e4"]),
                "object_requirements_passed": localization["passed_count"],
                "object_requirements_evaluated": localization["total"],
                "object_requirements_registered_total": localization[
                    "registered_total"
                ],
                "not_evaluable_object_requirements": localization["not_evaluable"],
                "failed_object_requirements": localization["failed"],
                "object_requirements_all_passed": localization["all_passed"],
                "segmentation_installable": segmentation["segmentation_installable"],
                "failed_segmentation_requirements": segmentation["failed"],
                "absolute_service_pass_count": int(
                    reference["absolute_service_pass_count"]
                ),
            }
        )
    return {
        **family_fields(size),
        "source": NOAE_UINT8_VALIDATION_RELPATH,
        "source_sha256": NOAE_UINT8_VALIDATION_SHA256,
        "contains_phase10b_measurement": False,
        "purpose": (
            "registration-time visibility of how hard the absolute requirements "
            "are on the frozen noAE UINT8+zstd path itself"
        ),
        "not_evaluable_requirements": [PERSON_PRIMARY_RANGE_RECALL_METRIC],
        "not_evaluable_reason": RANGE_METRIC_UNAVAILABLE_REASON,
        "primary_operating_range": PEDESTRIAN_PRIMARY_RANGE,
        "extended_diagnostic_range": PEDESTRIAN_EXTENDED_DIAGNOSTIC_RANGE,
        "range_provenance": PEDESTRIAN_RANGE_PROVENANCE,
        "evaluation_only_boundary_rule": EVALUATION_ONLY_BOUNDARY_RULE,
        "per_q": out,
    }


# ---------------------------------------------------------------------------
# One frame through the deployment path
# ---------------------------------------------------------------------------


def _transport_one(
    *,
    bottleneck: int,
    frame: torch.Tensor,
    autoencoder: Any,
    ranker: _CountingRanker,
    decoders: ae_family_dispatch.PreloadedAeDecoders,
    plan: continuous_q.ContinuousQ,
    wire: CountingWireCodec,
    device: torch.device,
    expected_tag: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Transmit and receive one frame, then audit every registered invariant.

    Returns the reconstructed FP32 C2 the frozen tail will consume, plus the
    per-frame integrity and cost row. Nothing here reimplements a codec step:
    the transmit side is ``ae_uint8_transport.encode_frame`` and the receive side
    is ``PreloadedAeDecoders.receive`` over the raw wire bytes. Every family
    check reads the value the *received header* declared and compares it against
    the family derived from ``--bottleneck``.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    width = latent_width(size)
    expected_family = family.family_id(size)
    label = family.family_label(size)

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
    if int(transport.family_id) != expected_family:
        raise guards.HybridQPayloadError(
            f"the transmitted frame is not {label}"
        )
    if int(transport.bottleneck) != width:
        raise guards.HybridQPayloadError("transmitted latent width drift")

    analytical = analytical_size(size, plan.wire_q)
    if packet.uncompressed_bytes != analytical.total_bytes:
        raise guards.HybridQPayloadError("pre-zstd AE payload size drift")
    if analytical.range_bytes != range_bytes(size):
        raise guards.HybridQPayloadError("per-channel range byte accounting drift")
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
            f"the header-selected decoder is not the preloaded selected {label}"
        )
    if received.family.family_id != expected_family:
        raise guards.HybridQPayloadError(f"received family is not {label}")
    if received.family.transported_channels != width:
        raise guards.HybridQPayloadError(
            f"received latent width is not {width} channels"
        )
    if received.family.routing_tag != expected_tag:
        raise guards.HybridQPayloadError("received routing tag is not the bound tag")
    if received.family.codec != "ae_latent_uint8":
        raise guards.HybridQPayloadError("received codec identity drift")
    if int(parsed.family_id) != expected_family:
        raise guards.HybridQPayloadError("received header family drift")
    if int(parsed.bottleneck) != width:
        raise guards.HybridQPayloadError("received header latent width drift")
    if int(parsed.routing_tag) != expected_tag:
        raise guards.HybridQPayloadError("received header routing tag drift")
    if int(parsed.header.q_e4) != plan.q_e4:
        raise guards.HybridQPayloadError("received header q drift")
    if continuous_q.quantize_q(received.q).q_e4 != plan.q_e4:
        raise guards.HybridQPayloadError("decoded AE q drift")
    guards.require_keep_cardinality(int(received.keep_count), plan.keep_count)
    guards.require_keep_cardinality(int(diagnostics.keep_mask.sum()), plan.keep_count)
    if int(parsed.header.keep_count) != plan.keep_count:
        raise guards.HybridQPayloadError("received header keep-count drift")
    if int(parsed.values.shape[0]) != plan.keep_count or int(
        parsed.values.shape[1]
    ) != width:
        raise guards.HybridQPayloadError("retained UINT8 value block shape drift")

    # The retained UINT8 cells are exactly the cells selection chose.
    if plan.is_bypass:
        expected_indices = torch.arange(
            ae_contract.AE_LATENT_CELLS, dtype=torch.int64
        )
    else:
        expected_indices = (
            transport.selection.keep_indices.detach().to(
                device="cpu", dtype=torch.int64
            )
        )
    if not torch.equal(parsed.keep_indices, expected_indices):
        raise guards.HybridQPayloadError(
            "the retained UINT8 cells are not the selected cells"
        )

    # Dropped cells scatter to exact zero across all transported latent channels
    # before the decoder ever runs.
    flat = diagnostics.latent.reshape(width, ae_contract.AE_LATENT_CELLS)
    occupied = (flat != 0).any(dim=0)
    dropped = ~diagnostics.keep_mask.reshape(-1)
    if int(dropped.sum()) != plan.drop_count:
        raise guards.HybridQPayloadError("reconstructed drop cardinality drift")
    if bool(occupied[dropped].any()):
        raise guards.HybridQNumericalError(
            "a dropped latent cell did not scatter to exact zero"
        )

    reconstructed = received.c2
    guards.require_frozen_c2(
        reconstructed, what=f"reconstructed {label} validation C2"
    )
    # q=0 bypasses the ranker but not the AE, so no q is ever an identity.
    if torch.equal(reconstructed, frame):
        raise guards.HybridQNumericalError(
            f"the {label} reconstruction is bit-identical to the original C2"
        )

    row = {
        "pre_zstd_bytes": int(packet.uncompressed_bytes),
        "zstd_bytes": int(packet.compressed_bytes),
        "keep_count": int(received.keep_count),
        "family_id": int(received.family.family_id),
        "latent_channels": int(received.family.transported_channels),
        "routing_tag": int(received.family.routing_tag),
        "ranker_invocations": int(ranker.invocations),
        "zstd_decompressions": int(decompressions),
        "transmit_ns": int(transmit_ns),
        "receive_ns": int(receive_ns),
    }
    return reconstructed, row


# ---------------------------------------------------------------------------
# One complete validation pass at one q
# ---------------------------------------------------------------------------


def _load_runtime(bottleneck: int, device: torch.device) -> dict[str, Any]:
    """The one frozen model/ranker and the registered validation ordering."""
    size = family.require_phase10_bottleneck(bottleneck)
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
        "bottleneck": size,
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
    """The one and only complete UINT8+zstd inference pass for one family/q."""
    size = family.require_phase10_bottleneck(runtime["bottleneck"])
    width = latent_width(size)
    label = family.family_label(size)
    expected_tag = routing_tag(size)
    expected_family = family.family_id(size)

    plan = continuous_q.quantize_q(q)
    if not plan.is_registered:
        raise guards.HybridQConfigError("Phase 10B accepts registered q only")
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
    families: set[int] = set()
    widths: set[int] = set()
    tags: set[int] = set()
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
                        bottleneck=size,
                        frame=c2[index],
                        autoencoder=autoencoder,
                        ranker=ranker,
                        decoders=decoders,
                        plan=plan,
                        wire=wire,
                        device=device,
                        expected_tag=expected_tag,
                    )
                    pre_zstd_sizes.add(frame_row["pre_zstd_bytes"])
                    pre_zstd_bytes.append(frame_row["pre_zstd_bytes"])
                    zstd_sizes.append(frame_row["zstd_bytes"])
                    keep_counts.add(frame_row["keep_count"])
                    families.add(frame_row["family_id"])
                    widths.add(frame_row["latent_channels"])
                    tags.add(frame_row["routing_tag"])
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
    analytical = analytical_size(size, plan.wire_q)
    if pre_zstd_sizes != {analytical.total_bytes}:
        raise guards.HybridQPayloadError(f"pre-zstd payload drift: {pre_zstd_sizes}")
    if len(zstd_sizes) != contract.VALIDATION_FRAMES:
        raise guards.HybridQPayloadError("zstd payload count drift")
    if len(pre_zstd_bytes) != contract.VALIDATION_FRAMES:
        raise guards.HybridQPayloadError("pre-zstd payload count drift")
    if keep_counts != {plan.keep_count}:
        raise guards.HybridQPayloadError(f"keep-count drift: {sorted(keep_counts)}")
    if families != {expected_family}:
        raise guards.HybridQPayloadError(f"received family drift: {sorted(families)}")
    if widths != {width}:
        raise guards.HybridQPayloadError(f"received latent width drift: {sorted(widths)}")
    if tags != {expected_tag}:
        raise guards.HybridQPayloadError("received routing tag drift")
    expected_ranker_calls = 0 if plan.is_bypass else contract.VALIDATION_FRAMES
    if ranker_invocations != expected_ranker_calls:
        raise guards.HybridQPayloadError("ranker invocation count drift")
    if decompressions != contract.VALIDATION_FRAMES:
        raise guards.HybridQPayloadError(
            "the pass did not perform exactly one zstd decompression per frame"
        )

    return {
        **family_fields(size),
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
            "family": label,
            "family_id": expected_family,
            "transported_latent_channels": width,
            "routing_tag": expected_tag,
            "analytical_pre_zstd_bytes": analytical.total_bytes,
            "analytical_breakdown": {
                "header_bytes": analytical.header_bytes,
                "mask_bytes": analytical.mask_bytes,
                "range_bytes": analytical.range_bytes,
                "value_bytes": analytical.value_bytes,
            },
            "range_bytes_derivation": (
                f"{width} transported latent channels x one FP32 min/max pair = "
                f"{range_bytes(size)} bytes, from "
                "ae_uint8_transport.range_byte_count(bottleneck)"
            ),
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
                "stable epoch-4 ranker and stable per-frame top-K selection "
                "(q>0 only)",
                f"{label} encoder on the complete frame",
                "per-channel range computation from the complete latent",
                "UINT8 quantization and family-labelled sparse AE framing",
                "mandatory zstd level-1 compression",
            ],
            "receive_includes": [
                "exactly one zstd decompression",
                "AE header inspection and preloaded-decoder selection by "
                "family/bottleneck/routing tag",
                "UINT8 dequantization and zero scatter",
                f"{label} decoder",
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
            "ae_encoder_bypassed": False,
            "ranked_original_fp32_c2_per_frame": not plan.is_bypass,
            "selection_independent_per_frame": True,
            "batched_or_cross_frame_selection_used": False,
            "stable_per_frame_top_k_for_q_above_zero": not plan.is_bypass,
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
            "received_family_matches_selected_family": True,
            "received_latent_width_matches_bottleneck": True,
            "received_routing_tag_matches_bound_tag": True,
            "received_family_ids": sorted(families),
            "received_latent_widths": sorted(widths),
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
# Per-q setting artifact and the proven Phase-9D durability semantics
# ---------------------------------------------------------------------------

# The four canonical-p025 person metrics, reported as diagnostics beside the
# AVO>=0.65 person view the gates and the secondary classification use.
CANONICAL_PERSON_METRICS = tuple(phase9d._CSV_CANONICAL)

PRESERVATION_KEY = "same_q_preservation_vs_noae_uint8_zstd"
DELTA_KEY = "protected_metric_deltas_vs_noae_uint8_zstd_same_q"


def _setting_document(
    *,
    bottleneck: int,
    raw: Mapping[str, Any],
    scored: Mapping[str, Any],
    reference: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """One durable, self-describing scientific record of one family/q pass."""
    size = family.require_phase10_bottleneck(bottleneck)
    q = float(raw["q"])
    preservation = common.evaluate_same_q_gates(
        reference["metrics"], scored["metrics"], baseline=SAME_Q_BASELINE_LABEL
    )
    deltas = {
        name: {
            "noae_uint8_zstd_same_q": float(reference["metrics"][name]),
            "ae_uint8_zstd_same_q": float(scored["metrics"][name]),
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
        raise guards.HybridQNumericalError(
            f"a scored {family.family_label(size)} metric is non-finite"
        )
    stress = q in contract.EVALUATION_STRESS_Q_VALUES
    return {
        "schema": setting_schema(size),
        "terminal": setting_terminal(size),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "run_identity_sha256": identity["sha256"],
        **dict(raw),
        "metrics": dict(scored["metrics"]),
        "canonical_person_metrics": dict(scored["canonical_person_metrics"]),
        RANGE_STRATIFIED_KEY: person_range_stratification(
            scored["person_avo_detail"]["distance_bins"], scored["metrics"]
        ),
        "canonical_person_metrics_role": (
            "diagnostic only; the gates and the secondary localization-priority "
            "classification use the AVO>=0.65 person view"
        ),
        "absolute_service_gates": dict(scored["absolute_service_gates"]),
        PRESERVATION_KEY: preservation,
        DELTA_KEY: deltas,
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
        "all_outputs_and_metrics_finite": True,
        "frozen_perception_state_unchanged": True,
        "stable_ranker_state_unchanged": True,
        "selected_ae_state_unchanged": True,
        "inference_passes_for_this_q": 1,
        "training_or_tuning": False,
        "threshold_nms_or_gate_change": False,
        "test_or_carla_access": False,
        "prediction_artifacts": {
            "root": str(raw["prediction_root"]),
            "removed_before_this_record": False,
            "removal_marker": f"{CLEANUP_DIRNAME}/{_q_slug(q)}.json",
            "durability_order": list(DURABILITY_ORDER),
            "rule": (
                "this record is the durable scientific completion record for "
                "this q and is fsynced into place before its predictions are "
                "removed, so an interruption can only lose scratch predictions"
            ),
        },
    }


def setting_path(output: Path, bottleneck: int, q: float) -> Path:
    family.require_phase10_bottleneck(bottleneck)
    return Path(output) / SETTINGS_DIRNAME / f"{_q_slug(q)}.json"


def cleanup_marker_path(output: Path, bottleneck: int, q: float) -> Path:
    family.require_phase10_bottleneck(bottleneck)
    return Path(output) / CLEANUP_DIRNAME / f"{_q_slug(q)}.json"


def prediction_root(output: Path, bottleneck: int, q: float) -> Path:
    family.require_phase10_bottleneck(bottleneck)
    return Path(output) / WORKING_DIRNAME / _q_slug(q)


def load_durable_setting(
    path: Path, bottleneck: int, q: float, identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Fully validate one durable per-q setting record before it may be reused.

    A setting JSON is the only thing that lets Phase 10B skip a q, so it is
    validated in full rather than spot-checked: schema and terminal, run
    identity, family identity, q, the registered frame count and single pass,
    the registered keep/drop cardinality and the family's exact analytical
    payload size, the finite-result flags, the expected ranker and zstd
    decompression counts, the frozen-state flags, and the complete metric and
    gate structure. Anything short of a complete, self-consistent record raises
    instead of being reused, overwritten or silently remeasured.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    plan = continuous_q.quantize_q(q)

    def fail(reason: str) -> None:
        raise guards.HybridQConfigError(f"{path}: {reason}")

    if (
        document.get("schema") != setting_schema(size)
        or document.get("terminal") != setting_terminal(size)
    ):
        fail("incomplete or foreign setting artifact")
    require_family_identity(document, size, what=str(path))
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

    # Payload: the pre-zstd size is exact, analytical and family-derived, so
    # every statistic of it must be that one value, and the wire must have one
    # sample per frame.
    analytical = analytical_size(size, plan.wire_q)
    payload = document.get("payload")
    if not isinstance(payload, Mapping):
        fail("no payload block")
    if int(payload.get("transported_latent_channels", -1)) != latent_width(size):
        fail("transported latent width mismatch")
    if int(payload.get("family_id", -1)) != family.family_id(size):
        fail("payload family id mismatch")
    if int(payload.get("routing_tag", -1)) != routing_tag(size):
        fail("payload routing tag mismatch")
    if int(payload.get("analytical_pre_zstd_bytes", -1)) != analytical.total_bytes:
        fail("analytical pre-zstd payload size mismatch")
    breakdown = payload.get("analytical_breakdown")
    if not isinstance(breakdown, Mapping):
        fail("no analytical payload breakdown")
    if int(breakdown.get("range_bytes", -1)) != range_bytes(size):
        fail("analytical range-byte accounting mismatch")
    if int(breakdown.get("value_bytes", -1)) != plan.keep_count * latent_width(size):
        fail("analytical value-byte accounting mismatch")
    pre_zstd = payload.get("pre_zstd_bytes")
    zstd = payload.get("zstd_bytes")
    if not isinstance(pre_zstd, Mapping) or not isinstance(zstd, Mapping):
        fail("no measured payload statistics")
    for name in ("mean", "median", "p95", "minimum", "maximum"):
        if float(pre_zstd.get(name, -1.0)) != float(analytical.total_bytes):
            fail(f"measured pre-zstd {name} is not the analytical payload size")
    for block, block_label in ((pre_zstd, "pre-zstd"), (zstd, "zstd")):
        if int(block.get("samples", -1)) != contract.VALIDATION_FRAMES:
            fail(f"{block_label} payload sample count mismatch")
        if not all(
            math.isfinite(float(block[name]))
            for name in ("mean", "median", "p95", "minimum", "maximum")
        ):
            fail(f"non-finite {block_label} payload statistic")
    if not bool(payload.get("zstd_mandatory")):
        fail("the record does not report zstd as mandatory")

    # Integrity: exactly the invocation, family, routing and decompression
    # declarations one complete pass owes.
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
    if list(integrity.get("received_family_ids", ())) != [family.family_id(size)]:
        fail("recorded received family drift")
    if list(integrity.get("received_latent_widths", ())) != [latent_width(size)]:
        fail("recorded received latent width drift")
    for flag in INTEGRITY_REQUIRED_TRUE:
        if not bool(integrity.get(flag)):
            fail(f"integrity flag {flag} is not set")
    for flag in INTEGRITY_REQUIRED_FALSE:
        if bool(integrity.get(flag, True)):
            fail(f"integrity flag {flag} must be false")

    # Finite-result and frozen-state declarations.
    for flag in (
        "all_outputs_and_metrics_finite",
        "frozen_perception_state_unchanged",
        "stable_ranker_state_unchanged",
        "selected_ae_state_unchanged",
    ):
        if not bool(document.get(flag)):
            fail(f"{flag} is not set")
    for flag in (
        "training_or_tuning",
        "threshold_nms_or_gate_change",
        "test_or_carla_access",
    ):
        if bool(document.get(flag, True)):
            fail(f"{flag} must be false")

    # Complete metric and gate structure.
    metrics = document.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(
        contract.PROTECTED_METRICS
    ):
        fail("protected metric set is incomplete")
    canonical = document.get("canonical_person_metrics")
    if not isinstance(canonical, Mapping) or set(canonical) != set(
        CANONICAL_PERSON_METRICS
    ):
        fail("canonical person metric set is incomplete")
    if not all(
        math.isfinite(float(value))
        for value in list(metrics.values()) + list(canonical.values())
    ):
        fail("a recorded metric is non-finite")

    stratified = document.get(RANGE_STRATIFIED_KEY)
    if not isinstance(stratified, Mapping):
        fail("no person range-stratification block")
    recorded_bins = stratified.get("bins")
    if not isinstance(recorded_bins, Mapping) or set(recorded_bins) != set(
        PEDESTRIAN_PRIMARY_RANGE_BINS + PEDESTRIAN_EXTENDED_RANGE_BINS
    ):
        fail("person range-stratification bin set is incomplete")
    rebuilt = person_range_stratification(recorded_bins, metrics)
    if float(
        stratified.get(PERSON_PRIMARY_RANGE_RECALL_METRIC, float("nan"))
    ) != float(rebuilt[PERSON_PRIMARY_RANGE_RECALL_METRIC]):
        fail("recorded primary-range person recall is inconsistent with its bins")
    if not all(
        bool(stratified.get(name)) == bool(value)
        for name, value in EVALUATION_ONLY_BOUNDARY_DECLARATIONS.items()
    ):
        fail("evaluation-only range declarations drift")

    service = document.get("absolute_service_gates")
    if not isinstance(service, Mapping) or len(
        service.get("targets", ())
    ) != SERVICE_GATE_COUNT:
        fail("absolute service-gate set is incomplete")
    targets = service["targets"]
    if {name for name, _target, _direction in contract.ABSOLUTE_SERVICE_TARGETS} != set(
        targets
    ):
        fail("absolute service-gate names drift")
    passed = sum(1 for row in targets.values() if bool(row.get("passed")))
    if int(service.get("pass_count", -1)) != passed:
        fail("absolute service-gate pass count is inconsistent")
    if sorted(service.get("failed", ())) != sorted(
        name for name, row in targets.items() if not bool(row.get("passed"))
    ):
        fail("absolute service-gate failure list is inconsistent")

    preservation = document.get(PRESERVATION_KEY)
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
    deltas = document.get(DELTA_KEY)
    if not isinstance(deltas, Mapping) or set(deltas) != set(
        contract.PROTECTED_METRICS
    ):
        fail("protected metric delta set is incomplete")

    reference = document.get("noae_same_q_reference")
    if not isinstance(reference, Mapping):
        fail("no frozen noAE same-q reference")
    if int(reference.get("q_e4", -1)) != plan.q_e4:
        fail("the recorded noAE reference is for a different q")
    if int(reference.get("retained_cells", -1)) != plan.keep_count:
        fail("the recorded noAE reference keep count drifts")
    if not 0 <= int(
        reference.get("absolute_service_pass_count", -1)
    ) <= SERVICE_GATE_COUNT:
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
    bottleneck: int,
    q: float,
    identity: Mapping[str, Any],
    setting_sha256: str,
    predictions: Path,
) -> dict[str, Any]:
    """Remove one q's scratch predictions, then mark the cleanup durable.

    Called only after that q's setting JSON is durably on disk. Idempotent: a
    resume that finds the directory already gone still writes the marker, and a
    resume that finds the marker complete does not come here at all.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    plan = continuous_q.quantize_q(q)
    predictions = Path(predictions)
    if predictions.exists():
        shutil.rmtree(predictions)
    document = {
        "schema": cleanup_schema(size),
        "terminal": cleanup_terminal(size),
        **family_fields(size),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "run_identity_sha256": identity["sha256"],
        "q": plan.wire_q,
        "q_e4": plan.q_e4,
        "setting_path": f"{SETTINGS_DIRNAME}/{_q_slug(q)}.json",
        "setting_sha256": str(setting_sha256),
        "prediction_root": str(predictions),
        "prediction_artifacts_removed_after_scoring": True,
        "durability_order": list(DURABILITY_ORDER),
    }
    path = cleanup_marker_path(output, size, q)
    _atomic_json(path, document)
    return {**document, "path": str(path)}


def cleanup_is_complete(
    output: Path,
    bottleneck: int,
    q: float,
    identity: Mapping[str, Any],
    setting_sha256: str,
) -> bool:
    """True only for a marker of this family that binds exactly this setting."""
    size = family.require_phase10_bottleneck(bottleneck)
    path = cleanup_marker_path(output, size, q)
    if not path.is_file():
        return False
    document = json.loads(path.read_text(encoding="utf-8"))
    plan = continuous_q.quantize_q(q)
    return bool(
        document.get("schema") == cleanup_schema(size)
        and document.get("terminal") == cleanup_terminal(size)
        and document.get("family") == family.family_label(size)
        and int(document.get("bottleneck", -1)) == size
        and document.get("run_identity_sha256") == identity["sha256"]
        and int(document.get("q_e4", -1)) == plan.q_e4
        and document.get("setting_sha256") == str(setting_sha256)
        and bool(document.get("prediction_artifacts_removed_after_scoring"))
    )


def reuse_or_complete(
    *, output: Path, bottleneck: int, q: float, identity: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Resume step for one q. Measures nothing, ever.

    Returns the fully validated durable setting when one exists, having finished
    that q's cleanup if an interruption left it unfinished. Returns None only
    when no durable record exists, which is the single case in which the caller
    is allowed to run the inference/evaluation pass.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    path = setting_path(output, size, q)
    if not path.is_file():
        return None
    document = load_durable_setting(path, size, q, identity)
    digest = sha256_file(path)
    if not cleanup_is_complete(output, size, q, identity, digest):
        marker = complete_cleanup(
            output=output,
            bottleneck=size,
            q=q,
            identity=identity,
            setting_sha256=digest,
            predictions=prediction_root(output, size, q),
        )
        print(
            json.dumps(
                {
                    "family": family.family_label(size),
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
    bottleneck: int, row: Mapping[str, Any], q0: Mapping[str, Any]
) -> dict[str, Any]:
    """Ratios against the three registered payload references."""
    size = family.require_phase10_bottleneck(bottleneck)
    label = family.family_label(size)
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
        "vs_selected_family_uint8_zstd_q0": {
            "family": label,
            "reference_pre_zstd_bytes": float(q0_payload["pre_zstd_bytes"]["mean"]),
            "reference_zstd_mean_bytes": float(q0_payload["zstd_bytes"]["mean"]),
            "reference_zstd_median_bytes": float(q0_payload["zstd_bytes"]["median"]),
            "pre_zstd_mean": pre["mean"] / float(q0_payload["pre_zstd_bytes"]["mean"]),
            "zstd_mean": zstd["mean"] / float(q0_payload["zstd_bytes"]["mean"]),
            "zstd_median": zstd["median"] / float(q0_payload["zstd_bytes"]["median"]),
        },
    }


_CSV_METRICS = tuple(contract.PROTECTED_METRICS)


def _csv_text(bottleneck: int, rows: Sequence[Mapping[str, Any]]) -> str:
    """One row per measured q. Every measured q appears, whatever the decision."""
    family.require_phase10_bottleneck(bottleneck)
    columns = [
        "family", "bottleneck", "q", "q_e4", "retained_cells",
        "pre_zstd_bytes_mean", "pre_zstd_bytes_median", "pre_zstd_bytes_p95",
        "zstd_bytes_mean", "zstd_bytes_median", "zstd_bytes_p95",
        "zstd_bytes_min", "zstd_bytes_max",
        "zstd_ratio_vs_framed_fp32_noae_q0_median",
        "zstd_ratio_vs_noae_uint8_zstd_same_q_median",
        "zstd_ratio_vs_family_uint8_zstd_q0_median",
        *_CSV_METRICS, *CANONICAL_PERSON_METRICS,
        "absolute_service_pass_count", "failed_absolute_service_gates",
        "same_q_preservation_pass_count", "same_q_preservation_all_passed",
        "failed_same_q_preservation_gates", "worst_normalized_degradation",
        "profile_designation", "influences_acceptance", "all_outputs_finite",
        # Secondary prospective classification, reported beside the primary
        # numbers it cannot change.
        "secondary_tier", "localization_requirements_passed",
        "failed_localization_requirements", "segmentation_installable",
        "segmentation_install_action", "failed_segmentation_requirements",
        "service_ready",
        # Range-stratified person reporting. Only person_avo_recall_0_30m is a
        # tier gate; the rest are reported and never gated. Per-band precision is
        # not derivable from the frozen recall slices.
        "person_avo_recall_0_30m", "person_avo_recall_00_10m",
        "person_avo_recall_10_20m", "person_avo_recall_20_30m",
        "person_avo_recall_30_40m", "person_avo_recall_20_40m_historical",
    ]
    columns += [f"degradation_{name}" for name in _CSV_METRICS]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        payload = row["payload"]
        ratios = row["payload_ratios"]
        preservation = row[PRESERVATION_KEY]
        deltas = row[DELTA_KEY]
        secondary = row["secondary_classification"]
        localization = secondary["localization_priority"]
        segmentation = secondary["segmentation"]
        stratified = row[RANGE_STRATIFIED_KEY]
        writer.writerow(
            {
                "family": row["family"],
                "bottleneck": row["bottleneck"],
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
                "zstd_ratio_vs_family_uint8_zstd_q0_median": ratios[
                    "vs_selected_family_uint8_zstd_q0"
                ]["zstd_median"],
                **{name: row["metrics"][name] for name in _CSV_METRICS},
                **{
                    name: row["canonical_person_metrics"][name]
                    for name in CANONICAL_PERSON_METRICS
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
                "secondary_tier": secondary["tier"],
                "localization_requirements_passed": (
                    f"{localization['passed_count']}/{localization['total']}"
                ),
                "failed_localization_requirements": ";".join(localization["failed"]),
                "segmentation_installable": segmentation["segmentation_installable"],
                "segmentation_install_action": segmentation["action"],
                "failed_segmentation_requirements": ";".join(segmentation["failed"]),
                "service_ready": secondary["service_readiness"]["service_ready"],
                "person_avo_recall_0_30m": stratified[
                    PERSON_PRIMARY_RANGE_RECALL_METRIC
                ],
                **{
                    f"person_avo_recall_{name}": stratified["bins"][name]["recall"]
                    for name in PEDESTRIAN_PRIMARY_RANGE_BINS
                    + PEDESTRIAN_EXTENDED_RANGE_BINS
                },
                "person_avo_recall_20_40m_historical": stratified[
                    "historical_20_40m"
                ]["recall"],
                **{
                    f"degradation_{name}": deltas[name]["degradation"]
                    for name in _CSV_METRICS
                },
            }
        )
    return stream.getvalue()


def _report_text(bottleneck: int, document: Mapping[str, Any]) -> str:
    size = family.require_phase10_bottleneck(bottleneck)
    label = family.family_label(size)
    width = latent_width(size)
    rows = document["curve"]
    acceptance = document["preregistered_interpretation"]
    secondary = document["secondary_prospective_classification"]
    lines = [
        f"# Phase 10B — selected {label} UINT8 + mandatory-zstd validation",
        "",
        f"Generated {document['generated_utc']} · terminal "
        f"`{terminal(size)}`",
        "",
        "One frozen measurement of the six registered q anchors on the "
        f"{contract.VALIDATION_FRAMES:,} registered validation frames, one "
        "inference/evaluation pass per q. Nothing was trained, tuned, "
        "recalibrated, reselected or removed; no threshold, NMS setting, scorer "
        "or geometry evaluator changed; test data and CARLA were never opened. "
        "Component latency below is current-host diagnostic evidence only — no "
        "Raspberry Pi and no OAI latency is claimed.",
        "",
        "## Deployment path measured",
        "",
        "```text",
        f"original FP32 C2 -> {label} encoder (complete frame)",
        "  -> ranges from the complete latent -> per-channel UINT8",
        "  -> stable per-frame top-K (q>0) -> family-labelled sparse wire",
        "  -> mandatory zstd-1 -> received raw bytes",
        "  -> exactly one decompression",
        "  -> decoder selected from header family/bottleneck/routing tag",
        f"  -> dequantize / zero scatter -> {label} decoder",
        "  -> unchanged frozen perception tail and p025 service policy",
        "```",
        "",
        f"Selected checkpoint "
        f"`{document['selected_ae']['selected_checkpoint_path']}` "
        f"(sha256 `{document['selected_ae']['selected_checkpoint_sha256']}`), "
        f"epoch {document['selected_ae']['epoch']}, {width}-channel latent, "
        f"routing tag `{document['selected_ae']['routing_tag_hex']}` derived "
        "from that full digest. The 32-bit tag routes a frame to the decoder "
        "that produced it; it is not the checkpoint's identity.",
        "",
        "## Payload",
        "",
        "| q | keep | pre-zstd mean B | median | p95 | zstd mean B | median | p95 | "
        f"vs framed FP32 noAE q0 | vs noAE UINT8+zstd same q | vs {label} "
        "UINT8+zstd q0 |",
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
            f"{ratios['vs_selected_family_uint8_zstd_q0']['zstd_median']:.6f} |"
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
        "AVO>=0.65 person P/R/F1/XY | person 20–40 m recall (historical) | "
        "vehicle IoU | "
        "person box-mask IoU | foreground mIoU | service gates | same-q gates | "
        "profile |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        m = row["metrics"]
        c = row["canonical_person_metrics"]
        preservation = row[PRESERVATION_KEY]
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
        "Canonical-p025 person metrics are diagnostics. The twelve preservation "
        "gates and the secondary localization-priority classification both use "
        "the AVO>=0.65 visible-object person view.",
        "",
        "## Pedestrian range stratification",
        "",
        f"Primary operating range: `{PEDESTRIAN_PRIMARY_RANGE}`. Extended "
        f"diagnostic range: `{PEDESTRIAN_EXTENDED_DIAGNOSTIC_RANGE}`. Only "
        f"`{PERSON_PRIMARY_RANGE_RECALL_METRIC} >= 0.70` is an absolute tier "
        "gate; every other row below is reported and never gated.",
        "",
        EVALUATION_ONLY_BOUNDARY_RULE,
        "",
        "| q | 0-10 m R | 10-20 m R | 20-30 m R | **0-30 m R (gate)** | "
        "30-40 m R (stress) | 20-40 m R (historical) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def _recall(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.6f}"

    for row in rows:
        stratified = row[RANGE_STRATIFIED_KEY]
        bins = stratified["bins"]
        lines.append(
            f"| {row['q']:.2f} | {_recall(bins['00_10m']['recall'])} | "
            f"{_recall(bins['10_20m']['recall'])} | "
            f"{_recall(bins['20_30m']['recall'])} | "
            f"**{_recall(stratified[PERSON_PRIMARY_RANGE_RECALL_METRIC])}** | "
            f"{_recall(bins['30_40m']['recall'])} | "
            f"{_recall(stratified['historical_20_40m']['recall'])} |"
        )
    lines += [
        "",
        "20-30 m is shown on its own so the cumulative 0-30 m result cannot hide "
        "boundary-band behaviour, and 30-40 m is retained as extended-range "
        "stress. " + PER_BAND_PRECISION_UNAVAILABLE_REASON,
        "",
        "Range provenance:",
        "",
        "> " + PEDESTRIAN_RANGE_PROVENANCE,
        "",
        "## Failed gates and exact degradations",
        "",
        "| q | failed same-q preservation gates | degradation / bound | "
        "failed absolute service gates |",
        "| ---: | --- | --- | --- |",
    ]
    for row in rows:
        preservation = row[PRESERVATION_KEY]
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
        "## Primary preregistered interpretation (relative, 12 gates)",
        "",
        acceptance["rule"],
        "",
        f"- q=0 condition: "
        f"**{'met' if acceptance['q0_condition']['passed'] else 'not met'}** "
        f"({acceptance['q0_condition']['preservation_gates_passed']}/{GATE_COUNT} "
        "same-q gates, "
        f"{acceptance['q0_condition']['absolute_service_pass_count']}/"
        f"{SERVICE_GATE_COUNT} absolute service gates against a "
        f"{BASELINE_SERVICE_PASS_COUNT}/{SERVICE_GATE_COUNT} baseline)",
        f"- qualifying primary q: {acceptance['qualifying_primary_q'] or 'none'}",
        f"- **decision: {acceptance['decision']}**",
        "- q=0.90 and q=0.98 are stress/emergency profiles regardless of their "
        "results and did not enter the decision",
        "- this decision is a *relative* preservation result against the frozen "
        "noAE UINT8+zstd row at the same q. It is not an absolute service claim, "
        "and it does not by itself authorize replacing the spatial-map "
        "segmentation layer",
        "- every measured q is reported above whatever this decision was: a "
        f"failed acceptance suppressed {document['integrity']['rows_suppressed_by_failed_acceptance']} rows",
        "",
        "## Secondary prospective classification (absolute AVO/object)",
        "",
        secondary["threshold_provenance"],
        "",
        "This is not an independent or untouched test-set confirmation. It "
        "changed no checkpoint selection, no primary acceptance terminal, and no "
        "threshold, NMS setting, model or scorer, and it neither erases nor "
        "reinterprets any preservation failure recorded above.",
        "",
        "| requirement | target |",
        "| --- | ---: |",
    ]
    for name, target, direction in LOCALIZATION_OBJECT_REQUIREMENTS:
        comparator = ">=" if direction == "higher" else "<="
        lines.append(f"| `{name}` | {comparator} {target} |")
    lines += [
        "",
        "The three segmentation outputs — vehicle IoU, person box-mask IoU and "
        "foreground mIoU — are measured and reported above and decide "
        "segmentation installability, but do not enter this classification.",
        "",
        "| q | tier | object requirements | failed | segmentation installable | "
        "segmentation action | 9/9 service ready |",
        "| ---: | --- | ---: | --- | ---: | --- | ---: |",
    ]
    for entry in secondary["profiles"]:
        localization = entry["localization_priority"]
        segmentation = entry["segmentation"]
        lines.append(
            f"| {entry['q']:.2f} | `{entry['tier']}` | "
            f"{localization['passed_count']}/{localization['total']} | "
            f"{', '.join(localization['failed']) or '—'} | "
            f"{segmentation['segmentation_installable']} | "
            f"`{segmentation['action']}` | "
            f"{entry['service_readiness']['service_ready']} |"
        )
    lines += [
        "",
        SEGMENTATION_INSTALL_RULE,
        "",
        SERVICE_READY_RULE,
        "",
        MASKING_POLICY,
        "",
        f"`{TIER_STATE_INFEASIBLE}` is {STATE_INFEASIBLE_DEFINITION}",
        "",
        "## Integrity",
        "",
        f"- family: {label}, family id "
        f"{document['scope']['family_id']}, {width} transported latent channels",
        f"- validation frames per q: {contract.VALIDATION_FRAMES:,}",
        f"- q settings completed exactly once: {len(rows)}/{len(Q_VALUES)}",
        f"- every frame carried the {label} family id, a {width}-channel latent "
        "and the bound routing tag in its own header",
        "- every frame was decompressed exactly once, and the decoder was "
        "discovered from the received header bytes alone",
        "- retained UINT8 cells were exactly the selected cells; dropped cells "
        "scattered to exact zero before reconstruction",
        f"- q=0 invoked the ranker zero times and {label} every time; no q "
        "produced an identity reconstruction",
        f"- frozen perception, stable ranker and selected {label} parameters and "
        "buffers were unchanged",
        "- per q the setting JSON was fsynced into place first, its predictions "
        "were removed only afterwards, and the cleanup marker was written last, "
        "so an interruption could only lose scratch predictions",
        "- only compact evidence is retained; no prediction directory survives",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Run identity, manifest and finalization
# ---------------------------------------------------------------------------


def run_identity(bottleneck: int, binding: Mapping[str, Any]) -> dict[str, Any]:
    """Everything a resumed run must be measuring the same thing under.

    The family, the exact scientific scope, the selected checkpoint and
    decision, the frozen noAE reference, the routing tag, the primary acceptance
    rule, the *whole* secondary registration, every frozen binding and the named
    runner sources. Any change to any of them changes the digest, and a resume
    against a different digest is refused rather than mixed. Because the
    secondary thresholds are inside the identity, they cannot be moved between
    an interrupted run and its resume.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    row = selected(size)
    package = dict(binding["ae_package_source_sha256"])
    missing = [name for name in RUNNER_SOURCES if name not in package]
    if missing:
        raise guards.HybridQConfigError(
            f"the AE package source map does not cover {missing}"
        )
    identity = {
        **family_fields(size),
        "schema": schema(size),
        "command": execute_token(size),
        "latent_channels": latent_width(size),
        "q": [float(q) for q in Q_VALUES],
        "q_e4": [continuous_q.quantize_q(q).q_e4 for q in Q_VALUES],
        "validation_frames": contract.VALIDATION_FRAMES,
        "inference_passes_per_q": 1,
        "selected_epoch": int(row["selected_epoch"]),
        "selected_checkpoint_sha256": row["selected_checkpoint_sha256"],
        "holdout_decision_sha256": row["holdout_decision_sha256"],
        "noae_reference_sha256": NOAE_UINT8_VALIDATION_SHA256,
        "routing_tag": routing_tag(size),
        "acceptance_rule": acceptance_rule(size),
        "acceptance_rule_source": ACCEPTANCE_RULE_SOURCE,
        "preservation_gate_count": GATE_COUNT,
        "absolute_service_gate_count": SERVICE_GATE_COUNT,
        "baseline_absolute_service_pass_count": BASELINE_SERVICE_PASS_COUNT,
        "secondary_classification": {
            "object_requirements": [
                [name, float(target), direction]
                for name, target, direction in LOCALIZATION_OBJECT_REQUIREMENTS
            ],
            "segmentation_install_requirements": [
                [name, float(target), direction]
                for name, target, direction in SEGMENTATION_INSTALL_REQUIREMENTS
            ],
            "threshold_provenance": LOCALIZATION_THRESHOLD_PROVENANCE,
            "tiers": [name for name, _definition in CLASSIFICATION_TIERS]
            + [TIER_STATE_INFEASIBLE],
            "segmentation_install_rule": SEGMENTATION_INSTALL_RULE,
            "service_ready_rule": SERVICE_READY_RULE,
            "masking_policy": MASKING_POLICY,
        },
        "binding": common.binding_fields(binding),
        "runner_sources": {name: package[name] for name in RUNNER_SOURCES},
        "runner_sha256": sha256_file(Path(__file__)),
    }
    return {**identity, "sha256": _identity_digest(identity)}


def manifest_document(bottleneck: int, identity: Mapping[str, Any]) -> dict[str, Any]:
    size = family.require_phase10_bottleneck(bottleneck)
    return {
        "schema": schema(size),
        **family_fields(size),
        "terminal_when_complete": terminal(size),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_identity_sha256": identity["sha256"],
        "run_identity": dict(identity),
        "q_values": [float(q) for q in Q_VALUES],
        "settings_directory": SETTINGS_DIRNAME,
        "cleanup_directory": CLEANUP_DIRNAME,
        "expected_settings": [f"{_q_slug(q)}.json" for q in Q_VALUES],
        "durability_order_per_q": list(DURABILITY_ORDER),
        "resume_rule": (
            "--resume requires a bit-identical run identity, reuses only fully "
            "validated durable settings, never remeasures a valid q, and "
            "refuses rather than overwrites an invalid record"
        ),
    }


def load_run_manifest(
    output: Path, bottleneck: int, identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Require an existing manifest that binds exactly this run identity."""
    size = family.require_phase10_bottleneck(bottleneck)
    path = Path(output) / manifest_filename(size)
    if not path.is_file():
        raise guards.HybridQConfigError(
            f"--resume requires an existing run manifest: {path} does not exist"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != schema(size):
        raise guards.HybridQConfigError("Phase-10B run manifest schema drift")
    require_family_identity(document, size, what=path.name)
    expected = manifest_document(size, identity)
    if document.get("run_identity_sha256") != expected["run_identity_sha256"]:
        raise guards.HybridQConfigError(
            "the run manifest binds a different run identity; refusing to resume "
            "into a run measured under different inputs"
        )
    if dict(document.get("run_identity", {})) != dict(identity):
        raise guards.HybridQConfigError("run manifest identity drift")
    if list(document.get("expected_settings", [])) != expected["expected_settings"]:
        raise guards.HybridQConfigError("run manifest expected-setting drift")
    return document


def finalize(
    *,
    bottleneck: int,
    output: Path,
    rows: list[dict[str, Any]],
    binding: Mapping[str, Any],
    identity: Mapping[str, Any],
    runtime: Mapping[str, Any],
    decision: Mapping[str, Any],
    scorers: Any,
    manifest: Mapping[str, Any],
    resumed: bool,
    reused: Sequence[float],
    executed: Sequence[float],
    default_cpu_threads: int,
    started: float,
) -> dict[str, Any]:
    """Emit the family result only once all six settings and markers exist."""
    size = family.require_phase10_bottleneck(bottleneck)
    label = family.family_label(size)
    run_terminal = terminal(size)

    if len(rows) != len(Q_VALUES):
        raise guards.HybridQConfigError(
            f"finalization requires all {len(Q_VALUES)} settings, {len(rows)} "
            "are present"
        )
    if [row["q_e4"] for row in rows] != [
        continuous_q.quantize_q(q).q_e4 for q in Q_VALUES
    ]:
        raise guards.HybridQConfigError("completed q order drift")
    q0 = rows[0]
    if continuous_q.quantize_q(float(q0["q"])).q_e4 != 0:
        raise guards.HybridQConfigError("the first completed setting is not q=0")

    for row in rows:
        row["payload_ratios"] = _payload_ratios(size, row, q0)

    # 1. The primary result: the Phase-9D rule, applied to the Phase-9D input
    #    fields, with only the family terminal relabelled.
    acceptance = evaluate_acceptance([acceptance_inputs(row) for row in rows], size)

    # 2. The secondary classification, computed *from* the primary result so it
    #    cannot disagree with it, and attached beside each measured row.
    secondary = classify_profiles(bottleneck=size, rows=rows, acceptance=acceptance)
    by_q_e4 = {int(entry["q_e4"]): entry for entry in secondary["profiles"]}
    for row in rows:
        row["secondary_classification"] = by_q_e4[int(row["q_e4"])]

    _require_state_unchanged(runtime)

    settings_sha256 = {
        _q_slug(row["q"]): sha256_file(setting_path(output, size, float(row["q"])))
        for row in rows
    }
    markers_complete = all(
        cleanup_is_complete(
            output, size, float(row["q"]), identity, settings_sha256[_q_slug(row["q"])]
        )
        for row in rows
    )

    document = {
        "schema": schema(size),
        "terminal": run_terminal,
        **family_fields(size),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            f"measure the selected {label} on the real UINT8 + mandatory-zstd "
            "deployment path at the six registered q anchors, and compare each "
            "row with the frozen noAE UINT8+zstd validation result at the same q "
            f"so the reported degradation isolates the {label} latent transport"
        ),
        "scope": {
            **family_fields(size),
            "latent_channels": latent_width(size),
            "ae128_touched": False,
            "validation_frames_per_q": contract.VALIDATION_FRAMES,
            "validation_episodes": list(contract.VALIDATION_EPISODES),
            "q_values": [float(q) for q in Q_VALUES],
            "stress_q_values": [float(q) for q in STRESS_Q_VALUES],
            "completed_settings": len(rows),
            "inference_passes_per_q": 1,
            "training_or_tuning": False,
            "checkpoint_selection_performed_here": False,
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
        "selected_ae": runtime["autoencoder_provenance"],
        "holdout_selection_decision": dict(decision),
        "transport": {
            "pipeline": [
                "original FP32 C2",
                f"selected {label} encoder on the complete frame",
                "per-channel UINT8 latent quantization, ranges from the complete "
                "latent before q dropping",
                "stable per-frame top-K cell selection for q>0",
                "family-labelled sparse AE latent wire",
                "mandatory zstd level 1",
                "received raw bytes",
                "exactly one zstd decompression",
                "decoder selected from the received header family, bottleneck "
                "and routing tag",
                "UINT8 dequantization and zero scatter",
                f"selected {label} decoder",
                "unchanged frozen perception tail and p025 service policy",
            ],
            "family_id": family.family_id(size),
            "transported_latent_channels": latent_width(size),
            "routing_tag": routing_tag(size),
            "range_bytes_per_frame": range_bytes(size),
            "analytical_payload_per_q": {
                _q_slug(q): {
                    "total_bytes": analytical_size(size, q).total_bytes,
                    "header_bytes": analytical_size(size, q).header_bytes,
                    "mask_bytes": analytical_size(size, q).mask_bytes,
                    "range_bytes": analytical_size(size, q).range_bytes,
                    "value_bytes": analytical_size(size, q).value_bytes,
                }
                for q in Q_VALUES
            },
            "q0_bypasses_ranker": True,
            "q0_bypasses_ae": False,
            "ranking_input": "original FP32 C2, independently per frame",
            "ranges": "per frame/channel from the complete AE latent before dropping",
            "zstd": implementation_report(),
            "zstd_mandatory": True,
            "zstd_level_tuned_here": False,
            "codec_logic_duplicated_here": False,
            "separate_ae64_and_ae32_implementations": False,
            "snap_continuous_q_called": False,
        },
        "same_q_reference": {
            "path": NOAE_UINT8_VALIDATION_RELPATH,
            "sha256": NOAE_UINT8_VALIDATION_SHA256,
            "description": SAME_Q_BASELINE_LABEL,
            "identical_reference_as_ae128_phase9d": True,
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
            "selected_family_uint8_zstd_q0": "this run, q=0 row",
        },
        "preregistered_interpretation": acceptance,
        "secondary_prospective_classification": secondary,
        "secondary_classification_reference_feasibility": reference_feasibility(size),
        "verdict_separation": {
            "primary": (
                "the original twelve-gate relative-preservation study and the "
                "family-level acceptance rule, unchanged"
            ),
            "secondary": (
                "an absolute AVO/object localization-priority classification of "
                "each measured profile, plus separate segmentation-installability "
                "and 9/9 service-readiness results"
            ),
            "secondary_changed_primary_terminal": False,
            "secondary_changed_checkpoint_selection": False,
            "secondary_changed_threshold_nms_model_or_scorer": False,
            "secondary_erased_or_reinterpreted_preservation_failures": False,
            "relative_preservation_is_not_service_ready": True,
            "relative_preservation_does_not_authorize_segmentation_install": True,
        },
        "curve": rows,
        "settings": {
            _q_slug(row["q"]): {
                "path": f"{SETTINGS_DIRNAME}/{_q_slug(row['q'])}.json",
                "sha256": settings_sha256[_q_slug(row["q"])],
            }
            for row in rows
        },
        "run_recovery": {
            "manifest": manifest_filename(size),
            "manifest_run_identity_sha256": manifest["run_identity_sha256"],
            "run_identity_sha256": identity["sha256"],
            "resumed": bool(resumed),
            "reused_q": [float(q) for q in reused],
            "executed_q": [float(q) for q in executed],
            "policy": (
                "one atomic durable record per completed q; a resumed run "
                "reuses only records that revalidate under this identity, "
                "re-measures nothing else, and refuses an invalid record rather "
                "than overwriting it"
            ),
        },
        "durability": {
            "order_per_q": list(DURABILITY_ORDER),
            "setting_json_is_the_completion_record": True,
            "predictions_removed_only_after_the_setting_is_durable": True,
            "cleanup_markers": {
                _q_slug(row["q"]): {
                    "path": f"{CLEANUP_DIRNAME}/{_q_slug(row['q'])}.json",
                    "terminal": cleanup_terminal(size),
                    "sha256": sha256_file(
                        cleanup_marker_path(output, size, float(row["q"]))
                    ),
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
            "ae_never_bypassed": all(
                not row["integrity"]["ae_encoder_bypassed"] for row in rows
            ),
            "decoder_always_selected_from_header": all(
                row["integrity"]["decoder_selected_from_received_header_bytes"]
                for row in rows
            ),
            "every_frame_carried_this_family": all(
                list(row["integrity"]["received_family_ids"])
                == [family.family_id(size)]
                for row in rows
            ),
            "every_frame_carried_this_latent_width": all(
                list(row["integrity"]["received_latent_widths"])
                == [latent_width(size)]
                for row in rows
            ),
            "frozen_perception_state_unchanged": True,
            "stable_ranker_state_unchanged": True,
            "selected_ae_state_unchanged": True,
            "every_q_has_a_durable_setting_and_cleanup_marker": markers_complete,
            "measured_q_rows_reported": len(rows),
            "measured_q_rows_registered": len(Q_VALUES),
            "rows_suppressed_by_failed_acceptance": 0,
            "every_measured_q_reported_regardless_of_acceptance": (
                len(rows) == len(Q_VALUES)
            ),
        },
        "wall_seconds_this_invocation": time.time() - started,
        "enriched_curve_note": (
            "each curve row is the durable per-q setting record read back off "
            "disk, plus the payload ratios and the secondary classification "
            "that can only be computed once every q exists; the durable record "
            "itself remains the completion record for its q"
        ),
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
    if not integrity["every_frame_carried_this_family"]:
        raise guards.HybridQPayloadError("a measured q recorded a foreign family")
    if not integrity["every_measured_q_reported_regardless_of_acceptance"]:
        raise guards.HybridQConfigError("a measured q row was suppressed")

    _atomic_json(output / result_json_filename(size), document)
    _atomic_write(output / result_csv_filename(size), _csv_text(size, rows))
    _atomic_write(
        output / report_filename(size), _report_text(size, document)
    )
    _atomic_write(output / run_terminal, f"{run_terminal} {document['generated_utc']}\n")
    return document


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The public command surface: one runner, one family per invocation.

    ``--bottleneck`` admits only the two Phase-10A families, so AE128 is refused
    at parse time and again by ``require_phase10_bottleneck``. There is
    deliberately no frame-limiting, smoke or bounded option: every pass measures
    all 3,345 registered validation frames.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Phase-10B selected AE64/AE32 UINT8 + mandatory-zstd deployment "
            "validation"
        )
    )
    parser.add_argument("--execute", required=True, choices=EXECUTE_TOKENS)
    parser.add_argument(
        "--bottleneck", required=True, type=int, choices=family.AE_PHASE10_BOTTLENECKS
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=DATALOADER_WORKERS)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # First, and before CUDA is touched or any output directory is created: the
    # execute token and --bottleneck must name the same in-scope family.
    bottleneck = require_token_agrees_with_bottleneck(args.execute, args.bottleneck)
    family.bind_process_family(bottleneck)
    label = family.family_label(bottleneck)
    run_terminal = terminal(bottleneck)
    row = selected(bottleneck)

    output: Path = args.output
    if args.resume:
        if not output.is_dir():
            raise guards.HybridQConfigError(
                f"--resume requires an existing run directory: {output} does not "
                "exist"
            )
        if (output / run_terminal).is_file():
            raise guards.HybridQConfigError(
                f"{output / run_terminal} already exists: this validation "
                "completed, and a completed result is not rewritten"
            )
    elif output.exists() and any(output.iterdir()):
        raise guards.HybridQConfigError(
            f"create-only without --resume: {output} already holds files"
        )

    binding = bind_inputs(bottleneck)
    decision = load_holdout_decision(bottleneck, binding)
    references = load_noae_reference()
    identity = run_identity(bottleneck, binding)

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Phase-10B {label} validation requires the qualified CUDA runtime"
        )
    device = torch.device("cuda:0")
    torch.manual_seed(contract.RANKER_INIT_SEED)
    default_cpu_threads = torch.get_num_threads()
    torch.set_num_threads(TORCH_CPU_THREADS)
    started = time.time()

    output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        manifest = load_run_manifest(output, bottleneck, identity)
    else:
        # Written before the first pass, atomically, then read back and checked
        # through exactly the validator a later --resume would use.
        _atomic_json(
            output / manifest_filename(bottleneck),
            manifest_document(bottleneck, identity),
        )
        manifest = load_run_manifest(output, bottleneck, identity)

    (output / SETTINGS_DIRNAME).mkdir(exist_ok=True)
    (output / CLEANUP_DIRNAME).mkdir(exist_ok=True)
    (output / WORKING_DIRNAME).mkdir(exist_ok=True)

    # A fresh run must not inherit completed measurements: an existing durable
    # record without --resume is refused rather than reused or overwritten.
    if not args.resume:
        existing = [
            float(q) for q in Q_VALUES if setting_path(output, bottleneck, q).is_file()
        ]
        if existing:
            raise guards.HybridQConfigError(
                "a fresh run cannot already hold completed setting records for "
                f"q={existing}; pass --resume to continue that run"
            )

    runtime = dict(_load_runtime(bottleneck, device))
    autoencoder, provenance = load_selected_ae(bottleneck, device, binding)
    decoders = ae_family_dispatch.PreloadedAeDecoders([autoencoder])
    if decoders.families != (family.family_id(bottleneck),):
        raise guards.HybridQConfigError(
            f"exactly one preloaded {label} decoder is expected"
        )
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
    reused_q: list[float] = []
    executed_q: list[float] = []

    print(
        json.dumps(
            {
                "family": label,
                "bottleneck": bottleneck,
                "latent_channels": latent_width(bottleneck),
                "selected_epoch": int(row["selected_epoch"]),
                "routing_tag_hex": f"0x{routing_tag(bottleneck):08x}",
                "run_identity_sha256": identity["sha256"],
                "resumed": bool(args.resume),
            }
        ),
        flush=True,
    )

    for q in Q_VALUES:
        path = setting_path(output, bottleneck, q)
        # A q with a valid durable record is never remeasured; at most its
        # interrupted cleanup is finished.
        durable = reuse_or_complete(
            output=output, bottleneck=bottleneck, q=q, identity=identity
        )
        if durable is not None:
            completed_rows.append(durable)
            reused_q.append(float(q))
            print(
                json.dumps(
                    {
                        "family": label,
                        "reused_completed_q": q,
                        "setting": str(path),
                    }
                ),
                flush=True,
            )
            continue

        predictions = prediction_root(output, bottleneck, q)
        if predictions.exists():
            # Scratch output of a pass that never produced a setting record. No
            # record references it, so it is discarded and re-measured in full
            # rather than partially reused.
            print(
                f"[{family.family_slug(bottleneck)}] discarding incomplete "
                f"prediction scratch {predictions.name} from an interrupted pass",
                flush=True,
            )
            shutil.rmtree(predictions)
        raw = run_validation_pass(
            runtime=runtime,
            q=q,
            output=predictions,
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
            bottleneck=bottleneck,
            raw=raw,
            scored=scored,
            reference=reference,
            identity=identity,
        )
        # The scientific completion record goes down first, fsynced into place,
        # and is then re-read through the same validator the resume path uses,
        # so the in-memory row is exactly the durable bytes.
        digest = _atomic_json(path, setting)
        setting = load_durable_setting(path, bottleneck, q, identity)
        # Only now: drop the scratch predictions, then mark the cleanup durable.
        complete_cleanup(
            output=output,
            bottleneck=bottleneck,
            q=q,
            identity=identity,
            setting_sha256=digest,
            predictions=predictions,
        )
        completed_rows.append(setting)
        executed_q.append(float(q))
        preservation = setting[PRESERVATION_KEY]
        print(
            json.dumps(
                {
                    "family": label,
                    "completed_q": q,
                    "frames": setting["frames"],
                    "zstd_bytes_median": setting["payload"]["zstd_bytes"]["median"],
                    "absolute_service_gates": setting["absolute_service_gates"][
                        "pass_count"
                    ],
                    "same_q_preservation_gates": preservation["gates_passed"],
                    "failed_same_q_preservation_gates": preservation["failed"],
                    "setting": str(path),
                    "sha256": digest,
                }
            ),
            flush=True,
        )

    _require_state_unchanged(runtime)
    document = finalize(
        bottleneck=bottleneck,
        output=output,
        rows=completed_rows,
        binding=binding,
        identity=identity,
        runtime=runtime,
        decision=decision,
        scorers=scorers,
        manifest=manifest,
        resumed=bool(args.resume),
        reused=reused_q,
        executed=executed_q,
        default_cpu_threads=default_cpu_threads,
        started=started,
    )
    work_dir = output / WORKING_DIRNAME
    if work_dir.exists() and not any(work_dir.iterdir()):
        work_dir.rmdir()
    acceptance = document["preregistered_interpretation"]
    secondary = document["secondary_prospective_classification"]
    print(
        json.dumps(
            {
                "family": label,
                "terminal": run_terminal,
                "output": str(output),
                "settings": len(document["curve"]),
                "decision": acceptance["decision"],
                "accepted": acceptance["accepted"],
                "qualifying_primary_q": acceptance["qualifying_primary_q"],
                "secondary_tier_counts": secondary["tier_counts"],
                "segmentation_installable_profiles": secondary[
                    "segmentation_installable_profiles"
                ],
                "service_ready_profiles": secondary["service_ready_profiles"],
                "all_finite": document["integrity"]["all_outputs_and_metrics_finite"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(run_terminal)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
