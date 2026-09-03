"""Shared family-aware Phase-10A configuration for AE64 and AE32.

One implementation, two families. Every family-dependent quantity -- execute
token, terminal, schema, checkpoint filename, selection directory, init seed and
architecture -- is derived from the single `bottleneck` argument, so AE64 and
AE32 cannot drift apart and neither can borrow the other's artifact.

The completed AE128 Phase-9C work is imported and reused, never edited: the
locked scientific configuration, the exact Stage-A/Stage-B schedule, the
sample-id-keyed Phase-4 teacher store, the atomic write primitives, the frozen
noAE same-q holdout reference and the registered preservation gates all come
from `ae_training_common`. This module adds only what a *different family* needs
and re-labels nothing else.

AE128 is deliberately not constructible here. `require_phase10_bottleneck`
admits 64 and 32 only, and every emitted schema, terminal, token and filename is
checked to carry its own family label and no other family's.

Nothing here loads a checkpoint, touches CUDA, reads a dataset or cache, trains,
infers or evaluates at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    contract,
    continuous_q,
    guards,
    training,
)
from . import ae_contract, ae_loss
from . import ae_training_common as common
from .ae_model import SplitFeatureAE, ae_parameters, build_split_feature_ae


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

PHASE = "phase10a"

# Phase 10A trains the two smaller registered families. AE128 is complete and is
# out of scope: it keeps its own runners, schemas and artifacts untouched.
AE_PHASE10_BOTTLENECKS = (64, 32)
AE_OUT_OF_SCOPE_BOTTLENECK = common.AE_TRAINING_BOTTLENECK  # 128

# The AE has exactly eight trainable tensors in every family: two weights and
# two biases on each side of the bottleneck.
AE_TRAINABLE_TENSOR_COUNT = 8


def require_phase10_bottleneck(bottleneck: int) -> int:
    """Only 64 and 32 are trainable in this phase; 128 is refused by name."""
    value = ae_contract.require_bottleneck(bottleneck)
    if value not in AE_PHASE10_BOTTLENECKS:
        raise guards.HybridQConfigError(
            f"bottleneck {value} is not a Phase-10A family "
            f"{AE_PHASE10_BOTTLENECKS}; the {value}-channel family is out of "
            "scope for this runner"
        )
    return value


def family_id(bottleneck: int) -> int:
    return ae_contract.family_for_bottleneck(require_phase10_bottleneck(bottleneck))


def family_label(bottleneck: int) -> str:
    """The registered family name, e.g. 'AE64'."""
    size = require_phase10_bottleneck(bottleneck)
    label = ae_contract.family_name(ae_contract.family_for_bottleneck(size))
    if label != f"AE{size}":
        raise guards.HybridQConfigError(
            f"registered family name {label!r} disagrees with bottleneck {size}"
        )
    return label


def family_slug(bottleneck: int) -> str:
    """Lower-case family label used inside filenames and schemas, e.g. 'ae64'."""
    return family_label(bottleneck).lower()


def require_family_labelled(text: str, bottleneck: int, *, what: str) -> str:
    """Fail closed unless `text` names this family and no other AE family.

    Applied to every emitted schema, terminal, token and filename, so an AE128
    (or cross-family) label cannot reach a Phase-10A artifact even by a typo.
    """
    size = require_phase10_bottleneck(bottleneck)
    if not isinstance(text, str) or not text:
        raise guards.HybridQConfigError(f"{what} must be a non-empty string")
    lowered = text.lower()
    if family_slug(size) not in lowered:
        raise guards.HybridQConfigError(
            f"{what} {text!r} does not identify {family_label(size)}"
        )
    for other in ae_contract.AE_BOTTLENECKS:
        if int(other) == size:
            continue
        if f"ae{int(other)}" in lowered:
            raise guards.HybridQConfigError(
                f"{what} {text!r} carries the AE{int(other)} label in an "
                f"{family_label(size)} run"
            )
    return text


# ---------------------------------------------------------------------------
# Execute tokens and terminals
# ---------------------------------------------------------------------------


def training_token(bottleneck: int) -> str:
    size = require_phase10_bottleneck(bottleneck)
    return require_family_labelled(
        f"SPLITFUSION_AE{size}_PHASE10_TRAINING", size, what="training execute token"
    )


def holdout_token(bottleneck: int) -> str:
    size = require_phase10_bottleneck(bottleneck)
    return require_family_labelled(
        f"SPLITFUSION_AE{size}_PHASE10_HOLDOUT_SELECTION",
        size,
        what="holdout execute token",
    )


def training_terminal(bottleneck: int) -> str:
    size = require_phase10_bottleneck(bottleneck)
    return require_family_labelled(
        f"SPLITFUSION_AE{size}_PHASE10_TRAINING_COMPLETE",
        size,
        what="training terminal",
    )


def holdout_terminal(bottleneck: int) -> str:
    size = require_phase10_bottleneck(bottleneck)
    return require_family_labelled(
        f"SPLITFUSION_AE{size}_PHASE10_HOLDOUT_CHECKPOINT_SELECTED",
        size,
        what="holdout terminal",
    )


TRAINING_EXECUTE_TOKENS = tuple(
    training_token(size) for size in AE_PHASE10_BOTTLENECKS
)
HOLDOUT_EXECUTE_TOKENS = tuple(holdout_token(size) for size in AE_PHASE10_BOTTLENECKS)


def require_token_agrees_with_bottleneck(
    token: str, bottleneck: int, *, kind: str
) -> int:
    """The execute token and `--bottleneck` must name the same family, exactly.

    `kind` is "training" or "holdout"; a training token passed to the selection
    command is refused as firmly as a family mismatch, so neither the family nor
    the command can be selected by accident.
    """
    if kind == "training":
        expected = {training_token(size): size for size in AE_PHASE10_BOTTLENECKS}
    elif kind == "holdout":
        expected = {holdout_token(size): size for size in AE_PHASE10_BOTTLENECKS}
    else:
        raise guards.HybridQConfigError(f"{kind!r} is not a Phase-10A command kind")
    if token not in expected:
        raise guards.HybridQConfigError(
            f"{token!r} is not a registered Phase-10A {kind} execute token "
            f"{sorted(expected)}"
        )
    size = require_phase10_bottleneck(bottleneck)
    if expected[token] != size:
        raise guards.HybridQConfigError(
            f"execute token {token} names {family_label(expected[token])} but "
            f"--bottleneck {size} names {family_label(size)}; they must agree"
        )
    return size


# ---------------------------------------------------------------------------
# Schemas and filenames
# ---------------------------------------------------------------------------


def _schema(bottleneck: int, kind: str) -> str:
    size = require_phase10_bottleneck(bottleneck)
    return require_family_labelled(
        f"splitfusion_fcos_ae{size}_phase10a_{kind}_v1", size, what=f"{kind} schema"
    )


def training_schema(bottleneck: int) -> str:
    return _schema(bottleneck, "training")


def recovery_schema(bottleneck: int) -> str:
    return _schema(bottleneck, "recovery")


def candidate_schema(bottleneck: int) -> str:
    return _schema(bottleneck, "candidate")


def holdout_schema(bottleneck: int) -> str:
    return _schema(bottleneck, "holdout_selection")


def candidate_filename(bottleneck: int, epoch: int) -> str:
    size = require_phase10_bottleneck(bottleneck)
    name = f"ae{size}_epoch_{common.require_training_epoch(epoch):02d}.pt"
    return require_family_labelled(name, size, what="candidate filename")


def recovery_filename(bottleneck: int, epoch: int) -> str:
    size = require_phase10_bottleneck(bottleneck)
    name = f"ae{size}_recovery_epoch_{common.require_training_epoch(epoch):02d}.pt"
    return require_family_labelled(name, size, what="recovery filename")


def recovery_glob(bottleneck: int) -> str:
    return f"ae{require_phase10_bottleneck(bottleneck)}_recovery_epoch_*.pt"


def epoch_from_recovery_filename(bottleneck: int, name: str) -> int:
    """Parse a recovery filename strictly; a foreign family is not parseable."""
    size = require_phase10_bottleneck(bottleneck)
    prefix = f"ae{size}_recovery_epoch_"
    if not name.startswith(prefix) or not name.endswith(".pt"):
        raise guards.HybridQConfigError(
            f"{name} is not an {family_label(size)} recovery checkpoint filename"
        )
    digits = name[len(prefix) : -len(".pt")]
    if not digits.isdigit():
        raise guards.HybridQConfigError(f"{name} carries no epoch number")
    epoch = common.require_training_epoch(int(digits))
    if name != recovery_filename(size, epoch):
        raise guards.HybridQConfigError(f"{name} recovery filename drift")
    return epoch


def holdout_selection_dirname(bottleneck: int) -> str:
    size = require_phase10_bottleneck(bottleneck)
    return require_family_labelled(
        f"holdout_selection_ae{size}", size, what="holdout selection directory"
    )


def holdout_selection_dir(training_dir: Path, bottleneck: int) -> Path:
    return Path(training_dir) / holdout_selection_dirname(bottleneck)


def training_report_filename(bottleneck: int) -> str:
    size = require_phase10_bottleneck(bottleneck)
    return require_family_labelled(
        f"ae{size}_training_report.json", size, what="training report filename"
    )


def epoch_summaries_filename(bottleneck: int) -> str:
    size = require_phase10_bottleneck(bottleneck)
    return require_family_labelled(
        f"ae{size}_epoch_summaries.json", size, what="epoch summary filename"
    )


def holdout_report_filename(bottleneck: int) -> str:
    size = require_phase10_bottleneck(bottleneck)
    return require_family_labelled(
        f"ae{size}_holdout_selection.json", size, what="holdout report filename"
    )


def holdout_manifest_schema(bottleneck: int) -> str:
    return _schema(bottleneck, "holdout_run_manifest")


def holdout_setting_schema(bottleneck: int) -> str:
    return _schema(bottleneck, "holdout_setting")


def holdout_manifest_filename(bottleneck: int) -> str:
    size = require_phase10_bottleneck(bottleneck)
    return require_family_labelled(
        f"ae{size}_holdout_run_manifest.json", size, what="holdout manifest filename"
    )


def setting_record_filename(bottleneck: int, epoch: int, q_e4: int) -> str:
    """One completed checkpoint/q pass, named by family, epoch and q."""
    size = require_phase10_bottleneck(bottleneck)
    if int(epoch) not in common.AE_CANDIDATE_EPOCHS:
        raise guards.HybridQConfigError(
            f"epoch {epoch} is not a registered candidate epoch "
            f"{common.AE_CANDIDATE_EPOCHS}"
        )
    value = int(q_e4)
    if value not in {
        continuous_q.quantize_q(float(q)).q_e4 for q in common.AE_HOLDOUT_Q_VALUES
    }:
        raise guards.HybridQConfigError(
            f"q_e4={value} is not a registered Phase-10A selection q"
        )
    name = f"ae{size}_epoch{int(epoch):02d}_q{value:04d}.json"
    return require_family_labelled(name, size, what="setting record filename")


# ---------------------------------------------------------------------------
# One family per process
# ---------------------------------------------------------------------------

_BOUND_FAMILY: int | None = None


def bind_process_family(bottleneck: int) -> int:
    """AE64 and AE32 must not be trained or selected in the same process.

    The two families own different optimizers, different RNG streams and
    different checkpoints; sharing a process would let one run's global state
    reach the other. The first bind wins and any later disagreement fails closed.
    """
    global _BOUND_FAMILY
    size = require_phase10_bottleneck(bottleneck)
    if _BOUND_FAMILY is not None and _BOUND_FAMILY != size:
        raise guards.HybridQOwnershipError(
            f"this process is already bound to {family_label(_BOUND_FAMILY)}; "
            f"{family_label(size)} must run in its own process"
        )
    _BOUND_FAMILY = size
    return size


def bound_family() -> int | None:
    return _BOUND_FAMILY


def _reset_process_family_for_tests() -> None:
    """Test-only hook: forget the bound family. Never called by a runner."""
    global _BOUND_FAMILY
    _BOUND_FAMILY = None


# ---------------------------------------------------------------------------
# The locked scientific configuration, inherited from AE128 Phase 9C
# ---------------------------------------------------------------------------

# Exactly the keys Phase 10A is permitted to change: the family identity and the
# deterministic initialization that follows from it. Everything else -- epochs,
# stages, learning rates, the Stage-B cycle, optimizer, weight decay, clipping,
# batch size, drop_last, augmentation, objective, candidate epochs, split use --
# is inherited from the completed AE128 configuration unchanged.
FAMILY_DEPENDENT_KEYS = ("family", "bottleneck", "init_seed")
PHASE10_ADDED_KEYS = ("phase", "configuration_inherited_from")


def training_configuration(bottleneck: int) -> dict[str, Any]:
    """The AE128 Phase-9C configuration with only the family fields re-derived.

    Built by copying `ae_training_common.training_configuration()` rather than
    restating it, so a scientific setting cannot silently differ between the
    completed AE128 run and these two families.
    """
    size = require_phase10_bottleneck(bottleneck)
    configuration = dict(common.training_configuration())
    configuration["family"] = family_label(size)
    configuration["bottleneck"] = size
    configuration["init_seed"] = ae_contract.ae_init_seed(size)
    configuration["phase"] = PHASE
    configuration["configuration_inherited_from"] = (
        "ae128_phase9c_locked_configuration, family fields re-derived"
    )
    delta = configuration_delta(size)
    if set(delta["changed"]) != set(FAMILY_DEPENDENT_KEYS):
        raise guards.HybridQConfigError(
            f"Phase-10A changed {sorted(delta['changed'])} of the locked "
            f"configuration; only {list(FAMILY_DEPENDENT_KEYS)} may differ"
        )
    return configuration


def configuration_delta(bottleneck: int) -> dict[str, Any]:
    """Exactly which locked-configuration keys this family changes, and how."""
    size = require_phase10_bottleneck(bottleneck)
    inherited = dict(common.training_configuration())
    mine = dict(inherited)
    mine["family"] = family_label(size)
    mine["bottleneck"] = size
    mine["init_seed"] = ae_contract.ae_init_seed(size)
    changed = {
        key: {"inherited": inherited[key], "phase10a": mine[key]}
        for key in inherited
        if inherited[key] != mine[key]
    }
    return {
        "changed": changed,
        "added": list(PHASE10_ADDED_KEYS),
        "unchanged_keys": len(inherited) - len(changed),
        "policy": (
            "only the family identity and the deterministic initialization that "
            "follows from it may differ from the completed AE128 configuration"
        ),
    }


def process_seed(bottleneck: int) -> int:
    """The per-family deterministic seed, `ae_contract.ae_init_seed(B)`.

    The AE itself is initialized from this seed inside a forked RNG, so the
    process seed only fixes any other draw a run might make. The sampler order is
    seeded per epoch by `contract.epoch_shuffle_seed`, which is family
    independent by design: AE64 and AE32 see the identical frame order.
    """
    return ae_contract.ae_init_seed(require_phase10_bottleneck(bottleneck))


# ---------------------------------------------------------------------------
# Family AE and its optimizer
# ---------------------------------------------------------------------------


def build_family_ae(bottleneck: int, device: torch.device) -> SplitFeatureAE:
    """The committed deterministic AE for one Phase-10A family, on `device`."""
    size = require_phase10_bottleneck(bottleneck)
    autoencoder = build_split_feature_ae(size).to(device)
    if autoencoder.bottleneck != size:
        raise guards.HybridQConfigError("AE bottleneck drift")
    if autoencoder.init_seed != ae_contract.ae_init_seed(size):
        raise guards.HybridQConfigError("AE initialization seed drift")
    if autoencoder.family_id != family_id(size):
        raise guards.HybridQConfigError("AE family id drift")
    if autoencoder.family_name != family_label(size):
        raise guards.HybridQConfigError("AE family name drift")
    return autoencoder


def build_family_optimizer(
    autoencoder: SplitFeatureAE,
    *,
    lr: float,
    frozen_modules: Sequence[torch.nn.Module],
) -> torch.optim.Optimizer:
    """Locked AdamW owning exactly this family's eight tensors and nothing else."""
    size = require_phase10_bottleneck(autoencoder.bottleneck)
    optimizer = training.build_ranker_optimizer(
        autoencoder,
        lr=float(lr),
        weight_decay=common.AE_WEIGHT_DECAY,
        frozen_modules=tuple(frozen_modules),
    )
    ae_loss.require_ae_only_optimizer(optimizer, autoencoder)
    owned = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    expected = ae_parameters(autoencoder)
    if len(expected) != AE_TRAINABLE_TENSOR_COUNT:
        raise guards.HybridQConfigError(
            f"{family_label(size)} exposes {len(expected)} trainable tensors, "
            f"the registered AE has {AE_TRAINABLE_TENSOR_COUNT}"
        )
    if len(owned) != len(expected):
        raise guards.HybridQOwnershipError(
            f"optimizer owns {len(owned)} tensors, {family_label(size)} has "
            f"{len(expected)}"
        )
    if {id(parameter) for parameter in owned} != {
        id(parameter) for parameter in expected
    }:
        raise guards.HybridQOwnershipError(
            f"the optimizer does not own exactly the {family_label(size)} tensors"
        )
    return optimizer


# ---------------------------------------------------------------------------
# Family-labelled atomic checkpoints
# ---------------------------------------------------------------------------

# The proven Phase-9C durability primitive, reused rather than reimplemented:
# write beside the destination, fsync, rename into place, fsync the directory.
_atomic_torch_save = common._atomic_torch_save


def family_fields(bottleneck: int) -> dict[str, Any]:
    """The family identity every Phase-10A artifact carries, in one place."""
    size = require_phase10_bottleneck(bottleneck)
    return {
        "phase": PHASE,
        "family": family_label(size),
        "family_id": family_id(size),
        "bottleneck": size,
        "init_seed": ae_contract.ae_init_seed(size),
    }


def require_family_fields(
    payload: Mapping[str, Any], bottleneck: int, *, what: str
) -> None:
    """Fail closed unless a loaded artifact declares exactly this family."""
    expected = family_fields(bottleneck)
    for name, value in expected.items():
        if name not in payload:
            raise guards.HybridQConfigError(f"{what} does not carry {name}")
        if payload[name] != value:
            raise guards.HybridQConfigError(
                f"{what} {name} is {payload[name]!r}, this run is {value!r}"
            )


def save_recovery(
    path: Path,
    *,
    bottleneck: int,
    epoch: int,
    autoencoder: SplitFeatureAE,
    optimizer: torch.optim.Optimizer,
    global_update_index: int,
    stage_b_position: int,
    order_identity: Mapping[str, Any],
    summary: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> str:
    """Everything an exact resume needs for one family, written atomically."""
    size = require_phase10_bottleneck(bottleneck)
    common.require_training_epoch(epoch)
    if autoencoder.bottleneck != size:
        raise guards.HybridQConfigError("recovery AE family drift")
    if Path(path).name != recovery_filename(size, epoch):
        raise guards.HybridQConfigError(
            f"{Path(path).name} is not the {family_label(size)} epoch-{epoch} "
            "recovery filename"
        )
    payload = {
        "schema": recovery_schema(size),
        **family_fields(size),
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "next_epoch_shuffle_seed": (
            contract.epoch_shuffle_seed(epoch + 1)
            if epoch < common.AE_TRAINING_EPOCHS
            else None
        ),
        "stage": common.stage_for_epoch(epoch),
        "next_stage": (
            common.stage_for_epoch(epoch + 1)
            if epoch < common.AE_TRAINING_EPOCHS
            else None
        ),
        "global_update_index": int(global_update_index),
        "stage_b_cycle_position": int(stage_b_position),
        "next_stage_b_q": common.stage_b_q_at(int(stage_b_position)),
        "autoencoder": autoencoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": common.rng_state(),
        "order_identity": dict(order_identity),
        "epoch_summary": dict(summary),
        "configuration": training_configuration(size),
        **common.binding_fields(binding),
    }
    return _atomic_torch_save(payload, path)


def save_candidate(
    path: Path,
    *,
    bottleneck: int,
    epoch: int,
    autoencoder: SplitFeatureAE,
    global_update_index: int,
    stage_b_position: int,
    binding: Mapping[str, Any],
) -> str:
    """One family selection candidate: AE weights plus the identity to bind them."""
    size = require_phase10_bottleneck(bottleneck)
    if int(epoch) not in common.AE_CANDIDATE_EPOCHS:
        raise guards.HybridQConfigError(
            f"epoch {epoch} is not a registered candidate epoch "
            f"{common.AE_CANDIDATE_EPOCHS}"
        )
    if autoencoder.bottleneck != size:
        raise guards.HybridQConfigError("candidate AE family drift")
    if Path(path).name != candidate_filename(size, epoch):
        raise guards.HybridQConfigError(
            f"{Path(path).name} is not the {family_label(size)} epoch-{epoch} "
            "candidate filename"
        )
    payload = {
        "schema": candidate_schema(size),
        **family_fields(size),
        "epoch": int(epoch),
        "stage": common.stage_for_epoch(epoch),
        "autoencoder": autoencoder.state_dict(),
        "parameter_count": autoencoder.parameter_count(),
        "global_update_index": int(global_update_index),
        "stage_b_cycle_position": int(stage_b_position),
        "configuration": training_configuration(size),
        **common.binding_fields(binding),
    }
    return _atomic_torch_save(payload, path)


def load_candidate(
    path: Path,
    bottleneck: int,
    epoch: int,
    device: torch.device,
    binding: Mapping[str, Any],
) -> tuple[SplitFeatureAE, dict[str, Any]]:
    """Load one family candidate for evaluation, frozen and in eval mode."""
    size = require_phase10_bottleneck(bottleneck)
    path = Path(path)
    if path.name != candidate_filename(size, epoch):
        raise guards.HybridQConfigError(
            f"{path.name} is not the {family_label(size)} epoch-{epoch} candidate"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema"] != candidate_schema(size):
        raise guards.HybridQConfigError(f"{path.name} candidate schema drift")
    require_family_fields(payload, size, what=path.name)
    if int(payload["epoch"]) != int(epoch):
        raise guards.HybridQConfigError(f"{path.name} epoch drift")
    if payload["configuration"] != training_configuration(size):
        raise guards.HybridQConfigError(
            f"{path.name} was written under a different locked configuration"
        )
    common.require_bindings(payload, binding, what=path.name)
    autoencoder = build_split_feature_ae(size)
    autoencoder.load_state_dict(payload["autoencoder"])
    if autoencoder.parameter_count() != int(payload["parameter_count"]):
        raise guards.HybridQConfigError(f"{path.name} parameter-count drift")
    autoencoder = autoencoder.to(device)
    common.freeze(autoencoder)
    guards.require_module_parameters_finite(autoencoder, f"candidate {path.name}")
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in ("autoencoder", "configuration")
    }
    del payload
    return autoencoder, metadata


def write_epoch_summaries(
    output: Path, bottleneck: int, summaries: Sequence[Mapping[str, Any]]
) -> str:
    """Atomically replace this family's external epoch-summary file."""
    size = require_phase10_bottleneck(bottleneck)
    return common.atomic_write_json(
        {
            "schema": training_schema(size),
            **family_fields(size),
            "epochs": [dict(summary) for summary in summaries],
        },
        Path(output) / epoch_summaries_filename(size),
    )


__all__ = [
    "AE_PHASE10_BOTTLENECKS",
    "bind_process_family",
    "build_family_ae",
    "build_family_optimizer",
    "candidate_filename",
    "candidate_schema",
    "configuration_delta",
    "epoch_summaries_filename",
    "family_fields",
    "family_label",
    "holdout_manifest_filename",
    "holdout_manifest_schema",
    "holdout_report_filename",
    "holdout_schema",
    "holdout_selection_dir",
    "holdout_setting_schema",
    "holdout_terminal",
    "holdout_token",
    "load_candidate",
    "process_seed",
    "recovery_filename",
    "recovery_schema",
    "require_family_fields",
    "require_phase10_bottleneck",
    "require_token_agrees_with_bottleneck",
    "save_candidate",
    "save_recovery",
    "setting_record_filename",
    "training_configuration",
    "training_report_filename",
    "training_schema",
    "training_terminal",
    "training_token",
    "write_epoch_summaries",
]
