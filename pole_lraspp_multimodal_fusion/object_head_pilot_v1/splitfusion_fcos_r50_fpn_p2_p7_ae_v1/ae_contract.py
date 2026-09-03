"""Registered AE constants and fail-closed validators for the Phase-9A package.

Every frozen quantity — split shape, registered q values, keep/drop formula,
bitmask layout, teacher groups, error classes — is imported from the frozen
hybrid-q contract rather than restated, so this module can only ever add AE
facts and never silently fork a frozen one.
"""

from __future__ import annotations

import torch

from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import contract, guards


# ---------------------------------------------------------------------------
# Families and the frozen split geometry they operate on
# ---------------------------------------------------------------------------

AE_SCHEMA = "splitfusion_fcos_ae_latent_transport_v1"

# Latent channel counts. Spatial resolution is never changed by the AE.
AE_BOTTLENECKS = (128, 64, 32)

AE_INPUT_CHANNELS = contract.SPLIT_CHANNELS  # 256
AE_LATENT_HEIGHT = contract.SPLIT_HEIGHT  # 112
AE_LATENT_WIDTH = contract.SPLIT_WIDTH  # 192
AE_LATENT_SPATIAL_SHAPE = contract.SPLIT_SPATIAL_SHAPE  # (112, 192)
AE_LATENT_CELLS = contract.SPLIT_CELLS  # 21504

# Deterministic initialization. The base seed matches the registered ranker
# seed; each family is separated by its own bottleneck so AE128, AE64 and AE32
# never share a random draw.
AE_INIT_BASE_SEED = 20260829

# Both residual context kernels are depthwise 3x3 with padding 1.
AE_CONTEXT_KERNEL = 3
AE_CONTEXT_PADDING = 1


def ae_init_seed(bottleneck: int) -> int:
    """Per-family initialization seed: base seed separated by bottleneck size."""
    return AE_INIT_BASE_SEED + int(require_bottleneck(bottleneck))


def mask_channel_index(bottleneck: int) -> int:
    """Decoder input channel that carries the reconstructed keep mask.

    The mask is appended after the B latent channels, so it is always the last
    decoder input channel and the first B channels stay in latent order.
    """
    return int(require_bottleneck(bottleneck))


# ---------------------------------------------------------------------------
# Per-frame model-family identity (provenance only; no registry, no loading)
# ---------------------------------------------------------------------------
#
# The deployed runtime holds one shared frozen front/tail, the noAE direct path
# and preloaded AE128/AE64/AE32 encoder-decoder pairs, and switches profile at
# runtime. A delayed or reordered packet must therefore never be decoded with
# whichever AE happens to be selected *now*, so family identity travels in the
# per-frame envelope rather than in mutable global state.
#
# The validated noAE UINT8 wire is frozen and is not re-framed here: its own
# envelope already identifies it unambiguously (magic HQ8\0, codec id 1, 256
# transported channels), which maps to family id 0. The new AE wire carries an
# explicit family id field plus a 32-bit routing tag.

AE_FAMILY_NOAE = 0
AE_FAMILY_AE128 = 1
AE_FAMILY_AE64 = 2
AE_FAMILY_AE32 = 3

AE_FAMILY_IDS = {
    AE_FAMILY_NOAE: "noAE",
    AE_FAMILY_AE128: "AE128",
    AE_FAMILY_AE64: "AE64",
    AE_FAMILY_AE32: "AE32",
}

# Registered bijection between AE family id and transported latent channels.
AE_FAMILY_BOTTLENECKS = {
    AE_FAMILY_AE128: 128,
    AE_FAMILY_AE64: 64,
    AE_FAMILY_AE32: 32,
}
AE_BOTTLENECK_FAMILIES = {
    bottleneck: family for family, bottleneck in AE_FAMILY_BOTTLENECKS.items()
}

# The routing tag is a 32-bit *discriminator* that lets the edge route a frame to
# the preloaded decoder that produced it. It is deliberately not a cryptographic
# checkpoint identity: 32 bits cannot authenticate a checkpoint, and the
# authoritative full SHA-256 stays bound by the existing profile registry. Its
# job is to stop a packet delayed across a profile switch from silently reaching
# a different AE, not to prove provenance against an adversary.
#
# 0 means "unbound". A freshly constructed AE is unbound because Phase 9A loads
# no checkpoint, and the deployable encode and dispatch paths both refuse it.
AE_UNBOUND_ROUTING_TAG = 0
AE_ROUTING_TAG_BYTES = 4


def family_for_bottleneck(bottleneck: int) -> int:
    """Registered AE family id for one latent channel count."""
    return AE_BOTTLENECK_FAMILIES[require_bottleneck(bottleneck)]


def bottleneck_for_family(family_id: int) -> int:
    """Registered latent channel count for one AE family id.

    noAE and unknown ids are rejected: they have no AE latent and must not be
    decoded by an AE decoder.
    """
    if isinstance(family_id, bool) or not isinstance(family_id, int):
        raise guards.HybridQConfigError(
            f"family id must be an int, got {type(family_id).__name__}"
        )
    if family_id not in AE_FAMILY_BOTTLENECKS:
        name = AE_FAMILY_IDS.get(int(family_id), "unregistered")
        raise guards.HybridQPayloadError(
            f"family id {family_id} ({name}) is not a registered AE family "
            f"{sorted(AE_FAMILY_BOTTLENECKS)}"
        )
    return AE_FAMILY_BOTTLENECKS[int(family_id)]


def family_name(family_id: int) -> str:
    return AE_FAMILY_IDS.get(int(family_id), f"unregistered({int(family_id)})")


def require_routing_tag(tag: int) -> int:
    """Field validator: one unsigned 32-bit routing tag, unbound (0) allowed."""
    if isinstance(tag, bool) or not isinstance(tag, int):
        raise guards.HybridQConfigError(
            f"routing tag must be an int, got {type(tag).__name__}"
        )
    if not 0 <= tag < 2 ** (8 * AE_ROUTING_TAG_BYTES):
        raise guards.HybridQConfigError(
            f"routing tag {tag} does not fit in {AE_ROUTING_TAG_BYTES} unsigned bytes"
        )
    return int(tag)


def require_bound_routing_tag(tag: int, *, what: str = "routing tag") -> int:
    """Deployable-path validator: an unbound (0) tag is refused.

    Encoding or dispatching with an unbound tag would give every family the same
    tag, which defeats the whole point of routing a delayed packet back to the
    AE that produced it.
    """
    value = require_routing_tag(tag)
    if value == AE_UNBOUND_ROUTING_TAG:
        raise guards.HybridQConfigError(
            f"{what} is unbound (0); the deployable path requires a bound "
            "routing tag from the profile registry"
        )
    return value


def routing_tag_from_sha256(digest: str) -> int:
    """Derive the routing tag from a registered checkpoint SHA-256 hex digest.

    Truncation to 32 bits is intentional and lossy: the result is a routing
    discriminator, not the checkpoint identity. The full digest stays with the
    profile registry, which remains the authority on which checkpoint a
    preloaded pair actually holds. Pure string arithmetic on a digest the caller
    already holds; this reads no file and loads no checkpoint.
    """
    if not isinstance(digest, str) or len(digest) != 64:
        raise guards.HybridQConfigError(
            "a routing tag requires a 64-character SHA-256 hex digest"
        )
    try:
        leading = int(digest[: 2 * AE_ROUTING_TAG_BYTES], 16)
    except ValueError as error:
        raise guards.HybridQConfigError("routing tag digest is not hexadecimal") from error
    return require_bound_routing_tag(leading, what="routing tag derived from digest")


# ---------------------------------------------------------------------------
# Intended later training protocol (documentation constants; nothing runs here)
# ---------------------------------------------------------------------------

AE_STAGE_A_Q = 0.00
AE_STAGE_B_Q_CYCLE = (0.00, 0.30, 0.50, 0.70)
AE_OPTIMIZATION_EXCLUDED_Q = contract.EVALUATION_STRESS_Q_VALUES  # 0.90, 0.98
AE_TASK_GROUPS = contract.TEACHER_GROUPS  # D, G, S, A
AE_MIN_VALID_TASK_GROUPS = 3
AE_TRAINABLE_SCOPE = "autoencoder_parameters_only"


# ---------------------------------------------------------------------------
# Fail-closed validators (frozen guard error classes, AE-specific subjects)
# ---------------------------------------------------------------------------


def require_bottleneck(bottleneck: int) -> int:
    """Only the three registered latent channel counts are constructible."""
    if isinstance(bottleneck, bool) or not isinstance(bottleneck, int):
        raise guards.HybridQConfigError(
            f"bottleneck must be an int, got {type(bottleneck).__name__}"
        )
    if bottleneck not in AE_BOTTLENECKS:
        raise guards.HybridQConfigError(
            f"bottleneck {bottleneck} is not a registered AE family {AE_BOTTLENECKS}"
        )
    return int(bottleneck)


def require_latent(
    latent: torch.Tensor,
    bottleneck: int,
    *,
    what: str = "AE latent",
    check_finite: bool = True,
) -> torch.Tensor:
    """One frame of latent: exactly [B, 112, 192] FP32, finite."""
    size = require_bottleneck(bottleneck)
    if not isinstance(latent, torch.Tensor):
        raise guards.HybridQPayloadError(f"{what} must be a torch.Tensor")
    expected = (size, AE_LATENT_HEIGHT, AE_LATENT_WIDTH)
    if tuple(latent.shape) != expected:
        raise guards.HybridQPayloadError(
            f"{what} must be {list(expected)}, got {list(latent.shape)}"
        )
    if latent.dtype is not torch.float32:
        raise guards.HybridQPayloadError(f"{what} must be float32, got {latent.dtype}")
    if check_finite:
        guards.require_finite(latent, what)
    return latent


def require_keep_mask(
    mask: torch.Tensor, *, what: str = "keep mask", expect_keep: int | None = None
) -> torch.Tensor:
    """One frame of binary keep mask: exactly [112, 192] bool."""
    if not isinstance(mask, torch.Tensor):
        raise guards.HybridQPayloadError(f"{what} must be a torch.Tensor")
    if tuple(mask.shape) != AE_LATENT_SPATIAL_SHAPE:
        raise guards.HybridQPayloadError(
            f"{what} must be {list(AE_LATENT_SPATIAL_SHAPE)}, got {list(mask.shape)}"
        )
    if mask.dtype is not torch.bool:
        raise guards.HybridQPayloadError(f"{what} must be boolean, got {mask.dtype}")
    if expect_keep is not None:
        guards.require_keep_cardinality(int(mask.sum()), int(expect_keep))
    return mask
