"""One shared, channel-parameterized feature autoencoder at the frozen C2 split.

`SplitFeatureAE(B)` is the *same* architecture for every registered family; only
the latent channel count B in {128, 64, 32} changes. Spatial resolution is
exactly 112x192 on both sides of the bottleneck, so the AE compresses channels
only and never resamples.

Design: lightweight and asymmetric, with no BatchNorm anywhere.

    encoder:  z        = P(C2) + depthwise_3x3(GELU(P(C2)))          P: 1x1 256->B
    decoder:  C2_hat   = E(z|m) + depthwise_3x3(GELU(E(z|m)))        E: 1x1 B+1->256

`m` is the *reconstructed* binary keep mask, concatenated as one extra decoder
input channel. It is not a new side channel: the sparse transport already
carries that mask in its header bitmask, so the decoder is only being told what
it can already read off the wire. At q=0 the mask is all ones.

No skip connection carrying original C2 crosses the transport boundary. The
decoder entry point accepts a latent and a mask and nothing else, so an
original-C2 shortcut is unrepresentable rather than merely discouraged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import guards
from . import ae_contract


@dataclass(frozen=True)
class AeComplexity:
    """Static parameter and MAC accounting for one family; nothing is measured."""

    bottleneck: int
    height: int
    width: int
    encoder_parameters: int
    decoder_parameters: int
    total_parameters: int
    encoder_macs: int
    decoder_macs: int
    total_macs: int


class SplitFeatureAE(nn.Module):
    """Channel autoencoder for the frozen [256,112,192] C2 tensor.

    Construction is deterministic *and* RNG-neutral: every layer is built and
    initialized inside `torch.random.fork_rng`, seeded by
    `ae_contract.ae_init_seed(B)`, so two constructions of the same family are
    bit-identical, different families never share a draw, and the caller's
    global RNG stream is left exactly where it was.

    Initial state is a plain low-rank channel projection: the encoder projection
    has orthonormal rows, the decoder latent weights are its transpose, the mask
    weights, both depthwise kernels and every bias are zero. So at
    initialization `decode(encode(x))` is the rank-B orthogonal projection of
    each cell's channel vector, and the two residual context branches contribute
    exactly nothing until they are trained.
    """

    def __init__(self, bottleneck: int) -> None:
        super().__init__()
        self.bottleneck = ae_contract.require_bottleneck(bottleneck)
        self.in_channels = ae_contract.AE_INPUT_CHANNELS
        self.mask_channel_index = ae_contract.mask_channel_index(self.bottleneck)
        self.init_seed = ae_contract.ae_init_seed(self.bottleneck)
        self.family_id = ae_contract.family_for_bottleneck(self.bottleneck)
        self.family_name = ae_contract.family_name(self.family_id)
        # Provenance only: Phase 9A loads no checkpoint, so this stays unbound.
        self.checkpoint_binding = ae_contract.AE_UNBOUND_CHECKPOINT_BINDING

        # Layer construction itself draws from the RNG (default conv init), so
        # the fork covers construction as well as the explicit initialization.
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(self.init_seed)
            self.project = nn.Conv2d(self.in_channels, self.bottleneck, kernel_size=1)
            self.latent_context = nn.Conv2d(
                self.bottleneck,
                self.bottleneck,
                kernel_size=ae_contract.AE_CONTEXT_KERNEL,
                padding=ae_contract.AE_CONTEXT_PADDING,
                groups=self.bottleneck,
            )
            self.expand = nn.Conv2d(
                self.bottleneck + 1, self.in_channels, kernel_size=1
            )
            self.spatial_context = nn.Conv2d(
                self.in_channels,
                self.in_channels,
                kernel_size=ae_contract.AE_CONTEXT_KERNEL,
                padding=ae_contract.AE_CONTEXT_PADDING,
                groups=self.in_channels,
            )
            self.activation = nn.GELU()
            self._initialize()

    # -- initialization ----------------------------------------------------

    @torch.no_grad()
    def _initialize(self) -> None:
        """Orthonormal projection, transposed decoder, zero everything else."""
        nn.init.orthogonal_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        nn.init.zeros_(self.latent_context.weight)
        nn.init.zeros_(self.latent_context.bias)

        nn.init.zeros_(self.expand.weight)
        projection = self.project.weight.reshape(self.bottleneck, self.in_channels)
        self.expand.weight[:, : self.bottleneck, 0, 0].copy_(projection.t())
        nn.init.zeros_(self.expand.bias)

        nn.init.zeros_(self.spatial_context.weight)
        nn.init.zeros_(self.spatial_context.bias)

    # -- shape handling ----------------------------------------------------

    def _as_batched(
        self, tensor: torch.Tensor, channels: int, what: str
    ) -> tuple[torch.Tensor, bool]:
        if not isinstance(tensor, torch.Tensor):
            raise guards.HybridQPayloadError(f"{what} must be a torch.Tensor")
        if tensor.dim() == 3:
            batched, squeeze = tensor.unsqueeze(0), True
        elif tensor.dim() == 4:
            batched, squeeze = tensor, False
        else:
            raise guards.HybridQPayloadError(
                f"{what} must be [C,H,W] or [N,C,H,W], got {list(tensor.shape)}"
            )
        if batched.shape[1] != channels:
            raise guards.HybridQPayloadError(
                f"{what} must have {channels} channels, got {batched.shape[1]}"
            )
        if tuple(batched.shape[2:]) != ae_contract.AE_LATENT_SPATIAL_SHAPE:
            raise guards.HybridQPayloadError(
                f"{what} spatial shape must be "
                f"{list(ae_contract.AE_LATENT_SPATIAL_SHAPE)}, "
                f"got {list(batched.shape[2:])}"
            )
        if batched.dtype is not torch.float32:
            raise guards.HybridQPayloadError(f"{what} must be float32, got {batched.dtype}")
        return batched, squeeze

    def _mask_channel(
        self, keep_mask: torch.Tensor | None, reference: torch.Tensor
    ) -> torch.Tensor:
        """Build the [N,1,H,W] FP32 mask plane; None means the q=0 all-one mask."""
        frames = int(reference.shape[0])
        if keep_mask is None:
            return torch.ones(
                frames,
                1,
                ae_contract.AE_LATENT_HEIGHT,
                ae_contract.AE_LATENT_WIDTH,
                dtype=reference.dtype,
                device=reference.device,
            )
        if not isinstance(keep_mask, torch.Tensor):
            raise guards.HybridQPayloadError("keep mask must be a torch.Tensor")
        if keep_mask.dtype is not torch.bool:
            raise guards.HybridQPayloadError(
                f"keep mask must be boolean, got {keep_mask.dtype}"
            )
        if keep_mask.dim() == 2:
            plane = keep_mask.reshape(1, 1, *keep_mask.shape).expand(frames, 1, -1, -1)
        elif keep_mask.dim() == 3:
            plane = keep_mask.unsqueeze(1)
        else:
            raise guards.HybridQPayloadError(
                f"keep mask must be [H,W] or [N,H,W], got {list(keep_mask.shape)}"
            )
        if tuple(plane.shape[2:]) != ae_contract.AE_LATENT_SPATIAL_SHAPE:
            raise guards.HybridQPayloadError(
                f"keep mask spatial shape must be "
                f"{list(ae_contract.AE_LATENT_SPATIAL_SHAPE)}, got {list(plane.shape[2:])}"
            )
        if int(plane.shape[0]) != frames:
            raise guards.HybridQPayloadError(
                f"keep mask carries {int(plane.shape[0])} frames, latent has {frames}"
            )
        return plane.to(device=reference.device, dtype=reference.dtype)

    # -- forward paths -----------------------------------------------------

    def encode(self, c2: torch.Tensor) -> torch.Tensor:
        """Original FP32 C2 -> dense latent. Accepts [256,H,W] or [N,256,H,W].

        This runs *before* any spatial dropping: the transport encodes the
        latent, so the AE must see the complete frame.
        """
        batched, squeeze = self._as_batched(c2, self.in_channels, "AE encoder input")
        projection = self.project(batched)
        latent = projection + self.latent_context(self.activation(projection))
        return latent.squeeze(0) if squeeze else latent

    def decode(
        self, latent: torch.Tensor, keep_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Latent + reconstructed keep mask -> reconstructed FP32 C2.

        `keep_mask=None` supplies the all-one mask, which is exactly the q=0
        case. No original-C2 argument exists, by design.
        """
        batched, squeeze = self._as_batched(
            latent, self.bottleneck, "AE decoder latent"
        )
        plane = self._mask_channel(keep_mask, batched)
        expansion = self.expand(torch.cat((batched, plane), dim=1))
        reconstructed = expansion + self.spatial_context(self.activation(expansion))
        return reconstructed.squeeze(0) if squeeze else reconstructed

    def forward(
        self, c2: torch.Tensor, keep_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Convenience encode-then-decode with no transport in between.

        This is not the deployed path and is not an identity at q=0: channel
        compression from 256 to B is lossy by construction.
        """
        return self.decode(self.encode(c2), keep_mask)

    # -- deployment provenance --------------------------------------------

    def bind_checkpoint(self, binding: int) -> "SplitFeatureAE":
        """Record which registered checkpoint this preloaded pair carries.

        Interface hook only: it stores one 32-bit provenance word so a decoded
        packet can be refused when it was produced by a different AE. It loads
        nothing and does not touch parameters.
        """
        self.checkpoint_binding = ae_contract.require_checkpoint_binding(binding)
        return self

    def wire_identity(self) -> dict[str, int | str]:
        """Exactly the family fields the per-frame envelope must agree with."""
        return {
            "family_id": self.family_id,
            "family_name": self.family_name,
            "bottleneck": self.bottleneck,
            "checkpoint_binding": self.checkpoint_binding,
        }

    # -- static accounting -------------------------------------------------

    def encoder_parameters(self) -> int:
        return sum(
            parameter.numel()
            for module in (self.project, self.latent_context)
            for parameter in module.parameters()
        )

    def decoder_parameters(self) -> int:
        return sum(
            parameter.numel()
            for module in (self.expand, self.spatial_context)
            for parameter in module.parameters()
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def complexity(
        self,
        height: int = ae_contract.AE_LATENT_HEIGHT,
        width: int = ae_contract.AE_LATENT_WIDTH,
    ) -> AeComplexity:
        """Analytical parameter and MAC accounting (biases excluded from MACs)."""
        cells = int(height) * int(width)
        kernel = ae_contract.AE_CONTEXT_KERNEL ** 2
        encoder_macs = (
            self.in_channels * self.bottleneck * cells + self.bottleneck * kernel * cells
        )
        decoder_macs = (
            (self.bottleneck + 1) * self.in_channels * cells
            + self.in_channels * kernel * cells
        )
        encoder_parameters = self.encoder_parameters()
        decoder_parameters = self.decoder_parameters()
        return AeComplexity(
            bottleneck=self.bottleneck,
            height=int(height),
            width=int(width),
            encoder_parameters=encoder_parameters,
            decoder_parameters=decoder_parameters,
            total_parameters=encoder_parameters + decoder_parameters,
            encoder_macs=encoder_macs,
            decoder_macs=decoder_macs,
            total_macs=encoder_macs + decoder_macs,
        )


def build_split_feature_ae(bottleneck: int) -> SplitFeatureAE:
    """Construct one registered AE family and cross-check its static accounting."""
    autoencoder = SplitFeatureAE(ae_contract.require_bottleneck(bottleneck))
    complexity = autoencoder.complexity()
    if complexity.total_parameters != autoencoder.parameter_count():
        raise guards.HybridQConfigError(
            "AE encoder/decoder parameter split does not cover every parameter"
        )
    guards.require_module_parameters_finite(autoencoder, "AE")
    return autoencoder


def ae_parameters(autoencoder: SplitFeatureAE) -> list[nn.Parameter]:
    """The only parameters any later optimizer is permitted to own."""
    if not isinstance(autoencoder, SplitFeatureAE):
        raise guards.HybridQConfigError("expected a SplitFeatureAE")
    return list(autoencoder.parameters())
