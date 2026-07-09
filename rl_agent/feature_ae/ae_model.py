"""Task-aware feature autoencoder for the fusion split point.

Compresses the backbone 'high' feature level (~960 ch) to a channel bottleneck {128,64,32} and back.
Structure mirrors the supervisor's rd_ae_b128 checkpoint (encoder 1x1, decoder 1x1, importance_head
bottleneck->bottleneck/2->bottleneck) but re-dimensioned for the fusion path (960-ch) instead of the
256-ch OD/FPN path. The importance head scores per-bottleneck-channel importance -> lets the agent drop
the least-important channels under a byte budget (workstream E), and gives a hook for informed loss.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class FeatureAE(nn.Module):
    def __init__(self, in_channels: int, bottleneck: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.bottleneck = int(bottleneck)
        self.encoder = nn.Conv2d(in_channels, bottleneck, kernel_size=1)
        self.decoder = nn.Conv2d(bottleneck, in_channels, kernel_size=1)
        self.importance_head = nn.Sequential(
            nn.Conv2d(bottleneck, max(1, bottleneck // 2), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, bottleneck // 2), bottleneck, kernel_size=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def channel_importance(self, z: torch.Tensor) -> torch.Tensor:
        """Per-bottleneck-channel importance in (0,1) (spatially pooled). Higher = keep under budget."""
        gate = torch.sigmoid(self.importance_head(z))          # (B, bottleneck, H, W)
        return gate.mean(dim=(2, 3))                            # (B, bottleneck)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        importance = torch.sigmoid(self.importance_head(z))     # per-cell gate (B, bottleneck, H, W)
        return x_hat, z, importance


class FeatureAEv2(nn.Module):
    """Nonlinear, spatially-aware feature AE (same bottleneck/payload as FeatureAE, better codec).

    v1 (FeatureAE) is a single 1x1 conv each way = a LINEAR, per-pixel, low-rank channel projection.
    It preserves segmentation (coarse, redundant) but discards the high-dimensional feature detail the
    object head regresses from -> detection collapses, and no loss reweighting recovers it (the info is
    gone at encode). v2 keeps the SAME bottleneck channel count (identical on-wire payload) but makes the
    encoder/decoder expressive: a 3x3 conv (spatial context) + hidden layer + GELU, so the bottleneck can
    carry object-critical structure. Principled fix: same compression rate, a codec with capacity to
    preserve the task-relevant subspace."""

    def __init__(self, in_channels: int, bottleneck: int, hidden: int = 0) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.bottleneck = int(bottleneck)
        h = int(hidden) if hidden else max(2 * int(bottleneck), 384)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, h, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(h, h, kernel_size=1), nn.GELU(),
            nn.Conv2d(h, bottleneck, kernel_size=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(bottleneck, h, kernel_size=1), nn.GELU(),
            nn.Conv2d(h, h, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(h, in_channels, kernel_size=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def channel_importance(self, z: torch.Tensor) -> torch.Tensor:
        return z.abs().mean(dim=(2, 3))

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z), z, None


def build_ae(arch: str, in_channels: int, bottleneck: int) -> nn.Module:
    """Factory keyed on the checkpoint's recorded arch, so eval/client rebuild the right class."""
    if str(arch).lower() in ("v2", "nonlinear", "spatial"):
        return FeatureAEv2(in_channels, bottleneck)
    return FeatureAE(in_channels, bottleneck)
