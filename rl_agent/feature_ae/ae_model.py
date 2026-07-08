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
