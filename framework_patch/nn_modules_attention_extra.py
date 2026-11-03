# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Attention modules added for the safety-helmet detection project (SE / CBAM).

These are *project-level* extensions to the stock Ultralytics modules. They are
registered in ``parse_model`` so that they can be referenced directly from a
model YAML, e.g.::

    - [-1, 1, CBAMBlock, [1024]]   # spatial + channel attention, output 1024 ch

Both modules accept ``(c1, c2)`` so that ``parse_model`` can feed them the
incoming channel count automatically, and internally insert a 1x1 projection
when ``c1 != c2`` (this keeps the module usable anywhere in the network without
having to reason about channel matching in the YAML).
"""

from __future__ import annotations

import torch
from torch import nn

from .conv import Conv

__all__ = ("SEAttention", "CBAMBlock")


class SEAttention(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018).

    Global-average-pools each channel, passes it through a bottleneck MLP and
    rescales the feature map with the resulting sigmoid weights. Cheap and
    effective for "which channels matter" re-weighting on small datasets.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels (defaults to ``c1`` when ``None``).
        reduction (int): Bottleneck reduction ratio, clamped so the hidden layer
            never drops below 8 channels.
    """

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 16):
        super().__init__()
        c2 = c2 or c1
        self.proj = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
        hidden = max(c2 // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c2, hidden, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, c2, 1, bias=True),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention."""
        y = self.proj(x)
        weight = self.act(self.fc(self.pool(y)))
        return y * weight


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018).

    Applies channel attention (shared MLP over average- and max-pooled
    descriptors) followed by spatial attention (7x7 conv over the channel-wise
    average/max maps). The spatial branch is what helps most on this project:
    it suppresses background clutter and makes partially occluded helmets /
    vests more separable, which is exactly the failure mode we care about.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels (defaults to ``c1`` when ``None``).
        kernel_size (int): Spatial attention kernel size (7 in the paper).
        reduction (int): Channel-bottleneck reduction ratio.
    """

    def __init__(self, c1: int, c2: int | None = None, kernel_size: int = 7, reduction: int = 16):
        super().__init__()
        c2 = c2 or c1
        self.proj = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()

        # --- channel attention: shared MLP, one pass for avg-pool and one for max-pool
        hidden = max(c2 // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(c2, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, bias=True),
        )
        self.channel_act = nn.Sigmoid()

        # --- spatial attention: 7x7 conv on the concatenated [avg, max] maps
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=True)
        self.spatial_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention followed by spatial attention."""
        y = self.proj(x)

        avg_out = self.mlp(torch.mean(y, dim=(2, 3), keepdim=True))
        max_out = self.mlp(torch.amax(y, dim=(2, 3), keepdim=True))
        y = y * self.channel_act(avg_out + max_out)

        avg_map = torch.mean(y, dim=1, keepdim=True)
        max_map = torch.amax(y, dim=1, keepdim=True)
        y = y * self.spatial_act(self.spatial_conv(torch.cat((avg_map, max_map), dim=1)))
        return y
