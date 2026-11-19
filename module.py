from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["TransformerContextEncoder", "FiLMConditioning"]


class TransformerContextEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 128,
        context_dim: int = 128,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
        position_grid_size: Tuple[int, int] = (16, 16),
    ) -> None:
        super().__init__()
        dimensions = (in_channels, context_dim, embed_dim, num_heads, num_layers,
                      feedforward_dim)
        if any(value <= 0 for value in dimensions):
            raise ValueError("Channel dimensions, heads and layer counts must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if len(position_grid_size) != 2 or any(size <= 0 for size in position_grid_size):
            raise ValueError("position_grid_size must contain two positive dimensions")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.in_channels = in_channels
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.input_projection = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.position_embedding = nn.Parameter(
            torch.empty(1, embed_dim, *position_grid_size)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
            enable_nested_tensor=False,
        )
        self.output_projection = nn.Linear(embed_dim, context_dim)

    def forward_features(self, features: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError("features must have shape [B, in_channels, H, W]")
        if features.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} CNN feature channels")
        if features.shape[2] == 0 or features.shape[3] == 0:
            raise ValueError("CNN feature height and width must be nonzero")

        projected = self.input_projection(features)
        batch, _, height, width = projected.shape
        positions = F.interpolate(
            self.position_embedding,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).to(dtype=projected.dtype)
        tokens = (projected + positions).flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        return tokens.transpose(1, 2).reshape(batch, self.embed_dim, height, width)

    def forward(self, features: Tensor) -> Tensor:
        spatial_features = self.forward_features(features)
        return self.output_projection(spatial_features.mean(dim=(2, 3)))


class FiLMConditioning(nn.Module):

    def __init__(
        self,
        context_dim: int = 128,
        feature_channels: int = 128,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if min(context_dim, feature_channels, hidden_dim) <= 0:
            raise ValueError("context_dim, feature_channels and hidden_dim must be positive")
        self.context_dim = context_dim
        self.feature_channels = feature_channels
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * feature_channels),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, hidden: Tensor, context: Tensor) -> Tensor:
        if hidden.ndim != 4:
            raise ValueError("hidden must have shape [B, feature_channels, H, W]")
        if hidden.shape[1] != self.feature_channels:
            raise ValueError(f"Expected {self.feature_channels} U-Net feature channels")
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError(f"context must have shape [B, {self.context_dim}]")
        if context.shape[0] != hidden.shape[0]:
            raise ValueError("hidden and context must have the same batch size")

        gamma, beta = self.mlp(context).to(dtype=hidden.dtype).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return (1 + gamma) * hidden + beta
