"""Conditional pixel-space Flow Matching and a non-neural inpainting baseline.

All images use floating-point BCHW RGB tensors (normally in [0, 1]); masks use
[B, 1, H, W], with 1=missing and 0=observed. The condition image fills holes
with -1. Flow states instead contain Gaussian noise/interpolated RGB in holes.
No pretrained weights or external Flow Matching library are required.
"""
import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from module import FiLMConditioning, TransformerContextEncoder

__all__ = ["UNetFlowMatching", "NeighborInterpolationBaseline",
           "make_masked_image", "square_mask", "interpolate_patch"]


def _prepare_mask(image: Tensor, mask: Tensor) -> Tensor:
    if image.ndim != 4 or image.shape[1] != 3 or min(image.shape) <= 0:
        raise ValueError("image must be nonempty RGB [B, 3, H, W]")
    if not image.is_floating_point():
        raise ValueError("image must be floating point, normally normalized to [0, 1]")
    if mask.shape != (image.shape[0], 1, image.shape[2], image.shape[3]):
        raise ValueError("mask must have shape [B, 1, H, W] matching image")
    if mask.device != image.device:
        raise ValueError("image and mask must be on the same device")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must be binary: missing=1, observed=0")
    return mask.to(dtype=image.dtype)


def make_masked_image(image: Tensor, mask: Tensor, fill_value: float = -1.0) -> Tensor:
    """Replace missing RGB with a constant; never change observed pixels."""
    mask = _prepare_mask(image, mask)
    return torch.where(mask.bool(), torch.full_like(image, fill_value), image)


def square_mask(image: Tensor, x: int, y: int, size: int) -> Tensor:
    """Make the same user-specified square for every BCHW image in a batch.

    ``x`` is the left column, ``y`` is the top row; the square must fit entirely
    inside the image. Make separate masks and concatenate for different squares.
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must have shape [B, 3, H, W]")
    if any(not isinstance(v, int) or isinstance(v, bool) for v in (x, y, size)):
        raise ValueError("x, y and size must be integers")
    if size <= 0 or x < 0 or y < 0 or x + size > image.shape[3] or y + size > image.shape[2]:
        raise ValueError("square must have positive size and fit inside the image")
    mask = image.new_zeros(image.shape[0], 1, *image.shape[-2:])
    mask[:, :, y:y + size, x:x + size] = 1
    return mask


def _norm(channels: int) -> nn.GroupNorm:
    groups = next(g for g in (8, 4, 2, 1) if channels % g == 0)
    return nn.GroupNorm(groups, channels)


def _pool_context(features: Tensor, target_size) -> Tensor:
    """Adaptive-average semantics without nondeterministic CUDA backward."""
    height, width = features.shape[-2:]
    out_h, out_w = target_size
    if (height, width) == (out_h, out_w):
        return features
    rows = []
    for y in range(out_h):
        top, bottom = y * height // out_h, ((y + 1) * height + out_h - 1) // out_h
        columns = []
        for x in range(out_w):
            left, right = x * width // out_w, ((x + 1) * width + out_w - 1) // out_w
            columns.append(features[:, :, top:bottom, left:right].mean(dim=(-2, -1)))
        rows.append(torch.stack(columns, dim=-1))
    return torch.stack(rows, dim=-2)


class _TimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.mlp = nn.Sequential(nn.Linear(dimension, dimension), nn.SiLU(),
                                 nn.Linear(dimension, dimension))

    def forward(self, time: Tensor) -> Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, device=time.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = time.float()[:, None] * frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if self.dimension % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding.to(dtype=self.mlp[0].weight.dtype))


class _ConditionedBlock(nn.Module):
    """Same visual conditioning location for additive and FiLM variants."""
    def __init__(self, in_channels: int, out_channels: int, time_dim: int,
                 context_dim: int, use_film: bool):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = _norm(out_channels)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.use_film = use_film
        self.condition = (FiLMConditioning(context_dim, out_channels, context_dim)
                          if use_film else nn.Linear(context_dim, out_channels))
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = _norm(out_channels)
        self.skip = (nn.Identity() if in_channels == out_channels
                     else nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, hidden: Tensor, time: Tensor, context: Tensor) -> Tensor:
        residual = self.skip(hidden)
        hidden = self.norm1(self.conv1(hidden))
        hidden = hidden + self.time_projection(time)[:, :, None, None]
        if self.use_film:
            hidden = self.condition(hidden, context)
        else:
            hidden = hidden + self.condition(context)[:, :, None, None]
        hidden = F.silu(hidden)
        return F.silu(self.norm2(self.conv2(hidden)) + residual)


class UNetFlowMatching(nn.Module):
    """A conditional U-Net with independent module-1/module-2 constructor flags.

    ``use_transformer=False, use_film=False`` is the basic system. Enable either
    or both flags for the other ablations; train each variant separately.
    ``forward(x_t, t, masked_image, mask)`` predicts velocity [B,3,H,W].
    ``flow_matching_loss(image, mask)`` creates its own masked conditions.
    ``sample(masked_image, mask)`` returns only the composited reconstructed RGB.

    The context CNN has three stride-2 layers. A maximum context grid bounds
    attention cost; Transformer and basic pooling see the same grid. All
    variants share U-Net topology, visual injection sites and time embedding.
    Module flags are constructor choices, not trained-weight-compatible toggles.
    """
    def __init__(self, base_channels: int = 32, context_dim: int = 128,
                 time_dim: int = 128, use_transformer: bool = False,
                 use_film: bool = False, transformer_heads: int = 4,
                 transformer_layers: int = 2, transformer_dropout: float = 0.1,
                 max_context_grid: int = 16, mask_fill_value: float = 0.0):
        super().__init__()
        if min(base_channels, context_dim, time_dim, max_context_grid) <= 0 or time_dim < 2:
            raise ValueError("dimensions/grid must be positive and time_dim must be >= 2")
        self.use_transformer = use_transformer
        self.use_film = use_film
        self.context_dim = context_dim
        self.max_context_grid = max_context_grid
        self.mask_fill_value = mask_fill_value
        channels = [base_channels * multiplier for multiplier in (1, 2, 4, 8)]
        self.context_cnn = nn.Sequential(
            nn.Conv2d(4, channels[0], 3, stride=2, padding=1), _norm(channels[0]), nn.SiLU(),
            nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1), _norm(channels[1]), nn.SiLU(),
            nn.Conv2d(channels[1], context_dim, 3, stride=2, padding=1), _norm(context_dim), nn.SiLU(),
        )
        self.context_encoder = (
            TransformerContextEncoder(
                in_channels=context_dim, context_dim=context_dim, embed_dim=context_dim,
                num_heads=transformer_heads, num_layers=transformer_layers,
                feedforward_dim=2 * context_dim, dropout=transformer_dropout,
                position_grid_size=(max_context_grid, max_context_grid))
            if use_transformer else nn.Linear(context_dim, context_dim)
        )
        self.time_embedding = _TimeEmbedding(time_dim)
        self.input_projection = nn.Conv2d(4, channels[0], 3, padding=1)
        self.encoders = nn.ModuleList([
            _ConditionedBlock(c, c, time_dim, context_dim, use_film) for c in channels
        ])
        self.downsamples = nn.ModuleList([
            nn.Conv2d(channels[i], channels[i + 1], 3, stride=2, padding=1)
            for i in range(3)
        ])
        self.decoders = nn.ModuleList([
            _ConditionedBlock(channels[i + 1] + channels[i], channels[i],
                              time_dim, context_dim, use_film)
            for i in (2, 1, 0)
        ])
        # Velocity is unbounded: no sigmoid/tanh on the output.
        self.output_projection = nn.Conv2d(channels[0], 3, 3, padding=1)

    def normalize_image(self, image: Tensor) -> Tensor:
        """Map display RGB [0,1] to model RGB [-1,1]."""
        return image * 2 - 1

    def denormalize_image(self, image: Tensor) -> Tensor:
        """Map model RGB [-1,1] back to display RGB [0,1]; no clipping."""
        return (image + 1) / 2

    def _encode_context(self, masked_image: Tensor, mask: Tensor) -> Tensor:
        # Sanitizing again prevents accidental ground-truth leakage if callers
        # provide a clean photograph to forward/sample instead of a masked one.
        observed = torch.where(mask.bool(), torch.full_like(masked_image, self.mask_fill_value), masked_image)
        features = self.context_cnn(torch.cat((observed, mask), dim=1))
        target_size = tuple(min(size, self.max_context_grid) for size in features.shape[-2:])
        features = _pool_context(features, target_size)
        if self.use_transformer:
            return self.context_encoder(features)
        return self.context_encoder(features.mean(dim=(2, 3)))

    def encode_context(self, masked_image: Tensor, mask: Tensor) -> Tensor:
        """Encode fixed observed context; sampling reuses it at every step."""
        mask = _prepare_mask(masked_image, mask)
        return self._encode_context(masked_image, mask)

    @staticmethod
    def _prepare_time(t, image: Tensor) -> Tensor:
        time = torch.as_tensor(t, device=image.device, dtype=torch.float32)
        if time.ndim == 0:
            time = time.expand(image.shape[0])
        elif time.shape == (image.shape[0], 1):
            time = time[:, 0]
        if time.shape != (image.shape[0],) or not torch.all((time >= 0) & (time <= 1)):
            raise ValueError("t must be a scalar, [B], or [B,1] with values in [0,1]")
        return time

    def _predict_velocity(self, state: Tensor, time: Tensor, mask: Tensor,
                          context: Tensor) -> Tensor:
        time_features = self.time_embedding(time)
        hidden = self.input_projection(torch.cat((state, mask), dim=1))
        skips = []
        for level, encoder in enumerate(self.encoders):
            hidden = encoder(hidden, time_features, context)
            if level < 3:
                skips.append(hidden)
                hidden = self.downsamples[level](hidden)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            hidden = decoder(torch.cat((hidden, skip), dim=1), time_features, context)
        velocity = self.output_projection(hidden)
        return torch.where(mask.bool(), velocity, torch.zeros_like(velocity))

    def forward(self, x_t: Tensor, t, masked_image: Tensor, mask: Tensor,
                context: Optional[Tensor] = None) -> Tensor:
        mask = _prepare_mask(masked_image, mask)
        if x_t.shape != masked_image.shape or x_t.device != masked_image.device or not x_t.is_floating_point():
            raise ValueError("x_t must match masked_image shape/device and be floating point")
        time = self._prepare_time(t, x_t)
        if context is None:
            context = self._encode_context(masked_image, mask)
        if context.shape != (x_t.shape[0], self.context_dim) or context.device != x_t.device:
            raise ValueError("context must be [B, context_dim] on the image device")
        state = torch.where(mask.bool(), x_t, masked_image)
        return self._predict_velocity(state, time, mask, context)

    def flow_matching_loss(self, image: Tensor, mask: Tensor, *, t=None,
                           noise: Optional[Tensor] = None,
                           generator: Optional[torch.Generator] = None) -> Tensor:
        """Masked velocity MSE on x_t=(1-t)*noise+t*target within each hole.

        Each image's loss is divided by its own masked RGB count, then batch
        averaged, so larger holes do not automatically dominate. Every training
        sample must contain a nonempty mask. This is a training objective, not
        the final reconstruction MAE evaluation metric.
        """
        mask = _prepare_mask(image, mask)
        counts = mask.sum(dim=(1, 2, 3)) * image.shape[1]
        if torch.any(counts == 0):
            raise ValueError("Every training image must have at least one masked pixel")
        if noise is None:
            noise = torch.randn(image.shape, device=image.device, dtype=image.dtype, generator=generator)
        if noise.shape != image.shape or noise.device != image.device or not noise.is_floating_point():
            raise ValueError("noise must match image shape/device and be floating point")
        if t is None:
            t = torch.rand(image.shape[0], device=image.device, generator=generator)
        time = self._prepare_time(t, image)
        blend = time[:, None, None, None].to(image.dtype)
        state = torch.where(mask.bool(), (1 - blend) * noise + blend * image, image)
        masked_image = torch.where(mask.bool(), torch.full_like(image, self.mask_fill_value), image)
        prediction = self(state, time, masked_image, mask)
        error = torch.where(mask.bool(), (prediction.float() - (image - noise).float()).square(), 0.0)
        return (error.sum(dim=(1, 2, 3)) / counts.float()).mean()

    @torch.no_grad()
    def sample(self, masked_image: Tensor, mask: Tensor, *, num_steps: int = 50,
               noise: Optional[Tensor] = None,
               generator: Optional[torch.Generator] = None, clamp: bool = True,
               step_callback=None) -> Tensor:
        """Integrate t=0 -> 1 with Euler; observed pixels stay exact.

        Euler uses one velocity evaluation per step. A fixed
        noise tensor or device-matched Generator makes comparisons reproducible.
        Context is computed once with dropout disabled. Only final missing RGB
        is clamped to [-1,1]; intermediate states and velocities are not clamped.
        Optional step_callback(step, state) receives isolated tensor snapshots
        at step 0 and after every integration step, without altering sampling.
        """
        mask = _prepare_mask(masked_image, mask)
        if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps <= 0:
            raise ValueError("num_steps must be a positive integer")
        if not mask.bool().any():
            return masked_image.clone()
        if noise is None:
            noise = torch.randn(masked_image.shape, device=masked_image.device,
                                dtype=masked_image.dtype, generator=generator)
        if noise.shape != masked_image.shape or noise.device != masked_image.device or not noise.is_floating_point():
            raise ValueError("noise must match masked_image shape/device and be floating point")
        was_training = self.training
        self.eval()
        try:
            context = self._encode_context(masked_image, mask)
            state = torch.where(mask.bool(), noise, masked_image)
            if step_callback is not None:
                step_callback(0, state.detach().clone())
            dt = 1.0 / num_steps
            for step in range(num_steps):
                time = torch.full((state.shape[0],), step / num_steps, device=state.device)
                velocity = self._predict_velocity(state, time, mask, context)
                proposed = state + dt * velocity
                state = torch.where(mask.bool(), proposed, masked_image)
                if step_callback is not None:
                    step_callback(step + 1, state.detach().clone())
            if clamp:
                state = state.clamp(-1, 1)
            return torch.where(mask.bool(), state, masked_image)
        finally:
            self.train(was_training)


class NeighborInterpolationBaseline(nn.Module):
    """Non-trainable nearest-observed-neighbour interpolation baseline.

    Call ``baseline(masked_image, mask)`` with RGB [B,3,H,W] and a binary
    [B,1,H,W] mask (1=missing). Copy the RGB of the nearest visible pixel using
    Euclidean spatial distance. Ties choose the smallest row-major pixel index.
    There are no parameters, iterations or random choices. Computation follows
    the inputs' device; train/eval mode does not change the result.
    """
    def forward(self, masked_image: Tensor, mask: Tensor) -> Tensor:
        return interpolate_patch(masked_image, mask)


@torch.no_grad()
def interpolate_patch(masked_image: Tensor, mask: Tensor) -> Tensor:
    """Copy each missing RGB pixel from the closest original observed pixel.

    Distances are squared Euclidean distances on integer pixel coordinates;
    exact ties use the smallest row-major index (top-to-bottom, left-to-right).
    Only visible pixels bordering a hole need to be searched: a closest visible
    pixel must have a masked four-neighbour, otherwise a step toward the query
    would be a closer visible pixel. This optimization does not change nearest
    neighbour results, including irregular/edge masks. Distances are chunked
    to bound temporary memory. No generated pixel becomes a source for another.

    Returns BCHW RGB with unchanged dtype/device and exact observed pixels.
    A completely covered image has no possible source and is rejected.
    """
    mask = _prepare_mask(masked_image, mask)
    if not mask.bool().any():
        return masked_image.clone()
    observed = ~mask.bool()
    if torch.any(observed.sum(dim=(1, 2, 3)) == 0):
        raise ValueError("Interpolation requires at least one observed pixel per image")
    kernel = masked_image.new_tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32)[None, None]
    boundary = observed & (F.conv2d(mask.float(), kernel, padding=1) > 0)
    result = masked_image.clone(memory_format=torch.contiguous_format)
    width = masked_image.shape[-1]
    for batch_index in range(len(masked_image)):
        missing = mask[batch_index, 0].reshape(-1).nonzero(as_tuple=True)[0]
        if missing.numel() == 0:
            continue
        sources = boundary[batch_index, 0].reshape(-1).nonzero(as_tuple=True)[0]
        source_y, source_x = sources // width, sources % width
        original = masked_image[batch_index].reshape(3, -1)
        reconstructed = result[batch_index].view(3, -1)
        for indices in missing.split(256):
            dy = indices[:, None] // width - source_y[None, :]
            dx = indices[:, None] % width - source_x[None, :]
            nearest = sources[(dy.square() + dx.square()).argmin(dim=1)]
            reconstructed[:, indices] = original[:, nearest]
    return result
