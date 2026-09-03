"""Small transformer implementation used by the released WhatMoves encoder."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn


def _rms_norm(x: Tensor, scale: Tensor, eps: float) -> Tensor:
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_square = torch.mean(x.to(dtype).square(), dim=-1, keepdim=True)
    normalized_scale = scale.to(dtype) * torch.rsqrt(mean_square + eps)
    return x * normalized_scale.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return _rms_norm(x, self.scale, self.eps)


class LinearSwiGLU(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, 2 * out_features, bias=False)
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        value, gate = F.linear(x, self.weight).chunk(2, dim=-1)
        return value * F.silu(gate)


class FeedForwardBlock(nn.Module):
    def __init__(self, width: int, hidden_width: int, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(width)
        self.up_proj = LinearSwiGLU(width, hidden_width)
        self.dropout = nn.Dropout(dropout)
        self.down_proj = nn.Linear(hidden_width, width, bias=False)
        nn.init.zeros_(self.down_proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        skip = x
        x = self.up_proj(self.norm(x))
        return self.down_proj(self.dropout(x)) + skip


def _scale_for_cosine_attention(
    query: Tensor,
    key: Tensor,
    scale: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    dtype = reduce(
        torch.promote_types,
        (query.dtype, key.dtype, scale.dtype, torch.float32),
    )
    query_square = query.to(dtype).square().sum(dim=-1, keepdim=True)
    key_square = key.to(dtype).square().sum(dim=-1, keepdim=True)
    root_scale = torch.sqrt(scale.to(dtype))
    query_scale = root_scale * torch.rsqrt(query_square + eps)
    key_scale = root_scale * torch.rsqrt(key_square + eps)
    return query * query_scale.to(query.dtype), key * key_scale.to(key.dtype)


@dataclass
class _AxisFrequencies:
    minimum: float
    maximum: float


class _AxialRoPE(nn.Module):
    def __init__(
        self,
        head_width: int,
        num_heads: int,
        axes: tuple[str, ...],
        partial_axes: int,
    ):
        super().__init__()
        self.head_width = head_width
        frequencies = {
            "t": _AxisFrequencies(1.0, 0.01),
            "h": _AxisFrequencies(math.pi, 10.0 * math.pi),
            "w": _AxisFrequencies(math.pi, 10.0 * math.pi),
        }
        axis_width = head_width // 2 // partial_axes
        for axis in axes:
            spec = frequencies[axis]
            values = torch.linspace(
                math.log(spec.minimum),
                math.log(spec.maximum),
                num_heads * axis_width + 1,
            )[:-1].exp()
            self.register_buffer(
                f"freqs_{axis}",
                values.reshape(axis_width, num_heads).mT.contiguous(),
            )
        self.axes = axes

    def forward(self, positions: Tensor) -> Tensor:
        parts = []
        for index, axis in enumerate(self.axes):
            frequencies = getattr(self, f"freqs_{axis}").to(positions.dtype)
            parts.append(positions[..., index, None, None] * frequencies)
        return torch.cat(parts, dim=-1)

    def apply(self, x: Tensor, theta: Tensor) -> Tensor:
        dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
        encoded_width = theta.shape[-1]
        first = x[..., :encoded_width].to(dtype)
        second = x[..., encoded_width : 2 * encoded_width].to(dtype)
        remainder = x[..., 2 * encoded_width :]
        cosine, sine = torch.cos(theta.to(dtype)), torch.sin(theta.to(dtype))
        rotated_first = first * cosine - second * sine
        rotated_second = second * cosine + first * sine
        return torch.cat(
            (
                rotated_first.to(x.dtype),
                rotated_second.to(x.dtype),
                remainder,
            ),
            dim=-1,
        )


class AxialRoPE2D(_AxialRoPE):
    def __init__(self, head_width: int, num_heads: int):
        super().__init__(head_width, num_heads, ("h", "w"), partial_axes=4)


class AxialRoPE2DT(_AxialRoPE):
    def __init__(self, head_width: int, num_heads: int):
        super().__init__(head_width, num_heads, ("t", "h", "w"), partial_axes=4)


def _centers(
    start: float,
    stop: float,
    count: int,
    *,
    device: torch.device,
) -> Tensor:
    edges = torch.linspace(start, stop, count + 1, device=device)
    return (edges[:-1] + edges[1:]) / 2


def position_grid_2d(
    batch: int,
    height: int,
    width: int,
    *,
    device: torch.device,
) -> Tensor:
    aspect = width / height
    y_min, y_max = (-1 / aspect, 1 / aspect) if aspect > 1 else (-1.0, 1.0)
    x_min, x_max = (-aspect, aspect) if aspect < 1 else (-1.0, 1.0)
    y = _centers(y_min, y_max, height, device=device)
    x = _centers(x_min, x_max, width, device=device)
    grid = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)
    return grid[None].expand(batch, height, width, 2)


def position_grid_2dt(
    batch: int,
    frames: int,
    height: int,
    width: int,
    *,
    device: torch.device,
) -> Tensor:
    time = torch.arange(frames, device=device, dtype=torch.float32)
    time = time.view(1, frames, 1).expand(batch, frames, 1)
    space = position_grid_2d(batch, height, width, device=device)
    return torch.cat(
        (
            repeat(time, "b t c -> b t h w c", h=height, w=width),
            repeat(space, "b h w c -> b t h w c", t=frames),
        ),
        dim=-1,
    )


class AttentionBlock(nn.Module):
    def __init__(
        self,
        width: int,
        head_width: int,
        rope: type[_AxialRoPE],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_head = head_width
        self.n_heads = width // head_width
        self.norm = RMSNorm(width)
        self.qkv_proj = nn.Linear(width, 3 * width, bias=False)
        self.scale = nn.Parameter(torch.full((self.n_heads,), 10.0))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        self.pos_emb = rope(head_width, self.n_heads)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        skip = x
        qkv = self.qkv_proj(self.norm(x))
        query, key, value = rearrange(
            qkv,
            "b l (q h d) -> q b h l d",
            q=3,
            d=self.d_head,
        )
        query, key = _scale_for_cosine_attention(
            query,
            key,
            self.scale[:, None, None],
        )
        theta = self.pos_emb(positions.to(qkv.dtype)).movedim(-2, -3)
        query = self.pos_emb.apply(query, theta)
        key = self.pos_emb.apply(key, theta)
        if attention_mask is not None and attention_mask.ndim == 3:
            attention_mask = attention_mask[:, None]
        x = F.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=1.0,
            attn_mask=attention_mask,
        )
        x = rearrange(x, "b h l d -> b l (h d)")
        return self.out_proj(self.dropout(x)) + skip


class TransformerLayer(nn.Module):
    def __init__(
        self,
        width: int,
        head_width: int,
        rope: type[_AxialRoPE],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = AttentionBlock(width, head_width, rope, dropout)
        self.ff = FeedForwardBlock(width, 3 * width, dropout)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.self_attn(x, positions, attention_mask)
        return self.ff(x)
