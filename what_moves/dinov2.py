# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by the WhatMoves authors: reduced to the fixed DINOv2-B/14-register
# inference architecture and preprocessing used by the released checkpoint.

"""Minimal DINOv2-B/14-register feature extractor used by WhatMoves.

This is a reduced implementation of the official Apache-2.0-licensed DINOv2
source at commit ``7764ea0f912e53c92e82eb78a2a1631e92725fc8``. It keeps only
the inference path and parameter names required by the WhatMoves checkpoint.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("Expected a pair")
        return value
    return (value, value)


class MLP(nn.Module):
    def __init__(self, width: int, hidden_width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden_width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_width, width)

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = self.fc1(tokens)
        tokens = self.act(tokens)
        return self.fc2(tokens)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        width: int = 768,
    ):
        super().__init__()
        image_height, image_width = _pair(image_size)
        patch_height, patch_width = _pair(patch_size)
        self.grid_size = (
            image_height // patch_height,
            image_width // patch_width,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_channels,
            width,
            kernel_size=(patch_height, patch_width),
            stride=(patch_height, patch_width),
        )

    def forward(self, image: Tensor) -> Tensor:
        patches = self.proj(image)
        return patches.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, width: int, num_heads: int = 12):
        super().__init__()
        if width % num_heads:
            raise ValueError("Attention width must be divisible by num_heads")
        self.num_heads = num_heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, tokens: Tensor) -> Tensor:
        batch, token_count, width = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch,
            token_count,
            3,
            self.num_heads,
            width // self.num_heads,
        )
        query, key, value = torch.unbind(qkv, dim=2)
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(
                batch,
                token_count,
                width,
            )
        )
        return self.proj(attended)


class LayerScale(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(width))

    def forward(self, tokens: Tensor) -> Tensor:
        return tokens * self.gamma


class Block(nn.Module):
    def __init__(self, width: int = 768, num_heads: int = 12):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.attn = Attention(width, num_heads=num_heads)
        self.ls1 = LayerScale(width)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = MLP(width, 4 * width)
        self.ls2 = LayerScale(width)

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.ls1(self.attn(self.norm1(tokens)))
        return tokens + self.ls2(self.mlp(self.norm2(tokens)))


class DinoVisionTransformer(nn.Module):
    """Fixed DINOv2-B/14-register architecture used by the checkpoint."""

    def __init__(self):
        super().__init__()
        self.embed_dim = 768
        self.patch_size = 14
        self.num_register_tokens = 4
        self.patch_embed = PatchEmbed(
            image_size=224,
            patch_size=self.patch_size,
            width=self.embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # Official DINOv2 register checkpoints store a 37x37 patch grid.
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 37 * 37, self.embed_dim))
        self.register_tokens = nn.Parameter(
            torch.zeros(1, self.num_register_tokens, self.embed_dim)
        )
        self.blocks = nn.ModuleList(
            Block(self.embed_dim, num_heads=12) for _ in range(12)
        )
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim))

    def _interpolate_position_encoding(
        self,
        tokens: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        patch_count = tokens.shape[1] - 1
        stored_count = self.pos_embed.shape[1] - 1
        if patch_count == stored_count and height == width:
            return self.pos_embed

        stored_side = math.isqrt(stored_count)
        if stored_side * stored_side != stored_count:
            raise RuntimeError("DINOv2 position embedding is not a square grid")
        patch_position = self.pos_embed[:, 1:].float()
        patch_position = patch_position.reshape(
            1,
            stored_side,
            stored_side,
            self.embed_dim,
        ).permute(0, 3, 1, 2)
        patch_position = F.interpolate(
            patch_position,
            size=(height // self.patch_size, width // self.patch_size),
            mode="bicubic",
            antialias=True,
        )
        patch_position = patch_position.permute(0, 2, 3, 1).reshape(
            1,
            -1,
            self.embed_dim,
        )
        positions = torch.cat((self.pos_embed[:, :1].float(), patch_position), dim=1)
        return positions.to(tokens.dtype)

    def _prepare_tokens(self, image: Tensor) -> Tensor:
        batch, _, height, width = image.shape
        patches = self.patch_embed(image)
        tokens = torch.cat(
            (self.cls_token.expand(batch, -1, -1), patches),
            dim=1,
        )
        tokens = tokens + self._interpolate_position_encoding(
            tokens,
            height,
            width,
        )
        return torch.cat(
            (
                tokens[:, :1],
                self.register_tokens.expand(batch, -1, -1),
                tokens[:, 1:],
            ),
            dim=1,
        )

    def forward_features(self, image: Tensor) -> dict[str, Tensor]:
        tokens = self._prepare_tokens(image)
        for block in self.blocks:
            tokens = block(tokens)
        normalized = self.norm(tokens)
        first_patch = 1 + self.num_register_tokens
        return {
            "x_norm_clstoken": normalized[:, 0],
            "x_norm_regtokens": normalized[:, 1:first_patch],
            "x_norm_patchtokens": normalized[:, first_patch:],
        }


class DINOv2(nn.Module):
    """Return a 16x16 DINOv2 feature grid for RGB images in ``[-1, 1]``."""

    image_size = 224
    embed_dim = 768

    def __init__(self):
        super().__init__()
        self.model = DinoVisionTransformer()

    @staticmethod
    def _resize(image: Tensor) -> Tensor:
        height, width = image.shape[-2:]
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        image = image[..., top : top + side, left : left + side]
        pooling_factor = side // DINOv2.image_size
        if pooling_factor > 1:
            image = F.avg_pool2d(image, pooling_factor)
        return F.interpolate(
            image,
            size=(DINOv2.image_size, DINOv2.image_size),
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError("DINOv2 input must be [B,H,W,3]")
        image = self._resize(image.movedim(-1, 1))
        image = image.add(1).div(2)
        mean = image.new_tensor((0.48145466, 0.4578275, 0.40821073))[:, None, None]
        std = image.new_tensor((0.26862954, 0.26130258, 0.27577711))[:, None, None]
        features = self.model.forward_features((image - mean) / std)[
            "x_norm_patchtokens"
        ]
        side = self.image_size // self.model.patch_size
        return features.reshape(image.shape[0], side, side, self.embed_dim)
