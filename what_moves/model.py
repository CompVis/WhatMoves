"""Self-contained inference model for localized WhatMoves embeddings.

The released model has a fixed architecture and no configuration-framework
dependency. It accepts the compact encoder-only release checkpoint as well as
compatible full training checkpoints, whose decoder parameters are ignored.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .dinov2 import DINOv2
from .transformer import (
    AxialRoPE2D,
    AxialRoPE2DT,
    RMSNorm,
    TransformerLayer,
    position_grid_2d,
    position_grid_2dt,
)


class TokenTransformer(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        head_width: int,
        rope: type[AxialRoPE2D] | type[AxialRoPE2DT],
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    width=width,
                    head_width=head_width,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.norm = RMSNorm(width)

    def forward(
        self,
        tokens: Tensor,
        positions: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            tokens = layer(tokens, positions, attention_mask)
        return self.norm(tokens)


class ContentEncoder(nn.Module):
    """Encode one static image and one or more region masks."""

    d_content = 512

    def __init__(self):
        super().__init__()
        self.frame_proj = nn.Linear(768, 512, bias=False)
        self.mask_proj = nn.Linear(1, 512, bias=False)
        self.cls_token = nn.Parameter(torch.randn(512) * 0.02)
        self.transformer = TokenTransformer(
            width=512,
            depth=6,
            head_width=64,
            rope=AxialRoPE2D,
        )
        self.out = nn.Sequential(
            nn.Linear(512, self.d_content, bias=False),
            RMSNorm(self.d_content),
        )

    def forward(
        self,
        frame_features: Tensor,
        masks: Tensor,
        valid: Tensor,
    ) -> Tensor:
        batch, regions, height, width, _ = frame_features.shape
        features = frame_features.reshape(batch * regions, height, width, -1)
        patches = self.frame_proj(features)
        patch_height, patch_width = patches.shape[1:3]
        mask_tokens = F.interpolate(
            masks.reshape(batch * regions, 1, *masks.shape[-2:]).to(features.dtype),
            size=(patch_height, patch_width),
            mode="area",
        ).movedim(1, -1)
        patches = patches + self.mask_proj(mask_tokens)

        tokens = rearrange(patches, "br h w c -> br (h w) c")
        cls = repeat(
            self.cls_token.to(tokens.dtype),
            "c -> br 1 c",
            br=batch * regions,
        )
        tokens = torch.cat((cls, tokens), dim=1)
        patch_positions = position_grid_2d(
            batch * regions,
            patch_height,
            patch_width,
            device=tokens.device,
        ).reshape(batch * regions, patch_height * patch_width, 2)
        positions = torch.cat(
            (patch_positions.new_zeros((batch * regions, 1, 2)), patch_positions),
            dim=1,
        )
        embeddings = self.out(self.transformer(tokens, positions)[:, 0])
        embeddings = embeddings.reshape(batch, regions, self.d_content)
        return torch.where(valid[:, :, None], embeddings, 0.0)


class MotionEncoder(nn.Module):
    """Query full-video features with static, region-specific content tokens."""

    d_motion = 384

    def __init__(self):
        super().__init__()
        self.width = 768
        self.frame_proj = nn.Linear(768, self.width, bias=False)
        self.content_proj = nn.Linear(ContentEncoder.d_content, self.width, bias=False)
        self.query_token = nn.Parameter(torch.randn(self.width) * 0.02)
        self.transformer = TokenTransformer(
            width=self.width,
            depth=12,
            head_width=64,
            rope=AxialRoPE2DT,
        )
        self.out = nn.Sequential(
            nn.Linear(self.width, self.d_motion, bias=False),
            RMSNorm(self.d_motion),
        )

    def forward(
        self,
        frame_features: Tensor,
        content: Tensor,
        valid: Tensor,
        *,
        query_steps: int,
    ) -> Tensor:
        batch, frames, height, width, _ = frame_features.shape
        regions = content.shape[1]
        if not 0 < query_steps <= frames:
            raise ValueError("query_steps must be in [1, number of frames]")

        frame_tokens = self.frame_proj(
            frame_features.reshape(batch * frames, height, width, -1)
        ).reshape(batch, frames, height, width, self.width)
        frame_tokens = rearrange(frame_tokens, "b t h w c -> b (t h w) c")
        frame_positions = position_grid_2dt(
            batch,
            frames,
            height,
            width,
            device=frame_features.device,
        ).reshape(batch, frames * height * width, 3)

        safe_content = torch.where(valid[:, :, None], content, 0.0)
        content_tokens = self.content_proj(safe_content)
        query_tokens = content_tokens[:, None] + self.query_token.to(
            content_tokens.dtype
        ).view(1, 1, 1, self.width)
        query_tokens = query_tokens.expand(
            batch,
            query_steps,
            regions,
            self.width,
        )
        time = torch.arange(
            query_steps,
            device=frame_features.device,
            dtype=torch.float32,
        ).view(1, query_steps, 1, 1)
        query_positions = torch.cat(
            (
                time.expand(batch, query_steps, regions, 1),
                torch.zeros(
                    batch,
                    query_steps,
                    regions,
                    2,
                    device=frame_features.device,
                ),
            ),
            dim=-1,
        )
        query_tokens = rearrange(query_tokens, "b t k c -> b (t k) c")
        query_positions = rearrange(query_positions, "b t k c -> b (t k) c")
        tokens = torch.cat((frame_tokens, query_tokens), dim=1)
        positions = torch.cat((frame_positions, query_positions), dim=1)

        frame_valid = torch.ones(
            batch,
            frame_tokens.shape[1],
            device=valid.device,
            dtype=torch.bool,
        )
        query_valid = repeat(valid.bool(), "b k -> b (t k)", t=query_steps)
        key_valid = torch.cat((frame_valid, query_valid), dim=1)
        attention_mask = key_valid[:, None].expand(-1, key_valid.shape[1], -1)
        tokens = self.transformer(tokens, positions, attention_mask)
        motion = tokens[:, -query_steps * regions :].reshape(
            batch,
            query_steps,
            regions,
            self.width,
        )
        motion = self.out(motion)
        return torch.where(valid[:, None, :, None], motion, 0.0)


def _sliding_window_plan(
    sequence_length: int,
    *,
    window_length: int,
    stride: int,
    emitted_steps: int,
) -> tuple[list[int], Tensor]:
    if sequence_length < window_length:
        raise ValueError(
            f"WhatMoves needs at least {window_length} frames, got {sequence_length}"
        )
    if not 1 <= stride <= emitted_steps:
        raise ValueError(f"window_stride must be in [1, {emitted_steps}]")

    final_start = sequence_length - window_length
    starts = list(range(0, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    seen: set[int] = set()
    gather = []
    for window_index, start in enumerate(starts):
        for relative_index in range(emitted_steps):
            absolute_index = start + relative_index
            if absolute_index not in seen:
                seen.add(absolute_index)
                gather.append(window_index * emitted_steps + relative_index)
    expected = list(range(sequence_length - window_length + emitted_steps))
    if sorted(seen) != expected:
        raise RuntimeError("sliding windows did not cover a contiguous motion stream")
    return starts, torch.tensor(gather, dtype=torch.long)


def _temporal_mask_frame_plan(
    valid: Tensor,
    starts: list[int],
    *,
    window_length: int,
) -> tuple[Tensor, Tensor]:
    """Select the first valid mask at or after each window start."""
    if valid.ndim != 3:
        raise ValueError("temporal mask validity must be [B,T,K]")
    selected = []
    available = []
    for start in starts:
        window_valid = valid[:, start : start + window_length]
        has_current = window_valid.any(dim=1)
        first_offset = window_valid.to(torch.int64).argmax(dim=1)
        selected.append(first_offset + start)
        available.append(has_current)
    return torch.stack(selected, dim=1), torch.stack(available, dim=1)


def _reuse_previous_content(candidates: Tensor, available: Tensor) -> Tensor:
    """Use the previous embedding where a window has no valid region mask."""
    if candidates.ndim != 4 or available.shape != candidates.shape[:3]:
        raise ValueError("content candidates/availability must be [B,W,K,D]/[B,W,K]")
    previous = torch.zeros_like(candidates[:, 0])
    previously_available = torch.zeros_like(available[:, 0])
    resolved = []
    for window_index in range(candidates.shape[1]):
        current_available = available[:, window_index]
        unavailable = ~current_available & ~previously_available
        if bool(unavailable.any()):
            batch_index, region_index = unavailable.nonzero()[0].tolist()
            raise ValueError(
                "Temporal masks cannot initialize region "
                f"{region_index} in batch {batch_index}: the first extraction "
                "window has no valid mask for that region"
            )
        current = torch.where(
            current_available[:, :, None],
            candidates[:, window_index],
            previous,
        )
        resolved.append(current)
        previous = current
        previously_available |= current_available
    return torch.stack(resolved, dim=1)


class WhatMoves(nn.Module):
    """Released localized-motion encoder.

    Inputs are floating-point RGB tensors in ``[-1, 1]`` with channels last.
    Masks select one or more regions in either one static reference frame or a
    frame-aligned temporal mask stream.

    ``encode_content(image, masks)`` returns ``[B,K,512]``.
    ``encode_motion(video, masks)`` returns ``[B,T-3,K,384]``. Omitting masks
    returns the learned global-motion stream as ``[B,T-3,384]``.
    """

    image_size = 256
    window_length = 8
    emitted_steps = 5
    lookahead = window_length - emitted_steps
    content_dim = ContentEncoder.d_content
    motion_dim = MotionEncoder.d_motion

    _CHECKPOINT_PREFIXES = (
        "frame_encoder.",
        "content_encoder.",
        "motion_encoder.",
        "global_motion_query",
    )

    def __init__(
        self,
        *,
        frame_chunk_size: int = 32,
        window_chunk_size: int = 8,
        window_stride: int = 1,
    ):
        """Build the fixed release architecture.

        ``frame_chunk_size`` and ``window_chunk_size`` are per-video counts;
        their effective forward batch sizes are respectively ``B * count``.
        Lower either value to reduce peak inference memory without changing
        the result.
        """
        super().__init__()
        if frame_chunk_size < 1 or window_chunk_size < 1:
            raise ValueError("frame and window chunk sizes must be positive")
        if not 1 <= window_stride <= self.emitted_steps:
            raise ValueError(f"window_stride must be in [1, {self.emitted_steps}]")
        self.frame_encoder = DINOv2()
        self.content_encoder = ContentEncoder()
        self.motion_encoder = MotionEncoder()
        self.global_motion_query = nn.Parameter(torch.zeros(self.content_dim))
        self.frame_chunk_size = int(frame_chunk_size)
        self.window_chunk_size = int(window_chunk_size)
        self.window_stride = int(window_stride)
        self._compute_dtype: torch.dtype | None = None

    @property
    def device(self) -> torch.device:
        return self.global_motion_query.device

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        mmap: bool = True,
        **model_kwargs,
    ) -> "WhatMoves":
        model = cls(**model_kwargs)
        model.load_checkpoint(checkpoint, mmap=mmap)
        model.requires_grad_(False).eval()
        model = model.to(device=device, dtype=dtype)
        model._compute_dtype = (
            torch.bfloat16 if dtype is None and model.device.type == "cuda" else dtype
        )
        return model

    def load_checkpoint(
        self,
        checkpoint: str | Path,
        *,
        mmap: bool = True,
    ) -> None:
        path = Path(checkpoint).expanduser()
        if path.is_dir():
            path = path / "model.pt"
        if not path.is_file():
            raise FileNotFoundError(f"WhatMoves checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", mmap=mmap, weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if not isinstance(state, dict):
            raise TypeError("WhatMoves checkpoint must contain a state dictionary")
        state = {
            key.removeprefix("_orig_mod."): value
            for key, value in state.items()
            if key.removeprefix("_orig_mod.").startswith(self._CHECKPOINT_PREFIXES)
        }
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "WhatMoves checkpoint does not match the released encoder: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )

    def train(self, mode: bool = True) -> "WhatMoves":
        super().train(False)
        return self

    def _autocast(self):
        if self.device.type == "cuda" and self._compute_dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            return torch.autocast("cuda", dtype=self._compute_dtype)
        return nullcontext()

    @staticmethod
    def _validate_image(image: Tensor, name: str) -> None:
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError(f"{name} must be [B,H,W,3]")
        if not image.is_floating_point():
            raise TypeError(f"{name} must be floating point in [-1, 1]")

    @staticmethod
    def _validate_masks(masks: Tensor, batch: int, name: str) -> None:
        if masks.ndim != 4 or masks.shape[0] != batch:
            raise ValueError(f"{name} must be [B,K,H,W]")

    @classmethod
    def _resize_image(cls, image: Tensor) -> Tensor:
        if image.shape[1:3] == (cls.image_size, cls.image_size):
            return image
        return F.interpolate(
            image.movedim(-1, 1),
            size=(cls.image_size, cls.image_size),
            mode="bilinear",
            antialias=True,
        ).movedim(1, -1)

    @classmethod
    def _resize_masks(cls, masks: Tensor) -> Tensor:
        if masks.shape[-2:] == (cls.image_size, cls.image_size):
            return masks.bool()
        return (
            F.interpolate(
                masks.float(),
                size=(cls.image_size, cls.image_size),
                mode="bilinear",
                antialias=True,
            )
            .round()
            .bool()
        )

    @classmethod
    def _resize_temporal_masks(cls, masks: Tensor) -> Tensor:
        batch, frames, regions, height, width = masks.shape
        if (height, width) == (cls.image_size, cls.image_size):
            return masks.bool()
        resized = F.interpolate(
            masks.reshape(batch * frames, regions, height, width).float(),
            size=(cls.image_size, cls.image_size),
            mode="bilinear",
            antialias=True,
        )
        return (
            resized.round()
            .bool()
            .reshape(
                batch,
                frames,
                regions,
                cls.image_size,
                cls.image_size,
            )
        )

    def _encode_frames(self, frames: Tensor) -> Tensor:
        chunks = []
        for start in range(0, frames.shape[1], self.frame_chunk_size):
            chunk = frames[:, start : start + self.frame_chunk_size]
            batch, count, height, width, channels = chunk.shape
            features = self.frame_encoder(
                chunk.reshape(batch * count, height, width, channels)
            )
            chunks.append(features.reshape(batch, count, *features.shape[1:]))
        return torch.cat(chunks, dim=1)

    def _encode_content(self, image: Tensor, masks: Tensor) -> Tensor:
        valid = masks.flatten(-2).any(dim=-1)
        frames = image[:, None].expand(-1, masks.shape[1], -1, -1, -1)
        features = self._encode_frames(frames)
        return self.content_encoder(features, masks, valid)

    def _encode_temporal_content(
        self,
        frame_features: Tensor,
        masks: Tensor,
        starts: list[int],
    ) -> Tensor:
        batch, _, regions = masks.shape[:3]
        valid = masks.flatten(-2).any(dim=-1)
        selected_frames, available = _temporal_mask_frame_plan(
            valid,
            starts,
            window_length=self.window_length,
        )
        candidates = []
        for first in range(0, len(starts), self.window_chunk_size):
            last = min(first + self.window_chunk_size, len(starts))
            window_count = last - first
            frame_indices = selected_frames[:, first:last]
            batch_indices = torch.arange(batch, device=masks.device)[:, None, None]
            region_indices = torch.arange(regions, device=masks.device)[None, None]
            selected_features = frame_features[batch_indices, frame_indices]
            selected_masks = masks[
                batch_indices,
                frame_indices,
                region_indices,
            ]
            candidate = self.content_encoder(
                selected_features.flatten(0, 1),
                selected_masks.flatten(0, 1),
                available[:, first:last].flatten(0, 1),
            )
            candidates.append(
                candidate.reshape(
                    batch,
                    window_count,
                    regions,
                    self.content_dim,
                )
            )
        return _reuse_previous_content(torch.cat(candidates, dim=1), available)

    @torch.inference_mode()
    def encode_content(self, image: Tensor, masks: Tensor) -> Tensor:
        """Encode ``image [B,H,W,3]`` and ``masks [B,K,H,W]``."""
        self._validate_image(image, "image")
        self._validate_masks(masks, image.shape[0], "masks")
        if image.shape[1:3] != masks.shape[-2:]:
            raise ValueError("image and masks must share their spatial shape")
        image = self._resize_image(image.to(self.device, dtype=torch.float32))
        masks = self._resize_masks(masks.to(self.device))
        with self._autocast():
            return self._encode_content(image, masks)

    @torch.inference_mode()
    def encode_motion(
        self,
        video: Tensor,
        masks: Tensor | None = None,
        *,
        content_image: Tensor | None = None,
        return_content: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Encode localized motion from ``video [B,T,H,W,3]``.

        Static ``masks [B,K,H,W]`` identify regions in ``content_image`` or,
        when no content image is supplied, in the first video frame. Temporal
        ``masks [B,T,K,H,W]`` update the content query for every extraction
        window. The first valid mask at or after the window start is used; a
        window without a valid mask reuses the previous content embedding.
        Every region must have a valid mask in the first extraction window.
        When ``masks`` is ``None``, the learned global-motion stream is returned.
        """
        if video.ndim != 5 or video.shape[-1] != 3:
            raise ValueError("video must be [B,T,H,W,3]")
        if not video.is_floating_point():
            raise TypeError("video must be floating point in [-1, 1]")
        temporal_masks = masks is not None and masks.ndim == 5
        if temporal_masks:
            if masks.shape[:2] != video.shape[:2]:
                raise ValueError(
                    "temporal masks must be [B,T,K,H,W] with the same B,T as video"
                )
            if content_image is not None:
                raise ValueError("content_image cannot be used with temporal masks")
            if video.shape[-3:-1] != masks.shape[-2:]:
                raise ValueError("video and masks must share their spatial shape")
        elif masks is not None:
            self._validate_masks(masks, video.shape[0], "masks")
            if video.shape[-3:-1] != masks.shape[-2:]:
                raise ValueError("video and masks must share their spatial shape")
        elif content_image is not None:
            raise ValueError("content_image is only meaningful with region masks")
        if video.shape[1] < self.window_length:
            raise ValueError(
                f"WhatMoves needs at least {self.window_length} video frames"
            )
        if masks is not None and not temporal_masks and content_image is None:
            content_image = video[:, 0]
        elif masks is not None and not temporal_masks:
            self._validate_image(content_image, "content_image")
            if content_image.shape[0] != video.shape[0]:
                raise ValueError("video and content_image batches must match")
            if content_image.shape[1:3] != masks.shape[-2:]:
                raise ValueError(
                    "content_image and masks must share their spatial shape"
                )

        video = video.to(self.device, dtype=torch.float32)
        batch, frames, height, width, channels = video.shape
        if (height, width) != (self.image_size, self.image_size):
            video = (
                F.interpolate(
                    video.reshape(batch * frames, height, width, channels).movedim(
                        -1, 1
                    ),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    antialias=True,
                )
                .movedim(1, -1)
                .reshape(
                    batch,
                    frames,
                    self.image_size,
                    self.image_size,
                    channels,
                )
            )
        if temporal_masks:
            masks = self._resize_temporal_masks(masks.to(self.device))
            valid = None
            regions = masks.shape[2]
        elif masks is not None:
            content_image = self._resize_image(
                content_image.to(self.device, dtype=torch.float32)
            )
            masks = self._resize_masks(masks.to(self.device))
            valid = masks.flatten(-2).any(dim=-1)
            regions = masks.shape[1]
        else:
            valid = torch.empty(batch, 0, device=self.device, dtype=torch.bool)
            regions = 0

        with self._autocast():
            frame_features = self._encode_frames(video)
            starts, gather = _sliding_window_plan(
                frames,
                window_length=self.window_length,
                stride=self.window_stride,
                emitted_steps=self.emitted_steps,
            )
            if temporal_masks:
                content = self._encode_temporal_content(
                    frame_features,
                    masks,
                    starts,
                )
            elif masks is not None:
                content = self._encode_content(content_image, masks)
            else:
                content = frame_features.new_empty((batch, 0, self.content_dim))
            outputs = []
            for first in range(0, len(starts), self.window_chunk_size):
                chunk_starts = starts[first : first + self.window_chunk_size]
                windows = torch.stack(
                    [
                        frame_features[:, start : start + self.window_length]
                        for start in chunk_starts
                    ],
                    dim=1,
                )
                window_count = windows.shape[1]
                windows = windows.flatten(0, 1)
                if temporal_masks:
                    local_content = content[:, first : first + window_count].flatten(
                        0, 1
                    )
                    local_valid = torch.ones(
                        local_content.shape[:2],
                        device=local_content.device,
                        dtype=torch.bool,
                    )
                else:
                    local_content = (
                        content[:, None].expand(-1, window_count, -1, -1).flatten(0, 1)
                    )
                    local_valid = (
                        valid[:, None].expand(-1, window_count, -1).flatten(0, 1)
                    )
                global_content = (
                    self.global_motion_query.to(
                        device=local_content.device,
                        dtype=local_content.dtype,
                    )
                    .view(1, 1, -1)
                    .expand(local_content.shape[0], 1, -1)
                )
                conditioned_content = torch.cat((local_content, global_content), dim=1)
                conditioned_valid = torch.cat(
                    (
                        local_valid,
                        torch.ones(
                            local_valid.shape[0],
                            1,
                            device=local_valid.device,
                            dtype=torch.bool,
                        ),
                    ),
                    dim=1,
                )
                motion = self.motion_encoder(
                    windows,
                    conditioned_content,
                    conditioned_valid,
                    query_steps=self.emitted_steps,
                )
                motion = motion[:, :, :regions] if regions else motion[:, :, -1:]
                outputs.append(
                    motion.reshape(
                        batch,
                        window_count,
                        self.emitted_steps,
                        max(regions, 1),
                        self.motion_dim,
                    )
                )
            flattened = torch.cat(outputs, dim=1).flatten(1, 2)
            motion = flattened.index_select(1, gather.to(flattened.device))
            if not regions:
                motion = motion[:, :, 0]
            elif temporal_masks and return_content:
                aligned_content = content[:, :, None].expand(
                    -1,
                    -1,
                    self.emitted_steps,
                    -1,
                    -1,
                )
                content = aligned_content.flatten(1, 2).index_select(
                    1,
                    gather.to(aligned_content.device),
                )
        return (motion, content) if return_content else motion

    def forward(self, video: Tensor, masks: Tensor | None = None) -> Tensor:
        return self.encode_motion(video, masks)


def load_model(
    checkpoint: str | Path,
    **kwargs,
) -> WhatMoves:
    """Convenience alias for :meth:`WhatMoves.from_pretrained`."""
    return WhatMoves.from_pretrained(checkpoint, **kwargs)
