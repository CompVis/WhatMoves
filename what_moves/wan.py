"""Minimal inference-only WhatMoves adapter for Wan2.2 I2V-A14B.

The adapter appends projected WhatMoves tokens to Wan's video-token sequence.
The frozen base model can also be evaluated directly: a base branch disables
both LoRA and motion-token concatenation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import reduce, wraps
import importlib
import logging
import math
from pathlib import Path
import threading
from types import MethodType
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import Tensor, nn

from .model import WhatMoves

logger = logging.getLogger(__name__)


def _serialized_inference(function):
    """Reject concurrent sampling on the stateful Wan runtime."""

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        if not self._inference_lock.acquire(blocking=False):
            raise RuntimeError(
                "WanMotionTransfer.sample() is not reentrant; wait for the active "
                "sample to finish before starting another one"
            )
        try:
            return function(self, *args, **kwargs)
        finally:
            self._inference_lock.release()

    return wrapped


def _wan_module(name: str):
    """Import a module from the bundled top-level Wan runtime."""
    return importlib.import_module(f"wan.{name}")


WAN_GUIDANCE_BASE_CFG = "base_cfg"
WAN_GUIDANCE_TEXT_CFG = "text_cfg"
WAN_GUIDANCE_JOINT_CFG = "joint_cfg"
WAN_GUIDANCE_MOTION_CFG = "motion_cfg"
WAN_GUIDANCE_ADDITIVE_CFG = "additive_cfg"
WAN_GUIDANCE_FACTORIZED_CFG = "factorized_cfg"

WAN_GUIDANCE_MODES = (
    WAN_GUIDANCE_BASE_CFG,
    WAN_GUIDANCE_TEXT_CFG,
    WAN_GUIDANCE_JOINT_CFG,
    WAN_GUIDANCE_MOTION_CFG,
    WAN_GUIDANCE_ADDITIVE_CFG,
    WAN_GUIDANCE_FACTORIZED_CFG,
)

_GUIDANCE_BRANCHES = {
    WAN_GUIDANCE_BASE_CFG: ("base_neg", "base_pos"),
    WAN_GUIDANCE_TEXT_CFG: ("adapter_pos", "adapter_neg"),
    WAN_GUIDANCE_JOINT_CFG: ("base_neg", "adapter_pos"),
    WAN_GUIDANCE_MOTION_CFG: ("base_pos", "adapter_pos"),
    WAN_GUIDANCE_ADDITIVE_CFG: ("base_neg", "base_pos", "adapter_pos"),
    WAN_GUIDANCE_FACTORIZED_CFG: (
        "base_neg",
        "base_pos",
        "adapter_pos",
        "adapter_neg",
    ),
}


@dataclass(frozen=True)
class _WanI2VConfig:
    high_noise_checkpoint: str = "high_noise_model"
    low_noise_checkpoint: str = "low_noise_model"
    vae_checkpoint: str = "Wan2.1_VAE.pth"
    t5_checkpoint: str = "models_t5_umt5-xxl-enc-bf16.pth"
    t5_tokenizer: str = "google/umt5-xxl"
    vae_stride: tuple[int, int, int] = (4, 8, 8)
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_length: int = 512
    train_timesteps: int = 1000
    default_steps: int = 40
    default_text_guidance: float = 3.5
    default_flow_shift: float = 5.0
    default_expert_boundary: float = 0.9
    negative_prompt: str = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
        "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
        "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )


def normalize_wan_guidance_mode(mode: str | None) -> str:
    normalized = str(mode or WAN_GUIDANCE_TEXT_CFG).strip().lower()
    if normalized not in _GUIDANCE_BRANCHES:
        raise ValueError(
            f"Unknown Wan guidance mode {mode!r}; expected one of: "
            + ", ".join(WAN_GUIDANCE_MODES)
        )
    return normalized


def wan_guidance_required_branches(mode: str | None) -> tuple[str, ...]:
    return _GUIDANCE_BRANCHES[normalize_wan_guidance_mode(mode)]


def _cfg(negative, positive, scale: float):
    if scale == 0.0:
        return negative
    if scale == 1.0:
        return positive
    return negative + scale * (positive - negative)


def combine_wan_guidance_predictions(
    mode: str | None,
    predictions: dict[str, object],
    *,
    text_guidance_scale: float,
    motion_guidance_scale: float,
):
    """Combine the denoising branches for one configured guidance mode."""
    mode = normalize_wan_guidance_mode(mode)
    missing = [
        name for name in wan_guidance_required_branches(mode) if name not in predictions
    ]
    if missing:
        raise ValueError(f"Missing Wan guidance prediction branches: {missing}")

    text_scale = float(text_guidance_scale)
    motion_scale = float(motion_guidance_scale)
    if mode == WAN_GUIDANCE_BASE_CFG:
        return _cfg(predictions["base_neg"], predictions["base_pos"], text_scale)
    if mode == WAN_GUIDANCE_TEXT_CFG:
        return _cfg(
            predictions["adapter_neg"],
            predictions["adapter_pos"],
            text_scale,
        )
    if mode == WAN_GUIDANCE_JOINT_CFG:
        return _cfg(
            predictions["base_neg"],
            predictions["adapter_pos"],
            text_scale,
        )
    if mode == WAN_GUIDANCE_MOTION_CFG:
        return _cfg(
            predictions["base_pos"],
            predictions["adapter_pos"],
            motion_scale,
        )
    if mode == WAN_GUIDANCE_ADDITIVE_CFG:
        base = _cfg(predictions["base_neg"], predictions["base_pos"], text_scale)
        return base + motion_scale * (
            predictions["adapter_pos"] - predictions["base_pos"]
        )

    base = _cfg(predictions["base_neg"], predictions["base_pos"], text_scale)
    adapted = _cfg(
        predictions["adapter_neg"],
        predictions["adapter_pos"],
        text_scale,
    )
    return base + motion_scale * (adapted - base)


@contextmanager
def _lora_disabled(model: nn.Module):
    layers = [
        module for module in model.modules() if isinstance(module, BaseTunerLayer)
    ]
    if not layers:
        raise RuntimeError("The Wan expert has no injected LoRA layers")
    enabled = [not layer.disable_adapters for layer in layers]
    parameter_states = {
        parameter: parameter.requires_grad
        for layer in layers
        for adapter_layer_name in layer.adapter_layer_names
        for parameter in getattr(layer, adapter_layer_name).parameters()
    }
    try:
        for layer in layers:
            layer.enable_adapters(False)
        yield
    finally:
        for layer, was_enabled in zip(layers, enabled):
            layer.enable_adapters(was_enabled)
        for parameter, requires_grad in parameter_states.items():
            parameter.requires_grad_(requires_grad)


@contextmanager
def _lora_scaled(model: nn.Module, scale: float):
    scale = float(scale)
    if not math.isfinite(scale):
        raise ValueError(f"Wan LoRA scale must be finite, got {scale}")
    if scale == 1.0:
        yield
        return

    states = {
        (layer, adapter): layer.scaling[adapter]
        for layer in model.modules()
        if isinstance(layer, BaseTunerLayer)
        for adapter in layer.active_adapters
        if adapter in layer.scaling
    }
    if not states:
        raise RuntimeError("Wan has no active LoRA adapters to scale")
    try:
        for (layer, adapter), original in states.items():
            layer.scaling[adapter] = original * scale
        yield
    finally:
        for (layer, adapter), original in states.items():
            layer.scaling[adapter] = original


def _resolve_adapter_checkpoint(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Wan adapter checkpoint not found: {path}")
    candidates = sorted(path.glob("*.pt"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No adapter checkpoint found under: {path}")
    raise ValueError(
        f"Multiple Wan adapter checkpoints found under {path}; select one explicitly"
    )


def _load_adapter_payload(path: Path) -> tuple[dict[str, Tensor], dict[str, object]]:
    payload = torch.load(
        path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise TypeError("Wan adapter checkpoint must contain a dictionary")
    state = payload.get("adapter", payload)
    if not isinstance(state, dict) or not all(
        isinstance(name, str) and isinstance(value, Tensor)
        for name, value in state.items()
    ):
        raise TypeError("Wan adapter state must be a string-to-tensor dictionary")
    metadata = {name: value for name, value in payload.items() if name != "adapter"}
    return state, metadata


def _adapter_hyperparameters(
    state: dict[str, Tensor],
    metadata: dict[str, object],
) -> tuple[int, float]:
    inferred_rank = next(
        (
            value.shape[0]
            for name, value in state.items()
            if ".lora_A." in name and value.ndim == 2
        ),
        None,
    )
    rank = int(metadata.get("lora_rank", inferred_rank or 0))
    alpha = float(metadata.get("lora_alpha", 0))
    if rank < 1 or alpha <= 0 or not math.isfinite(alpha):
        raise ValueError(
            "Wan adapter checkpoint must provide valid lora_rank and lora_alpha"
        )
    if inferred_rank is not None and rank != inferred_rank:
        raise ValueError(
            f"Wan adapter metadata rank {rank} does not match tensor rank "
            f"{inferred_rank}"
        )
    return rank, alpha


def _rms_norm(x: Tensor, scale: Tensor, eps: float) -> Tensor:
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    multiplier = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * multiplier.to(x.dtype)


class _RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return _rms_norm(x, self.scale, self.eps)


class _LinearSwiGLU(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features * 2, bias=False)
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        x, gate = F.linear(x, self.weight).chunk(2, dim=-1)
        return x * F.silu(gate)


class _FeedForwardBlock(nn.Module):
    def __init__(self, width: int, hidden_width: int):
        super().__init__()
        self.norm = _RMSNorm(width)
        self.up_proj = _LinearSwiGLU(width, hidden_width)
        self.down_proj = nn.Linear(hidden_width, width, bias=False)
        nn.init.zeros_(self.down_proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.down_proj(self.up_proj(self.norm(x)))


class _MappingNetwork(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.in_norm = _RMSNorm(width)
        self.blocks = nn.ModuleList([_FeedForwardBlock(width, 3 * width)])
        self.out_norm = _RMSNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)


def _apply_motion_rope(x: Tensor, positions: Tensor, freqs: Tensor) -> Tensor:
    head_dim = x.shape[-1]
    complex_dim = head_dim // 2
    frequency_groups = freqs.split(
        [complex_dim - 2 * (complex_dim // 3), complex_dim // 3, complex_dim // 3],
        dim=1,
    )
    positions = positions.round().long()
    indices = [
        positions[..., axis].clamp_(0, group.shape[0] - 1)
        for axis, group in enumerate(frequency_groups)
    ]
    token_freqs = torch.cat(
        [group[index] for group, index in zip(frequency_groups, indices)],
        dim=-1,
    ).unsqueeze(2)
    complex_tokens = torch.view_as_complex(
        x.to(torch.float64).reshape(*x.shape[:-1], -1, 2)
    )
    return torch.view_as_real(complex_tokens * token_freqs).flatten(-2).float()


def _motion_aware_self_attention(
    self,
    x: Tensor,
    seq_lens: Tensor,
    grid_sizes: Tensor,
    freqs: Tensor,
) -> Tensor:
    attention = _wan_module("modules.attention").attention
    rope_apply = _wan_module("modules.model").rope_apply

    batch, sequence = x.shape[:2]
    query = self.norm_q(self.q(x)).view(batch, sequence, self.num_heads, self.head_dim)
    key = self.norm_k(self.k(x)).view(batch, sequence, self.num_heads, self.head_dim)
    value = self.v(x).view(batch, sequence, self.num_heads, self.head_dim)

    positions = self.motion_data["positions"]
    motion_length = positions.shape[1]
    video_length = sequence - motion_length
    query_video = rope_apply(query[:, :video_length], grid_sizes, freqs)
    key_video = rope_apply(key[:, :video_length], grid_sizes, freqs)
    if motion_length:
        query = torch.cat(
            (
                query_video,
                _apply_motion_rope(query[:, video_length:], positions, freqs),
            ),
            dim=1,
        )
        key = torch.cat(
            (key_video, _apply_motion_rope(key[:, video_length:], positions, freqs)),
            dim=1,
        )
    else:
        query, key = query_video, key_video

    hidden = attention(
        q=query,
        k=key,
        v=value,
        k_lens=seq_lens,
        window_size=self.window_size,
    )
    return self.o(hidden.flatten(2))


def _motion_positions(
    masks: Tensor,
    motion_frames: int,
    latent_grid: tuple[int, int, int],
) -> Tensor:
    latent_t, latent_h, latent_w = latent_grid
    batch, regions, height, width = masks.shape
    temporal = torch.linspace(
        0,
        max(latent_t - 1, 0),
        motion_frames,
        device=masks.device,
        dtype=torch.float32,
    )
    positions = []
    for batch_index in range(batch):
        sample = []
        for region_index in range(regions):
            coordinates = masks[batch_index, region_index].nonzero()
            if not coordinates.numel():
                raise ValueError(
                    f"Target mask {region_index} in batch {batch_index} is empty"
                )
            center = coordinates.float().mean(dim=0)
            y = center[0] / max(height - 1, 1) * max(latent_h - 1, 0)
            x = center[1] / max(width - 1, 1) * max(latent_w - 1, 0)
            sample.append(
                torch.stack(
                    (temporal, y.expand_as(temporal), x.expand_as(temporal)), dim=-1
                )
            )
        positions.append(torch.cat(sample, dim=0))
    return torch.stack(positions)


def _as_tensor_list(value, name: str) -> list[Tensor]:
    if isinstance(value, Tensor):
        return [value]
    if (
        isinstance(value, Sequence)
        and value
        and all(isinstance(item, Tensor) for item in value)
    ):
        return list(value)
    raise TypeError(f"{name} must be a tensor or a nonempty sequence of tensors")


def _prompt_batch(value: str | Sequence[str], batch: int, name: str) -> list[str]:
    if isinstance(value, str):
        return [value] * batch
    prompts = list(value)
    if len(prompts) != batch or not all(isinstance(item, str) for item in prompts):
        raise ValueError(f"{name} must contain one string per batch element")
    return prompts


class WanMotionTransfer(nn.Module):
    """Inference-only WhatMoves adapter for Wan2.2 I2V-A14B."""

    config = _WanI2VConfig()

    def __init__(
        self,
        wan_checkpoint: str | Path,
        what_moves_checkpoint: str | Path,
        adapter_checkpoint: str | Path,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        flow_shift: float | None = None,
        expert_boundary: float | None = None,
        frame_chunk_size: int = 32,
        window_chunk_size: int = 8,
        window_stride: int = 1,
    ):
        super().__init__()
        self._device_locked = False
        self._inference_lock = threading.Lock()
        self.device_hint = torch.device(device)
        if self.device_hint.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Wan I2V-A14B inference requires a CUDA GPU")
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Wan dtype must be torch.bfloat16 or torch.float16")
        self.weight_dtype = dtype
        self.flow_shift = float(
            self.config.default_flow_shift if flow_shift is None else flow_shift
        )
        self.expert_boundary = float(
            self.config.default_expert_boundary
            if expert_boundary is None
            else expert_boundary
        )

        self.wan_checkpoint = Path(wan_checkpoint).expanduser().resolve()
        self.adapter_checkpoint = _resolve_adapter_checkpoint(adapter_checkpoint)
        self._validate_wan_checkpoint()
        adapter_state, adapter_metadata = _load_adapter_payload(self.adapter_checkpoint)
        self.lora_rank, self.lora_alpha = _adapter_hyperparameters(
            adapter_state,
            adapter_metadata,
        )

        WanModel = _wan_module("modules.model").WanModel
        T5EncoderModel = _wan_module("modules.t5").T5EncoderModel
        Wan2_1_VAE = _wan_module("modules.vae2_1").Wan2_1_VAE

        logger.info("Loading WhatMoves from %s", what_moves_checkpoint)
        self.what_moves = WhatMoves.from_pretrained(
            what_moves_checkpoint,
            device=self.device_hint,
            dtype=self.weight_dtype,
            frame_chunk_size=frame_chunk_size,
            window_chunk_size=window_chunk_size,
            window_stride=window_stride,
        )
        logger.info("Loading Wan high-noise expert")
        self.high_noise_model = WanModel.from_pretrained(
            self.wan_checkpoint,
            subfolder=self.config.high_noise_checkpoint,
            torch_dtype=self.weight_dtype,
        ).to(self.device_hint)
        logger.info("Loading Wan low-noise expert")
        self.low_noise_model = WanModel.from_pretrained(
            self.wan_checkpoint,
            subfolder=self.config.low_noise_checkpoint,
            torch_dtype=self.weight_dtype,
        ).to(self.device_hint)
        for name, model in self.experts.items():
            if model.model_type != "i2v":
                raise ValueError(
                    f"Wan {name} expert has model_type={model.model_type!r}"
                )

        self.text_encoder = T5EncoderModel(
            text_len=self.config.text_length,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            checkpoint_path=str(self.wan_checkpoint / self.config.t5_checkpoint),
            tokenizer_path=str(self.wan_checkpoint / self.config.t5_tokenizer),
        )
        self.vae = Wan2_1_VAE(
            vae_pth=str(self.wan_checkpoint / self.config.vae_checkpoint),
            dtype=self.weight_dtype,
            device=self.device_hint,
        )

        mapper_width = self.what_moves.motion_dim + self.what_moves.content_dim
        self.motion_norm = nn.LayerNorm(self.what_moves.motion_dim)
        self.content_norm = nn.LayerNorm(self.what_moves.content_dim)
        self.motion_proj = nn.Sequential(
            _MappingNetwork(mapper_width),
            nn.Linear(mapper_width, self.high_noise_model.dim),
        )
        self._inject_lora()
        super().to(self.device_hint)
        self._adapter_parameter_names = {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        self._load_adapter_state(adapter_state)
        del adapter_state
        self.eval().requires_grad_(False)
        self._device_locked = True

    def _apply(self, function, recurse: bool = True):
        if self._device_locked:
            raise RuntimeError(
                "WanMotionTransfer is device-bound. Select device and dtype when "
                "loading it instead of calling .to(), .cuda(), .cpu(), or dtype "
                "conversion methods afterwards."
            )
        return super()._apply(function, recurse=recurse)

    @property
    def device(self) -> torch.device:
        return self.high_noise_model.patch_embedding.weight.device

    @property
    def experts(self) -> dict[str, nn.Module]:
        return {
            "high_noise": self.high_noise_model,
            "low_noise": self.low_noise_model,
        }

    def _validate_wan_checkpoint(self) -> None:
        required = (
            self.wan_checkpoint / self.config.high_noise_checkpoint,
            self.wan_checkpoint / self.config.low_noise_checkpoint,
            self.wan_checkpoint / self.config.vae_checkpoint,
            self.wan_checkpoint / self.config.t5_checkpoint,
            self.wan_checkpoint / self.config.t5_tokenizer,
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Incomplete Wan I2V-A14B checkpoint: {missing}")

    def _inject_lora(self) -> None:
        self.what_moves.eval().requires_grad_(False)
        self.text_encoder.model.eval().requires_grad_(False)
        self.vae.model.eval().requires_grad_(False)
        target_modules = (
            r".*blocks\.\d+\.(?:self_attn|cross_attn)\.(?:q|k|v|o)$"
            r"|.*blocks\.\d+\.ffn\.(?:0|2)$"
        )
        config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.0,
            init_lora_weights=True,
            target_modules=target_modules,
        )
        self.motion_data: dict[str, Tensor] = {
            "positions": torch.empty(1, 0, 3, device=self.device_hint)
        }
        for name, model in self.experts.items():
            model.eval().requires_grad_(False)
            inject_adapter_in_model(config, model, adapter_name=name)
            for block in model.blocks:
                block.self_attn.motion_data = self.motion_data
                block.self_attn.forward = MethodType(
                    _motion_aware_self_attention,
                    block.self_attn,
                )
        self.motion_norm.requires_grad_(True)
        self.content_norm.requires_grad_(True)
        self.motion_proj.requires_grad_(True)

    def _load_adapter_state(self, state: dict[str, Tensor]) -> None:
        actual = set(state)
        missing = sorted(self._adapter_parameter_names - actual)
        unexpected = sorted(actual - self._adapter_parameter_names)
        if missing or unexpected:
            raise RuntimeError(
                "Wan adapter does not match the released model: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        self.load_state_dict(state, strict=False)

    def load_adapter(self, checkpoint: str | Path) -> "WanMotionTransfer":
        """Load another adapter with the same LoRA architecture."""
        path = _resolve_adapter_checkpoint(checkpoint)
        state, metadata = _load_adapter_payload(path)
        rank, alpha = _adapter_hyperparameters(state, metadata)
        if (rank, alpha) != (self.lora_rank, self.lora_alpha):
            raise ValueError(
                "Adapter LoRA architecture differs from the initialized model: "
                f"expected rank={self.lora_rank}, alpha={self.lora_alpha}; "
                f"got rank={rank}, alpha={alpha}"
            )
        self._load_adapter_state(state)
        self.adapter_checkpoint = path
        return self

    def train(self, mode: bool = True) -> "WanMotionTransfer":
        super().train(False)
        self.text_encoder.model.eval()
        self.vae.model.eval()
        return self

    @torch.no_grad()
    def _encode_text(self, prompts: list[str]) -> list[Tensor]:
        self.text_encoder.model.to(self.device)
        return self.text_encoder(prompts, self.device)

    @torch.no_grad()
    def _image_condition(
        self,
        image: Tensor,
        frame_count: int,
        latent_shape: tuple[int, int, int],
    ) -> Tensor:
        batch, height, width, _ = image.shape
        latent_t, latent_h, latent_w = latent_shape
        video = torch.zeros(
            batch,
            3,
            frame_count,
            height,
            width,
            device=self.device,
            dtype=self.weight_dtype,
        )
        video[:, :, 0] = image.permute(0, 3, 1, 2)
        image_latent = torch.stack(self.vae.encode(list(video)))
        mask = torch.zeros(
            batch,
            4,
            latent_t,
            latent_h,
            latent_w,
            device=self.device,
        )
        mask[:, :, 0] = 1
        return torch.cat((mask.to(image_latent.dtype), image_latent), dim=1)

    def _encode_motion_tokens(
        self,
        source_video: Tensor,
        source_masks: Tensor,
        source_content_image: Tensor | None,
        target_image: Tensor,
        target_masks: Tensor,
        latent_grid: tuple[int, int, int],
    ) -> tuple[Tensor, Tensor]:
        motion = self.what_moves.encode_motion(
            source_video,
            source_masks,
            content_image=source_content_image,
        )
        if motion.ndim != 4:
            raise ValueError(f"WhatMoves returned invalid shape {tuple(motion.shape)}")
        motion = self.motion_norm(motion.permute(0, 2, 1, 3))
        content = self.what_moves.encode_content(target_image, target_masks)
        if content.shape[:2] != motion.shape[:2]:
            raise ValueError(
                f"Target content {tuple(content.shape)} does not match motion "
                f"{tuple(motion.shape)}"
            )
        content = self.content_norm(content)[:, :, None].expand(
            -1, -1, motion.shape[2], -1
        )
        tokens = self.motion_proj(torch.cat((motion, content), dim=-1))
        positions = _motion_positions(target_masks, motion.shape[2], latent_grid)
        return tokens.flatten(1, 2), positions

    def _expert(self, timestep: Tensor) -> nn.Module:
        boundary = self.expert_boundary * self.config.train_timesteps
        if bool((timestep >= boundary).all()):
            return self.high_noise_model
        if bool((timestep < boundary).all()):
            return self.low_noise_model
        raise ValueError("A batch cannot mix high- and low-noise Wan experts")

    def _transformer_forward(
        self,
        model: nn.Module,
        latent: Tensor,
        image_condition: Tensor,
        timesteps: Tensor,
        context: list[Tensor],
        motion_tokens: Tensor,
        motion_positions: Tensor,
    ) -> Tensor:
        sinusoidal_embedding_1d = _wan_module("modules.model").sinusoidal_embedding_1d

        device = model.patch_embedding.weight.device
        if model.freqs.device != device:
            model.freqs = model.freqs.to(device)
        embedded = [
            model.patch_embedding(torch.cat((x, y), dim=0).unsqueeze(0))
            for x, y in zip(latent, image_condition)
        ]
        grid_sizes = torch.stack(
            [torch.tensor(item.shape[2:], device=device) for item in embedded]
        )
        video_tokens = [item.flatten(2).transpose(1, 2) for item in embedded]
        video_lengths = torch.tensor(
            [item.shape[1] for item in video_tokens],
            device=device,
        )
        if not bool((video_lengths == video_lengths[0]).all()):
            raise ValueError("Wan batches must have one fixed latent shape")
        video_length = int(video_lengths[0])
        hidden = torch.cat(video_tokens, dim=0)
        hidden = torch.cat((hidden, motion_tokens.to(hidden.dtype)), dim=1)
        joint_lengths = video_lengths + motion_tokens.shape[1]
        self.motion_data["positions"] = motion_positions

        joint_t = timesteps[:, None].expand(-1, hidden.shape[1])
        with torch.amp.autocast("cuda", dtype=torch.float32):
            batch = hidden.shape[0]
            time = sinusoidal_embedding_1d(
                model.freq_dim,
                joint_t.flatten(),
            ).unflatten(0, (batch, joint_t.shape[1]))
            time = model.time_embedding(time.float())
            modulation = model.time_projection(time).unflatten(2, (6, model.dim))
        text = model.text_embedding(
            torch.stack(
                [
                    torch.cat(
                        (
                            item,
                            item.new_zeros(
                                model.text_len - item.shape[0],
                                item.shape[1],
                            ),
                        )
                    )
                    for item in context
                ]
            )
        )
        kwargs = {
            "e": modulation,
            "seq_lens": joint_lengths,
            "grid_sizes": grid_sizes,
            "freqs": model.freqs,
            "context": text,
            "context_lens": None,
        }
        for block in model.blocks:
            hidden = block(hidden, **kwargs)
        output = model.head(hidden[:, :video_length], time[:, :video_length])
        return torch.stack(model.unpatchify(output, grid_sizes))

    def _prepare_sources(
        self,
        source_videos,
        source_masks,
        target_masks,
        source_content_images,
        *,
        target_image: Tensor,
    ) -> list[tuple[Tensor, Tensor, Tensor, Tensor | None]]:
        videos = _as_tensor_list(source_videos, "source_videos")
        masks = _as_tensor_list(source_masks, "source_masks")
        targets = _as_tensor_list(target_masks, "target_masks")
        references = (
            [None] * len(videos)
            if source_content_images is None
            else _as_tensor_list(source_content_images, "source_content_images")
        )
        if not (len(videos) == len(masks) == len(targets) == len(references)):
            raise ValueError(
                "source videos, masks, target masks, and content images must have "
                "equal length"
            )
        result = []
        batch = target_image.shape[0]
        video_cache: dict[int, Tensor] = {}
        for index, (video, source, target, reference) in enumerate(
            zip(videos, masks, targets, references)
        ):
            if video.ndim != 5 or video.shape[-1] != 3 or video.shape[0] != batch:
                raise ValueError(f"source_videos[{index}] must be [B,T,H,W,3]")
            if not video.is_floating_point():
                raise TypeError(f"source_videos[{index}] must be floating point")
            if video.shape[1] < self.what_moves.window_length:
                raise ValueError(
                    f"source_videos[{index}] has {video.shape[1]} frames, "
                    f"but {self.what_moves.window_length} are required"
                )
            if source.ndim == 4:
                if source.shape[0] != batch or source.shape[-2:] != video.shape[-3:-1]:
                    raise ValueError(
                        f"source_masks[{index}] must be [B,K,H,W] aligned to its video"
                    )
                regions = source.shape[1]
                if not bool(source.flatten(-2).any(-1).all()):
                    raise ValueError(f"source_masks[{index}] contains an empty region")
                if reference is not None:
                    if (
                        reference.ndim != 4
                        or reference.shape != (batch, *video.shape[-3:])
                        or not reference.is_floating_point()
                    ):
                        raise ValueError(
                            f"source_content_images[{index}] must be floating-point "
                            "[B,H,W,3] aligned to its source masks"
                        )
            elif source.ndim == 5:
                if (
                    source.shape[:2] != video.shape[:2]
                    or source.shape[-2:] != video.shape[-3:-1]
                ):
                    raise ValueError(
                        f"source_masks[{index}] must be [B,T,K,H,W] aligned to its video"
                    )
                regions = source.shape[2]
                if reference is not None:
                    raise ValueError(
                        "source_content_images cannot accompany temporal source masks"
                    )
            else:
                raise ValueError(
                    f"source_masks[{index}] must be [B,K,H,W] or [B,T,K,H,W]"
                )
            if regions < 1:
                raise ValueError(f"source_masks[{index}] contains no regions")
            if (
                target.ndim != 4
                or target.shape[:2] != (batch, regions)
                or target.shape[-2:] != target_image.shape[1:3]
            ):
                raise ValueError(
                    f"target_masks[{index}] must be [B,{regions},H,W] aligned "
                    "to target_image"
                )
            cached_video = video_cache.get(id(video))
            if cached_video is None:
                cached_video = video.to(self.device)
                video_cache[id(video)] = cached_video
            result.append(
                (
                    cached_video,
                    source.bool().to(self.device),
                    target.bool().to(self.device),
                    None if reference is None else reference.to(self.device),
                )
            )
        return result

    @torch.inference_mode()
    @_serialized_inference
    def sample(
        self,
        prompt: str | Sequence[str],
        target_image: Tensor,
        *,
        source_videos: Tensor | Sequence[Tensor] | None = None,
        source_masks: Tensor | Sequence[Tensor] | None = None,
        target_masks: Tensor | Sequence[Tensor] | None = None,
        source_content_images: Tensor | Sequence[Tensor] | None = None,
        negative_prompt: str | Sequence[str] | None = None,
        num_frames: int | None = None,
        num_inference_steps: int | None = None,
        text_guidance_scale: float | None = None,
        motion_guidance_scale: float = 1.0,
        guidance_mode: str = WAN_GUIDANCE_TEXT_CFG,
        lora_scale: float = 1.0,
        seed: int = 42,
        offload_text_encoder: bool = True,
        progress_callback: Callable[[str, int | None, int | None], None] | None = None,
    ) -> Tensor:
        """Generate ``[B,T,H,W,3]`` videos in ``[-1,1]``.

        Each tensor in the source/target sequences describes one source
        video and one or more region mappings. Source masks may be static
        ``[B,K,H,W]`` or temporal ``[B,T,K,H,W]``. Optional
        ``source_content_images`` identify the frame on which each static mask
        was drawn. Source and output durations may differ; motion positions are
        aligned over the output timeline. Base CFG needs no sources.
        """
        FlowUniPCMultistepScheduler = _wan_module(
            "utils.fm_solvers_unipc"
        ).FlowUniPCMultistepScheduler

        if (
            target_image.ndim != 4
            or target_image.shape[-1] != 3
            or not target_image.is_floating_point()
        ):
            raise ValueError("target_image must be floating-point [B,H,W,3]")
        batch, height, width, _ = target_image.shape
        spatial_multiple = self.config.vae_stride[1] * self.config.patch_size[1]
        if height % spatial_multiple or width % spatial_multiple:
            raise ValueError(
                f"Target height and width must be divisible by {spatial_multiple}"
            )
        mode = normalize_wan_guidance_mode(guidance_mode)
        branches = wan_guidance_required_branches(mode)
        needs_motion = any(name.startswith("adapter_") for name in branches)

        if num_frames is None:
            if needs_motion:
                videos = _as_tensor_list(source_videos, "source_videos")
                frame_count = min(int(video.shape[1]) for video in videos)
            else:
                frame_count = 25
        else:
            frame_count = int(num_frames)
        if frame_count < 1 or (frame_count - 1) % self.config.vae_stride[0]:
            raise ValueError("num_frames must be positive and have the form 4n+1")
        if needs_motion and frame_count < self.what_moves.window_length:
            raise ValueError(
                f"Motion transfer needs at least {self.what_moves.window_length} frames"
            )

        steps = (
            self.config.default_steps
            if num_inference_steps is None
            else int(num_inference_steps)
        )
        if steps < 1:
            raise ValueError("num_inference_steps must be positive")
        text_scale = float(
            self.config.default_text_guidance
            if text_guidance_scale is None
            else text_guidance_scale
        )
        motion_scale = float(motion_guidance_scale)
        lora_scale = float(lora_scale)
        for name, value in (
            ("text_guidance_scale", text_scale),
            ("motion_guidance_scale", motion_scale),
            ("lora_scale", lora_scale),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        target_image = target_image.to(self.device, dtype=torch.float32)
        prompts = _prompt_batch(prompt, batch, "prompt")
        negatives = _prompt_batch(
            self.config.negative_prompt if negative_prompt is None else negative_prompt,
            batch,
            "negative_prompt",
        )
        latent_t = (frame_count - 1) // self.config.vae_stride[0] + 1
        latent_h = height // self.config.vae_stride[1]
        latent_w = width // self.config.vae_stride[2]
        latent_grid = (
            latent_t,
            latent_h // self.config.patch_size[1],
            latent_w // self.config.patch_size[2],
        )

        def report(stage, current=None, total=None):
            if progress_callback is not None:
                progress_callback(stage, current, total)

        report("Preparation")
        with torch.autocast("cuda", dtype=self.weight_dtype):
            image_condition = self._image_condition(
                target_image,
                frame_count,
                (latent_t, latent_h, latent_w),
            )
            if needs_motion:
                sources = self._prepare_sources(
                    source_videos,
                    source_masks,
                    target_masks,
                    source_content_images,
                    target_image=target_image,
                )
                token_groups, position_groups = zip(
                    *[
                        self._encode_motion_tokens(
                            video,
                            source,
                            reference,
                            target_image,
                            target,
                            latent_grid,
                        )
                        for video, source, target, reference in sources
                    ]
                )
                motion_tokens = torch.cat(token_groups, dim=1)
                motion_positions = torch.cat(position_groups, dim=1)
            else:
                motion_tokens = torch.empty(
                    batch,
                    0,
                    self.high_noise_model.dim,
                    device=self.device,
                    dtype=self.weight_dtype,
                )
                motion_positions = torch.empty(
                    batch,
                    0,
                    3,
                    device=self.device,
                )

            positive_context = self._encode_text(prompts)
            negative_context = (
                self._encode_text(negatives)
                if any(name.endswith("_neg") for name in branches)
                else None
            )
            if offload_text_encoder:
                self.text_encoder.model.cpu()
                torch.cuda.empty_cache()

            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            latent = torch.randn(
                batch,
                16,
                latent_t,
                latent_h,
                latent_w,
                generator=generator,
                device=self.device,
            )
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.config.train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            scheduler.set_timesteps(steps, device=self.device, shift=self.flow_shift)

            report("Sampling", 0, steps)
            with _lora_scaled(self, lora_scale):
                for step_index, timestep in enumerate(scheduler.timesteps):
                    timesteps = timestep.expand(batch).to(self.device)
                    expert = self._expert(timesteps)
                    predictions = {}
                    for branch in branches:
                        adapted = branch.startswith("adapter_")
                        context = (
                            positive_context
                            if branch.endswith("_pos")
                            else negative_context
                        )
                        arguments = {
                            "model": expert,
                            "latent": latent,
                            "image_condition": image_condition,
                            "timesteps": timesteps,
                            "context": context,
                            "motion_tokens": (
                                motion_tokens if adapted else motion_tokens[:, :0]
                            ),
                            "motion_positions": (
                                motion_positions if adapted else motion_positions[:, :0]
                            ),
                        }
                        if adapted:
                            predictions[branch] = self._transformer_forward(**arguments)
                        else:
                            with _lora_disabled(expert):
                                predictions[branch] = self._transformer_forward(
                                    **arguments
                                )
                    prediction = combine_wan_guidance_predictions(
                        mode,
                        predictions,
                        text_guidance_scale=text_scale,
                        motion_guidance_scale=motion_scale,
                    )
                    latent = scheduler.step(
                        prediction,
                        timestep,
                        latent,
                        return_dict=False,
                        generator=generator,
                    )[0]
                    report("Sampling", step_index + 1, steps)

            report("VAE decoding")
            videos = self.vae.decode(list(latent))
        report("Finishing")
        return torch.stack(videos).permute(0, 2, 3, 4, 1)


def load_wan_model(
    wan_checkpoint: str | Path,
    what_moves_checkpoint: str | Path,
    adapter_checkpoint: str | Path,
    **kwargs,
) -> WanMotionTransfer:
    """Load the base Wan model, WhatMoves encoder, and trained adapter."""
    return WanMotionTransfer(
        wan_checkpoint,
        what_moves_checkpoint,
        adapter_checkpoint,
        **kwargs,
    )
