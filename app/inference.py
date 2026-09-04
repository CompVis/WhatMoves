"""Wan input preparation and Torch Hub model loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Callable

import numpy as np

from .config import AppConfig
from .media import decode_video_range, write_video


@dataclass(frozen=True)
class SourceSnapshot:
    video_path: Path
    trim_start: int
    trim_end: int
    source_masks: tuple[np.ndarray, ...]
    target_masks: tuple[np.ndarray, ...]
    reference_frames: tuple[int, ...]


@dataclass(frozen=True)
class GenerationSnapshot:
    target_image: np.ndarray
    sources: tuple[SourceSnapshot, ...]
    prompt: str
    negative_prompt: str | None
    steps: int
    seed: int
    guidance_mode: str
    text_guidance_scale: float
    motion_guidance_scale: float
    lora_scale: float
    output_path: Path


class WanService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._model = None
        self._status = "idle"
        self._lock = threading.RLock()

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def _load_model(self):
        """Load the release through the same Torch Hub API exposed to users."""
        if self._model is not None:
            return self._model
        self._set_status("loading")
        try:
            import torch

            kwargs = {
                "variant": self.config.model_variant,
                "device": self.config.device,
            }
            if self.config.wan_checkpoint is not None:
                kwargs["wan_checkpoint"] = str(self.config.wan_checkpoint)
            self._model = torch.hub.load(
                str(Path(__file__).resolve().parents[1]),
                "wan",
                source="local",
                **kwargs,
            )
        except Exception:
            self._set_status("error")
            raise
        self._set_status("ready")
        return self._model

    def close(self) -> None:
        with self._lock:
            self._model = None
            self._status = "idle"
        if self.config.device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    def generate(
        self,
        request: GenerationSnapshot,
        progress_callback: Callable[[str, int | None, int | None], None] | None = None,
    ) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        def report(stage, current=None, total=None):
            if progress_callback is not None:
                progress_callback(stage, current, total)

        height = self.config.output_height
        width = self.config.output_width
        frames = self.config.output_frames

        report("Preparation")
        target = torch.from_numpy(request.target_image.copy())
        target = (
            F.interpolate(
                target.permute(2, 0, 1)[None].float(),
                size=(height, width),
                mode="bilinear",
                antialias=True,
            )
            .permute(0, 2, 3, 1)
            .div(127.5)
            .sub(1)
        )

        source_videos = []
        source_masks = []
        target_masks = []
        source_content_images = []
        for source in request.sources:
            decoded = decode_video_range(
                source.video_path, source.trim_start, source.trim_end
            )
            video = torch.from_numpy(decoded.copy()).permute(0, 3, 1, 2).float()
            video = (
                F.interpolate(
                    video,
                    size=(height, width),
                    mode="bilinear",
                    antialias=True,
                )
                .permute(0, 2, 3, 1)
                .div(127.5)
                .sub(1)[None]
            )
            grouped: dict[int, list[int]] = {}
            for index, reference in enumerate(source.reference_frames):
                grouped.setdefault(reference, []).append(index)
            for reference, indices in grouped.items():
                local_reference = reference - source.trim_start
                if not 0 <= local_reference < decoded.shape[0]:
                    raise ValueError("A source mask lies outside its selected interval")
                source_mask = torch.from_numpy(
                    np.stack([source.source_masks[index] for index in indices])
                ).float()[None]
                source_mask = F.interpolate(
                    source_mask, size=(height, width), mode="nearest"
                ).bool()
                target_mask = torch.from_numpy(
                    np.stack([source.target_masks[index] for index in indices])
                ).float()[None]
                target_mask = F.interpolate(
                    target_mask, size=(height, width), mode="nearest"
                ).bool()
                reference_image = video[:, local_reference]
                source_videos.append(video)
                source_masks.append(source_mask)
                target_masks.append(target_mask)
                source_content_images.append(reference_image)

        # Decode and validate every user input before paying the cost of loading
        # the two Wan experts. This also makes bad frame ranges fail quickly.
        if self._model is None:
            report("Model loading")
        model = self._load_model()
        kwargs = {
            "prompt": request.prompt,
            "target_image": target,
            "negative_prompt": request.negative_prompt,
            "num_frames": frames,
            "num_inference_steps": request.steps,
            "text_guidance_scale": request.text_guidance_scale,
            "motion_guidance_scale": request.motion_guidance_scale,
            "guidance_mode": request.guidance_mode,
            "lora_scale": request.lora_scale,
            "seed": request.seed,
        }
        if request.guidance_mode != "base_cfg":
            kwargs.update(
                source_videos=source_videos,
                source_masks=source_masks,
                target_masks=target_masks,
                source_content_images=source_content_images,
            )
        self._set_status("generating")
        try:
            with torch.inference_mode():
                generated = model.sample(
                    **kwargs,
                    progress_callback=progress_callback,
                )
            video = (
                generated[0]
                .detach()
                .float()
                .cpu()
                .clamp(-1, 1)
                .add(1)
                .mul(127.5)
                .round()
                .byte()
                .numpy()
            )
            report("Finishing")
            write_video(request.output_path, video, fps=self.config.output_fps)
            return {
                "output_name": request.output_path.name,
                "frames": int(video.shape[0]),
                "width": int(video.shape[2]),
                "height": int(video.shape[1]),
                "fps": self.config.output_fps,
            }
        finally:
            self._set_status("ready")
