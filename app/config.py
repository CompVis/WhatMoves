"""Application configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

DEFAULT_VARIANT = "gated_static_step600000"


def _optional_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _runtime_root() -> Path:
    configured = os.environ.get("WHATMOVES_APP_TMP_DIR")
    if configured:
        return Path(configured).expanduser()
    job = os.environ.get("SLURM_JOB_ID", "local")
    suffix = f"{job}-{os.getpid()}"
    return Path("/tmp") / f"whatmoves-app-{suffix}"


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class AppConfig:
    model_variant: str = DEFAULT_VARIANT
    wan_checkpoint: Path | None = _optional_path("WHATMOVES_WAN_CHECKPOINT")
    sam_model_id: str = "facebook/sam2-hiera-large"
    device: str = "cuda"
    runtime_root: Path = _runtime_root()
    output_height: int = 480
    output_width: int = 704
    output_frames: int = 25
    output_fps: int = 8
    source_fps: int = 8
    max_image_side: int = 1600
    max_upload_bytes: int = 2 * 1024**3
    max_sources: int = _positive_int("WHATMOVES_APP_MAX_SOURCES", 8)
    max_targets: int = _positive_int("WHATMOVES_APP_MAX_TARGETS", 8)
    max_sam_frames: int = _positive_int("WHATMOVES_APP_MAX_SAM_FRAMES", 8)
    max_gpu_tasks: int = _positive_int("WHATMOVES_APP_MAX_GPU_TASKS", 32)
    session_ttl_seconds: int = _positive_int("WHATMOVES_APP_SESSION_TTL", 7200)
    cleanup_interval_seconds: int = _positive_int("WHATMOVES_APP_CLEANUP_INTERVAL", 60)
