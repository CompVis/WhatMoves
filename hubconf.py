"""PyTorch Hub entrypoints for WhatMoves and its Wan motion adapter."""

from __future__ import annotations

import hashlib as _hashlib
import json as _json
from pathlib import PurePosixPath as _PurePosixPath

import torch as _torch

dependencies = ["torch", "einops", "huggingface_hub"]

_WEIGHTS_REPO = "CompVis/WhatMoves"
_WEIGHTS_REVISION = "31820f6dbaa3f4f535bdb472d44db2c0bb03349c"
_WAN_BASE_MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B"
_WAN_BASE_REVISION = "206a9ee1b7bfaaf8f7e4d81335650533490646a3"
_DEFAULT_VARIANT = "gated_static_step600000"
_VARIANTS = {
    _DEFAULT_VARIANT: {
        "what_moves": "what_moves/gated_static_step600000.pt",
        "metadata": "wan/gated_static_step600000/metadata.json",
    }
}
_WAN_ALLOW_PATTERNS = (
    "high_noise_model/*",
    "low_noise_model/*",
    "Wan2.1_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/*",
)


def _variant_files(variant: str) -> dict[str, str]:
    try:
        return _VARIANTS[variant]
    except KeyError as error:
        choices = ", ".join(sorted(_VARIANTS))
        raise ValueError(
            f"Unknown WhatMoves variant {variant!r}; choose: {choices}"
        ) from error


def _download_file(
    filename: str,
    *,
    repo_id: str,
    revision: str | None,
    cache_dir: str | None,
    token: bool | str | None,
    force_download: bool,
    local_files_only: bool,
) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        force_download=force_download,
        local_files_only=local_files_only,
    )


def _download_kwargs(
    *,
    repo_id: str,
    revision: str | None,
    cache_dir: str | None,
    hf_token: bool | str | None,
    force_download: bool,
    local_files_only: bool,
) -> dict[str, object]:
    return {
        "repo_id": repo_id,
        "revision": revision,
        "cache_dir": cache_dir,
        "token": hf_token,
        "force_download": force_download,
        "local_files_only": local_files_only,
    }


def _release_revision(repo_id: str, revision: str | None) -> str | None:
    """Pin official release downloads while leaving custom repositories alone."""
    if revision is None and repo_id == _WEIGHTS_REPO:
        return _WEIGHTS_REVISION
    return revision


def _metadata(
    variant: str,
    *,
    repo_id: str,
    revision: str | None,
    cache_dir: str | None,
    hf_token: bool | str | None,
    force_download: bool,
    local_files_only: bool,
) -> dict[str, object]:
    files = _variant_files(variant)
    path = _download_file(
        files["metadata"],
        **_download_kwargs(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            hf_token=hf_token,
            force_download=force_download,
            local_files_only=local_files_only,
        ),
    )
    with open(path, encoding="utf-8") as file:
        metadata = _json.load(file)
    required = {
        "format": "whatmoves_wan_motion_adapter",
        "format_version": 1,
        "variant": variant,
    }
    mismatched = {
        name: metadata.get(name)
        for name, expected in required.items()
        if metadata.get(name) != expected
    }
    if mismatched:
        raise ValueError(f"Invalid Wan adapter metadata in {path}: {mismatched}")
    return metadata


def _metadata_path(metadata: dict[str, object], name: str) -> str:
    value = metadata.get(name)
    if not isinstance(value, str):
        raise ValueError(f"Wan adapter metadata has no string {name!r}")
    path = _PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Wan adapter metadata has unsafe {name!r}: {value!r}")
    return value


def _verify_download(
    path: str,
    filename: str,
    metadata: dict[str, object],
) -> None:
    hashes = metadata.get("sha256")
    expected = hashes.get(filename) if isinstance(hashes, dict) else None
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Wan adapter metadata has no SHA-256 for {filename!r}")
    digest = _hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"Checksum mismatch for {filename}: expected {expected}, got {actual}"
        )


def what_moves(
    *,
    variant: str = _DEFAULT_VARIANT,
    checkpoint: str | None = None,
    device: str | _torch.device = "cpu",
    dtype: _torch.dtype | None = None,
    repo_id: str = _WEIGHTS_REPO,
    revision: str | None = None,
    cache_dir: str | None = None,
    hf_token: bool | str | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
    verify_checksum: bool = True,
    **model_kwargs,
):
    """Load the released localized-motion encoder.

    By default the checkpoint is downloaded from ``CompVis/WhatMoves`` and
    cached by Hugging Face. Pass ``checkpoint`` to use a local file instead.
    Private repositories work after ``hf auth login`` or with ``HF_TOKEN``.
    """
    _variant_files(variant)
    if checkpoint is None:
        revision = _release_revision(repo_id, revision)
        metadata = _metadata(
            variant,
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            hf_token=hf_token,
            force_download=force_download,
            local_files_only=local_files_only,
        )
        filename = _metadata_path(metadata, "what_moves_checkpoint")
        checkpoint = _download_file(
            filename,
            **_download_kwargs(
                repo_id=repo_id,
                revision=revision,
                cache_dir=cache_dir,
                hf_token=hf_token,
                force_download=force_download,
                local_files_only=local_files_only,
            ),
        )
        if verify_checksum:
            _verify_download(checkpoint, filename, metadata)

    from what_moves import WhatMoves as _WhatMoves

    return _WhatMoves.from_pretrained(
        checkpoint,
        device=device,
        dtype=dtype,
        **model_kwargs,
    )


def wan(
    *,
    variant: str = _DEFAULT_VARIANT,
    wan_checkpoint: str | None = None,
    what_moves_checkpoint: str | None = None,
    adapter_checkpoint: str | None = None,
    device: str | _torch.device = "cuda",
    dtype: _torch.dtype = _torch.bfloat16,
    repo_id: str = _WEIGHTS_REPO,
    revision: str | None = None,
    base_model_id: str | None = None,
    base_revision: str | None = None,
    cache_dir: str | None = None,
    hf_token: bool | str | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
    verify_checksum: bool = True,
    **model_kwargs,
):
    """Load Wan2.2 I2V-A14B with the matching WhatMoves adapter.

    Release weights and metadata come from ``CompVis/WhatMoves``. The official
    Wan base snapshot is downloaded automatically unless ``wan_checkpoint`` is
    supplied. Any checkpoint argument can independently override its download.
    """
    _variant_files(variant)
    revision = _release_revision(repo_id, revision)
    metadata = None
    if (
        what_moves_checkpoint is None
        or adapter_checkpoint is None
        or (wan_checkpoint is None and base_model_id is None)
    ):
        metadata = _metadata(
            variant,
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            hf_token=hf_token,
            force_download=force_download,
            local_files_only=local_files_only,
        )
    download_kwargs = _download_kwargs(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        hf_token=hf_token,
        force_download=force_download,
        local_files_only=local_files_only,
    )
    if what_moves_checkpoint is None:
        filename = _metadata_path(metadata, "what_moves_checkpoint")
        what_moves_checkpoint = _download_file(
            filename,
            **download_kwargs,
        )
        if verify_checksum:
            _verify_download(what_moves_checkpoint, filename, metadata)
    if adapter_checkpoint is None:
        filename = _metadata_path(metadata, "adapter_checkpoint")
        adapter_checkpoint = _download_file(
            filename,
            **download_kwargs,
        )
        if verify_checksum:
            _verify_download(adapter_checkpoint, filename, metadata)
    if wan_checkpoint is None:
        if base_model_id is None:
            base_model_id = _metadata_path(metadata, "base_model_id")
        if base_revision is None and base_model_id == _WAN_BASE_MODEL_ID:
            base_revision = _WAN_BASE_REVISION
        from huggingface_hub import snapshot_download

        wan_checkpoint = snapshot_download(
            repo_id=base_model_id,
            revision=base_revision,
            cache_dir=cache_dir,
            token=hf_token,
            force_download=force_download,
            local_files_only=local_files_only,
            allow_patterns=list(_WAN_ALLOW_PATTERNS),
        )

    from what_moves import load_wan_model as _load_wan_model

    return _load_wan_model(
        wan_checkpoint,
        what_moves_checkpoint,
        adapter_checkpoint,
        device=device,
        dtype=dtype,
        **model_kwargs,
    )
