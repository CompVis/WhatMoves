"""Load the small, redistributable examples bundled with the release."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .media import decode_video_frame, probe_video, read_image
from .state import MaskRecord, Session, SourceAsset, TargetAsset, new_id


def _asset_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"Invalid example asset: {relative}")
    return path


def _mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as opened:
        image = opened.convert("L")
        if image.size != (shape[1], shape[0]):
            image = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
        mask = np.asarray(image) > 127
    if not mask.any():
        raise ValueError(f"Example mask is empty: {path.name}")
    return mask


def load_examples(
    session: Session,
    root: Path,
    *,
    source_fps: int,
    max_sources: int,
    max_targets: int,
    max_image_side: int,
) -> None:
    """Copy manifest-defined examples into one disposable browser session."""
    manifest_path = root / "examples.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1:
        raise ValueError("Unsupported examples manifest version")

    source_refs: dict[str, tuple[SourceAsset, dict[str, str]]] = {}
    for record in manifest.get("sources", [])[:max_sources]:
        source_id = new_id("source")
        source_path = _asset_path(root, record["video"])
        runtime_path = session.directory / f"{source_id}.mp4"
        shutil.copyfile(source_path, runtime_path)
        runtime_path.chmod(0o600)
        fps, frame_count = probe_video(runtime_path)
        if frame_count is None or frame_count < 1:
            raise ValueError(f"Example video has no known frames: {source_path.name}")
        if abs(fps - source_fps) > 0.05:
            raise ValueError(f"Example video must be encoded at {source_fps} fps")
        trim = record.get("trim", [0, frame_count - 1])
        trim_start = int(trim[0])
        trim_end = int(trim[1])
        if not 0 <= trim_start <= trim_end < frame_count:
            raise ValueError(f"Invalid example trim for {source_path.name}")
        frame = decode_video_frame(runtime_path, trim_start, max_image_side)
        source = SourceAsset(
            id=source_id,
            name=str(record.get("name", source_path.stem)),
            video_path=runtime_path,
            frame=frame,
            fps=float(source_fps),
            frame_count=frame_count,
            original_fps=float(source_fps),
            current_frame=trim_start,
            trim_start=trim_start,
            trim_end=trim_end,
            sam_status="idle",
            example=True,
        )
        mask_refs = {}
        for mask_record in record.get("masks", []):
            frame_index = int(mask_record["frame"])
            if not 0 <= frame_index < frame_count:
                raise ValueError("Example source mask frame is out of range")
            mask_id = new_id("mask")
            allocated = session.allocate_color()
            color = str(mask_record.get("color") or allocated)
            source.masks[mask_id] = MaskRecord(
                mask_id,
                _mask(_asset_path(root, mask_record["mask"]), frame.shape[:2]),
                color=color,
                frame_index=frame_index,
            )
            mask_refs[str(mask_record["id"])] = mask_id
            source.mask_revision += 1
        session.sources[source.id] = source
        source_refs[str(record["id"])] = (source, mask_refs)

    target_refs: dict[str, tuple[TargetAsset, dict[str, str]]] = {}
    for record in manifest.get("targets", [])[:max_targets]:
        target_id = new_id("target")
        image_path = _asset_path(root, record["image"])
        image = read_image(image_path, max_image_side)
        session.target_revision += 1
        target = TargetAsset(
            id=target_id,
            name=str(record.get("name", image_path.stem)),
            image=image,
            revision=session.target_revision,
            sam_status="idle",
            example=True,
        )
        mask_refs = {}
        for mask_record in record.get("masks", []):
            mask_id = new_id("mask")
            target.masks[mask_id] = MaskRecord(
                mask_id,
                _mask(_asset_path(root, mask_record["mask"]), image.shape[:2]),
            )
            mask_refs[str(mask_record["id"])] = mask_id
            target.mask_revision += 1
        session.targets[target.id] = target
        target_refs[str(record["id"])] = (target, mask_refs)

    for mapping in manifest.get("mappings", []):
        source, source_masks = source_refs[str(mapping["source"])]
        target, target_masks = target_refs[str(mapping["target"])]
        target_mask = target.masks[target_masks[str(mapping["target_mask"])]]
        target_mask.source_id = source.id
        target_mask.source_mask_id = source_masks[str(mapping["source_mask"])]
        target.mask_revision += 1
    session.selected_target_id = next(iter(session.targets), None)
