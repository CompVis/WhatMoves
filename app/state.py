"""In-memory browser-session state."""

from __future__ import annotations

import colorsys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from typing import Any
from uuid import uuid4

import numpy as np

MASK_COLORS = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#7A6FF0",
    "#32B8A6",
    "#EF6C96",
    "#8AAE3D",
    "#C98932",
)


@dataclass
class MaskRecord:
    id: str
    data: np.ndarray
    color: str | None = None
    source_id: str | None = None
    source_mask_id: str | None = None
    frame_index: int | None = None


@dataclass
class Draft:
    data: np.ndarray
    asset_revision: int
    prompt_revision: int
    frame_index: int | None = None


@dataclass
class TargetAsset:
    id: str
    name: str
    image: np.ndarray
    revision: int = 1
    sam_status: str = "queued"
    sam_error: str | None = None
    masks: OrderedDict[str, MaskRecord] = field(default_factory=OrderedDict)
    draft: Draft | None = None
    prediction_token: str | None = None
    hover_prediction_token: str | None = None
    mask_revision: int = 0
    example: bool = False


@dataclass
class SourceAsset:
    id: str
    name: str
    video_path: Path
    frame: np.ndarray
    fps: float
    frame_count: int
    original_fps: float
    current_frame: int = 0
    trim_start: int = 0
    trim_end: int = 0
    revision: int = 1
    sam_status: str = "queued"
    sam_error: str | None = None
    masks: OrderedDict[str, MaskRecord] = field(default_factory=OrderedDict)
    draft: Draft | None = None
    prediction_token: str | None = None
    hover_prediction_token: str | None = None
    frame_selection_token: str | None = None
    mask_revision: int = 0
    example: bool = False


@dataclass
class Session:
    id: str
    directory: Path
    targets: OrderedDict[str, TargetAsset] = field(default_factory=OrderedDict)
    selected_target_id: str | None = None
    sources: OrderedDict[str, SourceAsset] = field(default_factory=OrderedDict)
    next_color: int = 0
    target_revision: int = 0
    active_generation: str | None = None
    latest_generation: str | None = None
    failed_generation: str | None = None
    latest_output: Path | None = None
    input_revision: int = 0
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    closed: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def selected_target(self) -> TargetAsset | None:
        if self.selected_target_id is None:
            return None
        return self.targets.get(self.selected_target_id)

    def allocate_color(self) -> str:
        if self.next_color < len(MASK_COLORS):
            color = MASK_COLORS[self.next_color]
        else:
            hue = (0.13 + self.next_color * 0.61803398875) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.68, 0.94)
            color = "#" + "".join(f"{round(channel * 255):02X}" for channel in rgb)
        self.next_color += 1
        return color

    def touch(self) -> None:
        self.input_revision += 1
        self.accessed_at = time.time()

    def access(self) -> None:
        self.accessed_at = time.time()


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()

    def create(self) -> Session:
        session_id = uuid4().hex
        directory = self.root / session_id
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        session = Session(session_id, directory)
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            try:
                session = self._sessions[session_id]
            except KeyError as error:
                raise KeyError(f"Unknown session {session_id}") from error
        session.access()
        return session

    def pop(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.pop(session_id, None)

    def pop_expired(self, ttl_seconds: float) -> list[Session]:
        cutoff = time.time() - float(ttl_seconds)
        with self._lock:
            expired = [
                session
                for session in self._sessions.values()
                if session.accessed_at < cutoff
            ]
            for session in expired:
                self._sessions.pop(session.id, None)
        return expired

    def pop_all(self) -> list[Session]:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        return sessions


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


def clear_source_mappings(
    session: Session, source_id: str, mask_id: str | None = None
) -> None:
    for target in session.targets.values():
        changed = False
        for target_mask in target.masks.values():
            if target_mask.source_id != source_id:
                continue
            if mask_id is None or target_mask.source_mask_id == mask_id:
                target_mask.source_id = None
                target_mask.source_mask_id = None
                changed = True
        if changed:
            target.mask_revision += 1


def clear_excluded_source_mappings(session: Session, source: SourceAsset) -> int:
    """Clear mappings whose source observation falls outside the trim."""
    excluded = {
        mask.id
        for mask in source.masks.values()
        if mask.frame_index is None
        or not source.trim_start <= mask.frame_index <= source.trim_end
    }
    changed = 0
    for target in session.targets.values():
        target_changed = False
        for mask in target.masks.values():
            if mask.source_id == source.id and mask.source_mask_id in excluded:
                mask.source_id = None
                mask.source_mask_id = None
                changed += 1
                target_changed = True
        if target_changed:
            target.mask_revision += 1
    return changed


def source_mask_lookup(session: Session, source_id: str, mask_id: str) -> MaskRecord:
    try:
        return session.sources[source_id].masks[mask_id]
    except KeyError as error:
        raise KeyError(f"Unknown source mask {source_id}/{mask_id}") from error


def session_payload(session: Session) -> dict[str, Any]:
    def mask_payload(mask: MaskRecord, url: str, preview_url: str | None = None):
        payload = {
            "id": mask.id,
            "color": mask.color,
            "source_id": mask.source_id,
            "source_mask_id": mask.source_mask_id,
            "frame_index": mask.frame_index,
            "url": url,
        }
        if preview_url:
            payload["preview_url"] = preview_url
        return payload

    targets = []
    for target in session.targets.values():
        base = f"/api/sessions/{session.id}/targets/{target.id}"
        targets.append(
            {
                "id": target.id,
                "name": target.name,
                "revision": target.revision,
                "width": int(target.image.shape[1]),
                "height": int(target.image.shape[0]),
                "sam_status": target.sam_status,
                "sam_error": target.sam_error,
                "example": target.example,
                "image_url": f"{base}/image?v={target.revision}",
                "thumbnail_url": (
                    f"{base}/thumbnail.png?v={target.revision}-{target.mask_revision}"
                ),
                "masks": [
                    mask_payload(
                        mask,
                        f"{base}/masks/{mask.id}.png?v={target.mask_revision}",
                    )
                    for mask in target.masks.values()
                ],
            }
        )

    sources = []
    for source in session.sources.values():
        base = f"/api/sessions/{session.id}/sources/{source.id}"
        sources.append(
            {
                "id": source.id,
                "name": source.name,
                "revision": source.revision,
                "width": int(source.frame.shape[1]),
                "height": int(source.frame.shape[0]),
                "fps": source.fps,
                "original_fps": source.original_fps,
                "frame_count": source.frame_count,
                "current_frame": source.current_frame,
                "trim_start": source.trim_start,
                "trim_end": source.trim_end,
                "sam_status": source.sam_status,
                "sam_error": source.sam_error,
                "example": source.example,
                "image_url": (
                    f"{base}/frame.png?frame={source.current_frame}&v={source.revision}"
                ),
                "video_url": f"{base}/video?v={source.revision}",
                "thumbnail_url": f"{base}/thumbnail.png?v={source.revision}-{source.mask_revision}",
                "masks": [
                    mask_payload(
                        mask,
                        f"{base}/masks/{mask.id}.png?v={source.mask_revision}",
                        f"{base}/masks/{mask.id}/preview.png?v={source.mask_revision}",
                    )
                    for mask in source.masks.values()
                ],
            }
        )
    return {
        "id": session.id,
        "targets": targets,
        "selected_target_id": session.selected_target_id,
        "sources": sources,
        "active_generation": session.active_generation,
        "latest_generation": session.latest_generation,
        "input_revision": session.input_revision,
    }
