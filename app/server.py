"""FastAPI server for the WhatMoves annotation and generation workspace."""

from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import os
from pathlib import Path
import shutil
import threading
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import AppConfig
from .examples import load_examples
from .inference import GenerationSnapshot, SourceSnapshot, WanService
from .media import (
    composite_masks,
    decode_video_frame,
    jpeg_bytes,
    mask_png_bytes,
    png_bytes,
    read_image,
    resample_video,
)
from .sam_service import SamService
from .scheduler import GpuScheduler, QueueFull, Superseded
from .state import (
    Draft,
    MaskRecord,
    Session,
    SessionStore,
    SourceAsset,
    TargetAsset,
    clear_excluded_source_mappings,
    clear_source_mappings,
    new_id,
    session_payload,
    source_mask_lookup,
)

STATIC_ROOT = Path(__file__).resolve().parent / "static"
ASSETS_ROOT = Path(__file__).resolve().parent / "assets"


class PromptRequest(BaseModel):
    asset_revision: int
    prompt_revision: int
    positive: list[list[float]] = Field(default_factory=list)
    negative: list[list[float]] = Field(default_factory=list)
    box: list[float] | None = None
    frame_index: int | None = Field(default=None, ge=0)
    transient: bool = False


class CommitRequest(BaseModel):
    asset_revision: int
    prompt_revision: int
    frame_index: int | None = Field(default=None, ge=0)


class SourceSettings(BaseModel):
    current_frame: int | None = Field(default=None, ge=0)
    trim_start: int | None = Field(default=None, ge=0)
    trim_end: int | None = Field(default=None, ge=0)


class MappingRequest(BaseModel):
    source_id: str | None = None
    source_mask_id: str | None = None


class GenerateRequest(BaseModel):
    prompt: str = Field(default="", max_length=4096)
    negative_prompt: str | None = Field(default=None, max_length=4096)
    steps: int = Field(default=40, ge=1, le=100)
    seed: int = Field(default=42, ge=-(2**63), le=2**63 - 1)
    guidance_mode: Literal[
        "base_cfg",
        "text_cfg",
        "joint_cfg",
        "motion_cfg",
        "additive_cfg",
        "factorized_cfg",
    ] = "text_cfg"
    text_guidance_scale: float = Field(default=3.5, allow_inf_nan=False)
    motion_guidance_scale: float = Field(default=1.0, allow_inf_nan=False)
    lora_scale: float = Field(default=1.0, allow_inf_nan=False)


class Runtime:
    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        eager_embeddings: bool = True,
    ) -> None:
        self.config = config or AppConfig()
        self.eager_embeddings = eager_embeddings
        self.store = SessionStore(self.config.runtime_root)
        self.cpu = ThreadPoolExecutor(max_workers=4, thread_name_prefix="whatmoves-cpu")
        self.scheduler = GpuScheduler(max_pending=self.config.max_gpu_tasks)
        self.sam = SamService(
            self.config.sam_model_id,
            self.config.device,
            max_predictors=self.config.max_sam_frames,
        )
        self.wan = WanService(self.config)
        self._closing = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_sessions,
            name="whatmoves-cleanup",
            daemon=True,
        )
        self._cleanup_thread.start()

    def close(self) -> None:
        if self._closing.is_set():
            return
        self._closing.set()
        self._cleanup_thread.join()
        sessions = self.store.pop_all()
        for session in sessions:
            self.dispose_session(session)
        self.scheduler.close()
        for session in sessions:
            _remove_session_directory(self.config.runtime_root, session.directory)
        self.sam.close()
        self.wan.close()
        self.cpu.shutdown(wait=False, cancel_futures=True)
        try:
            self.config.runtime_root.rmdir()
        except OSError:
            pass

    def dispose_session(self, session: Session) -> None:
        """Release every live resource owned by one browser session."""
        with session.lock:
            if session.closed:
                return
            session.closed = True
            targets = list(session.targets.values())
            sources = list(session.sources.values())
            latest_task = session.latest_generation
            failed_task = session.failed_generation
            latest_output = session.latest_output
            active_task = session.active_generation
            session.targets.clear()
            session.selected_target_id = None
            session.sources.clear()
            session.latest_generation = None
            session.failed_generation = None
            session.latest_output = None
        self.scheduler.cancel_owner(session.id)
        self.sam.release(f"{session.id}:")
        for target in targets:
            _clear_asset(target)
        for source in sources:
            _clear_asset(source)
            source.video_path.unlink(missing_ok=True)
        if latest_output is not None:
            latest_output.unlink(missing_ok=True)
        if latest_task is not None and latest_task != active_task:
            self.scheduler.discard(latest_task)
        if failed_task is not None and failed_task != active_task:
            self.scheduler.discard(failed_task)
        active = self.scheduler.get(active_task) if active_task is not None else None
        if active is None or active.status != "running":
            if active_task is not None:
                self.scheduler.discard(active_task)
            _remove_session_directory(self.config.runtime_root, session.directory)

    def _cleanup_sessions(self) -> None:
        while not self._closing.wait(self.config.cleanup_interval_seconds):
            for session in self.store.pop_expired(self.config.session_ttl_seconds):
                self.dispose_session(session)


async def _run_cpu(runtime: Runtime, function, *args, **kwargs):
    # CPU-only is the lightweight test/development mode; production CUDA runs
    # media work in the dedicated pool so the HTTP event loop stays responsive.
    if runtime.config.device == "cpu":
        return function(*args, **kwargs)
    return await asyncio.wrap_future(runtime.cpu.submit(function, *args, **kwargs))


def _session(runtime: Runtime, session_id: str) -> Session:
    try:
        return runtime.store.get(session_id)
    except KeyError as error:
        raise HTTPException(404, str(error)) from error


def _remove_session_directory(root: Path, directory: Path) -> None:
    """Remove exactly one validated session directory."""
    root = root.resolve()
    directory = directory.resolve()
    if directory.parent != root or not directory.name:
        raise RuntimeError(f"Refusing to remove invalid session path: {directory}")
    shutil.rmtree(directory, ignore_errors=True)


def _clear_asset(asset: TargetAsset | SourceAsset) -> None:
    asset.draft = None
    asset.prediction_token = None
    asset.hover_prediction_token = None
    if isinstance(asset, SourceAsset):
        asset.frame_selection_token = None
    asset.masks.clear()


def _sam_prefix(session_id: str, kind: str, asset_id: str) -> str:
    return f"{session_id}:{kind}:{asset_id}:"


def _release_asset(runtime: Runtime, session_id: str, kind: str, asset_id: str) -> None:
    prefix = _sam_prefix(session_id, kind, asset_id)
    runtime.scheduler.cancel_prefix(f"sam:{prefix}")
    runtime.sam.release(prefix)


def _asset(session: Session, kind: str, asset_id: str):
    if kind == "target":
        try:
            return session.targets[asset_id]
        except KeyError as error:
            raise HTTPException(404, f"Unknown target {asset_id}") from error
    if kind == "source":
        try:
            return session.sources[asset_id]
        except KeyError as error:
            raise HTTPException(404, f"Unknown source {asset_id}") from error
    raise HTTPException(404, f"Unknown asset kind {kind}")


def _sam_key(
    session_id: str,
    kind: str,
    asset_id: str,
    revision: int,
    frame_index: int | None = None,
) -> str:
    frame = "image" if frame_index is None else f"frame-{frame_index}"
    return f"{session_id}:{kind}:{asset_id}:{revision}:{frame}"


def _store_upload_sync(upload: UploadFile, path: Path, limit: int) -> None:
    written = 0
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            while chunk := upload.file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, "Upload is too large")
                output.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _queue_embedding(
    runtime: Runtime,
    session: Session,
    kind: str,
    asset_id: str,
    revision: int,
    frame_index: int | None = None,
) -> None:
    # ZeroGPU only exposes a real CUDA device while a @spaces.GPU function is
    # active. In Gradio transport mode SAM prepares its embedding lazily inside
    # the decorated prediction call instead of starting background CUDA work.
    if not runtime.eager_embeddings:
        with session.lock:
            try:
                asset = _asset(session, kind, asset_id)
            except HTTPException:
                return
            if asset.revision == revision and (
                kind == "target" or asset.current_frame == frame_index
            ):
                asset.sam_status = "ready"
                asset.sam_error = None
        return

    key = _sam_key(session.id, kind, asset_id, revision, frame_index)

    if runtime.sam.has(key):
        with session.lock:
            try:
                asset = _asset(session, kind, asset_id)
            except HTTPException:
                return
            if asset.revision == revision and (
                kind == "target" or asset.current_frame == frame_index
            ):
                asset.sam_status = "ready"
                asset.sam_error = None
        return

    def prepare():
        with session.lock:
            try:
                current = _asset(session, kind, asset_id)
            except HTTPException:
                return None
            if (
                session.closed
                or current.revision != revision
                or (kind == "source" and current.current_frame != frame_index)
            ):
                return None
            image = current.image if kind == "target" else current.frame
        try:
            runtime.sam.prepare(key, image)
        except Exception as error:
            with session.lock:
                try:
                    asset = _asset(session, kind, asset_id)
                except HTTPException:
                    return None
                if asset.revision == revision and (
                    kind == "target" or asset.current_frame == frame_index
                ):
                    asset.sam_status = "error"
                    asset.sam_error = str(error)
            raise
        keep = False
        with session.lock:
            try:
                asset = _asset(session, kind, asset_id)
            except HTTPException:
                asset = None
            if (
                asset is not None
                and asset.revision == revision
                and (kind == "target" or asset.current_frame == frame_index)
            ):
                asset.sam_status = "ready"
                asset.sam_error = None
                keep = True
        if not keep:
            runtime.sam.release(key)
        return None

    try:
        runtime.scheduler.submit(
            prepare,
            priority=20,
            key=f"sam:{key}:embedding",
            owner=session.id,
            label=f"Prepare {kind} mask features",
        )
    except QueueFull as error:
        with session.lock:
            try:
                asset = _asset(session, kind, asset_id)
            except HTTPException:
                return
            if asset.revision == revision:
                asset.sam_status = "error"
                asset.sam_error = str(error)


def _state(runtime: Runtime, session: Session):
    with session.lock:
        payload = session_payload(session)
    payload["wan_status"] = runtime.wan.status
    payload["defaults"] = {
        "frames": runtime.config.output_frames,
        "width": runtime.config.output_width,
        "height": runtime.config.output_height,
        "fps": runtime.config.output_fps,
        "source_fps": runtime.config.source_fps,
        "minimum_source_frames": 8,
    }
    return payload


def _validate_prompts(request: PromptRequest) -> None:
    if not request.positive and not request.negative and request.box is None:
        raise HTTPException(400, "At least one point or box is required")
    for point in request.positive + request.negative:
        if len(point) != 2 or not all(0 <= float(value) <= 1 for value in point):
            raise HTTPException(400, "Prompt points must be normalized [x,y] pairs")
    if request.box is not None:
        if len(request.box) != 4 or not all(
            0 <= float(value) <= 1 for value in request.box
        ):
            raise HTTPException(400, "Box must be normalized [x0,y0,x1,y1]")
        if request.box[0] >= request.box[2] or request.box[1] >= request.box[3]:
            raise HTTPException(400, "Box must have positive width and height")


def _generation_snapshot(
    runtime: Runtime, session: Session, request: GenerateRequest
) -> GenerationSnapshot:
    target = session.selected_target
    if target is None:
        raise HTTPException(400, "Upload a target image first")
    grouped: dict[str, list[tuple[MaskRecord, MaskRecord]]] = {}
    for target_mask in target.masks.values():
        if target_mask.source_id is None or target_mask.source_mask_id is None:
            continue
        try:
            source_mask = source_mask_lookup(
                session, target_mask.source_id, target_mask.source_mask_id
            )
        except KeyError:
            continue
        grouped.setdefault(target_mask.source_id, []).append((source_mask, target_mask))
    if request.guidance_mode == "base_cfg":
        grouped.clear()
    elif not grouped:
        raise HTTPException(400, "Map at least one source mask to a target mask")

    run_id = new_id("run")
    output_path = session.directory / f"generation_{run_id}.mp4"
    sources = []
    try:
        for index, (source_id, pairs) in enumerate(grouped.items()):
            source = session.sources[source_id]
            selected_frames = source.trim_end - source.trim_start + 1
            if selected_frames < 8:
                raise HTTPException(
                    400,
                    f"{source.name} needs at least 8 selected 8-fps frames",
                )
            if any(
                mask.frame_index is None
                or not source.trim_start <= mask.frame_index <= source.trim_end
                for mask, _ in pairs
            ):
                raise HTTPException(409, "A mapped source mask lies outside its trim")
            private_path = session.directory / (
                f"input_{run_id}_{index}{source.video_path.suffix.lower()}"
            )
            os.link(source.video_path, private_path)
            sources.append(
                SourceSnapshot(
                    video_path=private_path,
                    trim_start=source.trim_start,
                    trim_end=source.trim_end,
                    source_masks=tuple(pair[0].data.copy() for pair in pairs),
                    target_masks=tuple(pair[1].data.copy() for pair in pairs),
                    reference_frames=tuple(pair[0].frame_index for pair in pairs),
                )
            )
    except Exception:
        for source in sources:
            source.video_path.unlink(missing_ok=True)
        raise
    return GenerationSnapshot(
        target_image=target.image.copy(),
        sources=tuple(sources),
        prompt=request.prompt.strip(),
        negative_prompt=(request.negative_prompt or "").strip() or None,
        steps=request.steps,
        seed=request.seed,
        guidance_mode=request.guidance_mode,
        text_guidance_scale=request.text_guidance_scale,
        motion_guidance_scale=request.motion_guidance_scale,
        lora_scale=request.lora_scale,
        output_path=output_path,
    )


def create_app(runtime: Runtime | None = None) -> FastAPI:
    runtime = runtime or Runtime()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        runtime.close()

    application = FastAPI(title="WhatMoves", lifespan=lifespan)
    application.state.runtime = runtime
    application.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")

    @application.get("/", response_class=HTMLResponse)
    async def index():
        return (STATIC_ROOT / "index.html").read_text()

    @application.get("/api/health")
    async def health():
        return {"status": "ok", "wan_status": runtime.wan.status}

    @application.post("/api/sessions")
    async def create_session():
        session = runtime.store.create()
        try:
            await _run_cpu(
                runtime,
                load_examples,
                session,
                ASSETS_ROOT,
                source_fps=runtime.config.source_fps,
                max_sources=runtime.config.max_sources,
                max_targets=runtime.config.max_targets,
                max_image_side=runtime.config.max_image_side,
            )
        except Exception:
            runtime.store.pop(session.id)
            runtime.dispose_session(session)
            raise
        source = next(iter(session.sources.values()), None)
        target = session.selected_target
        if source is not None:
            source.sam_status = "queued"
            _queue_embedding(
                runtime,
                session,
                "source",
                source.id,
                source.revision,
                source.current_frame,
            )
        if target is not None:
            target.sam_status = "queued"
            _queue_embedding(runtime, session, "target", target.id, target.revision)
        return _state(runtime, session)

    @application.get("/api/sessions/{session_id}")
    async def get_state(session_id: str):
        return _state(runtime, _session(runtime, session_id))

    @application.post("/api/sessions/{session_id}/targets")
    async def upload_target(session_id: str, file: UploadFile = File(...)):
        session = _session(runtime, session_id)
        with session.lock:
            if len(session.targets) >= runtime.config.max_targets:
                raise HTTPException(
                    409,
                    f"A session may contain at most {runtime.config.max_targets} targets",
                )
        target_id = new_id("target")
        suffix = Path(file.filename or "target.png").suffix.lower() or ".png"
        path = session.directory / f"{target_id}{suffix}"
        await _run_cpu(
            runtime, _store_upload_sync, file, path, runtime.config.max_upload_bytes
        )
        try:
            image = await _run_cpu(
                runtime, read_image, path, runtime.config.max_image_side
            )
        except Exception as error:
            raise HTTPException(
                400, f"Could not decode target image: {error}"
            ) from error
        finally:
            path.unlink(missing_ok=True)
        with session.lock:
            if session.closed:
                raise HTTPException(410, "The browser session has expired")
            if len(session.targets) >= runtime.config.max_targets:
                raise HTTPException(
                    409,
                    f"A session may contain at most {runtime.config.max_targets} targets",
                )
            session.target_revision += 1
            target = TargetAsset(
                id=target_id,
                name=Path(file.filename or "Target image").name,
                image=image,
                revision=session.target_revision,
            )
            session.targets[target.id] = target
            session.selected_target_id = target.id
            session.touch()
        _queue_embedding(runtime, session, "target", target.id, target.revision)
        return _state(runtime, session)

    @application.put("/api/sessions/{session_id}/targets/{target_id}/selection")
    async def select_target(session_id: str, target_id: str):
        session = _session(runtime, session_id)
        with session.lock:
            target = _asset(session, "target", target_id)
            session.selected_target_id = target.id
            if not runtime.sam.has(
                _sam_key(session.id, "target", target.id, target.revision)
            ):
                target.sam_status = "queued"
            session.touch()
        _queue_embedding(runtime, session, "target", target.id, target.revision)
        return _state(runtime, session)

    @application.delete("/api/sessions/{session_id}/targets/{target_id}")
    async def remove_target(session_id: str, target_id: str):
        session = _session(runtime, session_id)
        with session.lock:
            target = _asset(session, "target", target_id)
            del session.targets[target_id]
            if session.selected_target_id == target_id:
                session.selected_target_id = next(iter(session.targets), None)
            session.touch()
        _release_asset(runtime, session.id, "target", target_id)
        _clear_asset(target)
        return _state(runtime, session)

    @application.post("/api/sessions/{session_id}/sources")
    async def upload_source(session_id: str, file: UploadFile = File(...)):
        session = _session(runtime, session_id)
        with session.lock:
            if len(session.sources) >= runtime.config.max_sources:
                raise HTTPException(
                    409,
                    f"A session may contain at most {runtime.config.max_sources} sources",
                )
        suffix = Path(file.filename or "source.mp4").suffix.lower() or ".mp4"
        source_id = new_id("source")
        upload_path = session.directory / f"upload_{source_id}{suffix}"
        path = session.directory / f"{source_id}.mp4"
        await _run_cpu(
            runtime,
            _store_upload_sync,
            file,
            upload_path,
            runtime.config.max_upload_bytes,
        )
        try:
            original_fps, frame_count, frame = await _run_cpu(
                runtime,
                resample_video,
                upload_path,
                path,
                runtime.config.source_fps,
                runtime.config.max_image_side,
            )
        except Exception as error:
            path.unlink(missing_ok=True)
            raise HTTPException(
                400, f"Could not decode source video: {error}"
            ) from error
        finally:
            upload_path.unlink(missing_ok=True)
        source = SourceAsset(
            id=source_id,
            name=Path(file.filename or "Source video").name,
            video_path=path,
            frame=frame,
            fps=float(runtime.config.source_fps),
            frame_count=frame_count,
            original_fps=original_fps,
            trim_end=min(frame_count - 1, runtime.config.output_frames - 1),
        )
        with session.lock:
            if session.closed:
                path.unlink(missing_ok=True)
                raise HTTPException(410, "The browser session has expired")
            if len(session.sources) >= runtime.config.max_sources:
                path.unlink(missing_ok=True)
                raise HTTPException(
                    409,
                    f"A session may contain at most {runtime.config.max_sources} sources",
                )
            session.sources[source.id] = source
            session.touch()
        _queue_embedding(
            runtime,
            session,
            "source",
            source.id,
            source.revision,
            source.current_frame,
        )
        return _state(runtime, session)

    @application.patch("/api/sessions/{session_id}/sources/{source_id}")
    async def update_source(session_id: str, source_id: str, settings: SourceSettings):
        session = _session(runtime, session_id)
        with session.lock:
            source = _asset(session, "source", source_id)
            trim_start = (
                source.trim_start
                if settings.trim_start is None
                else settings.trim_start
            )
            trim_end = (
                source.trim_end if settings.trim_end is None else settings.trim_end
            )
            if not 0 <= trim_start <= trim_end < source.frame_count:
                raise HTTPException(400, "Invalid source trim interval")
            if source.frame_count >= 8 and trim_end - trim_start + 1 < 8:
                raise HTTPException(400, "Select at least eight 8-fps frames")
            new_frame = (
                source.current_frame
                if settings.current_frame is None
                else settings.current_frame
            )
            new_frame = min(source.frame_count - 1, max(0, new_frame))
            path = source.video_path
            frame_changed = new_frame != source.current_frame
            trim_changed = (
                trim_start != source.trim_start or trim_end != source.trim_end
            )
            trim_requested = (
                settings.trim_start is not None or settings.trim_end is not None
            )
            selection_token = new_id("frame") if frame_changed else None
            if frame_changed:
                source.frame_selection_token = selection_token
        frame = None
        if frame_changed:
            try:
                frame = await _run_cpu(
                    runtime,
                    decode_video_frame,
                    path,
                    new_frame,
                    runtime.config.max_image_side,
                )
            except Exception as error:
                raise HTTPException(400, str(error)) from error
        with session.lock:
            source = _asset(session, "source", source_id)
            if frame_changed and source.frame_selection_token != selection_token:
                raise HTTPException(409, "Superseded by a newer frame selection")
            if trim_requested:
                source.trim_start = trim_start
                source.trim_end = trim_end
            if trim_requested and trim_changed:
                clear_excluded_source_mappings(session, source)
            if frame_changed:
                source.current_frame = new_frame
                source.frame = frame
                source.frame_selection_token = None
                source.draft = None
                source.prediction_token = None
                source.hover_prediction_token = None
                source.sam_status = "queued"
                source.sam_error = None
            if trim_requested and trim_changed:
                session.touch()
            revision = source.revision
        if frame_changed:
            runtime.scheduler.cancel_prefix(
                f"sam:{_sam_prefix(session.id, 'source', source_id)}"
            )
            _queue_embedding(
                runtime,
                session,
                "source",
                source_id,
                revision,
                new_frame,
            )
        return _state(runtime, session)

    @application.delete("/api/sessions/{session_id}/sources/{source_id}")
    async def remove_source(session_id: str, source_id: str):
        session = _session(runtime, session_id)
        with session.lock:
            if source_id not in session.sources:
                raise HTTPException(404, f"Unknown source {source_id}")
            source = session.sources.pop(source_id)
            clear_source_mappings(session, source_id)
            session.touch()
        _release_asset(runtime, session.id, "source", source_id)
        _clear_asset(source)
        source.video_path.unlink(missing_ok=True)
        return _state(runtime, session)

    @application.post("/api/sessions/{session_id}/assets/{kind}/{asset_id}/predict")
    async def predict_mask(
        session_id: str, kind: str, asset_id: str, request: PromptRequest
    ):
        _validate_prompts(request)
        session = _session(runtime, session_id)
        with session.lock:
            asset = _asset(session, kind, asset_id)
            if asset.revision != request.asset_revision:
                raise HTTPException(409, "The underlying media has changed")
            frame_index = asset.current_frame if kind == "source" else None
            if kind == "source" and request.frame_index != frame_index:
                raise HTTPException(409, "The selected source frame has changed")
            key = _sam_key(
                session.id,
                kind,
                asset_id,
                asset.revision,
                frame_index,
            )
            prediction_token = new_id("prediction")
            token_name = (
                "hover_prediction_token" if request.transient else "prediction_token"
            )
            if not request.transient:
                asset.draft = None
                asset.hover_prediction_token = None
            setattr(asset, token_name, prediction_token)

        def run_prediction():
            with session.lock:
                try:
                    current = _asset(session, kind, asset_id)
                except HTTPException as error:
                    raise Superseded("Media removed during mask prediction") from error
                if (
                    session.closed
                    or current.revision != request.asset_revision
                    or (kind == "source" and current.current_frame != frame_index)
                    or getattr(current, token_name) != prediction_token
                ):
                    raise Superseded("Media changed during mask prediction")
                image = current.image if kind == "target" else current.frame
            mask, score = runtime.sam.predict(
                key,
                image,
                request.positive,
                request.negative,
                request.box,
            )
            with session.lock:
                try:
                    current = _asset(session, kind, asset_id)
                except HTTPException as error:
                    raise Superseded("Media removed during mask prediction") from error
                if (
                    current.revision != request.asset_revision
                    or (kind == "source" and current.current_frame != frame_index)
                    or getattr(current, token_name) != prediction_token
                ):
                    raise Superseded("Media changed during mask prediction")
                current.sam_status = "ready"
                current.sam_error = None
                if not request.transient:
                    current.draft = Draft(
                        mask,
                        request.asset_revision,
                        request.prompt_revision,
                        frame_index,
                    )
            return mask, score

        try:
            if runtime.config.device == "cpu":
                mask, score = run_prediction()
            else:
                try:
                    if not request.transient:
                        runtime.scheduler.cancel_prefix(
                            f"sam:{_sam_prefix(session.id, kind, asset_id)}"
                            "prediction:hover"
                        )
                    _, future = runtime.scheduler.submit(
                        run_prediction,
                        priority=30 if request.transient else 0,
                        key=(
                            f"sam:{_sam_prefix(session.id, kind, asset_id)}"
                            f"prediction:{'hover' if request.transient else 'commit'}"
                        ),
                        owner=session.id,
                        label="Update mask draft",
                    )
                except QueueFull as error:
                    raise HTTPException(503, str(error)) from error
                mask, score = await asyncio.wrap_future(future)
        except Superseded as error:
            raise HTTPException(409, str(error)) from error
        except Exception as error:
            raise HTTPException(500, f"SAM prediction failed: {error}") from error
        return {
            "asset_revision": request.asset_revision,
            "prompt_revision": request.prompt_revision,
            "score": score,
            "transient": request.transient,
            "mask": base64.b64encode(mask_png_bytes(mask)).decode("ascii"),
        }

    @application.delete("/api/sessions/{session_id}/assets/{kind}/{asset_id}/draft")
    async def clear_draft(
        session_id: str,
        kind: str,
        asset_id: str,
        revision: int,
    ):
        session = _session(runtime, session_id)
        runtime.scheduler.cancel_prefix(
            f"sam:{_sam_prefix(session.id, kind, asset_id)}prediction"
        )
        with session.lock:
            asset = _asset(session, kind, asset_id)
            if asset.revision == revision:
                asset.draft = None
                asset.prediction_token = None
                asset.hover_prediction_token = None
        return Response(status_code=204)

    @application.post("/api/sessions/{session_id}/assets/{kind}/{asset_id}/masks")
    async def commit_mask(
        session_id: str, kind: str, asset_id: str, request: CommitRequest
    ):
        session = _session(runtime, session_id)
        with session.lock:
            asset = _asset(session, kind, asset_id)
            draft = asset.draft
            if (
                draft is None
                or draft.asset_revision != request.asset_revision
                or draft.prompt_revision != request.prompt_revision
                or draft.frame_index != request.frame_index
            ):
                raise HTTPException(409, "The mask draft is no longer current")
            if not draft.data.any():
                raise HTTPException(400, "Cannot add an empty mask")
            mask_id = new_id("mask")
            color = session.allocate_color() if kind == "source" else None
            asset.masks[mask_id] = MaskRecord(
                mask_id,
                draft.data.copy(),
                color=color,
                frame_index=draft.frame_index if kind == "source" else None,
            )
            asset.draft = None
            asset.prediction_token = None
            asset.hover_prediction_token = None
            asset.mask_revision += 1
            session.touch()
        return _state(runtime, session)

    @application.delete(
        "/api/sessions/{session_id}/assets/{kind}/{asset_id}/masks/{mask_id}"
    )
    async def remove_mask(session_id: str, kind: str, asset_id: str, mask_id: str):
        session = _session(runtime, session_id)
        with session.lock:
            asset = _asset(session, kind, asset_id)
            if mask_id not in asset.masks:
                raise HTTPException(404, f"Unknown mask {mask_id}")
            del asset.masks[mask_id]
            asset.mask_revision += 1
            if kind == "source":
                clear_source_mappings(session, asset_id, mask_id)
            session.touch()
        return _state(runtime, session)

    @application.put(
        "/api/sessions/{session_id}/targets/{target_id}/masks/{target_mask_id}/mapping"
    )
    async def map_mask(
        session_id: str,
        target_id: str,
        target_mask_id: str,
        request: MappingRequest,
    ):
        session = _session(runtime, session_id)
        with session.lock:
            target = _asset(session, "target", target_id)
            if target_mask_id not in target.masks:
                raise HTTPException(404, f"Unknown target mask {target_mask_id}")
            target_mask = target.masks[target_mask_id]
            if request.source_id is None and request.source_mask_id is None:
                target_mask.source_id = None
                target_mask.source_mask_id = None
            elif request.source_id is None or request.source_mask_id is None:
                raise HTTPException(400, "Both source and source mask are required")
            else:
                try:
                    source_mask_lookup(
                        session, request.source_id, request.source_mask_id
                    )
                except KeyError as error:
                    raise HTTPException(404, str(error)) from error
                target_mask.source_id = request.source_id
                target_mask.source_mask_id = request.source_mask_id
            target.mask_revision += 1
            session.touch()
        return _state(runtime, session)

    @application.post("/api/sessions/{session_id}/generate")
    async def generate(session_id: str, request: GenerateRequest):
        session = _session(runtime, session_id)
        with session.lock:
            if session.closed:
                raise HTTPException(410, "The browser session has expired")
            if session.active_generation is not None:
                active = runtime.scheduler.get(session.active_generation)
                if active is not None and active.status in {"queued", "running"}:
                    raise HTTPException(409, "This session is already generating")
                session.active_generation = None
            snapshot = _generation_snapshot(runtime, session, request)
            input_revision = session.input_revision
            previous_failure = session.failed_generation
            session.failed_generation = None
            if previous_failure is not None:
                runtime.scheduler.discard(previous_failure)
            output_path = snapshot.output_path
            private_inputs = tuple(source.video_path for source in snapshot.sources)
            task_id = new_id("task")

            def report_progress(stage, current=None, total=None):
                runtime.scheduler.update_progress(
                    task_id,
                    stage,
                    current,
                    total,
                )

            def run_generation():
                try:
                    result = runtime.wan.generate(snapshot, report_progress)
                    result["video_url"] = (
                        f"/api/sessions/{session.id}/outputs/{result['output_name']}"
                    )
                    with session.lock:
                        result["inputs_changed"] = (
                            session.input_revision != input_revision
                        )
                    return result
                finally:
                    for path in private_inputs:
                        path.unlink(missing_ok=True)

            try:
                task_id, future = runtime.scheduler.submit(
                    run_generation,
                    priority=10,
                    key=f"generation:{session.id}",
                    owner=session.id,
                    label="Generate video",
                    retain_record=True,
                    task_id=task_id,
                )
            except (QueueFull, RuntimeError) as error:
                for path in private_inputs:
                    path.unlink(missing_ok=True)
                output_path.unlink(missing_ok=True)
                raise HTTPException(503, str(error)) from error
            session.active_generation = task_id

        def generation_finished(done):
            try:
                done.result()
            except Exception:
                succeeded = False
            else:
                succeeded = True

            old_task = None
            old_output = None
            with session.lock:
                if session.active_generation == task_id:
                    session.active_generation = None
                closed = session.closed
                if succeeded and not closed:
                    old_task = session.latest_generation
                    old_output = session.latest_output
                    session.latest_generation = task_id
                    session.latest_output = output_path
                elif not closed:
                    old_task = session.failed_generation
                    session.failed_generation = task_id
            if old_task is not None and old_task != task_id:
                runtime.scheduler.discard(old_task)
            if old_output is not None and old_output != output_path:
                old_output.unlink(missing_ok=True)
            if not succeeded or closed:
                output_path.unlink(missing_ok=True)
            if closed:
                runtime.scheduler.discard(task_id)
                _remove_session_directory(
                    runtime.config.runtime_root,
                    session.directory,
                )

        future.add_done_callback(generation_finished)
        return {"task_id": task_id}

    @application.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        session = runtime.store.pop(session_id)
        if session is None:
            raise HTTPException(404, f"Unknown session {session_id}")
        runtime.dispose_session(session)
        return Response(status_code=204)

    @application.get("/api/tasks/{task_id}")
    async def task_status(task_id: str):
        record = runtime.scheduler.get(task_id)
        if record is None:
            raise HTTPException(404, f"Unknown task {task_id}")
        return record.as_dict()

    @application.get("/api/sessions/{session_id}/targets/{target_id}/image")
    async def target_image(session_id: str, target_id: str):
        session = _session(runtime, session_id)
        with session.lock:
            content = png_bytes(_asset(session, "target", target_id).image)
        return Response(
            content,
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @application.get("/api/sessions/{session_id}/targets/{target_id}/thumbnail.png")
    async def target_thumbnail(session_id: str, target_id: str):
        session = _session(runtime, session_id)
        with session.lock:
            target = _asset(session, "target", target_id)
            masks = []
            for mask in target.masks.values():
                color = "#a8afb9"
                if mask.source_id and mask.source_mask_id:
                    try:
                        color = source_mask_lookup(
                            session, mask.source_id, mask.source_mask_id
                        ).color
                    except KeyError:
                        pass
                masks.append((mask.data, color))
            image = composite_masks(target.image, masks, max_size=(180, 120))
        return Response(
            png_bytes(image),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @application.get(
        "/api/sessions/{session_id}/targets/{target_id}/masks/{mask_id}.png"
    )
    async def target_mask_image(session_id: str, target_id: str, mask_id: str):
        session = _session(runtime, session_id)
        with session.lock:
            target = _asset(session, "target", target_id)
            if mask_id not in target.masks:
                raise HTTPException(404, f"Unknown target mask {mask_id}")
            content = mask_png_bytes(target.masks[mask_id].data)
        return Response(
            content,
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @application.get("/api/sessions/{session_id}/sources/{source_id}/frame.png")
    async def source_frame(session_id: str, source_id: str, frame: int | None = None):
        session = _session(runtime, session_id)
        with session.lock:
            source = _asset(session, "source", source_id)
            requested = source.current_frame if frame is None else frame
            if not 0 <= requested < source.frame_count:
                raise HTTPException(404, f"Unknown frame {requested}")
            path = source.video_path
            cached = source.frame.copy() if requested == source.current_frame else None
        image = cached
        if image is None:
            image = await _run_cpu(
                runtime,
                decode_video_frame,
                path,
                requested,
                runtime.config.max_image_side,
            )
        return Response(
            png_bytes(image),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @application.get("/api/sessions/{session_id}/sources/{source_id}/video")
    async def source_video(session_id: str, source_id: str):
        session = _session(runtime, session_id)
        with session.lock:
            path = _asset(session, "source", source_id).video_path
        if not path.is_file():
            raise HTTPException(404, "Source video is unavailable")
        return FileResponse(
            path,
            media_type="video/mp4",
            headers={"Cache-Control": "no-store", "Accept-Ranges": "bytes"},
        )

    @application.get(
        "/api/sessions/{session_id}/sources/{source_id}/frames/{frame_index}.jpg"
    )
    async def source_timeline_frame(session_id: str, source_id: str, frame_index: int):
        session = _session(runtime, session_id)
        with session.lock:
            source = _asset(session, "source", source_id)
            if not 0 <= frame_index < source.frame_count:
                raise HTTPException(404, f"Unknown frame {frame_index}")
            path = source.video_path
        image = await _run_cpu(
            runtime,
            decode_video_frame,
            path,
            frame_index,
            256,
        )
        return Response(
            jpeg_bytes(image),
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @application.get("/api/sessions/{session_id}/sources/{source_id}/thumbnail.png")
    async def source_thumbnail(session_id: str, source_id: str):
        session = _session(runtime, session_id)
        with session.lock:
            source = _asset(session, "source", source_id)
            frame_index = source.trim_start
            path = source.video_path
            masks = [
                (mask.data.copy(), mask.color)
                for mask in source.masks.values()
                if mask.frame_index == frame_index
            ]
        frame = await _run_cpu(runtime, decode_video_frame, path, frame_index, 320)
        image = composite_masks(frame, masks, max_size=(180, 120))
        return Response(
            png_bytes(image),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @application.get(
        "/api/sessions/{session_id}/sources/{source_id}/masks/{mask_id}.png"
    )
    async def source_mask_image(session_id: str, source_id: str, mask_id: str):
        session = _session(runtime, session_id)
        with session.lock:
            source = _asset(session, "source", source_id)
            if mask_id not in source.masks:
                raise HTTPException(404, f"Unknown mask {mask_id}")
            content = mask_png_bytes(source.masks[mask_id].data)
        return Response(
            content,
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @application.get(
        "/api/sessions/{session_id}/sources/{source_id}/masks/{mask_id}/preview.png"
    )
    async def source_mask_preview(session_id: str, source_id: str, mask_id: str):
        session = _session(runtime, session_id)
        with session.lock:
            source = _asset(session, "source", source_id)
            if mask_id not in source.masks:
                raise HTTPException(404, f"Unknown mask {mask_id}")
            mask = source.masks[mask_id]
            frame_index = mask.frame_index
            path = source.video_path
            mask_data = mask.data.copy()
            color = mask.color
        if frame_index is None:
            raise HTTPException(409, "Source mask has no reference frame")
        frame = await _run_cpu(runtime, decode_video_frame, path, frame_index, 320)
        image = composite_masks(frame, [(mask_data, color)], max_size=(160, 100))
        return Response(
            png_bytes(image),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @application.get("/api/sessions/{session_id}/outputs/{name}")
    async def output_video(session_id: str, name: str):
        session = _session(runtime, session_id)
        safe_name = Path(name).name
        with session.lock:
            path = session.latest_output
        if safe_name != name or path is None or path.name != name or not path.is_file():
            raise HTTPException(404, "Unknown output")
        return FileResponse(
            path,
            media_type="video/mp4",
            headers={"Cache-Control": "no-store"},
        )

    return application
