"""Small image, mask, and video I/O helpers."""

from __future__ import annotations

from fractions import Fraction
from io import BytesIO
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageOps


def read_image(path: Path, max_side: int = 1600) -> np.ndarray:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.asarray(image).copy()


def png_bytes(image: np.ndarray) -> bytes:
    buffer = BytesIO()
    mode = "L" if image.ndim == 2 else "RGB"
    Image.fromarray(image, mode=mode).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def jpeg_bytes(image: np.ndarray, quality: int = 82) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(
        buffer,
        format="JPEG",
        quality=int(quality),
        optimize=True,
    )
    return buffer.getvalue()


def mask_png_bytes(mask: np.ndarray) -> bytes:
    return png_bytes(np.asarray(mask, dtype=np.uint8) * 255)


def _rate(stream) -> float:
    value = stream.average_rate or stream.guessed_rate
    try:
        rate = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        rate = 0.0
    return rate if np.isfinite(rate) and rate > 0 else 8.0


def probe_video(path: Path) -> tuple[float, int | None]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = _rate(stream)
        count = int(stream.frames) if stream.frames else None
        if (
            count is None
            and stream.duration is not None
            and stream.time_base is not None
        ):
            count = max(1, int(round(float(stream.duration * stream.time_base) * fps)))
        return fps, count


def decode_video_frame(path: Path, index: int, max_side: int = 1600) -> np.ndarray:
    index = max(0, int(index))
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = _rate(stream)
        if index and stream.time_base is not None:
            timestamp = int((index / fps) / float(stream.time_base))
            container.seek(timestamp, stream=stream, any_frame=False, backward=True)
        closest = None
        closest_distance = float("inf")
        for decoded_index, frame in enumerate(container.decode(stream)):
            if frame.pts is not None and stream.time_base is not None:
                frame_index = round(float(frame.pts * stream.time_base) * fps)
            else:
                frame_index = decoded_index
            distance = abs(frame_index - index)
            if distance < closest_distance:
                closest = frame
                closest_distance = distance
            if frame_index < index:
                continue
            frame = closest
            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            if max(image.size) > max_side:
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            return np.asarray(image).copy()
    raise ValueError(f"Video has no frame {index}")


def _resized_rgb(array: np.ndarray, max_side: int, size=None) -> np.ndarray:
    image = Image.fromarray(array, mode="RGB")
    if size is None:
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        width = max(2, image.width - image.width % 2)
        height = max(2, image.height - image.height % 2)
        size = (width, height)
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image).copy()


def resample_video(
    source_path: Path,
    output_path: Path,
    fps: int = 8,
    max_side: int = 1600,
) -> tuple[float, int, np.ndarray]:
    """Create the exact, nearest-timestamp source sequence shown by the app."""
    target_fps = int(fps)
    if target_fps < 1:
        raise ValueError("Target FPS must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_output = None
    output_count = 0
    try:
        with av.open(str(source_path)) as input_container:
            input_stream = input_container.streams.video[0]
            original_fps = _rate(input_stream)
            with av.open(str(output_path), mode="w") as output_container:
                try:
                    output_stream = output_container.add_stream(
                        "libx264", rate=Fraction(target_fps, 1)
                    )
                except av.codec.codec.UnknownCodecError:
                    output_stream = output_container.add_stream(
                        "mpeg4", rate=Fraction(target_fps, 1)
                    )
                output_stream.pix_fmt = "yuv420p"
                output_stream.options = {"crf": "18", "preset": "veryfast"}
                size = None
                previous = None
                previous_time = 0.0
                first_time = None
                next_time = 0.0
                source_index = 0

                def emit(array):
                    nonlocal first_output, output_count, size
                    resized = _resized_rgb(array, max_side, size)
                    if size is None:
                        size = (int(resized.shape[1]), int(resized.shape[0]))
                        output_stream.width, output_stream.height = size
                    if first_output is None:
                        first_output = resized.copy()
                    encoded = av.VideoFrame.from_ndarray(resized, format="rgb24")
                    for packet in output_stream.encode(encoded):
                        output_container.mux(packet)
                    output_count += 1

                for frame in input_container.decode(input_stream):
                    raw_time = frame.time
                    if raw_time is None:
                        raw_time = source_index / original_fps
                    raw_time = float(raw_time)
                    if first_time is None:
                        first_time = raw_time
                    timestamp = max(0.0, raw_time - first_time)
                    if previous is not None:
                        timestamp = max(previous_time, timestamp)
                    array = frame.to_ndarray(format="rgb24")
                    if previous is not None:
                        boundary = (previous_time + timestamp) / 2
                        while next_time < boundary:
                            emit(previous)
                            next_time += 1 / target_fps
                    previous = array
                    previous_time = timestamp
                    source_index += 1
                if previous is None:
                    raise ValueError("Video contains no decodable frames")
                end_time = previous_time + 0.5 / original_fps
                while next_time < end_time or output_count == 0:
                    emit(previous)
                    next_time += 1 / target_fps
                for packet in output_stream.encode():
                    output_container.mux(packet)
        output_path.chmod(0o600)
        return original_fps, output_count, first_output
    except Exception:
        output_path.unlink(missing_ok=True)
        raise


def decode_video_range(path: Path, start: int, end: int) -> np.ndarray:
    start = max(0, int(start))
    end = int(end)
    if end < start:
        raise ValueError("Video range end precedes its start")
    wanted = end - start + 1
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index < start:
                continue
            if index > end:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    if len(frames) != wanted:
        raise ValueError(
            f"Source contains only {len(frames)} of frames {start} through {end}"
        )
    return np.stack(frames)


def decode_sampled_video(path: Path, start: int, stride: int, count: int) -> np.ndarray:
    start = max(0, int(start))
    stride = max(1, int(stride))
    wanted = {start + stride * index: index for index in range(int(count))}
    frames: list[np.ndarray | None] = [None] * int(count)
    final_index = max(wanted)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            slot = wanted.get(frame_index)
            if slot is not None:
                frames[slot] = frame.to_ndarray(format="rgb24")
            if frame_index >= final_index:
                break
    available = sum(frame is not None for frame in frames)
    if available != count:
        raise ValueError(
            f"Source contains only {available} of the requested {count} sampled frames "
            f"(start={start}, stride={stride})"
        )
    return np.stack(frames)


def _hex_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def composite_masks(
    image: np.ndarray,
    masks: list[tuple[np.ndarray, str]],
    *,
    max_size: tuple[int, int] | None = None,
) -> np.ndarray:
    output = image.astype(np.float32).copy()
    for mask, color in masks:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != output.shape[:2]:
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8), mode="L").resize(
                    (output.shape[1], output.shape[0]),
                    Image.Resampling.NEAREST,
                ),
                dtype=bool,
            )
        rgb = np.asarray(_hex_rgb(color), dtype=np.float32)
        output[mask] = output[mask] * 0.67 + rgb * 0.33
        edge = mask & ~(
            np.roll(mask, 1, 0)
            & np.roll(mask, -1, 0)
            & np.roll(mask, 1, 1)
            & np.roll(mask, -1, 1)
        )
        output[edge] = rgb
    rendered = Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))
    if max_size is not None:
        rendered.thumbnail(max_size, Image.Resampling.LANCZOS)
    return np.asarray(rendered).copy()


def write_video(path: Path, frames: np.ndarray, fps: int = 8) -> None:
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Video frames must be [T,H,W,3] uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        try:
            stream = container.add_stream("libx264", rate=Fraction(int(fps), 1))
        except av.codec.codec.UnknownCodecError:
            stream = container.add_stream("mpeg4", rate=Fraction(int(fps), 1))
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for array in frames:
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    path.chmod(0o600)
