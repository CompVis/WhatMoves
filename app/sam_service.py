"""Lazy shared SAM2 model with one cached predictor per uploaded asset."""

from __future__ import annotations

from contextlib import nullcontext
from collections import OrderedDict
import threading

import numpy as np


class SamService:
    def __init__(
        self, model_id: str, device: str = "cuda", max_predictors: int = 8
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self.max_predictors = max(1, int(max_predictors))
        self._predictors: OrderedDict[str, object] = OrderedDict()
        self._lock = threading.RLock()

    def has(self, key: str) -> bool:
        with self._lock:
            present = key in self._predictors
            if present:
                self._predictors.move_to_end(key)
            return present

    def prepare(self, key: str, image: np.ndarray) -> None:
        with self._lock:
            if key in self._predictors:
                return
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            if self._model is None:
                predictor = SAM2ImagePredictor.from_pretrained(
                    self.model_id, device=self.device
                )
                self._model = predictor.model
            else:
                predictor = SAM2ImagePredictor(self._model)
            with self._autocast():
                predictor.set_image(np.asarray(image, dtype=np.uint8))
            self._predictors[key] = predictor
            self._predictors.move_to_end(key)
            while len(self._predictors) > self.max_predictors:
                _, evicted = self._predictors.popitem(last=False)
                evicted.reset_predictor()

    def predict(
        self,
        key: str,
        image: np.ndarray,
        positive: list[list[float]],
        negative: list[list[float]],
        box: list[float] | None,
    ) -> tuple[np.ndarray, float]:
        with self._lock:
            self.prepare(key, image)
            predictor = self._predictors[key]
            self._predictors.move_to_end(key)
            points = positive + negative
            point_coords = None
            point_labels = None
            if points:
                height, width = image.shape[:2]
                point_coords = np.asarray(
                    [[point[0] * width, point[1] * height] for point in points],
                    dtype=np.float32,
                )
                point_labels = np.asarray(
                    [1] * len(positive) + [0] * len(negative), dtype=np.int32
                )
            pixel_box = None
            if box is not None:
                height, width = image.shape[:2]
                pixel_box = np.asarray(
                    [
                        box[0] * width,
                        box[1] * height,
                        box[2] * width,
                        box[3] * height,
                    ],
                    dtype=np.float32,
                )
            with self._autocast():
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=pixel_box,
                    multimask_output=True,
                )
            best = int(np.argmax(scores))
            return np.asarray(masks[best], dtype=bool), float(scores[best])

    def release(self, prefix: str) -> None:
        with self._lock:
            for key in [key for key in self._predictors if key.startswith(prefix)]:
                predictor = self._predictors.pop(key)
                predictor.reset_predictor()

    def close(self) -> None:
        with self._lock:
            for predictor in self._predictors.values():
                predictor.reset_predictor()
            self._predictors.clear()
            self._model = None

    def _autocast(self):
        if not self.device.startswith("cuda"):
            return nullcontext()
        import torch

        return torch.autocast("cuda", dtype=torch.bfloat16)
