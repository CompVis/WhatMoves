"""Gradio transport for the unchanged WhatMoves browser experience.

The public page and read-only media routes remain on the FastAPI application.
All browser mutations and GPU jobs travel through named Gradio endpoints, so
Hugging Face ZeroGPU can allocate hardware around the actual CUDA lifetime.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any

# Gradio reads its temp root during import. Use a process-private directory so
# multi-user machines never collide on a shared /tmp/gradio directory.
if "GRADIO_TEMP_DIR" not in os.environ:
    _runtime_temp = os.environ.get("WHATMOVES_APP_TMP_DIR")
    _gradio_temp = (
        Path(_runtime_temp) / "gradio"
        if _runtime_temp
        else Path("/tmp") / f"whatmoves-gradio-{os.getpid()}"
    )
    _gradio_temp.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.environ["GRADIO_TEMP_DIR"] = str(_gradio_temp)
os.environ.setdefault("MPLBACKEND", "Agg")

import gradio as gr
import httpx
import spaces

from .server import Runtime, create_app


def _upload_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        return Path(value)
    if isinstance(value, dict):
        candidate = value.get("path") or value.get("name")
        return Path(candidate) if candidate else None
    candidate = getattr(value, "name", None)
    return Path(candidate) if candidate else None


class GradioBridge:
    def __init__(self, service, runtime: Runtime) -> None:
        self.service = service
        self.runtime = runtime

    async def _request(
        self,
        method: str,
        path: str,
        body: str | None,
        upload: Any,
    ) -> dict[str, Any]:
        method = str(method or "GET").upper()
        path = str(path or "")
        if not path.startswith("/api/") or "://" in path:
            return {"ok": False, "status": 400, "data": "Invalid API path"}

        kwargs: dict[str, Any] = {}
        upload_path = _upload_path(upload)
        handle = None
        try:
            if upload_path is not None:
                handle = upload_path.open("rb")
                kwargs["files"] = {
                    "file": (upload_path.name, handle, "application/octet-stream")
                }
            elif body:
                kwargs["content"] = body.encode("utf-8")
                kwargs["headers"] = {"content-type": "application/json"}

            transport = httpx.ASGITransport(app=self.service)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://whatmoves.internal",
            ) as client:
                response = await client.request(method, path, **kwargs)
        except Exception as error:
            return {"ok": False, "status": 500, "data": str(error)}
        finally:
            if handle is not None:
                handle.close()

        if not response.content:
            data: Any = None
        else:
            try:
                data = response.json()
            except ValueError:
                data = response.text
        return {"ok": response.is_success, "status": response.status_code, "data": data}

    def request(
        self,
        method: str,
        path: str,
        body: str | None = None,
        upload: Any = None,
    ) -> dict[str, Any]:
        return asyncio.run(self._request(method, path, body, upload))

    def request_json(
        self,
        method: str,
        path: str,
        body: str | None = None,
        upload: Any = None,
    ) -> str:
        return json.dumps(self.request(method, path, body, upload), separators=(",", ":"))

    def generate(
        self,
        method: str,
        path: str,
        body: str | None = None,
        upload: Any = None,
        progress=gr.Progress(),
    ) -> dict[str, Any]:
        result = self.request(method, path, body, upload)
        if not result["ok"]:
            return result
        task_id = (result.get("data") or {}).get("task_id")
        if not task_id:
            return result

        while True:
            record = self.runtime.scheduler.get(task_id)
            if record is None:
                return {"ok": False, "status": 404, "data": "Generation task disappeared"}
            if record.progress_total and record.progress_current is not None:
                progress(
                    (record.progress_current, record.progress_total),
                    desc=record.stage,
                )
            else:
                progress(0, desc=record.stage)
            if record.status not in {"queued", "running"}:
                return result
            time.sleep(0.2)

    def generate_json(
        self,
        method: str,
        path: str,
        body: str | None = None,
        upload: Any = None,
        progress=gr.Progress(),
    ) -> str:
        return json.dumps(
            self.generate(method, path, body, upload, progress),
            separators=(",", ":"),
        )


def create_gradio_application():
    runtime = Runtime(eager_embeddings=False)
    service = create_app(runtime)
    bridge = GradioBridge(service, runtime)

    @spaces.GPU(size="large", duration=120)
    def sam_request(method, path, body=None, upload=None):
        return bridge.request_json(method, path, body, upload)

    @spaces.GPU(size="xlarge", duration=300)
    def generation_request(method, path, body=None, upload=None, progress=gr.Progress()):
        return bridge.generate_json(method, path, body, upload, progress)

    with gr.Blocks(title="What Moves?") as transport:
        method = gr.Textbox(visible=False)
        path = gr.Textbox(visible=False)
        body = gr.Textbox(visible=False)
        upload = gr.File(type="filepath", visible=False)
        output = gr.Textbox(visible=False)
        cpu_trigger = gr.Button(visible=False)
        sam_trigger = gr.Button(visible=False)
        generation_trigger = gr.Button(visible=False)

        inputs = [method, path, body, upload]
        cpu_trigger.click(
            bridge.request_json,
            inputs=inputs,
            outputs=output,
            api_name="api",
            queue=False,
        )
        sam_trigger.click(
            sam_request,
            inputs=inputs,
            outputs=output,
            api_name="sam_api",
            concurrency_limit=1,
        )
        generation_trigger.click(
            generation_request,
            inputs=inputs,
            outputs=output,
            api_name="generation_api",
            concurrency_limit=1,
        )

    transport.queue()
    return gr.mount_gradio_app(service, transport, path="/gradio")
