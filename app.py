"""Hugging Face Spaces entrypoint for the WhatMoves demo."""

from __future__ import annotations

import os

import uvicorn

from app.gradio_app import create_gradio_application


application = create_gradio_application()


if __name__ == "__main__":
    uvicorn.run(
        application,
        host=os.environ.get("WHATMOVES_APP_HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", os.environ.get("WHATMOVES_APP_PORT", "7860"))),
    )
