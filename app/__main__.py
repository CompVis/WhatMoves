"""Run the application with ``python -m app``."""

from __future__ import annotations

import argparse
import os

import uvicorn

from .gradio_app import create_gradio_application


def main() -> None:
    parser = argparse.ArgumentParser(description="WhatMoves interactive demo")
    parser.add_argument(
        "--host", default=os.environ.get("WHATMOVES_APP_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("WHATMOVES_APP_PORT", "7860")),
    )
    parser.add_argument(
        "--reload", action="store_true", help="Reload after source changes"
    )
    args = parser.parse_args()
    uvicorn.run(
        create_gradio_application(),
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
