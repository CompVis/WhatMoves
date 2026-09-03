"""Public WhatMoves inference API."""

from .model import WhatMoves, load_model

__all__ = ["WhatMoves", "load_model", "WanMotionTransfer", "load_wan_model"]


def __getattr__(name: str):
    if name in {"WanMotionTransfer", "load_wan_model"}:
        from .wan import WanMotionTransfer, load_wan_model

        return {
            "WanMotionTransfer": WanMotionTransfer,
            "load_wan_model": load_wan_model,
        }[name]
    raise AttributeError(name)
