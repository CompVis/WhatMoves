# Vendored Wan2.2 runtime

This directory contains the minimal inference runtime needed by the WhatMoves
Wan I2V-A14B adapter. The source files were reduced from the official
[Wan2.2 repository](https://github.com/Wan-Video/Wan2.2) at
commit `388807310646ed5f318a99f8e8d9ad28c5b65373`.

Wan2.2 is distributed under the Apache License 2.0; see `LICENSE.txt` in this
directory. Changes are limited to omitting unused pipelines and model variants,
removing unused imports, and using PyTorch's restricted tensor-only checkpoint
loader and current autocast API. Attention is routed through Wan's existing
PyTorch-SDPA fallback when FlashAttention is unavailable. Modified files are
marked in their headers.
