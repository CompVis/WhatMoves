# Third-party notices

WhatMoves contains reduced or vendored components from the projects below.
The root MIT license applies to original WhatMoves code; it does not replace
the licenses or copyright notices of these components.

## DINOv2

`what_moves/dinov2.py` is a reduced and modified implementation of
[DINOv2](https://github.com/facebookresearch/dinov2) by Meta Platforms, Inc.,
based on commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.

DINOv2 code and the pretrained DINOv2 parameters incorporated into the
WhatMoves checkpoint are licensed under the Apache License 2.0. A copy of that
license is included at `wan/LICENSE.txt`. The reduced implementation identifies
its modifications in the source header.

## Wan2.2

The `wan/` package is a reduced runtime from
[Wan2.2](https://github.com/Wan-Video/Wan2.2) by the Alibaba Wan Team, based on
commit `388807310646ed5f318a99f8e8d9ad28c5b65373`. It is licensed under the
Apache License 2.0; see `wan/LICENSE.txt`. Every modified vendored source file
carries an explicit modification notice.

The released checkpoints combine original WhatMoves parameters with components
derived from Apache-2.0-licensed DINOv2 or designed to operate as deltas over
Apache-2.0-licensed Wan2.2. See the WhatMoves Hugging Face model card for the
per-file breakdown. The repository's root MIT license applies only to original
WhatMoves material; all upstream notices and conditions remain in force.

## Hugging Face Transformers and Diffusers

The upstream Wan runtime's `wan/modules/t5.py` is derived from
[Transformers](https://github.com/huggingface/transformers), and
`wan/utils/fm_solvers_unipc.py` is derived from
[Diffusers](https://github.com/huggingface/diffusers). Both projects are
licensed under Apache License 2.0. Their provenance comments are retained in
the corresponding files; the bundled license text is at `wan/LICENSE.txt`.

## SAM2

The optional interactive app installs
[SAM2](https://github.com/facebookresearch/sam2) as an external dependency at a
pinned revision. SAM2 is not copied into this repository and is licensed under
Apache License 2.0.
