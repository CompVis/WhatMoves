# WhatMoves interactive app

The app is a build-free browser interface for localized motion transfer.
FastAPI hosts the unchanged custom HTML, CSS, JavaScript, and streaming media;
named Gradio endpoints carry browser operations and integrate GPU calls with
Hugging Face ZeroGPU. One in-process scheduler still serializes SAM2 and Wan
work.

Install the complete application from the repository root using Python 3.12. Model weights are downloaded on first use and are not stored in this repository:

```bash
SAM2_BUILD_CUDA=0 pip install -r requirements.txt --no-build-isolation
```

## Run locally

From the repository root:

```bash
source .venv/bin/activate
python -m app
```

Open <http://localhost:7860>. The default bind address is loopback-only. Use
`--port PORT` to choose another port.

## Run on Slurm and connect over SSH

Allocate one sufficiently large GPU. Substitute your cluster's partition and
GPU resource syntax:

```bash
srun -p GPU_PARTITION --gres=gpu:GPU_TYPE:1 --time=04:00:00 \
  --job-name=whatmoves-app --pty bash
```

On the allocated GPU node, run:

```bash
source /path/to/WhatMoves/.venv/bin/activate
cd /path/to/WhatMoves

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD"

# Optional: reuse an existing official Wan2.2 base-model directory.
export WHATMOVES_WAN_CHECKPOINT=/path/to/Wan2.2-I2V-A14B

export WHATMOVES_APP_PORT=$(python -c \
  'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "GPU node: $(hostname)"
echo "Remote port: $WHATMOVES_APP_PORT"
python -m app --host 0.0.0.0 --port "$WHATMOVES_APP_PORT"
```

Leave that process running. On your laptop, substitute the printed values and
your normal SSH login host or alias:

```bash
ssh -N -L 7860:GPU_NODE:REMOTE_PORT LOGIN_HOST
```

Then open <http://localhost:7860>. If local port 7860 is occupied, change only
the first port, for example `-L 7861:GPU_NODE:REMOTE_PORT`, and open
<http://localhost:7861>.

Each user should request a separate GPU, start a separate app process, and use
their own remote and laptop ports. Do not add Uvicorn workers: model and session
state are intentionally process-local, and the Wan runtime is not reentrant.

## Workflow

The app opens with a small, fully masked and mapped synthetic example. You can
generate from it immediately or remove it and upload your own media.

1. Upload one or more source videos on the left and target images on the right.
   Each rail keeps its uploads independent; generation uses only the selected
   target image.
2. Source videos are timestamp-resampled to the adapter's 8 fps training
   cadence. Play them or scrub the filmstrip to inspect the exact frames used
   by inference. The two handles select the source interval.
3. Select any source frame, then choose **Start masking**. New uploads enter
   masking mode automatically. Left-click adds a positive SAM2 point,
   right-click adds a negative point, and left-drag adds a box. Hovering shows
   a transient proposal without changing the committed prompt. Choose
   **Add mask** to store the committed proposal at that exact frame.
4. Colored source-mask previews remain visible at their timeline positions;
   selecting one jumps back to its frame. Drag a colored source mask onto a
   grey target mask, or click the target mask and choose a source from its menu.
5. Choose a prompt and generate. Honest stage progress is shown, including the
   exact sampling step.

On ordinary local and Slurm deployments, SAM2 image features are queued as soon
as media is uploaded. On ZeroGPU they are prepared lazily inside the first
decorated mask request, because CUDA exists only for the lifetime of that
request. Source features are cached per selected frame in a bounded LRU cache.
Wan loads lazily on the first generation. A
generation uses an immutable snapshot of the selected target, source intervals,
masks, and settings present when **Generate video** was pressed; you may
continue editing while it runs. The completed video is still shown and marked
**previous inputs** if those inputs changed meanwhile.

## Lifecycle and privacy

- Runtime data is stored in a mode-`0700` directory under
  `/tmp/whatmoves-app-<job>-<pid>` by default; media and output files use mode
  `0600`.
- Target upload files are deleted immediately after decoding. Uploaded source
  videos are replaced by their canonical 8 fps session copy. Removing a source
  or target drops its RGB arrays, masks, drafts, pending work, mappings, cached
  SAM2 features, and session files. Bundled app assets are copied into a
  session and are never modified.
- Uncommitted mask drafts are cancelled when their source is left and after a
  browser reload, because their point and box prompts exist only in that page.
- Only the latest successful generated video is retained. Generation input
  links are removed as soon as their job finishes.
- Browser sessions send a lightweight heartbeat and expire after two idle
  hours by default. Expiration or graceful server shutdown releases all
  associated memory and files.
- The server has no authentication. Keep it on loopback or a trusted compute
  network and access it through SSH; do not expose it directly to the internet.

The most useful environment overrides are:

| Variable | Default | Purpose |
|---|---:|---|
| `WHATMOVES_WAN_CHECKPOINT` | download through Hugging Face | Local Wan2.2 base directory |
| `WHATMOVES_APP_TMP_DIR` | `/tmp/whatmoves-app-<job>-<pid>` | Runtime-data root |
| `WHATMOVES_APP_SESSION_TTL` | `7200` | Idle-session lifetime in seconds |
| `WHATMOVES_APP_MAX_SOURCES` | `8` | Maximum source videos per session |
| `WHATMOVES_APP_MAX_TARGETS` | `8` | Maximum target images per session |
| `WHATMOVES_APP_MAX_SAM_FRAMES` | `8` | Maximum cached per-frame SAM2 predictors |
| `WHATMOVES_APP_MAX_GPU_TASKS` | `32` | Maximum queued/running GPU tasks |
| `WHATMOVES_APP_HOST` | `127.0.0.1` | Server bind address |
| `WHATMOVES_APP_PORT` | `7860` | Server port |

Use `--reload` only for frontend development. It is not intended for model
inference or production sessions.
