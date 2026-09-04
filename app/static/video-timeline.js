function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function timeLabel(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.max(0, seconds - minutes * 60);
  return `${minutes}:${remainder.toFixed(1).padStart(4, "0")}`;
}

export class VideoTimeline {
  constructor({ root, video, playButton, time, callbacks = {} }) {
    this.root = root;
    this.video = video;
    this.playButton = playButton;
    this.time = time;
    this.callbacks = callbacks;
    this.filmstrip = root.querySelector(".timeline-filmstrip");
    this.maskLayer = root.querySelector(".timeline-masks");
    this.playhead = root.querySelector(".timeline-playhead");
    this.selection = root.querySelector(".timeline-selection");
    this.before = root.querySelector(".timeline-before");
    this.after = root.querySelector(".timeline-after");
    this.leftHandle = root.querySelector('[data-handle="left"]');
    this.rightHandle = root.querySelector('[data-handle="right"]');
    this.source = null;
    this.currentFrame = 0;
    this.trimStart = 0;
    this.trimEnd = 0;
    this.drag = null;
    this.animation = null;
    this.thumbnailSignature = "";

    playButton.addEventListener("click", () => this.togglePlayback());
    root.addEventListener("pointerdown", (event) => this._pointerDown(event));
    video.addEventListener("play", () => this._playbackChanged(true));
    video.addEventListener("pause", () => this._playbackChanged(false));
    video.addEventListener("ended", () => this._loop());
    ["loadstart", "waiting", "seeking"].forEach((eventName) => {
      video.addEventListener(eventName, () => this.callbacks.onLoading?.(true));
    });
    ["loadeddata", "canplay", "playing", "seeked", "emptied", "error"].forEach((eventName) => {
      video.addEventListener(eventName, () => this.callbacks.onLoading?.(false));
    });
    this.resizeObserver = new ResizeObserver(() => this._renderThumbnails());
    this.resizeObserver.observe(root);
  }

  setSource(source) {
    const sameMedia = Boolean(
      source && this.source && source.id === this.source.id && source.revision === this.source.revision
    );
    this.source = source;
    this.root.classList.toggle("hidden", !source);
    if (!source) {
      this.callbacks.onLoading?.(false);
      this.video.pause();
      this.video.removeAttribute("src");
      this.filmstrip.replaceChildren();
      this.maskLayer.replaceChildren();
      return;
    }
    this.trimStart = source.trim_start;
    this.trimEnd = source.trim_end;
    const absoluteUrl = new URL(source.video_url, window.location.href).href;
    if (this.video.src !== absoluteUrl) {
      this.video.pause();
      this.video.src = source.video_url;
      this.video.load();
    }
    if (!sameMedia || (this.video.paused && !this.drag)) {
      this.currentFrame = source.current_frame;
      this._setVideoTime(this.currentFrame / source.fps);
    }
    this._renderThumbnails();
    this._renderMasks();
    this._renderGeometry();
  }

  togglePlayback() {
    if (!this.source) return;
    if (!this.video.paused) {
      this.video.pause();
      this.callbacks.onSeek?.(this.currentFrame, true);
      return;
    }
    if (this.currentFrame < this.trimStart || this.currentFrame >= this.trimEnd) {
      this._seekLocal(this.trimStart);
    }
    this.video.play().catch(() => {});
  }

  pause() {
    this.video.pause();
  }

  jumpToMask(mask) {
    this.pause();
    this._seekLocal(mask.frame_index);
    this.callbacks.onSeek?.(mask.frame_index, true);
  }

  _playbackChanged(playing) {
    this.playButton.innerHTML = playing
      ? '<svg class="playback-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h4v14H6zM14 5h4v14h-4z"></path></svg>'
      : '<svg class="playback-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"></path></svg>';
    this.playButton.setAttribute("aria-label", playing ? "Pause" : "Play");
    this.playButton.title = playing ? "Pause" : "Play";
    this.callbacks.onPlaybackChange?.(playing);
    cancelAnimationFrame(this.animation);
    if (playing) this._tick();
  }

  _tick() {
    if (!this.source || this.video.paused) return;
    const frame = clamp(
      Math.round(this.video.currentTime * this.source.fps),
      this.trimStart,
      this.trimEnd,
    );
    this.currentFrame = frame;
    if (this.video.currentTime >= (this.trimEnd + 0.75) / this.source.fps) {
      this._loop();
    } else {
      this._renderGeometry();
    }
    this.animation = requestAnimationFrame(() => this._tick());
  }

  _loop() {
    if (!this.source) return;
    this._seekLocal(this.trimStart);
    if (!this.video.paused) this.video.play().catch(() => {});
  }

  _seekLocal(frame) {
    if (!this.source) return;
    this.currentFrame = clamp(Math.round(frame), 0, this.source.frame_count - 1);
    this._setVideoTime(this.currentFrame / this.source.fps);
    this._renderGeometry();
  }

  _setVideoTime(seconds) {
    const apply = () => {
      try { this.video.currentTime = seconds; } catch (_) {}
    };
    if (this.video.readyState >= 1) apply();
    else this.video.addEventListener("loadedmetadata", apply, { once: true });
  }

  _frameAt(clientX) {
    const rect = this.root.getBoundingClientRect();
    const ratio = clamp((clientX - rect.left) / Math.max(rect.width, 1), 0, 1);
    return Math.round(ratio * Math.max(this.source.frame_count - 1, 0));
  }

  _pointerDown(event) {
    if (!this.source || event.button !== 0 || event.target.closest(".timeline-mask-preview")) return;
    const handle = event.target.closest(".trim-handle")?.dataset.handle;
    this.pause();
    this.drag = {
      mode: handle || "playhead",
      pointerId: event.pointerId,
      originalStart: this.trimStart,
      originalEnd: this.trimEnd,
    };
    this.root.setPointerCapture(event.pointerId);
    this.root.addEventListener("pointermove", this._boundMove ||= (value) => this._pointerMove(value));
    this.root.addEventListener("pointerup", this._boundUp ||= (value) => this._pointerUp(value));
    this.root.addEventListener("pointercancel", this._boundUp);
    this._pointerMove(event);
    event.preventDefault();
  }

  _pointerMove(event) {
    if (!this.drag || !this.source) return;
    const frame = this._frameAt(event.clientX);
    const minimum = Math.min(8, this.source.frame_count);
    if (this.drag.mode === "left") {
      this.trimStart = clamp(frame, 0, this.trimEnd - minimum + 1);
      this._seekLocal(Math.max(this.currentFrame, this.trimStart));
    } else if (this.drag.mode === "right") {
      this.trimEnd = clamp(frame, this.trimStart + minimum - 1, this.source.frame_count - 1);
      this._seekLocal(Math.min(this.currentFrame, this.trimEnd));
    } else {
      this._seekLocal(frame);
      this.callbacks.onSeek?.(this.currentFrame, false);
    }
    this._renderGeometry();
  }

  async _pointerUp(event) {
    if (!this.drag || !this.source) return;
    const drag = this.drag;
    this.drag = null;
    try { this.root.releasePointerCapture(event.pointerId); } catch (_) {}
    this.root.removeEventListener("pointermove", this._boundMove);
    this.root.removeEventListener("pointerup", this._boundUp);
    this.root.removeEventListener("pointercancel", this._boundUp);
    if (drag.mode === "playhead") {
      await this.callbacks.onSeek?.(this.currentFrame, true);
    } else {
      const accepted = await this.callbacks.onTrim?.(this.trimStart, this.trimEnd);
      if (accepted === false) {
        this.trimStart = drag.originalStart;
        this.trimEnd = drag.originalEnd;
        this._seekLocal(clamp(this.currentFrame, this.trimStart, this.trimEnd));
      }
    }
    this._renderGeometry();
  }

  _ratio(frame) {
    return this.source.frame_count <= 1 ? 0 : frame / (this.source.frame_count - 1);
  }

  _renderGeometry() {
    if (!this.source) return;
    const start = this._ratio(this.trimStart) * 100;
    const end = this._ratio(this.trimEnd) * 100;
    const current = this._ratio(this.currentFrame) * 100;
    this.selection.style.left = `${start}%`;
    this.selection.style.width = `${Math.max(0, end - start)}%`;
    this.before.style.width = `${start}%`;
    this.after.style.left = `${end}%`;
    this.after.style.width = `${100 - end}%`;
    this.leftHandle.style.left = `${start}%`;
    this.rightHandle.style.left = `${end}%`;
    this.playhead.style.left = `${current}%`;
    this.time.textContent = `${timeLabel(this.currentFrame / this.source.fps)}  ·  frame ${this.currentFrame + 1}/${this.source.frame_count}  ·  ${this.trimEnd - this.trimStart + 1} selected`;
  }

  _renderThumbnails() {
    if (!this.source || !this.root.clientWidth) return;
    const height = 50;
    const width = Math.max(48, height * this.source.width / this.source.height);
    const count = Math.max(1, Math.floor(this.root.clientWidth / (width + 3)));
    const indices = [];
    for (let index = 0; index < Math.min(count, this.source.frame_count); index += 1) {
      const denominator = Math.max(Math.min(count, this.source.frame_count) - 1, 1);
      indices.push(Math.round(index * (this.source.frame_count - 1) / denominator));
    }
    const signature = `${this.source.id}:${this.source.revision}:${indices.join(",")}`;
    if (signature === this.thumbnailSignature) return;
    this.thumbnailSignature = signature;
    this.filmstrip.replaceChildren(...indices.map((frame) => {
      const item = document.createElement("div");
      item.className = "timeline-thumbnail";
      item.style.width = `${width}px`;
      const image = document.createElement("img");
      image.src = `/api/sessions/${this.source.session_id}/sources/${this.source.id}/frames/${frame}.jpg`;
      image.alt = "";
      item.append(image);
      return item;
    }));
  }

  _renderMasks() {
    if (!this.source) return;
    this.maskLayer.replaceChildren(...this.source.masks.map((mask, index) => {
      const button = document.createElement("button");
      button.className = "timeline-mask-preview";
      if (mask.frame_index < this.trimStart || mask.frame_index > this.trimEnd) {
        button.classList.add("excluded");
      }
      button.style.left = `${this._ratio(mask.frame_index) * 100}%`;
      button.style.borderColor = mask.color;
      button.style.top = `${3 + (index % 3) * 5}px`;
      const excluded = button.classList.contains("excluded") ? " · outside selected interval" : "";
      button.title = `Mask ${index + 1} · ${timeLabel(mask.frame_index / this.source.fps)}${excluded}`;
      const image = document.createElement("img");
      image.src = mask.preview_url;
      image.alt = "";
      button.append(image);
      button.addEventListener("pointerdown", (event) => event.stopPropagation());
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        this.jumpToMask(mask);
        this.callbacks.onMaskClick?.(mask);
      });
      return button;
    }));
  }
}
