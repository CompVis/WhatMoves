const GREY = "#a8afb9";

function makeCanvas(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

async function bitmapFromUrl(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load image (${response.status})`);
  return createImageBitmap(await response.blob());
}

function boundaryDistance(pixels, width, height) {
  const infinity = 65535;
  const distance = new Uint16Array(width * height);
  distance.fill(infinity);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      const value = pixels[index];
      const boundary = (value && (x === 0 || y === 0 || x === width - 1 || y === height - 1)) ||
        (x && pixels[index - 1] !== value) ||
        (x + 1 < width && pixels[index + 1] !== value) ||
        (y && pixels[index - width] !== value) ||
        (y + 1 < height && pixels[index + width] !== value);
      if (boundary) distance[index] = 0;
    }
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      let best = distance[index];
      if (x) best = Math.min(best, distance[index - 1] + 3);
      if (y) best = Math.min(best, distance[index - width] + 3);
      if (x && y) best = Math.min(best, distance[index - width - 1] + 4);
      if (x + 1 < width && y) best = Math.min(best, distance[index - width + 1] + 4);
      distance[index] = best;
    }
  }
  for (let y = height - 1; y >= 0; y -= 1) {
    for (let x = width - 1; x >= 0; x -= 1) {
      const index = y * width + x;
      let best = distance[index];
      if (x + 1 < width) best = Math.min(best, distance[index + 1] + 3);
      if (y + 1 < height) best = Math.min(best, distance[index + width] + 3);
      if (x + 1 < width && y + 1 < height) best = Math.min(best, distance[index + width + 1] + 4);
      if (x && y + 1 < height) best = Math.min(best, distance[index + width - 1] + 4);
      distance[index] = best;
    }
  }
  return distance;
}

async function loadMask(url, width, height, record) {
  const bitmap = await bitmapFromUrl(url);
  const raster = makeCanvas(width, height);
  const rasterContext = raster.getContext("2d", { willReadFrequently: true });
  rasterContext.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  const rgba = rasterContext.getImageData(0, 0, width, height).data;
  const pixels = new Uint8Array(width * height);
  for (let index = 0; index < pixels.length; index += 1) {
    pixels[index] = rgba[index * 4] > 127 ? 1 : 0;
  }

  const alpha = makeCanvas(width, height);
  const alphaContext = alpha.getContext("2d");
  const alphaImage = alphaContext.createImageData(width, height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      if (!pixels[index]) continue;
      const rgbaIndex = index * 4;
      alphaImage.data[rgbaIndex] = 255;
      alphaImage.data[rgbaIndex + 1] = 255;
      alphaImage.data[rgbaIndex + 2] = 255;
      alphaImage.data[rgbaIndex + 3] = 255;
    }
  }
  alphaContext.putImageData(alphaImage, 0, 0);
  return {
    ...record,
    pixels,
    alpha,
    edgeDistance: boundaryDistance(pixels, width, height),
    edgeCanvases: new Map(),
  };
}

export class MaskCanvas {
  constructor({ host, canvas, role, callbacks = {} }) {
    this.host = host;
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.role = role;
    this.callbacks = callbacks;
    this.asset = null;
    this.assetSignature = "";
    this.base = null;
    this.masks = [];
    this.draft = null;
    this.hoverDraft = null;
    this.hoverPoint = null;
    this.positive = [];
    this.negative = [];
    this.box = null;
    this.promptRevision = 0;
    this.hovered = null;
    this.dropHovered = null;
    this.pinned = false;
    this.pointerInside = false;
    this.pointerDown = null;
    this.loadToken = 0;
    this.scratch = makeCanvas(1, 1);

    canvas.addEventListener("pointerenter", () => {
      this.pointerInside = true;
      this._modeChanged();
    });
    canvas.addEventListener("pointerleave", () => {
      this.pointerInside = false;
      if (!this.pointerDown) this.hovered = null;
      this.clearHoverDraft();
      this.hoverPoint = null;
      this.callbacks.onHoverPrompt?.(null);
      this._modeChanged();
      this.render();
    });
    canvas.addEventListener("pointerdown", (event) => this._pointerDown(event));
    canvas.addEventListener("pointermove", (event) => this._pointerMove(event));
    canvas.addEventListener("pointerup", (event) => this._pointerUp(event));
    canvas.addEventListener("pointercancel", () => this._cancelPointer());
    canvas.addEventListener("contextmenu", (event) => this._contextMenu(event));
    this.resizeObserver = new ResizeObserver(() => this._resizeDisplay());
    this.resizeObserver.observe(host);
  }

  async setAsset(asset, masks = []) {
    if (!asset) {
      this.loadToken += 1;
      this.asset = null;
      this.assetSignature = "";
      if (this.base) this.base.close();
      this.base = null;
      this.masks = [];
      this.clearDraft();
      this.host.classList.remove("has-image");
      this.callbacks.onLoading?.(false);
      return;
    }
    const signature = JSON.stringify([
      asset.image_url,
      masks.map((mask) => [mask.id, mask.url, mask.displayColor]),
    ]);
    if (signature === this.assetSignature) return;
    const mediaChanged = !this.asset || this.asset.image_url !== asset.image_url;
    const token = ++this.loadToken;
    this.callbacks.onLoading?.(true);
    try {
      const base = mediaChanged ? await bitmapFromUrl(asset.image_url) : this.base;
      if (token !== this.loadToken) {
        if (mediaChanged) base.close();
        return;
      }
      if (mediaChanged) {
        if (this.base) this.base.close();
        this.base = base;
        this.canvas.width = base.width;
        this.canvas.height = base.height;
        this.scratch = makeCanvas(base.width, base.height);
        this.clearDraft();
      }
      const loadedMasks = await Promise.all(
        masks.map((mask) => loadMask(mask.url, this.canvas.width, this.canvas.height, mask)),
      );
      if (token !== this.loadToken) return;
      this.asset = asset;
      this.assetSignature = signature;
      this.masks = loadedMasks;
      this.host.classList.add("has-image");
      this._resizeDisplay();
      this.render();
    } catch (error) {
      this.callbacks.onError?.(error);
    } finally {
      if (token === this.loadToken) this.callbacks.onLoading?.(false);
    }
  }

  togglePinned() {
    return this.setPinned(!this.pinned);
  }

  setPinned(value) {
    this.pinned = Boolean(value);
    if (!this.pinned) {
      this.clearHoverDraft();
      this.hoverPoint = null;
      this.callbacks.onHoverPrompt?.(null);
    }
    this._modeChanged();
    this.render();
    return this.pinned;
  }

  isMasking() {
    return Boolean(this.asset && this.pinned);
  }

  promptPayload() {
    return {
      asset_revision: this.asset?.revision,
      frame_index: this.role === "source" ? this.asset?.current_frame : null,
      prompt_revision: this.promptRevision,
      positive: this.positive,
      negative: this.negative,
      box: this.box,
    };
  }

  async setDraft(base64, promptRevision) {
    if (promptRevision !== this.promptRevision || !this.asset) return false;
    const record = { id: "draft", url: `data:image/png;base64,${base64}` };
    const draft = await loadMask(record.url, this.canvas.width, this.canvas.height, record);
    if (promptRevision !== this.promptRevision) return false;
    this.draft = draft;
    this.callbacks.onDraftState?.(true);
    this.render();
    return true;
  }

  async setHoverDraft(base64, promptRevision) {
    if (promptRevision !== this.promptRevision || !this.asset || !this.isMasking()) return false;
    const record = { id: "hover-draft", url: `data:image/png;base64,${base64}` };
    const draft = await loadMask(record.url, this.canvas.width, this.canvas.height, record);
    if (promptRevision !== this.promptRevision || !this.isMasking()) return false;
    this.hoverDraft = draft;
    this.render();
    return true;
  }

  clearHoverDraft() {
    if (!this.hoverDraft) return;
    this.hoverDraft = null;
    this.hoverPoint = null;
    this.render();
  }

  clearDraft() {
    this.positive = [];
    this.negative = [];
    this.box = null;
    this.draft = null;
    this.hoverDraft = null;
    this.promptRevision += 1;
    this.callbacks.onDraftState?.(false);
    this.render();
  }

  masksAtClient(clientX, clientY) {
    const point = this._point({ clientX, clientY });
    if (!point) return [];
    return this.masks
      .filter((mask) => mask.pixels[Math.floor(point.y) * this.canvas.width + Math.floor(point.x)])
      .reverse();
  }

  setDropHover(maskId) {
    if (this.dropHovered === maskId) return;
    this.dropHovered = maskId;
    this.render();
  }

  getMask(maskId) {
    return this.masks.find((value) => value.id === maskId) || null;
  }

  maskClientCenter(maskId) {
    const mask = this.masks.find((value) => value.id === maskId);
    if (!mask || !this.canvas.width || !this.canvas.height) return null;
    let minX = this.canvas.width;
    let minY = this.canvas.height;
    let maxX = -1;
    let maxY = -1;
    for (let y = 0; y < this.canvas.height; y += 1) {
      for (let x = 0; x < this.canvas.width; x += 1) {
        if (!mask.pixels[y * this.canvas.width + x]) continue;
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
    if (maxX < 0 || maxY < 0) return null;
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: rect.left + ((minX + maxX) / 2 / this.canvas.width) * rect.width,
      y: rect.top + ((minY + maxY) / 2 / this.canvas.height) * rect.height,
    };
  }

  render() {
    if (!this.base) return;
    const context = this.context;
    context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    context.drawImage(this.base, 0, 0, this.canvas.width, this.canvas.height);
    for (const mask of this.masks) {
      const color = mask.displayColor || mask.color || GREY;
      this._drawTint(mask.alpha, color, 0.28);
      const hovered = mask.id === this.hovered;
      const dropping = mask.id === this.dropHovered;
      this._drawEdge(mask, dropping ? "#ffffff" : color, hovered || dropping ? 4 : 2);
    }
    const preview = this.hoverDraft || this.draft;
    if (preview && this.isMasking()) {
      this._drawTint(preview.alpha, "#ffffff", this.hoverDraft ? 0.16 : 0.2);
      this._drawEdge(preview, "#ffffff", 2);
    }
    if (this.isMasking()) this._drawPrompts();
  }

  _drawTint(alpha, color, opacity) {
    const scratch = this.scratch;
    const scratchContext = scratch.getContext("2d");
    scratchContext.clearRect(0, 0, scratch.width, scratch.height);
    scratchContext.globalCompositeOperation = "source-over";
    scratchContext.fillStyle = color;
    scratchContext.fillRect(0, 0, scratch.width, scratch.height);
    scratchContext.globalCompositeOperation = "destination-in";
    scratchContext.drawImage(alpha, 0, 0);
    scratchContext.globalCompositeOperation = "source-over";
    this.context.save();
    this.context.globalAlpha = opacity;
    this.context.drawImage(scratch, 0, 0);
    this.context.restore();
  }

  _drawEdge(mask, color, cssRadius) {
    const rect = this.canvas.getBoundingClientRect();
    const radius = Math.max(1, Math.round(cssRadius * this.canvas.width / Math.max(rect.width, 1)));
    let edge = mask.edgeCanvases.get(radius);
    if (!edge) {
      edge = makeCanvas(this.canvas.width, this.canvas.height);
      const edgeContext = edge.getContext("2d");
      const image = edgeContext.createImageData(edge.width, edge.height);
      const threshold = radius * 3;
      for (let index = 0; index < mask.edgeDistance.length; index += 1) {
        if (mask.edgeDistance[index] > threshold) continue;
        const rgba = index * 4;
        image.data[rgba] = 255;
        image.data[rgba + 1] = 255;
        image.data[rgba + 2] = 255;
        image.data[rgba + 3] = 255;
      }
      edgeContext.putImageData(image, 0, 0);
      mask.edgeCanvases.set(radius, edge);
    }
    const scratchContext = this.scratch.getContext("2d");
    scratchContext.clearRect(0, 0, this.scratch.width, this.scratch.height);
    scratchContext.globalCompositeOperation = "source-over";
    scratchContext.fillStyle = color;
    scratchContext.fillRect(0, 0, this.scratch.width, this.scratch.height);
    scratchContext.globalCompositeOperation = "destination-in";
    scratchContext.drawImage(edge, 0, 0);
    scratchContext.globalCompositeOperation = "source-over";
    this.context.drawImage(this.scratch, 0, 0);
  }

  _drawPrompts() {
    const context = this.context;
    const rect = this.canvas.getBoundingClientRect();
    const scale = this.canvas.width / Math.max(rect.width, 1);
    const radius = Math.max(5, 5 * scale);
    const drawPoint = (point, fill, sign) => {
      const x = point[0] * this.canvas.width;
      const y = point[1] * this.canvas.height;
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.fillStyle = fill;
      context.fill();
      context.lineWidth = Math.max(2, 1.5 * scale);
      context.strokeStyle = "white";
      context.stroke();
      context.beginPath();
      context.moveTo(x - radius * 0.45, y);
      context.lineTo(x + radius * 0.45, y);
      if (sign === "+") {
        context.moveTo(x, y - radius * 0.45);
        context.lineTo(x, y + radius * 0.45);
      }
      context.strokeStyle = "white";
      context.lineWidth = Math.max(1.5, scale);
      context.stroke();
    };
    this.positive.forEach((point) => drawPoint(point, "#29b889", "+"));
    this.negative.forEach((point) => drawPoint(point, "#df5b61", "−"));
    if (this.hoverPoint) drawPoint(this.hoverPoint, "#28c997", "+");
    const box = this.pointerDown?.mode === "mask" && this.pointerDown.dragging
      ? this._normalizedBox(this.pointerDown.start, this.pointerDown.current)
      : this.box;
    if (box) {
      context.strokeStyle = "rgba(255,255,255,.95)";
      context.lineWidth = Math.max(2, 2 * scale);
      context.setLineDash([7 * scale, 5 * scale]);
      context.strokeRect(
        box[0] * this.canvas.width,
        box[1] * this.canvas.height,
        (box[2] - box[0]) * this.canvas.width,
        (box[3] - box[1]) * this.canvas.height,
      );
      context.setLineDash([]);
    }
  }

  _point(event) {
    if (!this.base) return null;
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const x = (event.clientX - rect.left) / rect.width * this.canvas.width;
    const y = (event.clientY - rect.top) / rect.height * this.canvas.height;
    if (x < 0 || y < 0 || x >= this.canvas.width || y >= this.canvas.height) return null;
    return { x, y, clientX: event.clientX, clientY: event.clientY };
  }

  _normalized(point) {
    return [
      Math.min(1, Math.max(0, point.x / this.canvas.width)),
      Math.min(1, Math.max(0, point.y / this.canvas.height)),
    ];
  }

  _normalizedBox(first, second) {
    const a = this._normalized(first);
    const b = this._normalized(second);
    return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
  }

  _pointerDown(event) {
    if (!this.asset || event.button !== 0) return;
    const point = this._point(event);
    if (!point) return;
    const mode = this.isMasking() ? "mask" : "normal";
    if (mode === "mask") {
      this.clearHoverDraft();
      this.hoverPoint = null;
      this.callbacks.onHoverPrompt?.(null);
    }
    const hit = this.masksAtClient(event.clientX, event.clientY)[0] || null;
    this.pointerDown = { start: point, current: point, mode, hit, dragging: false };
    this.canvas.setPointerCapture(event.pointerId);
    event.preventDefault();
  }

  _pointerMove(event) {
    const point = this._point(event);
    if (this.pointerDown) {
      const clientPoint = point || { clientX: event.clientX, clientY: event.clientY };
      const distance = Math.hypot(clientPoint.clientX - this.pointerDown.start.clientX, clientPoint.clientY - this.pointerDown.start.clientY);
      if (distance > 5) this.pointerDown.dragging = true;
      if (this.pointerDown.mode === "normal" && this.pointerDown.dragging && this.role === "source" && this.pointerDown.hit) {
        this.callbacks.onMappingDrag?.("move", this.pointerDown.hit, event.clientX, event.clientY);
      } else if (point) {
        this.pointerDown.current = point;
      }
      this.render();
      return;
    }
    if (this.isMasking() && point) {
      this.hovered = null;
      this.hoverPoint = this._normalized(point);
      const payload = this.promptPayload();
      payload.positive = [...payload.positive, this.hoverPoint];
      payload.transient = true;
      this.callbacks.onHoverPrompt?.(payload, point.clientX, point.clientY);
      this.render();
      return;
    }
    const next = this.masksAtClient(event.clientX, event.clientY)[0]?.id || null;
    this.hoverPoint = null;
    if (next !== this.hovered) {
      this.hovered = next;
      this._modeChanged();
      this.render();
    }
  }

  _pointerUp(event) {
    if (!this.pointerDown || event.button !== 0) return;
    const interaction = this.pointerDown;
    const point = this._point(event) || interaction.current;
    this.pointerDown = null;
    if (interaction.mode === "mask") {
      if (interaction.dragging) {
        const box = this._normalizedBox(interaction.start, point);
        if ((box[2] - box[0]) * this.canvas.width > 3 && (box[3] - box[1]) * this.canvas.height > 3) {
          this.box = box;
          this._promptsChanged();
        }
      } else {
        this.positive.push(this._normalized(point));
        this._promptsChanged();
      }
    } else if (interaction.dragging && this.role === "source" && interaction.hit) {
      this.callbacks.onMappingDrag?.("drop", interaction.hit, event.clientX, event.clientY);
    } else {
      const hits = this.masksAtClient(event.clientX, event.clientY);
      if (hits.length) this.callbacks.onMaskClick?.(hits, event.clientX, event.clientY);
    }
    this.render();
  }

  _contextMenu(event) {
    if (!this.isMasking() || !this.asset) return;
    event.preventDefault();
    const point = this._point(event);
    if (!point) return;
    this.negative.push(this._normalized(point));
    this._promptsChanged();
  }

  _cancelPointer() {
    if (this.pointerDown?.mode === "normal" && this.pointerDown.hit) {
      this.callbacks.onMappingDrag?.("cancel", this.pointerDown.hit, 0, 0);
    }
    this.pointerDown = null;
    this.render();
  }

  _promptsChanged() {
    this.promptRevision += 1;
    this.draft = null;
    this.hoverDraft = null;
    this.hoverPoint = null;
    this.callbacks.onHoverPrompt?.(null);
    this.callbacks.onDraftState?.(false);
    this.callbacks.onPromptsChanged?.(this.promptPayload());
    this.render();
  }

  _modeChanged() {
    const active = this.isMasking();
    if (!active) {
      this.hoverDraft = null;
      this.hoverPoint = null;
    }
    this.canvas.style.cursor = active ? "crosshair" : (this.hovered ? "pointer" : "default");
    this.callbacks.onModeChange?.(active, this.pinned);
  }

  _resizeDisplay() {
    if (!this.base) return;
    const width = this.host.clientWidth;
    const height = this.host.clientHeight;
    const scale = Math.min(width / this.canvas.width, height / this.canvas.height);
    this.canvas.style.width = `${Math.max(1, this.canvas.width * scale)}px`;
    this.canvas.style.height = `${Math.max(1, this.canvas.height * scale)}px`;
    this.render();
  }
}
