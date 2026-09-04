import { MaskCanvas } from "/static/mask-canvas.js?v=20260904h";
import { VideoTimeline } from "/static/video-timeline.js?v=20260904h";
import { Client, handle_file } from "/static/gradio-client.js?v=20260904j";

const $ = (id) => document.getElementById(id);
const elements = {
  startOver: $("startOver"),
  sourceRail: $("sourceRail"),
  targetRail: $("targetRail"),
  sourceFile: $("sourceFile"),
  targetFile: $("targetFile"),
  removeSource: $("removeSource"),
  removeTarget: $("removeTarget"),
  sourceRailUpload: $("sourceRailUpload"),
  sourceEmpty: $("sourceEmpty"),
  targetEmpty: $("targetEmpty"),
  sourceTools: $("sourceTools"),
  targetTools: $("targetTools"),
  sourceMaskMode: $("sourceMaskMode"),
  targetMaskMode: $("targetMaskMode"),
  sourceModeBadge: $("sourceModeBadge"),
  targetModeBadge: $("targetModeBadge"),
  sourceCanvasStatus: $("sourceCanvasStatus"),
  targetCanvasStatus: $("targetCanvasStatus"),
  sourceLoader: $("sourceLoader"),
  targetLoader: $("targetLoader"),
  sourceAddMask: $("sourceAddMask"),
  targetAddMask: $("targetAddMask"),
  sourceCancelDraft: $("sourceCancelDraft"),
  targetCancelDraft: $("targetCancelDraft"),
  sourceSettings: $("sourceSettings"),
  sourceVideo: $("sourceVideo"),
  sourcePlay: $("sourcePlay"),
  sourceTime: $("sourceTime"),
  sourceTimeline: $("sourceTimeline"),
  contextMenu: $("contextMenu"),
  dragGhost: $("dragGhost"),
  mappingHint: $("mappingHint"),
  toast: $("toast"),
  prompt: $("prompt"),
  negativePrompt: $("negativePrompt"),
  guidanceMode: $("guidanceMode"),
  steps: $("steps"),
  seed: $("seed"),
  textGuidance: $("textGuidance"),
  motionGuidance: $("motionGuidance"),
  loraScale: $("loraScale"),
  generateButton: $("generateButton"),
  generateButtonProgress: $("generateButtonProgress"),
  generateButtonLabel: $("generateButtonLabel"),
  generateButtonCount: $("generateButtonCount"),
  generateHint: $("generateHint"),
  generatedRail: $("generatedRail"),
  resultPanel: $("resultPanel"),
  resultVideo: $("resultVideo"),
  resultLoader: $("resultLoader"),
  resultMeta: $("resultMeta"),
  resultClose: $("resultClose"),
};

let sessionState = null;
let selectedSourceId = null;
let toastTimer = null;
let pollTimer = null;
let activeGeneration = null;
let activeGenerationForm = null;
let activeGenerationStatus = null;
let frameController = null;
let draggedMaskId = null;
let mappingHintTimer = null;
let exampleTutorialState = "pending";
let exampleTutorialScheduled = false;
let formTargetId = null;
let settingsDirty = false;
let settingsSaveTimer = null;
let targetSettingsSessionId = null;
let targetSettings = {};
let generatedOutputs = [];
const predictionControllers = { source: null, target: null };
const hoverControllers = { source: null, target: null };
const hoverTimers = { source: null, target: null };
const loadingReasons = { source: new Set(), target: new Set(), result: new Set() };

function setMediaLoading(kind, reason, active) {
  const reasons = loadingReasons[kind];
  const loader = kind === "source"
    ? elements.sourceLoader
    : kind === "target"
      ? elements.targetLoader
      : elements.resultLoader;
  if (active) reasons.add(reason);
  else reasons.delete(reason);
  loader.classList.toggle("hidden", reasons.size === 0);
}

let gradioClientPromise = null;

function getGradioClient() {
  if (!gradioClientPromise) {
    gradioClientPromise = Client.connect(new URL("/gradio", window.location.origin).href, {
      events: ["data", "status"],
    }).catch((error) => {
      gradioClientPromise = null;
      throw error;
    });
  }
  return gradioClientPromise;
}

async function request(url, options = {}) {
  const method = options.method || "GET";
  const signal = options.signal;
  if (signal?.aborted) throw new DOMException("Request aborted", "AbortError");

  let body = null;
  let upload = null;
  if (options.body instanceof FormData) {
    const file = options.body.get("file");
    if (file instanceof File) upload = handle_file(file);
  } else if (typeof options.body === "string") {
    body = options.body;
  }

  const endpoint = url.includes("/predict")
    ? "/sam_api"
    : url.split("?", 1)[0].endsWith("/generate")
      ? "/generation_api"
      : "/api";
  const client = await getGradioClient();
  const inputs = [method, url, body, upload];
  if (endpoint === "/api") {
    const result = await client.predict(endpoint, inputs);
    let envelope = result.data?.[0] ?? null;
    if (typeof envelope === "string") {
      try { envelope = JSON.parse(envelope); } catch (_) { /* Keep the server value for the error below. */ }
    }
    if (!envelope?.ok) {
      const payload = envelope?.data;
      const message = payload?.detail || payload || `Request failed (${envelope?.status || "unknown"})`;
      const error = new Error(message);
      error.status = envelope?.status;
      throw error;
    }
    return envelope.data;
  }
  const submission = client.submit(endpoint, inputs);
  const abort = () => submission.cancel();
  signal?.addEventListener("abort", abort, { once: true });

  let envelope = null;
  try {
    for await (const message of submission) {
      if (signal?.aborted) throw new DOMException("Request aborted", "AbortError");
      if (message.type === "status") {
        options.onGradioStatus?.(message);
        if (message.stage === "error") {
          throw new Error(typeof message.message === "string" ? message.message : "Gradio request failed");
        }
      } else if (message.type === "data") {
        envelope = message.data?.[0] ?? null;
      }
    }
  } finally {
    signal?.removeEventListener("abort", abort);
  }

  if (typeof envelope === "string") {
    try { envelope = JSON.parse(envelope); } catch (_) { /* Gradio may already return an object. */ }
  }
  if (!envelope?.ok) {
    const payload = envelope?.data;
    const message = payload?.detail || payload || `Request failed (${envelope?.status || "unknown"})`;
    const error = new Error(message);
    error.status = envelope?.status;
    throw error;
  }
  return envelope.data;
}

function jsonOptions(value, method = "POST") {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
  };
}

function showToast(message) {
  clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.remove("hidden");
  toastTimer = setTimeout(() => elements.toast.classList.add("hidden"), 4200);
}

function canvasStatus(kind, message = "", error = false) {
  const element = kind === "source" ? elements.sourceCanvasStatus : elements.targetCanvasStatus;
  element.textContent = message;
  element.style.color = error ? "#ff9a9c" : "";
  element.classList.toggle("hidden", !message);
}

function modeChanged(kind, active, pinned) {
  const badge = kind === "source" ? elements.sourceModeBadge : elements.targetModeBadge;
  const button = kind === "source" ? elements.sourceMaskMode : elements.targetMaskMode;
  badge.classList.toggle("hidden", !active);
  button.classList.toggle("active", pinned);
  button.textContent = pinned ? "Done masking" : "Start masking";
}

function draftChanged(kind, ready) {
  const add = kind === "source" ? elements.sourceAddMask : elements.targetAddMask;
  const cancel = kind === "source" ? elements.sourceCancelDraft : elements.targetCancelDraft;
  add.classList.toggle("hidden", !ready);
  cancel.classList.toggle("hidden", !ready);
}

const sourceCanvas = new MaskCanvas({
  host: $("sourceCanvasHost"),
  canvas: $("sourceCanvas"),
  role: "source",
  callbacks: {
    onError: (error) => showToast(error.message),
    onLoading: (active) => setMediaLoading("source", "canvas", active),
    onPromptsChanged: (payload) => predictMask("source", payload),
    onHoverPrompt: (payload) => scheduleHover("source", payload),
    onDraftState: (ready) => draftChanged("source", ready),
    onModeChange: (active, pinned) => modeChanged("source", active, pinned),
    onMaskClick: (masks, x, y) => openSourceMenu(masks, x, y),
    onMappingDrag: (phase, mask, x, y) => mappingDrag(phase, mask, x, y),
  },
});

const targetCanvas = new MaskCanvas({
  host: $("targetCanvasHost"),
  canvas: $("targetCanvas"),
  role: "target",
  callbacks: {
    onError: (error) => showToast(error.message),
    onLoading: (active) => setMediaLoading("target", "canvas", active),
    onPromptsChanged: (payload) => predictMask("target", payload),
    onHoverPrompt: (payload) => scheduleHover("target", payload),
    onDraftState: (ready) => draftChanged("target", ready),
    onModeChange: (active, pinned) => modeChanged("target", active, pinned),
    onMaskClick: (masks, x, y) => openTargetMenu(masks, x, y),
  },
});

const sourceTimeline = new VideoTimeline({
  root: elements.sourceTimeline,
  video: elements.sourceVideo,
  playButton: elements.sourcePlay,
  time: elements.sourceTime,
  callbacks: {
    onSeek: (frame, final) => selectSourceFrame(frame, final),
    onTrim: (start, end) => updateSourceTrim(start, end),
    onMaskClick: () => sourceCanvas.setPinned(false),
    onLoading: (active) => setMediaLoading(
      "source",
      "video",
      active && $("sourceCanvasHost").classList.contains("playing"),
    ),
    onPlaybackChange: (playing) => {
      $("sourceCanvasHost").classList.toggle("playing", playing);
      if (!playing) setMediaLoading("source", "video", false);
      if (playing) {
        sourceCanvas.setPinned(false);
        clearCanvasDraft("source");
      }
    },
  },
});

["loadstart", "waiting", "seeking"].forEach((eventName) => {
  elements.resultVideo.addEventListener(eventName, () => setMediaLoading("result", "video", true));
});
["loadeddata", "canplay", "playing", "seeked", "emptied", "error"].forEach((eventName) => {
  elements.resultVideo.addEventListener(eventName, () => setMediaLoading("result", "video", false));
});

function selectedSource() {
  return sessionState?.sources.find((source) => source.id === selectedSourceId) || null;
}

function selectedTarget() {
  return sessionState?.targets.find((target) => target.id === sessionState.selected_target_id) || null;
}

function sourceMaskEntries() {
  const entries = [];
  (sessionState?.sources || []).forEach((source, sourceIndex) => {
    source.masks.filter((mask) => mask.frame_index >= source.trim_start && mask.frame_index <= source.trim_end).forEach((mask, maskIndex) => {
      entries.push({
        ...mask,
        source,
        sourceLabel: `S${sourceIndex + 1}`,
        maskLabel: `M${maskIndex + 1}`,
      });
    });
  });
  return entries;
}

function mappedColor(targetMask) {
  const found = sourceMaskEntries().find(
    (mask) => mask.source.id === targetMask.source_id && mask.id === targetMask.source_mask_id,
  );
  return found?.color || "#a8afb9";
}

function isExampleTutorial(target = selectedTarget()) {
  return Boolean(
    target?.example &&
    sessionState?.sources.some((source) => source.example) &&
    target.masks.some((mask) => mask.source_id && mask.source_mask_id)
  );
}

function generationSettingsPayload() {
  const numberValue = (element, fallback, integer = false) => {
    const value = integer ? Number.parseInt(element.value, 10) : Number.parseFloat(element.value);
    return Number.isFinite(value) ? value : fallback;
  };
  return {
    prompt: elements.prompt.value,
    negative_prompt: elements.negativePrompt.value || null,
    steps: numberValue(elements.steps, 40, true),
    seed: numberValue(elements.seed, 42, true),
    guidance_mode: elements.guidanceMode.value,
    text_guidance_scale: numberValue(elements.textGuidance, 3.5),
    motion_guidance_scale: numberValue(elements.motionGuidance, 1.0),
    lora_scale: numberValue(elements.loraScale, 1.0),
  };
}

function defaultGenerationSettings() {
  return {
    prompt: "A bear and a dog together in the forest.",
    negative_prompt: null,
    steps: 40,
    seed: 42,
    guidance_mode: "text_cfg",
    text_guidance_scale: 3.5,
    motion_guidance_scale: 1.0,
    lora_scale: 1.0,
  };
}

function targetSettingsStorageKey(sessionId = targetSettingsSessionId) {
  return `whatmoves-target-settings-${sessionId}`;
}

function loadTargetSettings(sessionId) {
  targetSettingsSessionId = sessionId;
  try {
    targetSettings = JSON.parse(sessionStorage.getItem(targetSettingsStorageKey(sessionId)) || "{}");
  } catch (_) {
    targetSettings = {};
  }
}

function persistTargetSettings() {
  if (!targetSettingsSessionId) return;
  sessionStorage.setItem(targetSettingsStorageKey(), JSON.stringify(targetSettings));
}

function restoreGenerationSettings(target) {
  if (!target) return;
  const settings = targetSettings[target.id] || defaultGenerationSettings();
  elements.prompt.value = settings.prompt ?? "";
  elements.negativePrompt.value = settings.negative_prompt ?? "";
  elements.steps.value = settings.steps ?? 40;
  elements.seed.value = settings.seed ?? 42;
  elements.guidanceMode.value = settings.guidance_mode ?? "text_cfg";
  elements.textGuidance.value = settings.text_guidance_scale ?? 3.5;
  elements.motionGuidance.value = settings.motion_guidance_scale ?? 1.0;
  elements.loraScale.value = settings.lora_scale ?? 1.0;
  settingsDirty = false;
}

async function flushTargetSettings() {
  clearTimeout(settingsSaveTimer);
  settingsSaveTimer = null;
  if (!settingsDirty || !sessionState || !formTargetId) return true;
  targetSettings[formTargetId] = generationSettingsPayload();
  persistTargetSettings();
  settingsDirty = false;
  return true;
}

function scheduleTargetSettingsSave() {
  if (!formTargetId) return;
  settingsDirty = true;
  clearTimeout(settingsSaveTimer);
  settingsSaveTimer = setTimeout(flushTargetSettings, 450);
}

async function applyState(next) {
  if (targetSettingsSessionId !== next.id) loadTargetSettings(next.id);
  sessionState = next;
  if (!activeGeneration && next.active_generation) {
    activeGeneration = next.active_generation;
    activeGenerationForm = null;
    activeGenerationStatus = "queued";
    pollGeneration(activeGeneration);
  }
  if (!next.sources.some((source) => source.id === selectedSourceId)) {
    selectedSourceId = next.sources[0]?.id || null;
  }
  renderSourceRail();
  renderTargetRail();
  renderGeneratedRail();

  const source = selectedSource();
  elements.sourceTools.classList.toggle("hidden", !source);
  elements.sourceSettings.classList.toggle("hidden", !source);
  elements.removeSource.classList.toggle("hidden", !source);
  elements.sourceEmpty.classList.toggle("hidden", Boolean(source));
  if (source) {
    source.session_id = next.id;
    const masks = source.masks.filter((mask) => mask.frame_index === source.current_frame)
      .map((mask) => ({ ...mask, displayColor: mask.color }));
    await sourceCanvas.setAsset(source, masks);
    sourceTimeline.setSource(source);
    assetStatus("source", source);
  } else {
    await sourceCanvas.setAsset(null);
    sourceTimeline.setSource(null);
    canvasStatus("source");
  }

  const target = selectedTarget();
  if (target?.id !== formTargetId) {
    formTargetId = target?.id || null;
    restoreGenerationSettings(target);
  }
  elements.targetTools.classList.toggle("hidden", !target);
  elements.removeTarget.classList.toggle("hidden", !target);
  elements.targetEmpty.classList.toggle("hidden", Boolean(target));
  if (target) {
    const hideExampleAssignments = exampleTutorialState !== "complete" && isExampleTutorial(target);
    const masks = target.masks.map((mask) => ({
      ...mask,
      displayColor: hideExampleAssignments ? "#a8afb9" : mappedColor(mask),
    }));
    await targetCanvas.setAsset(target, masks);
    assetStatus("target", target);
  } else {
    await targetCanvas.setAsset(null);
    canvasStatus("target");
  }

  if (!source || (elements.sourceVideo.paused && !sourceTimeline.drag)) {
    $("sourceCanvasHost").classList.remove("playing");
  }

  renderModelStatus();
  updateGenerateState();
  scheduleStatePoll();
  scheduleMappingHint();
}

function renderModelStatus() {
  // Model state is reflected by the generation control and the page content.
}

function assetStatus(kind, asset) {
  if (asset.sam_status === "queued") canvasStatus(kind, "Preparing mask features…");
  else if (asset.sam_status === "error") canvasStatus(kind, asset.sam_error || "SAM failed", true);
  else canvasStatus(kind);
}

function renderSourceRail() {
  elements.sourceRail.replaceChildren();
  sessionState.sources.forEach((source, index) => {
    const button = document.createElement("button");
    button.className = `source-tile${source.id === selectedSourceId ? " selected" : ""}`;
    button.setAttribute("aria-label", `Select source video ${index + 1}`);
    const image = document.createElement("img");
    image.src = source.thumbnail_url;
    image.alt = "";
    button.append(image);
    button.addEventListener("click", async () => {
      if (source.id === selectedSourceId) return;
      await clearCanvasDraft("source");
      selectedSourceId = source.id;
      await applyState(sessionState);
    });
    elements.sourceRail.append(button);
  });
  const upload = document.createElement("button");
  upload.className = "source-upload-tile";
  upload.title = "Upload source video";
  upload.textContent = "+";
  upload.addEventListener("click", () => elements.sourceFile.click());
  elements.sourceRail.append(upload);
}

function renderTargetRail() {
  elements.targetRail.replaceChildren();
  sessionState.targets.forEach((target, index) => {
    const button = document.createElement("button");
    button.className = `target-tile${target.id === sessionState.selected_target_id ? " selected" : ""}`;
    button.setAttribute("aria-label", `Select start frame ${index + 1}`);
    const image = document.createElement("img");
    image.src = target.thumbnail_url;
    image.alt = "";
    button.append(image);
    button.addEventListener("click", async () => {
      if (target.id === sessionState.selected_target_id) return;
      if (!await flushTargetSettings()) return;
      await clearCanvasDraft("target");
      try {
        await applyState(await request(
          `/api/sessions/${sessionState.id}/targets/${target.id}/selection`,
          { method: "PUT" },
        ));
      } catch (error) {
        showToast(error.message);
      }
    });
    elements.targetRail.append(button);
  });
  const upload = document.createElement("button");
  upload.className = "target-upload-tile";
  upload.title = "Upload target image";
  upload.textContent = "+";
  upload.addEventListener("click", () => elements.targetFile.click());
  elements.targetRail.append(upload);
}

function outputUrl(output) {
  if (output.local_url) return output.local_url;
  return `${output.video_url}?v=${encodeURIComponent(output.task_id || output.output_name)}`;
}

async function rememberGeneratedOutput(output) {
  let record = { ...output };
  try {
    const response = await fetch(outputUrl(output));
    if (!response.ok) throw new Error(`Could not cache output (${response.status})`);
    record.local_url = URL.createObjectURL(await response.blob());
  } catch (_) {
    // The current server URL remains usable until a later generation replaces it.
  }
  generatedOutputs.push(record);
  while (generatedOutputs.length > 8) {
    const expired = generatedOutputs.shift();
    if (expired.local_url) URL.revokeObjectURL(expired.local_url);
  }
  renderGeneratedRail();
  return record;
}

function clearGeneratedOutputs() {
  generatedOutputs.forEach((output) => {
    if (output.local_url) URL.revokeObjectURL(output.local_url);
  });
  generatedOutputs = [];
  renderGeneratedRail();
}

function showGeneratedOutput(output) {
  if (!output?.video_url) return;
  elements.resultVideo.pause();
  elements.resultVideo.src = outputUrl(output);
  elements.resultMeta.textContent = output.frames
    ? `${output.frames} frames · ${output.width}×${output.height} · ${output.fps} fps`
    : "";
  elements.resultPanel.classList.remove("hidden");
  elements.resultVideo.load();
  elements.resultVideo.play().catch(() => {});
}

function renderGeneratedRail() {
  const outputs = generatedOutputs;
  elements.generatedRail.replaceChildren();
  elements.generatedRail.classList.toggle("hidden", outputs.length === 0);
  [...outputs].reverse().forEach((output, index) => {
    const button = document.createElement("button");
    button.className = "generated-tile";
    button.setAttribute("aria-label", `Open generated video ${index + 1}`);
    const video = document.createElement("video");
    video.src = outputUrl(output);
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.preload = "metadata";
    button.append(video);
    button.addEventListener("click", () => showGeneratedOutput(output));
    elements.generatedRail.append(button);
  });
}

function scheduleStatePoll() {
  clearTimeout(pollTimer);
  const preparing = sessionState?.targets.some((target) => target.sam_status === "queued") ||
    sessionState?.sources.some((source) => source.sam_status === "queued") ||
    ["loading", "generating"].includes(sessionState?.wan_status) ||
    Boolean(activeGeneration || sessionState?.active_generation);
  const delay = preparing ? 900 : (document.hidden ? 60000 : 30000);
  pollTimer = setTimeout(async () => {
    try {
      await applyState(await request(`/api/sessions/${sessionState.id}`));
    } catch (_) {
      scheduleStatePoll();
    }
  }, delay);
}

async function upload(kind, file) {
  if (!file || !sessionState) return;
  const body = new FormData();
  body.append("file", file);
  canvasStatus(kind, "Uploading…");
  try {
    predictionControllers[kind]?.abort();
    const endpoint = kind === "target" ? "targets" : "sources";
    const next = await request(`/api/sessions/${sessionState.id}/${endpoint}`, { method: "POST", body });
    if (kind === "source") {
      await clearCanvasDraft("source");
      selectedSourceId = next.sources.at(-1)?.id || selectedSourceId;
    }
    if (kind === "target") {
      targetCanvas.clearDraft();
    }
    await applyState(next);
    if (kind === "source") sourceCanvas.setPinned(true);
    if (kind === "target") targetCanvas.setPinned(true);
  } catch (error) {
    canvasStatus(kind);
    showToast(error.message);
  }
}

async function predictMask(kind, payload) {
  const assetId = kind === "target" ? selectedTarget()?.id : selectedSourceId;
  if (!assetId) return;
  hoverControllers[kind]?.abort();
  clearTimeout(hoverTimers[kind]);
  payload.transient = false;
  predictionControllers[kind]?.abort();
  const controller = new AbortController();
  predictionControllers[kind] = controller;
  canvasStatus(kind, "Updating mask…");
  try {
    const result = await request(
      `/api/sessions/${sessionState.id}/assets/${kind}/${assetId}/predict`,
      { ...jsonOptions(payload), signal: controller.signal },
    );
    const canvas = kind === "source" ? sourceCanvas : targetCanvas;
    await canvas.setDraft(result.mask, result.prompt_revision);
    canvasStatus(kind);
  } catch (error) {
    if (error.name === "AbortError" || error.status === 409) return;
    canvasStatus(kind, error.message, true);
    showToast(error.message);
  }
}

function scheduleHover(kind, payload) {
  clearTimeout(hoverTimers[kind]);
  hoverControllers[kind]?.abort();
  hoverControllers[kind] = null;
  const canvas = kind === "source" ? sourceCanvas : targetCanvas;
  if (!payload) {
    canvas.clearHoverDraft();
    return;
  }
  hoverTimers[kind] = setTimeout(async () => {
    const assetId = kind === "target" ? selectedTarget()?.id : selectedSourceId;
    if (!assetId) return;
    const controller = new AbortController();
    hoverControllers[kind] = controller;
    try {
      const result = await request(
        `/api/sessions/${sessionState.id}/assets/${kind}/${assetId}/predict`,
        { ...jsonOptions(payload), signal: controller.signal },
      );
      await canvas.setHoverDraft(result.mask, result.prompt_revision);
    } catch (error) {
      if (error.name !== "AbortError" && error.status !== 409) showToast(error.message);
    }
  }, 85);
}

async function commitMask(kind) {
  const canvas = kind === "source" ? sourceCanvas : targetCanvas;
  const assetId = kind === "target" ? selectedTarget()?.id : selectedSourceId;
  const payload = canvas.promptPayload();
  try {
    const next = await request(
      `/api/sessions/${sessionState.id}/assets/${kind}/${assetId}/masks`,
      jsonOptions({
        asset_revision: payload.asset_revision,
        prompt_revision: payload.prompt_revision,
        frame_index: payload.frame_index,
      }),
    );
    canvas.clearDraft();
    await applyState(next);
  } catch (error) {
    showToast(error.message);
  }
}

async function clearCanvasDraft(kind) {
  const canvas = kind === "source" ? sourceCanvas : targetCanvas;
  const asset = canvas.asset;
  predictionControllers[kind]?.abort();
  hoverControllers[kind]?.abort();
  clearTimeout(hoverTimers[kind]);
  predictionControllers[kind] = null;
  canvas.clearDraft();
  if (!asset || !sessionState) return;
  try {
    await request(
      `/api/sessions/${sessionState.id}/assets/${kind}/${asset.id}/draft?revision=${asset.revision}`,
      { method: "DELETE" },
    );
  } catch (error) {
    if (error.status !== 404 && error.status !== 409) showToast(error.message);
  }
}

async function discardUnrecoverableDrafts(state) {
  const assets = [];
  state.targets.forEach((target) => assets.push(["target", target.id, target.revision]));
  state.sources.forEach((source) => assets.push(["source", source.id, source.revision]));
  await Promise.allSettled(assets.map(([kind, assetId, revision]) => request(
    `/api/sessions/${state.id}/assets/${kind}/${assetId}/draft?revision=${revision}`,
    { method: "DELETE" },
  )));
}

function menuButton(label, action, { danger = false, color = null, image = null, detail = null } = {}) {
  const button = document.createElement("button");
  button.className = `menu-item${danger ? " danger" : ""}`;
  if (image) {
    const preview = document.createElement("img");
    preview.src = image;
    preview.alt = "";
    button.append(preview);
  }
  if (color) {
    const dot = document.createElement("span");
    dot.className = "color-dot";
    dot.style.background = color;
    button.append(dot);
  }
  const copy = document.createElement("span");
  copy.className = "menu-copy";
  const main = document.createElement("span");
  main.textContent = label;
  copy.append(main);
  if (detail) {
    const small = document.createElement("small");
    small.textContent = detail;
    copy.append(small);
  }
  button.append(copy);
  button.addEventListener("click", async (event) => {
    event.stopPropagation();
    hideMenu();
    await action();
  });
  return button;
}

function menuLabel(text) {
  const label = document.createElement("div");
  label.className = "menu-label";
  label.textContent = text;
  return label;
}

function divider() {
  const value = document.createElement("div");
  value.className = "menu-divider";
  return value;
}

function showMenu(children, x, y) {
  elements.contextMenu.replaceChildren(...children);
  elements.contextMenu.classList.remove("hidden");
  const width = elements.contextMenu.offsetWidth;
  const height = elements.contextMenu.offsetHeight;
  elements.contextMenu.style.left = `${Math.min(x + 6, window.innerWidth - width - 8)}px`;
  elements.contextMenu.style.top = `${Math.min(y + 6, window.innerHeight - height - 8)}px`;
}

function hideMenu() {
  elements.contextMenu.classList.add("hidden");
}

function openSourceMenu(masks, x, y) {
  const source = selectedSource();
  if (!source) return;
  const children = [menuLabel(masks.length > 1 ? "Masks here" : "Source mask")];
  masks.forEach((mask) => {
    const index = source.masks.findIndex((value) => value.id === mask.id);
    children.push(menuButton(
      masks.length > 1 ? `Remove S${sessionState.sources.indexOf(source) + 1} · M${index + 1}` : "Remove mask",
      () => removeMask("source", source.id, mask.id),
      { danger: true, color: mask.color },
    ));
  });
  showMenu(children, x, y);
}

function openTargetMenu(masks, x, y, selected = null) {
  const target = selectedTarget();
  if (!target) return;
  const mask = selected || masks[0];
  const targetIndex = target.masks.findIndex((value) => value.id === mask.id);
  const children = [];
  if (masks.length > 1 && !selected) {
    children.push(menuLabel("Target masks here"));
    masks.forEach((candidate) => {
      const index = target.masks.findIndex((value) => value.id === candidate.id);
      children.push(menuButton(`Target mask ${index + 1}`, () => openTargetMenu(masks, x, y, candidate), {
        color: candidate.displayColor,
      }));
    });
    children.push(divider());
  }
  children.push(menuLabel(`Target mask ${targetIndex + 1}`));
  children.push(menuButton("Remove mask", () => removeMask("target", target.id, mask.id), { danger: true }));
  const stored = target.masks.find((value) => value.id === mask.id);
  if (stored?.source_id) {
    children.push(menuButton("Clear motion", () => setMapping(mask.id, null, null)));
  }
  const sources = sourceMaskEntries();
  if (sources.length) {
    children.push(divider(), menuLabel("Use motion from"));
    sources.forEach((entry) => {
      children.push(menuButton(
        `${entry.sourceLabel} · ${entry.maskLabel}`,
        () => setMapping(mask.id, entry.source.id, entry.id),
        { color: entry.color, image: entry.preview_url, detail: entry.source.name },
      ));
    });
  }
  showMenu(children, x, y);
}

async function removeMask(kind, assetId, maskId) {
  try {
    const next = await request(
      `/api/sessions/${sessionState.id}/assets/${kind}/${assetId}/masks/${maskId}`,
      { method: "DELETE" },
    );
    await applyState(next);
  } catch (error) {
    showToast(error.message);
  }
}

async function setMapping(targetMaskId, sourceId, sourceMaskId) {
  const target = selectedTarget();
  if (!target) return false;
  try {
    const next = await request(
      `/api/sessions/${sessionState.id}/targets/${target.id}/masks/${targetMaskId}/mapping`,
      jsonOptions({ source_id: sourceId, source_mask_id: sourceMaskId }, "PUT"),
    );
    targetCanvas.setDropHover(null);
    await applyState(next);
    return true;
  } catch (error) {
    showToast(error.message);
    return false;
  }
}

function maskShape(element, mask, color) {
  element.replaceChildren();
  if (!mask?.alpha) return false;
  let minX = mask.alpha.width;
  let minY = mask.alpha.height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < mask.alpha.height; y += 1) {
    for (let x = 0; x < mask.alpha.width; x += 1) {
      if (!mask.pixels[y * mask.alpha.width + x]) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }
  if (maxX < 0 || maxY < 0) return false;
  const width = maxX - minX + 1;
  const height = maxY - minY + 1;
  const raster = document.createElement("canvas");
  raster.width = width;
  raster.height = height;
  const context = raster.getContext("2d");
  context.fillStyle = color || mask.color || "#25d5f4";
  context.fillRect(0, 0, raster.width, raster.height);
  context.globalCompositeOperation = "destination-in";
  context.drawImage(mask.alpha, minX, minY, width, height, 0, 0, width, height);
  const image = document.createElement("img");
  image.src = raster.toDataURL("image/png");
  image.alt = "";
  element.append(image);
  return true;
}

function hideDragGhost() {
  elements.dragGhost.classList.add("hidden");
  elements.dragGhost.replaceChildren();
  draggedMaskId = null;
}

function playMappingDrop(target, color, x, y) {
  const pop = document.createElement("span");
  pop.className = "mapping-drop-pop";
  pop.style.left = `${x}px`;
  pop.style.top = `${y}px`;
  if (!maskShape(pop, target, color)) return;
  document.body.append(pop);
  pop.addEventListener("animationend", () => pop.remove(), { once: true });
}

function scheduleMappingHint() {
  if (exampleTutorialState === "complete" || exampleTutorialScheduled) return;
  const target = selectedTarget();
  const source = selectedSource();
  if (!isExampleTutorial(target) || !source) return;
  const targetRecord = target.masks.find((value) =>
    value.source_id === source.id && sourceCanvas.getMask(value.source_mask_id)
  );
  const sourceMask = targetRecord && sourceCanvas.getMask(targetRecord.source_mask_id);
  const targetMask = targetRecord && targetCanvas.getMask(targetRecord.id);
  if (!targetRecord || !sourceMask || !targetMask) return;
  exampleTutorialScheduled = true;
  mappingHintTimer = setTimeout(() => {
    const start = sourceCanvas.maskClientCenter(sourceMask.id);
    const end = targetCanvas.maskClientCenter(targetMask.id);
    if (!start || !end) {
      exampleTutorialScheduled = false;
      return;
    }
    const token = document.createElement("span");
    token.className = "mapping-hint-token";
    token.style.left = `${start.x}px`;
    token.style.top = `${start.y}px`;
    token.style.setProperty("--hint-x", `${end.x - start.x}px`);
    token.style.setProperty("--hint-y", `${end.y - start.y}px`);
    maskShape(token, sourceMask, sourceMask.color);
    const pointer = document.createElement("span");
    pointer.className = "mapping-hint-pointer";
    pointer.innerHTML = '<svg viewBox="0 0 24 28" aria-hidden="true"><path d="M3 2v20l5.2-5.1 3.6 8.1 4.1-1.9-3.7-7.8H20L3 2Z"></path></svg>';
    const copy = document.createElement("span");
    copy.className = "mapping-hint-copy";
    copy.textContent = "Drag a source mask onto a target mask";
    elements.mappingHint.replaceChildren(token, pointer, copy);
    pointer.style.left = `${start.x + 8}px`;
    pointer.style.top = `${start.y + 8}px`;
    pointer.style.setProperty("--hint-x", `${end.x - start.x}px`);
    pointer.style.setProperty("--hint-y", `${end.y - start.y}px`);
    elements.mappingHint.classList.remove("hidden");
    token.addEventListener("animationend", () => {
      playMappingDrop(targetMask, sourceMask.color, end.x, end.y);
      exampleTutorialState = "complete";
      elements.mappingHint.classList.add("hidden");
      elements.mappingHint.replaceChildren();
      const currentTarget = selectedTarget();
      if (currentTarget?.id === target.id) {
        const masks = currentTarget.masks.map((mask) => ({
          ...mask,
          displayColor: mappedColor(mask),
        }));
        targetCanvas.setAsset(currentTarget, masks);
      }
    }, { once: true });
  }, 850);
}

function mappingDrag(phase, mask, x, y) {
  if (phase === "cancel") {
    hideDragGhost();
    targetCanvas.setDropHover(null);
    return;
  }
  if (phase === "move") {
    if (draggedMaskId !== mask.id) {
      elements.dragGhost.replaceChildren();
      const shape = document.createElement("span");
      shape.className = "drag-mask-shape";
      maskShape(shape, mask, mask.color);
      elements.dragGhost.append(shape);
      draggedMaskId = mask.id;
    }
    elements.dragGhost.style.left = `${x - 8}px`;
    elements.dragGhost.style.top = `${y - 8}px`;
    elements.dragGhost.classList.remove("hidden");
    const target = targetCanvas.masksAtClient(x, y)[0] || null;
    targetCanvas.setDropHover(target?.id || null);
    return;
  }
  const target = targetCanvas.masksAtClient(x, y)[0] || null;
  hideDragGhost();
  targetCanvas.setDropHover(null);
  if (target) {
    playMappingDrop(target, mask.color, x, y);
    setMapping(target.id, selectedSourceId, mask.id);
  }
}

async function selectSourceFrame(frame, final) {
  const source = selectedSource();
  if (!source || !final) {
    if (source) $("sourceCanvasHost").classList.add("playing");
    return true;
  }
  const selected = Math.max(0, Math.min(source.frame_count - 1, Math.round(frame)));
  $("sourceCanvasHost").classList.add("playing");
  frameController?.abort();
  frameController = new AbortController();
  try {
    canvasStatus("source", "Preparing selected frame…");
    const next = await request(
      `/api/sessions/${sessionState.id}/sources/${source.id}`,
      { ...jsonOptions({ current_frame: selected }, "PATCH"), signal: frameController.signal },
    );
    await applyState(next);
    return true;
  } catch (error) {
    if (error.name === "AbortError" || error.status === 409) return false;
    showToast(error.message);
    await applyState(sessionState);
    return false;
  }
}

async function updateSourceTrim(start, end) {
  const source = selectedSource();
  if (!source) return false;
  const excluded = source.masks.filter((mask) =>
    (mask.frame_index < start || mask.frame_index > end) &&
    sessionState.targets.some((target) => target.masks.some((value) =>
      value.source_id === source.id && value.source_mask_id === mask.id,
    )),
  );
  if (excluded.length && !window.confirm(
    `This trim excludes ${excluded.length} mapped source mask${excluded.length === 1 ? "" : "s"} and will clear those mappings. Continue?`,
  )) return false;
  try {
    const next = await request(
      `/api/sessions/${sessionState.id}/sources/${source.id}`,
      jsonOptions({ trim_start: start, trim_end: end }, "PATCH"),
    );
    await applyState(next);
    return true;
  } catch (error) {
    showToast(error.message);
    await applyState(sessionState);
    return false;
  }
}

function updateGenerateState() {
  const mode = elements.guidanceMode.value;
  const target = selectedTarget();
  const mapped = target?.masks.some((mask) => mask.source_id && mask.source_mask_id);
  const ready = Boolean(target && (mode === "base_cfg" || mapped));
  if (activeGeneration) {
    elements.generateButton.disabled = true;
    elements.generateHint.textContent = "GPU work continues in the background; you can keep editing inputs.";
    return;
  }
  elements.generateButton.disabled = !ready;
  elements.generateHint.textContent = !target
    ? "Upload a target image first."
    : mode === "base_cfg"
      ? "Base Wan ignores source masks."
      : mapped
        ? ""
        : "Create source and target masks, then drag a source mask onto a target mask.";
}

function setGenerateButtonStatus(label, count = "") {
  elements.generateButtonLabel.textContent = label;
  elements.generateButtonCount.textContent = count;
}

async function generate() {
  if (activeGeneration) return;
  const formSignature = generationFormSignature();
  const payload = {
    prompt: elements.prompt.value,
    negative_prompt: elements.negativePrompt.value || null,
    steps: Number.parseInt(elements.steps.value, 10),
    seed: Number.parseInt(elements.seed.value, 10),
    guidance_mode: elements.guidanceMode.value,
    text_guidance_scale: Number.parseFloat(elements.textGuidance.value),
    motion_guidance_scale: Number.parseFloat(elements.motionGuidance.value),
    lora_scale: Number.parseFloat(elements.loraScale.value),
  };
  try {
    activeGeneration = "gradio-pending";
    activeGenerationForm = formSignature;
    activeGenerationStatus = "queued";
    elements.generateButton.disabled = true;
    updateProgress({ stage: "Queued for GPU", progress_current: null, progress_total: null });
    const result = await request(
      `/api/sessions/${sessionState.id}/generate`,
      {
        ...jsonOptions(payload),
        onGradioStatus: (message) => {
          const progress = message.progress_data?.[0];
          if (progress?.desc) {
            updateProgress({
              stage: progress.desc,
              progress_current: progress.index,
              progress_total: progress.length,
            });
          } else if (message.stage === "pending") {
            const position = Number.isFinite(message.position) ? ` · ${message.position + 1} ahead` : "";
            updateProgress({ stage: `Queued for GPU${position}`, progress_current: null, progress_total: null });
          } else if (message.stage === "generating") {
            updateProgress({ stage: "Starting generation", progress_current: null, progress_total: null });
          }
        },
      },
    );
    activeGeneration = result.task_id;
    activeGenerationForm = formSignature;
    activeGenerationStatus = "queued";
    elements.generateButton.disabled = true;
    updateProgress({ stage: "Queued", progress_current: null, progress_total: null });
    elements.generateHint.textContent = "GPU work continues in the background; you can keep editing inputs.";
    renderModelStatus();
    pollGeneration(result.task_id);
  } catch (error) {
    activeGeneration = null;
    activeGenerationForm = null;
    activeGenerationStatus = null;
    updateProgress(null);
    updateGenerateState();
    showToast(error.message);
  }
}

function updateProgress(task = null) {
  if (!task) {
    elements.generateButton.classList.remove("generating");
    elements.generateButtonProgress.classList.remove("indeterminate");
    elements.generateButtonProgress.style.width = "0";
    setGenerateButtonStatus("Generate video");
    return;
  }
  elements.generateButton.classList.add("generating");
  const stage = task.stage || (task.status === "queued" ? "Queued" : "Working");
  const determinate = Number.isFinite(task.progress_current) && Number.isFinite(task.progress_total) && task.progress_total > 0;
  setGenerateButtonStatus(stage, determinate ? `${task.progress_current} / ${task.progress_total}` : "");
  elements.generateButtonProgress.classList.toggle("indeterminate", !determinate);
  if (determinate) {
    elements.generateButtonProgress.style.width = `${Math.max(0, Math.min(100, task.progress_current / task.progress_total * 100))}%`;
  } else {
    elements.generateButtonProgress.style.width = "";
  }
}

async function pollGeneration(taskId) {
  if (activeGeneration !== taskId) return;
  try {
    const task = await request(`/api/tasks/${taskId}`);
    updateProgress(task);
    if (task.status === "queued") {
      activeGenerationStatus = "queued";
    } else if (task.status === "running") {
      activeGenerationStatus = "running";
    } else if (task.status === "complete") {
      const formChanged = activeGenerationForm !== null &&
        activeGenerationForm !== generationFormSignature();
      activeGeneration = null;
      activeGenerationForm = null;
      activeGenerationStatus = null;
      updateProgress(null);
      const previousInputs = task.result.inputs_changed || formChanged;
      const rememberedOutput = await rememberGeneratedOutput(task.result);
      showGeneratedOutput(rememberedOutput);
      if (previousInputs) {
        showToast("Generation finished from the inputs captured when you clicked Generate.");
      }
      try {
        await applyState(await request(`/api/sessions/${sessionState.id}`));
      } catch (_) {
        renderModelStatus();
        updateGenerateState();
      }
      return;
    } else if (["failed", "superseded"].includes(task.status)) {
      activeGeneration = null;
      activeGenerationForm = null;
      activeGenerationStatus = null;
      updateProgress(null);
      showToast(task.error || "Generation failed");
      renderModelStatus();
      updateGenerateState();
      return;
    }
    renderModelStatus();
  } catch (error) {
    activeGeneration = null;
    activeGenerationForm = null;
    activeGenerationStatus = null;
    updateProgress(null);
    showToast(error.message);
    renderModelStatus();
    updateGenerateState();
    return;
  }
  setTimeout(() => pollGeneration(taskId), 1000);
}

function generationFormSignature() {
  return JSON.stringify([
    elements.prompt.value,
    elements.negativePrompt.value,
    elements.guidanceMode.value,
    elements.steps.value,
    elements.seed.value,
    elements.textGuidance.value,
    elements.motionGuidance.value,
    elements.loraScale.value,
  ]);
}

elements.sourceRailUpload.addEventListener("click", () => elements.sourceFile.click());
elements.sourceEmpty.addEventListener("click", () => elements.sourceFile.click());
elements.sourceFile.addEventListener("change", () => {
  upload("source", elements.sourceFile.files[0]);
  elements.sourceFile.value = "";
});
elements.targetEmpty.addEventListener("click", () => elements.targetFile.click());
elements.targetFile.addEventListener("change", () => {
  const file = elements.targetFile.files[0];
  elements.targetFile.value = "";
  flushTargetSettings().then((saved) => {
    if (saved) upload("target", file);
  });
});
elements.removeSource.addEventListener("click", async () => {
  const source = selectedSource();
  if (!source) return;
  predictionControllers.source?.abort();
  try {
    const next = await request(`/api/sessions/${sessionState.id}/sources/${source.id}`, { method: "DELETE" });
    await applyState(next);
  } catch (error) {
    showToast(error.message);
  }
});
elements.removeTarget.addEventListener("click", async () => {
  const target = selectedTarget();
  if (!target) return;
  predictionControllers.target?.abort();
  try {
    const next = await request(`/api/sessions/${sessionState.id}/targets/${target.id}`, { method: "DELETE" });
    await applyState(next);
  } catch (error) {
    showToast(error.message);
  }
});

elements.resultClose.addEventListener("click", () => {
  elements.resultVideo.pause();
  setMediaLoading("result", "video", false);
  elements.resultPanel.classList.add("hidden");
});

elements.startOver.addEventListener("click", async () => {
  if (!sessionState || !window.confirm("Remove the current media and start over?")) return;
  predictionControllers.source?.abort();
  predictionControllers.target?.abort();
  activeGeneration = null;
  activeGenerationForm = null;
  activeGenerationStatus = null;
  try {
    const oldSettingsKey = targetSettingsStorageKey(sessionState.id);
    await request(`/api/sessions/${sessionState.id}`, { method: "DELETE" });
    let next = await request("/api/sessions", { method: "POST" });
    for (const source of [...next.sources]) {
      next = await request(`/api/sessions/${next.id}/sources/${source.id}`, { method: "DELETE" });
    }
    for (const target of [...next.targets]) {
      next = await request(`/api/sessions/${next.id}/targets/${target.id}`, { method: "DELETE" });
    }
    sessionStorage.setItem("whatmoves-session", next.id);
    sessionStorage.removeItem(oldSettingsKey);
    selectedSourceId = null;
    formTargetId = null;
    settingsDirty = false;
    targetSettingsSessionId = null;
    targetSettings = {};
    exampleTutorialState = "complete";
    exampleTutorialScheduled = false;
    clearGeneratedOutputs();
    elements.resultVideo.pause();
    elements.resultVideo.removeAttribute("src");
    setMediaLoading("result", "video", false);
    elements.resultPanel.classList.add("hidden");
    updateProgress(null);
    await applyState(next);
  } catch (error) {
    showToast(error.message);
  }
});

elements.sourceMaskMode.addEventListener("click", async () => {
  if (sourceCanvas.pinned) {
    sourceCanvas.setPinned(false);
    return;
  }
  sourceTimeline.pause();
  if (await selectSourceFrame(sourceTimeline.currentFrame, true)) {
    sourceCanvas.setPinned(true);
  }
});
elements.targetMaskMode.addEventListener("click", () => targetCanvas.togglePinned());
elements.sourceAddMask.addEventListener("click", () => commitMask("source"));
elements.targetAddMask.addEventListener("click", () => commitMask("target"));
elements.sourceCancelDraft.addEventListener("click", () => clearCanvasDraft("source"));
elements.targetCancelDraft.addEventListener("click", () => clearCanvasDraft("target"));
[
  elements.prompt,
  elements.negativePrompt,
  elements.guidanceMode,
  elements.steps,
  elements.seed,
  elements.textGuidance,
  elements.motionGuidance,
  elements.loraScale,
].forEach((element) => {
  element.addEventListener("input", scheduleTargetSettingsSave);
  element.addEventListener("change", scheduleTargetSettingsSave);
});
elements.guidanceMode.addEventListener("change", updateGenerateState);
elements.generateButton.addEventListener("click", generate);

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    hideMenu();
    if (sourceCanvas.isMasking()) clearCanvasDraft("source");
    if (targetCanvas.isMasking()) clearCanvasDraft("target");
  }
});
document.addEventListener("pointerdown", (event) => {
  if (!elements.contextMenu.contains(event.target)) hideMenu();
});

async function start() {
  try {
    let initial = null;
    let resumed = false;
    const savedSession = sessionStorage.getItem("whatmoves-session");
    if (savedSession) {
      try {
        initial = await request(`/api/sessions/${savedSession}`);
        resumed = true;
      } catch (error) {
        if (error.status !== 404) throw error;
        sessionStorage.removeItem("whatmoves-session");
      }
    }
    if (!initial) {
      initial = await request("/api/sessions", { method: "POST" });
      sessionStorage.setItem("whatmoves-session", initial.id);
    }
    if (resumed) await discardUnrecoverableDrafts(initial);
    await applyState(initial);
    if (!activeGeneration && initial.latest_generation) {
      activeGeneration = initial.latest_generation;
      activeGenerationForm = null;
      activeGenerationStatus = "queued";
      updateGenerateState();
      pollGeneration(activeGeneration);
    }
  } catch (error) {
    const message = `Could not start the app: ${error.message}`;
    elements.generateHint.textContent = message;
    showToast(message);
  }
}

start();
