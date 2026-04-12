import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    getWidget,
    isSelectorTargetNodeDef,
    normalizeSelectedPath,
    normalizeSha256,
    resolvePreferredSelectorSlot,
    resolveSelectorTarget,
} from "./selector_targets.js";

const MENU_LABEL = "View Model Info...";
const STYLE_ID = "IPT-selector-model-info-window-style";
const WINDOW_MIN_WIDTH = 420;
const WINDOW_MIN_HEIGHT = 280;
const WINDOW_DEFAULT_WIDTH = 984;
const WINDOW_DEFAULT_HEIGHT = 616;
const INFO_POLL_MAX_ATTEMPTS = 8;
const INFO_POLL_INTERVAL_MS = 250;
const THUMBNAIL_POLL_MAX_ATTEMPTS = 12;
const THUMBNAIL_POLL_INTERVAL_MS = 350;
const LORA_TAG_ENSURE_TIMEOUT_MS = 2500;
const LORA_TAG_TERMINAL_STATES = new Set(["ready", "empty", "no_metadata", "unsupported", "error"]);
const MISSING_LOCAL_MESSAGE = "This model file is not available in the current environment.";
const RUNTIME_SETTING_CLIP_LAST_LAYER_KEY = "stop_at_clip_layer";
const RUNTIME_SETTING_SD3_SHIFT_KEY = "model_sampling_sd3_shift";
const COPYABLE_HASH_BUTTONS = [
    { algo: "sha256", label: "SHA256" },
    { algo: "crc32", label: "CRC32" },
    { algo: "blake3", label: "BLAKE3" },
    { algo: "autov3", label: "AUTOV3" },
    { algo: "autov1", label: "AUTOV1" },
    { algo: "autov2", label: "AUTOV2" },
];

function nextWindowZIndex() {
    if (typeof window === "undefined") {
        return 3000;
    }
    const current = Number(window.__iisWindowZIndex ?? 3000);
    const next = Number.isFinite(current) ? current + 1 : 3001;
    window.__iisWindowZIndex = next;
    return next;
}

function installStyle() {
    if (document.getElementById(STYLE_ID)) {
        return;
    }

    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .IPT-model-info-window {
            position: fixed;
            left: 48px;
            top: 48px;
            width: ${WINDOW_DEFAULT_WIDTH}px;
            height: ${WINDOW_DEFAULT_HEIGHT}px;
            min-width: ${WINDOW_MIN_WIDTH}px;
            min-height: ${WINDOW_MIN_HEIGHT}px;
            max-width: calc(100vw - 16px);
            max-height: calc(100vh - 16px);
            background: #15161a;
            border: 1px solid #3a3e49;
            border-radius: 8px;
            color: #e9edf6;
            box-shadow: 0 10px 35px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            resize: both;
        }
        .IPT-model-info-window-header {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 10px;
            background: #1d2028;
            border-bottom: 1px solid #2e3440;
            cursor: move;
            user-select: none;
        }
        .IPT-model-info-window-title {
            font-size: 14px;
            font-weight: 700;
            white-space: nowrap;
            overflow-x: auto;
            overflow-y: hidden;
            flex: 1 1 auto;
            user-select: text;
            cursor: text;
            scrollbar-width: none;
            -ms-overflow-style: none;
        }
        .IPT-model-info-window-title::-webkit-scrollbar {
            width: 0;
            height: 0;
            display: none;
        }
        .IPT-model-info-window-actions {
            display: flex;
            gap: 6px;
            flex: 0 0 auto;
        }
        .IPT-model-info-window-actions button {
            padding: 3px 8px;
            border-radius: 4px;
            border: 1px solid #4a5160;
            background: #202636;
            color: #e9edf6;
            cursor: pointer;
        }
        .IPT-model-info-window-actions button:hover {
            background: #2a3347;
        }
        .IPT-model-info-window-content {
            display: flex;
            flex-direction: column;
            gap: 12px;
            flex: 1 1 auto;
            overflow: auto;
            padding: 12px;
        }
        .IPT-model-info-status {
            font-size: 12px;
            color: #aeb7c8;
        }
        .IPT-model-info-missing-row {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: #ffd9a6;
            background: rgba(138, 87, 27, 0.18);
            border: 1px solid rgba(201, 138, 65, 0.38);
            border-radius: 6px;
            padding: 8px 10px;
        }
        .IPT-model-info-inline-link {
            color: #7fc0ff;
            text-decoration: underline;
            cursor: pointer;
            border: 0;
            background: transparent;
            padding: 0;
            font: inherit;
        }
        .IPT-model-info-inline-link:disabled {
            color: #77829b;
            text-decoration: none;
            cursor: default;
        }
        .IPT-model-info-fields {
            display: grid;
            grid-template-columns: minmax(82px, 108px) minmax(0, 1fr);
            gap: 7px 10px;
            align-items: start;
        }
        .IPT-model-info-main {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(260px, 360px);
            align-items: start;
            gap: 12px;
        }
        .IPT-model-info-main-left {
            min-width: 0;
        }
        .IPT-model-info-preview {
            align-self: start;
        }
        .IPT-model-info-preview-frame {
            width: 100%;
            height: 250px;
            border: 1px solid #303748;
            border-radius: 6px;
            background: #10141d;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            position: relative;
        }
        .IPT-model-info-preview-image {
            display: none;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .IPT-model-info-preview-status {
            color: #aeb7c8;
            font-size: 12px;
            text-align: center;
            padding: 0 12px;
        }
        .IPT-model-info-field-label {
            color: #8da0c2;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
            font-size: 12px;
            text-align: right;
            justify-self: end;
            padding-right: 2px;
            margin-top: 1px;
        }
        .IPT-model-info-field-value {
            color: #f2f6ff;
            word-break: break-word;
            font-size: 13px;
        }
        .IPT-model-info-field-text {
            min-width: 0;
        }
        .IPT-model-info-field-value a {
            color: #6eb8ff;
            text-decoration: underline;
        }
        .IPT-model-info-hash-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 8px;
        }
        .IPT-model-info-hash-button {
            padding: 3px 8px;
            border-radius: 4px;
            border: 1px solid #4a5160;
            background: #202636;
            color: #f2f6ff;
            cursor: pointer;
            font-size: 12px;
        }
        .IPT-model-info-hash-button:hover {
            background: #2a3347;
        }
        .IPT-model-info-lora-header {
            display: flex;
            gap: 8px;
            align-items: center;
            justify-content: space-between;
        }
        .IPT-model-info-lora-title {
            font-weight: 700;
            font-size: 13px;
        }
        .IPT-model-info-copy-button {
            padding: 3px 10px;
            border-radius: 4px;
            border: 1px solid #4a5160;
            background: #202636;
            color: #f2f6ff;
            cursor: pointer;
        }
        .IPT-model-info-copy-button:disabled {
            opacity: 0.5;
            cursor: default;
        }
        .IPT-model-info-settings {
            display: flex;
            flex-direction: column;
            gap: 10px;
            border: 1px solid #313748;
            border-radius: 6px;
            background: #171c27;
            padding: 10px;
        }
        .IPT-model-info-settings-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
        }
        .IPT-model-info-settings-title {
            font-weight: 700;
            font-size: 13px;
        }
        .IPT-model-info-settings-actions {
            display: flex;
            gap: 8px;
        }
        .IPT-model-info-settings-actions button {
            padding: 3px 10px;
            border-radius: 4px;
            border: 1px solid #4a5160;
            background: #202636;
            color: #f2f6ff;
            cursor: pointer;
        }
        .IPT-model-info-settings-actions button:disabled {
            opacity: 0.5;
            cursor: default;
        }
        .IPT-model-info-settings-grid {
            display: grid;
            grid-template-columns: minmax(120px, 160px) minmax(0, 1fr);
            gap: 8px 10px;
            align-items: center;
        }
        .IPT-model-info-settings-label {
            color: #9fb0ce;
            font-size: 12px;
        }
        .IPT-model-info-settings-input {
            width: 100%;
            min-width: 0;
            border: 1px solid #4a5160;
            border-radius: 4px;
            background: #10141d;
            color: #f2f6ff;
            padding: 6px 8px;
        }
        .IPT-model-info-settings-help {
            color: #aeb7c8;
            font-size: 12px;
        }
        .IPT-model-info-settings-status {
            color: #aeb7c8;
            font-size: 12px;
        }
        .IPT-model-info-settings-status.IPT-model-info-settings-status-error {
            color: #ffb4b4;
        }
        .IPT-model-info-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .IPT-model-info-tag {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border: 1px solid #4d5d74;
            border-radius: 6px;
            background: #2a3444;
            color: #e8efff;
            cursor: pointer;
            padding: 3px 8px;
            font-size: 13px;
        }
        .IPT-model-info-tag:hover {
            filter: brightness(1.06);
        }
        .IPT-model-info-tag.IPT-model-info-tag-selected {
            border-color: #5a9de8;
            background: #3d7fcd;
            color: #f8fbff;
            box-shadow: 0 0 0 1px rgba(29, 67, 112, 0.45) inset;
        }
        .IPT-model-info-tag-frequency {
            min-width: 18px;
            padding: 1px 5px;
            border-radius: 10px;
            text-align: center;
            color: #dcebff;
            background: #2f73bf;
            font-size: 12px;
            line-height: 1.2;
        }
        @media (max-width: 768px) {
            .IPT-model-info-window {
                left: 8px !important;
                top: 8px !important;
                width: calc(100vw - 16px);
                height: calc(100vh - 16px);
            }
            .IPT-model-info-fields {
                grid-template-columns: 1fr;
            }
            .IPT-model-info-main {
                grid-template-columns: 1fr;
            }
            .IPT-model-info-preview-frame {
                height: 220px;
            }
            .IPT-model-info-settings-grid {
                grid-template-columns: 1fr;
            }
        }
    `;
    document.body.appendChild(style);
}

function normalizeHashHints(value) {
    if (!Array.isArray(value)) {
        return [];
    }
    return value
        .map((item) => {
            const algo = String(item?.algo ?? "").trim().toLowerCase();
            const digest = String(item?.value ?? "").trim().toLowerCase();
            if (!algo || !digest || !/^[0-9a-f]+$/.test(digest)) {
                return null;
            }
            return { algo, value: digest };
        })
        .filter(Boolean);
}

function normalizeCopyableHashes(value) {
    if (!value || typeof value !== "object") {
        return [];
    }
    return COPYABLE_HASH_BUTTONS
        .map(({ algo, label }) => {
            const digest = String(value?.[algo] ?? "").trim().toLowerCase();
            if (!digest || !/^[0-9a-f]+$/.test(digest)) {
                return null;
            }
            return {
                algo,
                label,
                value: digest.toUpperCase(),
            };
        })
        .filter(Boolean);
}

function getSha256Widget(node, slot) {
    return getWidget(node, slot?.sha256WidgetName);
}

function resolveSelectedSha256(node, slot) {
    const widget = getSha256Widget(node, slot);
    if (!widget) {
        return "";
    }
    return normalizeSha256(widget.value);
}

function setSha256WidgetValue(node, slot, nextValue) {
    const widget = getSha256Widget(node, slot);
    if (!widget) {
        return;
    }
    const normalized = normalizeSha256(nextValue);
    if (!normalized || String(widget.value ?? "") === normalized) {
        return;
    }
    widget.value = normalized;
    node.setDirtyCanvas?.(true, true);
}

function resolveSelectedPath(node, slot) {
    const widget = getWidget(node, slot?.widgetName);
    if (!widget) {
        return "";
    }
    return normalizeSelectedPath(widget.value);
}

async function fetchModelReference({
    folderName,
    relativePath,
    sha256,
    nameRaw,
    hashHints,
    includeLoraTags = false,
    ensureLoraTags = false,
    ensureTimeoutMs = null,
}) {
    const response = await api.fetchApi("/ipt/model-reference/resolve", {
        method: "POST",
        headers: {
            "content-type": "application/json",
        },
        body: JSON.stringify({
            folder_name: folderName,
            relative_path: relativePath || undefined,
            sha256: normalizeSha256(sha256) || undefined,
            name_raw: nameRaw || relativePath || undefined,
            hash_hints: Array.isArray(hashHints) ? hashHints : undefined,
            enqueue_local_hash: true,
            resolve_remote: true,
            include_lora_tags: includeLoraTags,
            ensure_lora_tags: includeLoraTags && ensureLoraTags,
            ensure_timeout_ms: includeLoraTags && Number.isFinite(Number(ensureTimeoutMs))
                ? Math.max(0, Math.trunc(Number(ensureTimeoutMs)))
                : undefined,
        }),
    });
    if (!response.ok) {
        let errorText = `HTTP ${response.status}`;
        try {
            const body = await response.json();
            if (body?.error) {
                errorText = String(body.error);
            }
        } catch {
            // Use default error text.
        }
        throw new Error(errorText);
    }
    return response.json();
}

async function fetchModelThumbnail(folderName, relativePath) {
    const query = new URLSearchParams({
        folder_name: folderName,
        relative_path: relativePath,
    });
    return api.fetchApi(`/ipt/model-metadata/model-thumbnail?${query.toString()}`);
}

async function upsertModelRuntimeSettings({ folderName, relativePath, runtimeSettings }) {
    const response = await api.fetchApi("/ipt/model-runtime-settings/upsert", {
        method: "POST",
        headers: {
            "content-type": "application/json",
        },
        body: JSON.stringify({
            folder_name: folderName,
            relative_path: relativePath,
            runtime_settings: runtimeSettings || {},
        }),
    });
    if (!response.ok) {
        let errorText = `HTTP ${response.status}`;
        try {
            const body = await response.json();
            if (body?.error) {
                errorText = String(body.error);
            }
        } catch {
            // Use default error text.
        }
        throw new Error(errorText);
    }
    return response.json();
}

function waitMs(ms) {
    return new Promise((resolve) => {
        window.setTimeout(resolve, ms);
    });
}

function normalizeLoraTagsState(value) {
    const state = String(value ?? "").trim().toLowerCase();
    if (!state) {
        return "pending";
    }
    return state;
}

function isTerminalLoraTagsState(value) {
    return LORA_TAG_TERMINAL_STATES.has(normalizeLoraTagsState(value));
}

function getEmptyLoraTagsMessage(state) {
    const normalized = normalizeLoraTagsState(state);
    if (normalized === "pending") {
        return "LoRA tags are still being analyzed. Please wait and press Refresh.";
    }
    if (normalized === "no_metadata") {
        return "No local LoRA metadata was found for this model.";
    }
    if (normalized === "unsupported") {
        return "LoRA tags are available only for .safetensors files.";
    }
    if (normalized === "error") {
        return "Failed to analyze LoRA tags.";
    }
    return "No rows in lora_tag_frequency for this model.";
}

async function copyToClipboard(text) {
    const value = String(text ?? "");
    if (!value) {
        return false;
    }

    if (navigator?.clipboard?.writeText) {
        try {
            await navigator.clipboard.writeText(value);
            return true;
        } catch {
            // Fall back to execCommand.
        }
    }

    const textarea = document.createElement("textarea");
    textarea.value = value;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();

    let copied = false;
    try {
        copied = document.execCommand("copy");
    } catch {
        copied = false;
    } finally {
        document.body.removeChild(textarea);
    }
    return copied;
}

function parsePositiveInt(value) {
    const parsed = Number.parseInt(String(value ?? "").trim(), 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        return null;
    }
    return parsed;
}

function parseNullableInt(value) {
    const text = String(value ?? "").trim();
    if (!text) {
        return null;
    }
    const parsed = Number.parseInt(text, 10);
    if (!Number.isFinite(parsed)) {
        return null;
    }
    return parsed;
}

function parseNullableFloat(value) {
    const text = String(value ?? "").trim();
    if (!text) {
        return null;
    }
    const parsed = Number.parseFloat(text);
    if (!Number.isFinite(parsed)) {
        return null;
    }
    return parsed;
}

function buildCivitaiModelUrl(modelInfo) {
    const modelId = parsePositiveInt(modelInfo?.civitai_model_version_model_id);
    const modelVersionId = parsePositiveInt(modelInfo?.civitai_model_version_model_version_id);
    if (modelId === null || modelVersionId === null) {
        return "";
    }
    return `https://civitai.com/models/${modelId}?modelVersionId=${modelVersionId}`;
}

class ModelInfoWindow {
    constructor(node, slot, source) {
        this.node = node;
        this.slot = slot;
        this.source = {
            relativePath: String(source?.relativePath ?? source?.relative_path ?? "").trim(),
            sha256: normalizeSha256(source?.sha256),
            nameRaw: String(source?.nameRaw ?? source?.name_raw ?? "").trim(),
            hashHints: normalizeHashHints(source?.hashHints ?? source?.hash_hints ?? []),
        };
        this.selectedTags = new Set();
        this.tagOrder = [];
        this.element = null;
        this.content = null;
        this.status = null;
        this.copyButton = null;
        this.titleElement = null;
        this.thumbnailObjectUrl = "";
        this.thumbnailLoadToken = 0;
    }

    show() {
        installStyle();

        const panel = document.createElement("section");
        panel.className = "IPT-model-info-window";
        panel.style.left = "48px";
        panel.style.top = "48px";
        panel.style.zIndex = String(nextWindowZIndex());

        const header = document.createElement("header");
        header.className = "IPT-model-info-window-header";

        const title = document.createElement("div");
        title.className = "IPT-model-info-window-title";
        title.textContent = this.buildTitleText();

        const actions = document.createElement("div");
        actions.className = "IPT-model-info-window-actions";

        const refreshButton = document.createElement("button");
        refreshButton.type = "button";
        refreshButton.textContent = "Refresh";
        refreshButton.addEventListener("click", () => {
            this.load();
        });

        const closeButton = document.createElement("button");
        closeButton.type = "button";
        closeButton.textContent = "Close";
        closeButton.addEventListener("click", () => this.close());

        actions.append(refreshButton, closeButton);
        header.append(title, actions);

        const content = document.createElement("div");
        content.className = "IPT-model-info-window-content";

        panel.append(header, content);
        document.body.appendChild(panel);
        this.centerWindow(panel);

        panel.addEventListener("mousedown", () => {
            panel.style.zIndex = String(nextWindowZIndex());
        });

        this.attachDrag(panel, header);
        this.element = panel;
        this.content = content;
        this.titleElement = title;
        this.scrollTitleToEnd(title);

        this.load();
    }

    buildTitleText() {
        const label = this.source.relativePath || this.source.nameRaw || this.source.sha256 || "(unresolved reference)";
        return `${label} (${this.slot.folderName})`;
    }

    centerWindow(panel) {
        const width = panel.offsetWidth;
        const height = panel.offsetHeight;
        const maxLeft = Math.max(0, window.innerWidth - width);
        const maxTop = Math.max(0, window.innerHeight - height);
        panel.style.left = `${Math.floor(maxLeft / 2)}px`;
        panel.style.top = `${Math.floor(maxTop / 2)}px`;
    }

    scrollTitleToEnd(titleElement) {
        const apply = () => {
            titleElement.scrollLeft = titleElement.scrollWidth;
        };
        requestAnimationFrame(() => {
            apply();
            requestAnimationFrame(apply);
        });
    }

    attachDrag(panel, handle) {
        const onMouseDown = (event) => {
            if (event.button !== 0) {
                return;
            }
            const target = event.target instanceof Element ? event.target : null;
            if (target?.closest("button")) {
                return;
            }
            if (target?.closest(".IPT-model-info-window-title")) {
                return;
            }

            event.preventDefault();
            const rect = panel.getBoundingClientRect();
            const startOffsetX = event.clientX - rect.left;
            const startOffsetY = event.clientY - rect.top;

            const onMouseMove = (moveEvent) => {
                const maxLeft = Math.max(0, window.innerWidth - panel.offsetWidth);
                const maxTop = Math.max(0, window.innerHeight - panel.offsetHeight);
                const nextLeft = Math.min(maxLeft, Math.max(0, moveEvent.clientX - startOffsetX));
                const nextTop = Math.min(maxTop, Math.max(0, moveEvent.clientY - startOffsetY));
                panel.style.left = `${nextLeft}px`;
                panel.style.top = `${nextTop}px`;
            };

            const onMouseUp = () => {
                window.removeEventListener("mousemove", onMouseMove);
                window.removeEventListener("mouseup", onMouseUp);
            };

            window.addEventListener("mousemove", onMouseMove);
            window.addEventListener("mouseup", onMouseUp);
        };

        handle.addEventListener("mousedown", onMouseDown);
    }

    close() {
        this.cancelThumbnailLoad();
        this.revokeThumbnailObjectUrl();
        this.element?.remove();
        this.element = null;
    }

    setStatus(text) {
        if (!this.content) {
            return;
        }
        if (!this.status) {
            this.status = document.createElement("div");
            this.status.className = "IPT-model-info-status";
            this.content.appendChild(this.status);
        }
        this.status.textContent = text;
    }

    clearContent() {
        if (!this.content) {
            return;
        }
        this.cancelThumbnailLoad();
        this.revokeThumbnailObjectUrl();
        this.content.replaceChildren();
        this.status = null;
        this.copyButton = null;
        this.selectedTags.clear();
        this.tagOrder = [];
    }

    renderMissingLocalNotice(downloadCandidate) {
        if (!this.content) {
            return;
        }

        const row = document.createElement("div");
        row.className = "IPT-model-info-missing-row";

        const text = document.createElement("span");
        text.textContent = MISSING_LOCAL_MESSAGE;
        row.appendChild(text);

        const candidate = downloadCandidate && typeof downloadCandidate === "object" ? downloadCandidate : null;
        const url = String(candidate?.url ?? "").trim();
        const directory = String(candidate?.directory ?? "").trim();
        const name = String(candidate?.name ?? "").trim();
        if (url && directory && name) {
            const link = document.createElement("button");
            link.type = "button";
            link.className = "IPT-model-info-inline-link";
            link.textContent = "Download";
            link.addEventListener("click", async () => {
                link.disabled = true;
                const previous = link.textContent;
                try {
                    await startDownloadCandidate({
                        url,
                        directory,
                        name,
                    });
                    link.textContent = "Opened";
                } catch {
                    link.textContent = "Failed";
                } finally {
                    window.setTimeout(() => {
                        link.disabled = false;
                        link.textContent = previous;
                    }, 1200);
                }
            });
            row.appendChild(link);
        }

        this.content.appendChild(row);
    }

    renderModelFields(modelInfo, parentElement = null) {
        if (!this.content) {
            return;
        }

        const civitaiUrl = buildCivitaiModelUrl(modelInfo);
        const copyableHashes = normalizeCopyableHashes(modelInfo?.copyable_hashes);
        const fields = [
            ["Name", modelInfo?.civitai_model_name, civitaiUrl],
            ["Version", modelInfo?.civitai_model_version_name],
            ["Base Model", modelInfo?.civitai_model_version_base_model],
        ];

        const table = document.createElement("div");
        table.className = "IPT-model-info-fields";
        for (const [labelText, valueRaw, linkUrl] of fields) {
            const label = document.createElement("div");
            label.className = "IPT-model-info-field-label";
            label.textContent = labelText;

            const value = document.createElement("div");
            value.className = "IPT-model-info-field-value";
            const textNode = document.createElement("div");
            textNode.className = "IPT-model-info-field-text";
            const text = String(valueRaw ?? "").trim();
            if (text && linkUrl) {
                const anchor = document.createElement("a");
                anchor.href = linkUrl;
                anchor.target = "_blank";
                anchor.rel = "noopener noreferrer";
                anchor.textContent = text;
                textNode.appendChild(anchor);
            } else {
                textNode.textContent = text || "-";
            }
            value.appendChild(textNode);

            if (labelText === "Base Model" && copyableHashes.length > 0) {
                const hashButtons = document.createElement("div");
                hashButtons.className = "IPT-model-info-hash-buttons";
                for (const hashEntry of copyableHashes) {
                    const button = document.createElement("button");
                    button.type = "button";
                    button.className = "IPT-model-info-hash-button";
                    button.textContent = hashEntry.label;
                    button.title = hashEntry.value;
                    button.addEventListener("click", async () => {
                        button.disabled = true;
                        const originalLabel = hashEntry.label;
                        const copied = await copyToClipboard(hashEntry.value);
                        button.textContent = copied ? "Copied" : "Failed";
                        window.setTimeout(() => {
                            button.disabled = false;
                            button.textContent = originalLabel;
                        }, 900);
                    });
                    hashButtons.appendChild(button);
                }
                value.appendChild(hashButtons);
            }

            table.append(label, value);
        }
        (parentElement || this.content).appendChild(table);
    }

    cancelThumbnailLoad() {
        this.thumbnailLoadToken += 1;
    }

    revokeThumbnailObjectUrl() {
        if (this.thumbnailObjectUrl) {
            URL.revokeObjectURL(this.thumbnailObjectUrl);
            this.thumbnailObjectUrl = "";
        }
    }

    renderModelTopSection(folderName, relativePath, modelInfo, { showPreview = true } = {}) {
        if (!this.content) {
            return;
        }

        const main = document.createElement("div");
        main.className = "IPT-model-info-main";

        const left = document.createElement("div");
        left.className = "IPT-model-info-main-left";
        this.renderModelFields(modelInfo, left);

        const preview = document.createElement("div");
        preview.className = "IPT-model-info-preview";

        const frame = document.createElement("div");
        frame.className = "IPT-model-info-preview-frame";

        const image = document.createElement("img");
        image.className = "IPT-model-info-preview-image";
        image.alt = `Preview image for ${relativePath}`;
        image.loading = "lazy";
        image.decoding = "async";

        const status = document.createElement("div");
        status.className = "IPT-model-info-preview-status";
        status.textContent = "Loading preview image...";

        if (showPreview) {
            frame.append(image, status);
            preview.append(frame);
            main.append(left, preview);
        } else {
            main.append(left);
        }
        this.content.appendChild(main);

        if (showPreview) {
            void this.loadThumbnailImage(folderName, relativePath, image, status);
        }
    }

    async loadThumbnailImage(folderName, relativePath, imageElement, statusElement) {
        const token = this.thumbnailLoadToken;
        const setStatus = (text) => {
            if (this.thumbnailLoadToken !== token || !statusElement) {
                return;
            }
            statusElement.textContent = text;
            statusElement.style.display = text ? "" : "none";
        };

        for (let attempt = 1; attempt <= THUMBNAIL_POLL_MAX_ATTEMPTS; attempt += 1) {
            if (!this.element || this.thumbnailLoadToken !== token) {
                return;
            }

            let response = null;
            try {
                response = await fetchModelThumbnail(folderName, relativePath);
            } catch {
                if (attempt < THUMBNAIL_POLL_MAX_ATTEMPTS) {
                    setStatus(`Loading preview image... (${attempt + 1}/${THUMBNAIL_POLL_MAX_ATTEMPTS})`);
                    await waitMs(THUMBNAIL_POLL_INTERVAL_MS);
                    continue;
                }
                setStatus("Failed to load preview image.");
                return;
            }

            if (!this.element || this.thumbnailLoadToken !== token) {
                return;
            }

            if (response.status === 200) {
                const blob = await response.blob();
                if (!this.element || this.thumbnailLoadToken !== token) {
                    return;
                }
                if (!(blob instanceof Blob) || blob.size <= 0) {
                    setStatus("Preview image payload is empty.");
                    return;
                }

                const objectUrl = URL.createObjectURL(blob);
                this.revokeThumbnailObjectUrl();
                this.thumbnailObjectUrl = objectUrl;
                imageElement.src = objectUrl;
                imageElement.style.display = "block";
                setStatus("");
                return;
            }

            if (response.status === 404) {
                setStatus("No preview image available.");
                return;
            }

            if (response.status === 202) {
                if (attempt < THUMBNAIL_POLL_MAX_ATTEMPTS) {
                    setStatus(`Loading preview image... (${attempt + 1}/${THUMBNAIL_POLL_MAX_ATTEMPTS})`);
                    await waitMs(THUMBNAIL_POLL_INTERVAL_MS);
                    continue;
                }
                setStatus("Preview image is still preparing. Press Refresh.");
                return;
            }

            setStatus(`Failed to load preview image: HTTP ${response.status}`);
            return;
        }
    }

    renderRuntimeSettingsSection({
        folderName,
        relativePath,
        runtimeSettings,
        editable,
        hasLocalFile,
    }) {
        if (!this.content) {
            return;
        }
        if (folderName !== "checkpoints" && folderName !== "diffusion_models") {
            return;
        }

        const section = document.createElement("section");
        section.className = "IPT-model-info-settings";

        const header = document.createElement("div");
        header.className = "IPT-model-info-settings-header";

        const title = document.createElement("div");
        title.className = "IPT-model-info-settings-title";
        title.textContent = "Runtime settings";

        const actions = document.createElement("div");
        actions.className = "IPT-model-info-settings-actions";

        const saveButton = document.createElement("button");
        saveButton.type = "button";
        saveButton.textContent = "Save";

        const resetButton = document.createElement("button");
        resetButton.type = "button";
        resetButton.textContent = "Reset";

        actions.append(saveButton, resetButton);
        header.append(title, actions);
        section.appendChild(header);

        const help = document.createElement("div");
        help.className = "IPT-model-info-settings-help";
        help.textContent = folderName === "checkpoints"
            ? "Checkpoint applies CLIP Set Last Layer and ModelSamplingSD3."
            : "Diffusion Model applies ModelSamplingSD3.";
        section.appendChild(help);

        const grid = document.createElement("div");
        grid.className = "IPT-model-info-settings-grid";

        let clipLastLayerInput = null;
        if (folderName === "checkpoints") {
            const label = document.createElement("label");
            label.className = "IPT-model-info-settings-label";
            label.textContent = "CLIP last layer";

            clipLastLayerInput = document.createElement("input");
            clipLastLayerInput.type = "number";
            clipLastLayerInput.step = "1";
            clipLastLayerInput.max = "-1";
            clipLastLayerInput.placeholder = "unset";
            clipLastLayerInput.className = "IPT-model-info-settings-input";
            const clipValue = parseNullableInt(runtimeSettings?.[RUNTIME_SETTING_CLIP_LAST_LAYER_KEY]);
            clipLastLayerInput.value = clipValue === null ? "" : String(clipValue);

            grid.append(label, clipLastLayerInput);
        }

        const sd3Label = document.createElement("label");
        sd3Label.className = "IPT-model-info-settings-label";
        sd3Label.textContent = "ModelSamplingSD3 shift";

        const sd3ShiftInput = document.createElement("input");
        sd3ShiftInput.type = "number";
        sd3ShiftInput.step = "0.01";
        sd3ShiftInput.placeholder = "unset";
        sd3ShiftInput.className = "IPT-model-info-settings-input";
        const sd3ShiftValue = parseNullableFloat(runtimeSettings?.[RUNTIME_SETTING_SD3_SHIFT_KEY]);
        sd3ShiftInput.value = sd3ShiftValue === null ? "" : String(sd3ShiftValue);

        grid.append(sd3Label, sd3ShiftInput);
        section.appendChild(grid);

        const status = document.createElement("div");
        status.className = "IPT-model-info-settings-status";
        section.appendChild(status);

        const setStatus = (text, { isError = false } = {}) => {
            status.textContent = text;
            status.classList.toggle("IPT-model-info-settings-status-error", Boolean(isError));
        };

        const setDisabled = (disabled) => {
            saveButton.disabled = disabled;
            resetButton.disabled = disabled;
            sd3ShiftInput.disabled = disabled;
            if (clipLastLayerInput) {
                clipLastLayerInput.disabled = disabled;
            }
        };

        if (!editable || !relativePath) {
            setDisabled(true);
            if (!hasLocalFile) {
                setStatus("Settings can be edited only for local files.");
            } else {
                setStatus("Local model index is still preparing. Press Refresh.");
            }
            this.content.appendChild(section);
            return;
        }

        const collectRuntimeSettings = () => {
            const nextSettings = {};
            if (clipLastLayerInput) {
                const parsedClipLastLayer = parseNullableInt(clipLastLayerInput.value);
                if (parsedClipLastLayer !== null && parsedClipLastLayer <= -1) {
                    nextSettings[RUNTIME_SETTING_CLIP_LAST_LAYER_KEY] = parsedClipLastLayer;
                }
            }
            const parsedSd3Shift = parseNullableFloat(sd3ShiftInput.value);
            if (parsedSd3Shift !== null) {
                nextSettings[RUNTIME_SETTING_SD3_SHIFT_KEY] = parsedSd3Shift;
            }
            return nextSettings;
        };

        const runSave = async (runtimeSettingsPayload) => {
            setDisabled(true);
            setStatus("Saving...");
            try {
                const response = await upsertModelRuntimeSettings({
                    folderName,
                    relativePath,
                    runtimeSettings: runtimeSettingsPayload,
                });
                const nextSettings = response?.runtime_settings && typeof response.runtime_settings === "object"
                    ? response.runtime_settings
                    : {};
                if (clipLastLayerInput) {
                    const nextClipValue = parseNullableInt(nextSettings?.[RUNTIME_SETTING_CLIP_LAST_LAYER_KEY]);
                    clipLastLayerInput.value = nextClipValue === null ? "" : String(nextClipValue);
                }
                const nextShiftValue = parseNullableFloat(nextSettings?.[RUNTIME_SETTING_SD3_SHIFT_KEY]);
                sd3ShiftInput.value = nextShiftValue === null ? "" : String(nextShiftValue);
                setStatus("Saved.");
            } catch (error) {
                const message = error instanceof Error ? error.message : String(error);
                setStatus(`Save failed: ${message}`, { isError: true });
            } finally {
                setDisabled(false);
            }
        };

        saveButton.addEventListener("click", () => {
            void runSave(collectRuntimeSettings());
        });
        resetButton.addEventListener("click", () => {
            void runSave({});
        });

        setDisabled(false);
        setStatus("Settings are stored in the local metadata DB.");
        this.content.appendChild(section);
    }

    updateCopyButton() {
        if (!this.copyButton) {
            return;
        }
        const selectedCount = this.selectedTags.size;
        this.copyButton.disabled = selectedCount === 0;
        this.copyButton.textContent = selectedCount === 0 ? "Copy" : `Copy (${selectedCount})`;
    }

    renderLoraTags(tags, { state = "empty" } = {}) {
        if (!this.content) {
            return;
        }

        const tagItems = Array.isArray(tags) ? tags : [];

        const header = document.createElement("div");
        header.className = "IPT-model-info-lora-header";

        const title = document.createElement("div");
        title.className = "IPT-model-info-lora-title";
        title.textContent = "LoRA tags";

        const copyButton = document.createElement("button");
        copyButton.type = "button";
        copyButton.className = "IPT-model-info-copy-button";
        copyButton.textContent = "Copy";
        copyButton.disabled = true;
        copyButton.addEventListener("click", async () => {
            const selected = this.tagOrder.filter((tag) => this.selectedTags.has(tag));
            const copied = await copyToClipboard(selected.join(", "));
            const previous = copyButton.textContent;
            copyButton.textContent = copied ? "Copied" : "Copy failed";
            setTimeout(() => {
                if (this.copyButton === copyButton) {
                    this.updateCopyButton();
                } else {
                    copyButton.textContent = previous;
                }
            }, 900);
        });

        header.append(title, copyButton);
        this.content.appendChild(header);
        this.copyButton = copyButton;

        if (!tagItems.length) {
            const empty = document.createElement("div");
            empty.className = "IPT-model-info-status";
            empty.textContent = getEmptyLoraTagsMessage(state);
            this.content.appendChild(empty);
            this.updateCopyButton();
            return;
        }

        const wrap = document.createElement("div");
        wrap.className = "IPT-model-info-tags";
        for (const tagInfo of tagItems) {
            const tag = String(tagInfo?.tag ?? "").trim();
            if (!tag) {
                continue;
            }
            const frequency = Number(tagInfo?.frequency ?? 0);
            this.tagOrder.push(tag);

            const button = document.createElement("button");
            button.type = "button";
            button.className = "IPT-model-info-tag";

            const name = document.createElement("span");
            name.textContent = tag;

            const count = document.createElement("span");
            count.className = "IPT-model-info-tag-frequency";
            count.textContent = Number.isFinite(frequency) ? String(Math.max(0, Math.trunc(frequency))) : "0";

            button.append(name, count);
            button.addEventListener("click", () => {
                if (this.selectedTags.has(tag)) {
                    this.selectedTags.delete(tag);
                    button.classList.remove("IPT-model-info-tag-selected");
                } else {
                    this.selectedTags.add(tag);
                    button.classList.add("IPT-model-info-tag-selected");
                }
                this.updateCopyButton();
            });
            wrap.appendChild(button);
        }

        this.content.appendChild(wrap);
        this.updateCopyButton();
    }

    async load() {
        if (!this.content) {
            return;
        }

        this.clearContent();
        this.setStatus("Loading model metadata...");

        const folderName = this.slot.folderName;
        const relativePath = this.source.relativePath;
        const includeLoraTags = folderName === "loras";
        const ensureLoraTags = includeLoraTags;
        const ensureTimeoutMs = includeLoraTags ? LORA_TAG_ENSURE_TIMEOUT_MS : null;

        try {
            let payload = null;
            for (let attempt = 1; attempt <= INFO_POLL_MAX_ATTEMPTS; attempt += 1) {
                payload = await fetchModelReference({
                    folderName,
                    relativePath,
                    sha256: this.source.sha256,
                    nameRaw: this.source.nameRaw,
                    hashHints: this.source.hashHints,
                    includeLoraTags,
                    ensureLoraTags,
                    ensureTimeoutMs,
                });
                if (payload?.sha256) {
                    this.source.sha256 = normalizeSha256(payload.sha256);
                    setSha256WidgetValue(this.node, this.slot, this.source.sha256);
                }
                const loraTagsState = normalizeLoraTagsState(payload?.lora_tags_state);
                if (includeLoraTags) {
                    if (
                        payload?.local_status !== "present"
                        || isTerminalLoraTagsState(loraTagsState)
                    ) {
                        break;
                    }
                } else if (
                    payload?.model_info
                    || payload?.local_status !== "present"
                    || payload?.remote_status === "not_found"
                ) {
                    break;
                }
                if (!this.element || !this.content) {
                    return;
                }
                if (attempt < INFO_POLL_MAX_ATTEMPTS) {
                    if (includeLoraTags && loraTagsState === "pending") {
                        this.setStatus(`Analyzing LoRA tags... (${attempt + 1}/${INFO_POLL_MAX_ATTEMPTS})`);
                    } else {
                        this.setStatus(`Loading model metadata... (${attempt + 1}/${INFO_POLL_MAX_ATTEMPTS})`);
                    }
                    await waitMs(INFO_POLL_INTERVAL_MS);
                }
            }

            if (!this.element || !this.content) {
                return;
            }

            if (this.titleElement) {
                this.titleElement.textContent = this.buildTitleText();
                this.scrollTitleToEnd(this.titleElement);
            }

            this.clearContent();
            const resolvedRelativePath = String(payload?.local_match?.relative_path ?? relativePath ?? "").trim();
            const hasLocalFile = String(payload?.local_status ?? "") === "present";
            if (!payload?.model_info && hasLocalFile && payload?.remote_status !== "not_found") {
                this.setStatus("No metadata found yet. Please wait a moment and press Refresh.");
            }
            if (!hasLocalFile) {
                this.renderMissingLocalNotice(payload?.download_candidate || null);
            }

            if (payload?.model_info || payload?.remote_status === "not_found" || !hasLocalFile) {
                this.renderModelTopSection(folderName, resolvedRelativePath, {
                    ...(payload?.model_info || {}),
                    copyable_hashes: payload?.copyable_hashes || {},
                }, {
                    showPreview: hasLocalFile && Boolean(resolvedRelativePath),
                });
            }

            this.renderRuntimeSettingsSection({
                folderName,
                relativePath: resolvedRelativePath,
                runtimeSettings: payload?.runtime_settings || {},
                editable: Boolean(payload?.runtime_settings_editable),
                hasLocalFile,
            });

            if (folderName === "loras" && Array.isArray(payload?.lora_tags)) {
                this.renderLoraTags(payload.lora_tags || [], {
                    state: normalizeLoraTagsState(payload?.lora_tags_state),
                });
                return;
            }

            if (!payload?.model_info && !hasLocalFile) {
                const hasHashHints = Array.isArray(this.source.hashHints) && this.source.hashHints.length > 0;
                const message = payload?.remote_status === "not_found"
                    ? "No remote metadata was found for the stored SHA256."
                    : hasHashHints && !this.source.sha256
                        ? "Remote metadata could not be resolved from the available hash hints."
                        : "Remote metadata is not available yet.";
                this.setStatus(message);
            }
        } catch (error) {
            if (!this.element || !this.content) {
                return;
            }
            this.clearContent();
            const message = error instanceof Error ? error.message : String(error);
            this.setStatus(`Failed to load model metadata: ${message}`);
        }
    }
}

async function startDownloadCandidate(candidate) {
    const url = String(candidate?.url ?? "").trim();
    if (!url) {
        throw new Error("download_url_missing");
    }

    window.open(url, "_blank", "noopener,noreferrer");
}

function resolveReferenceSource(node, slot) {
    return {
        relativePath: resolveSelectedPath(node, slot),
        sha256: resolveSelectedSha256(node, slot),
        nameRaw: resolveSelectedPath(node, slot),
        hashHints: [],
    };
}

function resolveTargetByFolderName(folderName) {
    const normalized = String(folderName ?? "").trim();
    if (!normalized) {
        return null;
    }
    return { folderName: normalized };
}

function normalizeTargetToSlot(target, slot) {
    if (slot?.folderName) {
        return slot;
    }
    if (target?.folderName) {
        return target;
    }
    const selectors = Array.isArray(target?.selectors) ? target.selectors : [];
    return selectors[0] ?? null;
}

function openModelInfoWindow({ node = null, target = null, slot = null, source }) {
    const resolvedSlot = normalizeTargetToSlot(target, slot);
    if (!resolvedSlot) {
        return false;
    }
    new ModelInfoWindow(node, resolvedSlot, source || {}).show();
    return true;
}

function addMenuOption(node, options) {
    const target = resolveSelectorTarget(node);
    if (!target) {
        return;
    }

    const preferredWidgetNameAtMenuOpen = app.canvas?.getWidgetAtCursor?.()?.name ?? "";

    options.unshift({
        content: MENU_LABEL,
        callback: () => {
            const slot = resolvePreferredSelectorSlot(node, target, {
                preferredWidgetName: preferredWidgetNameAtMenuOpen,
                allowEmpty: false,
            });
            if (!slot) {
                alert("Select a model first.");
                return;
            }

            const source = resolveReferenceSource(node, slot);
            if (!source.relativePath && !source.sha256) {
                alert("Select a model first.");
                return;
            }
            new ModelInfoWindow(node, slot, source).show();
        },
    });
}

app.registerExtension({
    name: "IPT.SelectorModelInfoWindow",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isSelectorTargetNodeDef(nodeData)) {
            return;
        }
        if (nodeType.prototype.__iptSelectorModelInfoMenuPatched) {
            return;
        }

        const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function patchedGetExtraMenuOptions(_, options) {
            if (Array.isArray(options)) {
                addMenuOption(this, options);
            }
            return originalGetExtraMenuOptions?.apply(this, arguments);
        };

        nodeType.prototype.__iptSelectorModelInfoMenuPatched = true;
    },
});

if (typeof window !== "undefined") {
    window.__iisModelInfoWindow = {
        openModelInfoWindow,
        resolveTargetByFolderName,
    };
}
