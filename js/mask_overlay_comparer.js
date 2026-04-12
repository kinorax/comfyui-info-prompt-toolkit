import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE_TYPES = new Set([
    "IPT-MaskOverlayComparer",
    "MaskOverlayComparer",
    "Mask Overlay Comparer",
]);

const STYLE_ID = "IPT-mask-overlay-comparer-style";
const WIDGET_NAME = "mask_overlay_comparer";
const STATE_KEY = "__iptMaskOverlayComparerState";
const PATCHED_FLAG = "__iptMaskOverlayComparerPatched";
const PATCH_RETRY_FLAG = "__iptMaskOverlayComparerPatchRetryQueued";
const PATCH_RETRY_COUNT_KEY = "__iptMaskOverlayComparerPatchRetryCount";
const MIN_HEIGHT = 238;
const MIN_WIDTH = 160;
const NODE_MIN_WIDTH = 260;
const NODE_MIN_HEIGHT = 300;
const DEFAULT_SPLIT_RATIO = 0.5;
const MAX_PATCH_RETRIES = 3;

function getNodeTypeCandidates(node) {
    return [
        node?.comfyClass,
        node?.type,
        node?.constructor?.comfyClass,
        node?.constructor?.type,
        node?.title,
    ].filter(Boolean);
}

function isTargetNode(node) {
    return getNodeTypeCandidates(node).some((candidate) => TARGET_NODE_TYPES.has(candidate));
}

function isTargetNodeDef(nodeData) {
    const candidates = [
        nodeData?.name,
        nodeData?.display_name,
        nodeData?.type,
        nodeData?.node_id,
    ].filter(Boolean);
    return candidates.some((candidate) => TARGET_NODE_TYPES.has(candidate));
}

function chainCallback(original, callback) {
    return function chained(...args) {
        const result = original?.apply(this, args);
        callback?.apply(this, args);
        return result;
    };
}

function requestVueNodeRefresh(node) {
    const graph = node?.graph;
    if (!graph?.trigger || node?.id == null) {
        return;
    }

    graph.trigger("node:property:changed", {
        nodeId: node.id,
        property: "shape",
        newValue: node.shape,
    });
}

function installStyle() {
    if (document.getElementById(STYLE_ID)) {
        return;
    }

    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .IPT-mask-overlay-comparer {
            display: flex;
            flex-direction: column;
            gap: 8px;
            width: 100%;
            height: 100%;
            min-width: 0;
            padding: 0 8px 6px 8px;
            box-sizing: border-box;
        }
        .IPT-mask-overlay-comparer-frame {
            position: relative;
            flex: 1 1 auto;
            min-width: 0;
            min-height: 160px;
            border-radius: 8px;
            border: 1px solid #404656;
            overflow: hidden;
            background:
                linear-gradient(45deg, #232831 25%, transparent 25%),
                linear-gradient(-45deg, #232831 25%, transparent 25%),
                linear-gradient(45deg, transparent 75%, #232831 75%),
                linear-gradient(-45deg, transparent 75%, #232831 75%);
            background-size: 18px 18px;
            background-position: 0 0, 0 9px, 9px -9px, -9px 0;
            touch-action: none;
            user-select: none;
            cursor: ew-resize;
        }
        .IPT-mask-overlay-comparer-image {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
            pointer-events: none;
            background: #0f1217;
        }
        .IPT-mask-overlay-comparer-overlay-clip {
            display: contents;
        }
        .IPT-mask-overlay-comparer-divider {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 2px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 0 0 1px rgba(22, 24, 29, 0.35);
            transform: translateX(-1px);
            pointer-events: none;
        }
        .IPT-mask-overlay-comparer-handle {
            position: absolute;
            top: 50%;
            width: 28px;
            height: 28px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.96);
            color: #151821;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
            transform: translate(-50%, -50%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: 700;
            pointer-events: none;
        }
        .IPT-mask-overlay-comparer-status {
            font-size: 11px;
            line-height: 1.3;
            color: #9aa4b7;
            min-width: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .IPT-mask-overlay-comparer-labels {
            display: flex;
            justify-content: space-between;
            gap: 8px;
            font-size: 10px;
            color: #788399;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
    `;
    (document.head ?? document.body).appendChild(style);
}

function buildViewUrl(imageMeta) {
    if (!imageMeta || typeof imageMeta !== "object") {
        return "";
    }
    const filename = String(imageMeta.filename ?? "").trim();
    const imageType = String(imageMeta.type ?? "").trim();
    if (!(filename && imageType)) {
        return "";
    }
    const params = new URLSearchParams({
        filename,
        type: imageType,
    });
    const subfolder = String(imageMeta.subfolder ?? "").trim();
    if (subfolder) {
        params.set("subfolder", subfolder);
    }
    const path = `/view?${params.toString()}`;
    return typeof api?.apiURL === "function" ? api.apiURL(path) : path;
}

function clampRatio(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
        return DEFAULT_SPLIT_RATIO;
    }
    return Math.min(1, Math.max(0, numeric));
}

function resolveCanvasElement() {
    const directCanvas = app?.canvas?.canvas;
    if (directCanvas instanceof HTMLCanvasElement) {
        return directCanvas;
    }

    const fallbackCanvas = document.querySelector("canvas");
    return fallbackCanvas instanceof HTMLCanvasElement ? fallbackCanvas : null;
}

function forwardWheelToCanvas(event) {
    const canvas = resolveCanvasElement();
    if (!canvas) {
        return;
    }

    event.preventDefault();

    const forwardedEvent = new WheelEvent("wheel", {
        deltaX: event.deltaX,
        deltaY: event.deltaY,
        deltaZ: event.deltaZ,
        deltaMode: event.deltaMode,
        clientX: event.clientX,
        clientY: event.clientY,
        screenX: event.screenX,
        screenY: event.screenY,
        ctrlKey: event.ctrlKey,
        shiftKey: event.shiftKey,
        altKey: event.altKey,
        metaKey: event.metaKey,
        bubbles: true,
        cancelable: true,
    });
    canvas.dispatchEvent(forwardedEvent);
}

function createComparerState(node) {
    if (typeof node?.addDOMWidget !== "function") {
        return null;
    }

    installStyle();

    const host = document.createElement("div");
    host.className = "IPT-mask-overlay-comparer";
    host.style.minHeight = `${MIN_HEIGHT}px`;
    host.style.setProperty("--comfy-widget-min-height", `${MIN_HEIGHT}px`);
    host.style.setProperty("--comfy-widget-height", `${MIN_HEIGHT}px`);
    host.style.setProperty("--comfy-widget-max-height", `${MIN_HEIGHT}px`);

    const frame = document.createElement("div");
    frame.className = "IPT-mask-overlay-comparer-frame";

    const baseImage = document.createElement("img");
    baseImage.className = "IPT-mask-overlay-comparer-image";
    baseImage.alt = "Mask overlay comparer base";
    baseImage.decoding = "async";
    baseImage.style.display = "none";

    const overlayClip = document.createElement("div");
    overlayClip.className = "IPT-mask-overlay-comparer-overlay-clip";

    const overlayImage = document.createElement("img");
    overlayImage.className = "IPT-mask-overlay-comparer-image";
    overlayImage.alt = "Mask overlay comparer overlay";
    overlayImage.decoding = "async";
    overlayImage.style.display = "none";
    overlayClip.appendChild(overlayImage);

    const divider = document.createElement("div");
    divider.className = "IPT-mask-overlay-comparer-divider";

    const handle = document.createElement("div");
    handle.className = "IPT-mask-overlay-comparer-handle";
    handle.textContent = "<>";

    frame.append(baseImage, overlayClip, divider, handle);

    const labels = document.createElement("div");
    labels.className = "IPT-mask-overlay-comparer-labels";

    const leftLabel = document.createElement("span");
    leftLabel.textContent = "Overlay On";
    const rightLabel = document.createElement("span");
    rightLabel.textContent = "Overlay Off";
    labels.append(leftLabel, rightLabel);

    const status = document.createElement("div");
    status.className = "IPT-mask-overlay-comparer-status";
    status.textContent = "Preview: run the node to generate comparer images.";

    host.append(frame, labels, status);

    const widget = node.addDOMWidget(WIDGET_NAME, "mask_overlay_comparer", host, {
        canvasOnly: false,
        hideOnZoom: false,
        getMinHeight: () => MIN_HEIGHT,
        getHeight: () => MIN_HEIGHT,
        getMaxHeight: () => MIN_HEIGHT,
    });
    widget.serialize = false;
    widget.options = widget.options ?? {};
    widget.options.canvasOnly = false;
    widget.computeLayoutSize = () => ({
        minHeight: MIN_HEIGHT,
        minWidth: MIN_WIDTH,
    });

    const state = {
        widget,
        host,
        frame,
        baseImage,
        overlayClip,
        overlayImage,
        divider,
        handle,
        status,
        splitRatio: DEFAULT_SPLIT_RATIO,
        baseUrl: "",
        overlayUrl: "",
    };

    baseImage.addEventListener("load", () => {
        baseImage.style.display = "block";
    });
    overlayImage.addEventListener("load", () => {
        overlayImage.style.display = "block";
    });
    baseImage.addEventListener("error", () => {
        baseImage.style.display = "none";
    });
    overlayImage.addEventListener("error", () => {
        overlayImage.style.display = "none";
    });

    const updateFromPointer = (event) => {
        const rect = frame.getBoundingClientRect();
        if (!(rect.width > 0)) {
            return;
        }
        const nextRatio = clampRatio((event.clientX - rect.left) / rect.width);
        setSplitRatio(state, nextRatio);
    };

    frame.addEventListener("pointerdown", (event) => {
        updateFromPointer(event);
    });

    frame.addEventListener("pointermove", (event) => {
        updateFromPointer(event);
    });

    frame.addEventListener("wheel", forwardWheelToCanvas, { passive: false });

    setSplitRatio(state, DEFAULT_SPLIT_RATIO);
    return state;
}

function getComparerState(node) {
    const existing = node?.[STATE_KEY];
    if (existing) {
        return existing;
    }
    const created = createComparerState(node);
    if (created) {
        node[STATE_KEY] = created;
    }
    return created;
}

function setStatus(state, text, isError = false) {
    if (!state?.status) {
        return;
    }
    state.status.textContent = text;
    state.status.style.color = isError ? "#d78f8f" : "#9aa4b7";
}

function setSplitRatio(state, ratio) {
    if (!state?.frame) {
        return;
    }
    const nextRatio = clampRatio(ratio);
    state.splitRatio = nextRatio;
    const percent = `${(nextRatio * 100).toFixed(3)}%`;
    state.overlayImage.style.clipPath = `inset(0 ${`${((1 - nextRatio) * 100).toFixed(3)}%`} 0 0)`;
    state.divider.style.left = percent;
    state.handle.style.left = percent;
}

function parseNodeSizeValue(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : 0;
}

function getComputedNodeSize(node) {
    if (typeof node?.computeSize !== "function") {
        return null;
    }

    try {
        const computed = node.computeSize(Array.isArray(node.size) ? [...node.size] : undefined);
        if (!Array.isArray(computed) || computed.length < 2) {
            return null;
        }

        const width = parseNodeSizeValue(computed[0]);
        const height = parseNodeSizeValue(computed[1]);
        if (!(width > 0) || !(height > 0)) {
            return null;
        }

        return [width, height];
    } catch {
        return null;
    }
}

function schedulePatchRetry(node) {
    if (!node || node[PATCH_RETRY_FLAG]) {
        return;
    }

    const retryCount = Number(node[PATCH_RETRY_COUNT_KEY]) || 0;
    if (retryCount >= MAX_PATCH_RETRIES) {
        return;
    }
    if (typeof window === "undefined" || typeof window.requestAnimationFrame !== "function") {
        return;
    }

    node[PATCH_RETRY_COUNT_KEY] = retryCount + 1;
    node[PATCH_RETRY_FLAG] = true;
    window.requestAnimationFrame(() => {
        node[PATCH_RETRY_FLAG] = false;
        patchNode(node);
    });
}

function extractComparerPayload(output) {
    const normalizePayload = (value) => {
        if (Array.isArray(value)) {
            return value[0] && typeof value[0] === "object" ? value[0] : null;
        }
        return value && typeof value === "object" ? value : null;
    };

    if (Array.isArray(output)) {
        for (const item of output) {
            const payload = normalizePayload(item?.mask_overlay_comparer ?? item);
            if (payload?.base_image && payload?.overlay_image) {
                return payload;
            }
        }
        return null;
    }

    if (!output || typeof output !== "object") {
        return null;
    }

    const directValue = normalizePayload(output?.mask_overlay_comparer);
    if (directValue?.base_image && directValue?.overlay_image) {
        return directValue;
    }

    const uiValue = normalizePayload(output?.ui?.mask_overlay_comparer);
    if (uiValue?.base_image && uiValue?.overlay_image) {
        return uiValue;
    }

    if (Array.isArray(output.result)) {
        for (const item of output.result) {
            const payload = normalizePayload(item?.mask_overlay_comparer ?? item);
            if (payload?.base_image && payload?.overlay_image) {
                return payload;
            }
        }
    }

    return null;
}

function applyComparerPayload(node, payload) {
    const state = getComparerState(node);
    if (!state) {
        return;
    }

    if (!payload) {
        state.baseImage.removeAttribute("src");
        state.overlayImage.removeAttribute("src");
        state.baseImage.style.display = "none";
        state.overlayImage.style.display = "none";
        state.baseUrl = "";
        state.overlayUrl = "";
        setStatus(state, "Preview: run the node to generate comparer images.");
        return;
    }

    const baseUrl = buildViewUrl(payload.base_image);
    const overlayUrl = buildViewUrl(payload.overlay_image);
    if (!(baseUrl && overlayUrl)) {
        setStatus(state, "Preview: comparer images are missing.", true);
        return;
    }

    if (state.baseUrl !== baseUrl) {
        state.baseImage.style.display = "none";
        state.baseImage.src = baseUrl;
        state.baseUrl = baseUrl;
    }
    if (state.overlayUrl !== overlayUrl) {
        state.overlayImage.style.display = "none";
        state.overlayImage.src = overlayUrl;
        state.overlayUrl = overlayUrl;
    }

    setSplitRatio(state, payload.split_ratio);
    const statusText = String(payload.status ?? "").trim() || "Preview: ready";
    setStatus(state, statusText);
    node.setDirtyCanvas?.(true, true);
}

function patchNode(node) {
    if (!isTargetNode(node)) {
        return;
    }

    const hadState = Boolean(node?.[STATE_KEY]);
    const state = getComparerState(node);
    let layoutChanged = Boolean(state) && !hadState;

    if (!node[PATCHED_FLAG]) {
        if (!(Array.isArray(node.size) && node.size.length >= 2)) {
            schedulePatchRetry(node);
        } else {
            const currentWidth = parseNodeSizeValue(node.size[0]);
            const currentHeight = parseNodeSizeValue(node.size[1]);
            const computedSize = getComputedNodeSize(node);
            const computedWidth = computedSize?.[0] ?? 0;
            const computedHeight = computedSize?.[1] ?? 0;

            const nextWidth = Math.max(currentWidth, computedWidth, NODE_MIN_WIDTH);
            const nextHeight = Math.max(currentHeight, computedHeight, NODE_MIN_HEIGHT);
            layoutChanged = layoutChanged || node.size[0] !== nextWidth || node.size[1] !== nextHeight;
            if (layoutChanged && typeof node.setSize === "function") {
                node.setSize([nextWidth, nextHeight]);
            } else {
                node.size[0] = nextWidth;
                node.size[1] = nextHeight;
            }
            node[PATCHED_FLAG] = true;
            node[PATCH_RETRY_COUNT_KEY] = 0;
        }
    }

    if (layoutChanged) {
        node.setDirtyCanvas?.(true, true);
        requestVueNodeRefresh(node);
    }
}

app.registerExtension({
    name: "IPT.MaskOverlayComparer",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isTargetNodeDef(nodeData)) {
            return;
        }

        nodeType.prototype.onNodeCreated = chainCallback(
            nodeType.prototype.onNodeCreated,
            function onNodeCreated() {
                patchNode(this);
            },
        );

        nodeType.prototype.onConfigure = chainCallback(
            nodeType.prototype.onConfigure,
            function onConfigure() {
                patchNode(this);
            },
        );

        nodeType.prototype.onGraphConfigured = chainCallback(
            nodeType.prototype.onGraphConfigured,
            function onGraphConfigured() {
                patchNode(this);
            },
        );

        nodeType.prototype.onExecuted = chainCallback(
            nodeType.prototype.onExecuted,
            function onExecuted(output) {
                patchNode(this);
                applyComparerPayload(this, extractComparerPayload(output));
            },
        );
    },
    nodeCreated(node) {
        patchNode(node);
    },
    loadedGraphNode(node) {
        patchNode(node);
    },
    onNodeOutputsUpdated(nodeOutputs) {
        if (!app?.rootGraph || !nodeOutputs || typeof nodeOutputs !== "object") {
            return;
        }

        for (const node of app.rootGraph._nodes || []) {
            if (!isTargetNode(node)) {
                continue;
            }
            patchNode(node);
            const output = nodeOutputs[String(node.id)];
            applyComparerPayload(node, extractComparerPayload(output));
        }
    },
});
