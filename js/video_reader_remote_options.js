import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE_TYPES = new Set([
    "IPT-VideoReader",
    "VideoReader",
    "Video Reader",
]);

const PATH_SOURCE_WIDGET_NAME = "path_source";
const DIRECTORY_WIDGET_NAME = "directory";
const FILE_WIDGET_NAME = "file";
const FRAME_LOAD_CAP_WIDGET_NAME = "frame_load_cap";
const SKIP_FIRST_FRAMES_WIDGET_NAME = "skip_first_frames";
const SELECT_EVERY_NTH_WIDGET_NAME = "select_every_nth";

const PATCHED_FLAG = "__iisVideoReaderPathPatched";
const DIRECTORY_FETCH_TOKEN_KEY = "__iisVideoReaderDirectoryFetchToken";
const FILE_FETCH_TOKEN_KEY = "__iisVideoReaderFileFetchToken";
const PREVIEW_FETCH_TOKEN_KEY = "__iisVideoReaderPreviewFetchToken";
const PREVIEW_STATE_KEY = "__iisVideoReaderPreviewState";

const PREVIEW_WIDGET_NAME = "video_preview";
const PREVIEW_WIDGET_MIN_HEIGHT = 172;
const SOURCE_DISPLAY_SIZE_CACHE = new Map();
const SOURCE_DISPLAY_PROBE_TIMEOUT_MS = 2500;

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

function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) ?? null;
}

function normalizePathSource(value) {
    const source = String(value ?? "").trim().toLowerCase();
    return source === "output" ? "output" : "input";
}

function normalizeDirectory(value) {
    const directory = String(value ?? "").trim().replace(/\\+/g, "/");
    return directory || ".";
}

function normalizeFileName(value) {
    return String(value ?? "").trim();
}

function buildSourceViewPath(pathSource, directory, file) {
    const normalizedDirectory = normalizeDirectory(directory);
    const params = normalizedDirectory === "."
        ? new URLSearchParams({ filename: file, type: pathSource })
        : new URLSearchParams({ filename: file, subfolder: normalizedDirectory, type: pathSource });
    return `/view?${params.toString()}`;
}

function makeSourceDisplaySizeKey(queryParams) {
    return `${queryParams.path_source}::${queryParams.directory}::${queryParams.file}`;
}

async function probeSourceDisplaySize(queryParams) {
    if (!queryParams?.file) {
        return null;
    }
    const cacheKey = makeSourceDisplaySizeKey(queryParams);
    const cached = SOURCE_DISPLAY_SIZE_CACHE.get(cacheKey);
    if (cached) {
        return cached;
    }

    const sourcePath = buildSourceViewPath(queryParams.path_source, queryParams.directory, queryParams.file);
    const src = typeof api?.apiURL === "function" ? api.apiURL(sourcePath) : sourcePath;

    const probed = await new Promise((resolve) => {
        const video = document.createElement("video");
        let finished = false;

        const complete = (value) => {
            if (finished) {
                return;
            }
            finished = true;
            video.removeAttribute("src");
            video.load?.();
            resolve(value);
        };

        const timer = setTimeout(() => {
            complete(null);
        }, SOURCE_DISPLAY_PROBE_TIMEOUT_MS);

        video.preload = "metadata";
        video.muted = true;
        video.playsInline = true;
        video.onloadedmetadata = () => {
            clearTimeout(timer);
            const width = Number(video.videoWidth) || 0;
            const height = Number(video.videoHeight) || 0;
            if (width > 0 && height > 0) {
                complete({ width, height });
                return;
            }
            complete(null);
        };
        video.onerror = () => {
            clearTimeout(timer);
            complete(null);
        };
        video.src = src;
    });

    if (probed && probed.width > 0 && probed.height > 0) {
        SOURCE_DISPLAY_SIZE_CACHE.set(cacheKey, probed);
        return probed;
    }
    return null;
}

function normalizeInteger(value, fallback, minimum) {
    const n = Number(value);
    const normalized = Number.isFinite(n) ? Math.trunc(n) : fallback;
    return Math.max(minimum, normalized);
}

function normalizeDirectoryOptions(rawValues) {
    const values = [];
    const seen = new Set();

    const push = (value) => {
        const text = normalizeDirectory(value);
        if (!text || seen.has(text)) {
            return;
        }
        seen.add(text);
        values.push(text);
    };

    if (Array.isArray(rawValues)) {
        for (const value of rawValues) {
            push(value);
        }
    }

    push(".");

    values.sort((a, b) => {
        if (a === "." && b !== ".") {
            return -1;
        }
        if (b === "." && a !== ".") {
            return 1;
        }
        return a.localeCompare(b, undefined, { sensitivity: "base" });
    });

    return values;
}

function normalizeFileOptions(rawValues) {
    const values = [];
    const seen = new Set();

    const push = (value) => {
        const text = normalizeFileName(value);
        if (!text || seen.has(text)) {
            return;
        }
        seen.add(text);
        values.push(text);
    };

    if (Array.isArray(rawValues)) {
        for (const value of rawValues) {
            push(value);
        }
    }

    values.sort((a, b) => b.localeCompare(a, undefined, { sensitivity: "base" }));
    return values;
}

async function fetchDirectoryOptions(pathSource) {
    if (!api || typeof api.fetchApi !== "function") {
        return ["."];
    }

    const query = new URLSearchParams({ path_source: pathSource }).toString();
    const response = await api.fetchApi(`/iis/video-reader/directories?${query}`, {
        method: "GET",
        cache: "no-store",
    });

    if (!response?.ok) {
        throw new Error(`failed to fetch directories (${response?.status ?? "unknown"})`);
    }

    const payload = await response.json();
    return normalizeDirectoryOptions(payload);
}

async function fetchFileOptions(pathSource, directory) {
    if (!api || typeof api.fetchApi !== "function") {
        return [];
    }

    const query = new URLSearchParams({
        path_source: pathSource,
        directory,
    }).toString();
    const response = await api.fetchApi(`/iis/video-reader/files?${query}`, {
        method: "GET",
        cache: "no-store",
    });

    if (!response?.ok) {
        throw new Error(`failed to fetch files (${response?.status ?? "unknown"})`);
    }

    const payload = await response.json();
    return normalizeFileOptions(payload);
}

function createPreviewState(node) {
    if (typeof node?.addDOMWidget !== "function") {
        return null;
    }

    const host = document.createElement("div");
    host.style.display = "flex";
    host.style.flexDirection = "column";
    host.style.gap = "4px";
    host.style.width = "100%";
    host.style.height = "100%";
    host.style.padding = "0 8px 2px 8px";
    host.style.boxSizing = "border-box";

    const video = document.createElement("video");
    video.controls = true;
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    video.preload = "metadata";
    video.style.width = "100%";
    video.style.height = "0";
    video.style.flex = "1 1 auto";
    video.style.minHeight = "72px";
    video.style.objectFit = "contain";
    video.style.background = "#111";
    video.style.border = "1px solid #444";
    video.style.borderRadius = "6px";

    const status = document.createElement("div");
    status.style.fontSize = "11px";
    status.style.lineHeight = "1.2";
    status.style.color = "#999";
    status.style.userSelect = "none";
    status.style.whiteSpace = "nowrap";
    status.style.overflow = "hidden";
    status.style.textOverflow = "ellipsis";
    status.textContent = "Preview: waiting for file selection";

    host.append(video, status);

    const widget = node.addDOMWidget(PREVIEW_WIDGET_NAME, "video", host, {
        canvasOnly: true,
        hideOnZoom: false,
    });
    widget.serialize = false;
    widget.computeLayoutSize = () => ({
        minHeight: PREVIEW_WIDGET_MIN_HEIGHT,
        minWidth: 220,
    });

    const state = {
        widget,
        host,
        video,
        status,
        currentUrl: "",
    };

    video.addEventListener("error", () => {
        state.status.style.color = "#d66";
        state.status.textContent = "Preview: failed to load";
    });

    return state;
}

function getPreviewState(node) {
    const current = node?.[PREVIEW_STATE_KEY];
    if (current) {
        return current;
    }
    const created = createPreviewState(node);
    if (created) {
        node[PREVIEW_STATE_KEY] = created;
    }
    return created;
}

function setPreviewStatus(previewState, message, isError = false) {
    if (!previewState?.status) {
        return;
    }
    previewState.status.style.color = isError ? "#d66" : "#999";
    previewState.status.textContent = message;
}

function clearPreviewVideo(previewState) {
    if (!previewState?.video) {
        return;
    }
    previewState.video.pause?.();
    previewState.video.removeAttribute("src");
    previewState.video.load?.();
    previewState.currentUrl = "";
}

function buildPreviewQuery(node) {
    const pathSourceWidget = getWidget(node, PATH_SOURCE_WIDGET_NAME);
    const directoryWidget = getWidget(node, DIRECTORY_WIDGET_NAME);
    const fileWidget = getWidget(node, FILE_WIDGET_NAME);
    const frameCapWidget = getWidget(node, FRAME_LOAD_CAP_WIDGET_NAME);
    const skipWidget = getWidget(node, SKIP_FIRST_FRAMES_WIDGET_NAME);
    const nthWidget = getWidget(node, SELECT_EVERY_NTH_WIDGET_NAME);

    if (!pathSourceWidget || !directoryWidget || !fileWidget || !frameCapWidget || !skipWidget || !nthWidget) {
        return null;
    }

    return {
        path_source: normalizePathSource(pathSourceWidget.value),
        directory: normalizeDirectory(directoryWidget.value),
        file: normalizeFileName(fileWidget.value),
        frame_load_cap: String(normalizeInteger(frameCapWidget.value, 0, 0)),
        skip_first_frames: String(normalizeInteger(skipWidget.value, 0, 0)),
        select_every_nth: String(normalizeInteger(nthWidget.value, 1, 1)),
    };
}

async function fetchPreviewPayload(queryParams) {
    const sourceDisplaySize = await probeSourceDisplaySize(queryParams);
    const requestParams = { ...queryParams };
    if (sourceDisplaySize) {
        requestParams.source_display_width_hint = String(sourceDisplaySize.width);
        requestParams.source_display_height_hint = String(sourceDisplaySize.height);
    }

    const query = new URLSearchParams(requestParams).toString();
    const response = await api.fetchApi(`/iis/video-reader/preview?${query}`, {
        method: "GET",
        cache: "no-store",
    });

    let payload = null;
    try {
        payload = await response.json();
    } catch (error) {
        payload = null;
    }

    if (!response?.ok || !payload || payload.ok !== true || !payload.url) {
        const message = payload?.error ? String(payload.error) : `HTTP ${response?.status ?? "unknown"}`;
        throw new Error(message);
    }

    return payload;
}

async function refreshPreview(node) {
    if (!isTargetNode(node)) {
        return;
    }

    const previewState = getPreviewState(node);
    if (!previewState) {
        return;
    }

    const queryParams = buildPreviewQuery(node);
    if (!queryParams) {
        setPreviewStatus(previewState, "Preview: waiting for parameters");
        clearPreviewVideo(previewState);
        return;
    }
    if (!queryParams.file) {
        setPreviewStatus(previewState, "Preview: waiting for file selection");
        clearPreviewVideo(previewState);
        return;
    }

    const requestToken = (Number(node[PREVIEW_FETCH_TOKEN_KEY]) || 0) + 1;
    node[PREVIEW_FETCH_TOKEN_KEY] = requestToken;

    setPreviewStatus(previewState, "Preview: loading...");

    try {
        const payload = await fetchPreviewPayload(queryParams);
        if (node[PREVIEW_FETCH_TOKEN_KEY] !== requestToken) {
            return;
        }

        const mode = String(payload.mode ?? "reflect");
        const cacheToken = payload.cache_key
            ? String(payload.cache_key)
            : `passthrough:${queryParams.path_source}:${queryParams.directory}:${queryParams.file}`;
        const mediaUrl = `${payload.url}${payload.url.includes("?") ? "&" : "?"}cache_key=${encodeURIComponent(cacheToken)}`;
        const resolvedUrl = typeof api?.apiURL === "function" ? api.apiURL(mediaUrl) : mediaUrl;

        if (previewState.currentUrl !== resolvedUrl) {
            previewState.video.src = resolvedUrl;
            previewState.video.load?.();
            previewState.currentUrl = resolvedUrl;
        }

        previewState.video.play?.().catch(() => {
            // ignore autoplay restrictions; controls are enabled.
        });

        const selectedFrames = payload.selected_frames;
        const previewFps = payload.preview_fps;
        if (mode === "passthrough") {
            const thresholdFrames = payload.threshold_frames;
            const reason = payload.reason ? String(payload.reason) : "threshold";
            if (thresholdFrames != null) {
                setPreviewStatus(previewState, `Preview: passthrough (${reason}, threshold=${thresholdFrames} frames)`);
            } else {
                setPreviewStatus(previewState, `Preview: passthrough (${reason})`);
            }
        } else if (selectedFrames != null && previewFps != null) {
            setPreviewStatus(previewState, `Preview: ${selectedFrames} frames @ ${Number(previewFps).toFixed(3)} fps`);
        } else {
            setPreviewStatus(previewState, "Preview: ready");
        }

        node.setDirtyCanvas?.(true, true);
    } catch (error) {
        if (node[PREVIEW_FETCH_TOKEN_KEY] !== requestToken) {
            return;
        }
        clearPreviewVideo(previewState);
        setPreviewStatus(previewState, `Preview: ${String(error?.message ?? error)}`, true);
    }
}

function applyDirectoryOptions(node, options) {
    const directoryWidget = getWidget(node, DIRECTORY_WIDGET_NAME);
    if (!directoryWidget) {
        return;
    }

    const values = normalizeDirectoryOptions(options);
    directoryWidget.options = directoryWidget.options ?? {};
    directoryWidget.options.values = values;
    directoryWidget.options.options = values;

    const currentValue = normalizeDirectory(directoryWidget.value);
    if (!values.includes(currentValue)) {
        directoryWidget.value = values[0] ?? ".";
        directoryWidget.callback?.(directoryWidget.value);
    }

    node.setDirtyCanvas?.(true, true);
}

function applyFileOptions(node, options) {
    const fileWidget = getWidget(node, FILE_WIDGET_NAME);
    if (!fileWidget) {
        return;
    }

    const values = normalizeFileOptions(options);
    fileWidget.options = fileWidget.options ?? {};
    fileWidget.options.values = values;
    fileWidget.options.options = values;

    const currentValue = normalizeFileName(fileWidget.value);
    if (!values.includes(currentValue)) {
        fileWidget.value = values[0] ?? "";
        fileWidget.callback?.(fileWidget.value);
    }

    node.setDirtyCanvas?.(true, true);
}

async function refreshFileOptions(node) {
    if (!isTargetNode(node)) {
        return;
    }

    const pathSourceWidget = getWidget(node, PATH_SOURCE_WIDGET_NAME);
    const directoryWidget = getWidget(node, DIRECTORY_WIDGET_NAME);
    const fileWidget = getWidget(node, FILE_WIDGET_NAME);
    if (!pathSourceWidget || !directoryWidget || !fileWidget) {
        return;
    }

    const pathSource = normalizePathSource(pathSourceWidget.value);
    const directory = normalizeDirectory(directoryWidget.value);
    const requestToken = (Number(node[FILE_FETCH_TOKEN_KEY]) || 0) + 1;
    node[FILE_FETCH_TOKEN_KEY] = requestToken;

    try {
        const options = await fetchFileOptions(pathSource, directory);
        if (node[FILE_FETCH_TOKEN_KEY] !== requestToken) {
            return;
        }
        applyFileOptions(node, options);
        await refreshPreview(node);
    } catch (error) {
        console.warn("[IPT.VideoReaderPathCombo] failed to refresh file options", error);
    }
}

async function refreshDirectoryOptions(node) {
    if (!isTargetNode(node)) {
        return;
    }

    const pathSourceWidget = getWidget(node, PATH_SOURCE_WIDGET_NAME);
    const directoryWidget = getWidget(node, DIRECTORY_WIDGET_NAME);
    if (!pathSourceWidget || !directoryWidget) {
        return;
    }

    const pathSource = normalizePathSource(pathSourceWidget.value);
    const requestToken = (Number(node[DIRECTORY_FETCH_TOKEN_KEY]) || 0) + 1;
    node[DIRECTORY_FETCH_TOKEN_KEY] = requestToken;

    try {
        const options = await fetchDirectoryOptions(pathSource);
        if (node[DIRECTORY_FETCH_TOKEN_KEY] !== requestToken) {
            return;
        }
        applyDirectoryOptions(node, options);
        await refreshFileOptions(node);
    } catch (error) {
        console.warn("[IPT.VideoReaderPathCombo] failed to refresh directory options", error);
    }
}

function patchNode(node) {
    if (!isTargetNode(node)) {
        return;
    }

    const pathSourceWidget = getWidget(node, PATH_SOURCE_WIDGET_NAME);
    const directoryWidget = getWidget(node, DIRECTORY_WIDGET_NAME);
    const fileWidget = getWidget(node, FILE_WIDGET_NAME);
    const frameCapWidget = getWidget(node, FRAME_LOAD_CAP_WIDGET_NAME);
    const skipWidget = getWidget(node, SKIP_FIRST_FRAMES_WIDGET_NAME);
    const nthWidget = getWidget(node, SELECT_EVERY_NTH_WIDGET_NAME);
    if (!pathSourceWidget || !directoryWidget || !fileWidget || !frameCapWidget || !skipWidget || !nthWidget) {
        return;
    }

    getPreviewState(node);

    if (!node[PATCHED_FLAG]) {
        pathSourceWidget.callback = chainCallback(pathSourceWidget.callback, () => {
            void refreshDirectoryOptions(node);
        });
        directoryWidget.callback = chainCallback(directoryWidget.callback, () => {
            void refreshFileOptions(node);
        });
        fileWidget.callback = chainCallback(fileWidget.callback, () => {
            void refreshPreview(node);
        });
        frameCapWidget.callback = chainCallback(frameCapWidget.callback, () => {
            void refreshPreview(node);
        });
        skipWidget.callback = chainCallback(skipWidget.callback, () => {
            void refreshPreview(node);
        });
        nthWidget.callback = chainCallback(nthWidget.callback, () => {
            void refreshPreview(node);
        });
        node[PATCHED_FLAG] = true;
    }

    void refreshDirectoryOptions(node);
}

app.registerExtension({
    name: "IPT.VideoReaderPathCombo",
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
    },
    nodeCreated(node) {
        patchNode(node);
    },
    loadedGraphNode(node) {
        patchNode(node);
    },
});
