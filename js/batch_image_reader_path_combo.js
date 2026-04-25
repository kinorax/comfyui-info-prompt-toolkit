// Copyright 2026 kinorax
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE_TYPES = new Set([
    "IPT-ImageDirectoryReader",
    "IPT-BatchImageReader",
    "BatchImageReader",
    "ImageDirectoryReader",
    "Batch Image Reader",
    "Image Directory Reader",
    "IPT-CaptionFileSaver",
    "CaptionFileSaver",
    "Caption File Saver",
]);

const PATH_SOURCE_WIDGET_NAME = "path_source";
const PATH_WIDGET_NAME = "path";
const PATCHED_FLAG = "__iisBatchImageReaderPathPatched";
const FETCH_TOKEN_KEY = "__iisBatchImageReaderPathFetchToken";

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

function normalizeDirectoryOptions(rawValues) {
    const values = [];
    const seen = new Set();

    const push = (value) => {
        const text = String(value ?? "").trim().replace(/\\+/g, "/");
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

async function fetchPathOptions(pathSource) {
    if (!api || typeof api.fetchApi !== "function") {
        return ["."];
    }

    const query = new URLSearchParams({ path_source: pathSource }).toString();
    const response = await api.fetchApi(`/iis/batch-image-reader/directories?${query}`, {
        method: "GET",
        cache: "no-store",
    });

    if (!response?.ok) {
        throw new Error(`failed to fetch directories (${response?.status ?? "unknown"})`);
    }

    const payload = await response.json();
    if (!payload || payload.ok !== true || !Array.isArray(payload.directories)) {
        throw new Error("invalid directory payload");
    }

    return normalizeDirectoryOptions(payload.directories);
}

function applyPathOptions(node, options) {
    const pathWidget = getWidget(node, PATH_WIDGET_NAME);
    if (!pathWidget) {
        return;
    }

    const values = normalizeDirectoryOptions(options);

    pathWidget.options = pathWidget.options ?? {};
    pathWidget.options.values = values;
    pathWidget.options.options = values;

    const currentValue = String(pathWidget.value ?? "").trim();
    if (!values.includes(currentValue)) {
        pathWidget.value = values[0] ?? ".";
        pathWidget.callback?.(pathWidget.value);
    }

    node.setDirtyCanvas?.(true, true);
}

async function refreshPathOptions(node) {
    if (!isTargetNode(node)) {
        return;
    }

    const sourceWidget = getWidget(node, PATH_SOURCE_WIDGET_NAME);
    const pathWidget = getWidget(node, PATH_WIDGET_NAME);
    if (!sourceWidget || !pathWidget) {
        return;
    }

    const pathSource = normalizePathSource(sourceWidget.value);
    const requestToken = (Number(node[FETCH_TOKEN_KEY]) || 0) + 1;
    node[FETCH_TOKEN_KEY] = requestToken;

    try {
        const options = await fetchPathOptions(pathSource);
        if (node[FETCH_TOKEN_KEY] !== requestToken) {
            return;
        }
        applyPathOptions(node, options);
    } catch (error) {
        console.warn("[IPT.BatchImageReaderPathCombo] failed to refresh path options", error);
    }
}

function patchNode(node) {
    if (!isTargetNode(node)) {
        return;
    }

    const sourceWidget = getWidget(node, PATH_SOURCE_WIDGET_NAME);
    if (!sourceWidget) {
        return;
    }

    if (!node[PATCHED_FLAG]) {
        sourceWidget.callback = chainCallback(sourceWidget.callback, () => {
            void refreshPathOptions(node);
        });
        node[PATCHED_FLAG] = true;
    }

    void refreshPathOptions(node);
}

app.registerExtension({
    name: "IPT.BatchImageReaderPathCombo",
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
