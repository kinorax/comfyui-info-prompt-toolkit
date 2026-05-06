// Copyright 2026 kinorax
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE_TYPES = new Set([
    "IPT-ReleaseMemory",
    "ReleaseMemory",
    "Release Memory",
]);

const BUTTON_LABEL = "Release Now";
const RUNNING_LABEL = "Releasing...";
const DONE_LABEL = "Released";
const FAILED_LABEL = "Failed";
const BUTTON_RESTORE_DELAY_MS = 1200;
const PATCHED_FLAG = "__iptReleaseMemoryButtonPatched";
const RUNNING_FLAG = "__iptReleaseMemoryRunning";

const OPTION_WIDGETS = [
    ["generation_runtime", true],
    ["sam3_runtime", true],
    ["pixai_tagger_runtime", true],
    ["gc_cuda_cleanup", true],
];

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

function boolOrDefault(value, defaultValue) {
    if (value == null) {
        return defaultValue;
    }
    if (typeof value === "boolean") {
        return value;
    }
    if (typeof value === "number") {
        return Boolean(value);
    }

    const text = String(value).trim().toLowerCase();
    if (["1", "true", "yes", "on"].includes(text)) {
        return true;
    }
    if (["0", "false", "no", "off"].includes(text)) {
        return false;
    }
    return defaultValue;
}

function buildPayload(node) {
    const payload = {};
    for (const [widgetName, defaultValue] of OPTION_WIDGETS) {
        const widget = getWidget(node, widgetName);
        payload[widgetName] = boolOrDefault(widget?.value, defaultValue);
    }
    return payload;
}

function setButtonLabel(node, widget, label) {
    if (!widget) {
        return;
    }
    widget.label = label;
    node?.setDirtyCanvas?.(true, true);
}

function setButtonDisabled(node, widget, disabled) {
    if (!widget) {
        return;
    }
    widget.disabled = Boolean(disabled);
    node?.setDirtyCanvas?.(true, true);
}

function showError(message) {
    const text = String(message || "Release Memory failed");
    if (typeof alert === "function") {
        alert(text);
    } else {
        console.warn("[IPT.ReleaseMemory]", text);
    }
}

async function releaseNow(node, widget) {
    if (!isTargetNode(node) || node?.[RUNNING_FLAG]) {
        return;
    }
    if (!api || typeof api.fetchApi !== "function") {
        showError("ComfyUI API is unavailable.");
        return;
    }

    node[RUNNING_FLAG] = true;
    setButtonDisabled(node, widget, true);
    setButtonLabel(node, widget, RUNNING_LABEL);

    try {
        const response = await api.fetchApi("/ipt/release-memory", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(buildPayload(node)),
        });

        let payload = null;
        try {
            payload = await response.json();
        } catch {
            payload = null;
        }

        if (!response?.ok || !payload?.ok) {
            const message = payload?.message || payload?.error || `HTTP ${response?.status ?? "unknown"}`;
            throw new Error(message);
        }

        console.info("[IPT.ReleaseMemory] released", payload);
        setButtonLabel(node, widget, DONE_LABEL);
    } catch (error) {
        console.warn("[IPT.ReleaseMemory] release failed", error);
        setButtonLabel(node, widget, FAILED_LABEL);
        showError(error?.message ?? error);
    } finally {
        window.setTimeout(() => {
            node[RUNNING_FLAG] = false;
            setButtonDisabled(node, widget, false);
            setButtonLabel(node, widget, BUTTON_LABEL);
        }, BUTTON_RESTORE_DELAY_MS);
    }
}

function ensureReleaseButton(node) {
    if (!isTargetNode(node) || typeof node?.addWidget !== "function") {
        return;
    }

    const existing = getWidget(node, BUTTON_LABEL);
    if (existing) {
        existing.serialize = false;
        existing.label = BUTTON_LABEL;
        return;
    }

    const widget = node.addWidget("button", BUTTON_LABEL, BUTTON_LABEL, () => {
        void releaseNow(node, widget);
    });
    widget.serialize = false;
    widget.label = BUTTON_LABEL;
}

app.registerExtension({
    name: "IPT.ReleaseMemory",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isTargetNodeDef(nodeData)) {
            return;
        }
        if (nodeType.prototype[PATCHED_FLAG]) {
            return;
        }

        nodeType.prototype.onNodeCreated = chainCallback(
            nodeType.prototype.onNodeCreated,
            function onNodeCreated() {
                ensureReleaseButton(this);
            },
        );

        nodeType.prototype.onConfigure = chainCallback(
            nodeType.prototype.onConfigure,
            function onConfigure() {
                ensureReleaseButton(this);
            },
        );

        nodeType.prototype.onGraphConfigured = chainCallback(
            nodeType.prototype.onGraphConfigured,
            function onGraphConfigured() {
                ensureReleaseButton(this);
            },
        );

        nodeType.prototype[PATCHED_FLAG] = true;
    },
});
