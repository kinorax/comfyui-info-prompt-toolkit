import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    getSelectorSlots,
    getWidget,
    isSelectorTargetNodeDef,
    normalizeSelectedPath,
    normalizeSha256,
    resolveSelectorTarget,
} from "./selector_targets.js";

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

function syncNodeWidgets(node) {
    if (!node) {
        return;
    }

    if (Array.isArray(node.widgets)) {
        try {
            node.widgets = [...node.widgets];
        } catch {
            // Legacy frontends may expose widgets as a plain array only.
        }
    }

    node.setDirtyCanvas?.(true, true);
    requestVueNodeRefresh(node);
}

function hideSocketlessWidget(widget) {
    if (!widget) {
        return false;
    }

    const previousHidden = Boolean(widget.hidden);
    const previousOptionsHidden = Boolean(widget.options?.hidden);
    const previousReadOnly = widget.options?.readonly;
    const previousReadOnlyAlt = widget.options?.read_only;
    const previousDisabled = widget.options?.disabled;
    const options = widget.options ?? {};

    widget.hidden = true;
    widget.options = options;
    widget.options.hidden = true;
    widget.options.readonly = true;
    widget.options.read_only = true;
    widget.options.disabled = false;

    return (
        previousHidden !== true
        || previousOptionsHidden !== true
        || previousReadOnly !== true
        || previousReadOnlyAlt !== true
        || previousDisabled !== false
    );
}

function getSha256Widget(node, slot) {
    return getWidget(node, slot?.sha256WidgetName);
}

function hideSha256Widget(node, slot) {
    return hideSocketlessWidget(getSha256Widget(node, slot));
}

function setSha256WidgetValue(node, slot, nextValue, { clearOnly = false } = {}) {
    const widget = getSha256Widget(node, slot);
    if (!widget) {
        return;
    }

    const normalized = normalizeSha256(nextValue);
    if (!normalized && !clearOnly) {
        return;
    }

    const finalValue = normalized || "";
    if (String(widget.value ?? "") === finalValue) {
        return;
    }

    widget.value = finalValue;
    node.setDirtyCanvas?.(true, true);
}

function getSlotState(node, slot) {
    const stateKey = "__iptSelectorSha256State";
    node[stateKey] = node[stateKey] ?? {};
    const slotKey = String(slot?.widgetName ?? "");
    node[stateKey][slotKey] = node[stateKey][slotKey] ?? {
        requestId: 0,
        selection: "",
    };
    return node[stateKey][slotKey];
}

async function enqueuePriority(folderName, relativePath) {
    const folder = String(folderName ?? "").trim();
    const relative = normalizeSelectedPath(relativePath);
    if (!folder || !relative || !api || typeof api.fetchApi !== "function") {
        return;
    }

    try {
        await api.fetchApi("/ipt/model-metadata/queue-priority", {
            method: "POST",
            headers: {
                "content-type": "application/json",
            },
            body: JSON.stringify({
                folder_name: folder,
                relative_path: relative,
            }),
        });
    } catch (error) {
        console.warn("[IPT.SelectorMetadataQueue] enqueue failed", error);
    }
}

async function resolveReference(folderName, relativePath, sha256, { enqueueLocalHash = false } = {}) {
    const folder = String(folderName ?? "").trim();
    const relative = normalizeSelectedPath(relativePath);
    const digest = normalizeSha256(sha256);
    if (!folder || !api || typeof api.fetchApi !== "function") {
        return null;
    }

    const response = await api.fetchApi("/ipt/model-reference/resolve", {
        method: "POST",
        headers: {
            "content-type": "application/json",
        },
        body: JSON.stringify({
            folder_name: folder,
            relative_path: relative || undefined,
            sha256: digest || undefined,
            name_raw: relative || undefined,
            enqueue_local_hash: Boolean(enqueueLocalHash),
            resolve_remote: false,
            include_lora_tags: false,
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

function refreshSha256ForSelection(node, slot, { clearBeforeRequest = false, preserveOnEmpty = true } = {}) {
    const selectionWidget = getWidget(node, slot.widgetName);
    const sha256Widget = getSha256Widget(node, slot);
    if (!selectionWidget || !sha256Widget) {
        return;
    }

    const selected = normalizeSelectedPath(selectionWidget.value);
    const currentSha256 = normalizeSha256(sha256Widget.value);
    const requestSelection = `${slot.folderName}:${selected}`;
    const slotState = getSlotState(node, slot);

    slotState.selection = requestSelection;
    slotState.requestId += 1;
    const requestId = slotState.requestId;

    if (!selected) {
        setSha256WidgetValue(node, slot, "", { clearOnly: true });
        return;
    }

    if (clearBeforeRequest) {
        setSha256WidgetValue(node, slot, "", { clearOnly: true });
    }

    const requestSha256 = clearBeforeRequest ? "" : currentSha256;
    void resolveReference(slot.folderName, selected, requestSha256, { enqueueLocalHash: true })
        .then((payload) => {
            if (!payload || slotState.requestId !== requestId) {
                return;
            }
            if (slotState.selection !== requestSelection) {
                return;
            }

            const nextSha256 = normalizeSha256(payload?.sha256);
            if (nextSha256) {
                setSha256WidgetValue(node, slot, nextSha256);
                return;
            }

            if (!preserveOnEmpty) {
                setSha256WidgetValue(node, slot, "", { clearOnly: true });
            }
        })
        .catch((error) => {
            if (slotState.requestId !== requestId) {
                return;
            }
            console.warn("[IPT.SelectorMetadataQueue] sha256 resolve failed", error);
        });
}

function patchNode(node) {
    const target = resolveSelectorTarget(node);
    if (!target) {
        return;
    }

    let widgetLayoutChanged = false;

    for (const slot of getSelectorSlots(target)) {
        widgetLayoutChanged = hideSha256Widget(node, slot) || widgetLayoutChanged;

        const widget = getWidget(node, slot.widgetName);
        if (!widget) {
            continue;
        }

        const patchFlagKey = `__iptSelectorQueuePatched_${slot.widgetName}`;
        if (!widget[patchFlagKey]) {
            widget.callback = chainCallback(widget.callback, (value) => {
                const selected = normalizeSelectedPath(value ?? widget.value);
                void enqueuePriority(slot.folderName, selected);
                refreshSha256ForSelection(node, slot, {
                    clearBeforeRequest: true,
                    preserveOnEmpty: false,
                });
            });
            widget[patchFlagKey] = true;
        }

        refreshSha256ForSelection(node, slot, {
            clearBeforeRequest: false,
            preserveOnEmpty: true,
        });
    }

    if (widgetLayoutChanged) {
        syncNodeWidgets(node);
    }
}

app.registerExtension({
    name: "IPT.SelectorMetadataQueue",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isSelectorTargetNodeDef(nodeData)) {
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
