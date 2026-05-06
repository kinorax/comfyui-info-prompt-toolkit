// Copyright 2026 kinorax
import { app } from "../../scripts/app.js";

const TARGET_NODE_TYPES = new Set([
    "IPT-ImageSaver",
    "ImageSaver",
    "Image Saver",
]);
const FORMAT_WIDGET_NAME = "output_format";
const QUALITY_WIDGET_NAME = "quality";
const CALLBACK_PATCHED_FLAG = "__iptImageSaverFormatCallbackPatched";
const SYNC_FLAG = "__iptImageSaverRuntimeSyncing";

function getNodeTypeCandidates(node) {
    return [
        node?.comfyClass,
        node?.type,
        node?.constructor?.comfyClass,
        node?.constructor?.type,
        node?.title,
    ].filter(Boolean);
}

function getNodeDefCandidates(nodeData) {
    return [
        nodeData?.name,
        nodeData?.display_name,
        nodeData?.type,
        nodeData?.node_id,
    ].filter(Boolean);
}

function isTargetNode(node) {
    return getNodeTypeCandidates(node).some((candidate) => TARGET_NODE_TYPES.has(candidate));
}

function isTargetNodeDef(nodeData) {
    return getNodeDefCandidates(nodeData).some((candidate) => TARGET_NODE_TYPES.has(candidate));
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

function syncNodeLayout(node) {
    if (Array.isArray(node?.widgets)) {
        try {
            node.widgets = [...node.widgets];
        } catch {
            // Legacy frontends may expose widgets as a plain array only.
        }
    }
    node?.setDirtyCanvas?.(true, true);
    if (Array.isArray(node?.size) && typeof node?.setSize === "function") {
        node.setSize([...node.size]);
    }
    requestVueNodeRefresh(node);
}

function setWidgetVisibility(widget, visible) {
    if (!widget) {
        return false;
    }

    const options = widget.options ?? {};
    const nextHidden = !visible;
    const previousHidden = Boolean(widget.hidden);
    const previousOptionsHidden = Boolean(options.hidden);

    widget.hidden = nextHidden;
    widget.options = options;
    widget.options.hidden = nextHidden;

    return previousHidden !== nextHidden || previousOptionsHidden !== nextHidden;
}

function patchFormatWidget(node) {
    const formatWidget = getWidget(node, FORMAT_WIDGET_NAME);
    if (!formatWidget || formatWidget[CALLBACK_PATCHED_FLAG]) {
        return;
    }

    formatWidget.callback = chainCallback(formatWidget.callback, () => {
        syncImageSaverWidgets(node);
    });
    formatWidget[CALLBACK_PATCHED_FLAG] = true;
}

function syncImageSaverWidgets(node) {
    if (!isTargetNode(node) || node?.[SYNC_FLAG]) {
        return;
    }
    node[SYNC_FLAG] = true;

    try {
        const formatWidget = getWidget(node, FORMAT_WIDGET_NAME);
        const qualityWidget = getWidget(node, QUALITY_WIDGET_NAME);
        if (!formatWidget) {
            return;
        }

        patchFormatWidget(node);

        const formatValue = String(formatWidget.value ?? "").trim().toLowerCase();
        const layoutChanged = setWidgetVisibility(qualityWidget, formatValue !== "png");

        if (layoutChanged) {
            syncNodeLayout(node);
        }
    } finally {
        node[SYNC_FLAG] = false;
    }
}

app.registerExtension({
    name: "IPT.ImageSaverRuntimeSync",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isTargetNodeDef(nodeData)) {
            return;
        }

        nodeType.prototype.onNodeCreated = chainCallback(
            nodeType.prototype.onNodeCreated,
            function onNodeCreated() {
                syncImageSaverWidgets(this);
            },
        );

        nodeType.prototype.onConfigure = chainCallback(
            nodeType.prototype.onConfigure,
            function onConfigure() {
                syncImageSaverWidgets(this);
            },
        );

        nodeType.prototype.onGraphConfigured = chainCallback(
            nodeType.prototype.onGraphConfigured,
            function onGraphConfigured() {
                syncImageSaverWidgets(this);
            },
        );
    },
    nodeCreated(node) {
        syncImageSaverWidgets(node);
    },
    loadedGraphNode(node) {
        syncImageSaverWidgets(node);
    },
});
