// Copyright 2026 kinorax
import { app } from "../../scripts/app.js";

const TARGET_NODE_TYPES = new Set([
    "IPT-VideoSaver",
    "VideoSaver",
    "Video Saver",
]);
const CODEC_WIDGET_NAME = "codec";
const AV1_CRF_WIDGET_NAME = "av1_crf";
const H264_CRF_WIDGET_NAME = "h264_crf";
const FRAME_RATE_WIDGET_NAME = "frame_rate";
const CALLBACK_PATCHED_FLAG = "__iptVideoSaverCodecCallbackPatched";
const SYNC_FLAG = "__iptVideoSaverRuntimeSyncing";

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

function applyFrameRateWidgetOptions(widget) {
    if (!widget) {
        return false;
    }

    widget.options = widget.options ?? {};
    const previousPrecision = widget.options.precision;
    const previousStep = widget.options.step;
    const previousStep2 = widget.options.step2;
    const previousRound = widget.options.round;

    widget.options.precision = 3;
    widget.options.step = 1;
    widget.options.step2 = 1;
    widget.options.round = 0.001;

    return (
        previousPrecision !== 3
        || previousStep !== 1
        || previousStep2 !== 1
        || previousRound !== 0.001
    );
}

function patchCodecWidget(node) {
    const codecWidget = getWidget(node, CODEC_WIDGET_NAME);
    if (!codecWidget || codecWidget[CALLBACK_PATCHED_FLAG]) {
        return;
    }

    codecWidget.callback = chainCallback(codecWidget.callback, () => {
        syncVideoSaverWidgets(node);
    });
    codecWidget[CALLBACK_PATCHED_FLAG] = true;
}

function syncVideoSaverWidgets(node) {
    if (!isTargetNode(node) || node?.[SYNC_FLAG]) {
        return;
    }
    node[SYNC_FLAG] = true;

    try {
        const codecWidget = getWidget(node, CODEC_WIDGET_NAME);
        const av1CrfWidget = getWidget(node, AV1_CRF_WIDGET_NAME);
        const h264CrfWidget = getWidget(node, H264_CRF_WIDGET_NAME);
        const frameRateWidget = getWidget(node, FRAME_RATE_WIDGET_NAME);
        if (!codecWidget) {
            return;
        }

        patchCodecWidget(node);

        const codecValue = String(codecWidget.value ?? "").trim().toLowerCase();
        let layoutChanged = false;
        layoutChanged = setWidgetVisibility(av1CrfWidget, codecValue !== "h264") || layoutChanged;
        layoutChanged = setWidgetVisibility(h264CrfWidget, codecValue === "h264") || layoutChanged;
        layoutChanged = applyFrameRateWidgetOptions(frameRateWidget) || layoutChanged;

        if (layoutChanged) {
            syncNodeLayout(node);
        }
    } finally {
        node[SYNC_FLAG] = false;
    }
}

app.registerExtension({
    name: "IPT.VideoSaverRuntimeSync",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isTargetNodeDef(nodeData)) {
            return;
        }

        nodeType.prototype.onNodeCreated = chainCallback(
            nodeType.prototype.onNodeCreated,
            function onNodeCreated() {
                syncVideoSaverWidgets(this);
            },
        );

        nodeType.prototype.onConfigure = chainCallback(
            nodeType.prototype.onConfigure,
            function onConfigure() {
                syncVideoSaverWidgets(this);
            },
        );

        nodeType.prototype.onGraphConfigured = chainCallback(
            nodeType.prototype.onGraphConfigured,
            function onGraphConfigured() {
                syncVideoSaverWidgets(this);
            },
        );
    },
    nodeCreated(node) {
        syncVideoSaverWidgets(node);
    },
    loadedGraphNode(node) {
        syncVideoSaverWidgets(node);
    },
});
