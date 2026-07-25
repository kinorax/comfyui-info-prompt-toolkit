// Copyright 2026 kinorax
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE_TYPES = new Set([
    "IPT-VideoSaver",
    "IPT-VideoSaverV2",
    "VideoSaver",
    "VideoSaverV2",
    "Video Saver",
    "Video Saver (Deprecated)",
]);
const VIDEO_SAVER_V2_NODE_TYPES = new Set([
    "IPT-VideoSaverV2",
    "VideoSaverV2",
]);
const CODEC_WIDGET_NAME = "codec";
const AV1_CRF_WIDGET_NAME = "av1_crf";
const H264_CRF_WIDGET_NAME = "h264_crf";
const VP9_CRF_WIDGET_NAME = "vp9_crf";
const FRAME_RATE_WIDGET_NAME = "frame_rate";
const COLOR_ENCODING_WIDGET_NAME = "color_encoding";
const COLOR_RANGE_WIDGET_NAME = "color_range";
const CHROMA_SUBSAMPLING_WIDGET_NAME = "chroma_subsampling";
const FFV1_CODEC_VALUES = new Set([
    "ffv1_v3_rgb8",
    "ffv1_v3_rgb16",
]);
const VIDEO_SAVER_NOTIFICATION_UI_KEY = "ipt_video_saver_notifications";
const CALLBACK_PATCHED_FLAG = "__iptVideoSaverCodecCallbackPatched";
const OPTION_LABELS_PATCHED_FLAG = "__iptVideoSaverOptionLabelsPatched";
const SYNC_FLAG = "__iptVideoSaverRuntimeSyncing";
const SHOWN_NOTIFICATION_LIMIT = 200;
const shownNotificationKeys = new Set();
const shownNotificationOrder = [];
const CODEC_OPTION_LABELS = Object.freeze({
    av1: "AV1 (MP4)",
    h264: "H.264 (MP4)",
    vp9: "VP9 (WebM)",
    ffv1_v3_rgb8: "FFV1 v3 RGB 8-bit (MKV)",
    ffv1_v3_rgb16: "FFV1 v3 RGB 16-bit (MKV)",
});
const COLOR_ENCODING_OPTION_LABELS = Object.freeze({
    "BT.709": "BT.709",
    "BT.709 (10-bit)": "BT.709 (10-bit)",
    "BT.2020": "BT.2020",
});
const COLOR_RANGE_OPTION_LABELS = Object.freeze({
    limited: "Limited",
    full: "Full",
});
const CHROMA_SUBSAMPLING_OPTION_LABELS = Object.freeze({
    "4:2:0": "4:2:0",
    "4:4:4": "4:4:4",
});

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

function isVideoSaverV2Node(node) {
    return getNodeTypeCandidates(node).some(
        (candidate) => VIDEO_SAVER_V2_NODE_TYPES.has(candidate),
    );
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

function applyOptionLabels(widget, labels) {
    if (!widget || widget[OPTION_LABELS_PATCHED_FLAG]) {
        return false;
    }

    widget.options = widget.options ?? {};
    const originalGetOptionLabel = widget.options.getOptionLabel;
    widget.options.getOptionLabel = (value) => {
        const normalizedValue = String(value ?? "");
        return labels[normalizedValue]
            ?? originalGetOptionLabel?.(value)
            ?? normalizedValue;
    };
    widget[OPTION_LABELS_PATCHED_FLAG] = true;
    return true;
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
        const vp9CrfWidget = getWidget(node, VP9_CRF_WIDGET_NAME);
        const frameRateWidget = getWidget(node, FRAME_RATE_WIDGET_NAME);
        const colorEncodingWidget = getWidget(node, COLOR_ENCODING_WIDGET_NAME);
        const colorRangeWidget = getWidget(node, COLOR_RANGE_WIDGET_NAME);
        const chromaSubsamplingWidget = getWidget(node, CHROMA_SUBSAMPLING_WIDGET_NAME);
        if (!codecWidget) {
            return;
        }

        patchCodecWidget(node);

        const codecValue = String(codecWidget.value ?? "").trim().toLowerCase();
        let layoutChanged = false;
        layoutChanged = setWidgetVisibility(av1CrfWidget, codecValue === "av1") || layoutChanged;
        layoutChanged = setWidgetVisibility(h264CrfWidget, codecValue === "h264") || layoutChanged;
        layoutChanged = setWidgetVisibility(vp9CrfWidget, codecValue === "vp9") || layoutChanged;
        layoutChanged = applyFrameRateWidgetOptions(frameRateWidget) || layoutChanged;
        if (isVideoSaverV2Node(node)) {
            const showSelectableColorEncoding = !FFV1_CODEC_VALUES.has(codecValue);
            layoutChanged = setWidgetVisibility(
                colorEncodingWidget,
                showSelectableColorEncoding,
            ) || layoutChanged;
            layoutChanged = setWidgetVisibility(
                colorRangeWidget,
                showSelectableColorEncoding,
            ) || layoutChanged;
            layoutChanged = setWidgetVisibility(
                chromaSubsamplingWidget,
                showSelectableColorEncoding,
            ) || layoutChanged;
            layoutChanged = applyOptionLabels(codecWidget, CODEC_OPTION_LABELS) || layoutChanged;
            layoutChanged = applyOptionLabels(
                colorEncodingWidget,
                COLOR_ENCODING_OPTION_LABELS,
            ) || layoutChanged;
            layoutChanged = applyOptionLabels(
                colorRangeWidget,
                COLOR_RANGE_OPTION_LABELS,
            ) || layoutChanged;
            layoutChanged = applyOptionLabels(
                chromaSubsamplingWidget,
                CHROMA_SUBSAMPLING_OPTION_LABELS,
            ) || layoutChanged;
        }

        if (layoutChanged) {
            syncNodeLayout(node);
        }
    } finally {
        node[SYNC_FLAG] = false;
    }
}

function showVideoSaverNotification(notification) {
    const summary = String(
        notification?.summary ?? "Video Saver: fallback applied",
    );
    const detail = String(notification?.detail ?? "");
    const toast = app?.extensionManager?.toast;

    if (typeof toast?.add === "function") {
        toast.add({
            severity: "warn",
            summary,
            detail,
            life: 12000,
        });
        return;
    }
    if (typeof toast?.addAlert === "function") {
        toast.addAlert(`${summary}\n${detail}`);
        return;
    }
    globalThis.alert?.(`${summary}\n\n${detail}`);
}

function rememberNotification(key) {
    if (shownNotificationKeys.has(key)) {
        return false;
    }

    shownNotificationKeys.add(key);
    shownNotificationOrder.push(key);
    if (shownNotificationOrder.length > SHOWN_NOTIFICATION_LIMIT) {
        const expiredKey = shownNotificationOrder.shift();
        shownNotificationKeys.delete(expiredKey);
    }
    return true;
}

function handleVideoSaverExecuted(event) {
    const execution = event?.detail;
    const notifications = execution?.output?.[VIDEO_SAVER_NOTIFICATION_UI_KEY];
    if (!Array.isArray(notifications)) {
        return;
    }

    for (const notification of notifications) {
        if (!notification || typeof notification !== "object") {
            continue;
        }
        const key = [
            execution?.prompt_id ?? "",
            execution?.node ?? "",
            notification.summary ?? "",
            notification.detail ?? "",
        ].join(":");
        if (rememberNotification(key)) {
            showVideoSaverNotification(notification);
        }
    }
}

app.registerExtension({
    name: "IPT.VideoSaverRuntimeSync",
    setup() {
        api.addEventListener("executed", handleVideoSaverExecuted);
    },
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
