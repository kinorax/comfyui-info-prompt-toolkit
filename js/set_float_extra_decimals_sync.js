import { app } from "../../scripts/app.js";

const TARGET_NODE_TYPES = new Set([
    "IPT-SetFloatExtra",
    "SetFloatExtra",
    "Set Float Extra",
]);

const DECIMALS_WIDGET_NAME = "decimals";
const VALUE_WIDGET_NAME = "value";
const PATCHED_FLAG = "__iisSetFloatExtraPatched";

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

function normalizeRangeBound(value, fallback) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
}

function clampDecimals(raw, min, max) {
    const n = Number(raw);
    if (!Number.isFinite(n)) {
        return min;
    }
    const v = Math.trunc(n);
    return Math.max(min, Math.min(max, v));
}

function stepFromDecimals(decimals) {
    if (decimals <= 0) {
        return 1;
    }
    return Number((1 / Math.pow(10, decimals)).toFixed(decimals));
}

function syncSetFloatExtraWidgets(node) {
    const valueWidget = getWidget(node, VALUE_WIDGET_NAME);
    const decimalsWidget = getWidget(node, DECIMALS_WIDGET_NAME);
    if (!valueWidget || !decimalsWidget || !valueWidget.options) {
        return;
    }

    const minDecimals = normalizeRangeBound(decimalsWidget.options?.min, 0);
    const maxDecimals = normalizeRangeBound(decimalsWidget.options?.max, 10);
    const decimals = clampDecimals(decimalsWidget.value, minDecimals, maxDecimals);
    const step = stepFromDecimals(decimals);

    valueWidget.options.precision = decimals;
    valueWidget.options.step2 = step;
    valueWidget.options.step = step * 10;
    valueWidget.options.round = step;

    const currentValue = Number(valueWidget.value);
    if (Number.isFinite(currentValue)) {
        valueWidget.value = Number(currentValue.toFixed(decimals));
    }

    node.setDirtyCanvas?.(true, true);
}

function patchNode(node) {
    if (!isTargetNode(node)) {
        return;
    }

    const decimalsWidget = getWidget(node, DECIMALS_WIDGET_NAME);
    if (decimalsWidget && !decimalsWidget[PATCHED_FLAG]) {
        decimalsWidget.callback = chainCallback(decimalsWidget.callback, () => {
            syncSetFloatExtraWidgets(node);
        });
        decimalsWidget[PATCHED_FLAG] = true;
    }

    syncSetFloatExtraWidgets(node);
}

app.registerExtension({
    name: "IPT.SetFloatExtraDecimalsSync",
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
