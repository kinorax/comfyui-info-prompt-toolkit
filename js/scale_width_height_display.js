// Copyright 2026 kinorax
import { app } from "../../scripts/app.js";

const TARGET_NODE_TYPES = new Set([
    "IPT-ScaleWidthHeight",
    "ScaleWidthHeight",
    "Scale (width x height)",
]);

const ACTUAL_RATIO_WIDGET_NAME = "actual_ratio";

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

function normalizeTextOrNull(value) {
    if (value == null) {
        return null;
    }
    const text = String(value).trim();
    return text ? text : null;
}

function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) ?? null;
}

function removeInputSocketIfPresent(node, name) {
    if (!node?.inputs?.length || typeof node.removeInput !== "function") {
        return;
    }

    const index = node.inputs.findIndex((input) => input?.name === name);
    if (index >= 0) {
        node.removeInput(index);
    }
}

function applyReadOnly(widget) {
    if (!widget) {
        return;
    }

    widget.options = widget.options ?? {};
    widget.options.readonly = true;
    widget.options.read_only = true;
    widget.options.disabled = false;

    const element = widget.element ?? widget.inputEl ?? null;
    if (element && typeof element === "object") {
        if ("readOnly" in element) {
            element.readOnly = true;
        }
        if ("disabled" in element) {
            element.disabled = false;
        }
    }
}

function applyActualRatioWidgetValue(node) {
    const widget = getWidget(node, ACTUAL_RATIO_WIDGET_NAME);
    if (!widget) {
        return;
    }

    const ratioText = normalizeTextOrNull(node.__iisScaleActualRatioValue) ?? "";
    if (widget.value !== ratioText) {
        widget.value = ratioText;
        widget.callback?.(ratioText);
    }
}

function setupActualRatioDisplayField(node) {
    removeInputSocketIfPresent(node, ACTUAL_RATIO_WIDGET_NAME);
    applyReadOnly(getWidget(node, ACTUAL_RATIO_WIDGET_NAME));
    applyActualRatioWidgetValue(node);
}

function extractActualRatioFromOutput(output) {
    if (!output || typeof output !== "object") {
        return null;
    }

    const uiValue = output?.ui?.actual_ratio;
    if (Array.isArray(uiValue) && uiValue.length > 0) {
        return normalizeTextOrNull(uiValue[0]);
    }
    return normalizeTextOrNull(uiValue);
}

function updateCachedActualRatioFromOutput(node, output) {
    const ratioText = extractActualRatioFromOutput(output);
    if (ratioText != null) {
        node.__iisScaleActualRatioValue = ratioText;
    }
}

app.registerExtension({
    name: "IPT.ScaleWidthHeightDisplay",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isTargetNodeDef(nodeData)) {
            return;
        }

        nodeType.prototype.onNodeCreated = chainCallback(
            nodeType.prototype.onNodeCreated,
            function onNodeCreated() {
                setupActualRatioDisplayField(this);
            },
        );

        nodeType.prototype.onConfigure = chainCallback(
            nodeType.prototype.onConfigure,
            function onConfigure() {
                setupActualRatioDisplayField(this);
            },
        );

        nodeType.prototype.onGraphConfigured = chainCallback(
            nodeType.prototype.onGraphConfigured,
            function onGraphConfigured() {
                setupActualRatioDisplayField(this);
            },
        );

        nodeType.prototype.onExecuted = chainCallback(
            nodeType.prototype.onExecuted,
            function onExecuted(output) {
                updateCachedActualRatioFromOutput(this, output);
                setupActualRatioDisplayField(this);
            },
        );
    },
    nodeCreated(node) {
        if (isTargetNode(node)) {
            setupActualRatioDisplayField(node);
        }
    },
    loadedGraphNode(node) {
        if (isTargetNode(node)) {
            setupActualRatioDisplayField(node);
        }
    },
    onNodeOutputsUpdated(nodeOutputs) {
        if (!app?.rootGraph || !nodeOutputs || typeof nodeOutputs !== "object") {
            return;
        }

        for (const node of app.rootGraph._nodes || []) {
            if (!isTargetNode(node)) {
                continue;
            }

            const output = nodeOutputs[String(node.id)];
            if (output) {
                updateCachedActualRatioFromOutput(node, output);
            }
            setupActualRatioDisplayField(node);
        }
    },
});
