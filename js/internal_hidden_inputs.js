import { app } from "../../scripts/app.js";

const TARGETS = [
    {
        nodeTypes: new Set([
            "IPT-DetailerStart",
            "DetailerStart",
            "Detailer Start",
        ]),
        widgetNames: ["detailer_index"],
        inputNames: [],
    },
    {
        nodeTypes: new Set([
            "IPT-DetailerEnd",
            "DetailerEnd",
            "Detailer End",
        ]),
        widgetNames: [],
        inputNames: ["accumulated_image"],
    },
    {
        nodeTypes: new Set([
            "IPT-XYPlotStart",
            "XYPlotStart",
            "XY Plot Start",
        ]),
        widgetNames: ["loop_index"],
        inputNames: [],
    },
    {
        nodeTypes: new Set([
            "IPT-XYPlotEnd",
            "XYPlotEnd",
            "XY Plot End",
        ]),
        widgetNames: [],
        inputNames: ["accumulated_images"],
    },
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

function getNodeDefCandidates(nodeData) {
    return [
        nodeData?.name,
        nodeData?.display_name,
        nodeData?.type,
        nodeData?.node_id,
    ].filter(Boolean);
}

function resolveTargetByCandidates(candidates) {
    return TARGETS.find((target) => candidates.some((candidate) => target.nodeTypes.has(candidate))) ?? null;
}

function isTargetNode(node) {
    return resolveTargetByCandidates(getNodeTypeCandidates(node)) != null;
}

function isTargetNodeDef(nodeData) {
    return resolveTargetByCandidates(getNodeDefCandidates(nodeData)) != null;
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

function removeInputSocketIfPresent(node, name) {
    if (!node?.inputs?.length || typeof node.removeInput !== "function") {
        return false;
    }

    const index = node.inputs.findIndex((input) => input?.name === name);
    if (index < 0) {
        return false;
    }

    node.removeInput(index);
    return true;
}

function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) ?? null;
}

function hideWidgetIfPresent(node, name) {
    const widget = getWidget(node, name);
    if (!widget) {
        return false;
    }

    const previousHidden = Boolean(widget.hidden);
    const previousOptionsHidden = Boolean(widget.options?.hidden);
    const options = widget.options ?? {};

    widget.hidden = true;
    widget.options = options;
    widget.options.hidden = true;

    return previousHidden !== true || previousOptionsHidden !== true;
}

function applyInternalHiddenInputs(node) {
    const target = resolveTargetByCandidates(getNodeTypeCandidates(node));
    if (!target) {
        return;
    }

    let layoutChanged = false;
    for (const widgetName of target.widgetNames ?? []) {
        layoutChanged = hideWidgetIfPresent(node, widgetName) || layoutChanged;
    }
    for (const inputName of target.inputNames) {
        layoutChanged = removeInputSocketIfPresent(node, inputName) || layoutChanged;
    }

    if (layoutChanged) {
        syncNodeLayout(node);
    }
}

app.registerExtension({
    name: "IPT.InternalHiddenInputs",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isTargetNodeDef(nodeData)) {
            return;
        }

        nodeType.prototype.onNodeCreated = chainCallback(
            nodeType.prototype.onNodeCreated,
            function onNodeCreated() {
                applyInternalHiddenInputs(this);
            },
        );

        nodeType.prototype.onConfigure = chainCallback(
            nodeType.prototype.onConfigure,
            function onConfigure() {
                applyInternalHiddenInputs(this);
            },
        );

        nodeType.prototype.onGraphConfigured = chainCallback(
            nodeType.prototype.onGraphConfigured,
            function onGraphConfigured() {
                applyInternalHiddenInputs(this);
            },
        );
    },
    nodeCreated(node) {
        if (isTargetNode(node)) {
            applyInternalHiddenInputs(node);
        }
    },
    loadedGraphNode(node) {
        if (isTargetNode(node)) {
            applyInternalHiddenInputs(node);
        }
    },
});
