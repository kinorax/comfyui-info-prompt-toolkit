// Copyright 2026 kinorax
import { app } from "../../scripts/app.js";

const TARGET_NODE_TYPES = new Set([
    "IPT-SeedGenerator",
    "SeedGenerator",
    "Seed Generator",
]);

const SEED_OUTPUT_INDEX = 0;
const DEFAULT_OUTPUT_LABEL = "seed";
const SEED_WIDGET_NAME = "generated_seed";

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

function normalizeSeedTextOrNone(value) {
    if (value == null) {
        return null;
    }
    const text = String(value).trim();
    return text ? text : null;
}

function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) ?? null;
}

function removeSeedInputSocketIfPresent(node) {
    if (!node?.inputs?.length || typeof node.removeInput !== "function") {
        return;
    }

    const index = node.inputs.findIndex((input) => input?.name === SEED_WIDGET_NAME);
    if (index >= 0) {
        node.removeInput(index);
    }
}

function applySeedWidgetValue(node) {
    const widget = getWidget(node, SEED_WIDGET_NAME);
    if (!widget) {
        return;
    }

    const cachedSeedText = normalizeSeedTextOrNone(node.__iisSeedGeneratorValue);
    const widgetSeedText = normalizeSeedTextOrNone(widget.value);
    const seedText = cachedSeedText ?? widgetSeedText ?? "";

    if (cachedSeedText == null && widgetSeedText != null) {
        node.__iisSeedGeneratorValue = widgetSeedText;
    }

    if (widget.value !== seedText) {
        widget.value = seedText;
        widget.callback?.(seedText);
    }
}

function applySeedWidgetReadOnly(node) {
    const widget = getWidget(node, SEED_WIDGET_NAME);
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

function setupSeedDisplayField(node) {
    removeSeedInputSocketIfPresent(node);
    applySeedWidgetReadOnly(node);
    applySeedWidgetValue(node);
}

function extractSeedTextDeep(value, depth = 0, seen = new Set()) {
    if (depth > 5 || value == null) {
        return null;
    }

    const valueType = typeof value;
    if (valueType === "string" || valueType === "number" || valueType === "bigint") {
        return normalizeSeedTextOrNone(value);
    }

    if (Array.isArray(value)) {
        for (const item of value) {
            const seed = extractSeedTextDeep(item, depth + 1, seen);
            if (seed != null) {
                return seed;
            }
        }
        return null;
    }

    if (valueType !== "object") {
        return null;
    }

    if (seen.has(value)) {
        return null;
    }
    seen.add(value);

    const preferredKeys = [
        "seed",
        "SEED",
        "result",
        "output",
        "outputs",
        "ui",
        "generated_seed",
        "value",
        "0",
    ];

    for (const key of preferredKeys) {
        if (!Object.prototype.hasOwnProperty.call(value, key)) {
            continue;
        }
        const seed = extractSeedTextDeep(value[key], depth + 1, seen);
        if (seed != null) {
            return seed;
        }
    }

    for (const [key, nestedValue] of Object.entries(value)) {
        if (preferredKeys.includes(key)) {
            continue;
        }
        const seed = extractSeedTextDeep(nestedValue, depth + 1, seen);
        if (seed != null) {
            return seed;
        }
    }

    return null;
}

function extractSeedTextFromOutput(output) {
    if (Array.isArray(output) && output.length > 0) {
        return normalizeSeedTextOrNone(output[0]);
    }

    if (!output || typeof output !== "object") {
        return null;
    }

    const uiSeed = output?.ui?.generated_seed;
    if (Array.isArray(uiSeed) && uiSeed.length > 0) {
        return normalizeSeedTextOrNone(uiSeed[0]);
    }
    if (uiSeed != null) {
        return normalizeSeedTextOrNone(uiSeed);
    }

    if (Array.isArray(output.result) && output.result.length > 0) {
        return normalizeSeedTextOrNone(output.result[0]);
    }

    return extractSeedTextDeep(output);
}

function updateCachedSeedFromOutput(node, output) {
    const seedText = extractSeedTextFromOutput(output);
    if (seedText) {
        node.__iisSeedGeneratorValue = seedText;
    }
}

function applySeedLabel(node) {
    const output = node?.outputs?.[SEED_OUTPUT_INDEX];
    if (!output) {
        return;
    }

    const label = DEFAULT_OUTPUT_LABEL;
    let changed = false;

    if (output.label !== label) {
        output.label = label;
        changed = true;
    }
    if (output.name !== label) {
        output.name = label;
        changed = true;
    }
    if (output.localized_name !== label) {
        output.localized_name = label;
        changed = true;
    }

    if (!changed) {
        setupSeedDisplayField(node);
        return;
    }

    node.setDirtyCanvas?.(true, true);
    requestVueNodeRefresh(node);
    setupSeedDisplayField(node);
}

function chainCallback(original, callback) {
    return function chained(...args) {
        const result = original?.apply(this, args);
        callback?.apply(this, args);
        return result;
    };
}

app.registerExtension({
    name: "IPT.SeedGeneratorDisplay",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isTargetNodeDef(nodeData)) {
            return;
        }

        nodeType.prototype.onNodeCreated = chainCallback(
            nodeType.prototype.onNodeCreated,
            function onNodeCreated() {
                setupSeedDisplayField(this);
                applySeedLabel(this);
            },
        );

        nodeType.prototype.onConfigure = chainCallback(
            nodeType.prototype.onConfigure,
            function onConfigure() {
                setupSeedDisplayField(this);
                applySeedLabel(this);
            },
        );

        nodeType.prototype.onGraphConfigured = chainCallback(
            nodeType.prototype.onGraphConfigured,
            function onGraphConfigured() {
                setupSeedDisplayField(this);
                applySeedLabel(this);
            },
        );

        nodeType.prototype.onExecuted = chainCallback(
            nodeType.prototype.onExecuted,
            function onExecuted(output) {
                updateCachedSeedFromOutput(this, output);
                setupSeedDisplayField(this);
                applySeedLabel(this);
            },
        );
    },
    nodeCreated(node) {
        if (isTargetNode(node)) {
            setupSeedDisplayField(node);
            applySeedLabel(node);
        }
    },
    loadedGraphNode(node) {
        if (isTargetNode(node)) {
            setupSeedDisplayField(node);
            applySeedLabel(node);
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
                updateCachedSeedFromOutput(node, output);
            }
            setupSeedDisplayField(node);
            applySeedLabel(node);
        }
    },
});
