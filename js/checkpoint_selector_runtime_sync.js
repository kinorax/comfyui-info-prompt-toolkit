// Copyright 2026 kinorax
import { app } from "../../scripts/app.js";
import {
    getSelectorSlots,
    getWidget,
    normalizeWidgetValue,
    resolveSelectorTarget,
} from "./selector_targets.js";

function extractSelectorValue(output, slot) {
    if (Array.isArray(output) && output.length > slot.resultIndex) {
        return normalizeWidgetValue(output[slot.resultIndex]);
    }

    if (!output || typeof output !== "object") {
        return undefined;
    }

    const directValue = output?.[slot.widgetName];
    if (Array.isArray(directValue)) {
        return normalizeWidgetValue(directValue[0]);
    }
    if (Object.prototype.hasOwnProperty.call(output, slot.widgetName)) {
        return normalizeWidgetValue(directValue);
    }

    const uiValue = output?.ui?.[slot.widgetName];
    if (Array.isArray(uiValue)) {
        return normalizeWidgetValue(uiValue[0]);
    }
    if (Object.prototype.hasOwnProperty.call(output?.ui ?? {}, slot.widgetName)) {
        return normalizeWidgetValue(uiValue);
    }

    if (Array.isArray(output.result) && output.result.length > slot.resultIndex) {
        return normalizeWidgetValue(output.result[slot.resultIndex]);
    }

    return undefined;
}

function applyWidgetValue(node, slot, value) {
    const widget = getWidget(node, slot.widgetName);
    if (!widget) {
        return;
    }

    if (widget.value === value) {
        return;
    }

    const optionValues = Array.isArray(widget.options?.values) ? widget.options.values : null;
    if (optionValues && !optionValues.includes(value)) {
        widget.options.values = [...optionValues, value];
    }

    widget.value = value;
    widget.callback?.(value);
    node.setDirtyCanvas?.(true, true);
}

async function syncSelectorWidgets(node, slotUpdates) {
    const pendingUpdates = [];

    for (const [slot, rawValue] of slotUpdates) {
        const nextValue = normalizeWidgetValue(rawValue);
        const widget = getWidget(node, slot.widgetName);
        if (!widget) {
            continue;
        }
        if (normalizeWidgetValue(widget.value) === nextValue) {
            continue;
        }

        const pendingKey = `__iisSelectorSyncPending_${slot.widgetName}`;
        if (node[pendingKey] === nextValue) {
            continue;
        }

        node[pendingKey] = nextValue;
        pendingUpdates.push([slot, nextValue, pendingKey]);
    }

    if (!pendingUpdates.length) {
        return;
    }

    try {
        if (typeof app.refreshComboInNodes === "function") {
            await app.refreshComboInNodes();
        }
    } catch {
    } finally {
        for (const [slot, nextValue, pendingKey] of pendingUpdates) {
            applyWidgetValue(node, slot, nextValue);
            node[pendingKey] = null;
        }
    }
}

function chainCallback(original, callback) {
    return function chained(...args) {
        const result = original?.apply(this, args);
        callback?.apply(this, args);
        return result;
    };
}

function queueSyncFromOutput(node, output) {
    const target = resolveSelectorTarget(node);
    if (!target) {
        return;
    }

    const slotUpdates = [];
    for (const slot of getSelectorSlots(target)) {
        const value = extractSelectorValue(output, slot);
        if (value === undefined) {
            continue;
        }
        slotUpdates.push([slot, value]);
    }

    if (!slotUpdates.length) {
        return;
    }

    void syncSelectorWidgets(node, slotUpdates);
}

app.registerExtension({
    name: "IPT.SelectorRuntimeSync",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!resolveSelectorTarget(nodeData)) {
            return;
        }
        if (nodeType.prototype.__iisSelectorRuntimeSyncPatched) {
            return;
        }

        nodeType.prototype.onExecuted = chainCallback(
            nodeType.prototype.onExecuted,
            function onExecuted(output) {
                queueSyncFromOutput(this, output);
            },
        );

        nodeType.prototype.__iisSelectorRuntimeSyncPatched = true;
    },
    onNodeOutputsUpdated(nodeOutputs) {
        if (!app?.rootGraph || !nodeOutputs || typeof nodeOutputs !== "object") {
            return;
        }

        for (const node of app.rootGraph._nodes || []) {
            if (!resolveSelectorTarget(node)) {
                continue;
            }

            const output = nodeOutputs[String(node.id)];
            if (!output) {
                continue;
            }

            queueSyncFromOutput(node, output);
        }
    },
});
