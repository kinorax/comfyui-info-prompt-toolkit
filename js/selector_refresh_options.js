import { app } from "../../scripts/app.js";
import {
    getSelectorSlots,
    getWidget,
    isSelectorTargetNodeDef,
    resolveSelectorTarget,
} from "./selector_targets.js";

const MENU_LABEL = "Refresh Selector Options";

function copyValue(value) {
    if (value == null || typeof value !== "object") {
        return value;
    }

    if (typeof structuredClone === "function") {
        try {
            return structuredClone(value);
        } catch {
            // Fall through to shallow copy.
        }
    }

    if (Array.isArray(value)) {
        return [...value];
    }

    return { ...value };
}

async function refreshSelectorOptions(node) {
    const target = resolveSelectorTarget(node);
    if (!target || typeof app.refreshComboInNodes !== "function") {
        return;
    }

    const preservedValues = new Map();
    for (const slot of getSelectorSlots(target)) {
        const widget = getWidget(node, slot.widgetName);
        if (!widget) {
            continue;
        }
        preservedValues.set(slot.widgetName, copyValue(widget.value));
    }

    try {
        await app.refreshComboInNodes();
    } catch (error) {
        console.warn("[IPT.SelectorRefreshOptions] refresh failed", error);
        return;
    }

    let changed = false;
    for (const slot of getSelectorSlots(target)) {
        const refreshedWidget = getWidget(node, slot.widgetName);
        const preservedValue = preservedValues.get(slot.widgetName);
        if (!refreshedWidget || preservedValue === undefined) {
            continue;
        }

        // Keep the previously selected raw value even if it is no longer present in
        // the refreshed options. Execution-side validation will still resolve it to
        // None when the backend no longer accepts it.
        refreshedWidget.value = preservedValue;
        changed = true;
    }

    if (changed) {
        node.setDirtyCanvas?.(true, true);
    }
}

function addMenuOption(node, options) {
    if (!resolveSelectorTarget(node)) {
        return;
    }

    options.unshift({
        content: MENU_LABEL,
        callback: () => {
            void refreshSelectorOptions(node);
        },
    });
}

app.registerExtension({
    name: "IPT.SelectorRefreshOptions",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isSelectorTargetNodeDef(nodeData)) {
            return;
        }
        if (nodeType.prototype.__iptSelectorRefreshOptionsMenuPatched) {
            return;
        }

        const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function patchedGetExtraMenuOptions(_, options) {
            if (Array.isArray(options)) {
                addMenuOption(this, options);
            }
            return originalGetExtraMenuOptions?.apply(this, arguments);
        };

        nodeType.prototype.__iptSelectorRefreshOptionsMenuPatched = true;
    },
});
