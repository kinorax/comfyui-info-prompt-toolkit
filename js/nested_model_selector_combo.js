// Copyright 2026 kinorax
import { app } from "../../scripts/app.js";
import {
    getNestedTargetWidgetNames,
    resolveNestedComboTarget,
} from "./selector_targets.js";
import {
    applyNestedComboTree,
    installNestedComboStyle,
} from "./nested_combo_menu.js";

function getWidgetOptionValues(widget) {
    const optionValues = widget?.options?.values;
    if (!Array.isArray(optionValues) || optionValues.length === 0) {
        return null;
    }

    const normalized = optionValues
        .filter((value) => typeof value === "string" && value)
        .map((value) => String(value));
    return normalized.length > 0 ? new Set(normalized) : null;
}

function installObserver() {
    const observer = new MutationObserver((mutations) => {
        const node = app.canvas?.current_node;
        const target = resolveNestedComboTarget(node);
        if (!target) {
            return;
        }

        const overWidget = app.canvas?.getWidgetAtCursor?.();
        const targetWidgetNames = new Set(getNestedTargetWidgetNames(target));
        if (!overWidget || !targetWidgetNames.has(overWidget.name)) {
            return;
        }
        const optionValues = getWidgetOptionValues(overWidget);
        if (optionValues === null) {
            return;
        }

        for (const mutation of mutations) {
            for (const addedNode of mutation.addedNodes) {
                if (!addedNode?.classList?.contains("litecontextmenu")) {
                    continue;
                }

                requestAnimationFrame(() => {
                    applyNestedComboTree(addedNode, [...optionValues]);
                });
            }
        }
    });

    observer.observe(document.body, { childList: true, subtree: false });
}

app.registerExtension({
    name: "IPT.NestedModelSelectorCombo",
    init() {
        installNestedComboStyle();
        installObserver();
    },
});
