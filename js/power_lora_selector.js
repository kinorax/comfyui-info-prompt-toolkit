// Copyright 2026 kinorax
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { clearStaleClassicWidgetWidth } from "./classic_widget_width_compat.js";
import {
    applyNestedComboTree,
    installNestedComboStyle,
} from "./nested_combo_menu.js";

const TARGET_NODE_TYPES = new Set([
    "IPT-PowerLoraSelector",
    "PowerLoraSelector",
    "Power Lora Selector",
]);
const BACKING_WIDGET_NAME = "items";
const EDITOR_WIDGET_NAME = "IPT-power-lora-selector-editor";
const EDITOR_WIDGET_TYPE = "IPT-PowerLoraSelectorEditor";
const STYLE_ID = "ipt-power-lora-selector-style";
const STATE_KEY = "__iptPowerLoraSelectorEditorState";
const ROW_HEIGHT = 34;
const ADD_BUTTON_HEIGHT = 28;
const VERTICAL_PADDING = 8;
const BOTTOM_CLEARANCE = 10;
const MINIMUM_NODE_WIDTH = 500;
const MIN_STRENGTH = -100;
const MAX_STRENGTH = 100;

let nextRowId = 1;

function chainCallback(original, callback) {
    return function chained(...args) {
        const result = original?.apply(this, args);
        callback?.apply(this, args);
        return result;
    };
}

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

function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) ?? null;
}

function normalizeOptions(value) {
    if (!Array.isArray(value)) {
        return [];
    }
    return [...new Set(value.map((item) => String(item ?? "").trim()).filter(Boolean))];
}

function getLoraOptionsFromNodeData(nodeData) {
    for (const sectionName of ["required", "optional"]) {
        const entry = nodeData?.input?.[sectionName]?.[BACKING_WIDGET_NAME];
        const options = normalizeOptions(entry?.[1]?.lora_options);
        if (options.length) {
            return options;
        }
    }
    return [];
}

function parseBoolean(value, fallback = true) {
    if (typeof value === "boolean") {
        return value;
    }
    if (typeof value === "number") {
        return value !== 0;
    }
    if (value == null) {
        return fallback;
    }
    return !["", "0", "false", "no", "off"].includes(String(value).trim().toLowerCase());
}

function normalizeSha256(value) {
    const text = String(value ?? "").trim().toLowerCase();
    return /^[0-9a-f]{64}$/.test(text) ? text : "";
}

function normalizeStrength(value) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
        return 1;
    }
    return Math.max(MIN_STRENGTH, Math.min(MAX_STRENGTH, parsed));
}

function makeRow(loraName, strength = 1, enabled = true, sha256 = "") {
    return {
        id: nextRowId++,
        loraName: String(loraName ?? "").trim(),
        strength: normalizeStrength(strength),
        enabled: Boolean(enabled),
        sha256: normalizeSha256(sha256),
        requestId: 0,
    };
}

function parseRows(value) {
    let rawRows = value;
    if (typeof value === "string") {
        const text = value.trim();
        if (!text) {
            return [];
        }
        try {
            rawRows = JSON.parse(text);
        } catch {
            return [];
        }
    }
    if (rawRows && typeof rawRows === "object" && !Array.isArray(rawRows)) {
        rawRows = rawRows.items;
    }
    if (!Array.isArray(rawRows)) {
        return [];
    }

    return rawRows
        .filter((rawRow) => rawRow && typeof rawRow === "object" && !Array.isArray(rawRow))
        .map((rawRow) => makeRow(
            rawRow.lora_name ?? rawRow.name,
            rawRow.strength,
            parseBoolean(rawRow.enabled, true),
            rawRow.sha256,
        ))
        .filter((row) => row.loraName);
}

function serializeRows(rows) {
    return JSON.stringify(rows.map((row) => ({
        enabled: Boolean(row.enabled),
        lora_name: row.loraName,
        strength: normalizeStrength(row.strength),
        sha256: normalizeSha256(row.sha256),
    })));
}

function markCurrentLoraMenuItem(menu, currentValue) {
    const menuElement = menu?.root instanceof HTMLElement ? menu.root : null;
    if (!menuElement || !currentValue) {
        return;
    }

    const clearMarkedItem = () => {
        const markedItem = menuElement.querySelector('[data-ipt-power-lora-current="1"]');
        if (!markedItem) {
            return;
        }
        markedItem.style.removeProperty("background-color");
        markedItem.style.removeProperty("color");
        markedItem.removeAttribute("aria-current");
        markedItem.removeAttribute("data-ipt-power-lora-current");
    };

    // ComfyUI's contextMenuFilter can take over selection while filtering or
    // using the keyboard. Remove our initial marker before that happens.
    const filter = menuElement.querySelector(".comfy-context-menu-filter");
    filter?.addEventListener("input", clearMarkedItem, { capture: true });
    filter?.addEventListener("keydown", (event) => {
        if (["ArrowUp", "ArrowRight", "ArrowDown", "ArrowLeft"].includes(event.key)) {
            clearMarkedItem();
        }
    }, { capture: true });

    // contextMenuFilter applies its initial highlight on the first animation
    // frame. Run after it, then replace that fallback with this row's value.
    requestAnimationFrame(() => {
        if (!menuElement.isConnected) {
            return;
        }

        const itemElements = [...menuElement.querySelectorAll(".litemenu-entry[data-value]")];
        const selectedItem = itemElements.find((itemElement) => (
            itemElement.getAttribute("data-value") === currentValue
        ));
        if (!selectedItem) {
            return;
        }

        for (const itemElement of itemElements) {
            itemElement.style.removeProperty("background-color");
            itemElement.style.removeProperty("color");
            itemElement.removeAttribute("aria-current");
            itemElement.removeAttribute("data-ipt-power-lora-current");
        }

        selectedItem.style.setProperty("background-color", "#ccc", "important");
        selectedItem.style.setProperty("color", "#000", "important");
        selectedItem.setAttribute("aria-current", "true");
        selectedItem.setAttribute("data-ipt-power-lora-current", "1");
    });
}

function openLoraMenu(event, row, optionValues, onSelect) {
    const ContextMenu = globalThis.LiteGraph?.ContextMenu;
    if (typeof ContextMenu !== "function") {
        alert("LoRA selector menu is unavailable.");
        return;
    }

    const menu = new ContextMenu(optionValues, {
        scale: Math.max(1, Number(app.canvas?.ds?.scale) || 1),
        event,
        className: "dark",
        callback: (value) => {
            const selected = String(value ?? "").trim();
            if (selected && selected !== row.loraName) {
                onSelect(selected);
            }
        },
    });
    applyNestedComboTree(menu, optionValues);
    markCurrentLoraMenuItem(menu, row.loraName);
}

function installStyleOnce() {
    installNestedComboStyle();
    if (document.getElementById(STYLE_ID)) {
        return;
    }

    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .ipt-power-lora-selector {
            width: 100%;
            min-width: 0;
            box-sizing: border-box;
            padding: 4px 0;
            overflow: visible;
            color: var(--component-node-foreground, #ddd);
            font: 12px sans-serif;
        }
        .ipt-power-lora-selector__rows {
            display: flex;
            flex-direction: column;
            gap: 4px;
            overflow: visible;
        }
        .ipt-power-lora-selector__row {
            min-width: 0;
            height: ${ROW_HEIGHT - 4}px;
            display: grid;
            grid-template-columns: 34px minmax(140px, 1fr) 72px 28px 24px;
            align-items: center;
            gap: 5px;
            box-sizing: border-box;
        }
        .ipt-power-lora-selector__combo,
        .ipt-power-lora-selector__strength {
            width: 100%;
            min-width: 0;
            height: 26px;
            box-sizing: border-box;
            color: var(--component-node-foreground, #ddd);
            background: var(--component-node-widget-background, #222);
            border: 1px solid var(--component-node-border, #666);
            border-radius: 6px;
            outline: none;
            font: inherit;
        }
        .ipt-power-lora-selector__combo {
            position: relative;
            padding: 2px 24px 2px 8px;
            overflow: hidden;
            text-align: left;
            text-overflow: ellipsis;
            white-space: nowrap;
            cursor: pointer;
        }
        .ipt-power-lora-selector__combo::after {
            content: "▾";
            position: absolute;
            top: 50%;
            right: 8px;
            transform: translateY(-50%);
            color: var(--component-node-foreground-secondary, #aaa);
            pointer-events: none;
        }
        .ipt-power-lora-selector__strength {
            padding: 2px 5px;
            text-align: right;
        }
        .ipt-power-lora-selector__combo:focus,
        .ipt-power-lora-selector__strength:focus {
            border-color: var(--p-primary-color, #6b9cff);
        }
        .ipt-power-lora-selector__info,
        .ipt-power-lora-selector__delete,
        .ipt-power-lora-selector__add {
            color: var(--component-node-foreground, #ddd);
            background: var(--component-node-widget-background, #2b2b2b);
            border: 1px solid var(--component-node-border, #666);
            border-radius: 6px;
            cursor: pointer;
            font: inherit;
        }
        .ipt-power-lora-selector__switch {
            position: relative;
            width: 32px;
            height: 18px;
            padding: 0;
            border: 1px solid var(--component-node-border, #666);
            border-radius: 999px;
            background: var(--component-node-widget-background, #333);
            cursor: pointer;
        }
        .ipt-power-lora-selector__switch::after {
            content: "";
            position: absolute;
            top: 2px;
            left: 2px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--component-node-foreground-secondary, #aaa);
            transition: transform 100ms ease, background-color 100ms ease;
        }
        .ipt-power-lora-selector__switch[aria-checked="true"] {
            background: var(--p-primary-color, #4778d7);
            border-color: var(--p-primary-color, #4778d7);
        }
        .ipt-power-lora-selector__switch[aria-checked="true"]::after {
            transform: translateX(14px);
            background: #fff;
        }
        .ipt-power-lora-selector__info,
        .ipt-power-lora-selector__delete {
            height: 24px;
            padding: 0;
            background: transparent;
        }
        .ipt-power-lora-selector__info {
            width: 28px;
            font-size: 15px;
        }
        .ipt-power-lora-selector__delete {
            width: 24px;
            border-color: transparent;
            color: var(--component-node-foreground-secondary, #aaa);
            font-size: 16px;
        }
        .ipt-power-lora-selector__add {
            width: 100%;
            height: ${ADD_BUTTON_HEIGHT}px;
            margin-top: 6px;
        }
        .ipt-power-lora-selector__info:hover,
        .ipt-power-lora-selector__delete:hover,
        .ipt-power-lora-selector__add:hover {
            filter: brightness(1.15);
        }
        .ipt-power-lora-selector__row.is-off .ipt-power-lora-selector__combo,
        .ipt-power-lora-selector__row.is-off .ipt-power-lora-selector__strength {
            opacity: 0.6;
        }
        .ipt-power-lora-selector button:focus-visible,
        .ipt-power-lora-selector input:focus-visible {
            outline: 2px solid var(--p-primary-color, #6b9cff);
            outline-offset: 1px;
        }
        .ipt-power-lora-selector button:disabled,
        .ipt-power-lora-selector input:disabled {
            cursor: default;
            opacity: 0.5;
        }
        .ipt-power-lora-selector.is-disabled {
            opacity: 0.65;
        }
    `;
    document.head.appendChild(style);
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
    node?.setDirtyCanvas?.(true, true);
    if (Array.isArray(node?.size) && typeof node?.setSize === "function") {
        let minimumHeight = node.size[1];
        try {
            minimumHeight = Number(node.computeSize?.()?.[1]) || node.size[1];
        } catch {
            minimumHeight = node.size[1];
        }
        node.setSize([
            Math.max(node.size[0], MINIMUM_NODE_WIDTH),
            Math.max(node.size[1], minimumHeight),
        ]);
    }
    requestVueNodeRefresh(node);
}

function refreshWidgetList(node) {
    if (Array.isArray(node?.widgets)) {
        try {
            node.widgets = [...node.widgets];
        } catch {
            // Legacy frontends may expose widgets as a plain array only.
        }
    }
    syncNodeLayout(node);
}

function hideBackingWidget(widget) {
    widget.options = widget.options ?? {};
    widget.hidden = true;
    widget.options.hidden = true;
}

async function refreshRowSha256(row, rows, commit) {
    const relativePath = row.loraName;
    row.requestId += 1;
    const requestId = row.requestId;
    row.sha256 = "";
    commit();

    try {
        await api.fetchApi("/ipt/model-metadata/queue-priority", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ folder_name: "loras", relative_path: relativePath }),
        });
        const response = await api.fetchApi("/ipt/model-reference/resolve", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
                folder_name: "loras",
                relative_path: relativePath,
                name_raw: relativePath,
                enqueue_local_hash: true,
                resolve_remote: false,
                include_lora_tags: false,
            }),
        });
        if (!response.ok) {
            return;
        }
        const payload = await response.json();
        if (!rows.includes(row) || row.requestId !== requestId || row.loraName !== relativePath) {
            return;
        }
        const sha256 = normalizeSha256(payload?.sha256);
        if (sha256) {
            row.sha256 = sha256;
            commit();
        }
    } catch (error) {
        console.warn("[IPT] Power Lora Selector failed to resolve LoRA SHA256.", error);
    }
}

function openModelInfo(row) {
    const bridge = globalThis.window?.__iisModelInfoWindow;
    if (!bridge || typeof bridge.openModelInfoWindow !== "function") {
        alert("View Model Info is unavailable.");
        return;
    }
    const target = bridge.resolveTargetByFolderName?.("loras") ?? { folderName: "loras" };
    bridge.openModelInfoWindow({
        node: null,
        target,
        source: {
            relativePath: row.loraName,
            sha256: normalizeSha256(row.sha256),
            nameRaw: row.loraName,
            hashHints: [],
        },
    });
}

function createEditor(node, backingWidget, initialOptions) {
    installStyleOnce();

    const host = document.createElement("div");
    host.className = "ipt-power-lora-selector";
    const rowsElement = document.createElement("div");
    rowsElement.className = "ipt-power-lora-selector__rows";
    const addButton = document.createElement("button");
    addButton.type = "button";
    addButton.className = "ipt-power-lora-selector__add";
    addButton.textContent = "Add Lora";
    host.append(rowsElement, addButton);

    let rows = parseRows(backingWidget.value);
    let loraOptions = normalizeOptions(initialOptions);
    let editorWidget = null;
    let editorHeight = 0;
    let disabled = false;

    const updateHeight = () => {
        editorHeight = VERTICAL_PADDING
            + (rows.length * ROW_HEIGHT)
            + ADD_BUTTON_HEIGHT
            + BOTTOM_CLEARANCE;
        const cssHeight = `${editorHeight}px`;
        host.style.height = cssHeight;
        host.style.setProperty("--comfy-widget-min-height", cssHeight);
        host.style.setProperty("--comfy-widget-height", cssHeight);
        host.style.setProperty("--comfy-widget-max-height", cssHeight);
    };

    const commit = () => {
        const value = serializeRows(rows);
        if (backingWidget.value !== value) {
            backingWidget.value = value;
            backingWidget.callback?.(value);
        }
        node.setDirtyCanvas?.(true, true);
    };

    const render = (focusRowId = null) => {
        rowsElement.replaceChildren();

        for (const row of rows) {
            const rowElement = document.createElement("div");
            rowElement.className = "ipt-power-lora-selector__row";
            rowElement.classList.toggle("is-off", !row.enabled);

            const toggle = document.createElement("button");
            toggle.type = "button";
            toggle.className = "ipt-power-lora-selector__switch";
            toggle.setAttribute("role", "switch");
            toggle.setAttribute("aria-checked", String(row.enabled));
            toggle.setAttribute("aria-label", `Enable ${row.loraName}`);
            toggle.title = row.enabled ? "ON" : "OFF";
            toggle.disabled = disabled;
            toggle.addEventListener("click", () => {
                row.enabled = !row.enabled;
                toggle.setAttribute("aria-checked", String(row.enabled));
                toggle.title = row.enabled ? "ON" : "OFF";
                rowElement.classList.toggle("is-off", !row.enabled);
                commit();
            });

            const options = loraOptions.includes(row.loraName)
                ? loraOptions
                : [row.loraName, ...loraOptions].filter(Boolean);
            const combo = document.createElement("button");
            combo.type = "button";
            combo.className = "ipt-power-lora-selector__combo";
            combo.textContent = row.loraName;
            combo.title = row.loraName;
            combo.dataset.rowId = String(row.id);
            combo.setAttribute("aria-label", `lora_name: ${row.loraName}`);
            combo.disabled = disabled;
            combo.addEventListener("click", (event) => {
                openLoraMenu(event, row, options, (selected) => {
                    row.loraName = selected;
                    commit();
                    render();
                    void refreshRowSha256(row, rows, commit);
                });
            });

            const strength = document.createElement("input");
            strength.className = "ipt-power-lora-selector__strength";
            strength.type = "number";
            strength.min = String(MIN_STRENGTH);
            strength.max = String(MAX_STRENGTH);
            strength.step = "0.01";
            strength.value = String(row.strength);
            strength.setAttribute("aria-label", `Strength for ${row.loraName}`);
            strength.disabled = disabled;
            const commitStrength = () => {
                row.strength = normalizeStrength(strength.value);
                strength.value = String(row.strength);
                commit();
            };
            strength.addEventListener("change", commitStrength);
            strength.addEventListener("keydown", (event) => {
                if (event.key === "Enter") {
                    commitStrength();
                }
            });

            const info = document.createElement("button");
            info.type = "button";
            info.className = "ipt-power-lora-selector__info";
            info.textContent = "ⓘ";
            info.title = "View Model Info...";
            info.setAttribute("aria-label", `View Model Info for ${row.loraName}`);
            info.disabled = disabled;
            info.addEventListener("click", () => openModelInfo(row));

            const remove = document.createElement("button");
            remove.type = "button";
            remove.className = "ipt-power-lora-selector__delete";
            remove.textContent = "×";
            remove.title = "Remove LoRA row";
            remove.setAttribute("aria-label", `Remove ${row.loraName}`);
            remove.disabled = disabled;
            remove.addEventListener("click", () => {
                rows = rows.filter((candidate) => candidate !== row);
                commit();
                updateHeight();
                render();
                syncNodeLayout(node);
            });

            rowElement.append(toggle, combo, strength, info, remove);
            rowsElement.appendChild(rowElement);
        }

        addButton.disabled = disabled || loraOptions.length === 0;
        addButton.title = loraOptions.length ? "Add Lora" : "No LoRA models are available";

        if (focusRowId != null) {
            queueMicrotask(() => {
                const index = rows.findIndex((row) => row.id === focusRowId);
                rowsElement.children[index]?.querySelector(".ipt-power-lora-selector__combo")?.focus();
            });
        }
    };

    const setDisabled = (nextDisabled) => {
        const normalized = Boolean(nextDisabled);
        if (disabled === normalized) {
            return;
        }
        disabled = normalized;
        host.classList.toggle("is-disabled", disabled);
        render();
    };

    const setValue = (value) => {
        const serializedCurrent = serializeRows(rows);
        const parsedRows = parseRows(value);
        const serializedNext = serializeRows(parsedRows);
        if (serializedCurrent === serializedNext) {
            return;
        }
        rows = parsedRows;
        updateHeight();
        render();
        syncNodeLayout(node);
    };

    const setOptions = (value) => {
        const normalized = normalizeOptions(value);
        if (!normalized.length || JSON.stringify(normalized) === JSON.stringify(loraOptions)) {
            return;
        }
        loraOptions = normalized;
        render();
    };

    addButton.addEventListener("click", () => {
        if (!loraOptions.length) {
            return;
        }
        const row = makeRow(loraOptions[0]);
        rows.push(row);
        commit();
        updateHeight();
        render(row.id);
        syncNodeLayout(node);
        void refreshRowSha256(row, rows, commit);
    });

    updateHeight();
    render();

    editorWidget = node.addDOMWidget(
        EDITOR_WIDGET_NAME,
        EDITOR_WIDGET_TYPE,
        host,
        {
            serialize: false,
            canvasOnly: false,
            getMinHeight: () => editorHeight,
            getHeight: () => editorHeight,
            getMaxHeight: () => editorHeight,
            onDraw: (widget) => {
                clearStaleClassicWidgetWidth(widget);
                setDisabled(Boolean(widget.computedDisabled));
            },
            getValue: () => serializeRows(rows),
            setValue,
        },
    );
    editorWidget.serialize = false;
    editorWidget.options = editorWidget.options ?? {};
    editorWidget.options.serialize = false;
    editorWidget.options.canvasOnly = false;

    for (const row of rows) {
        if (!row.sha256) {
            void refreshRowSha256(row, rows, commit);
        }
    }

    return { widget: editorWidget, setValue, setOptions };
}

function installEditor(node, loraOptions = []) {
    if (!isTargetNode(node) || typeof node?.addDOMWidget !== "function") {
        return;
    }
    const backingWidget = getWidget(node, BACKING_WIDGET_NAME);
    if (!backingWidget) {
        return;
    }

    const widgetOptions = normalizeOptions(backingWidget.options?.lora_options);
    const resolvedOptions = normalizeOptions(loraOptions).length ? loraOptions : widgetOptions;
    let state = node[STATE_KEY];
    if (!state) {
        try {
            state = createEditor(node, backingWidget, resolvedOptions);
        } catch (error) {
            console.error("[IPT] Failed to create Power Lora Selector editor.", error);
            return;
        }
        node[STATE_KEY] = state;
        hideBackingWidget(backingWidget);
        refreshWidgetList(node);
        return;
    }

    hideBackingWidget(backingWidget);
    state.setOptions(resolvedOptions);
    state.setValue(backingWidget.value);
    syncNodeLayout(node);
}

app.registerExtension({
    name: "IPT.PowerLoraSelectorEditor",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isTargetNodeDef(nodeData)) {
            return;
        }
        const loraOptions = getLoraOptionsFromNodeData(nodeData);
        nodeType.prototype.onNodeCreated = chainCallback(
            nodeType.prototype.onNodeCreated,
            function onNodeCreated() {
                installEditor(this, loraOptions);
            },
        );
        nodeType.prototype.onConfigure = chainCallback(
            nodeType.prototype.onConfigure,
            function onConfigure() {
                installEditor(this, loraOptions);
            },
        );
        nodeType.prototype.onGraphConfigured = chainCallback(
            nodeType.prototype.onGraphConfigured,
            function onGraphConfigured() {
                installEditor(this, loraOptions);
            },
        );
    },
    nodeCreated(node) {
        installEditor(node);
    },
    loadedGraphNode(node) {
        installEditor(node);
    },
});
