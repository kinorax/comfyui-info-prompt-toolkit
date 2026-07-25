// Copyright 2026 kinorax
import { app } from "../../scripts/app.js";
import { clearStaleClassicWidgetWidth } from "./classic_widget_width_compat.js";

const TARGET_NODE_TYPES = new Set([
    "IPT-ImageInfoEncryptionTargets",
    "ImageInfoEncryptionTargets",
    "Image Info Encryption Targets",
]);
const BACKING_WIDGET_NAME = "extra_keys";
const EDITOR_WIDGET_NAME = "IPT-image-info-encryption-extra-keys-editor";
const EDITOR_WIDGET_TYPE = "IPT-ImageInfoEncryptionExtraKeysEditor";
const STYLE_ID = "ipt-image-info-encryption-targets-style";
const STATE_KEY = "__iptImageInfoEncryptionTargetsEditorState";
const ROW_HEIGHT = 30;
const HEADER_HEIGHT = 22;
const VERTICAL_PADDING = 8;
const BOTTOM_CLEARANCE = 12;

let nextRowId = 1;

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
    const text = String(value).trim().toLowerCase();
    if (["1", "true", "yes", "on"].includes(text)) {
        return true;
    }
    if (["0", "false", "no", "off"].includes(text)) {
        return false;
    }
    return fallback;
}

function makeRow(key = "", enabled = true) {
    return {
        id: nextRowId++,
        key: String(key ?? ""),
        enabled: Boolean(enabled),
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
            if (text.startsWith("[") || text.startsWith("{")) {
                return [];
            }
            rawRows = text.split(",");
        }
    }

    if (rawRows && typeof rawRows === "object" && !Array.isArray(rawRows)) {
        rawRows = rawRows.items ?? rawRows.extra_keys;
    }
    if (!Array.isArray(rawRows)) {
        return [];
    }

    const rows = [];
    for (const rawRow of rawRows) {
        let key = rawRow;
        let enabled = true;
        if (Array.isArray(rawRow)) {
            if (!rawRow.length) {
                continue;
            }
            [key] = rawRow;
            enabled = parseBoolean(rawRow[1], true);
        } else if (rawRow && typeof rawRow === "object") {
            key = rawRow.key ?? rawRow.name ?? "";
            enabled = parseBoolean(rawRow.enabled, true);
        }

        const normalizedKey = String(key ?? "").trim();
        if (normalizedKey) {
            rows.push(makeRow(normalizedKey, enabled));
        }
    }
    return rows;
}

function serializableRows(rows) {
    return rows
        .map((row) => ({
            key: String(row.key ?? "").trim(),
            enabled: Boolean(row.enabled),
        }))
        .filter((row) => row.key);
}

function serializeRows(rows) {
    return JSON.stringify(serializableRows(rows));
}

function ensureTrailingEmptyRow(rows) {
    const nonEmptyRows = rows.filter((row) => String(row.key ?? "").trim());
    nonEmptyRows.push(makeRow());
    return nonEmptyRows;
}

function installStyleOnce() {
    if (document.getElementById(STYLE_ID)) {
        return;
    }

    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .ipt-encryption-extra-key-list {
            width: 100%;
            min-width: 0;
            box-sizing: border-box;
            padding: 4px 0px;
            overflow: visible;
            color: var(--component-node-foreground, #ddd);
            font: 12px sans-serif;
        }
        .ipt-encryption-extra-key-list__header {
            height: ${HEADER_HEIGHT}px;
            display: flex;
            align-items: center;
            color: var(--component-node-foreground-secondary, #aaa);
            user-select: none;
        }
        .ipt-encryption-extra-key-list__rows {
            display: flex;
            flex-direction: column;
            gap: 4px;
            overflow: visible;
        }
        .ipt-encryption-extra-key-list__row {
            min-width: 0;
            height: ${ROW_HEIGHT - 4}px;
            display: grid;
            grid-template-columns: 34px minmax(0, 1fr) 24px;
            align-items: center;
            gap: 6px;
            box-sizing: border-box;
        }
        .ipt-encryption-extra-key-list__input {
            width: 100%;
            min-width: 0;
            height: 24px;
            box-sizing: border-box;
            padding: 2px 8px;
            color: var(--component-node-foreground, #ddd);
            background: var(--component-node-widget-background, #222);
            border: 1px solid var(--component-node-border, #666);
            border-radius: 6px;
            outline: none;
            font: inherit;
        }
        .ipt-encryption-extra-key-list__input:focus {
            border-color: var(--p-primary-color, #6b9cff);
        }
        .ipt-encryption-extra-key-list__input::placeholder {
            color: var(--component-node-foreground-secondary, #999);
        }
        .ipt-encryption-extra-key-list__switch {
            position: relative;
            width: 32px;
            height: 18px;
            padding: 0;
            border: 1px solid var(--component-node-border, #666);
            border-radius: 999px;
            background: var(--component-node-widget-background, #333);
            cursor: pointer;
        }
        .ipt-encryption-extra-key-list__switch::after {
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
        .ipt-encryption-extra-key-list__switch[aria-checked="true"] {
            background: var(--p-primary-color, #4778d7);
            border-color: var(--p-primary-color, #4778d7);
        }
        .ipt-encryption-extra-key-list__switch[aria-checked="true"]::after {
            transform: translateX(14px);
            background: #fff;
        }
        .ipt-encryption-extra-key-list__switch:focus-visible,
        .ipt-encryption-extra-key-list__delete:focus-visible {
            outline: 2px solid var(--p-primary-color, #6b9cff);
            outline-offset: 1px;
        }
        .ipt-encryption-extra-key-list__switch.is-placeholder,
        .ipt-encryption-extra-key-list__delete.is-placeholder {
            visibility: hidden;
            pointer-events: none;
        }
        .ipt-encryption-extra-key-list__delete {
            width: 24px;
            height: 24px;
            padding: 0;
            color: var(--component-node-foreground-secondary, #aaa);
            background: transparent;
            border: 0;
            border-radius: 5px;
            cursor: pointer;
            font: 16px/1 sans-serif;
        }
        .ipt-encryption-extra-key-list__delete:hover {
            color: var(--component-node-foreground, #fff);
            background: color-mix(in srgb, var(--component-node-widget-background, #333) 80%, white 20%);
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
            node.size[0],
            Math.max(node.size[1], minimumHeight),
        ]);
    }
    requestVueNodeRefresh(node);
}

function hideBackingWidget(widget) {
    if (!widget) {
        return;
    }
    widget.options = widget.options ?? {};
    widget.hidden = true;
    widget.options.hidden = true;
}

function createEditor(node, backingWidget) {
    installStyleOnce();

    const host = document.createElement("div");
    host.className = "ipt-encryption-extra-key-list";

    const header = document.createElement("div");
    header.className = "ipt-encryption-extra-key-list__header";
    header.textContent = "extra keys";

    const rowsElement = document.createElement("div");
    rowsElement.className = "ipt-encryption-extra-key-list__rows";
    host.append(header, rowsElement);

    let rows = ensureTrailingEmptyRow(parseRows(backingWidget.value));
    let editorHeight = 0;

    const updateHeight = () => {
        editorHeight = VERTICAL_PADDING
            + HEADER_HEIGHT
            + (rows.length * ROW_HEIGHT)
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

    const focusRow = (rowId) => {
        queueMicrotask(() => {
            const input = rowsElement.querySelector(`[data-row-id="${rowId}"]`);
            input?.focus();
            if (input && typeof input.setSelectionRange === "function") {
                const end = input.value.length;
                input.setSelectionRange(end, end);
            }
        });
    };

    const render = (focusRowId = null) => {
        rowsElement.replaceChildren();

        for (let index = 0; index < rows.length; index += 1) {
            const row = rows[index];
            const isPlaceholder = index === rows.length - 1 && !String(row.key ?? "").trim();
            const rowElement = document.createElement("div");
            rowElement.className = "ipt-encryption-extra-key-list__row";

            const toggle = document.createElement("button");
            toggle.type = "button";
            toggle.className = "ipt-encryption-extra-key-list__switch";
            toggle.setAttribute("role", "switch");
            toggle.setAttribute("aria-label", `Enable encryption for ${row.key || "key"}`);
            toggle.setAttribute("aria-checked", String(Boolean(row.enabled)));
            toggle.classList.toggle("is-placeholder", isPlaceholder);
            toggle.tabIndex = isPlaceholder ? -1 : 0;
            toggle.addEventListener("click", () => {
                row.enabled = !row.enabled;
                toggle.setAttribute("aria-checked", String(row.enabled));
                commit();
            });

            const input = document.createElement("input");
            input.className = "ipt-encryption-extra-key-list__input";
            input.type = "text";
            input.value = row.key;
            input.placeholder = isPlaceholder ? "extra key to encrypt..." : "";
            input.dataset.rowId = String(row.id);
            input.setAttribute("aria-label", "Extra key to encrypt");
            input.addEventListener("input", () => {
                const wasPlaceholder = row === rows[rows.length - 1];
                row.key = input.value;
                commit();
                if (wasPlaceholder && row.key.trim()) {
                    rows.push(makeRow());
                    updateHeight();
                    render(row.id);
                    syncNodeLayout(node);
                }
            });
            input.addEventListener("blur", () => {
                if (row !== rows[rows.length - 1] && !row.key.trim()) {
                    rows = ensureTrailingEmptyRow(rows.filter((candidate) => candidate !== row));
                    commit();
                    updateHeight();
                    render();
                }
            });
            input.addEventListener("keydown", (event) => {
                if (event.key !== "Enter" || !row.key.trim()) {
                    return;
                }
                event.preventDefault();
                const nextRow = rows[index + 1] ?? rows[rows.length - 1];
                focusRow(nextRow.id);
            });

            const remove = document.createElement("button");
            remove.type = "button";
            remove.className = "ipt-encryption-extra-key-list__delete";
            remove.textContent = "×";
            remove.title = "Remove key row";
            remove.setAttribute("aria-label", `Remove ${row.key || "key"} row`);
            remove.classList.toggle("is-placeholder", isPlaceholder);
            remove.tabIndex = isPlaceholder ? -1 : 0;
            remove.addEventListener("click", () => {
                rows = ensureTrailingEmptyRow(rows.filter((candidate) => candidate !== row));
                commit();
                updateHeight();
                render();
            });

            rowElement.append(toggle, input, remove);
            rowsElement.appendChild(rowElement);
        }

        if (focusRowId != null) {
            focusRow(focusRowId);
        }
    };

    const setValue = (value) => {
        const serializedCurrent = serializeRows(rows);
        const parsedRows = parseRows(value);
        const serializedNext = serializeRows(parsedRows);
        if (serializedCurrent === serializedNext) {
            return;
        }
        rows = ensureTrailingEmptyRow(parsedRows);
        updateHeight();
        render();
        syncNodeLayout(node);
    };

    updateHeight();
    render();

    const editorWidget = node.addDOMWidget(
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
            },
            getValue: () => serializeRows(rows),
            setValue,
        },
    );
    editorWidget.serialize = false;
    editorWidget.options = editorWidget.options ?? {};
    editorWidget.options.serialize = false;
    editorWidget.options.canvasOnly = false;

    return {
        widget: editorWidget,
        setValue,
    };
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

function installEditor(node) {
    if (!isTargetNode(node) || typeof node?.addDOMWidget !== "function") {
        return;
    }

    const backingWidget = getWidget(node, BACKING_WIDGET_NAME);
    if (!backingWidget) {
        return;
    }

    let state = node[STATE_KEY];
    if (!state) {
        try {
            state = createEditor(node, backingWidget);
        } catch (error) {
            console.error("[IPT] Failed to create Image Info Encryption Targets editor.", error);
            return;
        }
        node[STATE_KEY] = state;
        hideBackingWidget(backingWidget);
        refreshWidgetList(node);
    } else {
        hideBackingWidget(backingWidget);
        state.setValue(backingWidget.value);
        syncNodeLayout(node);
    }
}

app.registerExtension({
    name: "IPT.ImageInfoEncryptionTargetsEditor",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isTargetNodeDef(nodeData)) {
            return;
        }

        nodeType.prototype.onNodeCreated = chainCallback(
            nodeType.prototype.onNodeCreated,
            function onNodeCreated() {
                installEditor(this);
            },
        );
        nodeType.prototype.onConfigure = chainCallback(
            nodeType.prototype.onConfigure,
            function onConfigure() {
                installEditor(this);
            },
        );
        nodeType.prototype.onGraphConfigured = chainCallback(
            nodeType.prototype.onGraphConfigured,
            function onGraphConfigured() {
                installEditor(this);
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
