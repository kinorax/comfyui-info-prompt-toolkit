import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const MENU_LABEL = "Check Referenced Models...";
const STYLE_ID = "IPT-image-reader-model-check-style";
const WINDOW_WIDTH = 860;
const WINDOW_HEIGHT = 620;

function nextWindowZIndex() {
    if (typeof window === "undefined") {
        return 3000;
    }
    const current = Number(window.__iisWindowZIndex ?? 3000);
    const next = Number.isFinite(current) ? current + 1 : 3001;
    window.__iisWindowZIndex = next;
    return next;
}

function installStyle() {
    if (document.getElementById(STYLE_ID)) {
        return;
    }

    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .IPT-image-reader-check-window {
            position: fixed;
            left: 48px;
            top: 48px;
            width: ${WINDOW_WIDTH}px;
            height: ${WINDOW_HEIGHT}px;
            max-width: calc(100vw - 16px);
            max-height: calc(100vh - 16px);
            min-width: 420px;
            min-height: 280px;
            background: #15161a;
            border: 1px solid #3a3e49;
            border-radius: 8px;
            color: #e9edf6;
            box-shadow: 0 10px 35px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            resize: both;
            z-index: 3200;
        }
        .IPT-image-reader-check-header {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 10px;
            background: #1d2028;
            border-bottom: 1px solid #2e3440;
            cursor: move;
            user-select: none;
        }
        .IPT-image-reader-check-title {
            font-size: 14px;
            font-weight: 700;
            flex: 1 1 auto;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .IPT-image-reader-check-actions {
            display: flex;
            gap: 6px;
        }
        .IPT-image-reader-check-actions button,
        .IPT-image-reader-check-row button {
            padding: 3px 8px;
            border-radius: 4px;
            border: 1px solid #4a5160;
            background: #202636;
            color: #e9edf6;
            cursor: pointer;
        }
        .IPT-image-reader-check-actions button:hover,
        .IPT-image-reader-check-row button:hover {
            background: #2a3347;
        }
        .IPT-image-reader-check-content {
            display: flex;
            flex-direction: column;
            gap: 10px;
            flex: 1 1 auto;
            overflow: auto;
            padding: 12px;
        }
        .IPT-image-reader-check-status {
            font-size: 12px;
            color: #aeb7c8;
        }
        .IPT-image-reader-check-summary {
            font-size: 12px;
            color: #aeb7c8;
        }
        .IPT-image-reader-check-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .IPT-image-reader-check-row {
            display: grid;
            grid-template-columns: 88px minmax(0, 1fr) 92px 132px;
            gap: 10px;
            align-items: center;
            border: 1px solid #303748;
            border-radius: 6px;
            background: #11151d;
            padding: 10px;
        }
        .IPT-image-reader-check-kind {
            font-size: 12px;
            color: #8da0c2;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
        }
        .IPT-image-reader-check-main {
            min-width: 0;
        }
        .IPT-image-reader-check-name {
            font-size: 13px;
            color: #f2f6ff;
            word-break: break-word;
        }
        .IPT-image-reader-check-sub {
            margin-top: 4px;
            font-size: 12px;
            color: #9aa7bf;
            word-break: break-word;
        }
        .IPT-image-reader-check-local {
            font-size: 12px;
            color: #d7e6ff;
        }
        .IPT-image-reader-check-local.IPT-image-reader-check-local-missing {
            color: #ffb18f;
        }
        @media (max-width: 768px) {
            .IPT-image-reader-check-window {
                left: 8px !important;
                top: 8px !important;
                width: calc(100vw - 16px);
                height: calc(100vh - 16px);
            }
            .IPT-image-reader-check-row {
                grid-template-columns: 1fr;
                align-items: start;
            }
        }
    `;
    document.body.appendChild(style);
}

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

function isTargetNode(node) {
    const candidates = getNodeTypeCandidates(node);
    return candidates.some((candidate) => (
        candidate === "IPT-ImageReader"
        || candidate === "ImageReader"
        || candidate === "Image Reader"
    ));
}

function isTargetNodeDef(nodeData) {
    const candidates = [
        nodeData?.name,
        nodeData?.display_name,
        nodeData?.type,
        nodeData?.node_id,
    ].filter(Boolean);
    return candidates.some((candidate) => (
        candidate === "IPT-ImageReader"
        || candidate === "ImageReader"
        || candidate === "Image Reader"
    ));
}

function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) ?? null;
}

function resolveSelectedImage(node) {
    const widget = getWidget(node, "image");
    if (!widget) {
        return "";
    }
    return String(widget.value ?? "").trim();
}

async function fetchReferencedModels(image) {
    const response = await api.fetchApi("/ipt/image-reader/model-check", {
        method: "POST",
        headers: {
            "content-type": "application/json",
        },
        body: JSON.stringify({ image }),
    });
    if (!response.ok) {
        let errorText = `HTTP ${response.status}`;
        try {
            const body = await response.json();
            if (body?.error) {
                errorText = String(body.error);
            }
        } catch {
            // Use default text.
        }
        throw new Error(errorText);
    }
    return response.json();
}

function localStatusText(item) {
    const status = String(item?.local_status ?? "").trim();
    if (status === "present") {
        return "Present";
    }
    if (status === "missing") {
        return "Missing";
    }
    return status || "-";
}

function sublineText(item) {
    const segments = [];
    const nameRaw = String(item?.name_raw ?? "").trim();
    const strength = typeof item?.strength === "number" ? item.strength : null;
    if (nameRaw) {
        segments.push(`Raw: ${nameRaw}`);
    }
    if (strength !== null) {
        segments.push(`Strength: ${strength}`);
    }
    const localRelativePath = String(item?.local_match?.relative_path ?? "").trim();
    if (localRelativePath) {
        segments.push(`Local: ${localRelativePath}`);
    }
    return segments.join(" | ");
}

function openViewModelInfo(item) {
    const bridge = window.__iisModelInfoWindow;
    if (!bridge || typeof bridge.openModelInfoWindow !== "function") {
        alert("View Model Info is unavailable.");
        return;
    }

    const target = bridge.resolveTargetByFolderName?.(item.folder_hint) ?? null;
    bridge.openModelInfoWindow({
        node: null,
        target,
        source: item.view_model_info_source || {},
    });
}

class ImageReaderModelCheckWindow {
    constructor(node, image) {
        this.node = node;
        this.image = image;
        this.element = null;
        this.content = null;
    }

    show() {
        installStyle();

        const panel = document.createElement("section");
        panel.className = "IPT-image-reader-check-window";
        panel.style.zIndex = String(nextWindowZIndex());

        const header = document.createElement("header");
        header.className = "IPT-image-reader-check-header";

        const title = document.createElement("div");
        title.className = "IPT-image-reader-check-title";
        title.textContent = `Check Referenced Models (${this.image})`;

        const actions = document.createElement("div");
        actions.className = "IPT-image-reader-check-actions";

        const refreshButton = document.createElement("button");
        refreshButton.type = "button";
        refreshButton.textContent = "Refresh";
        refreshButton.addEventListener("click", () => this.load());

        const closeButton = document.createElement("button");
        closeButton.type = "button";
        closeButton.textContent = "Close";
        closeButton.addEventListener("click", () => this.close());

        actions.append(refreshButton, closeButton);
        header.append(title, actions);

        const content = document.createElement("div");
        content.className = "IPT-image-reader-check-content";

        panel.append(header, content);
        document.body.appendChild(panel);
        panel.addEventListener("mousedown", () => {
            panel.style.zIndex = String(nextWindowZIndex());
        });

        this.attachDrag(panel, header);
        this.centerWindow(panel);
        this.element = panel;
        this.content = content;

        this.load();
    }

    centerWindow(panel) {
        const width = panel.offsetWidth;
        const height = panel.offsetHeight;
        const maxLeft = Math.max(0, window.innerWidth - width);
        const maxTop = Math.max(0, window.innerHeight - height);
        panel.style.left = `${Math.floor(maxLeft / 2)}px`;
        panel.style.top = `${Math.floor(maxTop / 2)}px`;
    }

    attachDrag(panel, handle) {
        const onMouseDown = (event) => {
            if (event.button !== 0) {
                return;
            }
            const target = event.target instanceof Element ? event.target : null;
            if (target?.closest("button")) {
                return;
            }

            event.preventDefault();
            const rect = panel.getBoundingClientRect();
            const startOffsetX = event.clientX - rect.left;
            const startOffsetY = event.clientY - rect.top;

            const onMouseMove = (moveEvent) => {
                const maxLeft = Math.max(0, window.innerWidth - panel.offsetWidth);
                const maxTop = Math.max(0, window.innerHeight - panel.offsetHeight);
                const nextLeft = Math.min(maxLeft, Math.max(0, moveEvent.clientX - startOffsetX));
                const nextTop = Math.min(maxTop, Math.max(0, moveEvent.clientY - startOffsetY));
                panel.style.left = `${nextLeft}px`;
                panel.style.top = `${nextTop}px`;
            };

            const onMouseUp = () => {
                window.removeEventListener("mousemove", onMouseMove);
                window.removeEventListener("mouseup", onMouseUp);
            };

            window.addEventListener("mousemove", onMouseMove);
            window.addEventListener("mouseup", onMouseUp);
        };

        handle.addEventListener("mousedown", onMouseDown);
    }

    close() {
        this.element?.remove();
        this.element = null;
        this.content = null;
    }

    setStatus(text) {
        if (!this.content) {
            return;
        }
        this.content.replaceChildren();
        const status = document.createElement("div");
        status.className = "IPT-image-reader-check-status";
        status.textContent = text;
        this.content.appendChild(status);
    }

    render(payload) {
        if (!this.content) {
            return;
        }
        this.content.replaceChildren();

        const infotextFound = Boolean(payload?.infotext_found);
        const items = Array.isArray(payload?.items) ? payload.items : [];

        if (!infotextFound) {
            this.setStatus("No infotext was found in the selected image.");
            return;
        }

        if (!items.length) {
            this.setStatus("No referenced models were found in the selected image metadata.");
            return;
        }

        const summary = document.createElement("div");
        summary.className = "IPT-image-reader-check-summary";
        summary.textContent = `Referenced: ${items.length}`;
        this.content.appendChild(summary);

        const list = document.createElement("div");
        list.className = "IPT-image-reader-check-list";

        for (const item of items) {
            const row = document.createElement("div");
            row.className = "IPT-image-reader-check-row";

            const kind = document.createElement("div");
            kind.className = "IPT-image-reader-check-kind";
            kind.textContent = String(item?.kind_label ?? item?.kind ?? "-");

            const main = document.createElement("div");
            main.className = "IPT-image-reader-check-main";

            const name = document.createElement("div");
            name.className = "IPT-image-reader-check-name";
            name.textContent = String(item?.display_name ?? item?.name_raw ?? "(unknown)");

            const sub = document.createElement("div");
            sub.className = "IPT-image-reader-check-sub";
            sub.textContent = sublineText(item) || `Folder hint: ${String(item?.folder_hint ?? "-")}`;

            main.append(name, sub);

            const local = document.createElement("div");
            local.className = "IPT-image-reader-check-local";
            if (String(item?.local_status ?? "") !== "present") {
                local.classList.add("IPT-image-reader-check-local-missing");
            }
            local.textContent = localStatusText(item);

            const action = document.createElement("div");
            const button = document.createElement("button");
            button.type = "button";
            button.textContent = "View Model Info...";
            button.addEventListener("click", () => openViewModelInfo(item));
            action.appendChild(button);

            row.append(kind, main, local, action);
            list.appendChild(row);
        }

        this.content.appendChild(list);
    }

    async load() {
        this.setStatus("Loading referenced models...");
        try {
            const payload = await fetchReferencedModels(this.image);
            this.render(payload);
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            this.setStatus(`Failed to inspect referenced models: ${message}`);
        }
    }
}

function addMenuOption(node, options) {
    if (!isTargetNode(node)) {
        return;
    }

    options.unshift({
        content: MENU_LABEL,
        callback: () => {
            const image = resolveSelectedImage(node);
            if (!image) {
                alert("Select an image first.");
                return;
            }
            new ImageReaderModelCheckWindow(node, image).show();
        },
    });
}

app.registerExtension({
    name: "IPT.ImageReaderModelCheckWindow",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isTargetNodeDef(nodeData)) {
            return;
        }
        if (nodeType.prototype.__iptImageReaderModelCheckMenuPatched) {
            return;
        }

        const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function patchedGetExtraMenuOptions(_, options) {
            if (Array.isArray(options)) {
                addMenuOption(this, options);
            }
            return originalGetExtraMenuOptions?.apply(this, arguments);
        };

        nodeType.prototype.__iptImageReaderModelCheckMenuPatched = true;
    },
});
