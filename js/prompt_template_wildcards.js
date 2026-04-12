import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE_TYPES = new Set([
    "IPT-PromptTemplate",
    "PromptTemplate",
    "Prompt Template",
]);

const TEMPLATE_WIDGET_NAME = "template";
const CONTROLLER_KEY = "__iptPromptTemplateWildcardController";
const STYLE_ID = "ipt-prompt-template-wildcards-style";
const MAX_RESULTS = 100;
const UPDATE_DELAY_MS = 60;
const BLUR_HIDE_DELAY_MS = 120;
const MIRROR_STYLE_PROPERTIES = [
    "boxSizing",
    "width",
    "height",
    "overflowX",
    "overflowY",
    "borderTopWidth",
    "borderRightWidth",
    "borderBottomWidth",
    "borderLeftWidth",
    "paddingTop",
    "paddingRight",
    "paddingBottom",
    "paddingLeft",
    "fontStyle",
    "fontVariant",
    "fontWeight",
    "fontStretch",
    "fontSize",
    "fontFamily",
    "lineHeight",
    "letterSpacing",
    "textTransform",
    "textIndent",
    "textDecoration",
    "textAlign",
    "tabSize",
    "whiteSpace",
    "wordBreak",
    "overflowWrap",
];

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

function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) ?? null;
}

function resolveTextarea(widget) {
    const element = widget?.element ?? widget?.inputEl ?? null;
    return element instanceof HTMLTextAreaElement ? element : null;
}

function wrapIndex(index, length) {
    if (!Number.isInteger(length) || length <= 0) {
        return -1;
    }
    if (!Number.isInteger(index)) {
        return 0;
    }
    return ((index % length) + length) % length;
}

function installStyleOnce() {
    if (document.getElementById(STYLE_ID)) {
        return;
    }

    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ipt-prompt-template-wildcards {
    position: fixed;
    z-index: 9999;
    display: none;
    min-width: 220px;
    max-width: min(460px, calc(100vw - 24px));
    max-height: min(320px, calc(100vh - 24px));
    overflow-y: auto;
    padding: 6px;
    border: 1px solid var(--border-color, rgba(255, 255, 255, 0.14));
    border-radius: 10px;
    background: var(--comfy-menu-bg, rgba(24, 27, 33, 0.97));
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(10px);
}

.ipt-prompt-template-wildcards[data-visible="true"] {
    display: block;
}

.ipt-prompt-template-wildcards-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.ipt-prompt-template-wildcards-item {
    width: 100%;
    border: 0;
    border-radius: 8px;
    padding: 8px 10px;
    background: transparent;
    color: var(--input-text, #f4f5f7);
    font: inherit;
    font-size: 12px;
    line-height: 1.35;
    text-align: left;
    cursor: pointer;
}

.ipt-prompt-template-wildcards-item:hover,
.ipt-prompt-template-wildcards-item[data-active="true"],
.ipt-prompt-template-wildcards-item:focus-visible {
    outline: none;
    background: rgba(255, 255, 255, 0.08);
}

.ipt-prompt-template-wildcards-primary {
    display: block;
    word-break: break-word;
}

.ipt-prompt-template-wildcards-secondary {
    display: block;
    margin-top: 4px;
    white-space: pre-wrap;
    word-break: break-word;
    opacity: 0.72;
}
`;
    document.head.appendChild(style);
}

function findOpenWildcardStart(text) {
    let openStart = -1;
    let searchIndex = 0;

    while (searchIndex < text.length) {
        const markerIndex = text.indexOf("__", searchIndex);
        if (markerIndex < 0) {
            break;
        }

        openStart = openStart < 0 ? markerIndex : -1;
        searchIndex = markerIndex + 2;
    }

    if (openStart < 0) {
        return -1;
    }

    const candidate = text.slice(openStart + 2);
    if (candidate.includes("\n") || candidate.includes("\r")) {
        return -1;
    }

    return openStart;
}

function findWildcardReplaceEnd(value, caretIndex) {
    const afterCaret = value.slice(caretIndex);
    const closingOffset = afterCaret.indexOf("__");
    if (closingOffset < 0) {
        return caretIndex;
    }

    const newlineOffset = afterCaret.search(/[\r\n]/);
    if (newlineOffset >= 0 && newlineOffset < closingOffset) {
        return caretIndex;
    }

    return caretIndex + closingOffset + 2;
}

function getSelectorContext(textarea) {
    if (!(textarea instanceof HTMLTextAreaElement)) {
        return null;
    }

    const caretIndex = textarea.selectionStart;
    const beforeCaret = textarea.value.slice(0, caretIndex);
    const hashIndex = beforeCaret.lastIndexOf("#");
    if (hashIndex < 0) {
        return null;
    }

    const selectorQuery = beforeCaret.slice(hashIndex + 1);
    if (/[^0-9]/.test(selectorQuery)) {
        return null;
    }

    const beforeHash = beforeCaret.slice(0, hashIndex);
    if (!beforeHash.endsWith("__")) {
        return null;
    }

    const openStart = beforeHash.lastIndexOf("__", beforeHash.length - 3);
    if (openStart < 0) {
        return null;
    }

    const tokenPath = beforeHash.slice(openStart + 2, beforeHash.length - 2);
    if (!tokenPath || tokenPath.includes("\n") || tokenPath.includes("\r")) {
        return null;
    }

    return {
        mode: "items",
        tokenPath,
        query: selectorQuery,
        replaceStart: hashIndex + 1,
        replaceEnd: caretIndex,
    };
}

function getWildcardContext(textarea) {
    if (!(textarea instanceof HTMLTextAreaElement)) {
        return null;
    }

    const caretIndex = textarea.selectionStart;
    const beforeCaret = textarea.value.slice(0, caretIndex);
    const openStart = findOpenWildcardStart(beforeCaret);
    if (openStart < 0) {
        return null;
    }

    return {
        mode: "wildcards",
        query: beforeCaret.slice(openStart + 2),
        replaceStart: openStart,
        replaceEnd: findWildcardReplaceEnd(textarea.value, caretIndex),
    };
}

function getCaretViewportPosition(textarea) {
    const rect = textarea.getBoundingClientRect();
    const style = window.getComputedStyle(textarea);
    const mirror = document.createElement("div");

    mirror.style.position = "fixed";
    mirror.style.left = `${rect.left}px`;
    mirror.style.top = `${rect.top}px`;
    mirror.style.visibility = "hidden";
    mirror.style.pointerEvents = "none";
    mirror.style.whiteSpace = "pre-wrap";
    mirror.style.overflowWrap = "break-word";

    for (const property of MIRROR_STYLE_PROPERTIES) {
        mirror.style[property] = style[property];
    }

    mirror.textContent = textarea.value.slice(0, textarea.selectionStart);

    const marker = document.createElement("span");
    marker.textContent = textarea.value.slice(textarea.selectionStart) || ".";
    mirror.appendChild(marker);

    document.body.appendChild(mirror);

    const lineHeight = Number.parseFloat(style.lineHeight) || (Number.parseFloat(style.fontSize) * 1.2) || 16;
    const top = rect.top + marker.offsetTop - textarea.scrollTop + lineHeight;
    const left = rect.left + marker.offsetLeft - textarea.scrollLeft;

    document.body.removeChild(mirror);
    return { top, left };
}

class PromptTemplateWildcardController {
    constructor(node, widget, textarea) {
        this.node = node;
        this.widget = widget;
        this.textarea = textarea;
        this.dropdown = document.createElement("div");
        this.dropdown.className = "ipt-prompt-template-wildcards";
        this.dropdown.dataset.visible = "false";
        this.list = document.createElement("div");
        this.list.className = "ipt-prompt-template-wildcards-list";
        this.dropdown.appendChild(this.list);
        document.body.appendChild(this.dropdown);

        this.entries = [];
        this.context = null;
        this.lastRequestKey = null;
        this.requestCache = new Map();
        this.inflightRequests = new Map();
        this.requestSerial = 0;
        this.updateTimer = null;
        this.selectedIndex = 0;
        this.dismissedSnapshot = null;
        this.boundHandlers = {
            input: () => this.scheduleUpdate(),
            click: () => this.scheduleUpdate(),
            keydown: (event) => this.handleKeyDown(event),
            keyup: () => this.scheduleUpdate(),
            scroll: () => this.scheduleUpdate(),
            blur: () => window.setTimeout(() => this.hide(), BLUR_HIDE_DELAY_MS),
            resize: () => this.reposition(),
            pointerdown: (event) => this.handleDocumentPointerDown(event),
        };

        this.attach(textarea);
    }

    setTarget(widget, textarea) {
        this.widget = widget;
        this.attach(textarea);
    }

    attach(textarea) {
        if (this.textarea === textarea && this.isAttached) {
            return;
        }

        this.detach();
        this.textarea = textarea;
        if (!(this.textarea instanceof HTMLTextAreaElement)) {
            return;
        }

        this.textarea.addEventListener("input", this.boundHandlers.input);
        this.textarea.addEventListener("click", this.boundHandlers.click);
        this.textarea.addEventListener("keydown", this.boundHandlers.keydown);
        this.textarea.addEventListener("keyup", this.boundHandlers.keyup);
        this.textarea.addEventListener("scroll", this.boundHandlers.scroll);
        this.textarea.addEventListener("blur", this.boundHandlers.blur);
        window.addEventListener("resize", this.boundHandlers.resize);
        document.addEventListener("pointerdown", this.boundHandlers.pointerdown, true);
        this.isAttached = true;
    }

    detach() {
        if (!this.isAttached || !(this.textarea instanceof HTMLTextAreaElement)) {
            return;
        }

        this.textarea.removeEventListener("input", this.boundHandlers.input);
        this.textarea.removeEventListener("click", this.boundHandlers.click);
        this.textarea.removeEventListener("keydown", this.boundHandlers.keydown);
        this.textarea.removeEventListener("keyup", this.boundHandlers.keyup);
        this.textarea.removeEventListener("scroll", this.boundHandlers.scroll);
        this.textarea.removeEventListener("blur", this.boundHandlers.blur);
        window.removeEventListener("resize", this.boundHandlers.resize);
        document.removeEventListener("pointerdown", this.boundHandlers.pointerdown, true);
        this.isAttached = false;
    }

    destroy() {
        if (this.updateTimer != null) {
            window.clearTimeout(this.updateTimer);
            this.updateTimer = null;
        }
        this.detach();
        this.dropdown.remove();
        this.entries = [];
        this.context = null;
        this.requestCache.clear();
        this.inflightRequests.clear();
    }

    scheduleUpdate() {
        if (this.updateTimer != null) {
            window.clearTimeout(this.updateTimer);
        }

        this.updateTimer = window.setTimeout(() => {
            this.updateTimer = null;
            void this.update();
        }, UPDATE_DELAY_MS);
    }

    captureSnapshot() {
        if (!(this.textarea instanceof HTMLTextAreaElement)) {
            return null;
        }
        if (this.textarea.selectionStart == null || this.textarea.selectionEnd == null) {
            return null;
        }

        return {
            value: this.textarea.value,
            selectionStart: this.textarea.selectionStart,
            selectionEnd: this.textarea.selectionEnd,
        };
    }

    matchesDismissedSnapshot() {
        if (!(this.textarea instanceof HTMLTextAreaElement) || !this.dismissedSnapshot) {
            return false;
        }

        const snapshot = this.captureSnapshot();
        if (!snapshot) {
            this.dismissedSnapshot = null;
            return false;
        }

        const matches = (
            snapshot.value === this.dismissedSnapshot.value
            && snapshot.selectionStart === this.dismissedSnapshot.selectionStart
            && snapshot.selectionEnd === this.dismissedSnapshot.selectionEnd
        );
        if (!matches) {
            this.dismissedSnapshot = null;
        }
        return matches;
    }

    dismissCurrentSelection() {
        this.dismissedSnapshot = this.captureSnapshot();
    }

    getContext() {
        if (!(this.textarea instanceof HTMLTextAreaElement)) {
            return null;
        }
        if (document.activeElement !== this.textarea) {
            return null;
        }
        if (this.textarea.selectionStart == null || this.textarea.selectionEnd == null) {
            return null;
        }
        if (this.textarea.selectionStart !== this.textarea.selectionEnd) {
            return null;
        }
        if (this.matchesDismissedSnapshot()) {
            return null;
        }

        const selectorContext = getSelectorContext(this.textarea);
        const baseContext = selectorContext ?? getWildcardContext(this.textarea);
        if (!baseContext) {
            return null;
        }

        return {
            ...baseContext,
            position: getCaretViewportPosition(this.textarea),
        };
    }

    makeRequestKey(context) {
        if (context.mode === "items") {
            return `items:${context.tokenPath}:${context.query}`;
        }
        return `wildcards:${context.query}`;
    }

    isOpen() {
        return this.dropdown.dataset.visible === "true";
    }

    getCachedEntries(requestKey) {
        if (typeof requestKey !== "string" || !requestKey) {
            return null;
        }
        return this.requestCache.get(requestKey) ?? null;
    }

    setCachedEntries(requestKey, entries) {
        if (typeof requestKey !== "string" || !requestKey) {
            return;
        }
        this.requestCache.set(requestKey, Array.isArray(entries) ? entries : []);
    }

    requestEntries(context, requestKey) {
        if (typeof requestKey !== "string" || !requestKey) {
            return Promise.resolve([]);
        }

        const existingRequest = this.inflightRequests.get(requestKey);
        if (existingRequest) {
            return existingRequest;
        }

        const request = this.fetchEntries(context)
            .then((entries) => {
                if (entries == null) {
                    return this.getCachedEntries(requestKey) ?? [];
                }
                const normalizedEntries = Array.isArray(entries) ? entries : [];
                this.setCachedEntries(requestKey, normalizedEntries);
                return normalizedEntries;
            })
            .finally(() => {
                if (this.inflightRequests.get(requestKey) === request) {
                    this.inflightRequests.delete(requestKey);
                }
            });

        this.inflightRequests.set(requestKey, request);
        return request;
    }

    revalidateEntries(context, requestKey) {
        void this.requestEntries(context, requestKey);
    }

    async update() {
        const context = this.getContext();
        if (!context) {
            this.hide();
            return;
        }

        this.context = context;
        const requestKey = this.makeRequestKey(context);
        if (requestKey === this.lastRequestKey) {
            if (this.entries.length > 0) {
                this.renderEntries();
                this.show();
            } else {
                this.hide();
            }
            return;
        }

        this.lastRequestKey = requestKey;
        this.selectedIndex = 0;

        const cachedEntries = this.getCachedEntries(requestKey);
        if (cachedEntries != null) {
            this.entries = cachedEntries;
            if (cachedEntries.length > 0) {
                this.renderEntries();
                this.show();
            } else {
                this.hide();
            }
            this.revalidateEntries(context, requestKey);
            return;
        }

        const requestSerial = ++this.requestSerial;
        const entries = await this.requestEntries(context, requestKey);
        if (requestSerial !== this.requestSerial || requestKey !== this.lastRequestKey) {
            return;
        }

        this.entries = entries;
        this.selectedIndex = 0;

        if (entries.length === 0) {
            this.hide();
            return;
        }

        this.renderEntries();
        this.show();
    }

    async fetchEntries(context) {
        if (!api || typeof api.fetchApi !== "function") {
            return null;
        }

        const params = new URLSearchParams({
            mode: context.mode,
            q: context.query,
            limit: String(MAX_RESULTS),
        });
        if (context.mode === "items") {
            params.set("token", context.tokenPath);
        }

        try {
            const response = await api.fetchApi(`/ipt/prompt-template/wildcards?${params.toString()}`, {
                method: "GET",
                cache: "no-store",
            });
            if (!response?.ok) {
                return null;
            }

            const payload = await response.json();
            if (!payload || payload.ok !== true || !Array.isArray(payload.entries)) {
                return null;
            }

            if (context.mode === "items") {
                return payload.entries
                    .map((entry) => ({
                        mode: "items",
                        index: Number(entry?.index),
                        insertText: String(entry?.insert_text ?? "").trim(),
                        displayText: String(entry?.display_text ?? "").trim(),
                        description: String(entry?.description ?? ""),
                        weight: Number(entry?.weight),
                        randomEnabled: entry?.random_enabled !== false,
                        selectorToken: String(entry?.selector_token ?? "").trim(),
                        value: String(entry?.value ?? "").trim(),
                    }))
                    .filter((entry) => Number.isFinite(entry.index) && entry.index > 0 && entry.insertText);
            }

            return payload.entries
                .map((entry) => ({
                    mode: "wildcards",
                    path: String(entry?.path ?? "").trim(),
                    insertText: String(entry?.token ?? "").trim(),
                }))
                .filter((entry) => entry.path && entry.insertText);
        } catch (error) {
            console.warn("[IPT.PromptTemplateWildcards] failed to fetch wildcard suggestions", error);
            return null;
        }
    }

    setSelectedIndex(index, { scroll = true } = {}) {
        this.selectedIndex = wrapIndex(index, this.entries.length);

        const buttons = Array.from(this.list.children);
        for (const [buttonIndex, button] of buttons.entries()) {
            if (!(button instanceof HTMLElement)) {
                continue;
            }
            button.dataset.active = buttonIndex === this.selectedIndex ? "true" : "false";
        }

        if (!scroll || this.selectedIndex < 0) {
            return;
        }

        const activeButton = buttons[this.selectedIndex];
        if (activeButton instanceof HTMLElement) {
            activeButton.scrollIntoView({ block: "nearest" });
        }
    }

    moveSelection(delta) {
        if (this.entries.length === 0) {
            return;
        }
        this.setSelectedIndex(this.selectedIndex + delta);
    }

    getSelectedEntry() {
        if (this.entries.length === 0) {
            return null;
        }
        const index = this.selectedIndex >= 0 ? this.selectedIndex : 0;
        return this.entries[index] ?? null;
    }

    selectSelectedEntry() {
        const entry = this.getSelectedEntry();
        if (!entry) {
            return;
        }
        this.insertEntry(entry);
    }

    renderEntries() {
        this.list.replaceChildren();

        for (const [index, entry] of this.entries.entries()) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "ipt-prompt-template-wildcards-item";
            button.dataset.active = "false";

            const primary = document.createElement("span");
            primary.className = "ipt-prompt-template-wildcards-primary";
            if (entry.mode === "items") {
                primary.textContent = `#${entry.index}  ${entry.displayText || entry.value}`;
            } else {
                primary.textContent = entry.path;
            }
            button.appendChild(primary);

            const secondaryLines = [];
            if (entry.mode === "items" && entry.randomEnabled === false) {
                secondaryLines.push("random off (weight 0)");
            }
            if (entry.mode === "items" && entry.description) {
                secondaryLines.push(entry.description);
            }
            if (secondaryLines.length > 0) {
                const secondary = document.createElement("span");
                secondary.className = "ipt-prompt-template-wildcards-secondary";
                secondary.textContent = secondaryLines.join("\n");
                button.appendChild(secondary);
            }

            button.addEventListener("mousedown", (event) => {
                event.preventDefault();
            });
            button.addEventListener("mouseenter", () => {
                this.setSelectedIndex(index, { scroll: false });
            });
            button.addEventListener("click", (event) => {
                event.preventDefault();
                this.insertEntry(entry);
            });

            this.list.appendChild(button);
        }

        this.setSelectedIndex(this.selectedIndex, { scroll: false });
    }

    insertEntry(entry) {
        if (!(this.textarea instanceof HTMLTextAreaElement) || !this.context) {
            return;
        }

        const contextMode = this.context.mode;
        this.textarea.focus();
        this.textarea.setSelectionRange(this.context.replaceStart, this.context.replaceEnd);
        this.textarea.setRangeText(entry.insertText, this.context.replaceStart, this.context.replaceEnd, "end");
        this.textarea.dispatchEvent(new Event("input", { bubbles: true }));
        if (contextMode === "items") {
            this.dismissCurrentSelection();
        }
        this.hide();
    }

    handleKeyDown(event) {
        if (!(event instanceof KeyboardEvent) || event.isComposing || !this.isOpen()) {
            return;
        }

        switch (event.key) {
            case "ArrowUp":
                event.preventDefault();
                event.stopPropagation();
                this.moveSelection(-1);
                break;
            case "ArrowDown":
                event.preventDefault();
                event.stopPropagation();
                this.moveSelection(1);
                break;
            case "Enter":
                event.preventDefault();
                event.stopPropagation();
                this.selectSelectedEntry();
                break;
            case "Escape":
                event.preventDefault();
                event.stopPropagation();
                this.dismissCurrentSelection();
                this.hide();
                break;
            default:
                break;
        }
    }

    handleDocumentPointerDown(event) {
        if (!this.isOpen()) {
            return;
        }

        const target = event.target;
        if (!(target instanceof Node)) {
            return;
        }
        if (this.dropdown.contains(target) || target === this.textarea) {
            return;
        }

        this.hide();
    }

    show() {
        if (!this.context) {
            this.hide();
            return;
        }

        this.dropdown.dataset.visible = "true";
        this.dropdown.style.display = "block";
        this.reposition();
    }

    reposition() {
        if (this.dropdown.dataset.visible !== "true" || !this.context) {
            return;
        }

        const dropdownRect = this.dropdown.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        let top = this.context.position.top + 6;
        let left = this.context.position.left;

        if (left + dropdownRect.width > viewportWidth - 12) {
            left = Math.max(12, viewportWidth - dropdownRect.width - 12);
        }
        if (top + dropdownRect.height > viewportHeight - 12) {
            top = Math.max(12, this.context.position.top - dropdownRect.height - 10);
        }

        this.dropdown.style.top = `${Math.round(top)}px`;
        this.dropdown.style.left = `${Math.round(left)}px`;
    }

    hide() {
        this.dropdown.dataset.visible = "false";
        this.dropdown.style.display = "none";
        this.context = null;
        this.lastRequestKey = null;
        this.selectedIndex = 0;
    }
}

function patchNode(node) {
    if (!isTargetNode(node)) {
        return;
    }

    const widget = getWidget(node, TEMPLATE_WIDGET_NAME);
    const textarea = resolveTextarea(widget);
    if (!widget || !textarea) {
        return;
    }

    const existingController = node[CONTROLLER_KEY];
    if (existingController instanceof PromptTemplateWildcardController) {
        existingController.setTarget(widget, textarea);
        return;
    }

    node[CONTROLLER_KEY] = new PromptTemplateWildcardController(node, widget, textarea);
}

function destroyNodeController(node) {
    const controller = node?.[CONTROLLER_KEY];
    if (!(controller instanceof PromptTemplateWildcardController)) {
        return;
    }

    controller.destroy();
    node[CONTROLLER_KEY] = null;
}

app.registerExtension({
    name: "IPT.PromptTemplateWildcards",
    init() {
        installStyleOnce();
    },
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

        nodeType.prototype.onRemoved = chainCallback(
            nodeType.prototype.onRemoved,
            function onRemoved() {
                destroyNodeController(this);
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
