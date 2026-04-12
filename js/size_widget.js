import { app } from "../../scripts/app.js";

const SIZE_TYPE = "IPT-Size";
const SIZE_PATTERN = /^\s*(-?\d+)\s*[xX]\s*(-?\d+)\s*$/;
const DEFAULT_WIDTH = 512;
const DEFAULT_HEIGHT = 512;
const DEFAULT_MIN = 16;
const DEFAULT_MAX = 16384;
const DEFAULT_STEP = 1;
const STYLE_ID = "ipt-size-widget-style";
const NODE_WIDGET_HEIGHT = 20;
const NODE_WIDGET_MARGIN = 0;

function toInt(value, fallback) {
    const n = Number(value);
    if (!Number.isFinite(n)) {
        return Math.trunc(fallback);
    }
    return Math.trunc(n);
}

function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
}

function resolvePalette() {
    const liteGraph = globalThis.LiteGraph ?? null;
    return {
        background: liteGraph?.WIDGET_BGCOLOR ?? "#222",
        outline: liteGraph?.WIDGET_OUTLINE_COLOR ?? "#666",
        text: liteGraph?.WIDGET_TEXT_COLOR ?? "#DDD",
        subtext: liteGraph?.WIDGET_SECONDARY_TEXT_COLOR ?? "#999",
        disabledText: liteGraph?.WIDGET_DISABLED_TEXT_COLOR ?? "#666",
    };
}

function installStyleOnce() {
    if (document.getElementById(STYLE_ID)) {
        return;
    }

    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .ipt-size-widget-input {
            -moz-appearance: textfield;
        }
        .ipt-size-widget-input::-webkit-outer-spin-button,
        .ipt-size-widget-input::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
    `;
    document.head.appendChild(style);
}

function parseRawSize(value) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
        const width = toInt(value.width, NaN);
        const height = toInt(value.height, NaN);
        if (Number.isFinite(width) && Number.isFinite(height)) {
            return { width, height };
        }
    }

    if (Array.isArray(value) && value.length >= 2) {
        const width = toInt(value[0], NaN);
        const height = toInt(value[1], NaN);
        if (Number.isFinite(width) && Number.isFinite(height)) {
            return { width, height };
        }
    }

    if (typeof value === "string") {
        const matched = SIZE_PATTERN.exec(value);
        if (matched) {
            const width = toInt(matched[1], NaN);
            const height = toInt(matched[2], NaN);
            if (Number.isFinite(width) && Number.isFinite(height)) {
                return { width, height };
            }
        }
    }

    return null;
}

function normalizeSize(value, fallback, minimum, maximum) {
    const parsed = parseRawSize(value);
    const source = parsed ?? fallback;
    return {
        width: clamp(toInt(source.width, fallback.width), minimum, maximum),
        height: clamp(toInt(source.height, fallback.height), minimum, maximum),
    };
}

function createNumberInput(minimum, maximum, step, palette) {
    const input = document.createElement("input");
    input.className = "ipt-size-widget-input";
    input.type = "number";
    input.min = String(minimum);
    input.max = String(maximum);
    input.step = String(step);
    input.style.width = "0";
    input.style.flex = "1 1 auto";
    input.style.minWidth = "0";
    input.style.boxSizing = "border-box";
    input.style.padding = "0 4px";
    input.style.margin = "0";
    input.style.background = "transparent";
    input.style.border = "none";
    input.style.outline = "none";
    input.style.textAlign = "right";
    input.style.fontSize = "12px";
    input.style.lineHeight = "1";
    input.style.color = palette.text;
    return input;
}

function createLabeledValue(labelText, input, palette) {
    const group = document.createElement("div");
    group.style.display = "flex";
    group.style.alignItems = "center";
    group.style.gap = "6px";
    group.style.flex = "1 1 0";
    group.style.minWidth = "0";

    const label = document.createElement("span");
    label.textContent = labelText;
    label.style.color = palette.subtext;
    label.style.fontSize = "11px";
    label.style.lineHeight = "1";
    label.style.whiteSpace = "nowrap";
    label.style.userSelect = "none";

    group.appendChild(label);
    group.appendChild(input);

    return { group, label };
}

app.registerExtension({
    name: "IPT.SizeWidget",
    getCustomWidgets() {
        installStyleOnce();
        return {
            [SIZE_TYPE](node, inputName, inputData) {
                const palette = resolvePalette();
                const options = inputData?.[1] ?? {};
                const minimum = toInt(options.min, DEFAULT_MIN);
                const maximum = toInt(options.max, DEFAULT_MAX);
                const step = toInt(options.step, DEFAULT_STEP);
                const defaultSize = normalizeSize(
                    options.default,
                    { width: DEFAULT_WIDTH, height: DEFAULT_HEIGHT },
                    minimum,
                    maximum,
                );
                let state = { ...defaultSize };

                const host = document.createElement("div");
                host.style.display = "flex";
                host.style.alignItems = "center";
                host.style.width = "100%";
                host.style.height = "100%";
                host.style.boxSizing = "border-box";
                host.style.padding = "0 15px";

                const container = document.createElement("div");
                container.style.display = "flex";
                container.style.alignItems = "center";
                container.style.justifyContent = "space-between";
                container.style.gap = "4px";
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.boxSizing = "border-box";
                container.style.padding = "0 8px";
                container.style.border = `1px solid ${palette.outline}`;
                container.style.borderRadius = "999px";
                container.style.background = palette.background;
                container.style.overflow = "hidden";

                const widthInput = createNumberInput(minimum, maximum, step, palette);
                const heightInput = createNumberInput(minimum, maximum, step, palette);
                const widthField = createLabeledValue("width", widthInput, palette);
                const heightField = createLabeledValue("height", heightInput, palette);
                widthField.group.style.padding = "0 3px 0 2px";
                heightField.group.style.padding = "0 2px 0 3px";
                const separator = document.createElement("span");
                separator.textContent = "x";
                separator.style.color = palette.subtext;
                separator.style.fontSize = "12px";
                separator.style.lineHeight = "1";
                separator.style.userSelect = "none";
                separator.style.margin = "0 2px";

                container.appendChild(widthField.group);
                container.appendChild(separator);
                container.appendChild(heightField.group);
                host.appendChild(container);
                let lastDisabled = null;

                const syncInputs = () => {
                    widthInput.value = String(state.width);
                    heightInput.value = String(state.height);
                };

                const widget = node.addDOMWidget(inputName, SIZE_TYPE, host, {
                    margin: NODE_WIDGET_MARGIN,
                    getMinHeight: () => NODE_WIDGET_HEIGHT,
                    getMaxHeight: () => NODE_WIDGET_HEIGHT,
                    getHeight: () => NODE_WIDGET_HEIGHT,
                    getValue: () => ({ width: state.width, height: state.height }),
                    setValue: (value) => {
                        state = normalizeSize(value, state, minimum, maximum);
                        syncInputs();
                    },
                    onDraw: (w) => {
                        const disabled = Boolean(w.computedDisabled);
                        if (lastDisabled !== disabled) {
                            widthInput.disabled = disabled;
                            heightInput.disabled = disabled;
                            const textColor = disabled ? palette.disabledText : palette.text;
                            widthInput.style.color = textColor;
                            heightInput.style.color = textColor;
                            const labelColor = disabled ? palette.disabledText : palette.subtext;
                            widthField.label.style.color = labelColor;
                            heightField.label.style.color = labelColor;
                            separator.style.color = disabled ? palette.disabledText : palette.subtext;
                            container.style.opacity = disabled ? "0.5" : "1";
                            lastDisabled = disabled;
                        }
                    },
                });

                const commit = () => {
                    const next = normalizeSize(
                        {
                            width: widthInput.value,
                            height: heightInput.value,
                        },
                        state,
                        minimum,
                        maximum,
                    );
                    state = next;
                    syncInputs();
                    widget.callback?.(widget.value);
                };

                widthInput.addEventListener("change", commit);
                heightInput.addEventListener("change", commit);
                widthInput.addEventListener("keydown", (event) => {
                    if (event.key === "Enter") {
                        commit();
                    }
                });
                heightInput.addEventListener("keydown", (event) => {
                    if (event.key === "Enter") {
                        commit();
                    }
                });

                syncInputs();
                return { widget };
            },
        };
    },
});
