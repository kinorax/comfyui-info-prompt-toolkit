import { app } from "../../scripts/app.js";

const TARGET_NODE_TYPES = new Set([
    "IPT-AspectRatioToSize",
    "AspectRatioToSize",
    "Aspect Ratio to Size",
]);

const WIDTH_RATIO_WIDGET_NAME = "width_ratio";
const HEIGHT_RATIO_WIDGET_NAME = "height_ratio";
const MIN_UNIT_WIDGET_NAME = "min_unit";
const WIDTH_WIDGET_NAME = "width";
const HEIGHT_WIDGET_NAME = "height";
const ACTUAL_RATIO_WIDGET_NAME = "actual_ratio";

const PATCHED_FLAG = "__iisAspectRatioToSizePatched";
const CALLBACK_PATCHED_FLAG = "__iisAspectRatioToSizeCallbackPatched";
const SYNC_FLAG = "__iisAspectRatioToSizeSyncing";
const LAST_ANCHOR_KEY = "__iisAspectRatioToSizeLastAnchor";

const ANCHOR_WIDTH = "width";
const ANCHOR_HEIGHT = "height";

const MIN_DIMENSION = 8;
const MAX_DIMENSION = 16384;
const UNIT_OPTIONS = new Set([8, 16, 32, 64]);
const DEFAULT_WIDTH_RATIO = 10;
const DEFAULT_HEIGHT_RATIO = 16;
const DEFAULT_MIN_UNIT = 32;
const DEFAULT_WIDTH = 864;
const DEFAULT_HEIGHT = 1376;

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

function removeInputSocketIfPresent(node, name) {
    if (!node?.inputs?.length || typeof node.removeInput !== "function") {
        return;
    }

    const index = node.inputs.findIndex((input) => input?.name === name);
    if (index >= 0) {
        node.removeInput(index);
    }
}

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

function normalizeRatio(value, fallback = 1) {
    const n = toInt(value, fallback);
    return n >= 1 ? n : 1;
}

function normalizeUnit(value, fallback = 8) {
    const n = toInt(value, fallback);
    if (UNIT_OPTIONS.has(n)) {
        return n;
    }
    return fallback;
}

function normalizeRangeBound(value, fallback) {
    const n = Number(value);
    return Number.isFinite(n) ? Math.trunc(n) : fallback;
}

function normalizeDimension(widget, fallback) {
    const minimum = normalizeRangeBound(widget?.options?.min, MIN_DIMENSION);
    const maximum = normalizeRangeBound(widget?.options?.max, MAX_DIMENSION);
    return clamp(toInt(widget?.value, fallback), minimum, maximum);
}

function floorToUnit(value, unit) {
    return Math.floor(value / unit) * unit;
}

function ceilToUnit(value, unit) {
    return Math.ceil(value / unit) * unit;
}

function getUnitBounds(unit, minimum, maximum) {
    const minMultiple = ceilToUnit(minimum, unit);
    const maxMultiple = floorToUnit(maximum, unit);
    if (maxMultiple < minMultiple) {
        return { minimum, maximum };
    }
    return { minimum: minMultiple, maximum: maxMultiple };
}

function normalizeAnchorDimension(value, unit, minimum, maximum) {
    const bounds = getUnitBounds(unit, minimum, maximum);
    const floored = floorToUnit(value, unit);
    return clamp(floored, bounds.minimum, bounds.maximum);
}

function chooseBestCandidate(target, unit, minimum, maximum, errorFn) {
    const floorValue = clamp(floorToUnit(target, unit), minimum, maximum);
    const ceilValue = clamp(ceilToUnit(target, unit), minimum, maximum);
    const candidates = [...new Set([floorValue, ceilValue])];
    if (!candidates.length) {
        return clamp(unit, minimum, maximum);
    }

    candidates.sort((a, b) => {
        const diff = errorFn(a) - errorFn(b);
        if (diff !== 0) {
            return diff;
        }
        return a - b;
    });
    return candidates[0];
}

function resolveFromWidth(width, widthRatio, heightRatio, minUnit, minimum, maximum) {
    const bounds = getUnitBounds(minUnit, minimum, maximum);
    const normalizedWidth = normalizeAnchorDimension(width, minUnit, minimum, maximum);
    const targetHeight = (normalizedWidth * heightRatio) / widthRatio;
    const height = chooseBestCandidate(
        targetHeight,
        minUnit,
        bounds.minimum,
        bounds.maximum,
        (candidate) => Math.abs((normalizedWidth * heightRatio) - (candidate * widthRatio)),
    );
    return { width: normalizedWidth, height };
}

function resolveFromHeight(height, widthRatio, heightRatio, minUnit, minimum, maximum) {
    const bounds = getUnitBounds(minUnit, minimum, maximum);
    const normalizedHeight = normalizeAnchorDimension(height, minUnit, minimum, maximum);
    const targetWidth = (normalizedHeight * widthRatio) / heightRatio;
    const width = chooseBestCandidate(
        targetWidth,
        minUnit,
        bounds.minimum,
        bounds.maximum,
        (candidate) => Math.abs((candidate * heightRatio) - (normalizedHeight * widthRatio)),
    );
    return { width, height: normalizedHeight };
}

function inferAnchor(width, height, widthRatio, heightRatio, minUnit, minimum, maximum) {
    const fromWidth = resolveFromWidth(width, widthRatio, heightRatio, minUnit, minimum, maximum);
    const fromHeight = resolveFromHeight(height, widthRatio, heightRatio, minUnit, minimum, maximum);
    const widthDelta = Math.abs(fromHeight.width - width);
    const heightDelta = Math.abs(fromWidth.height - height);
    if (heightDelta < widthDelta) {
        return ANCHOR_WIDTH;
    }
    if (widthDelta < heightDelta) {
        return ANCHOR_HEIGHT;
    }
    return ANCHOR_WIDTH;
}

function formatRatioValue(value, decimals = 4) {
    if (!Number.isFinite(value)) {
        return null;
    }

    const rounded = Math.round(value);
    if (Math.abs(value - rounded) < 1e-9) {
        return String(rounded);
    }

    const fixed = value.toFixed(decimals).replace(/\.?0+$/, "");
    return fixed || "0";
}

function actualRatioText(widthRatio, width, height) {
    if (width <= 0 || height <= 0) {
        return "-";
    }
    const ratioHeight = (height * widthRatio) / width;
    const rendered = formatRatioValue(ratioHeight);
    if (!rendered) {
        return "-";
    }
    return `${widthRatio} : ${rendered}`;
}

function applyReadOnly(widget) {
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

function applyDimensionStep(widget, minUnit) {
    if (!widget) {
        return;
    }

    widget.options = widget.options ?? {};
    widget.options.step = minUnit;
    widget.options.step2 = minUnit;
    widget.options.round = minUnit;
}

function patchWidgetCallback(node, widgetName, callback) {
    const widget = getWidget(node, widgetName);
    if (!widget || widget[CALLBACK_PATCHED_FLAG]) {
        return;
    }

    widget.callback = chainCallback(widget.callback, callback);
    widget[CALLBACK_PATCHED_FLAG] = true;
}

function syncNode(node, anchor = null) {
    if (!node || node[SYNC_FLAG]) {
        return;
    }
    node[SYNC_FLAG] = true;

    try {
        const widthRatioWidget = getWidget(node, WIDTH_RATIO_WIDGET_NAME);
        const heightRatioWidget = getWidget(node, HEIGHT_RATIO_WIDGET_NAME);
        const minUnitWidget = getWidget(node, MIN_UNIT_WIDGET_NAME);
        const widthWidget = getWidget(node, WIDTH_WIDGET_NAME);
        const heightWidget = getWidget(node, HEIGHT_WIDGET_NAME);
        const actualRatioWidget = getWidget(node, ACTUAL_RATIO_WIDGET_NAME);
        if (!widthRatioWidget || !heightRatioWidget || !minUnitWidget || !widthWidget || !heightWidget || !actualRatioWidget) {
            return;
        }

        const widthRatio = normalizeRatio(widthRatioWidget.value, DEFAULT_WIDTH_RATIO);
        const heightRatio = normalizeRatio(heightRatioWidget.value, DEFAULT_HEIGHT_RATIO);
        const minUnit = normalizeUnit(minUnitWidget.value, DEFAULT_MIN_UNIT);
        const width = normalizeDimension(widthWidget, DEFAULT_WIDTH);
        const height = normalizeDimension(heightWidget, DEFAULT_HEIGHT);

        widthRatioWidget.value = widthRatio;
        heightRatioWidget.value = heightRatio;
        minUnitWidget.value = String(minUnit);

        applyDimensionStep(widthWidget, minUnit);
        applyDimensionStep(heightWidget, minUnit);

        const minimum = Math.max(
            normalizeRangeBound(widthWidget.options?.min, MIN_DIMENSION),
            normalizeRangeBound(heightWidget.options?.min, MIN_DIMENSION),
        );
        const maximum = Math.min(
            normalizeRangeBound(widthWidget.options?.max, MAX_DIMENSION),
            normalizeRangeBound(heightWidget.options?.max, MAX_DIMENSION),
        );

        const resolvedAnchor = anchor ?? node[LAST_ANCHOR_KEY] ?? inferAnchor(
            width,
            height,
            widthRatio,
            heightRatio,
            minUnit,
            minimum,
            maximum,
        );

        const resolved = resolvedAnchor === ANCHOR_HEIGHT
            ? resolveFromHeight(height, widthRatio, heightRatio, minUnit, minimum, maximum)
            : resolveFromWidth(width, widthRatio, heightRatio, minUnit, minimum, maximum);

        node[LAST_ANCHOR_KEY] = resolvedAnchor;
        widthWidget.value = resolved.width;
        heightWidget.value = resolved.height;

        applyReadOnly(actualRatioWidget);
        actualRatioWidget.value = actualRatioText(widthRatio, resolved.width, resolved.height);
        actualRatioWidget.callback?.(actualRatioWidget.value);

        removeInputSocketIfPresent(node, WIDTH_RATIO_WIDGET_NAME);
        removeInputSocketIfPresent(node, HEIGHT_RATIO_WIDGET_NAME);
        removeInputSocketIfPresent(node, MIN_UNIT_WIDGET_NAME);
        removeInputSocketIfPresent(node, WIDTH_WIDGET_NAME);
        removeInputSocketIfPresent(node, HEIGHT_WIDGET_NAME);
        removeInputSocketIfPresent(node, ACTUAL_RATIO_WIDGET_NAME);

        node.setDirtyCanvas?.(true, true);
    } finally {
        node[SYNC_FLAG] = false;
    }
}

function patchNode(node) {
    if (!isTargetNode(node)) {
        return;
    }

    if (!node[PATCHED_FLAG]) {
        patchWidgetCallback(node, WIDTH_WIDGET_NAME, () => {
            node[LAST_ANCHOR_KEY] = ANCHOR_WIDTH;
            syncNode(node, ANCHOR_WIDTH);
        });

        patchWidgetCallback(node, HEIGHT_WIDGET_NAME, () => {
            node[LAST_ANCHOR_KEY] = ANCHOR_HEIGHT;
            syncNode(node, ANCHOR_HEIGHT);
        });

        patchWidgetCallback(node, WIDTH_RATIO_WIDGET_NAME, () => {
            syncNode(node);
        });

        patchWidgetCallback(node, HEIGHT_RATIO_WIDGET_NAME, () => {
            syncNode(node);
        });

        patchWidgetCallback(node, MIN_UNIT_WIDGET_NAME, () => {
            syncNode(node);
        });

        node[PATCHED_FLAG] = true;
    }

    syncNode(node);
}

app.registerExtension({
    name: "IPT.AspectRatioToSize",
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
    },
    nodeCreated(node) {
        patchNode(node);
    },
    loadedGraphNode(node) {
        patchNode(node);
    },
});
