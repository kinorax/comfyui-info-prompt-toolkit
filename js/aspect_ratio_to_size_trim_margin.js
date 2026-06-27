// Copyright 2026 kinorax
import { app } from "../../scripts/app.js";

const TARGET_NODE_TYPES = new Set([
    "IPT-AspectRatioToSizeTrimMargin",
    "AspectRatioToSizeTrimMargin",
    "Aspect Ratio to Size (Trim Margin)",
]);

const WIDTH_RATIO_WIDGET_NAME = "width_ratio";
const HEIGHT_RATIO_WIDGET_NAME = "height_ratio";
const MIN_UNIT_WIDGET_NAME = "min_unit";
const MARGIN_TOP_WIDGET_NAME = "margin_top";
const MARGIN_RIGHT_WIDGET_NAME = "margin_right";
const MARGIN_BOTTOM_WIDGET_NAME = "margin_bottom";
const MARGIN_LEFT_WIDGET_NAME = "margin_left";
const WIDTH_WIDGET_NAME = "width";
const HEIGHT_WIDGET_NAME = "height";
const ACTUAL_WIDTH_WIDGET_NAME = "actual_width";
const ACTUAL_HEIGHT_WIDGET_NAME = "actual_height";
const ACTUAL_RATIO_WIDGET_NAME = "actual_ratio";

const PATCHED_FLAG = "__iptAspectRatioTrimMarginPatched";
const CALLBACK_PATCHED_FLAG = "__iptAspectRatioTrimMarginCallbackPatched";
const SYNC_FLAG = "__iptAspectRatioTrimMarginSyncing";
const LAST_ANCHOR_KEY = "__iptAspectRatioTrimMarginLastAnchor";

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
const LEGACY_WIDGET_VALUE_COUNT = 10;

function getNodeTypeCandidates(node) {
    return [node?.comfyClass, node?.type, node?.constructor?.comfyClass, node?.constructor?.type, node?.title].filter(Boolean);
}

function isTargetNode(node) {
    return getNodeTypeCandidates(node).some((candidate) => TARGET_NODE_TYPES.has(candidate));
}

function isTargetNodeDef(nodeData) {
    return [nodeData?.name, nodeData?.display_name, nodeData?.type, nodeData?.node_id]
        .filter(Boolean)
        .some((candidate) => TARGET_NODE_TYPES.has(candidate));
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
    const parsed = Number(value);
    return Number.isFinite(parsed) ? Math.trunc(parsed) : Math.trunc(fallback);
}

function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
}

function normalizeRatio(value, fallback) {
    return Math.max(1, toInt(value, fallback));
}

function normalizeUnit(value) {
    const parsed = toInt(value, DEFAULT_MIN_UNIT);
    return UNIT_OPTIONS.has(parsed) ? parsed : DEFAULT_MIN_UNIT;
}

function normalizeRangeBound(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? Math.trunc(parsed) : fallback;
}

function normalizeDimension(widget, fallback) {
    const minimum = normalizeRangeBound(widget?.options?.min, MIN_DIMENSION);
    const maximum = normalizeRangeBound(widget?.options?.max, MAX_DIMENSION);
    return clamp(toInt(widget?.value, fallback), minimum, maximum);
}

function applyMarginOptions(widget, minUnit) {
    if (!widget) {
        return;
    }
    widget.options = widget.options ?? {};
    widget.options.min = 0;
    widget.options.max = minUnit;
    widget.options.step = 1;
    widget.options.step2 = 1;
    widget.options.round = 1;
}

function normalizeMarginPair(firstWidget, secondWidget, minUnit, changedMarginName = null) {
    applyMarginOptions(firstWidget, minUnit);
    applyMarginOptions(secondWidget, minUnit);

    const rawFirst = Math.max(0, toInt(firstWidget?.value, 0));
    const rawSecond = Math.max(0, toInt(secondWidget?.value, 0));
    let first = clamp(rawFirst, 0, minUnit);
    let second = clamp(rawSecond, 0, minUnit);

    if (changedMarginName === firstWidget?.name) {
        second = first === 0 ? 0 : minUnit - first;
    } else if (changedMarginName === secondWidget?.name) {
        first = second === 0 ? 0 : minUnit - second;
    } else {
        const total = rawFirst + rawSecond;
        if (total === 0) {
            first = 0;
            second = 0;
        } else if (total !== minUnit || rawFirst > minUnit || rawSecond > minUnit) {
            first = clamp(Math.round((rawFirst * minUnit) / total), 0, minUnit);
            second = minUnit - first;
        }
    }

    firstWidget.value = first;
    secondWidget.value = second;
    return { first, second, total: first + second };
}

function floorToUnit(value, unit) {
    return Math.floor(value / unit) * unit;
}

function ceilToUnit(value, unit) {
    return Math.ceil(value / unit) * unit;
}

function getSamplingBounds(marginTotal, unit, minimum, maximum) {
    return {
        minimum: ceilToUnit(minimum + marginTotal, unit),
        maximum: floorToUnit(maximum, unit),
    };
}

function normalizeAnchorContent(value, marginTotal, unit, minimum, maximum) {
    const bounds = getSamplingBounds(marginTotal, unit, minimum, maximum);
    const maximumContent = bounds.maximum - marginTotal;
    const content = clamp(value, minimum, maximumContent);
    const sampling = clamp(floorToUnit(content + marginTotal, unit), bounds.minimum, bounds.maximum);
    return sampling - marginTotal;
}

function chooseBestContentCandidate(targetContent, marginTotal, unit, minimum, maximum, errorFn) {
    const bounds = getSamplingBounds(marginTotal, unit, minimum, maximum);
    const targetSampling = targetContent + marginTotal;
    const candidates = [...new Set([
        clamp(floorToUnit(targetSampling, unit), bounds.minimum, bounds.maximum) - marginTotal,
        clamp(ceilToUnit(targetSampling, unit), bounds.minimum, bounds.maximum) - marginTotal,
    ])];
    candidates.sort((a, b) => {
        const difference = errorFn(a) - errorFn(b);
        return difference !== 0 ? difference : a - b;
    });
    return candidates[0];
}

function resolveFromWidth(width, widthRatio, heightRatio, minUnit, horizontalMargin, verticalMargin, minimum, maximum) {
    const normalizedWidth = normalizeAnchorContent(width, horizontalMargin, minUnit, minimum, maximum);
    const targetHeight = (normalizedWidth * heightRatio) / widthRatio;
    const height = chooseBestContentCandidate(
        targetHeight,
        verticalMargin,
        minUnit,
        minimum,
        maximum,
        (candidate) => Math.abs((normalizedWidth * heightRatio) - (candidate * widthRatio)),
    );
    return { width: normalizedWidth, height };
}

function resolveFromHeight(height, widthRatio, heightRatio, minUnit, horizontalMargin, verticalMargin, minimum, maximum) {
    const normalizedHeight = normalizeAnchorContent(height, verticalMargin, minUnit, minimum, maximum);
    const targetWidth = (normalizedHeight * widthRatio) / heightRatio;
    const width = chooseBestContentCandidate(
        targetWidth,
        horizontalMargin,
        minUnit,
        minimum,
        maximum,
        (candidate) => Math.abs((candidate * heightRatio) - (normalizedHeight * widthRatio)),
    );
    return { width, height: normalizedHeight };
}

function inferAnchor(width, height, widthRatio, heightRatio, minUnit, horizontalMargin, verticalMargin, minimum, maximum) {
    const widthBounds = getSamplingBounds(horizontalMargin, minUnit, minimum, maximum);
    const heightBounds = getSamplingBounds(verticalMargin, minUnit, minimum, maximum);
    const normalizedWidth = clamp(width, minimum, widthBounds.maximum - horizontalMargin);
    const normalizedHeight = clamp(height, minimum, heightBounds.maximum - verticalMargin);
    const fromWidth = resolveFromWidth(normalizedWidth, widthRatio, heightRatio, minUnit, horizontalMargin, verticalMargin, minimum, maximum);
    const fromHeight = resolveFromHeight(normalizedHeight, widthRatio, heightRatio, minUnit, horizontalMargin, verticalMargin, minimum, maximum);
    const widthDelta = Math.abs(fromHeight.width - normalizedWidth);
    const heightDelta = Math.abs(fromWidth.height - normalizedHeight);
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
    return value.toFixed(decimals).replace(/\.?0+$/, "") || "0";
}

function actualRatioText(widthRatio, width, height) {
    if (width <= 0 || height <= 0) {
        return "-";
    }
    const rendered = formatRatioValue((height * widthRatio) / width);
    return rendered ? `${widthRatio} : ${rendered}` : "-";
}

function applyReadOnly(widget, disabled = false) {
    if (!widget) {
        return;
    }
    widget.options = widget.options ?? {};
    widget.options.readonly = true;
    widget.options.read_only = true;
    widget.options.disabled = disabled;
    const element = widget.element ?? widget.inputEl ?? null;
    if (element && typeof element === "object") {
        if ("readOnly" in element) {
            element.readOnly = true;
        }
        if ("disabled" in element) {
            element.disabled = disabled;
        }
    }
}

function applyActualDimensionOptions(widget, marginTotal, minUnit) {
    if (!widget) {
        return;
    }
    const bounds = getSamplingBounds(marginTotal, minUnit, MIN_DIMENSION, MAX_DIMENSION);
    widget.options = widget.options ?? {};
    widget.options.min = bounds.minimum - marginTotal;
    widget.options.max = bounds.maximum - marginTotal;
    widget.options.step = minUnit;
    widget.options.step2 = minUnit;
    widget.options.round = 1;
}

function migrateLegacyWidgetValues(node, config) {
    const values = config?.widgets_values;
    if (!Array.isArray(values) || values.length !== LEGACY_WIDGET_VALUE_COUNT) {
        return;
    }
    const top = toInt(values[3], 0);
    const right = toInt(values[4], 0);
    const bottom = toInt(values[5], 0);
    const left = toInt(values[6], 0);
    const actualWidth = toInt(values[7], DEFAULT_WIDTH);
    const actualHeight = toInt(values[8], DEFAULT_HEIGHT);
    values.splice(
        0,
        values.length,
        ...values.slice(0, 7),
        actualWidth + left + right,
        actualHeight + top + bottom,
        actualWidth,
        actualHeight,
        values[9],
    );
    let valueIndex = 0;
    for (const widget of node?.widgets ?? []) {
        if (widget?.serialize === false) {
            continue;
        }
        if (valueIndex >= values.length) {
            break;
        }
        widget.value = values[valueIndex++];
    }
}

function patchWidgetCallback(node, widgetName, callback) {
    const widget = getWidget(node, widgetName);
    if (!widget || widget[CALLBACK_PATCHED_FLAG]) {
        return;
    }
    widget.callback = chainCallback(widget.callback, callback);
    widget[CALLBACK_PATCHED_FLAG] = true;
}

function syncNode(node, anchor = null, changedMarginName = null) {
    if (!node || node[SYNC_FLAG]) {
        return;
    }
    node[SYNC_FLAG] = true;
    try {
        const widthRatioWidget = getWidget(node, WIDTH_RATIO_WIDGET_NAME);
        const heightRatioWidget = getWidget(node, HEIGHT_RATIO_WIDGET_NAME);
        const minUnitWidget = getWidget(node, MIN_UNIT_WIDGET_NAME);
        const marginTopWidget = getWidget(node, MARGIN_TOP_WIDGET_NAME);
        const marginRightWidget = getWidget(node, MARGIN_RIGHT_WIDGET_NAME);
        const marginBottomWidget = getWidget(node, MARGIN_BOTTOM_WIDGET_NAME);
        const marginLeftWidget = getWidget(node, MARGIN_LEFT_WIDGET_NAME);
        const widthWidget = getWidget(node, WIDTH_WIDGET_NAME);
        const heightWidget = getWidget(node, HEIGHT_WIDGET_NAME);
        const actualWidthWidget = getWidget(node, ACTUAL_WIDTH_WIDGET_NAME);
        const actualHeightWidget = getWidget(node, ACTUAL_HEIGHT_WIDGET_NAME);
        const actualRatioWidget = getWidget(node, ACTUAL_RATIO_WIDGET_NAME);
        const required = [
            widthRatioWidget, heightRatioWidget, minUnitWidget,
            marginTopWidget, marginRightWidget, marginBottomWidget, marginLeftWidget,
            widthWidget, heightWidget, actualWidthWidget, actualHeightWidget, actualRatioWidget,
        ];
        if (required.some((widget) => !widget)) {
            return;
        }

        const widthRatio = normalizeRatio(widthRatioWidget.value, DEFAULT_WIDTH_RATIO);
        const heightRatio = normalizeRatio(heightRatioWidget.value, DEFAULT_HEIGHT_RATIO);
        const minUnit = normalizeUnit(minUnitWidget.value);
        const verticalMargins = normalizeMarginPair(marginTopWidget, marginBottomWidget, minUnit, changedMarginName);
        const horizontalMargins = normalizeMarginPair(marginLeftWidget, marginRightWidget, minUnit, changedMarginName);
        applyActualDimensionOptions(actualWidthWidget, horizontalMargins.total, minUnit);
        applyActualDimensionOptions(actualHeightWidget, verticalMargins.total, minUnit);
        const actualWidth = normalizeDimension(actualWidthWidget, DEFAULT_WIDTH);
        const actualHeight = normalizeDimension(actualHeightWidget, DEFAULT_HEIGHT);

        widthRatioWidget.value = widthRatio;
        heightRatioWidget.value = heightRatio;
        minUnitWidget.value = String(minUnit);
        const resolvedAnchor = anchor ?? node[LAST_ANCHOR_KEY] ?? inferAnchor(
            actualWidth,
            actualHeight,
            widthRatio,
            heightRatio,
            minUnit,
            horizontalMargins.total,
            verticalMargins.total,
            MIN_DIMENSION,
            MAX_DIMENSION,
        );
        const resolved = resolvedAnchor === ANCHOR_HEIGHT
            ? resolveFromHeight(actualHeight, widthRatio, heightRatio, minUnit, horizontalMargins.total, verticalMargins.total, MIN_DIMENSION, MAX_DIMENSION)
            : resolveFromWidth(actualWidth, widthRatio, heightRatio, minUnit, horizontalMargins.total, verticalMargins.total, MIN_DIMENSION, MAX_DIMENSION);

        node[LAST_ANCHOR_KEY] = resolvedAnchor;
        actualWidthWidget.value = resolved.width;
        actualHeightWidget.value = resolved.height;
        widthWidget.value = resolved.width + horizontalMargins.total;
        heightWidget.value = resolved.height + verticalMargins.total;
        applyReadOnly(widthWidget, true);
        applyReadOnly(heightWidget, true);
        applyReadOnly(actualRatioWidget);
        actualRatioWidget.value = actualRatioText(widthRatio, resolved.width, resolved.height);
        actualRatioWidget.callback?.(actualRatioWidget.value);

        for (const name of [
            WIDTH_RATIO_WIDGET_NAME, HEIGHT_RATIO_WIDGET_NAME, MIN_UNIT_WIDGET_NAME,
            MARGIN_TOP_WIDGET_NAME, MARGIN_RIGHT_WIDGET_NAME, MARGIN_BOTTOM_WIDGET_NAME, MARGIN_LEFT_WIDGET_NAME,
            WIDTH_WIDGET_NAME, HEIGHT_WIDGET_NAME, ACTUAL_WIDTH_WIDGET_NAME, ACTUAL_HEIGHT_WIDGET_NAME,
            ACTUAL_RATIO_WIDGET_NAME,
        ]) {
            removeInputSocketIfPresent(node, name);
        }
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
        patchWidgetCallback(node, ACTUAL_WIDTH_WIDGET_NAME, () => {
            node[LAST_ANCHOR_KEY] = ANCHOR_WIDTH;
            syncNode(node, ANCHOR_WIDTH);
        });
        patchWidgetCallback(node, ACTUAL_HEIGHT_WIDGET_NAME, () => {
            node[LAST_ANCHOR_KEY] = ANCHOR_HEIGHT;
            syncNode(node, ANCHOR_HEIGHT);
        });
        patchWidgetCallback(node, WIDTH_WIDGET_NAME, () => syncNode(node));
        patchWidgetCallback(node, HEIGHT_WIDGET_NAME, () => syncNode(node));
        for (const name of [
            WIDTH_RATIO_WIDGET_NAME, HEIGHT_RATIO_WIDGET_NAME, MIN_UNIT_WIDGET_NAME,
        ]) {
            patchWidgetCallback(node, name, () => syncNode(node));
        }
        for (const name of [
            MARGIN_TOP_WIDGET_NAME, MARGIN_RIGHT_WIDGET_NAME, MARGIN_BOTTOM_WIDGET_NAME, MARGIN_LEFT_WIDGET_NAME,
        ]) {
            patchWidgetCallback(node, name, () => syncNode(node, null, name));
        }
        node[PATCHED_FLAG] = true;
    }
    syncNode(node);
}

app.registerExtension({
    name: "IPT.AspectRatioToSizeTrimMargin",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isTargetNodeDef(nodeData)) {
            return;
        }
        nodeType.prototype.onNodeCreated = chainCallback(nodeType.prototype.onNodeCreated, function onNodeCreated() {
            patchNode(this);
        });
        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function onConfigure(config, ...args) {
            migrateLegacyWidgetValues(this, config);
            const result = originalOnConfigure?.call(this, config, ...args);
            patchNode(this);
            return result;
        };
        nodeType.prototype.onGraphConfigured = chainCallback(nodeType.prototype.onGraphConfigured, function onGraphConfigured() {
            patchNode(this);
        });
    },
    nodeCreated(node) {
        patchNode(node);
    },
    loadedGraphNode(node) {
        patchNode(node);
    },
});
