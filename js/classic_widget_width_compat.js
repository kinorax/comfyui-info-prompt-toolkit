// Copyright 2026 kinorax

export function clearStaleClassicWidgetWidth(widget) {
    const liteGraph = globalThis.LiteGraph;
    if (liteGraph?.vueNodesMode !== false || widget?.width === undefined) {
        return;
    }

    // Affected ComfyUI frontends leak the Vue wrapper width into the shared widget.
    // Classic rendering must fall back to the owning node width so resize stays aligned.
    widget.width = undefined;
}
