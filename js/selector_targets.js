const RAW_SELECTOR_TARGETS = [
    {
        nodeTypes: ["IPT-CheckpointSelector", "CheckpointSelector", "Checkpoint Selector"],
        selectors: [
            {
                widgetName: "checkpoint",
                folderName: "checkpoints",
                sha256WidgetName: "sha256",
                resultIndex: 1,
                nestedCombo: true,
            },
        ],
    },
    {
        nodeTypes: ["IPT-DiffusionModelSelector", "DiffusionModelSelector", "Diffusion Model Selector"],
        selectors: [
            {
                widgetName: "diffusion_model",
                folderName: "diffusion_models",
                sha256WidgetName: "sha256",
                resultIndex: 1,
                nestedCombo: true,
            },
        ],
    },
    {
        nodeTypes: ["IPT-UnetModelSelector", "UnetModelSelector", "Unet Model Selector"],
        selectors: [
            {
                widgetName: "unet_model",
                folderName: "unet",
                sha256WidgetName: "sha256",
                resultIndex: 1,
                nestedCombo: true,
            },
        ],
    },
    {
        nodeTypes: ["IPT-LoraSelector", "LoraSelector", "Lora Selector"],
        selectors: [
            {
                widgetName: "lora",
                folderName: "loras",
                sha256WidgetName: "sha256",
                resultIndex: 1,
                nestedCombo: true,
            },
        ],
    },
    {
        nodeTypes: ["IPT-VaeSelector", "VaeSelector", "Vae Selector"],
        selectors: [
            {
                widgetName: "vae",
                folderName: "vae",
                sha256WidgetName: "sha256",
                resultIndex: 0,
                nestedCombo: true,
            },
        ],
    },
    {
        nodeTypes: ["IPT-ClipSelector", "ClipSelector", "CLIP Selector"],
        selectors: [
            {
                widgetName: "clip_name",
                folderName: "text_encoders",
                sha256WidgetName: "sha256",
                resultIndex: 1,
                nestedCombo: true,
            },
        ],
    },
    {
        nodeTypes: ["IPT-DualClipSelector", "DualClipSelector", "Dual CLIP Selector"],
        selectors: [
            {
                widgetName: "clip_name1",
                folderName: "text_encoders",
                sha256WidgetName: "sha256_1",
                resultIndex: 1,
                nestedCombo: true,
            },
            {
                widgetName: "clip_name2",
                folderName: "text_encoders",
                sha256WidgetName: "sha256_2",
                resultIndex: 2,
                nestedCombo: true,
            },
        ],
    },
    {
        nodeTypes: ["IPT-TripleClipSelector", "TripleClipSelector", "Triple CLIP Selector"],
        selectors: [
            {
                widgetName: "clip_name1",
                folderName: "text_encoders",
                sha256WidgetName: "sha256_1",
                resultIndex: 1,
                nestedCombo: true,
            },
            {
                widgetName: "clip_name2",
                folderName: "text_encoders",
                sha256WidgetName: "sha256_2",
                resultIndex: 2,
                nestedCombo: true,
            },
            {
                widgetName: "clip_name3",
                folderName: "text_encoders",
                sha256WidgetName: "sha256_3",
                resultIndex: 3,
                nestedCombo: true,
            },
        ],
    },
    {
        nodeTypes: ["IPT-QuadrupleClipSelector", "QuadrupleClipSelector", "Quadruple CLIP Selector"],
        selectors: [
            {
                widgetName: "clip_name1",
                folderName: "text_encoders",
                sha256WidgetName: "sha256_1",
                resultIndex: 1,
                nestedCombo: true,
            },
            {
                widgetName: "clip_name2",
                folderName: "text_encoders",
                sha256WidgetName: "sha256_2",
                resultIndex: 2,
                nestedCombo: true,
            },
            {
                widgetName: "clip_name3",
                folderName: "text_encoders",
                sha256WidgetName: "sha256_3",
                resultIndex: 3,
                nestedCombo: true,
            },
            {
                widgetName: "clip_name4",
                folderName: "text_encoders",
                sha256WidgetName: "sha256_4",
                resultIndex: 4,
                nestedCombo: true,
            },
        ],
    },
];

const RAW_NESTED_COMBO_ONLY_TARGETS = [
    {
        nodeTypes: ["IPT-ImageDirectoryReader", "IPT-BatchImageReader", "BatchImageReader", "ImageDirectoryReader", "Batch Image Reader", "Image Directory Reader"],
        widgetNames: ["path"],
    },
    {
        nodeTypes: ["IPT-CaptionFileSaver", "CaptionFileSaver", "Caption File Saver"],
        widgetNames: ["path"],
    },
];

function cloneSelectors(selectors) {
    if (!Array.isArray(selectors)) {
        return [];
    }
    return selectors.map((selector) => ({ ...selector }));
}

function cloneTarget(rawTarget) {
    return {
        ...rawTarget,
        nodeTypes: new Set(Array.isArray(rawTarget?.nodeTypes) ? rawTarget.nodeTypes : []),
        selectors: cloneSelectors(rawTarget?.selectors),
        widgetNames: Array.isArray(rawTarget?.widgetNames) ? [...rawTarget.widgetNames] : undefined,
    };
}

export const SELECTOR_TARGETS = RAW_SELECTOR_TARGETS.map(cloneTarget);
export const NESTED_COMBO_TARGETS = [
    ...RAW_SELECTOR_TARGETS.map((target) => ({
        nodeTypes: target.nodeTypes,
        widgetNames: cloneSelectors(target.selectors)
            .filter((selector) => selector.nestedCombo !== false)
            .map((selector) => selector.widgetName),
    })).filter((target) => target.widgetNames.length > 0),
    ...RAW_NESTED_COMBO_ONLY_TARGETS,
].map(cloneTarget);

export function getNodeTypeCandidates(nodeOrNodeData) {
    return [
        nodeOrNodeData?.comfyClass,
        nodeOrNodeData?.type,
        nodeOrNodeData?.constructor?.comfyClass,
        nodeOrNodeData?.constructor?.type,
        nodeOrNodeData?.title,
        nodeOrNodeData?.name,
        nodeOrNodeData?.display_name,
        nodeOrNodeData?.node_id,
    ].filter(Boolean);
}

export function resolveTarget(nodeOrNodeData, targets) {
    const candidates = getNodeTypeCandidates(nodeOrNodeData);
    for (const target of targets) {
        if (candidates.some((candidate) => target.nodeTypes.has(candidate))) {
            return target;
        }
    }
    return null;
}

export function resolveSelectorTarget(nodeOrNodeData) {
    return resolveTarget(nodeOrNodeData, SELECTOR_TARGETS);
}

export function resolveNestedComboTarget(nodeOrNodeData) {
    return resolveTarget(nodeOrNodeData, NESTED_COMBO_TARGETS);
}

export function isSelectorTargetNodeDef(nodeData) {
    return resolveSelectorTarget(nodeData) !== null;
}

export function getSelectorSlots(target) {
    return Array.isArray(target?.selectors) ? target.selectors : [];
}

export function getSelectorSlotByWidgetName(target, widgetName) {
    const normalizedWidgetName = String(widgetName ?? "").trim();
    if (!normalizedWidgetName) {
        return null;
    }
    return getSelectorSlots(target).find((slot) => slot.widgetName === normalizedWidgetName) ?? null;
}

export function getNestedTargetWidgetNames(target) {
    if (Array.isArray(target?.widgetNames)) {
        return target.widgetNames;
    }
    return getSelectorSlots(target)
        .filter((slot) => slot.nestedCombo !== false)
        .map((slot) => slot.widgetName);
}

export function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) ?? null;
}

export function normalizeSelectedPath(value) {
    let selected = value;
    if (selected && typeof selected === "object" && "content" in selected) {
        selected = selected.content;
    }
    const text = String(selected ?? "").trim();
    if (!text || text === "None") {
        return "";
    }
    return text;
}

export function normalizeWidgetValue(value) {
    return String(value ?? "").trim();
}

export function normalizeSha256(value) {
    const text = String(value ?? "").trim().toLowerCase();
    if (!/^[0-9a-f]{64}$/.test(text)) {
        return "";
    }
    return text;
}

export function resolvePreferredSelectorSlot(
    node,
    target,
    {
        preferredWidgetName = "",
        allowEmpty = false,
    } = {},
) {
    const preferred = getSelectorSlotByWidgetName(target, preferredWidgetName);
    if (preferred) {
        return preferred;
    }

    const slots = getSelectorSlots(target);
    if (allowEmpty) {
        return slots[0] ?? null;
    }

    for (const slot of slots) {
        const widget = getWidget(node, slot.widgetName);
        if (normalizeSelectedPath(widget?.value)) {
            return slot;
        }
    }

    return slots[0] ?? null;
}
