// Copyright 2026 kinorax
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SETTINGS_ENDPOINT = "/ipt/settings";
const SYNC_DEBOUNCE_MS = 250;

const SETTING_IDS = {
    cacheRetention: "InfoPromptToolkit.UseLoadedModelCache.99.cacheRetention",
    maxCacheEntries: "InfoPromptToolkit.UseLoadedModelCache.98.maxCacheEntries",
    memoryBudgetRatio: "InfoPromptToolkit.UseLoadedModelCache.97.memoryBudgetRatio",
    cacheLoggingEnabled: "InfoPromptToolkit.UseLoadedModelCache.01.cacheLoggingEnabled",
};

const LEGACY_SETTING_IDS = {
    cacheRetention: [
        "InfoPromptToolkit.UseLoadedModelCache.01.cacheRetention",
        "InfoPromptToolkit.UseLoadedModelCache.cacheRetention",
    ],
    maxCacheEntries: [
        "InfoPromptToolkit.UseLoadedModelCache.02.maxCacheEntries",
        "InfoPromptToolkit.UseLoadedModelCache.maxCacheEntries",
    ],
    memoryBudgetRatio: [
        "InfoPromptToolkit.UseLoadedModelCache.03.memoryBudgetRatio",
        "InfoPromptToolkit.UseLoadedModelCache.memoryBudgetRatio",
    ],
    cacheLogLevel: [
        "InfoPromptToolkit.UseLoadedModelCache.cacheLogLevel",
    ],
};

const DEFAULTS = {
    cacheRetention: "auto",
    maxCacheEntries: 1,
    memoryBudgetRatio: 0.72,
    cacheLoggingEnabled: true,
};

let syncTimer = null;

function clampNumber(value, defaultValue, min, max) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
        return defaultValue;
    }
    return Math.min(max, Math.max(min, number));
}

function readSetting(id, defaultValue) {
    try {
        const settingService = app?.extensionManager?.setting;
        if (settingService && typeof settingService.get === "function") {
            const value = settingService.get(id);
            return value == null ? defaultValue : value;
        }
    } catch (error) {
        console.warn("[IPT.Settings] failed to read setting", id, error);
    }
    return defaultValue;
}

function boolOrDefault(value, defaultValue) {
    if (value == null) {
        return defaultValue;
    }
    if (typeof value === "boolean") {
        return value;
    }
    if (typeof value === "number") {
        return Boolean(value);
    }

    const text = String(value).trim().toLowerCase();
    if (["1", "true", "yes", "on"].includes(text)) {
        return true;
    }
    if (["0", "false", "no", "off"].includes(text)) {
        return false;
    }
    return defaultValue;
}

function readSettingWithLegacy(id, legacyIds, defaultValue) {
    const value = readSetting(id, undefined);
    if (value != null) {
        return value;
    }

    for (const legacyId of legacyIds) {
        const legacyValue = readSetting(legacyId, undefined);
        if (legacyValue != null) {
            return legacyValue;
        }
    }
    return defaultValue;
}

function readCacheLoggingEnabled() {
    const value = readSetting(SETTING_IDS.cacheLoggingEnabled, undefined);
    if (value != null) {
        return boolOrDefault(value, DEFAULTS.cacheLoggingEnabled);
    }

    const legacyLevel = String(readSettingWithLegacy("", LEGACY_SETTING_IDS.cacheLogLevel, "summary") || "").trim().toLowerCase();
    return legacyLevel !== "off";
}

function buildPayload() {
    return {
        use_loaded_model_cache: {
            cache_retention: String(
                readSettingWithLegacy(SETTING_IDS.cacheRetention, LEGACY_SETTING_IDS.cacheRetention, DEFAULTS.cacheRetention),
            ),
            max_cache_entries: Math.round(
                clampNumber(
                    readSettingWithLegacy(SETTING_IDS.maxCacheEntries, LEGACY_SETTING_IDS.maxCacheEntries, DEFAULTS.maxCacheEntries),
                    DEFAULTS.maxCacheEntries,
                    1,
                    9,
                ),
            ),
            memory_budget_ratio: clampNumber(
                readSettingWithLegacy(SETTING_IDS.memoryBudgetRatio, LEGACY_SETTING_IDS.memoryBudgetRatio, DEFAULTS.memoryBudgetRatio),
                DEFAULTS.memoryBudgetRatio,
                0.01,
                1,
            ),
            cache_log_enabled: readCacheLoggingEnabled(),
        },
    };
}

async function syncSettings() {
    if (!api || typeof api.fetchApi !== "function") {
        return;
    }

    try {
        await api.fetchApi(SETTINGS_ENDPOINT, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(buildPayload()),
        });
    } catch (error) {
        console.warn("[IPT.Settings] failed to sync settings", error);
    }
}

function scheduleSettingsSync() {
    if (syncTimer != null) {
        window.clearTimeout(syncTimer);
    }
    syncTimer = window.setTimeout(() => {
        syncTimer = null;
        void syncSettings();
    }, SYNC_DEBOUNCE_MS);
}

const cacheCategory = ["Info-Prompt-Toolkit", "Use Loaded Model Cache"];
const categoryOrder = {
    cacheRetention: "99 Cache retention",
    maxCacheEntries: "98 Max cache entries",
    memoryBudgetRatio: "97 Memory budget ratio",
    cacheLoggingEnabled: "01 Use cache logging",
};

app.registerExtension({
    name: "IPT.Settings",
    settings: [
        {
            id: SETTING_IDS.cacheLoggingEnabled,
            name: "Use cache logging",
            type: "boolean",
            defaultValue: DEFAULTS.cacheLoggingEnabled,
            category: [...cacheCategory, categoryOrder.cacheLoggingEnabled],
            tooltip: "Print Use Loaded Model cache messages to the ComfyUI console.",
            onChange: scheduleSettingsSync,
        },
        {
            id: SETTING_IDS.memoryBudgetRatio,
            name: "Memory budget ratio",
            type: "number",
            defaultValue: DEFAULTS.memoryBudgetRatio,
            attrs: {
                min: 0.01,
                max: 1,
                step: 0.01,
                showButtons: true,
            },
            category: [...cacheCategory, categoryOrder.memoryBudgetRatio],
            tooltip: "Auto keeps cached runtimes only while their estimated total size fits within this ratio of detected total memory.",
            onChange: scheduleSettingsSync,
        },
        {
            id: SETTING_IDS.maxCacheEntries,
            name: "Max cache entries",
            type: "number",
            defaultValue: DEFAULTS.maxCacheEntries,
            attrs: {
                min: 1,
                max: 9,
                step: 1,
                showButtons: true,
            },
            category: [...cacheCategory, categoryOrder.maxCacheEntries],
            tooltip: "Maximum number of Use Loaded Model runtime bundles to keep. The latest entry is always kept.",
            onChange: scheduleSettingsSync,
        },
        {
            id: SETTING_IDS.cacheRetention,
            name: "Cache retention",
            type: "combo",
            defaultValue: DEFAULTS.cacheRetention,
            options: [
                { text: "Auto", value: "auto" },
                { text: "Fixed", value: "fixed" },
            ],
            category: [...cacheCategory, categoryOrder.cacheRetention],
            tooltip: "Controls whether Use Loaded Model prunes cached runtimes by estimated memory usage or keeps a fixed count.",
            onChange: scheduleSettingsSync,
        },
    ],
    setup() {
        scheduleSettingsSync();
    },
});
