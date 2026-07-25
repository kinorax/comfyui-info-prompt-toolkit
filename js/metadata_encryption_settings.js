// Copyright 2026 kinorax
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SETTING_ID = "InfoPromptToolkit.MetadataEncryption.99.keys";
const SETTINGS_ENDPOINT = "/ipt/settings";
const PROTECT_ENDPOINT = "/ipt/protected-settings/protect";
const UNPROTECT_ENDPOINT = "/ipt/protected-settings/unprotect";
const RANDOM_KEY_ENDPOINT = "/ipt/metadata-encryption/random-key";
const DEVICE_DEFAULT_ENDPOINT = "/ipt/metadata-encryption/device-default";

const ENVELOPE_FORMAT = "ipt.protected-setting";
const KEY_PATTERN = /^[A-Za-z0-9_-]{43}$/;

const TEXT = {
    en: {
        loading: "Loading encryption keys…",
        primary: "Primary encryption key",
        fallback: "Fallback decryption key",
        fallbackHelp: "Used only when the primary key cannot decrypt the metadata.",
        deviceDerived: "Device-derived",
        custom: "Custom",
        randomFallback: "Random fallback",
        notSet: "Not set",
        show: "Show",
        hide: "Hide",
        copy: "Copy",
        generate: "Generate random key",
        restore: "Restore device default",
        clearFallback: "Clear fallback key",
        cancel: "Cancel",
        apply: "Apply",
        retrySync: "Synchronize selected copy",
        useComfy: "Use ComfyUI copy",
        useBackend: "Use backend copy",
        invalidPrimary: "Enter a valid AES-256 key. (43-character Base64URL)",
        invalidFallback: "Enter a valid AES-256 key. (43-character Base64URL)",
        duplicate: "The fallback key must differ from the primary key.",
        copied: "Encryption key copied.",
        copyFailed: "Could not copy the key.",
        loadFailed: "Could not load the encryption keys. The saved value was not changed.",
        saveFailed: "Could not save both copies of the encryption keys. Retry synchronization.",
        saved: "Encryption keys saved.",
        syncWarning: "The ComfyUI and backend copies differ. Choose the copy to keep, then synchronize it.",
        oneCopyWarning: "Only one valid saved copy was found. Synchronize it before changing the keys.",
        conflictWarning: "Both saved copies have the same revision but different contents.",
    },
    ja: {
        loading: "暗号化鍵を読み込んでいます…",
        primary: "暗号化・復号の主鍵",
        fallback: "復号用の予備鍵",
        fallbackHelp: "主鍵で復号できなかった場合だけ使用します。",
        deviceDerived: "端末由来",
        custom: "カスタム",
        randomFallback: "ランダムfallback",
        notSet: "未設定",
        show: "表示",
        hide: "隠す",
        copy: "コピー",
        generate: "ランダム鍵を生成",
        restore: "端末既定値に戻す",
        clearFallback: "予備鍵を削除",
        cancel: "取消",
        apply: "適用",
        retrySync: "選択した値を同期",
        useComfy: "ComfyUI側を使用",
        useBackend: "backend側を使用",
        invalidPrimary: "有効なAES-256鍵を入力してください。（Base64URL形式、43文字）",
        invalidFallback: "有効なAES-256鍵を入力してください。（Base64URL形式、43文字）",
        duplicate: "予備鍵には主鍵と異なる鍵を入力してください。",
        copied: "暗号化鍵をコピーしました。",
        copyFailed: "鍵をコピーできませんでした。",
        loadFailed: "暗号化鍵を読み込めませんでした。保存値は変更していません。",
        saveFailed: "暗号化鍵を両方の保存先へ保存できませんでした。同期を再試行してください。",
        saved: "暗号化鍵を保存しました。",
        syncWarning: "ComfyUI側とbackend側の保存値が異なります。残す値を選び、同期してください。",
        oneCopyWarning: "有効な保存値が片方にだけあります。鍵を変更する前に同期してください。",
        conflictWarning: "同じrevisionの異なる保存値が見つかりました。",
    },
};

function installStyles() {
    if (document.getElementById("ipt-metadata-encryption-settings-style")) {
        return;
    }
    const style = document.createElement("style");
    style.id = "ipt-metadata-encryption-settings-style";
    style.textContent = `
        .ipt-key-settings__form-row { display: grid !important; grid-template-columns: minmax(4.5rem, 6rem) minmax(0, 1fr); align-items: start !important; min-width: 0; }
        .ipt-key-settings__form-label, .ipt-key-settings__form-input, .ipt-key-settings__host { min-width: 0; }
        .ipt-key-settings__form-input { width: 100%; justify-content: stretch !important; }
        .ipt-key-settings__host { width: 100%; container-type: inline-size; }
        .ipt-key-settings { width: 100%; min-width: 0; max-width: 42rem; display: grid; gap: 0.8rem; overflow-wrap: anywhere; }
        .ipt-key-settings * { box-sizing: border-box; }
        .ipt-key-settings__section { display: grid; gap: 0.35rem; padding: 0.65rem; border: 1px solid var(--border-color, #5555); border-radius: 0.5rem; }
        .ipt-key-settings__heading { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 0.5rem; font-weight: 600; }
        .ipt-key-settings__badge { padding: 0.1rem 0.45rem; border-radius: 999px; background: var(--comfy-input-bg, #333); font-size: 0.75rem; font-weight: 400; white-space: nowrap; }
        .ipt-key-settings__input-row { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: 0.35rem; }
        .ipt-key-settings input { width: 100%; min-width: 0; padding: 0.45rem 0.55rem; color: var(--input-text, inherit); background: var(--comfy-input-bg, #222); border: 1px solid var(--border-color, #666); border-radius: 0.35rem; }
        .ipt-key-settings button { padding: 0.4rem 0.65rem; color: var(--input-text, inherit); background: var(--comfy-input-bg, #333); border: 1px solid var(--border-color, #666); border-radius: 0.35rem; cursor: pointer; }
        .ipt-key-settings button:hover:not(:disabled) { filter: brightness(1.12); }
        .ipt-key-settings button:disabled { opacity: 0.5; cursor: not-allowed; }
        .ipt-key-settings__actions { display: flex; flex-wrap: wrap; gap: 0.4rem; }
        .ipt-key-settings__footer { display: flex; justify-content: flex-end; gap: 0.45rem; }
        .ipt-key-settings__help { color: var(--descrip-text, #aaa); font-size: 0.8rem; }
        .ipt-key-settings__status { min-height: 1.25rem; font-size: 0.82rem; }
        .ipt-key-settings__status[data-kind="error"] { color: var(--error-text, #ff7878); }
        .ipt-key-settings__status[data-kind="warning"] { color: var(--warning-text, #e5b95c); }
        .ipt-key-settings__status[data-kind="success"] { color: var(--success-text, #65c98c); }
        .ipt-key-settings__conflict { display: grid; gap: 0.5rem; }
        .ipt-key-settings__conflict[hidden] { display: none !important; }
        .ipt-key-settings__error { color: var(--error-text, #ff7878); font-size: 0.78rem; }
        .ipt-key-settings__error:empty { display: none; }
        @container (max-width: 30rem) {
            .ipt-key-settings__input-row { grid-template-columns: minmax(0, 1fr) auto; }
            .ipt-key-settings__input-row button:last-child { grid-column: 2; }
        }
        @container (max-width: 22rem) {
            .ipt-key-settings__input-row { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
            .ipt-key-settings__input-row input { grid-column: 1 / -1; }
            .ipt-key-settings__input-row button:last-child { grid-column: auto; }
            .ipt-key-settings__input-row button { min-width: 0; }
            .ipt-key-settings__actions button { flex: 1 1 auto; }
            .ipt-key-settings__footer { flex-wrap: wrap; }
        }
        @media (max-width: 700px) {
            .ipt-key-settings__form-row { grid-template-columns: minmax(0, 1fr); }
        }
    `;
    document.head.appendChild(style);
}

function prepareResponsiveSettingsHost(root) {
    queueMicrotask(() => {
        // Current ComfyUI wraps custom values in FormItem and CustomFormValue.
        // Add scoped classes only when the expected setting host is present; otherwise the root CSS remains the fallback.
        const valueHost = root.parentElement;
        const formInput = valueHost?.parentElement;
        const formRow = formInput?.parentElement;
        const settingItem = root.closest("[data-setting-id]");
        if (
            !valueHost
            || !formInput
            || !formRow
            || settingItem?.getAttribute("data-setting-id") !== SETTING_ID
            || !settingItem.contains(formRow)
        ) {
            return;
        }

        valueHost.classList.add("ipt-key-settings__host");
        formInput.classList.add("ipt-key-settings__form-input");
        formRow.classList.add("ipt-key-settings__form-row");
        for (const child of formRow.children) {
            if (child !== formInput) {
                child.classList.add("ipt-key-settings__form-label");
            }
        }
    });
}

function localeText() {
    let locale = "";
    try {
        locale = String(app?.extensionManager?.setting?.get?.("Comfy.Locale") || "");
    } catch (_error) {
        locale = "";
    }
    if (!locale) {
        locale = document.documentElement.lang || navigator.language || "en";
    }
    return locale.toLowerCase().startsWith("ja") ? TEXT.ja : TEXT.en;
}

function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) {
        node.className = className;
    }
    if (text != null) {
        node.textContent = text;
    }
    return node;
}

function isEnvelope(value) {
    return Boolean(
        value
        && typeof value === "object"
        && value.format === ENVELOPE_FORMAT
        && Number.isInteger(value.revision),
    );
}

function isLogicalKeyValue(value) {
    return Boolean(
        value
        && typeof value === "object"
        && value.version === 1
        && typeof value.primary_key === "string"
        && typeof value.fallback_key === "string",
    );
}

function envelopeEquals(left, right) {
    if (!isEnvelope(left) || !isEnvelope(right)) {
        return false;
    }
    const fields = ["format", "version", "cipher", "kdf", "key_source", "revision", "salt", "nonce", "ciphertext"];
    return fields.every((field) => left[field] === right[field]);
}

function envelopeRevision(value) {
    return isEnvelope(value) ? value.revision : 0;
}

function normalizeKeyInput(value, allowEmpty = false) {
    let text = String(value || "").trim();
    if (text.endsWith("=") && text.indexOf("=") === text.length - 1) {
        text = text.slice(0, -1);
    }
    if (!text && allowEmpty) {
        return "";
    }
    return KEY_PATTERN.test(text) ? text : null;
}

async function fetchJson(endpoint, options = {}) {
    if (!api || typeof api.fetchApi !== "function") {
        throw new Error("ComfyUI API is unavailable.");
    }
    const response = await api.fetchApi(endpoint, options);
    let payload = null;
    try {
        payload = await response.json();
    } catch (_error) {
        payload = null;
    }
    if (!response.ok || !payload?.ok) {
        throw new Error(payload?.error || `Request failed (${response.status}).`);
    }
    return payload;
}

function postJson(endpoint, payload = {}) {
    return fetchJson(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
}

async function storeComfyEnvelope(envelope) {
    const settingService = app?.extensionManager?.setting;
    if (settingService && typeof settingService.set === "function") {
        await Promise.resolve(settingService.set(SETTING_ID, envelope));
        return;
    }
    if (app?.ui?.settings?.setSettingValueAsync) {
        await app.ui.settings.setSettingValueAsync(SETTING_ID, envelope);
        return;
    }
    throw new Error("ComfyUI setting service is unavailable.");
}

async function storeBackendEnvelope(envelope) {
    await postJson(SETTINGS_ENDPOINT, {
        protected_settings: {
            [SETTING_ID]: envelope,
        },
    });
}

async function storeBothCopies(envelope) {
    await storeBackendEnvelope(envelope);
    await storeComfyEnvelope(envelope);
}

async function unprotectEnvelope(envelope) {
    const payload = await postJson(UNPROTECT_ENDPOINT, {
        setting_id: SETTING_ID,
        envelope,
    });
    return payload.value;
}

async function tryUnprotectEnvelope(envelope) {
    if (!isEnvelope(envelope)) {
        return null;
    }
    try {
        return await unprotectEnvelope(envelope);
    } catch (_error) {
        return null;
    }
}

async function protectValue(value, revision, keySource) {
    const body = {
        setting_id: SETTING_ID,
        value,
        revision,
    };
    if (keySource) {
        body.key_source = keySource;
    }
    const payload = await postJson(PROTECT_ENDPOINT, body);
    return payload.envelope;
}

function showToast(severity, summary, detail) {
    try {
        app?.extensionManager?.toast?.add?.({ severity, summary, detail, life: 4000 });
    } catch (_error) {
        // The inline status remains available when the toast service is unavailable.
    }
}

async function resolveSavedState(frontendValue) {
    const payload = await fetchJson(SETTINGS_ENDPOINT);
    let frontendEnvelope = isEnvelope(frontendValue) ? frontendValue : null;
    let backendEnvelope = payload.protected_settings?.[SETTING_ID] || null;

    if (isLogicalKeyValue(frontendValue)) {
        const revision = Math.max(1, envelopeRevision(backendEnvelope) + 1);
        frontendEnvelope = await protectValue(
            frontendValue,
            revision,
            backendEnvelope?.key_source,
        );
        await storeBothCopies(frontendEnvelope);
        backendEnvelope = frontendEnvelope;
    }

    if (!frontendEnvelope && backendEnvelope) {
        await storeComfyEnvelope(backendEnvelope);
        frontendEnvelope = backendEnvelope;
    } else if (frontendEnvelope && !backendEnvelope) {
        const value = await unprotectEnvelope(frontendEnvelope);
        if (value) {
            await storeBackendEnvelope(frontendEnvelope);
            backendEnvelope = frontendEnvelope;
        }
    }

    if (!frontendEnvelope && !backendEnvelope) {
        throw new Error("No protected metadata encryption setting was returned.");
    }

    if (envelopeEquals(frontendEnvelope, backendEnvelope)) {
        const value = await unprotectEnvelope(frontendEnvelope);
        return {
            value,
            selectedEnvelope: frontendEnvelope,
            frontendEnvelope,
            backendEnvelope,
            conflict: null,
        };
    }

    const [frontendPlain, backendPlain] = await Promise.all([
        tryUnprotectEnvelope(frontendEnvelope),
        tryUnprotectEnvelope(backendEnvelope),
    ]);

    if (!frontendPlain && !backendPlain) {
        throw new Error("Neither protected setting copy could be decrypted.");
    }

    let selectedSide = "backend";
    if (frontendPlain && !backendPlain) {
        selectedSide = "frontend";
    } else if (frontendPlain && backendPlain) {
        selectedSide = envelopeRevision(frontendEnvelope) > envelopeRevision(backendEnvelope)
            ? "frontend"
            : "backend";
    }

    const selectedEnvelope = selectedSide === "frontend" ? frontendEnvelope : backendEnvelope;
    const value = selectedSide === "frontend" ? frontendPlain : backendPlain;
    return {
        value,
        selectedEnvelope,
        frontendEnvelope,
        backendEnvelope,
        conflict: {
            frontendValid: Boolean(frontendPlain),
            backendValid: Boolean(backendPlain),
            sameRevision: envelopeRevision(frontendEnvelope) === envelopeRevision(backendEnvelope),
            selectedSide,
        },
    };
}

function createKeySection({ title, badge, help, isFallback, text, onDraftChange }) {
    const section = element("div", "ipt-key-settings__section");
    const heading = element("div", "ipt-key-settings__heading");
    const headingText = element("span", "", title);
    const badgeNode = element("span", "ipt-key-settings__badge", badge);
    heading.append(headingText, badgeNode);

    const inputRow = element("div", "ipt-key-settings__input-row");
    const input = document.createElement("input");
    input.type = "password";
    input.autocomplete = "new-password";
    input.autocapitalize = "off";
    input.spellcheck = false;
    input.placeholder = isFallback ? text.notSet : "";
    const showButton = element("button", "", text.show);
    showButton.type = "button";
    const copyButton = element("button", "", text.copy);
    copyButton.type = "button";
    inputRow.append(input, showButton, copyButton);

    const error = element("div", "ipt-key-settings__error");
    section.append(heading, inputRow, error);
    if (help) {
        section.append(element("div", "ipt-key-settings__help", help));
    }
    const actions = element("div", "ipt-key-settings__actions");
    section.append(actions);

    input.addEventListener("input", () => {
        error.textContent = "";
        onDraftChange(input.value);
    });
    showButton.addEventListener("click", () => {
        const showing = input.type === "text";
        input.type = showing ? "password" : "text";
        showButton.textContent = showing ? text.show : text.hide;
    });
    copyButton.addEventListener("click", async () => {
        if (!input.value) {
            return;
        }
        try {
            await navigator.clipboard.writeText(input.value);
            showToast("success", text.copied);
        } catch (_error) {
            showToast("error", text.copyFailed);
        }
    });

    return {
        section,
        input,
        badge: badgeNode,
        actions,
        error,
        copyButton,
        resetVisibility() {
            input.type = "password";
            showButton.textContent = text.show;
        },
    };
}

function renderMetadataEncryptionSetting(_name, _setter, frontendValue) {
    installStyles();
    const text = localeText();
    const root = element("div", "ipt-key-settings");
    prepareResponsiveSettingsHost(root);
    const status = element("div", "ipt-key-settings__status", text.loading);
    status.dataset.kind = "info";
    root.append(status);

    const setStatus = (message, kind = "info") => {
        status.textContent = message || "";
        status.dataset.kind = kind;
    };

    void (async () => {
        let state;
        let deviceDefault = null;
        try {
            [state, deviceDefault] = await Promise.all([
                resolveSavedState(frontendValue),
                postJson(DEVICE_DEFAULT_ENDPOINT),
            ]);
        } catch (_error) {
            setStatus(text.loadFailed, "error");
            showToast("error", text.loadFailed);
            return;
        }

        if (!root.isConnected) {
            return;
        }

        const saved = {
            version: 1,
            primary_key: state.value.primary_key,
            fallback_key: state.value.fallback_key || "",
        };
        const draft = { ...saved };
        let selectedEnvelope = state.selectedEnvelope;
        let frontendEnvelope = state.frontendEnvelope;
        let backendEnvelope = state.backendEnvelope;
        let busy = false;

        root.innerHTML = "";
        root.append(status);

        const primary = createKeySection({
            title: text.primary,
            badge: text.custom,
            help: "",
            isFallback: false,
            text,
            onDraftChange: (value) => {
                draft.primary_key = value;
                updateBadges();
            },
        });
        const fallback = createKeySection({
            title: text.fallback,
            badge: text.notSet,
            help: text.fallbackHelp,
            isFallback: true,
            text,
            onDraftChange: (value) => {
                draft.fallback_key = value;
                updateBadges();
            },
        });

        primary.input.value = draft.primary_key;
        fallback.input.value = draft.fallback_key;
        fallback.copyButton.disabled = !draft.fallback_key;

        const generateButton = element("button", "", text.generate);
        generateButton.type = "button";
        const restoreButton = element("button", "", text.restore);
        restoreButton.type = "button";
        primary.actions.append(generateButton, restoreButton);

        const clearFallbackButton = element("button", "", text.clearFallback);
        clearFallbackButton.type = "button";
        fallback.actions.append(clearFallbackButton);

        const conflictPanel = element("div", "ipt-key-settings__conflict");
        conflictPanel.hidden = true;
        const conflictMessage = element("div", "");
        const conflictActions = element("div", "ipt-key-settings__actions");
        conflictPanel.append(conflictMessage, conflictActions);

        const footer = element("div", "ipt-key-settings__footer");
        const cancelButton = element("button", "", text.cancel);
        const applyButton = element("button", "", text.apply);
        cancelButton.type = "button";
        applyButton.type = "button";
        footer.append(cancelButton, applyButton);

        root.append(primary.section, fallback.section, conflictPanel, footer);

        function updateBadges() {
            if (deviceDefault?.reproducible && draft.primary_key === deviceDefault.key) {
                primary.badge.textContent = text.deviceDerived;
            } else if (!deviceDefault?.reproducible && draft.primary_key === deviceDefault?.key) {
                primary.badge.textContent = text.randomFallback;
            } else {
                primary.badge.textContent = text.custom;
            }
            fallback.badge.textContent = draft.fallback_key ? text.custom : text.notSet;
            fallback.copyButton.disabled = !draft.fallback_key;
        }

        function setBusy(nextBusy) {
            busy = nextBusy;
            for (const button of root.querySelectorAll("button")) {
                button.disabled = nextBusy;
            }
            fallback.copyButton.disabled = nextBusy || !draft.fallback_key;
        }

        function validateDraft() {
            primary.error.textContent = "";
            fallback.error.textContent = "";
            const primaryKey = normalizeKeyInput(draft.primary_key, false);
            const fallbackKey = normalizeKeyInput(draft.fallback_key, true);
            if (primaryKey == null) {
                primary.error.textContent = text.invalidPrimary;
                return null;
            }
            if (fallbackKey == null) {
                fallback.error.textContent = text.invalidFallback;
                return null;
            }
            if (fallbackKey && fallbackKey === primaryKey) {
                fallback.error.textContent = text.duplicate;
                return null;
            }
            return { version: 1, primary_key: primaryKey, fallback_key: fallbackKey };
        }

        async function saveValue(nextValue) {
            if (busy) {
                return;
            }
            setBusy(true);
            setStatus("");
            try {
                const revision = Math.max(
                    envelopeRevision(frontendEnvelope),
                    envelopeRevision(backendEnvelope),
                    envelopeRevision(selectedEnvelope),
                ) + 1;
                const envelope = await protectValue(
                    nextValue,
                    revision,
                    selectedEnvelope?.key_source,
                );
                await storeBothCopies(envelope);
                Object.assign(saved, nextValue);
                Object.assign(draft, nextValue);
                selectedEnvelope = envelope;
                frontendEnvelope = envelope;
                backendEnvelope = envelope;
                conflictPanel.hidden = true;
                setStatus(text.saved, "success");
                showToast("success", text.saved);
            } catch (_error) {
                setStatus(text.saveFailed, "error");
                showToast("error", text.saveFailed);
            } finally {
                setBusy(false);
            }
        }

        async function synchronizeEnvelope(envelope) {
            if (!envelope || busy) {
                return;
            }
            setBusy(true);
            try {
                await storeBothCopies(envelope);
                frontendEnvelope = envelope;
                backendEnvelope = envelope;
                selectedEnvelope = envelope;
                conflictPanel.hidden = true;
                setStatus(text.saved, "success");
            } catch (_error) {
                setStatus(text.saveFailed, "error");
            } finally {
                setBusy(false);
            }
        }

        function configureConflict() {
            if (!state.conflict) {
                conflictPanel.hidden = true;
                setStatus("");
                return;
            }
            conflictPanel.hidden = false;
            conflictActions.innerHTML = "";
            if (state.conflict.sameRevision && state.conflict.frontendValid && state.conflict.backendValid) {
                conflictMessage.textContent = `${text.conflictWarning} ${text.syncWarning}`;
                const useComfy = element("button", "", text.useComfy);
                const useBackend = element("button", "", text.useBackend);
                useComfy.type = "button";
                useBackend.type = "button";
                useComfy.addEventListener("click", () => void synchronizeEnvelope(frontendEnvelope));
                useBackend.addEventListener("click", () => void synchronizeEnvelope(backendEnvelope));
                conflictActions.append(useComfy, useBackend);
            } else {
                conflictMessage.textContent = text.oneCopyWarning;
                const retry = element("button", "", text.retrySync);
                retry.type = "button";
                retry.addEventListener("click", () => void synchronizeEnvelope(selectedEnvelope));
                conflictActions.append(retry);
            }
            setStatus(text.syncWarning, "warning");
        }

        generateButton.addEventListener("click", async () => {
            setBusy(true);
            try {
                const payload = await postJson(RANDOM_KEY_ENDPOINT);
                draft.primary_key = payload.key;
                primary.input.value = draft.primary_key;
                primary.error.textContent = "";
                updateBadges();
            } catch (_error) {
                setStatus(text.loadFailed, "error");
            } finally {
                setBusy(false);
            }
        });

        restoreButton.addEventListener("click", () => {
            draft.primary_key = deviceDefault.key;
            primary.input.value = draft.primary_key;
            primary.error.textContent = "";
            updateBadges();
        });

        clearFallbackButton.addEventListener("click", () => {
            draft.fallback_key = "";
            fallback.input.value = "";
            fallback.error.textContent = "";
            updateBadges();
        });

        cancelButton.addEventListener("click", () => {
            Object.assign(draft, saved);
            primary.input.value = draft.primary_key;
            fallback.input.value = draft.fallback_key;
            primary.error.textContent = "";
            fallback.error.textContent = "";
            primary.resetVisibility();
            fallback.resetVisibility();
            updateBadges();
            setStatus("");
        });

        applyButton.addEventListener("click", () => {
            const nextValue = validateDraft();
            if (!nextValue) {
                return;
            }
            void saveValue(nextValue);
        });

        updateBadges();
        configureConflict();
    })();

    return root;
}

app.registerExtension({
    name: "IPT.MetadataEncryptionSettings",
    settings: [
        {
            id: SETTING_ID,
            name: "Encryption keys",
            type: renderMetadataEncryptionSetting,
            defaultValue: null,
            category: ["Info-Prompt-Toolkit", "Metadata Encryption", "99 Encryption keys"],
            tooltip: "Configure the primary metadata encryption key and the fallback decryption key.",
            sortOrder: 99,
        },
    ],
});
