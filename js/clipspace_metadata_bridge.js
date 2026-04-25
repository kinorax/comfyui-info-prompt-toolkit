// Copyright 2026 kinorax
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function safeParseJson(value) {
    if (!value || typeof value !== "string") {
        return null;
    }
    try {
        return JSON.parse(value);
    } catch {
        return null;
    }
}

function pathFromRef(ref) {
    if (!ref || typeof ref !== "object" || !ref.filename) {
        return null;
    }
    if (ref.subfolder) {
        return `${ref.subfolder}/${ref.filename}`;
    }
    return `${ref.filename}`;
}

function annotatedFromRef(ref) {
    const path = pathFromRef(ref);
    if (!path) {
        return null;
    }
    const type = ref.type;
    if (type) {
        return `${path} [${type}]`;
    }
    return path;
}

function buildMaskedCandidates(maskedRef, fallbackFileName) {
    const out = new Set();

    const path = pathFromRef(maskedRef);
    const annotated = annotatedFromRef(maskedRef);
    if (path) {
        out.add(path);
    }
    if (annotated) {
        out.add(annotated);
    }

    const fileName = fallbackFileName || maskedRef?.filename;
    if (fileName) {
        out.add(fileName);
        if (maskedRef?.type) {
            out.add(`${fileName} [${maskedRef.type}]`);
        }

        if (!fileName.includes("/") && /^clipspace-/.test(fileName)) {
            out.add(`clipspace/${fileName}`);
            out.add(`clipspace/${fileName} [input]`);
            if (maskedRef?.type) {
                out.add(`clipspace/${fileName} [${maskedRef.type}]`);
            }
        }

        // When mask is saved, clipspace-painted-<stamp>.png is often generated next
        // and used as source of clipspace-painted-masked-<stamp>.png. Pre-map it.
        const maskMatch = /^clipspace-mask-(\d+)\.png$/.exec(fileName);
        if (maskMatch) {
            const stamp = maskMatch[1];
            const paintedName = `clipspace-painted-${stamp}.png`;
            out.add(paintedName);
            out.add(`clipspace/${paintedName}`);
            out.add(`clipspace/${paintedName} [input]`);
            if (maskedRef?.type) {
                out.add(`${paintedName} [${maskedRef.type}]`);
                out.add(`clipspace/${paintedName} [${maskedRef.type}]`);
            }
        }
    }

    return [...out];
}

function postClipspaceSourceMapping(originalFetchApi, sourceAnnotated, maskedRef, imageFileName) {
    const payload = {
        source_annotated: sourceAnnotated,
        masked_candidates: buildMaskedCandidates(maskedRef, imageFileName),
    };

    void originalFetchApi("/iis/clipspace-source", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    }).catch(() => {
        // Never interrupt normal upload flow.
    });
}

function trySendMappingFromFormData(formData, originalFetchApi) {
    const imageFile = formData.get("image");
    const imageFileName = imageFile && typeof imageFile.name === "string" ? imageFile.name : "";

    const sourceRef = safeParseJson(formData.get("original_ref"));
    const sourceAnnotated = annotatedFromRef(sourceRef);

    if (!(sourceAnnotated && imageFileName)) {
        return;
    }

    const type = typeof formData.get("type") === "string" ? formData.get("type") : "input";
    const subfolderField = formData.get("subfolder");
    const subfolder = typeof subfolderField === "string"
        ? subfolderField
        : (imageFileName.startsWith("clipspace-") ? "clipspace" : "");

    const maskedRef = {
        filename: imageFileName,
        subfolder,
        type,
    };

    postClipspaceSourceMapping(originalFetchApi, sourceAnnotated, maskedRef, imageFileName);
}

function installUploadMaskHook() {
    if (!api || typeof api.fetchApi !== "function") {
        return;
    }
    if (api.__iisUploadMaskHookInstalled) {
        return;
    }

    const originalFetchApi = api.fetchApi.bind(api);

    api.fetchApi = async (route, options) => {
        try {
            const routeText = String(route || "");
            const isUploadMask = routeText.includes("/upload/mask");
            const isUploadImage = routeText.includes("/upload/image");
            const formData = options?.body;

            if ((isUploadMask || isUploadImage) && formData instanceof FormData) {
                trySendMappingFromFormData(formData, originalFetchApi);
            }
        } catch {
            // Never block normal upload flow.
        }

        return originalFetchApi(route, options);
    };

    api.__iisUploadMaskHookInstalled = true;
}

app.registerExtension({
    name: "IPT.ClipspaceMetadataBridge",
    init() {
        installUploadMaskHook();
    },
});
