import { app } from "../../scripts/app.js";
import {
    getNestedTargetWidgetNames,
    resolveNestedComboTarget,
} from "./selector_targets.js";

function installStyle() {
    if (document.getElementById("IPT-nested-model-combo-style")) {
        return;
    }

    const style = document.createElement("style");
    style.id = "IPT-nested-model-combo-style";
    style.textContent = `
        .IPT-combo-folder {
            opacity: 0.75;
        }
        .IPT-combo-folder:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        .IPT-combo-folder-arrow {
            display: inline-block;
            width: 14px;
        }
        .IPT-combo-prefix {
            display: none;
        }

        /* Keep filtering usable after labels are shortened to basename. */
        .litecontextmenu.IPT-nested-combo-menu:has(input:not(:placeholder-shown)) .IPT-combo-folder-contents {
            display: block !important;
        }
        .litecontextmenu.IPT-nested-combo-menu:has(input:not(:placeholder-shown)) .IPT-combo-folder {
            display: none;
        }
        .litecontextmenu.IPT-nested-combo-menu:has(input:not(:placeholder-shown)) .IPT-combo-prefix {
            display: inline;
        }
        .litecontextmenu.IPT-nested-combo-menu:has(input:not(:placeholder-shown)) .litemenu-entry {
            padding-left: 2px !important;
        }
    `;
    document.body.appendChild(style);
}

function splitPath(pathText) {
    return String(pathText || "").split(/[\\/]+/).filter(Boolean);
}

function createFolderElement(name) {
    const folder = document.createElement("div");
    folder.className = "litemenu-entry IPT-combo-folder";
    folder.innerHTML = `<span class="IPT-combo-folder-arrow">></span> ${name}`;
    return folder;
}

function addToggleHandler(folderElement, contentsElement) {
    const toggle = (event) => {
        event.preventDefault();
        event.stopPropagation();

        const isClosed = contentsElement.style.display === "none";
        contentsElement.style.display = isClosed ? "block" : "none";
        const arrow = folderElement.querySelector(".IPT-combo-folder-arrow");
        if (arrow) {
            arrow.textContent = isClosed ? "v" : ">";
        }
    };

    folderElement.addEventListener("click", toggle);
}

function getWidgetOptionValues(widget) {
    const optionValues = widget?.options?.values;
    if (!Array.isArray(optionValues) || optionValues.length === 0) {
        return null;
    }

    const normalized = optionValues
        .filter((value) => typeof value === "string" && value)
        .map((value) => String(value));
    return normalized.length > 0 ? new Set(normalized) : null;
}

function canApplyNestedTree(menu, widget) {
    const optionValues = getWidgetOptionValues(widget);
    if (optionValues === null) {
        return false;
    }

    const itemElements = [...menu.querySelectorAll(".litemenu-entry[data-value]")];
    if (!itemElements.length) {
        return false;
    }

    return itemElements.every((itemElement) => optionValues.has(String(itemElement.getAttribute("data-value") ?? "")));
}

function buildNestedTree(menu, widget) {
    if (menu.dataset.iisNestedApplied === "1") {
        return;
    }

    if (!canApplyNestedTree(menu, widget)) {
        return;
    }

    const itemElements = [...menu.querySelectorAll(".litemenu-entry[data-value]")];
    if (!itemElements.length) {
        return;
    }

    const itemListSymbol = Symbol("items");
    const folderTree = new Map();
    let hasSubfolder = false;

    for (const itemElement of itemElements) {
        const value = itemElement.getAttribute("data-value");
        if (!value) {
            continue;
        }

        const pathSegments = splitPath(value);
        if (!pathSegments.length) {
            continue;
        }

        // Show only basename as the visible label.
        itemElement.textContent = pathSegments[pathSegments.length - 1];

        if (pathSegments.length <= 1) {
            continue;
        }

        hasSubfolder = true;

        // Keep full path searchable when user types in the filter input.
        const prefix = document.createElement("span");
        prefix.className = "IPT-combo-prefix";
        prefix.textContent = `${pathSegments.slice(0, -1).join("/")}/`;
        itemElement.prepend(prefix);

        itemElement.remove();

        let currentLevel = folderTree;
        for (let i = 0; i < pathSegments.length - 1; i += 1) {
            const folder = pathSegments[i];
            if (!currentLevel.has(folder)) {
                currentLevel.set(folder, new Map());
            }
            currentLevel = currentLevel.get(folder);
        }

        if (!currentLevel.has(itemListSymbol)) {
            currentLevel.set(itemListSymbol, []);
        }
        currentLevel.get(itemListSymbol).push(itemElement);
    }

    if (!hasSubfolder) {
        return;
    }

    const parent = itemElements[0].parentElement || menu;
    menu.classList.add("IPT-nested-combo-menu");
    const treeRoot = document.createElement("div");

    const appendFolders = (container, folderMap, level = 0) => {
        for (const [folderName, content] of folderMap.entries()) {
            if (folderName === itemListSymbol) {
                continue;
            }

            const folderElement = createFolderElement(folderName);
            folderElement.style.paddingLeft = `${(level * 10) + 6}px`;

            const childContainer = document.createElement("div");
            childContainer.className = "IPT-combo-folder-contents";
            childContainer.style.display = "none";

            const leafItems = content.get(itemListSymbol) || [];
            for (const leafItem of leafItems) {
                leafItem.style.paddingLeft = `${((level + 1) * 10) + 14}px`;
                childContainer.appendChild(leafItem);
            }

            appendFolders(childContainer, content, level + 1);
            addToggleHandler(folderElement, childContainer);

            container.appendChild(folderElement);
            container.appendChild(childContainer);
        }
    };

    appendFolders(treeRoot, folderTree, 0);

    const firstMenuItem = parent.querySelector(".litemenu-entry");
    if (firstMenuItem) {
        parent.insertBefore(treeRoot, firstMenuItem);
    } else {
        parent.appendChild(treeRoot);
    }

    menu.dataset.iisNestedApplied = "1";
}

function installObserver() {
    const observer = new MutationObserver((mutations) => {
        const node = app.canvas?.current_node;
        const target = resolveNestedComboTarget(node);
        if (!target) {
            return;
        }

        const overWidget = app.canvas?.getWidgetAtCursor?.();
        const targetWidgetNames = new Set(getNestedTargetWidgetNames(target));
        if (!overWidget || !targetWidgetNames.has(overWidget.name)) {
            return;
        }

        for (const mutation of mutations) {
            for (const addedNode of mutation.addedNodes) {
                if (!addedNode?.classList?.contains("litecontextmenu")) {
                    continue;
                }

                requestAnimationFrame(() => {
                    buildNestedTree(addedNode, overWidget);
                });
            }
        }
    });

    observer.observe(document.body, { childList: true, subtree: false });
}

app.registerExtension({
    name: "IPT.NestedModelSelectorCombo",
    init() {
        installStyle();
        installObserver();
    },
});
