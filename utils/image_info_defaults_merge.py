from __future__ import annotations


def is_unset(value: object) -> bool:
    return value is None


def merge_extras_missing_keys(base_extras: object, defaults_extras: object) -> dict[str, object] | None:
    if base_extras is None:
        if isinstance(defaults_extras, dict) and defaults_extras:
            return dict(defaults_extras)
        return None

    if not isinstance(base_extras, dict):
        return None

    output = dict(base_extras)
    if isinstance(defaults_extras, dict):
        for key, value in defaults_extras.items():
            if key not in output:
                output[key] = value

    if output:
        return output
    return None


def merge_image_info_missing_values(
    base_image_info: object,
    defaults_image_info: object,
    *,
    extras_key: str = "extras",
    positive_key: str = "positive",
    lora_stack_key: str = "lora_stack",
    preserve_lora_stack_when_positive_present: bool = False,
) -> dict[str, object]:
    output = dict(base_image_info) if isinstance(base_image_info, dict) else {}
    if not isinstance(defaults_image_info, dict):
        return output

    lock_lora_stack = preserve_lora_stack_when_positive_present and (not is_unset(output.get(positive_key)))

    for key, value in defaults_image_info.items():
        if lock_lora_stack and key == lora_stack_key:
            continue

        if key == extras_key:
            merged_extras = merge_extras_missing_keys(output.get(extras_key), value)
            if merged_extras is not None:
                output[extras_key] = merged_extras
            continue

        if is_unset(output.get(key)):
            output[key] = value

    return output
