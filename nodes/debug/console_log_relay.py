# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any

from comfy_api.latest import io as c_io

from ... import const as Const
from ...utils import cast as Cast

_MATCH_TEMPLATE = c_io.MatchType.Template("debug_console_log_relay", c_io.AnyType)


def _bool_or_default(value: Any, default: bool) -> bool:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _bool_or_default(value[0], default)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _short_text(value: Any, max_length: int = 120) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<unreprable {type(value).__name__}>"

    text = text.replace("\n", "\\n")
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}..."


def _type_name(value: Any) -> str:
    if value is None:
        return "NoneType"
    return type(value).__name__


class ConsoleLogRelay(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-ConsoleLogRelay",
            display_name="Console Log Relay",
            category=Const.CATEGORY_DEBUG,
            hidden=[
                c_io.Hidden.unique_id,
            ],
            inputs=[
                c_io.MatchType.Input(
                    "value",
                    template=_MATCH_TEMPLATE,
                    tooltip="Pass-through value to log to the server console",
                ),
                c_io.String.Input(
                    "label",
                    default="",
                    optional=True,
                    tooltip="Optional message prefix shown in console logs",
                ),
                c_io.Boolean.Input(
                    "enabled",
                    default=True,
                    tooltip="If false, this node logs nothing and only relays the value",
                ),
            ],
            outputs=[
                c_io.MatchType.Output(
                    template=_MATCH_TEMPLATE,
                    id=Cast.out_id("value"),
                    display_name="value",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        value: Any = None,
        label: Any = "",
        enabled: Any = True,
    ) -> bool | str:
        return True

    @classmethod
    def execute(
        cls,
        value: Any = None,
        label: Any = "",
        enabled: Any = True,
    ) -> c_io.NodeOutput:
        if _bool_or_default(enabled, True):
            unique_id = getattr(cls.hidden, "unique_id", None)
            label_text = str(label).strip() if label is not None else ""
            if len(label_text) == 0:
                label_text = "-"

            print(
                "[IPT-ConsoleLogRelay]"
                f" node={unique_id}"
                f" label={label_text}"
                f" type={_type_name(value)}"
                f" value={_short_text(value)}"
            )

        return c_io.NodeOutput(value)
