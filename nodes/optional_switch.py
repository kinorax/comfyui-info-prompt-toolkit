# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast

_MISSING = object()
_MATCH_TEMPLATE = c_io.MatchType.Template("optional_switch", c_io.AnyType)


class OptionalSwitch(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-OptionalSwitch",
            display_name="Switch (Optional)",
            category=Const.CATEGORY_UTILITY,
            description=(
                "Lazily evaluates only the branch selected by switch. "
                "Returns None when the selected branch is unconnected."
            ),
            search_aliases=[
                "optional switch",
                "if else",
                "conditional",
                "branch",
                "nullable switch",
            ],
            inputs=[
                c_io.Boolean.Input("switch"),
                c_io.MatchType.Input(
                    "on_false",
                    template=_MATCH_TEMPLATE,
                    optional=True,
                    lazy=True,
                    tooltip="Value returned when switch is false. May be unconnected.",
                ),
                c_io.MatchType.Input(
                    "on_true",
                    template=_MATCH_TEMPLATE,
                    optional=True,
                    lazy=True,
                    tooltip="Value returned when switch is true. May be unconnected.",
                ),
            ],
            outputs=[
                c_io.MatchType.Output(
                    template=_MATCH_TEMPLATE,
                    id=Cast.out_id("output"),
                    display_name="output",
                    tooltip="Selected value, or None when the selected branch is unconnected.",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        switch: bool,
        on_false: object = _MISSING,
        on_true: object = _MISSING,
    ) -> bool:
        return True

    @classmethod
    def check_lazy_status(
        cls,
        switch: bool,
        on_false: object = _MISSING,
        on_true: object = _MISSING,
    ) -> list[str]:
        selected_name = "on_true" if switch else "on_false"
        selected_value = on_true if switch else on_false

        if selected_value is _MISSING:
            return []
        if selected_value is None:
            return [selected_name]
        return []

    @classmethod
    def execute(
        cls,
        switch: bool,
        on_false: object = _MISSING,
        on_true: object = _MISSING,
    ) -> c_io.NodeOutput:
        selected_value = on_true if switch else on_false
        if selected_value is _MISSING:
            selected_value = None
        return c_io.NodeOutput(selected_value)
