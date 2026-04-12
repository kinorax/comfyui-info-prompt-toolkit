from typing_extensions import override
from comfy_api.latest import ComfyExtension, io as c_io
from .nodes.lora_selector import LoraSelector
from .nodes.prompt.pixai_tagger import PixAITagger
from .nodes.prompt.prompt_template import PromptTemplate
from .nodes.set_string_extra import SetStringExtra
from .nodes.aspect_ratio_to_size import AspectRatioToSize
from .nodes.mask.mask_overlay_comparer import MaskOverlayComparer
from .nodes.mask.grow_mask import GrowMask
from .nodes.mask.remove_small_soft_mask_regions import RemoveSmallSoftMaskRegions
from .nodes.mask.sam3_text_mask import Sam3PromptToMask
from .utils import clipspace_bridge as _clipspace_bridge  # noqa: F401
from .utils import image_reader_model_check_api as _image_reader_model_check_api  # noqa: F401
from .utils import model_lora_metadata_api as _model_lora_metadata_api  # noqa: F401
from .utils import prompt_template_wildcards_api as _prompt_template_wildcards_api  # noqa: F401
from .utils.model_lora_metadata_pipeline import get_shared_metadata_pipeline as _get_shared_metadata_pipeline

WEB_DIRECTORY = "./js"
__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]

_get_shared_metadata_pipeline(start=True)


class _Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[c_io.ComfyNode]]:
        return [
            LoraSelector,
            PixAITagger,
            PromptTemplate,
            SetStringExtra,
            AspectRatioToSize,
            MaskOverlayComparer,
            GrowMask,
            RemoveSmallSoftMaskRegions,
            Sam3PromptToMask,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    return _Extension()
