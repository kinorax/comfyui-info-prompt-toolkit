# Copyright 2026 kinorax
from __future__ import annotations

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io as c_io
from .nodes.image_info_context import ImageInfoContext
from .nodes.image_info_defaults import ImageInfoDefaults
from .nodes.image_info_fallback import ImageInfoFallback
from .nodes.any_switch_any import AnySwitchAny
from .nodes.debug.console_log_relay import ConsoleLogRelay
from .nodes.debug.image_batch_count_debug import ImageBatchCountDebug
from .nodes.debug.image_info_count_debug import ImageInfoCountDebug
from .nodes.debug.latent_batch_count_debug import LatentBatchCountDebug
from .nodes.sampler_selector import SamplerSelector
from .nodes.scheduler_selector import SchedulerSelector
from .nodes.sampler_params import SamplerParams
from .nodes.split_sampler_params import SplitSamplerParams
from .nodes.sampler_custom_from_params import SamplerCustomFromParams
from .nodes.sampler_custom_from_params_tiled import SamplerCustomFromParamsTiled
from .nodes.lora_selector import LoraSelector
from .nodes.checkpoint_selector import CheckpointSelector
from .nodes.use_loaded_model import UseLoadedModel
from .nodes.load_new_model import LoadNewModel
# from .nodes.model_merge import ModelMerge
from .nodes.vae_selector import VaeSelector
from .nodes.clip_selector import ClipSelector
from .nodes.dual_clip_selector import DualClipSelector
from .nodes.triple_clip_selector import TripleClipSelector
from .nodes.quadruple_clip_selector import QuadrupleClipSelector
from .nodes.unet_model_selector import UnetModelSelector
from .nodes.diffusion_model_selector import DiffusionModelSelector
# from .nodes.model_sampling_auraflow import ModelSamplingAuraFlow
from .nodes.image_reader import ImageReader
from .nodes.image_list_to_batch import ImageListToBatch
from .nodes.video_reader import VideoReader
from .nodes.batch_image_reader import ImageDirectoryReader
from .nodes.caption_file_saver import CaptionFileSaver
from .nodes.image_saver import ImageSaver
from .nodes.video_saver import VideoSaver
from .nodes.image_info_to_infotext import ImageInfoToInfotext
from .nodes.infotext_to_image_info import InfotextToImageInfo
from .nodes.prompt_to_lora_stack import PromptToLoraStack
from .nodes.combine_lora_stacks import CombineLoraStacks
from .nodes.lora_stack_lorader import LoraStackLorader
from .nodes.prompt.combine_prompts import CombinePrompts
from .nodes.prompt.merge_caption_tokens import MergeCaptionTokens
from .nodes.prompt.remove_caption_tokens import RemoveCaptionTokens
from .nodes.prompt.remove_prompt_comments import RemovePromptComments
from .nodes.prompt.normalize_prompt_tokens import NormalizePromptTokens
from .nodes.prompt.pixai_tagger import PixAITagger
from .nodes.prompt.prompt_template import PromptTemplate
from .nodes.prompt.strip_prompt_weights import FlattenPromptForCaption
from .nodes.set_string_extra import SetStringExtra
from .nodes.set_lora_stack_extra import SetLoraStackExtra
from .nodes.set_sampler_params_extra import SetSamplerParamsExtra
from .nodes.set_int_extra import SetIntExtra
from .nodes.set_float_extra import SetFloatExtra
from .nodes.set_size_extra import SetSizeExtra
from .nodes.aspect_ratio_to_size import AspectRatioToSize
from .nodes.scale_width_height import ScaleWidthHeight
from .nodes.remove_image_info_extra_key import RemoveImageInfoExtraKeys
from .nodes.remove_image_info_main_field import RemoveImageInfoMainFields
from .nodes.release_memory import ReleaseMemory
from .nodes.get_string_extra import GetStringExtra
from .nodes.get_lora_stack_extra import GetLoraStackExtra
from .nodes.get_sampler_params_extra import GetSamplerParamsExtra
from .nodes.get_int_extra import GetIntExtra
from .nodes.get_float_extra import GetFloatExtra
from .nodes.get_size_extra import GetSizeExtra
from .nodes.split_width_height import SplitWidthHeight
from .nodes.seed_generator import SeedGenerator
from .nodes.mask.mask_overlay_comparer import MaskOverlayComparer
from .nodes.mask.detailer_start import DetailerStart
from .nodes.mask.detailer_end import DetailerEnd
from .nodes.mask.grow_mask import GrowMask
from .nodes.mask.remove_small_mask_regions import RemoveSmallMaskRegions
from .nodes.mask.remove_small_soft_mask_regions import RemoveSmallSoftMaskRegions
from .nodes.mask.sam3_text_mask import Sam3PromptToMask
from .nodes.xy_plot.xy_plot_modifier import XYPlotModifier
from .nodes.xy_plot.xy_plot_start import XYPlotStart
from .nodes.xy_plot.xy_plot_end import XYPlotEnd
from .utils import clipspace_bridge as _clipspace_bridge  # noqa: F401
from .utils import image_batch_reader_directory_api as _image_batch_reader_directory_api  # noqa: F401
from .utils import image_reader_model_check_api as _image_reader_model_check_api  # noqa: F401
from .utils import model_lora_metadata_api as _model_lora_metadata_api  # noqa: F401
from .utils import prompt_template_wildcards_api as _prompt_template_wildcards_api  # noqa: F401
from .utils import release_memory_api as _release_memory_api  # noqa: F401
from .utils import video_reader_remote_options_api as _video_reader_remote_options_api  # noqa: F401
from .utils.development_nodes import load_development_node_list as _load_development_node_list
from .utils.model_lora_metadata_pipeline import get_shared_metadata_pipeline as _get_shared_metadata_pipeline

WEB_DIRECTORY = "./js"
__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]

_get_shared_metadata_pipeline(start=True)


class _Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[c_io.ComfyNode]]:
        node_list: list[type[c_io.ComfyNode]] = [
            ImageInfoContext,
            ImageInfoDefaults,
            ImageInfoFallback,
            AnySwitchAny,
            ConsoleLogRelay,
            ImageBatchCountDebug,
            ImageInfoCountDebug,
            LatentBatchCountDebug,
            SamplerSelector,
            SchedulerSelector,
            SamplerParams,
            SplitSamplerParams,
            SamplerCustomFromParams,
            SamplerCustomFromParamsTiled,
            LoraSelector,
            CheckpointSelector,
            LoadNewModel,
            # ModelMerge,
            UseLoadedModel,
            VaeSelector,
            ClipSelector,
            DualClipSelector,
            TripleClipSelector,
            QuadrupleClipSelector,
            UnetModelSelector,
            DiffusionModelSelector,
            # ModelSamplingAuraFlow,
            ImageReader,
            ImageListToBatch,
            VideoReader,
            ImageDirectoryReader,
            CaptionFileSaver,
            ImageSaver,
            VideoSaver,
            ImageInfoToInfotext,
            InfotextToImageInfo,
            PromptToLoraStack,
            CombineLoraStacks,
            LoraStackLorader,
            CombinePrompts,
            MergeCaptionTokens,
            RemoveCaptionTokens,
            RemovePromptComments,
            NormalizePromptTokens,
            PixAITagger,
            PromptTemplate,
            FlattenPromptForCaption,
            SetStringExtra,
            SetLoraStackExtra,
            SetSamplerParamsExtra,
            SetIntExtra,
            SetFloatExtra,
            SetSizeExtra,
            AspectRatioToSize,
            ScaleWidthHeight,
            RemoveImageInfoExtraKeys,
            RemoveImageInfoMainFields,
            ReleaseMemory,
            GetStringExtra,
            GetLoraStackExtra,
            GetSamplerParamsExtra,
            GetIntExtra,
            GetFloatExtra,
            GetSizeExtra,
            SplitWidthHeight,
            SeedGenerator,
            MaskOverlayComparer,
            DetailerStart,
            DetailerEnd,
            GrowMask,
            RemoveSmallMaskRegions,
            RemoveSmallSoftMaskRegions,
            Sam3PromptToMask,
            XYPlotModifier,
            XYPlotStart,
            XYPlotEnd,
        ]
        node_list.extend(_load_development_node_list(__package__, __file__))
        return node_list


async def comfy_entrypoint() -> ComfyExtension:
    return _Extension()
