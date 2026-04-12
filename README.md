[English](README.md) | [日本語](README.ja.md)

# ComfyUI-Info-Prompt-Toolkit

### Current Registry Status
<p><strong>This branch is currently limiting the number of registered nodes while basic behavior is being verified for ComfyUI registry submission.</strong><br>
<strong>The nodes currently published in this branch are <code>Prompt Template</code>, <code>SAM3 Prompt To Mask</code>, <code>PixAI Tagger</code>, <code>Mask Overlay Comparer</code>, <code>Aspect Ratio to Size</code>, <code>Grow Mask</code>, <code>Remove Small Soft Mask Regions</code>, <code>SetStringExtra</code>, and <code>LoraSelector</code>.</strong><br>
<strong>The remaining nodes, around 60 in total, are planned to be published soon.</strong></p>

This extension node collection is built around simplifying ComfyUI wiring and improving reusability, so trial results are easier to carry into the next production pass.  
Key features include Civitai-compatible image metadata saving, same-name `.txt` caption saving, XY Plot, Tiled Sampling (`SDXL (with ControlNet Tile)` and `Anima`), SAM3, Detailer, PixAI Tagger, wildcards, and Dynamic Prompts to strengthen your workflow.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kinorax/comfyui-info-prompt-toolkit.git
cd comfyui-info-prompt-toolkit
pip install -r requirements.txt
```

> Note: `SAM3 Prompt To Mask` and `PixAI Tagger` require manual placement of model files under `ComfyUI/models` after installation. See the relevant sections in `Core Nodes` for details.

## Core Nodes

### SamplerCustom (Sampler Params, Tiled) / SamplerCustom (Sampler Params)

<a href="assets/readme/samplercustom-sampler-params-pair.webp"><img align="left" hspace="16" src="assets/readme/samplercustom-sampler-params-pair.webp" alt="SamplerCustom pair" width="210"></a>
<ul>
  <li>Accepts <code>sampler_params</code> and performs sampling using the official <code>KSamplerSelect</code> / <code>BasicScheduler</code> / <code>SamplerCustom</code> nodes (no custom sampling logic is implemented).</li>
  <li>The Tiled version uses the same sampling settings as the standard version and runs inference by splitting into spatial tiles.</li>
  <li>The Tiled version is useful not only under VRAM constraints, but also when you need output resolutions beyond a model's practical supported size.</li>
  <li>Models with verified operation for the Tiled version are currently <code>SDXL (with ControlNet Tile)</code> and <code>Anima</code>. Other models may still work, but compatibility is not guaranteed.</li>
</ul>
<br clear="left">

### Sampler Params

<a href="assets/readme/sampler-params.webp"><img align="left" hspace="16" src="assets/readme/sampler-params.webp" alt="Sampler Params" width="210"></a>
<ul>
  <li>Aggregates sampler/scheduler/steps/denoise/seed/cfg into a single <code>sampler_params</code> bundle.</li>
  <li>Improves both wiring simplicity and configuration reuse.</li>
</ul>
<br clear="left">

### XY Plot Start / XY Plot Modifier / XY Plot End

<a href="assets/readme/xy-plot-start-modifier-end.webp"><img align="left" hspace="16" src="assets/readme/xy-plot-start-modifier-end.webp" alt="XY Plot nodes" width="210"></a>
<ul>
  <li><code>XY Plot Modifier</code> chains axis-specific change sets (<code>label</code> + <code>changes</code>) as an array and defines X/Y condition variations.</li>
  <li><code>XY Plot Start</code> applies X/Y modifiers to the base <code>image_info</code>, generates per-cell <code>image_info</code>, and outputs in <code>list</code> (all at once) or <code>loop</code> (iterative) mode.</li>
  <li><code>XY Plot End</code> receives <code>xy_plot_info</code> and generated images, then composes the axis-labeled grid image <code>xy_plot_image</code>.</li>
  <li>In <code>loop</code> mode, <code>Start</code> and <code>End</code> cooperate via <code>loop_control</code> to process all cells sequentially.</li>
</ul>
<br clear="left">

### Image Info Context

<a href="assets/readme/image-info-context.webp"><img align="left" hspace="16" src="assets/readme/image-info-context.webp" alt="Image Info Context" width="210"></a>
<ul>
  <li><code>Image Info Context</code> reconstructs updated <code>image_info</code> by reflecting only the input fields that are actually connected.</li>
  <li>When <code>positive</code> is connected, it extracts <code>&lt;lora:...:...&gt;</code> tags and applies the tag-removed <code>positive</code> text and extracted values to <code>lora_stack</code>.</li>
  <li><code>extras</code> is merged into existing <code>image_info.extras</code>, and duplicate keys are overwritten by input-side values.</li>
  <li>In addition to updated fields, it outputs <code>base_sampler_params</code> (with fixed <code>denoise=1.0</code>) for easy connection to downstream sampling nodes.</li>
</ul>
<br clear="left">

### Image Info Fallback

<a href="assets/readme/image-info-fallback.webp"><img align="left" hspace="16" src="assets/readme/image-info-fallback.webp" alt="Image Info Fallback" width="210"></a>
<ul>
  <li><code>Image Info Fallback</code> fills missing fields in the primary <code>image_info</code> from <code>image_info_fallback</code>.</li>
  <li>Fallback is limited to missing fields, and fields that already have values in the primary side are not overwritten.</li>
  <li><code>extras</code> is merged by adding only missing keys, while existing keys are preserved.</li>
  <li>When primary <code>positive</code> is present, <code>lora_stack</code> is not supplemented from the fallback side.</li>
</ul>
<br clear="left">

### Detailer Start / Detailer End

<a href="assets/readme/detailer-start-end.webp"><img align="left" hspace="16" src="assets/readme/detailer-start-end.webp" alt="Detailer Start and End" width="210"></a>
<ul>
  <li><code>Detailer Start</code> detects a <code>bounding box (bbox)</code> from regions where the input <code>mask</code> value is greater than <code>0.0</code>, then creates an expanded processing region using <code>crop_margin_scale</code>.</li>
  <li>The processing region is pre-upscaled by <code>upscale_factor</code>, then rounded up to a size divisible by <code>mini_unit</code> (8/16/32/64) and arranged with <code>center padding</code>.</li>
  <li>For compositing back into the original image, the graded values of the input <code>mask</code> are preserved as a <code>soft mask</code>, and that information is stored in <code>detailer_control</code>.</li>
  <li><code>Detailer End</code> receives <code>inpainted_image</code> and <code>composite</code>s it back into the original image using the preserved <code>soft mask</code>.</li>
  <li>If <code>inpainted_image</code> resolution differs from the expected <code>canvas</code> size, it is automatically resized before restoration.</li>
  <li>If no target region is detected, <code>Detailer End</code> does not evaluate <code>inpainted_image</code> and returns the original image unchanged.</li>
</ul>
<br clear="left">

### SAM3 Prompt To Mask

<a href="assets/readme/sam3-prompt-to-mask.webp"><img align="left" hspace="16" src="assets/readme/sam3-prompt-to-mask.webp" alt="SAM3 Prompt To Mask" width="210"></a>
<ul>
  <li><code>SAM3 Prompt To Mask</code> is a single-purpose node that generates a <code>soft mask</code> from text prompts. Comma-separated <code>prompt</code> tokens are treated as OR conditions, and accepted results can be controlled with <code>score_threshold</code> and <code>combine_mode</code> (<code>union</code> / <code>top1</code>).</li>
  <li>The output is a non-binarized <code>soft mask</code>; thresholding and region cleanup are intended to be handled by downstream nodes.</li>
  <li>This node is designed with a strong focus on keeping additional <code>pip</code>-installed dependencies to a minimum, because some existing SAM3-related extensions can significantly alter Python environments.</li>
  <li>Under that policy, SAM3-related functionality is intentionally limited to this single purpose, and no further feature additions are planned at this time.</li>
  <li>Place <code>sam3.pt</code> in <code>ComfyUI/models/sam3</code> (operation with SAM3.1 and later is unverified).</li>
</ul>
<br clear="left">

### PixAI Tagger

<a href="assets/readme/pixai-tagger.webp"><img align="left" hspace="16" src="assets/readme/pixai-tagger.webp" alt="PixAI Tagger" width="210"></a>
<ul>
  <li><code>PixAI Tagger</code> estimates <code>general</code> / <code>character</code> / <code>ip</code> tags from images and outputs prompt-ready text.</li>
  <li>Tag selection is controlled by <code>mode</code>: <code>threshold</code> uses score thresholds, while <code>topk</code> keeps top-ranked tags per category.</li>
  <li>This node does not depend on <code>onnxruntime</code> or <code>onnxruntime-gpu</code> and runs on a local PyTorch implementation.</li>
  <li>By avoiding additional ONNX Runtime installation, the design prioritizes resilience against Python environment changes.</li>
  <li>Use the PixAI Tagger v0.9 bundle (<code>model_v0.9.pth</code> / <code>tags_v0.9_13k.json</code> / <code>char_ip_map.json</code>) placed in <code>ComfyUI/models/pixai_tagger</code>.</li>
</ul>
<br clear="left">

### Prompt Template

<a href="assets/readme/prompt-template.webp"><img align="left" hspace="16" src="assets/readme/prompt-template.webp" alt="Prompt Template" width="210"></a>
<ul>
  <li><code>Prompt Template</code> generates final prompts from template text. It handles <code>//</code> / <code>/*...*/</code> comment removal, wildcard expansion, Dynamic Prompts expansion, and <code>$key</code> replacement in one node.</li>
  <li>Wildcards are loaded from <code>ComfyUI/user/info_prompt_toolkit/wildcards</code> <code>.txt</code> files and support <code>__name__</code> (random) and <code>__name__#N</code> (fixed selection), including weighted lines in <code>weight::value</code> format.</li>
  <li>Dynamic Prompts supports <code>{a|b}</code> plus weighted options, multi-select, range selection, and the <code>@</code> cycle sampler.</li>
  <li>When <code>seed</code> is connected, random expansion becomes reproducible; without it, results are non-deterministic.</li>
  <li>When <code>extras</code> is connected, <code>$key</code> placeholders are replaced while missing keys are left as-is. <code>suffix</code> is appended after template expansion.</li>
  <li>Outputs include raw <code>string</code> and token-normalized <code>normalized_string</code>.</li>
</ul>
<br clear="left">

### Image Reader / Image Saver

<a href="assets/readme/image-reader-and-image-saver.webp"><img align="left" hspace="16" src="assets/readme/image-reader-and-image-saver.webp" alt="Image Reader and Image Saver" width="210"></a>
<ul>
  <li><code>Image Reader</code> and <code>Image Saver</code> are core nodes that support reusable workflows for images that retain generation data (<code>A1111 infotext</code> / <code>image_info</code>).</li>
  <li><code>Image Saver</code> can embed <code>A1111 infotext</code> into WebP metadata in a format that is easier for Civitai Post images to recognize prompt, model, and multiple LoRA references.</li>
  <li><code>Check Referenced Models...</code> in the <code>Image Reader</code> right-click menu parses referenced entries from image infotext and lists Model / Refiner / Detailer / CLIP / VAE / LoRA items.</li>
  <li>The same window also shows local availability for each reference (Present / Missing), and can jump to <code>View Model Info...</code> for deeper inspection.</li>
  <li><code>Image Saver</code> supports both automatic serial naming and explicit <code>file_stem</code> naming, so you can switch naming strategy by workflow.</li>
  <li>When <code>write_caption</code> is enabled, a same-name <code>.txt</code> caption file is written alongside each image.</li>
</ul>
<br clear="left">

### Image Directory Reader

<a href="assets/readme/image-directory-reader.webp"><img align="left" hspace="16" src="assets/readme/image-directory-reader.webp" alt="Image Directory Reader" width="210"></a>
<ul>
  <li><code>Image Directory Reader</code> loads multiple images from subdirectories under ComfyUI <code>input</code> / <code>output</code> and returns corresponding data as aligned list outputs.</li>
  <li>You can choose the source with <code>path_source</code> and <code>path</code> (relative directory), then control the selection range with <code>start_index</code> and <code>image_load_limit</code> (<code>0</code> = unlimited).</li>
  <li>Target images are selected after case-insensitive filename sorting, making batch selection results easier to reproduce.</li>
  <li>For each image, it outputs <code>image</code>, <code>image_info</code> reconstructed and normalized from A1111 infotext, extensionless filename <code>file_stem</code>, and same-name <code>.txt</code> <code>caption</code> (empty string when missing).</li>
  <li>It also outputs <code>path_source</code> and <code>path</code>, so it can be connected directly to <code>Caption File Saver</code>.</li>
</ul>
<br clear="left">

### Video Reader / Video Saver

<a href="assets/readme/video-reader-and-video-saver.webp"><img align="left" hspace="16" src="assets/readme/video-reader-and-video-saver.webp" alt="Video Reader and Video Saver" width="210"></a>
<ul>
  <li><code>Video Reader</code> and <code>Video Saver</code> are a paired set of nodes that support video reuse workflows while retaining generation metadata (<code>source_image_info</code> / <code>video_info</code>).</li>
  <li><code>Video Reader</code> outputs <code>image</code> as a list. If downstream nodes expect batch-oriented image input or do not assume list expansion, route through <code>Image List To Batch</code>.</li>
  <li>In <code>Video Saver</code>, <code>crf</code> controls the quality/file-size tradeoff; lower values generally produce higher quality and larger files.</li>
  <li>In <code>Video Saver</code>, <code>preset</code> controls the encode speed/compression-efficiency tradeoff; higher values are generally faster, while lower values are generally more efficient.</li>
  <li><code>source_image_info</code> and <code>video_info</code> are handled as separate contexts and are not merged automatically.</li>
  <li>When connected to <code>Video Saver</code>, <code>source_image_info</code> and <code>video_info</code> are saved as separate infotexts, making condition tracking and comparison easier during reuse.</li>
</ul>
<br clear="left">

### Mask Overlay Comparer

<a href="assets/readme/mask-overlay-comparer.webp"><img align="left" hspace="16" src="assets/readme/mask-overlay-comparer.webp" alt="Mask Overlay Comparer" width="210"></a>
<ul>
  <li><code>Mask Overlay Comparer</code> displays a slider-based comparison between the original <code>image</code> and a mask-overlayed view, making mask coverage easier to inspect visually.</li>
  <li>The comparer switches between two views: an easier-to-read background image and the image with mask overlay applied.</li>
  <li>The <code>mask</code> is visualized without binarization, using raw values in the <code>0.0..1.0</code> range.</li>
  <li>Visualization behavior is: <code>mask = 0.0</code> keeps the image unchanged, <code>mask = 1.0</code> renders black, and intermediate values are shown as a blue-tinted overlay.</li>
  <li>For preview generation, use <code>image</code> and <code>mask</code> at the same resolution.</li>
</ul>
<br clear="left">

### Aspect Ratio to Size

<a href="assets/readme/aspect-ratio-to-size.webp"><img align="left" hspace="16" src="assets/readme/aspect-ratio-to-size.webp" alt="Aspect Ratio to Size" width="210"></a>
<ul>
  <li><code>Aspect Ratio to Size</code> recalculates <code>width</code> / <code>height</code> from node-UI widget edits before execution, so you can adjust values while checking <code>actual_ratio</code>.</li>
  <li>It resolves size from <code>width_ratio : height_ratio</code> and a base size, using either <code>width</code> or <code>height</code> as the anchor and deriving the other side from the ratio.</li>
  <li>The resolved size is aligned to <code>min_unit</code> (8 / 16 / 32 / 64), which helps keep resolutions on practical step sizes.</li>
  <li><code>width x height</code> is a combined size output that carries width and height on a single line, making it easy to pass into nodes that accept size input.</li>
</ul>
<br clear="left">

### Load New Model / Use Loaded Model

<a href="assets/readme/load-new-model-and-use-loaded-model.webp"><img align="left" hspace="16" src="assets/readme/load-new-model-and-use-loaded-model.webp" alt="Load New Model and Use Loaded Model" width="210"></a>
<ul>
  <li><code>Load New Model</code> and <code>Use Loaded Model</code> are a paired workflow that separates model loading from model reuse.</li>
  <li><code>Load New Model</code> loads <code>model</code>, <code>clip</code>, and <code>vae</code> from inputs provided by selector nodes.</li>
  <li>If you use <code>TorchCompile</code>-related nodes, place them after <code>Load New Model</code>, then connect their outputs to <code>Use Loaded Model</code> via <code>loaded_model</code>, <code>loaded_clip</code>, and <code>loaded_vae</code>.</li>
  <li><code>Use Loaded Model</code> switches by condition set (such as <code>model</code> plus <code>lora_stack</code>), which helps reduce duplicate loads when the same setup is reused.</li>
  <li><code>lora_stack</code> matching is order- and strength-sensitive, so changing order or values is treated as a different condition set.</li>
  <li>Supported targets are checkpoint-based and diffusion_models-based models.</li>
</ul>
<br clear="left">

## License

- The original source code of this project is offered under [GPL-3.0-or-later](LICENSE).
- This repository also includes vendored `sam3`, which is distributed under the [SAM License](vendor/LICENSE.sam3).
- See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for third-party license details, including notes on separately obtained optional models.
- Model files and other external assets obtained separately by users remain subject to their respective upstream licenses and terms.
