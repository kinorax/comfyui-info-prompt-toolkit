# Third-Party Licenses

This document summarizes major third-party code and assets that are bundled with, or explicitly referenced by, this repository.

It is not a full audit of all transitive dependencies in the base ComfyUI environment.

## Bundled Third-Party Code

| Component | Scope in this repository | Upstream license | Notes |
|---|---|---|---|
| `sam3` | Vendored under `vendor/sam3` | `SAM License` | License text is included at `vendor/LICENSE.sam3`. The separately obtained `sam3.pt` checkpoint remains subject to upstream terms. |

## Python Runtime Dependencies Declared in `requirements.txt`

| Package | Why it is required here | Reported upstream license | Upstream reference |
|---|---|---|---|
| `timm` | Local PixAI Tagger runtime and vendored `sam3` | `Apache-2.0` | https://pypi.org/project/timm/ |
| `ftfy` | Vendored `sam3` tokenizer path | `Apache-2.0` | https://pypi.org/project/ftfy/ |
| `regex` | Vendored `sam3` tokenizer path | `Apache-2.0` and `CNRI-Python` | https://github.com/mrabarnett/mrab-regex/blob/master/LICENSE.txt |
| `iopath` | Vendored `sam3` model loading utilities | `MIT` | https://github.com/facebookresearch/iopath |
| `typing_extensions` | Extension entrypoint typing helpers and vendored `sam3` | `PSF-2.0` | https://github.com/python/typing_extensions/blob/main/LICENSE |

## Optional Video Dependency

| Package | When it is used | Reported upstream license | Upstream reference |
|---|---|---|---|
| `av` | Video Reader / Video Saver runtime paths and video-backed thumbnail decode paths, when available in the environment | `BSD-3-Clause` | https://github.com/PyAV-Org/PyAV/blob/main/LICENSE.txt |

## Optional Models Obtained Separately By Users

### SAM3 Checkpoint

- `sam3.pt` is not bundled in this repository.
- The SAM3 checkpoint should be obtained and used under the upstream SAM terms.
- Official repository: https://github.com/facebookresearch/sam3

### PixAI Tagger v0.9 Bundle

- The PixAI Tagger model bundle is not bundled in this repository.
- The upstream Hugging Face model page currently reports `apache-2.0` and gated access.
- Files referenced by this repository include `model_v0.9.pth`, `tags_v0.9_13k.json`, and `char_ip_map.json`.
- Upstream model page: https://huggingface.co/pixai-labs/pixai-tagger-v0.9
- Users are responsible for following the upstream model terms and any data-origin notices associated with that model.
