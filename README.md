# ComfyUI-MVAdapter

This extension integrates [MV-Adapter](https://github.com/huanngzh/MV-Adapter) into ComfyUI, allowing users to generate multi-view consistent images from text prompts or single images directly within the ComfyUI interface.

## Features

* Generate multi-view consistent images from text prompts
* Generate multi-view consistent images from a single image
* More features are coming...

## Installation

### From Source

* Clone or download this repository into your `ComfyUI/custom_nodes/` directory.
* Install the required dependencies by running `pip install -r requirements.txt`.

## Notes

### Workflows

We provide the example workflows in `workflows` directory.

Note that our code depends on diffusers, and will automatically download the model weights from Hugging Face to the hf cache path at the first time. The `ckpt_name` in the node corresponds to the model name in Hugging Face, such as `stabilityai/stable-diffusion-xl-base-1.0`.

We also provide the nodes `Ldm**Loader` to support loading text-to-image models in `ldm` format. Please see the workflow files with the suffix `_ldm.json`.

### GPU Memory

If your GPU resources are limited, we recommend using the following configuration:

* Use [madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) as VAE. If using ldm-format pipeline, remember to set `upcast_fp32` to `False`.

![upcast_fp32_to_false](assets/comfyui_ldm_vae.png)

* Set `enable_vae_slicing` in the Diffusers Model Makeup node to `True`.

![enable_vae_slicing](assets/comfyui_model_makeup.png)

However, since SDXL is used as the base model, it still requires about 13G to 14G GPU memory.

## Usage

### Text to Multi-view Images

![comfyui_t2mv](assets/comfyui_t2mv.png)

* `workflows/t2mv_sdxl_diffusers.json` for loading diffusers-format models
* `workflows/t2mv_sdxl_ldm.json` for loading ldm-format models

### Image to Multi-view Images

![comfyui_i2mv](assets/comfyui_i2mv.png)

* `workflows/i2mv_sdxl_diffusers.json` for loading diffusers-format models
* `workflows/i2mv_sdxl_ldm.json` for loading ldm-format models
