# Adapted from https://github.com/Limitex/ComfyUI-Diffusers/blob/main/nodes.py
import copy
import os
import torch
from safetensors.torch import load_file
from torchvision import transforms
from .utils import (
    SCHEDULERS,
    PIPELINES,
    MVADAPTERS,
    vae_pt_to_vae_diffuser,
    convert_images_to_tensors,
    convert_tensors_to_images,
    prepare_camera_embed,
    preprocess_image,
)
from comfy.model_management import get_torch_device
import folder_paths

from diffusers import StableDiffusionXLPipeline, AutoencoderKL, ControlNetModel
from transformers import AutoModelForImageSegmentation

from .mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from .mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler


class DiffusersMVPipelineLoader:
    def __init__(self):
        self.hf_dir = folder_paths.get_folder_paths("diffusers")[0]
        self.dtype = torch.float16

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (
                    "STRING",
                    {"default": "stabilityai/stable-diffusion-xl-base-1.0"},
                ),
                "pipeline_name": (
                    list(PIPELINES.keys()),
                    {"default": "MVAdapterT2MVSDXLPipeline"},
                ),
            }
        }

    RETURN_TYPES = (
        "PIPELINE",
        "AUTOENCODER",
        "SCHEDULER",
    )

    FUNCTION = "create_pipeline"

    CATEGORY = "MV-Adapter"

    def create_pipeline(self, ckpt_name, pipeline_name):
        pipeline_class = PIPELINES[pipeline_name]
        pipe = pipeline_class.from_pretrained(
            pretrained_model_name_or_path=ckpt_name,
            torch_dtype=self.dtype,
            cache_dir=self.hf_dir,
        )
        return (pipe, pipe.vae, pipe.scheduler)


class LdmPipelineLoader:
    def __init__(self):
        self.hf_dir = folder_paths.get_folder_paths("diffusers")[0]
        self.dtype = torch.float16

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "pipeline_name": (
                    list(PIPELINES.keys()),
                    {"default": "MVAdapterT2MVSDXLPipeline"},
                ),
            }
        }

    RETURN_TYPES = (
        "PIPELINE",
        "AUTOENCODER",
        "SCHEDULER",
    )

    FUNCTION = "create_pipeline"

    CATEGORY = "MV-Adapter"

    def create_pipeline(self, ckpt_name, pipeline_name):
        pipeline_class = PIPELINES[pipeline_name]

        pipe = pipeline_class.from_single_file(
            pretrained_model_link_or_path=folder_paths.get_full_path(
                "checkpoints", ckpt_name
            ),
            torch_dtype=self.dtype,
            cache_dir=self.hf_dir,
        )

        return (pipe, pipe.vae, pipe.scheduler)


class DiffusersMVVaeLoader:
    def __init__(self):
        self.hf_dir = folder_paths.get_folder_paths("diffusers")[0]
        self.dtype = torch.float16

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (
                    "STRING",
                    {"default": "madebyollin/sdxl-vae-fp16-fix"},
                ),
            }
        }

    RETURN_TYPES = ("AUTOENCODER",)

    FUNCTION = "create_pipeline"

    CATEGORY = "MV-Adapter"

    def create_pipeline(self, vae_name):
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=vae_name,
            torch_dtype=self.dtype,
            cache_dir=self.hf_dir,
        )

        return (vae,)


class LdmVaeLoader:
    def __init__(self):
        self.dtype = torch.float16

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "upcast_fp32": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("AUTOENCODER",)

    FUNCTION = "create_pipeline"

    CATEGORY = "MV-Adapter"

    def create_pipeline(self, vae_name, upcast_fp32):
        vae = vae_pt_to_vae_diffuser(
            folder_paths.get_full_path("vae", vae_name), force_upcast=upcast_fp32
        ).to(self.dtype)

        return (vae,)


class DiffusersMVSchedulerLoader:
    def __init__(self):
        self.hf_dir = folder_paths.get_folder_paths("diffusers")[0]
        self.dtype = torch.float16

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "scheduler_name": (list(SCHEDULERS.keys()),),
                "shift_snr": ("BOOLEAN", {"default": True}),
                "shift_mode": (
                    list(ShiftSNRScheduler.SHIFT_MODES),
                    {"default": "interpolated"},
                ),
                "shift_scale": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.0, "max": 50.0, "step": 1.0},
                ),
            }
        }

    RETURN_TYPES = ("SCHEDULER",)

    FUNCTION = "load_scheduler"

    CATEGORY = "MV-Adapter"

    def load_scheduler(
        self, pipeline, scheduler_name, shift_snr, shift_mode, shift_scale
    ):
        scheduler = SCHEDULERS[scheduler_name].from_config(
            pipeline.scheduler.config, torch_dtype=self.dtype
        )
        if shift_snr:
            scheduler = ShiftSNRScheduler.from_scheduler(
                scheduler,
                shift_mode=shift_mode,
                shift_scale=shift_scale,
                scheduler_class=scheduler.__class__,
            )
        return (scheduler,)


class CustomLoraModelLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
                "enable": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "last_lora_node": (
                    "BOOLEAN",
                    {"default": True},
                ),
            }
        }

    RETURN_TYPES = ("PIPELINE",)
    FUNCTION = "load_lora"
    CATEGORY = "MV-Adapter"

    def load_lora(self, pipeline, lora_name, strength_model, enable, last_lora_node):
        if not hasattr(pipeline, "loaded_loras"):
            pipeline.loaded_loras = []

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_dir = os.path.dirname(lora_path)
        lora_name = os.path.basename(lora_path)
        lora = None
        if enable:
            if self.loaded_lora is not None:
                if self.loaded_lora[0] == lora_path:
                    lora = self.loaded_lora[1]
                else:
                    temp = self.loaded_lora
                    pipeline.delete_adapters(temp[1])
                    pipeline.loaded_loras = [(name, strength) for (name, strength) in pipeline.loaded_loras if name != temp[1]]
                    self.loaded_lora = None

            if lora is None:
                adapter_name = lora_name.rsplit(".", 1)[0]
                pipeline.load_lora_weights(
                    lora_dir, weight_name=lora_name, adapter_name=adapter_name
                )
                pipeline.set_adapters(adapter_name, strength_model)
                self.loaded_lora = (lora_path, adapter_name)
                lora = adapter_name

                pipeline.loaded_loras.append((adapter_name, strength_model))
        else:
            # Delete the loaded lora
            if self.loaded_lora is not None:
                temp = self.loaded_lora
                pipeline.delete_adapters(temp[1])
                pipeline.loaded_loras = [(name, strength) for (name, strength) in pipeline.loaded_loras if name != temp[1]]
                self.loaded_lora = None

        if last_lora_node:
            adapter_names = [x[0] for x in pipeline.loaded_loras]
            strengths = [x[1] for x in pipeline.loaded_loras]
            pipeline.set_adapters(adapter_names, strengths)

            print(adapter_names)

        return (pipeline,)


class ControlNetModelLoader:
    def __init__(self):
        self.loaded_controlnet = None
        self.dtype = torch.float16
        self.torch_device = get_torch_device()
        self.hf_dir = folder_paths.get_folder_paths("diffusers")[0]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "controlnet_name": (
                    "STRING",
                    {"default": "xinsir/controlnet-scribble-sdxl-1.0"},
                ),
            }
        }

    RETURN_TYPES = ("PIPELINE",)
    FUNCTION = "load_controlnet"
    CATEGORY = "MV-Adapter"

    def load_controlnet(self, pipeline, controlnet_name):
        controlnet = None
        if self.loaded_controlnet is not None:
            if self.loaded_controlnet == controlnet_name:
                controlnet = self.loaded_controlnet
            else:
                del pipeline.controlnet
                self.loaded_controlnet = None

        if controlnet is None:
            controlnet = ControlNetModel.from_pretrained(
                controlnet_name, cache_dir=self.hf_dir, torch_dtype=self.dtype
            )
            pipeline.controlnet = controlnet
            pipeline.controlnet.to(device=self.torch_device, dtype=self.dtype)

            self.loaded_controlnet = controlnet_name
            controlnet = controlnet_name

        return (pipeline,)


class DiffusersMVModelMakeup:
    def __init__(self):
        self.hf_dir = folder_paths.get_folder_paths("diffusers")[0]
        self.torch_device = get_torch_device()
        self.dtype = torch.float16

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "scheduler": ("SCHEDULER",),
                "autoencoder": ("AUTOENCODER",),
                "load_mvadapter": ("BOOLEAN", {"default": True}),
                "adapter_path": ("STRING", {"default": "huanngzh/mv-adapter"}),
                "adapter_name": (
                    MVADAPTERS,
                    {"default": "mvadapter_t2mv_sdxl.safetensors"},
                ),
                "num_views": ("INT", {"default": 6, "min": 1, "max": 12}),
            },
            "optional": {
                "enable_vae_slicing": ("BOOLEAN", {"default": True}),
                "enable_vae_tiling": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("PIPELINE",)

    FUNCTION = "makeup_pipeline"

    CATEGORY = "MV-Adapter"

    def makeup_pipeline(
        self,
        pipeline,
        scheduler,
        autoencoder,
        load_mvadapter,
        adapter_path,
        adapter_name,
        num_views,
        enable_vae_slicing=True,
        enable_vae_tiling=False,
    ):
        pipeline.vae = autoencoder
        pipeline.scheduler = scheduler

        if load_mvadapter:
            pipeline.init_custom_adapter(num_views=num_views)
            pipeline.load_custom_adapter(
                adapter_path, weight_name=adapter_name, cache_dir=self.hf_dir
            )
            pipeline.cond_encoder.to(device=self.torch_device, dtype=self.dtype)

        pipeline = pipeline.to(self.torch_device, self.dtype)

        if enable_vae_slicing:
            pipeline.enable_vae_slicing()
        if enable_vae_tiling:
            pipeline.enable_vae_tiling()

        return (pipeline,)


class DiffusersSampler:
    def __init__(self):
        self.torch_device = get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "a photo of a cat"},
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "watermark, ugly, deformed, noisy, blurry, low contrast",
                    },
                ),
                "width": ("INT", {"default": 768, "min": 1, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 768, "min": 1, "max": 2048, "step": 1}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 2000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 7.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "sample"

    CATEGORY = "MV-Adapter"

    def sample(
        self,
        pipeline,
        prompt,
        negative_prompt,
        height,
        width,
        steps,
        cfg,
        seed,
    ):
        images = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            negative_prompt=negative_prompt,
            generator=torch.Generator(self.torch_device).manual_seed(seed),
        ).images
        return (convert_images_to_tensors(images),)


class DiffusersMVSampler:
    def __init__(self):
        self.torch_device = get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "num_views": ("INT", {"default": 6, "min": 1, "max": 12}),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "an astronaut riding a horse"},
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "watermark, ugly, deformed, noisy, blurry, low contrast",
                    },
                ),
                "width": ("INT", {"default": 768, "min": 1, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 768, "min": 1, "max": 2048, "step": 1}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 2000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 7.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "controlnet_image": ("IMAGE",),
                "controlnet_conditioning_scale": ("FLOAT", {"default": 1.0}),
                "azimuth_degrees": ("LIST", {"default": [0, 45, 90, 180, 270, 315]}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "sample"

    CATEGORY = "MV-Adapter"

    def sample(
        self,
        pipeline,
        num_views,
        prompt,
        negative_prompt,
        height,
        width,
        steps,
        cfg,
        seed,
        reference_image=None,
        controlnet_image=None,
        controlnet_conditioning_scale=1.0,
        azimuth_degrees=[0, 45, 90, 180, 270, 315],
    ):
        num_views = len(azimuth_degrees)
        control_images = prepare_camera_embed(
            num_views, width, self.torch_device, azimuth_degrees
        )

        pipe_kwargs = {}
        if reference_image is not None:
            pipe_kwargs.update(
                {
                    "reference_image": convert_tensors_to_images(reference_image)[0],
                    "reference_conditioning_scale": 1.0,
                }
            )
        if controlnet_image is not None:
            controlnet_image = convert_tensors_to_images(controlnet_image)
            pipe_kwargs.update(
                {
                    "controlnet_image": controlnet_image,
                    "controlnet_conditioning_scale": controlnet_conditioning_scale,
                }
            )

        images = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            num_images_per_prompt=num_views,
            control_image=control_images,
            control_conditioning_scale=1.0,
            negative_prompt=negative_prompt,
            generator=torch.Generator(self.torch_device).manual_seed(seed),
            cross_attention_kwargs={"num_views": num_views},
            **pipe_kwargs,
        ).images
        return (convert_images_to_tensors(images),)


class BiRefNet:
    def __init__(self):
        self.hf_dir = folder_paths.get_folder_paths("diffusers")[0]
        self.torch_device = get_torch_device()
        self.dtype = torch.float32

    RETURN_TYPES = ("FUNCTION",)

    FUNCTION = "load_model_fn"

    CATEGORY = "MV-Adapter"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"ckpt_name": ("STRING", {"default": "briaai/RMBG-2.0"})}
        }

    def remove_bg(self, image, net, transform, device):
        image_size = image.size
        input_images = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = net(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image

    def load_model_fn(self, ckpt_name):
        model = AutoModelForImageSegmentation.from_pretrained(
            ckpt_name, trust_remote_code=True, cache_dir=self.hf_dir
        ).to(self.torch_device, self.dtype)

        transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        remove_bg_fn = lambda x: self.remove_bg(
            x, model, transform_image, self.torch_device
        )
        return (remove_bg_fn,)


class ImagePreprocessor:
    def __init__(self):
        self.torch_device = get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "remove_bg_fn": ("FUNCTION",),
                "image": ("IMAGE",),
                "height": ("INT", {"default": 768, "min": 1, "max": 2048, "step": 1}),
                "width": ("INT", {"default": 768, "min": 1, "max": 2048, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "process"

    def process(self, remove_bg_fn, image, height, width):
        images = convert_tensors_to_images(image)
        images = [
            preprocess_image(remove_bg_fn(img.convert("RGB")), height, width)
            for img in images
        ]

        return (convert_images_to_tensors(images),)


class ControlImagePreprocessor:
    def __init__(self):
        self.torch_device = get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "front_view": ("IMAGE",),
                "front_right_view": ("IMAGE",),
                "right_view": ("IMAGE",),
                "back_view": ("IMAGE",),
                "left_view": ("IMAGE",),
                "front_left_view": ("IMAGE",),
                "width": ("INT", {"default": 768, "min": 1, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 768, "min": 1, "max": 2048, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "process"

    def process(
        self,
        front_view,
        front_right_view,
        right_view,
        back_view,
        left_view,
        front_left_view,
        width,
        height,
    ):
        images = torch.cat(
            [
                front_view,
                front_right_view,
                right_view,
                back_view,
                left_view,
                front_left_view,
            ],
            dim=0,
        )
        images = convert_tensors_to_images(images)
        images = [img.resize((width, height)).convert("RGB") for img in images]
        return (convert_images_to_tensors(images),)


class ViewSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "front_view": ("BOOLEAN", {"default": True}),
                "front_right_view": ("BOOLEAN", {"default": True}),
                "right_view": ("BOOLEAN", {"default": True}),
                "back_view": ("BOOLEAN", {"default": True}),
                "left_view": ("BOOLEAN", {"default": True}),
                "front_left_view": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "process"
    CATEGORY = "MV-Adapter"

    def process(
        self,
        front_view,
        front_right_view,
        right_view,
        back_view,
        left_view,
        front_left_view,
    ):
        azimuth_deg = []
        if front_view:
            azimuth_deg.append(0)
        if front_right_view:
            azimuth_deg.append(45)
        if right_view:
            azimuth_deg.append(90)
        if back_view:
            azimuth_deg.append(180)
        if left_view:
            azimuth_deg.append(270)
        if front_left_view:
            azimuth_deg.append(315)

        return (azimuth_deg,)


NODE_CLASS_MAPPINGS = {
    "LdmPipelineLoader": LdmPipelineLoader,
    "LdmVaeLoader": LdmVaeLoader,
    "DiffusersMVPipelineLoader": DiffusersMVPipelineLoader,
    "DiffusersMVVaeLoader": DiffusersMVVaeLoader,
    "DiffusersMVSchedulerLoader": DiffusersMVSchedulerLoader,
    "DiffusersMVModelMakeup": DiffusersMVModelMakeup,
    "CustomLoraModelLoader": CustomLoraModelLoader,
    "DiffusersMVSampler": DiffusersMVSampler,
    "BiRefNet": BiRefNet,
    "ImagePreprocessor": ImagePreprocessor,
    "ControlNetModelLoader": ControlNetModelLoader,
    "ControlImagePreprocessor": ControlImagePreprocessor,
    "ViewSelector": ViewSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LdmPipelineLoader": "LDM Pipeline Loader",
    "LdmVaeLoader": "LDM Vae Loader",
    "DiffusersMVPipelineLoader": "Diffusers MV Pipeline Loader",
    "DiffusersMVVaeLoader": "Diffusers MV Vae Loader",
    "DiffusersMVSchedulerLoader": "Diffusers MV Scheduler Loader",
    "DiffusersMVModelMakeup": "Diffusers MV Model Makeup",
    "CustomLoraModelLoader": "Custom Lora Model Loader",
    "DiffusersMVSampler": "Diffusers MV Sampler",
    "BiRefNet": "BiRefNet",
    "ImagePreprocessor": "Image Preprocessor",
    "ControlNetModelLoader": "ControlNet Model Loader",
    "ControlImagePreprocessor": "Control Image Preprocessor",
    "ViewSelector": "View Selector",
}
