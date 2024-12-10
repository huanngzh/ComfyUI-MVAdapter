import torch
from comfy.model_management import get_torch_device
from ..utils import convert_images_to_tensors, prepare_camera_embed

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
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "sample"

    CATEGORY = "Diffusers"

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
    ):
        control_images = prepare_camera_embed(num_views, width, self.torch_device)

        pipe_kwargs = {}
        if reference_image is not None:
            pipe_kwargs.update(
                {
                    "reference_image": convert_tensors_to_images(reference_image)[0],
                    "reference_conditioning_scale": 1.0,
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
            **pipe_kwargs,
        ).images
        return (convert_images_to_tensors(images),)
