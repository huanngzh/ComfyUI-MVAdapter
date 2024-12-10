import torch
import folder_paths
from comfy.model_management import get_torch_device
from ..utils import MVADAPTERS

class DiffusersModelMakeup:
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
            },
        }

    RETURN_TYPES = ("PIPELINE",)

    FUNCTION = "makeup_pipeline"

    CATEGORY = "Diffusers"

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

        return (pipeline,)
