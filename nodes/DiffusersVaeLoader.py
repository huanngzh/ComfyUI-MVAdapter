import torch

import folder_paths
from diffusers import AutoencoderKL

class DiffusersVaeLoader:
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

    CATEGORY = "Diffusers"

    def create_pipeline(self, vae_name):
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=vae_name,
            torch_dtype=self.dtype,
            cache_dir=self.hf_dir,
        )

        return (vae,)

