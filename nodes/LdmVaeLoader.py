import torch
import folder_paths
from ..utils import vae_pt_to_vae_diffuser

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

    CATEGORY = "Diffusers"

    def create_pipeline(self, vae_name, upcast_fp32):
        vae = vae_pt_to_vae_diffuser(
            folder_paths.get_full_path("vae", vae_name), force_upcast=upcast_fp32
        ).to(self.dtype)

        return (vae,)
