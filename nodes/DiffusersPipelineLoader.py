import torch
import folder_paths
from ..mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from ..utils import PIPELINES

class DiffusersPipelineLoader:
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

    CATEGORY = "Diffusers"

    def create_pipeline(self, ckpt_name, pipeline_name):
        pipeline_class = PIPELINES[pipeline_name]
        pipe = pipeline_class.from_pretrained(
            pretrained_model_name_or_path=ckpt_name,
            torch_dtype=self.dtype,
            cache_dir=self.hf_dir,
        )
        return (pipe, pipe.vae, pipe.scheduler)

