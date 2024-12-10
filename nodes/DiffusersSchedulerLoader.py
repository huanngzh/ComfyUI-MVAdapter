import torch

import folder_paths
from ..mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from ..utils import SCHEDULERS

class DiffusersSchedulerLoader:
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

    CATEGORY = "Diffusers"

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
