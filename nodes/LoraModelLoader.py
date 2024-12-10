import folder_paths
import os

class LoraModelLoader:
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
            }
        }

    RETURN_TYPES = ("PIPELINE",)
    FUNCTION = "load_lora"
    CATEGORY = "Diffusers"

    def load_lora(self, pipeline, lora_name, strength_model):
        if strength_model == 0:
            return (pipeline,)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_dir = os.path.dirname(lora_path)
        lora_name = os.path.basename(lora_path)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                pipeline.delete_adapters(temp[1])
                self.loaded_lora = None

        if lora is None:
            adapter_name = lora_name.rsplit(".", 1)[0]
            pipeline.load_lora_weights(
                lora_dir, weight_name=lora_name, adapter_name=adapter_name
            )
            pipeline.set_adapters(adapter_name, strength_model)
            self.loaded_lora = (lora_path, adapter_name)
            lora = adapter_name

        return (pipeline,)
