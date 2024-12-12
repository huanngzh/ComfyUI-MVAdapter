from comfy.model_management import get_torch_device
from ..utils import convert_images_to_tensors, convert_tensors_to_images, preprocess_image

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

