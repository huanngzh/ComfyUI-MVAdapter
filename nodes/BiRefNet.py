import torch
import folder_paths
from transformers import AutoModelForImageSegmentation

from comfy.model_management import get_torch_device
from torchvision import transforms

class BiRefNet:
    def __init__(self):
        self.hf_dir = folder_paths.get_folder_paths("diffusers")[0]
        self.torch_device = get_torch_device()
        self.dtype = torch.float32

    RETURN_TYPES = ("FUNCTION",)

    FUNCTION = "load_model_fn"

    CATEGORY = "Diffusers"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"ckpt_name": ("STRING", {"default": "ZhengPeng7/BiRefNet"})}
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
