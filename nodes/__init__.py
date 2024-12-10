from .BiRefNet import BiRefNet
from .DiffusersModelMakeup import DiffusersModelMakeup
from .DiffusersMVSampler import DiffusersMVSampler
from .DiffusersPipelineLoader import DiffusersPipelineLoader
from .DiffusersSampler import DiffusersSampler
from .DiffusersSchedulerLoader import DiffusersSchedulerLoader
from .DiffusersVaeLoader import DiffusersVaeLoader
from .ImagePreprocessor import ImagePreprocessor
from .LdmPipelineLoader import LdmPipelineLoader
from .LdmVaeLoader import LdmVaeLoader
from .LoraModelLoader import LoraModelLoader

NODE_CLASS_MAPPINGS = {
    "LdmPipelineLoader": LdmPipelineLoader,
    "LdmVaeLoader": LdmVaeLoader,
    "DiffusersPipelineLoader": DiffusersPipelineLoader,
    "DiffusersVaeLoader": DiffusersVaeLoader,
    "DiffusersSchedulerLoader": DiffusersSchedulerLoader,
    "DiffusersModelMakeup": DiffusersModelMakeup,
    "LoraModelLoader": LoraModelLoader,
    "DiffusersSampler": DiffusersSampler,
    "DiffusersMVSampler": DiffusersMVSampler,
    "BiRefNet": BiRefNet,
    "ImagePreprocessor": ImagePreprocessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LdmPipelineLoader": "LDM Pipeline Loader",
    "LdmVaeLoader": "LDM Vae Loader",
    "DiffusersPipelineLoader": "Diffusers Pipeline Loader",
    "DiffusersVaeLoader": "Diffusers Vae Loader",
    "DiffusersSchedulerLoader": "Diffusers Scheduler Loader",
    "DiffusersModelMakeup": "Diffusers Model Makeup",
    "LoraModelLoader": "Lora Model Loader",
    "DiffusersSampler": "Diffusers Sampler",
    "DiffusersMVSampler": "Diffusers MV Sampler",
    "BiRefNet": "BiRefNet",
    "ImagePreprocessor": "Image Preprocessor",
}

