from typing import Tuple, Dict, Any, Callable
import os

from lingua import LanguageDetectorBuilder, LanguageDetector
from PIL import Image
import numpy as np
from boto3_type_annotations.s3 import ServiceResource

from shared.constants import WORKER_VERSION
from models.stable_diffusion.helpers import download_sd_models_concurrently
from models.stable_diffusion.constants import SD_MODELS, SD_MODEL_CACHE
from diffusers import (
    StableDiffusionPipeline,
)
from models.swinir.helpers import get_args_swinir, define_model_swinir
from models.swinir.constants import TASKS_SWINIR, MODELS_SWINIR, DEVICE_SWINIR
from models.download.download import download_models
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution


def setup(
    s3: ServiceResource, bucket_name: str
) -> Tuple[
    Dict[str, Any],
    Callable[[np.ndarray | Image.Image, Any, Any], Image.Image],
    Any,
    LanguageDetector,
]:
    print(f"⏳ Setup has started - Version: {WORKER_VERSION}")

    download_models(s3, bucket_name)

    txt2img_pipes: dict[
        str,
        StableDiffusionPipeline,
    ] = {}

    for key in SD_MODELS:
        print(f"⏳ Loading SD model: {key}")
        pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODELS[key]["id"],
            torch_dtype=SD_MODELS[key]["torch_dtype"],
            cache_dir=SD_MODEL_CACHE,
        )
        txt2img_pipes[key] = pipe.to("cuda")
        txt2img_pipes[key].enable_xformers_memory_efficient_attention()
        print(f"✅ Loaded SD model: {key}")

    # For upscaler
    upscaler_args = get_args_swinir()
    upscaler_args.task = TASKS_SWINIR["Real-World Image Super-Resolution-Large"]
    upscaler_args.scale = 4
    upscaler_args.model_path = MODELS_SWINIR["real_sr"]["large"]
    upscaler_args.large_model = True
    upscaler_pipe = define_model_swinir(upscaler_args)
    upscaler_pipe.eval()
    upscaler_pipe = upscaler_pipe.to(DEVICE_SWINIR)
    print("✅ Loaded upscaler")

    upscaler_processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr")
    upscaler_pipe = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr")
    upscaler_pipe = upscaler_pipe.to("cuda")
    upscaler = {
        "pipe": upscaler_pipe,
        "processor": upscaler_processor
    }

    # For translator
    language_detector_pipe = (
        LanguageDetectorBuilder.from_all_languages()
        .with_preloaded_language_models()
        .build()
    )
    print("✅ Loaded language detector")

    print("✅ Setup is done!")

    return txt2img_pipes, upscaler, language_detector_pipe
