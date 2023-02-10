from models.stable_diffusion.constants import SD_MODELS_ALL, SD_MODEL_CACHE
from diffusers import StableDiffusionPipeline
import concurrent.futures


def download_sd_model_from_hf(key):
    model_id = SD_MODELS_ALL[key]["id"]
    print(f"⏳ Downloading model: {model_id}")
    StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=SD_MODELS_ALL[key]["torch_dtype"],
        cache_dir=SD_MODEL_CACHE,
    )
    print(f"✅ Downloaded model: {key}")
    return {"key": key}


def download_sd_models_from_hf():
    for key in SD_MODELS_ALL:
        download_sd_model_from_hf(key)


def download_sd_models_concurrently_from_hf():
    with concurrent.futures.ThreadPoolExecutor(10) as executor:
        # Start the download tasks
        download_tasks = [
            executor.submit(download_sd_model_from_hf, key) for key in SD_MODELS_ALL
        ]
        # Wait for all tasks to complete
        results = [
            task.result() for task in concurrent.futures.as_completed(download_tasks)
        ]
    executor.shutdown(wait=True)


if __name__ == "__main__":
    download_sd_models_from_hf()
    print("✅ Downloaded all models successfully")
