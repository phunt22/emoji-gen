import torch
from pathlib import Path
from typing import Dict, Any
import os

# GPU settings
CUDA_ENABLED = torch.cuda.is_available()
DEVICE = 'cuda' if CUDA_ENABLED else 'cpu'
DTYPE = torch.float16 if CUDA_ENABLED else torch.float32

# Model defaults
DEFAULT_MODEL = 'sd-v1.5'
MODEL_ID_MAP = {
    "sd-v1.5": "runwayml/stable-diffusion-v1-5",
    "FLUX.1": "black-forest-labs/FLUX.1-dev",
    "SD-XL": "stabilityai/stable-diffusion-xl-base-1.0"
    # "test": "test_model_fake_model",
    # ADD MODELS HERE!
}

# Path settings
MODEL_LIST_PATH = Path("model_list.json")
FINE_TUNED_MODELS_DIR = Path("fine_tuned_models")
DEFAULT_OUTPUT_PATH = Path.cwd() / "generated_emojis"

# Fine-tuning dataset defaults
DEFAULT_DATASET = 'data/emojisPruned.json'

# Helper methods
def get_model_path(model_name: str) -> str:
    """Get the full model path for a given model name."""
    if model_name in MODEL_ID_MAP:
        return MODEL_ID_MAP[model_name]

    fine_tuned_path = FINE_TUNED_MODELS_DIR / model_name
    if fine_tuned_path.exists():
        return str(fine_tuned_path)
    raise ValueError(f"Unknown model: {model_name}")

def is_fine_tuned_model(model_name: str) -> bool:
    """Check if a model is a fine-tuned model."""
    return (FINE_TUNED_MODELS_DIR / model_name).exists()

def get_available_models() -> Dict[str, Any]:
    """Get all available models and their information."""
    models = {}
    # Base models from MODEL_ID_MAP
    for model_id, model_path in MODEL_ID_MAP.items():
        models[model_id] = {
            "path": model_path,
            "type": "base"
        }

    # Fine-tuned models from FINE_TUNED_MODELS_DIR
    if FINE_TUNED_MODELS_DIR.exists():
        for model_dir in FINE_TUNED_MODELS_DIR.iterdir():
            if model_dir.is_dir():
                models[model_dir.name] = {
                    "path": str(model_dir),
                    "type": "fine-tuned"
                }
    return models 
