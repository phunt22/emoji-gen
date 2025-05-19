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
    "SD-XL": "stabilityai/stable-diffusion-xl-base-1.0",
    "FLUX.1": "black-forest-labs/FLUX.1-dev",
    
    # "test": "test_model_fake_model",
    # ADD MODELS HERE!
}

# Path settings
MODEL_LIST_PATH = Path("model_list.json")
FINE_TUNED_MODELS_DIR = Path("fine_tuned_models")
DEFAULT_OUTPUT_PATH = Path.cwd() / "generated_emojis"

# Fine-tuning dataset defaults
DEFAULT_DATASET = 'data/emojisPruned.json'

# Data paths
DATA_DIR = Path("data")
DATA_SPLITS_DIR = DATA_DIR / "splits"
TRAIN_DATA_PATH = str(DATA_SPLITS_DIR / "train_emoji_data.json")
VAL_DATA_PATH = str(DATA_SPLITS_DIR / "val_emoji_data.json")
TEST_DATA_PATH = str(DATA_SPLITS_DIR / "test_emoji_data.json")
EMOJI_DATA_PATH = str(DATA_DIR / "emojisPruned.json")

# Fine-tuning data split ratios, etc.
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Validate split ratios
TOTAL_RATIO = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
if not (0.99 <= TOTAL_RATIO <= 1.01):
    raise ValueError(f"Split ratios must sum to 1.0, got {TOTAL_RATIO}")

# Random seed for data splitting
DATA_SPLIT_SEED = int(os.getenv('DATA_SPLIT_SEED', '42'))
MODELS_DIR = Path("models")

# helper methods
def get_model_path(model_name: str) -> str:
    if not model_name:
        raise ValueError("Model name cannot be empty")
        
    # Check base models first
    if model_name in MODEL_ID_MAP:
        return MODEL_ID_MAP[model_name]

    fine_tuned_path = FINE_TUNED_MODELS_DIR / model_name
    if fine_tuned_path.exists():
        if not fine_tuned_path.is_dir():
            raise ValueError(f"Model path exists but is not a directory: {fine_tuned_path}")
        return str(fine_tuned_path)
        
    raise ValueError(f"Model not found: {model_name}")

def is_fine_tuned_model(model_name: str) -> bool:
    
    if not model_name:
        return False
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
