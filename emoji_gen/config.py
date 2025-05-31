import torch
from pathlib import Path
from typing import Dict, Any
import os

# Get project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# GPU settings
CUDA_ENABLED = torch.cuda.is_available()
DEVICE = 'cuda' if CUDA_ENABLED else 'cpu'
# Using float32 for better compatibility, even though it uses more memory
DTYPE = torch.float32

# Model defaults
DEFAULT_MODEL = 'sd3'
MODEL_ID_MAP = {
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
    "SD-XL": "stabilityai/stable-diffusion-xl-base-1.0",    

    # quality of life :)
    "sd-xl": "stabilityai/stable-diffusion-xl-base-1.0",    
    # "test": "test_model_fake_model",
    # ADD BASE MODELS HERE!
}

# Path settings
MODEL_LIST_PATH = PROJECT_ROOT / "model_list.json"
FINE_TUNED_MODELS_DIR = PROJECT_ROOT / "fine_tuned_models"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "generated_emojis"

DATA_DIR = PROJECT_ROOT / "data"
DREAMBOOTH_DATA_DIR = DATA_DIR / "dreambooth"
TRAIN_DATA_PATH = str(DREAMBOOTH_DATA_DIR / "class_images")  # Directory with training images
VAL_DATA_PATH = str(DREAMBOOTH_DATA_DIR / "instance_images")  # Directory with validation images
TEST_DATA_PATH_IMAGES = str(DREAMBOOTH_DATA_DIR / "test_images") # For evaluation images
TEST_METADATA_PATH = str(Path(TEST_DATA_PATH_IMAGES) / "test_metadata.json") # For evaluation prompts/metadata


EMOJI_DATA_PATH = str(DATA_DIR / "emojisPruned.json") # Master pruned list used by dreambooth_preparation

# # Fine-tuning data split ratios, etc. (Primarily for the old JSON split method)
# # These ratios are less relevant if the primary path is direct image folders.
# TRAIN_RATIO = 0.8 # Adjusted to match dreambooth_preparation internal split
# VAL_RATIO = 0.1   # Adjusted
# TEST_RATIO = 0.1    # No test set for DreamBooth in the new setup

TEST_COUNT = 50
INSTANCE_COUNT = 50

# TOTAL_RATIO = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
# if not (0.999 <= TOTAL_RATIO <= 1.001):
#     raise ValueError(f"Configured split ratios (TRAIN, VAL, TEST) must sum to 1.0, got {TOTAL_RATIO}")

# Random seed for data splitting
DATA_SPLIT_SEED = int(os.getenv('DATA_SPLIT_SEED', '42'))
MODELS_DIR = PROJECT_ROOT / "models"

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


Path(DEFAULT_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(FINE_TUNED_MODELS_DIR).mkdir(parents=True, exist_ok=True)
Path(DREAMBOOTH_DATA_DIR).mkdir(parents=True, exist_ok=True)
# Instance, Validation, and Test image directories will be created by dreambooth_preparation.setup_folders() 
