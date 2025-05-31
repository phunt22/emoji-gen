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
    # ADD MODELS HERE!
}

# Path settings
MODEL_LIST_PATH = PROJECT_ROOT / "model_list.json"
FINE_TUNED_MODELS_DIR = PROJECT_ROOT / "fine_tuned_models"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "generated_emojis"

# --- New DreamBooth Data Paths ---
# These paths point to directories of images, not JSON files.
# The `emoji-dev prepare` command (using `dreambooth_preparation.download_emojis`)
# will create these directories and populate them with images.
DATA_DIR = PROJECT_ROOT / "data"
DREAMBOOTH_DATA_DIR = DATA_DIR / "dreambooth"
TRAIN_DATA_PATH = str(DREAMBOOTH_DATA_DIR / "instance_images")  # Directory with training images
VAL_DATA_PATH = str(DREAMBOOTH_DATA_DIR / "validation_images")  # Directory with validation images
TEST_DATA_PATH_IMAGES = str(DREAMBOOTH_DATA_DIR / "test_images") # For evaluation images
TEST_METADATA_PATH = str(Path(TEST_DATA_PATH_IMAGES) / "test_metadata.json") # For evaluation prompts/metadata

# --- Legacy Data Paths (for JSON-based approach, if needed for other utilities) ---
# Fine-tuning dataset defaults (original JSON path)
# DEFAULT_DATASET = str(PROJECT_ROOT / 'data/emojisPruned.json')
# Original data paths that pointed to JSON splits
# OLD_DATA_SPLITS_DIR = DATA_DIR / "splits_json" # Renamed to avoid conflict if old files exist
# OLD_TRAIN_JSON_PATH = str(OLD_DATA_SPLITS_DIR / "train_emoji_data.json")
# OLD_VAL_JSON_PATH = str(OLD_DATA_SPLITS_DIR / "val_emoji_data.json")
# OLD_TEST_JSON_PATH = str(OLD_DATA_SPLITS_DIR / "test_emoji_data.json")
EMOJI_DATA_PATH = str(DATA_DIR / "emojisPruned.json") # Master pruned list used by dreambooth_preparation

# Fine-tuning data split ratios, etc. (Primarily for the old JSON split method)
# The new dreambooth_preparation.download_emojis uses a fixed 0.9 train_val_split internally.
# These ratios are less relevant if the primary path is direct image folders.
TRAIN_RATIO = 0.8 # Adjusted to match dreambooth_preparation internal split
VAL_RATIO = 0.1   # Adjusted
TEST_RATIO = 0.1    # No test set for DreamBooth in the new setup

TOTAL_RATIO = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
if not (0.999 <= TOTAL_RATIO <= 1.001):
    raise ValueError(f"Configured split ratios (TRAIN, VAL, TEST) must sum to 1.0, got {TOTAL_RATIO}")

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

def get_models_config() -> Dict[str, Any]:
    return {
        "default_model": DEFAULT_MODEL,
        "model_id_map": MODEL_ID_MAP
    }

def get_data_config() -> Dict[str, Any]:
    return {
        "train_data_path": TRAIN_DATA_PATH,
        "val_data_path": VAL_DATA_PATH,
        "test_data_path_images": TEST_DATA_PATH_IMAGES,
        "test_metadata_path": TEST_METADATA_PATH,
        "emoji_data_path_json": EMOJI_DATA_PATH,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "data_split_seed": DATA_SPLIT_SEED
    }

Path(DEFAULT_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(FINE_TUNED_MODELS_DIR).mkdir(parents=True, exist_ok=True)
Path(DREAMBOOTH_DATA_DIR).mkdir(parents=True, exist_ok=True)
# Instance, Validation, and Test image directories will be created by dreambooth_preparation.setup_folders() 
