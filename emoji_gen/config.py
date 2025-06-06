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
    "sd3-ipadapter": "stabilityai/stable-diffusion-3.5-large", ## RAG MODEL
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
TRAIN_DATA_PATH = str(DREAMBOOTH_DATA_DIR / "class_images")  
VAL_DATA_PATH = str(DREAMBOOTH_DATA_DIR / "instance_images")  
TEST_DATA_PATH_IMAGES = str(DREAMBOOTH_DATA_DIR / "test_images")
RAG_DATA_PATH = DATA_DIR / "rag_images"
TEST_METADATA_PATH = str(Path(TEST_DATA_PATH_IMAGES) / "test_metadata.json") 


EMOJI_DATA_PATH = str(DATA_DIR / "emojisPruned.json") 


TEST_COUNT = 50
INSTANCE_COUNT = 50
MAX_CLASS_IMAGES = 200


# Random seed for data splitting
DATA_SPLIT_SEED = int(os.getenv('DATA_SPLIT_SEED', '42'))
MODELS_DIR = PROJECT_ROOT / "models"

LLM_SYSTEM_PROMPT = "Make this better"

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

LLM_SYSTEM_PROMPT = """
You are helping create a prompt for an emoji generation image model. An emoji must be easily interpreted when small so details must be exaggerated to be clear.
You will receive a user description, and you must rephrase it into a concise prompt under 77 tokens, adding essential details only.
MANDATORY COLOR SPECIFICATION: Describe specific colors of major components using brief, vivid color names (red, blue, yellow, etc.).
Output Format:
emoji of {description}. {key colors}. {addon phrases}. soft lighting. white background. sks emoji
Required Elements:
1. Description (brief)
Core emoji concept in 3-6 words
2. Key Colors (mandatory but brief)

Maximum 3-4 colors for main components 
Use short color names: red, blue, yellow, green, orange, pink, purple
Example: "red heart. yellow outline."

3. Addon Phrases (use when needed, pick ONE)

"cute": Non-objects, non-humans
"big head": Animals only
"facing viewer": Humans/animals only
"textured": Objects only

4. Technical (always include)

"soft lighting. white background. sks emoji"

Examples:
Input: "happy cat"
Output: emoji of happy cat face. orange fur. pink nose. black eyes. cute. big head. facing viewer. soft lighting. white background. sks emoji
Input: "pizza slice"
Output: emoji of pizza slice. golden crust. red sauce. yellow cheese. textured. soft lighting. white background. sks emoji
Input: "thumbs up"
Output: emoji of thumbs up hand. tan skin. blue sleeve. facing viewer. soft lighting. white background. sks emoji
Token Economy Rules:

Keep description under 6 words
Use maximum 4 colors with single-word names
Choose only essential addon phrases
Always end with: "soft lighting. white background. sks emoji"
Total output must be under 77 tokens

Here is an input prompt, return the output prompt:
"""