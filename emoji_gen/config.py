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



# For proejct
# Smiling cat with heart eyes emoji
# smiling cat with heart eyes sks emoji
# emoji of smiling cat with heart eyes. orange fur. red hearts. black eyes. cute. big head. facing viewer. soft lighting. white background. sks emoji
# emoji of smiling cat with heart eyes. orange fur. red hearts. black eyes. cute. big head. facing viewer. soft lighting. white background



# Overthinking gremlin with iced coffee emoji
# Chaotic raccoon in leather jacket emoji
# Tired wizard on hoverboard emoji
# Pizza slice loves donut emoji
# Crying sushi with umbrella emoji
# Judgmental avocado sips matcha emoji
# Skater cat kickflips rainbow emoji
# Frog in hoodie side-eyes void emoji
# Cowboy snail with lasso emoji
# Corgi in crown throws shade emoji
# Ghost DJ at haunted club emoji
# Cyberpunk mushroom with LED eyes emoji
# Alien sips bubble tea on UFO emoji
# Dragon does yoga in garden emoji
# Mermaid takes waterproof selfies emoji
# Buffering emoji stuck loading emoji
# Laptop with googly eyes panics emoji
# Goth hamster writes poetry emoji
# Banana lifts weights flexing emoji
# Sassy taco in high heels emoji
# Soft rage emoji holds calm sign emoji
# Romantic donut plans candlelit date emoji
# 404 error face looks existential emoji
# Bunny CEO on carrot phone emoji
# Wizard pancake casts syrup spells emoji
# AI with flower crown has crisis emoji
# Dead inside emoji with sparkly eyeliner emoji
# Suspicious cloud plans something sketchy emoji
# Judgy sun wears sunglasses at night emoji
# Happy volcano gives hugs emoji

# emoji of overthinking gremlin with iced coffee. green skin. brown coffee. black eyes. cute. facing viewer. soft lighting. white background. sks emoji
# emoji of chaotic raccoon in leather jacket. gray fur. black jacket. white face. big head. facing viewer. soft lighting. white background. sks emoji
# emoji of tired wizard on hoverboard. purple robe. gray beard. blue board. facing viewer. soft lighting. white background. sks emoji
# emoji of pizza slice hugging donut. yellow cheese. red sauce. pink donut. textured. soft lighting. white background. sks emoji
# emoji of crying sushi holding umbrella. white rice. pink fish. blue umbrella. cute. facing viewer. soft lighting. white background. sks emoji
# emoji of judgmental avocado sipping matcha. green skin. brown pit. yellow cup. cute. facing viewer. soft lighting. white background. sks emoji
# emoji of skater cat kickflipping rainbow. orange fur. black board. rainbow trail. big head. facing viewer. soft lighting. white background. sks emoji
# emoji of frog in hoodie side-eyeing void. green skin. blue hoodie. black eyes. big head. facing viewer. soft lighting. white background. sks emoji
# emoji of cowboy snail with lasso. brown shell. tan hat. yellow lasso. cute. big head. facing viewer. soft lighting. white background. sks emoji
# emoji of corgi in crown throwing shade. orange fur. gold crown. black glasses. big head. facing viewer. soft lighting. white background. sks emoji
# emoji of ghost DJ at haunted club. white ghost. black headphones. purple lights. cute. facing viewer. soft lighting. white background. sks emoji
# emoji of cyberpunk mushroom with LED eyes. red cap. white stem. blue eyes. textured. soft lighting. white background. sks emoji
# emoji of alien sipping bubble tea on UFO. green skin. pink drink. silver UFO. facing viewer. soft lighting. white background. sks emoji
# emoji of dragon doing yoga in garden. green scales. purple mat. yellow flowers. facing viewer. soft lighting. white background. sks emoji
# emoji of mermaid taking waterproof selfies. pink hair. blue tail. white phone. facing viewer. soft lighting. white background. sks emoji
# emoji of buffering face stuck loading. blue circle. gray face. white bar. cute. facing viewer. soft lighting. white background. sks emoji
# emoji of laptop with googly eyes panicking. gray screen. black keys. white eyes. textured. soft lighting. white background. sks emoji
# emoji of goth hamster writing poetry. brown fur. black outfit. purple pen. big head. facing viewer. soft lighting. white background. sks emoji
# emoji of banana lifting weights flexing. yellow peel. black weights. white teeth. cute. facing viewer. soft lighting. white background. sks emoji
# emoji of sassy taco in high heels. yellow shell. green lettuce. red shoes. cute. facing viewer. soft lighting. white background. sks emoji
# emoji of soft rage face holding calm sign. red face. white sign. black eyes. cute. facing viewer. soft lighting. white background. sks emoji
# emoji of romantic donut planning candlelit date. pink frosting. white candle. red heart. cute. facing viewer. soft lighting. white background. sks emoji
# emoji of 404 error face looking existential. blue face. white text. black eyes. cute. facing viewer. soft lighting. white background. sks emoji
# emoji of bunny CEO on carrot phone. white fur. orange phone. blue suit. big head. facing viewer. soft lighting. white background. sks emoji
# emoji of wizard pancake casting syrup spells. brown pancake. yellow syrup. purple hat. textured. facing viewer. soft lighting. white background. sks emoji
# emoji of AI with flower crown having crisis. silver face. pink flowers. blue eyes. facing viewer. soft lighting. white background. sks emoji
# emoji of dead inside face with sparkly eyeliner. gray face. purple glitter. black eyes. cute. facing viewer. soft lighting. white background. sks emoji
# emoji of suspicious cloud planning something sketchy. white cloud. black eyes. gray shadow. cute. facing viewer. soft lighting. white background. sks emoji
# emoji of judgy sun wearing sunglasses at night. yellow rays. black glasses. blue background. cute. facing viewer. soft lighting. white background. sks emoji
# emoji of happy volcano giving hugs. brown rock. red lava. pink cheeks. cute. facing viewer. soft lighting. white background. sks emoji
