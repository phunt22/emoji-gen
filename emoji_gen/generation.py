from pathlib import Path
from datetime import datetime
from PIL import Image
from typing import Optional
from emoji_gen.models.model_manager import model_manager
import torch


# generates an emoji from a prompt using the active model (from model_manager specified in dev_cli)
def generate_emoji(
    prompt: str,
    output_path: Optional[str] = None,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    num_images: int = 1 ## maybe not needed ???
):
   
    try:
        # get model from the cache (inits default if needed)
        model = model_manager.get_active_model()
        if not model:
            return {"status": "error", "error": "Failed to initialize model"}
        
        # generate image
        result = model(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images ## maybe not needed ???
        )
        
        # save image
        image_path = save_image(prompt, result.images[0], output_path)
        return {"status": "success", "image_path": str(image_path)}
    
    except Exception as e:
        return {"status": "error", "error": str(e)}

def save_image(prompt, image, output_path: Optional[str] = None) -> Path:
    """Save the generated image to disk and return the path."""
    base_path = Path(output_path or "generated_emojis")
    base_path.mkdir(parents=True, exist_ok=True)
    
    image_name = "".join(c if c.isalnum() else "_" for c in prompt)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{image_name}_{timestamp}.png"
    image_path = base_path / file_name
    
    image.save(image_path)
    return image_path

def list_available_models():
    """List all available models with their information."""
    return model_manager.list_available_models() 