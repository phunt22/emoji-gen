from pathlib import Path
from datetime import datetime
from PIL import Image
from typing import Optional
from emoji_gen.models.cache import model_cache
import torch

def generate_emoji(
    prompt: str,
    model_choice: str,
    output_path: Optional[str] = None,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    num_images: int = 1
):
    """Generate an emoji based on the prompt using the specified model."""
    try:
        # Get model from cache
        model = model_cache.get_model(model_choice)
        if not model:
            return {"status": "error", "error": f"Model {model_choice} not found"}
        
        # Generate image
        result = model(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images
        )
        
        # Save image
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
    return {
        "models": model_cache.list_models(),
        "current_model": model_cache.get_current_model_id(),
        "gpu_available": torch.cuda.is_available(),
    } 