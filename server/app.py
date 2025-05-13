from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from models.cache import model_cache
import torch
from datetime import datetime
from pathlib import Path
from models.fine_tune import EmojiFineTuner

app = FastAPI(title="Emoji Gen API", description="Generate emojis from text")

class GenerationRequest(BaseModel):
    prompt: str
    model_choice: str
    output_path: Optional[str] = None

class GenerationResponse(BaseModel):
    image_path: Optional[str] = None
    error: Optional[str] = None

class ModelRequest(BaseModel):
    model_name: str

class FineTuneRequest(BaseModel):
    base_model: str
    dataset_path: str
    model_name: str


# saves image and returns the path
def save_image(prompt, image, output_path: Optional[str] = None) -> Path:
    base_path = Path(output_path or "generated_emojis")
    base_path.mkdir(parents=True, exist_ok=True)

    image_name = "".join(c if c is c.isalnum() else "_" for c in prompt)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{image_name}_{timestamp}.png"
    image_path = base_path / file_name

    image.save(image_path)
    return image_path


@app.post("/generate", response_model=GenerationResponse)
async def generate_emoji(
        request: GenerationRequest
    ):
    try:
        pass
        # get model
        model = model_cache.get_model(request.model_choice)
        if not model:
            return GenerationResponse(error=f"Model {request.model_choice} not found")
        
        # generate image
        # tune the steps for inference, etc.
        result = model(
            prompt=request.prompt,
            num_inference_steps=25,
            guidance_scale=7.5,
            num_images_per_prompt=1
        )

        # save image
        image_path = save_image(request.prompt, result.images[0], request.output_path)
        return GenerationResponse(image_path=str(image_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/list")
async def list_models():
    try:
        return {
        "models": model_cache.list_models(),
        "current_model": model_cache.get_current_model_id(),
        "gpu_available": torch.cuda.is_available(),
    }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")
    

@app.post("/status")
async def status():
    try:
        return {
        "status": "ok",
        "message": "Server is running",
        "models": model_cache.list_models(),
        "current_model": model_cache.get_current_model_id(),
        "gpu_available": torch.cuda.is_available(),
    }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check status:{str(e)}")
    

@app.post("/fine-tune")
async def fine_tune_model(request: FineTuneRequest):
    try:
        fine_tuner = EmojiFineTuner(request.base_model)
        output_path = fine_tuner.train(
            dataset_path=request.dataset_path,
            model_name=request.model_name
        )
        
        # Add the new model to the cache
        model_cache.MODEL_ID_MAP[request.model_name] = output_path
        
        return {
            "status": "success",
            "message": f"Model fine-tuned and saved to {output_path}",
            "model_name": request.model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fine-tuned-models")
async def list_fine_tuned_models():
    try:
        fine_tuner = EmojiFineTuner("runwayml/stable-diffusion-v1-5")
        return {
            "models": fine_tuner.list_fine_tuned_models()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    