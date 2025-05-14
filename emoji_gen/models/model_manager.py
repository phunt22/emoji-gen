from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, FluxPipeline
from typing import Optional, Dict, Tuple, Union
from emoji_gen.config import MODEL_ID_MAP, FINE_TUNED_MODELS_DIR, DEFAULT_MODEL

class ModelManager:
    def __init__(self):
        self._active_model = None
        self._model_id = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._initialized = False

    def initialize_model(self, model_name: str = DEFAULT_MODEL) -> Tuple[bool, str]:
        try:
            # return early if we already have the model
            if self._initialized and model_name == self._model_id:
                return True, "Model already initialized"

            
            if model_name in MODEL_ID_MAP:
                model_path = MODEL_ID_MAP[model_name]
            else:
                model_path = str(FINE_TUNED_MODELS_DIR / model_name)
                if not Path(model_path).exists():
                    return False, f"Model not found: {model_name}"


            # clean up existing model if any
            if self._active_model is not None:
                del self._active_model
                torch.cuda.empty_cache()

            # load new model

            # if model_path.includes
            # TODO conditionally load based on the model

            model_path_lower = str(model_path).lower()
            if "stable" in model_path_lower:
                # if SD model
                self._active_model = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=self._dtype
                ).to(self._device)
            elif "flux" in model_path_lower:
                # if FLUX model
                self._active_model = FluxPipeline.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16
                ).to(self._device)
            else:
                print(f"Cannot find model {model_path}. Make sure that you put it in MODEL_ID_MAP in config.py")
                return

            self._model_id = model_name
            self._initialized = True
            return True, f"Successfully initialized model: {model_name}"

        except Exception as e:
            self.cleanup()
            print(f"Error initializing model: {str(e)}")
            return False, f"Error initializing model: {str(e)}"

    def get_active_model(self) -> Optional[Union[FluxPipeline, StableDiffusionPipeline]]:
        """Get the currently active model."""
        if not self._initialized:
            success, _ = self.initialize_model()  # try to initialize default model at the start
            if not success:
                return None
        return self._active_model

    def cleanup(self):
        """Clean up resources."""
        if self._active_model is not None:
            del self._active_model
            torch.cuda.empty_cache()
        self._active_model = None
        self._model_id = None
        self._initialized = False

    def list_available_models(self) -> Dict[str, Dict]:
        """List all available models and their information."""
        models = {}
        
        # base models from MODEL_ID_MAP
        for model_id, model_path in MODEL_ID_MAP.items():
            models[model_id] = {
                "path": model_path,
                "type": "base"
            }

        # fine-tuned models
        if FINE_TUNED_MODELS_DIR.exists():
            for model_dir in FINE_TUNED_MODELS_DIR.iterdir():
                if model_dir.is_dir():
                    models[model_dir.name] = {
                        "path": str(model_dir),
                        "type": "fine-tuned"
                    }
        
        return models

# global model manager instance
model_manager = ModelManager() 