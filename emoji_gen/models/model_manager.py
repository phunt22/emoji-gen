from pathlib import Path
import torch
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionXLPipeline, 
    StableDiffusion3Pipeline,
    FluxPipeline,
    DiffusionPipeline
)
from typing import Optional, Dict, Tuple, Union
from emoji_gen.config import MODEL_ID_MAP, FINE_TUNED_MODELS_DIR, DEFAULT_MODEL
import gc
import json
from huggingface_hub import RepoCard
from huggingface_hub.utils import EntryNotFoundError

class ModelManager:
    def __init__(self):
        self.active_model = None
        self._model_id = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._initialized = False

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        try:
            if hasattr(self, 'active_model') and self.active_model is not None:
                del self.active_model
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
        finally:
            self.active_model = None
            self._model_id = None
            self._initialized = False

    def _find_lora_weights_in_directory(self, model_dir: Path) -> Tuple[bool, Optional[Path], Optional[str]]:
       
        possible_lora_files = [
            # main one
            "pytorch_lora_weights.safetensors",
            # alternative, not needed yet
            "adapter_model.safetensors",
            "adapter_model.bin"
        ]
        
        # check direct directory first
        for lora_file in possible_lora_files:
            if (model_dir / lora_file).exists():
                # try to read base model from metadata.json
                base_model = self._read_base_model_from_metadata(model_dir)
                return True, model_dir, base_model
        
        # check subdirectories (shouldnt happen, but did initially in testing)
        for subdir in model_dir.iterdir():
            if subdir.is_dir():
                for lora_file in possible_lora_files:
                    if (subdir / lora_file).exists():
                        base_model = self._read_base_model_from_metadata(subdir)
                        return True, subdir, base_model
        
        return False, None, None

    def _read_base_model_from_metadata(self, weights_dir: Path) -> Optional[str]:
        
        metadata_path = weights_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get('base_model')
            except Exception as e:
                print(f"Warning: Could not read metadata.json: {e}")
        return None

    # def _get_pipeline_class_from_model_id(self, model_id: str):
    #     model_id_lower = model_id.lower()
        
    #     if "xl" in model_id_lower or "sdxl" in model_id_lower:
    #         return StableDiffusionXLPipeline
    #     elif "sd3" in model_id_lower or "stable-diffusion-3" in model_id_lower:
    #         return StableDiffusion3Pipeline
    #     else:
    #     #    default to SDXL
    #         print(f"Warning: Defaulting to StableDiffusionXLPipeline for model ID: {model_id}. If this is not an XL model, pipeline selection might be incorrect.")
    #         return StableDiffusionXLPipeline

    def initialize_model(self, model_name: str = DEFAULT_MODEL) -> Tuple[bool, str]:
        try:
            # return early if we already have the model
            if self._initialized and model_name == self._model_id:
                return True, "Model already initialized"

            self.cleanup()
            
            # base model from config
            if model_name in MODEL_ID_MAP:
                model_path = MODEL_ID_MAP[model_name]

                if "xl" in model_name: 
                    self.active_model = StableDiffusionXLPipeline.from_pretrained(
                        model_path,
                        torch_dtype=self._dtype,
                        use_safetensors=True, 
                        variant="fp16" if self._device == "cuda" else None ## ensure we are on GPU
                    ).to(self._device)
                else: ## sd3
                    self.active_model = StableDiffusion3Pipeline.from_pretrained(
                        model_path,
                        torch_dtype=self._dtype,
                        use_safetensors=True, 
                        variant=None
                    ).to(self._device)
                
                # self.active_model = pipeline_class.from_pretrained(
                #     model_path,
                #     torch_dtype=self._dtype,
                #     use_safetensors=True, 
                #     variant="fp16" if self._device == "cuda" and pipeline_class == StableDiffusionXLPipeline else None # variant typically for XL fp16
                # ).to(self._device)
            
            # fine tuned model
            else:
                local_path = FINE_TUNED_MODELS_DIR / model_name
                if not local_path.exists():
                    return False, f"Model '{model_name}' not found in {FINE_TUNED_MODELS_DIR}"
                
                is_lora, weights_path, base_model = self._find_lora_weights_in_directory(local_path)

                if is_lora:
                    if not base_model:
                        return False, f"LoRA Model '{model_name}' found but base model wasn't"

                    print(f"Loading LoRA model '{model_name}' (base: {base_model})")
                    print(f"LoRA weights at: {weights_path}")

                    base_model_path = MODEL_ID_MAP.get(base_model, base_model)

                    self.active_model = DiffusionPipeline.from_pretrained(
                        base_model_path,
                        torch_dtype=self._dtype,
                        use_safetensors=True, 
                        variant="fp16" if self._device == "cuda" and "sdxl" in model_name else None # var
                    )
                
                else:
                    # add functionality for other FT like FFT or plain dreambooth
                    pass

                if self.active_model is None:
                    return False, f"Failed to load model '{model_name}'"
                
                self._model_id = model_name
                self._initialized = True

                pipeline_type = type(self.active_model).__name__
                print(f"Successfully loaded {pipeline_type}")

                return True, f"Successfully loaded {pipeline_type}"
            
        except Exception as e:
            self.cleanup()
            return False, f"Error loading model '{model_name}': {str(e)}"


    def get_active_model(self) -> Optional[Union[FluxPipeline, StableDiffusionPipeline]]:
        if not self._initialized:
            success, _ = self.initialize_model()  # try to initialize default model at the start
            if not success:
                return None
        return self.active_model

    def get_available_models(self) -> Dict[str, Dict]:
        # empty dict, append base and fine models
        models = {}
        
        # base models from MODEL_ID_MAP
        for model_id_key, model_path_val in MODEL_ID_MAP.items(): 
            models[model_id_key] = {
                "path": model_path_val,
                "type": "base",
            }

        # fine-tuned models
        if FINE_TUNED_MODELS_DIR.exists():
            for model_dir in FINE_TUNED_MODELS_DIR.iterdir(): 
                if model_dir.is_dir():
                    is_lora, weights_path, base_model = self._find_lora_weights_in_directory(model_dir)
                    

                    if is_lora:
                        models[model_dir.name] = {
                            "path": str(model_dir),
                            "type": "lora",
                            "base_model": base_model,
                            "weights_path": str(weights_path)
                        }
                    else:
                        models[model_dir.name] = {
                            "path": str(model_dir),
                            "type": "full-finetune"
                        }

        return models

# global model manager instance
model_manager = ModelManager() 