from pathlib import Path
import torch
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionXLPipeline, 
    StableDiffusion3Pipeline,
    FluxPipeline
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

    def _get_pipeline_class_from_model_id(self, model_id: str):
        model_id_lower = model_id.lower()
        
        if "xl" in model_id_lower or "sdxl" in model_id_lower:
            return StableDiffusionXLPipeline
        elif "sd3" in model_id_lower or "stable-diffusion-3" in model_id_lower:
            return StableDiffusion3Pipeline
        else:
        #    default to SDXL
            print(f"Warning: Defaulting to StableDiffusionXLPipeline for model ID: {model_id}. If this is not an XL model, pipeline selection might be incorrect.")
            return StableDiffusionXLPipeline

    def initialize_model(self, model_name: str = DEFAULT_MODEL) -> Tuple[bool, str]:
        try:
            # return early if we already have the model
            if self._initialized and model_name == self._model_id:
                return True, "Model already initialized"

            self.cleanup()
            
            # 1: Determine if it's a base model, local model, or Hub model
            if model_name in MODEL_ID_MAP:
                model_path = MODEL_ID_MAP[model_name]
                pipeline_class = self._get_pipeline_class_from_model_id(model_path)
                
                print(f"INFO: Loading base model: {model_path} using {pipeline_class.__name__}")
                self.active_model = pipeline_class.from_pretrained(
                    model_path,
                    torch_dtype=self._dtype,
                    use_safetensors=True, 
                    variant="fp16" if self._device == "cuda" and pipeline_class == StableDiffusionXLPipeline else None # variant typically for XL fp16
                ).to(self._device)
                
            else:
                # Check local fine-tuned directory first
                local_path = FINE_TUNED_MODELS_DIR / model_name
                if local_path.exists() and local_path.is_dir():
                    is_lora, weights_path, base_model_from_metadata = self._find_lora_weights_in_directory(local_path)
                    
                    if is_lora:
                        if not base_model_from_metadata:
                            return False, f"Error: LoRA detected for '{model_name}' at '{weights_path}', but base model could not be determined from metadata.json."
                        
                        print(f"INFO: Loading local LoRA model '{model_name}' based on: {base_model_from_metadata}")
                        print(f"INFO: LoRA weights at: {weights_path}")
                        
                        base_model_resolved_path = MODEL_ID_MAP.get(base_model_from_metadata, base_model_from_metadata)
                        pipeline_class = self._get_pipeline_class_from_model_id(base_model_resolved_path)
                        
                        print(f"INFO: Loading base for LoRA: {base_model_resolved_path} using {pipeline_class.__name__}")
                        self.active_model = pipeline_class.from_pretrained(
                            base_model_resolved_path,
                            torch_dtype=self._dtype,
                            use_safetensors=True,
                            variant="fp16" if self._device == "cuda" and pipeline_class == StableDiffusionXLPipeline else None
                        ).to(self._device)
                        
                        # Load LoRA weights
                        print(f"INFO: Applying LoRA weights from {weights_path} to {model_name}")
                        self.active_model.load_lora_weights(str(weights_path))
                        
                    else:
                        # Try as FFT
                        print(f"INFO: Loading local full fine-tuned model: {local_path}")
                        pipeline_class = self._get_pipeline_class_from_model_id(model_name) 
                        
                        self.active_model = pipeline_class.from_pretrained(
                            str(local_path),
                            torch_dtype=self._dtype,
                            use_safetensors=True
                        ).to(self._device)
                
                else:
                    # Try Hugging Face Hub
                    try:
                        # from huggingface_hub import RepoCard # Already imported at top level
                        print(f"INFO: Checking Hugging Face Hub for: {model_name}")
                        
                        card = RepoCard.load(model_name)
                        base_model_from_hf_card = card.data.to_dict().get("base_model")
                        
                        if base_model_from_hf_card:
                            
                            print(f"INFO: Loading Hub LoRA model '{model_name}' based on: {base_model_from_hf_card}")
                            
                            base_model_resolved_path_hub = MODEL_ID_MAP.get(base_model_from_hf_card, base_model_from_hf_card)
                            pipeline_class = self._get_pipeline_class_from_model_id(base_model_resolved_path_hub)

                            print(f"INFO: Loading base for Hub LoRA: {base_model_resolved_path_hub} using {pipeline_class.__name__}")
                            self.active_model = pipeline_class.from_pretrained(
                                base_model_resolved_path_hub,
                                torch_dtype=self._dtype,
                                use_safetensors=True,
                                variant="fp16" if self._device == "cuda" and pipeline_class == StableDiffusionXLPipeline else None
                            ).to(self._device)
                            
                            # Load LoRA weights from Hub
                            print(f"INFO: Applying Hub LoRA weights for {model_name}")
                            self.active_model.load_lora_weights(model_name) # model_name is the Hub ID for the LoRA
                            
                        else:
                            # Hub full model
                            print(f"INFO: Loading full model from Hub: {model_name}")
                            pipeline_class = self._get_pipeline_class_from_model_id(model_name)
                            
                            self.active_model = pipeline_class.from_pretrained(
                                model_name, # model_name is the Hub ID
                                torch_dtype=self._dtype,
                                use_safetensors=True
                            ).to(self._device)
                    
                    except EntryNotFoundError:
                        return False, f"Model '{model_name}' not found locally or on Hugging Face Hub (EntryNotFoundError)."
                    except ImportError:
                         return False, "Huggingface_hub library is not installed. Please install it to use models from Hugging Face Hub."
                    except Exception as e:
                        return False, f"Error accessing or loading '{model_name}' from Hugging Face Hub: {str(e)}"
            
            if self.active_model is None: # should ideally be caught by specific errors above
                return False, f"Model '{model_name}' failed to load (self.active_model is None after attempts)."
            
            self._model_id = model_name
            self._initialized = True
            
            if is_lora:
                 return True, f"Successfully initialized LoRA model: {model_name} (base: {base_model_from_metadata if 'base_model_from_metadata' in locals() else base_model_from_hf_card})"
            elif model_name not in MODEL_ID_MAP and (not local_path.exists() or not local_path.is_dir()): # Hub model check
                 return True, f"Successfully initialized Hub model: {model_name}"
            else: 
                 return True, f"Successfully initialized model: {model_name}"

        except Exception as e:
            self.cleanup()
            return False, f"Critical error during model initialization for '{model_name}': {str(e)}"

    def get_active_model(self) -> Optional[Union[FluxPipeline, StableDiffusionPipeline]]:
        """Get the currently active model."""
        if not self._initialized:
            success, _ = self.initialize_model()  # try to initialize default model at the start
            if not success:
                return None
        return self.active_model

    def get_available_models(self) -> Dict[str, Dict]:
        """List all available models and their information."""
        models = {}
        
        # base models from MODEL_ID_MAP
        for model_id_key, model_path_val in MODEL_ID_MAP.items(): 
            pipeline_class = self._get_pipeline_class_from_model_id(model_path_val)
            models[model_id_key] = {
                "path": model_path_val,
                "type": "base",
                "pipeline": pipeline_class.__name__ if pipeline_class else "Unknown"
            }

        # fine-tuned models
        if FINE_TUNED_MODELS_DIR.exists():
            for model_dir_item in FINE_TUNED_MODELS_DIR.iterdir(): 
                if model_dir_item.is_dir():
                    model_name_key = model_dir_item.name 
                    is_lora, weights_path, base_model = self._find_lora_weights_in_directory(model_dir_item)
                    
                    if is_lora:
                        # determine pipeline based on base_model, or default if base_model is None (but _find_lora should ideally find it)
                        pipeline_base_identifier = base_model if base_model else "sdxl" 
                        if base_model and base_model in MODEL_ID_MAP: # if alias like "sd-xl"
                            pipeline_base_identifier = MODEL_ID_MAP[base_model]

                        pipeline_class = self._get_pipeline_class_from_model_id(pipeline_base_identifier)
                        models[model_name_key] = {
                            "path": str(model_dir_item),
                            "type": "fine-tuned-lora",
                            "base_model": base_model, #  from metadata.json
                            "weights_path": str(weights_path) if weights_path else None,
                            "pipeline": pipeline_class.__name__ if pipeline_class else "Unknown"
                        }
                    else:
                        # not lora, FFT. 
                        # TODO add more support, implement FFT
                        pipeline_class = self._get_pipeline_class_from_model_id(model_name_key)
                        models[model_name_key] = {
                            "path": str(model_dir_item),
                            "type": "fine-tuned-full",
                            "pipeline": pipeline_class.__name__ if pipeline_class else "Unknown"
                        }
        
        return models

# global model manager instance
model_manager = ModelManager() 