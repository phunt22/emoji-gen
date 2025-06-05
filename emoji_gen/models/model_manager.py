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


    def _infer_base_model_from_name(self, model_name: str) -> str:
        model_name_lower = model_name.lower()
        model_id = None
        
        if 'sd3' in model_name_lower or 'sd-3' in model_name_lower:
            model_id =  'sd3'
        else:
            model_id =  'sd-xl'

        return model_id

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        try:
            if hasattr(self, 'active_model') and self.active_model is not None:
                del self.active_model

                import gc
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache() ## empty again bc why not

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

        base_model = self._infer_base_model_from_name(model_dir.name)
        
        # check direct directory first
        all_checkpoints = sorted([d for d in model_dir.glob("checkpoint-*") if d.is_dir()], key=lambda p: int(p.name.split("-")[-1]))

        # check direct directory
        for ckpt_dir in reversed(all_checkpoints):  ## start from latest checkpoint
            for lora_file in possible_lora_files:
                if (ckpt_dir / lora_file).exists():
                    base_model = self._infer_base_model_from_name(model_dir)
                    return True, ckpt_dir, base_model
                
        # check direct directory
        for lora_file in possible_lora_files:
            if (model_dir / lora_file).exists():
                base_model = self._infer_base_model_from_name(model_dir)
                return True, model_dir, base_model
        
        # check subdirectories (shouldnt happen, but did initially in testing)
        for subdir in model_dir.iterdir():
            if subdir.is_dir():
                for lora_file in possible_lora_files:
                    if (subdir / lora_file).exists():
                        base_model = self._infer_base_model_from_name(model_dir)
                        return True, subdir, base_model
        
        return False, None, None

    # def _read_base_model_from_metadata(self, weights_dir: Path) -> Optional[str]:        
    #     metadata_path = weights_dir / "metadata.json"
    #     if metadata_path.exists():
    #         try:
    #             with open(metadata_path, 'r') as f:
    #                 metadata = json.load(f)
    #                 return metadata.get('base_model')
    #         except Exception as e:
    #             print(f"Warning: Could not read metadata.json: {e}")
    #     return None

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
            
            if model_name in MODEL_ID_MAP:
                model_path = MODEL_ID_MAP[model_name]

                if model_name == "sd3-ipadapter": ## RAG MODEL
                    # this is from docs
                    print(f"Loading {model_name} pipeline")
                    self.active_model = DiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=self._dtype, ## potentially bfloat16
                        # use_safetensors=True ##  maybe???
                    ).to(self._device)
                elif "xl" in model_name: 
                    print(f"Loading {model_name} pipeline")
                    self.active_model = StableDiffusionXLPipeline.from_pretrained(
                        model_path,
                        torch_dtype=self._dtype,
                        use_safetensors=True, 
                        variant="fp16" if self._device == "cuda" else None ## ensure we are on GPU
                    ).to(self._device)
                elif "sd3" in model_name: # base sd3, not the ip-adapter one
                    print(f"Loading {model_name} pipeline")
                    self.active_model = StableDiffusion3Pipeline.from_pretrained(
                        model_path,
                        torch_dtype=self._dtype,
                        use_safetensors=True, 
                        # variant=None # SD3 typically doesn't use variants like XL fp16
                    ).to(self._device)
                else: 
                    ## fallback to other models. Shouldnt happen since fine tuned models are hanlded in else branch
                    print(f"Warning: Loading model '{model_path}' with generic DiffusionPipeline.from_pretrained. Ensure this is appropriate.")
                    self.active_model = DiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=self._dtype
                    ).to(self._device)

                self._model_id = model_name
                self._initialized = True

                return True, f"Successfully init {model_name} (base)"
                
                # self.active_model = pipeline_class.from_pretrained(
                #     model_path,
                #     torch_dtype=self._dtype,
                #     use_safetensors=True, 
                #     variant="fp16" if self._device == "cuda" and pipeline_class == StableDiffusionXLPipeline else None # variant typically for XL fp16
                # ).to(self._device)
            
            # fine tuned model
            else:
                model_specific_dir = Path(FINE_TUNED_MODELS_DIR) / model_name

                if not model_specific_dir.exists() or not model_specific_dir.is_dir():
                    return False, f"Fine-tuned model directory for '{model_name}' not found at '{model_specific_dir}' or is not a directory."
                
                is_lora, weights_path, base_model = self._find_lora_weights_in_directory(model_specific_dir)

                if is_lora:
                    if not base_model:
                        error_message = (
                            f"LoRA Model '{model_name}' found (weights at '{weights_path}'), "
                            f"but its base model could not be determined. Ensure 'metadata.json' "
                            f"with a 'base_model' key exists in the model's main directory: '{model_specific_dir}'."
                        )
                        return False, error_message

                    print(f"Loading LoRA model '{model_name}' (base: {base_model})")
                    print(f"LoRA weights at: {weights_path}")

                    base_model_path = MODEL_ID_MAP.get(base_model, base_model)

                    # from docs
                    self.active_model = DiffusionPipeline.from_pretrained(
                        base_model_path,
                        torch_dtype=self._dtype,
                        use_safetensors=True, 
                        variant="fp16" if self._device == "cuda" and "sdxl" in model_name else None # var
                    ).to(self._device)
                    self.active_model.load_lora_weights(str(weights_path))
                
                else:
                    # add functionality for other FT like FFT or plain dreambooth
                    return False, f"FFT not implemented yet"

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