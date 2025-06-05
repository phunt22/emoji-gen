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

    def _find_lora_weights_in_directory(self, model_dir: Path, specific_checkpoint: Optional[str] = None) -> Tuple[bool, Optional[Path], Optional[str]]:
        possible_lora_files = [
            "pytorch_lora_weights.safetensors",
            # not needed I think?
            "adapter_model.safetensors", 
            "adapter_model.bin"
        ]

        base_model = self._infer_base_model_from_name(model_dir.name)
        
        # If specific checkpoint requested, check that first
        if specific_checkpoint:
            checkpoint_dir = model_dir / f"checkpoint-{specific_checkpoint}"
            if checkpoint_dir.exists() and checkpoint_dir.is_dir():
                for lora_file in possible_lora_files:
                    if (checkpoint_dir / lora_file).exists():
                        print(f"Found LoRA weights in requested checkpoint: {checkpoint_dir / lora_file}")
                        return True, checkpoint_dir, base_model
            
            # check nested structure: model_dir/model_name/checkpoint-X/
            for subdir in model_dir.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('checkpoint-'):
                    nested_checkpoint_dir = subdir / f"checkpoint-{specific_checkpoint}"
                    if nested_checkpoint_dir.exists() and nested_checkpoint_dir.is_dir():
                        for lora_file in possible_lora_files:
                            if (nested_checkpoint_dir / lora_file).exists():
                                print(f"Found LoRA weights in requested nested checkpoint: {nested_checkpoint_dir / lora_file}")
                                return True, nested_checkpoint_dir, base_model
            
            print(f"Warning: Requested checkpoint-{specific_checkpoint} not found, falling back to latest")
        
        
        all_checkpoints = []
        
        # direct checkpoints
        all_checkpoints.extend([d for d in model_dir.glob("checkpoint-*") if d.is_dir()])
        
        # nested checkpoints
        for subdir in model_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('checkpoint-'):
                all_checkpoints.extend([d for d in subdir.glob("checkpoint-*") if d.is_dir()])
        
        # sort by number
        all_checkpoints = sorted(all_checkpoints, key=lambda p: int(p.name.split("-")[-1]))

        # check direct directory
        for ckpt_dir in reversed(all_checkpoints):  ## start from latest checkpoint
            for lora_file in possible_lora_files:
                if (ckpt_dir / lora_file).exists():
                    print(f"Found LoRA weights in checkpoint: {ckpt_dir / lora_file}")
                    return True, ckpt_dir, base_model
                
        # check direct directory
        for lora_file in possible_lora_files:
            if (model_dir / lora_file).exists():
                print(f"Found LoRA weights in main directory: {model_dir / lora_file}")
                return True, model_dir, base_model
        
        # check subdirectories (shouldnt happen, but did initially in testing)
        for subdir in model_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('checkpoint-'):
                for lora_file in possible_lora_files:
                    if (subdir / lora_file).exists():
                        print(f"Found LoRA weights in subdirectory: {subdir / lora_file}")
                        return True, subdir, base_model
        
        return False, None, None

    def _get_pipeline_class_for_base_model(self, base_model: str):
        """
        Get the appropriate pipeline class for a base model
        """
        if base_model == 'sd3':
            return StableDiffusion3Pipeline
        elif base_model in ['sd-xl', 'sdxl']:
            return StableDiffusionXLPipeline
        else:
            print(f"Warning: Unknown base model '{base_model}', defaulting to StableDiffusionXLPipeline")
            return StableDiffusionXLPipeline

    def initialize_model(self, model_name: str = DEFAULT_MODEL) -> Tuple[bool, str]:
        try:
            specific_checkpoint = None
            if ':checkpoint-' in model_name:
                model_name, checkpoint_part = model_name.split(':checkpoint-', 1)
                specific_checkpoint = checkpoint_part
                print(f"Requested specific checkpoint: {specific_checkpoint}")
            
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
                elif "xl" in model_name.lower(): 
                    print(f"Loading {model_name} pipeline")
                    self.active_model = StableDiffusionXLPipeline.from_pretrained(
                        model_path,
                        torch_dtype=self._dtype,
                        use_safetensors=True, 
                        variant="fp16" if self._device == "cuda" else None ## ensure we are on GPU
                    ).to(self._device)
                elif "sd3" in model_name.lower(): # base sd3, not the ip-adapter one
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
                
                is_lora, weights_path, base_model = self._find_lora_weights_in_directory(model_specific_dir, specific_checkpoint)

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
                    
                    # Get appropriate pipeline class
                    pipeline_class = self._get_pipeline_class_for_base_model(base_model)

                    # Load base model with appropriate pipeline
                    self.active_model = pipeline_class.from_pretrained(
                        base_model_path,
                        torch_dtype=self._dtype,
                        use_safetensors=True, 
                        variant="fp16" if self._device == "cuda" and base_model in ['sd-xl', 'sdxl'] else None
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