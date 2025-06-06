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
            
            if self._initialized and model_name == self._model_id and self.active_model is not None:
                return True, "Model already initialized"

            self.cleanup()
            
            if model_name in MODEL_ID_MAP:
                model_path = MODEL_ID_MAP[model_name]

                if model_name == "sd3-ipadapter":
                    print("SD3.5 IP-Adapter model requested. Loading directly from InstantX...")
                    try:
                        pipeline = DiffusionPipeline.from_pretrained(
                            torch_dtype=self._dtype,
                            use_safetensors=True,
                        )

                        # memops
                        if self._device == "cuda":
                            print("Applying memory optimizations")
                            pipeline.enable_model_cpu_offload()
                            if hasattr(pipeline, 'enable_attention_slicing'):
                                pipeline.enable_attention_slicing()
                        else:
                            pipeline.to(self._device)
                        self.active_model = pipeline
                    except Exception as e:
                        print(f"Error loading SD3.5 IP-Adapter {e}")
                        self.cleanup()
                        return False



                load_args = {
                    "torch_dtype": self._dtype,
                    "use_safetensors": True,
                }

                if model_name == "sd3-ipadapter":
                    print("IP-Adapter model requested. Loading base SD3 model first...")
                    
                    # 1. load  base SD3 model
                    base_model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
                    pipeline = StableDiffusion3Pipeline.from_pretrained(
                        base_model_id,
                        torch_dtype=self._dtype,
                        use_safetensors=True
                    )
                    
                    # 2. load IP-Adapter weights
                    ip_adapter_id = "InstantX/SD3.5-Large-IP-Adapter"
                    print(f"Loading and attaching IP-Adapter weights from '{ip_adapter_id}'...")
                    pipeline.load_ip_adapter(ip_adapter_id, subfolder="sd3_ip-adapter", weight_name="ip-adapter.safetensors")
                    
                    # 3. apply memory optimizations
                    if self._device == "cuda":
                        print("Applying memory optimizations for CUDA device (CPU offload, attention slicing)...")
                        pipeline.enable_model_cpu_offload()
                        pipeline.enable_attention_slicing()
                    else:
                        pipeline.to(self._device)

                    self.active_model = pipeline                    
                elif "xl" in model_name.lower(): 
                    pipeline_class = StableDiffusionXLPipeline
                    if self._device == "cuda":
                        load_args["variant"] = "fp16"
                elif "sd3" in model_name.lower():
                    pipeline_class = StableDiffusion3Pipeline
                else: 
                    pipeline_class = DiffusionPipeline

                print(f"Loading base model '{model_name}' with {pipeline_class.__name__}...")
                
                # load model on CPU first
                pipeline = pipeline_class.from_pretrained(model_path, **load_args)

                if self._device == "cuda":
                    print("Applying memory optimizations for CUDA device (CPU offload, attention slicing)...")
                    pipeline.enable_model_cpu_offload()
                    pipeline.enable_attention_slicing()
                else:
                    pipeline.to(self._device)

                self.active_model = pipeline
            
            # fine tuned model
            else:
                model_specific_dir = Path(FINE_TUNED_MODELS_DIR) / model_name

                if not model_specific_dir.exists() or not model_specific_dir.is_dir():
                    return False, f"Fine-tuned model directory for '{model_name}' not found at '{model_specific_dir}' or is not a directory."
                
                is_lora, weights_path, base_model = self._find_lora_weights_in_directory(model_specific_dir, specific_checkpoint)

                if is_lora:
                    print(f"Loading LoRA model '{model_name}' (base: {base_model}) from weights at: {weights_path}")
                    
                    # START TEMPORARY, REALLY BAD CODE
                    # config is out of sync bc trained on sd3.5 and sd3, I was lazy to make a better fix
                    if base_model == 'sd3':
                        base_model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
                    else:
                        base_model_path = MODEL_ID_MAP.get(base_model, base_model)
                    # END END END 

                    pipeline_class = self._get_pipeline_class_for_base_model(base_model)
                    
                    load_args = { "torch_dtype": self._dtype, "use_safetensors": True }
                    if self._device == "cuda" and base_model in ['sd-xl', 'sdxl']:
                        load_args["variant"] = "fp16"

                    # load base on cpu, then load lora wei
                    print(f"Loading base model '{base_model_path}' to CPU...")
                    pipeline = pipeline_class.from_pretrained(base_model_path, **load_args)
                    print(f"Fusing LoRA weights from '{weights_path}' on CPU...")
                    pipeline.load_lora_weights(weights_path)
                    
                    # apply optimizations, move to cuda
                    if self._device == "cuda":
                        print("Applying memory optimizations for CUDA (CPU offload, attention slicing)...")
                        pipeline.enable_model_cpu_offload()
                        pipeline.enable_attention_slicing()
                    else:
                        pipeline.to(self._device)
                    self.active_model = pipeline
                
                else:
                    base_model = self._infer_base_model_from_name(model_name)
                    pipeline_class = self._get_pipeline_class_for_base_model(base_model)
                    
                    load_args = { "torch_dtype": self._dtype, "use_safetensors": True }
                    if self._device == "cuda" and base_model in ['sd-xl', 'sdxl']:
                        load_args["variant"] = "fp16"

                    try:
                        # load on CPU first
                        pipeline = pipeline_class.from_pretrained(str(model_specific_dir), **load_args)
                        
                        # apply optimizations
                        if self._device == "cuda":
                           print("Applying memory optimizations for CUDA (CPU offload, attention slicing)...")
                           pipeline.enable_model_cpu_offload()
                           pipeline.enable_attention_slicing()
                        else:
                            pipeline.to(self._device)
                        self.active_model = pipeline
                    except Exception as e:
                        return False, f"Failed to load full fine-tune model '{model_name}': {str(e)}"

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