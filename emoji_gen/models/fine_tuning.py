import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from emoji_gen.config import (
    DATA_DIR
)

EMOJI_DATA = Path(DATA_DIR / "emoji")

class EmojiFineTuner:
    
    def __init__(self, base_model_id: str):
        self.base_model_id = base_model_id
        self.logger = logging.getLogger(__name__)
        
        # Determine model type from model ID
        self.model_type = self._detect_model_type(base_model_id)
        self.scripts_dir = Path(__file__).parent.parent.parent / "scripts" 
        self.output_base_dir = Path("fine_tuned_models") # this will be relative to where the script is run
        self.output_base_dir.mkdir(exist_ok=True)

    def _detect_model_type(self, model_id: str) -> str:
        """Detect whether model is SDXL or SD3 based on model ID"""
        model_id_lower = model_id.lower()
        
        if "sdxl" in model_id_lower or "stable-diffusion-xl" in model_id_lower:
            return "sdxl"
        elif "sd3" in model_id_lower or "stable-diffusion-3" in model_id_lower:
            return "sd3"
        else:
            self.logger.warning(f"Unknown model type for {model_id}, defaulting to SDXL. Training script might not be optimal.")
            return "sdxl" # Defaulting, but training script choice might be an issue
    
    # def _legacy_prepare_emoji_data_from_json(self, train_data_json_path: str, temp_dir: Path) -> Path:
    #     """(Legacy) Convert emoji JSON data to image directory structure for DreamBooth."""
    #     self.logger.info(f"Using legacy JSON data preparation from {train_data_json_path}")
    #     instance_dir = temp_dir / "instance_images_from_json"
    #     instance_dir.mkdir(exist_ok=True)
        
    #     with open(train_data_json_path, 'r') as f:
    #         emoji_data = json.load(f)
        
    #     copied_count = 0
    #     for i, emoji_item in enumerate(emoji_data):
    #         img_path_str = emoji_item.get('image_path')
    #         if img_path_str:
    #             src_path = Path(img_path_str)
    #             if not src_path.is_absolute() and not src_path.exists():
    #                 pass 

    #             if src_path.exists():
    #                 dst_path = instance_dir / f"emoji_{i:03d}{src_path.suffix}"
    #                 try:
    #                     shutil.copy2(src_path, dst_path)
    #                     copied_count += 1
    #                 except Exception as e:
    #                     self.logger.error(f"Failed to copy {src_path} to {dst_path}: {e}")
    #             else:
    #                 self.logger.warning(f"Image path not found in JSON, skipping: {src_path} (Original path in JSON: {img_path_str})")
    #         else:
    #             self.logger.warning(f"Missing 'image_path' for item {i} in JSON, skipping.")

    #     self.logger.info(f"Prepared {copied_count} training images from JSON into {instance_dir}")
    #     if copied_count == 0 and emoji_data: 
    #          self.logger.error(f"JSON at {train_data_json_path} was processed, but no images were copied. Check 'image_path' entries.")
    #     return instance_dir
    
    def _get_training_script_path(self) -> Path:
        """Get the appropriate training script based on model type"""
        if self.model_type == "sdxl":
            script_name = "train_dreambooth_lora_sdxl.py"
        else:
            self.logger.info(f"Model type is {self.model_type}. Using train_dreambooth_lora_sd3.py. Adjust if a specific script for {self.model_type} exists.")
            script_name = "train_dreambooth_lora_sd3.py"
        
        script_path = self.scripts_dir / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Training script not found: {script_path}. Expected in {self.scripts_dir}")
        
        return script_path

    def _build_training_command(self, 
                              instance_data_dir: Path,
                              output_dir: Path,
                              validation_data_dir: Optional[Path] = None,
                              **kwargs) -> List[str]:
        """Build the training command based on model type and parameters"""

        from emoji_gen.config import (
            VAL_DATA_PATH, ## instance
            TRAIN_DATA_PATH ## class
        )

        script_path = self._get_training_script_path()

        cmd = [
            "accelerate", "launch", str(script_path),
            "--pretrained_model_name_or_path", self.base_model_id,
             "--output_dir", str(output_dir),
            # instance and class data
            # train and val are slight misnomers but its okay :)
            # "--instance_data_dir", VAL_DATA_PATH,
            # "--instance_prompt", str(kwargs.get('instance_prompt', 'sks emoji')),
            # "--class_data_dir", TRAIN_DATA_PATH,
            # "--class_prompt", "emoji",

            "--dataset_name", str(EMOJI_DATA),
            "--caption_column", "text",
            "--image_column", "file_name",
            "--instance_prompt", "an sks emoji"



            # training parameters
            "--mixed_precision", kwargs.get("mixed_precision", "fp16"),
            "--train_batch_size", str(kwargs.get('batch_size', 1)),
            "--gradient_accumulation_steps", str(kwargs.get('gradient_accumulation_steps', 4)),
            "--learning_rate", str(kwargs.get('learning_rate', 1e-4)),
            "--lr_scheduler", kwargs.get("lr_scheduler", "cosine"),
            "--lr_warmup_steps", str(kwargs.get("lr_warmup_steps", "0")),

            # should be ~5k steps for SDXL and ~4k for SD3
            "--max_train_steps", str(kwargs.get('max_train_steps', 6000)), ## can always rollback to checkpoint
            # checkpoint is every 500 by default
            "--seed", str(kwargs.get('seed', 42)),
        ]

        # conditionally add resolution
        resolution = kwargs.get('resolution')
        if resolution is not None:
            cmd.extend(["--resolution", str(resolution)])

        # instance_prompt = kwargs.get('instance_prompt', 'sks emoji')
        # cmd.extend(["--instance_prompt", instance_prompt])
        

        # NOTE:
        # Validation temporarily removed, since we arent actually making descisions based off of it.
        # Using the validation set for instance 

        # validation_prompt = kwargs.get('validation_prompt')
        # if validation_prompt: # Only add validation flags if a prompt is given
        #     cmd.extend(["--validation_prompt", validation_prompt])
        #     # Check if validation_data_dir (for image folder) is provided
        #     if validation_data_dir and validation_data_dir.exists() and any(validation_data_dir.iterdir()):
        #         cmd.extend(["--validation_data_dir", str(validation_data_dir)])
        #         self.logger.info(f"Using validation image directory: {validation_data_dir}")
            
        #     # validation_epochs is a common arg, but some scripts might use validation_steps
        #     validation_epochs = kwargs.get('validation_epochs')
        #     if validation_epochs is not None:
        #          cmd.extend(["--validation_epochs", str(validation_epochs)])
        #     else:
        #         # Alternative: use validation_steps if your script prefers that
        #         validation_steps = kwargs.get('validation_steps')
        #         if validation_steps is not None:
        #             cmd.extend(["--num_validation_images", str(kwargs.get('num_validation_images', 4))]) # Often paired with validation_steps
        #             cmd.extend(["--validation_steps", str(validation_steps)])

        if self.model_type == "sdxl":
            vae_path = kwargs.get('vae_path', 'madebyollin/sdxl-vae-fp16-fix')
            if vae_path: # Only add if a VAE path is actually provided/needed
                cmd.extend(["--pretrained_vae_model_name_or_path", vae_path])
            
            # defaults to true for all of these
            if kwargs.get("enable_xformers_memory_efficient_attention", True):
                 cmd.append("--enable_xformers_memory_efficient_attention")
            if kwargs.get("gradient_checkpointing", True):
                 cmd.append("--gradient_checkpointing")
            if kwargs.get("use_8bit_adam", True):
                 cmd.append("--use_8bit_adam")
        
        lora_rank = kwargs.get('lora_rank')
        if lora_rank is not None:
            cmd.extend(["--rank", str(lora_rank)])
        

        # dont push anywhere by default
        if kwargs.get('report_to'):
            cmd.extend(["--report_to", kwargs['report_to']])
        if kwargs.get('push_to_hub', False):
            cmd.append("--push_to_hub")
            hub_model_id = kwargs.get('hub_model_id')
            if hub_model_id:
                 cmd.extend(["--hub_model_id", hub_model_id])
        
        checkpointing_steps = kwargs.get('checkpointing_steps')
        if checkpointing_steps:
            cmd.extend(["--checkpointing_steps", str(checkpointing_steps)])
            cmd.extend(["--checkpoints_total_limit", str(kwargs.get('checkpoints_total_limit', 2))])

        return cmd
    
    def train_dreambooth(self, 
                        train_data_path: str,
                        model_name: Optional[str] = None,
                        val_data_path: Optional[str] = None,
                        output_dir: Optional[str] = None, 
                        **kwargs) -> str:
        
        base_output_dir = Path(output_dir) if output_dir else self.output_base_dir
        
        _model_name = model_name
        if not _model_name:
            safe_base_id = self.base_model_id.split('/')[-1].replace('-','_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            _model_name = f"{safe_base_id}_dreambooth_{timestamp}"
        else:
            _model_name = _model_name.replace("/", "_").replace("\\", "_")

        model_final_output_dir = Path(kwargs.get('output_dir', base_output_dir / model_name))
        model_final_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting DreamBooth training for {self.model_type.upper()} model")
        self.logger.info(f"Base model: {self.base_model_id}")
        self.logger.info(f"Output model name: {_model_name}")
        self.logger.info(f"Model will be saved to: {model_final_output_dir}")

        train_input_path = Path(train_data_path)
        validation_input_dir_for_command: Optional[Path] = None


        if val_data_path:
            val_input_path = Path(val_data_path)
            if val_input_path.is_dir() and any(val_input_path.iterdir()):
                self.logger.info(f"Using direct validation image directory: {val_input_path}")
                validation_input_dir_for_command = val_input_path
            elif val_input_path.is_file() and val_input_path.suffix.lower() == ".json":
                self.logger.info(f"Validation data path is a JSON file: {val_input_path}. Make sure you meant to do this.")
            else:
                self.logger.warning(f"Validation data path {val_input_path} is not a valid directory or JSON file. Validation might be skipped or fail.")

        if train_input_path.is_dir():
            self.logger.info(f"Training data is a directory: {train_input_path}. Using directly.")
            instance_dir_for_command = train_input_path
            
            if not any(instance_dir_for_command.iterdir()):
                self.logger.error(f"Instance data directory {instance_dir_for_command} is empty.")
                raise ValueError(f"No training images found in {instance_dir_for_command}. Training aborted.")

            training_cmd_args = self._build_training_command(
                instance_data_dir=instance_dir_for_command,
                output_dir=model_final_output_dir,
                validation_data_dir=validation_input_dir_for_command, 
                **kwargs
            )
            self._execute_training_command(training_cmd_args, model_final_output_dir)
        
        # elif train_input_path.is_file() and train_input_path.suffix.lower() == ".json":
        #     self.logger.info(f"Training data is a JSON file: {train_input_path}. Using legacy preparation.")
        #     with tempfile.TemporaryDirectory() as temp_dir_str:
        #         temp_path = Path(temp_dir_str)
        #         # instance_dir_for_command = self._legacy_prepare_emoji_data_from_json(str(train_input_path), temp_path)
                
        #         if not any(instance_dir_for_command.iterdir()):
        #              self.logger.error(f"Instance data directory {instance_dir_for_command} (from JSON) is empty.")
        #              raise ValueError(f"No training images prepared from {train_input_path}. Training aborted.")

        #         training_cmd_args = self._build_training_command(
        #             instance_data_dir=instance_dir_for_command,
        #             output_dir=model_final_output_dir,
        #             validation_data_dir=validation_input_dir_for_command, 
        #             **kwargs
        #         )
        #         self._execute_training_command(training_cmd_args, model_final_output_dir)
        else:
            raise FileNotFoundError(f"Train data path {train_input_path} is not a valid directory or .json file.")
        
        # save model metadata
        serializable_kwargs = {k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()}
        metadata = {
            "model_name": _model_name, 
            "base_model": self.base_model_id,
            "model_type": self.model_type,
            "training_date": datetime.now().isoformat(),
            "training_params": serializable_kwargs,
            "instance_prompt": kwargs.get('instance_prompt', 'sks emoji'),
            "lora_rank": kwargs.get('lora_rank')
        }
        
        metadata_path = model_final_output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model and metadata saved to: {model_final_output_dir}")
        return str(model_final_output_dir)

    def _execute_training_command(self, training_cmd_args: List[str], model_output_dir: Path):
        command_str = ' '.join(map(str, training_cmd_args))
        self.logger.info(f"Training command: {command_str}")
        
        script_to_execute = Path(training_cmd_args[2]) # accelerate launch SCRIPT_PATH ...
        script_exec_cwd = script_to_execute.parent 
        self.logger.info(f"Executing training script from CWD: {script_exec_cwd}")

        try:
            process = subprocess.run(training_cmd_args, cwd=script_exec_cwd, check=True)
            # process = subprocess.Popen(
            #    training_cmd_args, 
            #    text=True, 
            #    cwd=script_exec_cwd,
            #    env=os.environ.copy() # Ensure the subprocess inherits the current environment
            # )
 
        except FileNotFoundError as e:
             self.logger.error(f"The training script {script_to_execute} was not found or another part of the command caused a FileNotFoundError. Error: {e}")
             self.logger.error(f"Please ensure that 'accelerate' is installed and accessible, and the script path is correct.")
             raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during training execution: {e}")
            raise
    
    @staticmethod
    def list_fine_tuned_models(models_base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        print("Listing fine tuned models...")
        logger = logging.getLogger(__name__ + ".list_fine_tuned_models") 
        actual_models_dir = Path(models_base_dir) if models_base_dir else Path("fine_tuned_models")
        
        if not actual_models_dir.exists() or not actual_models_dir.is_dir():
            logger.info(f"Fine-tuned models directory not found or is not a directory: {actual_models_dir}")
            return []
        
        models_info = []
        for model_dir_item in actual_models_dir.iterdir():
            if model_dir_item.is_dir():
                metadata_path = model_dir_item / "metadata.json"
                model_entry = {
                    "name": model_dir_item.name,
                    "path": str(model_dir_item.resolve()),
                    "base_model": "unknown",
                    "model_type": "unknown", 
                    "training_date": "unknown",
                    "instance_prompt": "unknown",
                    "lora_rank": "unknown"
                }
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        model_entry.update({
                            "base_model": metadata.get("base_model", model_entry["base_model"]),
                            "model_type": metadata.get("model_type", model_entry["model_type"]),
                            "training_date": metadata.get("training_date", model_entry["training_date"]),
                            "instance_prompt": metadata.get("instance_prompt", model_entry["instance_prompt"]),
                            "lora_rank": metadata.get("lora_rank", model_entry["lora_rank"])
                        })
                    except json.JSONDecodeError:
                        logger.warning(f"Corrupted metadata for model {model_dir_item.name}, listing with defaults.")
                else:
                    logger.warning(f"No metadata.json found for model {model_dir_item.name}, listing with defaults.")
                models_info.append(model_entry)
        
        return sorted(
            models_info, 
            key=lambda x: x["training_date"] if x["training_date"] != "unknown" else "0000-00-00T00:00:00", 
            reverse=True
        )
