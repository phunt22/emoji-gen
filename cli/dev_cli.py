import argparse
from pathlib import Path
import os
import subprocess
from dotenv import load_dotenv
import random
import json
import logging
import time
from datetime import datetime

from emoji_gen.data_utils.get_emoji_list import main as get_emoji_list
from emoji_gen.data_utils.prune_emoji_list import main as prune_emoji_list
from emoji_gen.data_utils.dreambooth_preparation import organize_emojis, verify_dreambooth_structure

from emoji_gen.models.fine_tuning import EmojiFineTuner
from emoji_gen.models.model_manager import model_manager
from emoji_gen.server_client import is_server_running, set_model_remote
from emoji_gen.server import start_server
from emoji_gen.config import (
    DEFAULT_MODEL,
    EMOJI_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH_IMAGES,
    TEST_METADATA_PATH,
    get_model_path,
    FINE_TUNED_MODELS_DIR
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# load env
load_dotenv()

def prepare_and_split_data():
    """Prepare emoji dataset for DreamBooth: download, process, and split images."""
    logger.info("🔄 Starting emoji data preparation for DreamBooth training and evaluation...")
    print("🔄 Preparing emoji data for DreamBooth training and evaluation...")
    
    logger.info("\n1️⃣ Collecting initial emoji metadata (get_emoji_list)...")
    print("\n1️⃣ Collecting initial emoji metadata...")
    get_emoji_list()
    
    logger.info("\n2️⃣ Pruning emoji list (prune_emoji_list)...")
    print("\n2️⃣ Pruning emoji list...")
    prune_emoji_list()
    
    logger.info(f"\n3️⃣ Downloading & organizing images from {EMOJI_DATA_PATH} into DreamBooth structure (dreambooth_download_emojis)...")
    print(f"\n3️⃣ Downloading and organizing emoji images (using config ratios)...")

    organize_emojis()        
    logger.info("\n4️⃣ Verifying DreamBooth data structure (verify_dreambooth_structure)...")
    print("\n4️⃣ Verifying DreamBooth data structure...")

    if verify_dreambooth_structure():
        print(f"\n🚀 Ready for fine-tuning: emoji-dev fine-tune")
    else:
        print("❌ Data preparation failed or structure is not as expected.")

def handle_finetune(args):
    try:
        model_id_to_finetune = get_model_path(args.model)
        fine_tuner = EmojiFineTuner(base_model_id=model_id_to_finetune)
        output_model_name = args.output
        if not output_model_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name_for_dir = model_id_to_finetune.split('/')[-1].replace('-', '_').replace('.','_')
            output_model_name = f"{base_name_for_dir}_dreambooth_{timestamp}"

        expected_output_path = FINE_TUNED_MODELS_DIR / output_model_name
        expected_output_path.mkdir(parents=True, exist_ok=True)


        logger.info(f"Starting fine-tuning process...")
        logger.info(f"Base Model ID: {model_id_to_finetune}")
        logger.info(f"Detected Model Type for Training Script: {fine_tuner.model_type}")
        logger.info(f"User-provided output name (or auto-generated): {output_model_name}")
        logger.info(f"Instance (Training) Data Path (from config): {TRAIN_DATA_PATH}")
        logger.info(f"Validation Data Path (from config, if used by script): {VAL_DATA_PATH}")
        train_dir = Path(TRAIN_DATA_PATH)
        if not train_dir.is_dir() or not any(train_dir.iterdir()):
            logger.error(f"Training data directory {train_dir} is empty or does not exist.")
            print(f"❌ Training data directory {train_dir} is empty or not found. Run 'emoji-dev prepare' first.")
            return
        training_params_from_cli = {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'max_train_steps': args.max_train_steps,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'lora_rank': args.lora_rank,
            'instance_prompt': args.instance_prompt,
            'class_prompt': getattr(args, 'class_prompt', None),
            'validation_prompt': args.validation_prompt,
            'validation_epochs': args.validation_epochs,
            'seed': args.seed,
            'report_to': args.report_to if args.report_to else None,
            'push_to_hub': args.push_to_hub,
            'hub_model_id': getattr(args, 'hub_model_id', None),
            'mixed_precision': getattr(args, 'mixed_precision', 'fp16'),
            'lr_scheduler': getattr(args, 'lr_scheduler', 'constant'),
            'lr_warmup_steps': getattr(args, 'lr_warmup_steps', 0),
            'checkpointing_steps': getattr(args, 'checkpointing_steps', None),
            'checkpoints_total_limit': getattr(args, 'checkpoints_total_limit', 2),
            'output_dir': str(expected_output_path) ## otherwise weights are outputted in scripts folder
        }

        # Set resolution based on model type, defaulting if not provided via CLI
        cli_resolution = getattr(args, 'resolution', None) # Get user-provided resolution or None

        if fine_tuner.model_type == "sd3":
            training_params_from_cli['resolution'] = cli_resolution if cli_resolution is not None else 1024
        elif fine_tuner.model_type == "sdxl":
            training_params_from_cli['resolution'] = cli_resolution if cli_resolution is not None else 1024
            # SDXL specific params
            training_params_from_cli['vae_path'] = args.vae_path
            training_params_from_cli['enable_xformers_memory_efficient_attention'] = getattr(args, 'enable_xformers', True)
            training_params_from_cli['gradient_checkpointing'] = getattr(args, 'gradient_checkpointing', True)
            training_params_from_cli['use_8bit_adam'] = getattr(args, 'use_8bit_adam', True)
        else: # Fallback for other model types
            training_params_from_cli['resolution'] = cli_resolution if cli_resolution is not None else 512
        
        # Ensure resolution is not None if it was defaulted, otherwise fine_tuning.py will omit it
        # This is mainly for the "other" case if its default ends up being None
        if training_params_from_cli.get('resolution') is None and fine_tuner.model_type not in ["sd3", "sdxl"]:
             logger.warning(f"Resolution for model type {fine_tuner.model_type} is not explicitly set and no CLI arg provided. Training script defaults will apply.")
        elif training_params_from_cli.get('resolution') is None and fine_tuner.model_type in ["sd3", "sdxl"]:
             # This case should ideally not be hit if defaults (1024) are applied correctly
             logger.error(f"CRITICAL: Resolution for model type {fine_tuner.model_type} defaulted to None. This should not happen. Forcing 1024.")
             training_params_from_cli['resolution'] = 1024

        val_dir_for_trainer = VAL_DATA_PATH
        if not Path(val_dir_for_trainer).is_dir() or not any(Path(val_dir_for_trainer).iterdir()):
            logger.warning(f"Validation image directory {val_dir_for_trainer} is empty or not found. Effective validation might be skipped by training script.")
            val_dir_for_trainer = None
        logger.info("Starting DreamBooth training via fine_tuner...")
        logger.info(f"Effective training parameters: {json.dumps({k: str(v) if isinstance(v, Path) else v for k,v in training_params_from_cli.items()}, indent=2)}")
        final_model_output_path = fine_tuner.train_dreambooth(
            train_data_path=TRAIN_DATA_PATH,
            val_data_path=val_dir_for_trainer,
            model_name=output_model_name,
            **training_params_from_cli
        )
        print(f"\n🎉 Fine-tuning completed successfully!")
        print(f"📁 Model saved to: {final_model_output_path}")
        print(f"🏷️  Final Model name (in fine_tuned_models dir): {Path(final_model_output_path).name}")
        print(f"🤖 Base model used: {model_id_to_finetune}")
        print(f"📊 Trained Model Type: {fine_tuner.model_type.upper()}")
        print(f"\n💡 To use your fine-tuned model (example):")
        print(f"   emoji-dev set-model {Path(final_model_output_path).name}")
        print(f"   emoji-gen '{args.validation_prompt if args.validation_prompt else 'sks emoji'}'")
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}", exc_info=True)

def handle_list_models():
    models = model_manager.get_available_models()
    print("Available base models:")
    if isinstance(models, dict):
        for model_key, model_info in models.items(): 
            print(f"  - {model_key} (Type: {model_info.get('type', 'N/A')}, Path/ID: {model_info.get('path', 'N/A')})")
    elif isinstance(models, list):
         for model_name in models:
            print(f"  - {model_name}")
    else:
        print("Could not retrieve model list in expected format.")

def handle_set_model(args):
    server_running, server_info = is_server_running()
    if server_running:
        print(f"Server is running with model: {server_info['model']}")
        result = set_model_remote(args.model)
        if result["status"] == "success":
            print(f"Successfully set model to {args.model} on server")
        else:
            print(f"Error: {result['error']}")
    else:
        print("Server not running, setting model locally...")
        success, message = model_manager.initialize_model(args.model)
        if success:
            print(f"Successfully set model to {args.model}")
        else:
            print(f"Error: {message}")

def handle_list_finetuned():
    listed_models = EmojiFineTuner.list_fine_tuned_models()
    if not listed_models:
        print("No fine-tuned models found in the default 'fine_tuned_models' directory.")
        print("Run 'emoji-dev fine-tune' to create your first model!")
    else:
        print(f"📚 Found {len(listed_models)} fine-tuned models:")
        for model_info_dict in listed_models:
            print(f"\n  🤖 {model_info_dict.get('name', 'Unknown Name')}")
            print(f"     Path: {model_info_dict.get('path', 'N/A')}")
            print(f"     Base Model: {model_info_dict.get('base_model', 'unknown')}")
            print(f"     Type: {model_info_dict.get('model_type', 'unknown').upper()}")
            date_str = model_info_dict.get('training_date', 'unknown')
            if date_str != 'unknown' and 'T' in date_str:
                date_str = date_str.split('T')[0]
            print(f"     Training Date: {date_str}")
            print(f"     Instance Prompt: {model_info_dict.get('instance_prompt', 'unknown')}")
            print(f"     LoRA Rank: {model_info_dict.get('lora_rank', 'N/A')}")

def handle_server(args):
    print(f"Starting server with model {args.model}...")
    start_server(args.model, args.host, args.port, args.debug)

def handle_server_status(args):
    server_running, server_info = is_server_running()
    if server_running:
        print(f"✅ Server is running")
        print(f"🤖 Active model: {server_info['model']}")
        print(f"💻 Device: {server_info['device']}")
    else:
        print("❌ Server is not running")
        print("💡 Start with: emoji-dev start-server")

def handle_sync(args):
    vm_host = os.getenv('GCP_VM_EXTERNAL_IP')
    if not vm_host: print("❌ Error: GCP_VM_EXTERNAL_IP environment variable not set."); return
    local_sync_dir = os.getenv('EMOJI_LOCAL_SYNC_DIR')
    if not local_sync_dir: print("❌ Error: EMOJI_LOCAL_SYNC_DIR environment variable not set."); return
    local_sync_dir_path = Path(local_sync_dir); local_sync_dir_path.mkdir(parents=True, exist_ok=True)
    instance_name = os.getenv('GCP_INSTANCE_NAME')
    if not instance_name: print("❌ Error: GCP_INSTANCE_NAME environment variable not set."); return
    user_at_instance = f"{os.getenv('USER', 'your_gcp_user')}@instance-{instance_name}"
    print(f"🔄 Syncing images from VM {user_at_instance} to {local_sync_dir_path}")
    try:
        scp_cmd = [
            'gcloud', 'compute', 'scp', '--recurse',
            f'{user_at_instance}:~/emoji-gen/generated_emojis/*',
            str(local_sync_dir_path)
        ]
        subprocess.run(scp_cmd, check=True) ## check=True is what gives the per line command output
        print("✅ Sync complete!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error syncing images. Return code: {e.returncode}")
        print(f"   STDOUT: {e.stdout}")
        print(f"   STDERR: {e.stderr}")
    except Exception as e:
        print(f"❌ Unexpected error during sync: {e}")

def main():
    parser = argparse.ArgumentParser(description='Emoji Generation Development CLI')
    subparsers = parser.add_subparsers(dest='command', help='Commands', required=True)
    
    list_parser = subparsers.add_parser('list-models', help='List available base models')
    set_model_parser = subparsers.add_parser('set-model', help='Set active model')
    set_model_parser.add_argument('model', help='Model name or path')
    prepare_parser = subparsers.add_parser('prepare', help='Prepare emoji data (download, process, split for DreamBooth)')
        
    finetune_parser = subparsers.add_parser('fine-tune', help='Fine-tune model using DreamBooth LoRA with image folders.')
    finetune_parser.add_argument('--model', default=DEFAULT_MODEL, help='Base model ID or path to fine-tune (default: %(default)s)')
    finetune_parser.add_argument('--output', help='Output name for the fine-tuned model directory (auto-generated if not provided)')
    training_group = finetune_parser.add_argument_group('Core Training Parameters')
    training_group.add_argument('--max-train-steps', type=int, default=500, help='Number of training steps (default: %(default)s)')
    training_group.add_argument('--batch-size', type=int, default=1, help='Training batch size (per device) (default: %(default)s)')
    training_group.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate (default: %(default)s)')
    training_group.add_argument('--gradient-accumulation-steps', type=int, default=4, help='Gradient accumulation steps (default: %(default)s)')
    training_group.add_argument('--lora-rank', type=int, default=32, help='LoRA rank (e.g., 32). If not provided, LoRA might not be used or script might use its own default.')
    training_group.add_argument('--seed', type=int, default=42, help='Random seed (default: %(default)s)')
    training_group.add_argument('--resolution', type=int, help='Resolution for training images (e.g., 512 or 1024). Default depends on model type.')
    training_group.add_argument('--mixed-precision', default='fp16', choices=['no', 'fp16', 'bf16'], help='Mixed precision training (default: %(default)s)')
    training_group.add_argument('--lr-scheduler', default='constant', help='LR scheduler type (e.g., constant, linear, cosine) (default: %(default)s)')
    training_group.add_argument('--lr-warmup-steps', type=int, default=0, help='LR warmup steps (default: %(default)s)')
    training_group.add_argument('--checkpointing-steps', type=int, help='Save a checkpoint every X steps.')
    training_group.add_argument('--checkpoints-total-limit', type=int, default=2, help='Max number of checkpoints to store (default: %(default)s)')
    prompt_group = finetune_parser.add_argument_group('Prompt and Validation Parameters')
    prompt_group.add_argument('--instance-prompt', default='sks emoji', help='Instance prompt for training (default: %(default)s)')
    prompt_group.add_argument('--class-prompt', help='Class prompt for regularization. If used, --with_prior_preservation is enabled.')
    prompt_group.add_argument('--validation-prompt', help='Validation prompt. If provided, validation will be attempted.')
    prompt_group.add_argument('--validation-epochs', type=int, help='Run validation every X epochs. Some scripts might use validation_steps instead.')
    sdxl_group = finetune_parser.add_argument_group('SDXL-Specific Parameters')
    sdxl_group.add_argument('--vae-path', default='madebyollin/sdxl-vae-fp16-fix', help='Path to VAE for SDXL models (default: %(default)s)')
    sdxl_group.add_argument('--enable-xformers', type=bool, default=True, help='Enable xformers memory efficient attention (SDXL default: True)')
    sdxl_group.add_argument('--gradient-checkpointing', type=bool, default=True, help='Enable gradient checkpointing (SDXL default: True)')
    sdxl_group.add_argument('--use-8bit-adam', type=bool, default=True, help='Use 8-bit Adam optimizer (SDXL default: True)')
    output_group = finetune_parser.add_argument_group('Output and Tracking')
    output_group.add_argument('--report-to', choices=['wandb', 'tensorboard'], help='Experiment tracking platform (e.g., wandb, tensorboard)')
    output_group.add_argument('--push-to-hub', action='store_true', help='Push trained model to Hugging Face Hub')
    output_group.add_argument('--hub-model-id', help='Repository ID for Hugging Face Hub (e.g., your-username/my-emoji-lora)')

    list_finetuned_parser = subparsers.add_parser('list-finetuned', help='List fine-tuned models')
    server_parser = subparsers.add_parser("start-server", help="Start the emoji generation server")
    server_parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model ID to use (default: %(default)s from config)")
    server_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: %(default)s)")
    server_parser.add_argument("--port", type=int, default=5000, help="Port number (default: %(default)s)")
    server_parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    status_parser = subparsers.add_parser("server-status", help="Check server status")
    sync_parser = subparsers.add_parser("sync", help="Sync generated images from VM")
    
    args = parser.parse_args()
    
    if args.command == 'list-models': handle_list_models()
    elif args.command == 'set-model': handle_set_model(args)
    elif args.command == 'prepare': prepare_and_split_data()
    elif args.command == 'fine-tune': handle_finetune(args)
    elif args.command == 'list-finetuned': handle_list_finetuned()
    elif args.command == 'start-server': handle_server(args)
    elif args.command == 'server-status': handle_server_status(args)
    elif args.command == 'sync': handle_sync(args)

if __name__ == '__main__':
    main() 

