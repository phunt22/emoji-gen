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

from emoji_gen.data_utils import get_emoji_list, prune_emoji_list, download_emojis
from emoji_gen.models.fine_tuning import EmojiFineTuner
from emoji_gen.data_utils.split_data import split_emoji_data
from emoji_gen.models.model_manager import model_manager
from emoji_gen.server_client import is_server_running, set_model_remote
from emoji_gen.server import start_server
from emoji_gen.hyperparameter_tuning import tune_hyperparameters
from emoji_gen.config import (
    DEFAULT_MODEL,
    EMOJI_DATA_PATH,
    DATA_SPLITS_DIR,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH
)

# load env
load_dotenv()

def prepare_and_split_data(output_dir: str = "data/splits"):
    """Prepare emoji dataset and split it into train/val/test sets"""
    print("Grabbing emoji list...")
    get_emoji_list()
    print("Pruning emoji list...")
    prune_emoji_list()
    print("Downloading emoji list...")
    download_emojis()
    print("Splitting data...")
    split_emoji_data(
        data_path="data/emojisPruned.json",
        output_dir=output_dir,
        # seed and split are in method
    )


def handle_finetune(args):
    try:
        fine_tuner = EmojiFineTuner(base_model_id=args.model)

        # using the fixed data paths to the folders
        train_data_path = TRAIN_DATA_PATH
        val_data_path = VAL_DATA_PATH
        
        if args.method == "lora":
            output_path = fine_tuner.train_lora(
                train_data_path=train_data_path,
                val_data_path=val_data_path,
                model_name=args.output,
                learning_rate=args.learning_rate,
                lora_rank=args.lora_rank,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                output_dir="fine_tuned_models",
            )
        elif args.method == "dreambooth":
            output_path = fine_tuner.train_dreambooth(
                train_data_path=train_data_path,
                val_data_path=val_data_path,
                model_name=args.output,
                instance_prompt=args.instance_prompt,
                class_prompt=args.class_prompt,
                learning_rate=args.learning_rate,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                output_dir="fine_tuned_models",
            )
        elif args.method == "full":
            output_path = fine_tuner.train_full(
                train_data_path=train_data_path,
                val_data_path=val_data_path,
                model_name=args.output,
                learning_rate=args.learning_rate,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                output_dir="fine_tuned_models",
            )
        else:
            raise ValueError(f"Unknown fine-tuning method: {args.method}")
            
        print(f"Fine-tuning completed. Model saved to: {output_path}")
    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        raise

def handle_list_models():
    models = model_manager.get_available_models()
    print("Available models:")
    for model in models:
        print(f"- {model}")

def handle_set_model(args):
    # check if server is up
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
    fine_tuner = EmojiFineTuner(DEFAULT_MODEL)
    models = fine_tuner.list_fine_tuned_models()
    if not models:
        print("No fine-tuned models found")
    else:
        print(f"{len(models)} Fine-tuned models:")
        for model in models:
            print(f"- {model}")

def handle_server(args):
    print(f"Starting server with model {args.model}...")
    start_server(args.model, args.host, args.port, args.debug)

def handle_server_status(args):
    server_running, server_info = is_server_running()
    if server_running:
        print(f"Server is running")
        print(f"Active model: {server_info['model']}")
        print(f"Device: {server_info['device']}")
    else:
        print("Server is not running")

def handle_sync(args):
    # get VM host from environment variable
    vm_host = os.getenv('GCP_VM_EXTERNAL_IP')
    if not vm_host:
        print("Error: GCP_VM_EXTERNAL_IP environment variable not set")
        print("Please set it to your VM's IP address")
        return
    
    # get local sync directory from environment variable (where we want to save the images)
    local_sync_dir = os.getenv('EMOJI_LOCAL_SYNC_DIR')
    if not local_sync_dir:
        print("Error: EMOJI_LOCAL_SYNC_DIR environment variable not set")
        print("Please set it to your desired local directory")
        return
    
    # create local directory if it doesn't exist
    local_sync_dir = Path(local_sync_dir)
    local_sync_dir.mkdir(parents=True, exist_ok=True)

    # grab instance name from env s.t. we can ssh into it
    instance_name = os.getenv('GCP_INSTANCE_NAME')
    if not instance_name:
        print("Error: GCP_INSTANCE_NAME environment variable not set")
        print("Please set it to your VM's instance name")
        return
    
    print(f"Syncing images from VM to {local_sync_dir}")
    
    # use scp to copy all images as they are in the VM to the local directory
    try:
        scp_cmd = [
            'gcloud',
            'compute',
            'scp',
            '--recurse',  
            f'instance-{instance_name}:~/emoji-gen/generated_emojis/*',  ## make sure we can get folders, not just files
            str(local_sync_dir)  
        ]
        
        subprocess.run(scp_cmd, check=True)
        print("Sync complete!")
        
    except subprocess.CalledProcessError as e:
        print(f"Error syncing images: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Emoji Generation Development CLI')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available models')
    
    # Set model command
    set_model_parser = subparsers.add_parser('set-model', help='Set default model')
    set_model_parser.add_argument('model', help='Model name or path')
    
    # Prepare command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare data for training')
    
    # Fine-tune command
    finetune_parser = subparsers.add_parser('fine-tune', help='Fine-tune model with automatic hyperparameter tuning')
    finetune_parser.add_argument('--model', default=DEFAULT_MODEL,
                               help='Base model to fine-tune')
    finetune_parser.add_argument('--output', help='Output name for fine-tuned model')
    finetune_parser.add_argument('--method', choices=['lora', 'dreambooth', 'full'],
                               default='lora', help='Fine-tuning method')
    finetune_parser.add_argument('--epochs', type=int, default=100,
                               help='Number of training epochs')
    finetune_parser.add_argument('--batch-size', type=int, default=4,
                               help='Training batch size')
    finetune_parser.add_argument('--learning-rate', type=float, default=1e-4,
                               help='Learning rate')
    finetune_parser.add_argument('--lora-rank', type=int, default=4,
                               help='LoRA rank (for LoRA method)')
    finetune_parser.add_argument('--instance-prompt', default='a photo of sks emoji',
                               help='Instance prompt (for Dreambooth)')
    finetune_parser.add_argument('--class-prompt', default='a photo of emoji',
                               help='Class prompt (for Dreambooth)')
    finetune_parser.add_argument('--skip-tuning', action='store_true',
                               help='Skip hyperparameter tuning and use default parameters')
    
    # List fine-tuned models command
    list_finetuned_parser = subparsers.add_parser('list-finetuned',
                                                help='List fine-tuned models')
    
    # Start server command
    server_parser = subparsers.add_parser("start-server", help="Start the emoji generation server")
    server_parser.add_argument("--model", type=str, default="sd-v1.5", help="Model ID to use for generation")
    server_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    server_parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    server_parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    # Server status command
    status_parser = subparsers.add_parser("server-status", help="Check server status")
    
    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync generated images to local machine")
    
    args = parser.parse_args()
    
    if args.command == 'list-models':
        print("\nAvailable models:")
        models = model_manager.get_available_models()
        for model in models:
            print(f"- {model}")
            
    elif args.command == 'set-model':
        handle_set_model(args)
        
    elif args.command == 'prepare':
        prepare_and_split_data()
        
    elif args.command == 'fine-tune':
        # Generate output name if not provided
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"{args.model.split('/')[-1]}_{args.method}_{timestamp}"
        
        print(f"\nFine-tuning model: {args.model}")
        print(f"Method: {args.method}")
        print(f"Output: {args.output}")
        
        if not args.skip_tuning:
            print("\nFinding optimal hyperparameters...")
            best_params = tune_hyperparameters(
                train_data_path=TRAIN_DATA_PATH,
                val_data_path=VAL_DATA_PATH,
                base_model=args.model,
                method=args.method,
                num_samples=5,  # Quick search
                max_epochs=3    # Quick search
            )
            print("\nBest parameters found:")
            print(json.dumps(best_params, indent=2))
            
            # override CLI with best params
            args.learning_rate = best_params['learning_rate']
            args.batch_size = best_params['batch_size']
            if args.method == 'lora':
                args.lora_rank = best_params['lora_rank']
                args.lora_alpha = best_params['lora_alpha']
                args.lora_dropout = best_params['lora_dropout']
            elif args.method == 'dreambooth':
                args.instance_prompt = best_params['instance_prompt']
                args.class_prompt = best_params['class_prompt']
            else:  # full
                args.weight_decay = best_params['weight_decay']
                args.warmup_steps = best_params['warmup_steps']
                args.scheduler = best_params['scheduler']
        
        print("\nStarting fine-tuning...")
        tuner = EmojiFineTuner(args.model)
        
        if args.method == 'lora':
            tuner.train_lora(
                train_data_path=TRAIN_DATA_PATH,
                val_data_path=VAL_DATA_PATH,
                model_name=args.output,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout
            )
        elif args.method == 'dreambooth':
            tuner.train_dreambooth(
                train_data_path=TRAIN_DATA_PATH,
                val_data_path=VAL_DATA_PATH,
                output_name=args.output,
                instance_prompt=args.instance_prompt,
                class_prompt=args.class_prompt,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
        else:  # full
            tuner.train_full(
                train_data_path=TRAIN_DATA_PATH,
                val_data_path=VAL_DATA_PATH,
                output_name=args.output,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
            
        print(f"\nFine-tuning complete! Model saved as: {args.output}")
            
    elif args.command == 'list-finetuned':
        print("\nFine-tuned models:")
        models = EmojiFineTuner.list_finetuned_models()
        for model in models:
            print(f"- {model}")
            
    elif args.command == 'start-server':
        handle_server(args)
    elif args.command == 'server-status':
        handle_server_status(args)
    elif args.command == 'sync':
        handle_sync(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 

