import argparse
from pathlib import Path
from emoji_gen.models.model_manager import model_manager
from emoji_gen.data_utils import get_emoji_list, prune_emoji_list, download_emojis
from emoji_gen.models.fine_tune import EmojiFineTuner
from emoji_gen.config import DEFAULT_MODEL, DEFAULT_DATASET
from emoji_gen.generation import list_available_models
from emoji_gen.server_client import is_server_running, set_model_remote
import torch
import os
import subprocess
from dotenv import load_dotenv
import sys
import importlib.util
from emoji_gen.server import start_server

# load env
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Emoji Generation Developer Tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List models command
    list_parser = subparsers.add_parser("list-models", help="List available models")
    
    # Set model command
    set_model_parser = subparsers.add_parser("set-model", help="Set the active model")
    set_model_parser.add_argument("model_name", type=str, help="Name of the model to set as active")
    
    # Prepare dataset command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare emoji dataset for fine-tuning")
    
    # Fine-tune command
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune a model on emoji dataset")
    finetune_parser.add_argument("--base-model", type=str, required=True, help="Base model to fine-tune")
    finetune_parser.add_argument("--dataset", type=str, required=True, help="Path to prepared dataset")
    finetune_parser.add_argument("--output", type=str, required=True, help="Output directory for fine-tuned model")
    finetune_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    
    # List fine-tuned models command
    list_ft_parser = subparsers.add_parser("list-finetuned", help="List fine-tuned models")
    
    # Start server command
    server_parser = subparsers.add_parser("start-server", help="Start the emoji generation server")
    server_parser.add_argument("--model", type=str, default="sd-v1.5", help="Model ID to use for generation (default: sd-v1.5)")
    server_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    server_parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    server_parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    # Server status command
    status_parser = subparsers.add_parser("server-status", help="Check server status")
    
    args = parser.parse_args()
    
    if args.command == "list-models":
        models = model_manager.list_available_models()
        print("Available models:")
        for model in models:
            print(f"- {model}")
            
    elif args.command == "set-model":
        # Check if server is running
        server_running, server_info = is_server_running()
        if server_running:
            print(f"Server is running with model: {server_info['model']}")
            result = set_model_remote(args.model_name)
            if result["status"] == "success":
                print(f"Successfully set model to {args.model_name} on server")
            else:
                print(f"Error: {result['error']}")
        else:
            print("Server not running, setting model locally...")
            success, message = model_manager.initialize_model(args.model_name)
            if success:
                print(f"Successfully set model to {args.model_name}")
            else:
                print(f"Error: {message}")
                
    elif args.command == "prepare-dataset":
       run_prepare_emojis()
        
    elif args.command == "finetune":
        print(f"Starting fine-tuning of {args.base_model} on dataset {args.dataset}")
        fine_tuner = EmojiFineTuner(args.base_model)
        output_path = fine_tuner.train(args.dataset, args.output, num_epochs=args.epochs)
        print(f"Fine-tuning complete. Model saved to {output_path}")
        
    elif args.command == "list-finetuned":
        fine_tuner = EmojiFineTuner(DEFAULT_MODEL)
        models = fine_tuner.list_fine_tuned_models()
        if not models:
            print("No fine-tuned models found")
        else:
            print(f"{len(models)} Fine-tuned models:")
            for model in models:
                print(f"- {model}")
            
    elif args.command == "start-server":
        print(f"Starting server with model {args.model}...")
        start_server(args.model, args.host, args.port, args.debug)
        
    elif args.command == "server-status":
        server_running, server_info = is_server_running()
        if server_running:
            print(f"Server is running")
            print(f"Active model: {server_info['model']}")
            print(f"Device: {server_info['device']}")
        else:
            print("Server is not running")
            
    else:
        parser.print_help()
        sys.exit(1)

def start_server(model_id, host, port, debug):
    """Start the emoji generation server."""
    server_running, server_info = is_server_running(f"http://{host}:{port}")
    if server_running:
        print(f"Server is already running at http://{host}:{port}")
        print(f"Current model: {server_info['model']}")
        return
    
    # Import server module
    try:
        from emoji_gen.server import start_server as run_server
        run_server(model_id, host, port, debug)
    except ImportError:
        print("Server module not found. Make sure you have Flask installed.")
        print("You can install it with: pip install flask")
        
def check_server_status():
    """Check if the server is running."""
    server_running, server_info = is_server_running()
    if server_running:
        print(f"Server is running with model: {server_info['model']}")
        print(f"Device: {server_info['device']}")
    else:
        print("Server is not running. Start it with 'emoji-dev start-server'")

def run_prepare_emojis():
    """Prepare emoji dataset for training."""
    print("Grabbing emoji list...")
    get_emoji_list()
    print("Pruning emoji list...")
    prune_emoji_list()
    print("Downloading emoji list...")
    download_emojis()
    print("✅ Finished preparing emoji data")

def sync_images():
    """Sync generated images from VM to local machine."""


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
            f'instance-{instance_name}:~/emoji-gen/generated_emojis/*.png',
            str(local_sync_dir)
        ]
        subprocess.run(scp_cmd, check=True)
        print("Sync complete!")
        
    except subprocess.CalledProcessError as e:
        print(f"Error syncing images: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main() 

