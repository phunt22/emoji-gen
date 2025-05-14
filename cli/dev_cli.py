import argparse
from pathlib import Path
from emoji_gen.models.model_manager import model_manager
from emoji_gen.data_utils import get_emoji_list, prune_emoji_list, download_emojis
from emoji_gen.models.fine_tune import EmojiFineTuner
from emoji_gen.config import DEFAULT_MODEL, DEFAULT_DATASET
from emoji_gen.generation import list_available_models
import torch
import os
import subprocess
from dotenv import load_dotenv

# load env
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Dev CLI for EmojiGen")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # list models 
    list_parser = subparsers.add_parser("list", help="List available models")

    # set model
    set_model_parser = subparsers.add_parser("set-model", help="Set the active model")
    set_model_parser.add_argument("model", help="Model name to set as active")



    # prepare (emoji data for fine-tuning)
    prepare_parser = subparsers.add_parser("prepare", help="Prepare emoji dataset for training")

    # fine tune (TODO IMPLEMENT)
    fine_tune_parser = subparsers.add_parser("fine-tune", help="Fine-tune a model")
    fine_tune_parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                                help="Model key to use (e.g., 'sd-v1.5')")
    fine_tune_parser.add_argument("--dataset", type=str,
                                help="Path to dataset for fine-tuning")
    fine_tune_parser.add_argument("--output-name", type=str,
                                help="Name for the fine-tuned model")

    # list fine tuned models
    list_fine_tuned_parser = subparsers.add_parser("list-fine-tuned", 
                                                 help="List all fine-tuned models")

    args = parser.parse_args()

    try:
        if args.command == "list":
            models = model_manager.list_available_models()
            print("\nAvailable Models:")
            for name, info in models.items():
                print(f"- {name} ({info['type']})")
                print(f"  Path: {info['path']}")
            
            if torch.cuda.is_available():
                print("\nGPU is available for inference")
            else:
                print("\nWARNING: GPU is not available, using CPU (slow)")

        elif args.command == "set-model":
            success, message = model_manager.initialize_model(args.model)
            print(message)

        elif args.command == "prepare":
            run_prepare_emojis()

        elif args.command == "fine-tune":
            dataset = args.dataset or DEFAULT_DATASET
            output_name = args.output_name or f"fine_tuned_{args.model}"
            run_fine_tune(args.model, dataset, output_name)

        elif args.command == "list-fine-tuned":
            list_fine_tuned_models()

    except Exception as e:
        print(f"Stopped due to error: {e}")

# TODO IMPLEMENT
def run_fine_tune(base_model: str, dataset_path: str, output_name: str):
    """Run fine-tuning on the specified model.""" 
    print(f"Starting fine-tuning of {base_model} on dataset {dataset_path}")
    fine_tuner = EmojiFineTuner(base_model)
    output_path = fine_tuner.train(dataset_path, output_name)
    print(f"Fine-tuning complete. Model saved to {output_path}")

# TODO implement (after fine-tuning is implemented)
def list_fine_tuned_models():
    """List all fine-tuned models."""
    fine_tuner = EmojiFineTuner(DEFAULT_MODEL)
    models = fine_tuner.list_fine_tuned_models()
    if not models:
        print("No fine-tuned models found")
    else:
        print(f"{len(models)} Fine-tuned models:")
    for model in models:
        print(f"- {model}")

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

