import argparse
from emoji_gen.data_utils import get_emoji_list, prune_emoji_list, download_emojis as downloadEmojiList
import os
import torch
from diffusers import StableDiffusionPipeline
from server.models.fine_tune import EmojiFineTuner
import subprocess
import sys
import time
from server.config import (
    DEFAULT_MODEL, DEFAULT_PORT, DEFAULT_DATASET,
    DEFAULT_HOST, get_available_models
)

# TODO add google cloud commands here
def main():
    parser = argparse.ArgumentParser(description="Dev CLI for EmojiGen")
    parser.add_argument("task", 
                        choices=["prepare", "start", "list-models", "fine-tune", "list-fine-tuned"],
                        help="Choose a dev task"
                        )
    
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Model key to use (e.g., 'sd-v1.5')")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--dataset", type=str,
                        help="Path to dataset for fine-tuning")
    parser.add_argument("--output-name", type=str,
                        help="Name for the fine-tuned model")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="Port to run the server on")

    args = parser.parse_args()
    try:
        if args.task == "prepare":
            run_prepare_emojis()

        elif args.task == "start":
            run_serve(port=args.port)
        elif args.task == "fine-tune":
            dataset = args.dataset or DEFAULT_DATASET
            output_name = args.output_name or f"fine_tuned_{args.model}"
            
            run_fine_tune(args.model, dataset, output_name)
        elif args.task == "list-fine-tuned":
            list_fine_tuned_models()
        elif args.task == "list-models":
            list_models()
    except Exception as e:
        print(f"Stopped due to error: {e}")

def list_models():
    """List all available models."""
    models = get_available_models()
    print("\nAvailable models:")
    for model_id, info in models.items():
        print(f"- {model_id} ({info['type']})")
        print(f"  Path: {info['path']}")

def run_fine_tune(base_model: str, dataset_path: str, output_name: str):
    """Run fine-tuning on the specified model.""" 
    print(f"Starting fine-tuning of {base_model} on dataset {dataset_path}")
    fine_tuner = EmojiFineTuner(base_model)
    output_path = fine_tuner.train(dataset_path, output_name)
    print(f"Fine-tuning complete. Model saved to {output_path}")

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
    

def run_serve(port: int = DEFAULT_PORT):
    """Start the inference server."""
    print(f"Starting inference server on port {port}...")
    
    # get the path to the server dir
    server_dir = os.path.join(os.path.dirname(__file__), "..", "server")
    
    # start the server with uvicorn
    try:
        server_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", DEFAULT_HOST, "--port", str(port)],
            cwd=server_dir
        )
        
        print(f"Server started! Press Ctrl+C to stop.")
        print(f"API available at: http://localhost:{port}")
        print("Available endpoints:")
        print("  - POST /generate")
        print("  - GET /models/list")
        print("  - GET /status")
        
        # Keep the process running
        server_process.wait()
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server_process.terminate()
        server_process.wait()
        print("Server stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")
        if server_process:
            server_process.terminate()

def run_prepare_emojis():
    print("Grabbing emoji list...")
    get_emoji_list()
    print("Pruning emoji list...")
    prune_emoji_list()
    print("Downloading emoji list...")
    downloadEmojiList()
    print("✅ Finished preparing emoji data")

if __name__ == "__main__":
    main() 

