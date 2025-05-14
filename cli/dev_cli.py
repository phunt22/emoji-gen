import argparse
from emoji_gen.data_utils import get_emoji_list, prune_emoji_list, download_emojis
from emoji_gen.models.fine_tune import EmojiFineTuner
import sys
from emoji_gen.config import DEFAULT_MODEL, DEFAULT_DATASET
from emoji_gen.generation import list_available_models

def main():
    parser = argparse.ArgumentParser(description="Dev CLI for EmojiGen")
    parser.add_argument("task", 
                        choices=["prepare", "list-models", "fine-tune", "list-fine-tuned"],
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

    args = parser.parse_args()
    
    try:
        if args.task == "prepare":
            run_prepare_emojis()
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
    models_info = list_available_models()
    print("\nAvailable models:")
    for model_id, info in models_info["models"].items():
        print(f"- {model_id} ({info['type']})")
        print(f"  Path: {info['path']}")
    
    if models_info["gpu_available"]:
        print("\nGPU is available for inference")
    else:
        print("\nWARNING: GPU is not available, using CPU (slow)")

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

def run_prepare_emojis():
    """Prepare emoji dataset for training."""
    print("Grabbing emoji list...")
    get_emoji_list()
    print("Pruning emoji list...")
    prune_emoji_list()
    print("Downloading emoji list...")
    download_emojis()
    print("✅ Finished preparing emoji data")

if __name__ == "__main__":
    main() 

