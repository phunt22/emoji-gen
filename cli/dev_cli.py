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
from emoji_gen.utils.cache_rag_embeddings import compute_and_cache_rag_embeddings

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
from emoji_gen.utils.aux_models import get_clip_pipeline
from PIL import Image
import torch

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
    
    logger.info("\n3️⃣ Computing and caching RAG caption embeddings...")
    print("\n3️⃣ Computing and caching RAG caption embeddings...")
    compute_and_cache_rag_embeddings()
    
    logger.info(f"\n4️⃣ Downloading & organizing images from {EMOJI_DATA_PATH} into DreamBooth structure (dreambooth_download_emojis)...")
    print(f"\n4️⃣ Downloading and organizing emoji images (using config ratios)...")

    organize_emojis()        
    logger.info("\n5️⃣ Verifying DreamBooth data structure (verify_dreambooth_structure)...")
    print("\n5️⃣ Verifying DreamBooth data structure...")

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



def handle_test(args):
    use_rag = args.use_rag or False
    use_llm = args.use_llm or False
    num_prompts_to_test = args.num
    output_dir_name_arg = args.name
    custom_prompt_file = args.prompt_file
    num_inference_steps = args.num_steps
    guidance_scale = args.guidance

    model_identifier = args.model

    logger.info(f"Starting test run for model: {model_identifier}")
    logger.info(f"Using RAG: {use_rag}, Using LLM: {use_llm}")
    logger.info(f"Inference steps: {num_inference_steps}, Guidance scale: {guidance_scale}")
    logger.info(f"Targeting {num_prompts_to_test} unique prompts, 1 image per prompt.")

    # attempt to load CLIP model
    clip_model, clip_processor = None, None
    try:
        clip_model, clip_processor = get_clip_pipeline()
        if clip_model and clip_processor:
            logger.info("CLIP model loaded successfully for score calculation.")
        else:
            logger.warning("CLIP model or processor not available. CLIP scores will not be calculated.")
    except Exception as e:
        logger.error(f"Failed to load CLIP model for scoring: {e}", exc_info=True)
        clip_model, clip_processor = None, None # Ensure they are None

    selected_prompts = []
    prompt_source_info = "unknown"
    random.seed(42) 

    # LOADING PROMPTS
    # if custom file, load it
    if custom_prompt_file:
        try:
            prompt_file_path = Path(custom_prompt_file)
            if prompt_file_path.is_file():
                with open(prompt_file_path, 'r') as f:
                    available_prompts = [line.strip() for line in f if line.strip()]
                if available_prompts:
                    if len(available_prompts) <= num_prompts_to_test:
                        selected_prompts = available_prompts
                        random.shuffle(selected_prompts)
                    else:
                        selected_prompts = random.sample(available_prompts, num_prompts_to_test)
                    logger.info(f"Selected {len(selected_prompts)} prompts from custom file: {custom_prompt_file}")
                    prompt_source_info = f"custom_file:{custom_prompt_file} (selected={len(selected_prompts)}/{len(available_prompts)})"
                else:
                    logger.warning(f"Custom prompt file {custom_prompt_file} was empty. Falling back.")
            else:
                logger.warning(f"Custom prompt file {custom_prompt_file} not found. Falling back.")
        except Exception as e:
            logger.error(f"Error reading custom prompt file {custom_prompt_file}: {e}. Falling back.", exc_info=True)
    
    # if metadata file, load it
    if not selected_prompts and TEST_METADATA_PATH and Path(TEST_METADATA_PATH).exists():
        try:
            with open(TEST_METADATA_PATH, 'r') as f:
                all_metadata_prompts = [item.get("processed") for item in json.load(f) if item.get("processed")]
            
            if all_metadata_prompts:
                if len(all_metadata_prompts) <= num_prompts_to_test:
                    selected_prompts = all_metadata_prompts
                    random.shuffle(selected_prompts) # Shuffle for consistent order
                    logger.info(f"Selected all {len(selected_prompts)} available prompts from {TEST_METADATA_PATH} (shuffled with seed 42).")
                else:
                    selected_prompts = random.sample(all_metadata_prompts, num_prompts_to_test)
                    logger.info(f"Randomly sampled {len(selected_prompts)} prompts from {TEST_METADATA_PATH} (seed 42).")
                prompt_source_info = f"metadata:{TEST_METADATA_PATH} (selected={len(selected_prompts)}/{len(all_metadata_prompts)}, target_sample_size={num_prompts_to_test})"
            else:
                logger.warning(f"No valid prompts found in {TEST_METADATA_PATH}. Falling back.")
        except Exception as e:
            logger.error(f"Error loading or sampling prompts from {TEST_METADATA_PATH}: {e}. Falling back.", exc_info=True)

    if not selected_prompts:
        logger.error("CRITICAL: No prompts to test with. Check logs.")
        print("❌ CRITICAL: No prompts to test with. Check logs.")
        return
    
    # RUN INFERENCE
    # now actual inference on testing
    test_run_results = []
    
    if output_dir_name_arg:
        safe_output_dir_name = "".join(c if c.isalnum() else "_" for c in output_dir_name_arg)[:50].strip("_")
        if not safe_output_dir_name: safe_output_dir_name = "test_run"
    else:
        safe_output_dir_name = "test_run"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_test_output_dir = Path(TEST_DATA_PATH_IMAGES or "generated_tests")
    model_id_str = str(model_identifier).replace('/','_') if model_identifier else "unknown_model"
    current_test_run_dir = base_test_output_dir / f"{safe_output_dir_name}_{model_id_str}_{timestamp}"
    current_test_run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Test outputs will be saved to: {current_test_run_dir}")
    logger.info(f"Actually processing {len(selected_prompts)} unique prompts.")

    from emoji_gen.generation import generate_emoji

    overall_start_time = time.time()

    for i, prompt_text in enumerate(selected_prompts):
        logger.info(f"Processing prompt {i+1}/{len(selected_prompts)}: '{prompt_text}'")
        
        gen_start_time = time.time()
        generation_result = generate_emoji(
            prompt=prompt_text,
            output_path=str(current_test_run_dir),
            use_rag=use_rag,
            use_llm=use_llm,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        gen_end_time = time.time()
        duration_seconds = round(gen_end_time - gen_start_time, 2)
        
        generated_output_details = {
            "image_path": None,
            "duration_seconds": duration_seconds,
            "status": "error",
            "clip_score": None
        }

        if generation_result["status"] == "success":
            image_path_str = generation_result["image_path"]
            generated_output_details.update({
                "image_path": image_path_str,
                "status": "success",
            })
            logger.info(f"Successfully generated image: {image_path_str} in {duration_seconds}s")
            print(f"  🖼️ Image for '{prompt_text}' saved to: {image_path_str} (took {duration_seconds}s)")

            # CLIP Score if available
            if clip_model and clip_processor and image_path_str:
                try:
                    image = Image.open(image_path_str).convert("RGB")
                    inputs = clip_processor(text=[prompt_text], images=[image], return_tensors="pt", padding=True, truncation=True)
                    
                    # move inputs to the CLIP model's device
                    inputs = {k: v.to(clip_model.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = clip_model(**inputs)
                    
                    text_features = clip_model.get_text_features(**inputs)
                    image_features = clip_model.get_image_features(**inputs)
                    
                    text_features_norm = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                    image_features_norm = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    
                    similarity = (text_features_norm @ image_features_norm.T).item()
                    # [-1, 1]

                    score = round(similarity, 4) 

                    generated_output_details["clip_score"] = score
                    logger.info(f"Calculated CLIP score for '{prompt_text}': {score}")
                    print(f"    📊 CLIP Score: {score}")
                except Exception as e_clip:
                    logger.error(f"Error calculating CLIP score for {image_path_str}: {e_clip}", exc_info=True)
                    generated_output_details["clip_score"] = "error_calculating"
        else:
            error_msg = generation_result.get("error", "Unknown error during generation")
            generated_output_details["error_message"] = error_msg
            logger.error(f"Failed to generate image for prompt '{prompt_text}' in {duration_seconds}s: {error_msg}")
        
        test_run_results.append({
            "prompt_index": i,
            "original_prompt": prompt_text,
            "generated_output": generated_output_details, 
            "llm_augmentation_used": use_llm,
            "rag_used": use_rag
        })

    overall_end_time = time.time()
    total_run_duration_seconds = round(overall_end_time - overall_start_time, 2)

    summary_file_path = current_test_run_dir / "test_summary.json"
    with open(summary_file_path, 'w') as f:
        json.dump({
            "test_run_parameters": {
                "model_tested": model_identifier,
                "use_rag": use_rag,
                "use_llm": use_llm,
                "num_prompts_requested_arg": num_prompts_to_test, # This is args.num
                "actual_prompts_processed_count": len(selected_prompts),
                "images_per_prompt_generated": 1, # Fixed to 1
                "output_directory_name_arg": output_dir_name_arg,
                "actual_output_directory": str(current_test_run_dir),
                "prompt_source": prompt_source_info
            },
            "overall_test_duration_seconds": total_run_duration_seconds,
            "results_per_prompt": test_run_results 
        }, f, indent=2)
    
    logger.info(f"Test run complete. Total duration: {total_run_duration_seconds}s. Summary saved to: {summary_file_path}")
    print(f"\n✅ Test run finished! Total time: {total_run_duration_seconds}s")
    print(f"📂 All outputs and summary saved in: {current_test_run_dir}")

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
    
    # Test command parser
    test_parser = subparsers.add_parser('test', help='Run a test generation suite')
    # TODO add prompt file path default
    test_parser.add_argument('--model', type=str, required=True, help='Model name or path to test (must be in ModelManager or a local path)')
    test_parser.add_argument('--name', type=str, help='Optional name for the test run directory (i.e. "awesome_rag_experiment")')
    test_parser.add_argument('--num', type=int, default=1, help='Number of unique prompts to test (randomly sampled with seed).')
    test_parser.add_argument('--prompt-file', type=str, help='Optional path to a .txt file containing prompts (one per line). Overrides default prompt loading.')
    # inference params
    test_parser.add_argument('--use-rag', action='store_true', help='Enable Retrieval Augmented Generation (RAG) using IP-Adapter')
    test_parser.add_argument('--use-llm', action='store_true', help='Enable LLM prompt augmentation')
    test_parser.add_argument('--num-steps', type=int, default=25, help='Number of inference steps to take (default: %(default)s)')
    test_parser.add_argument('--guidance', type=float, default=7.5, help='Guidance scale (default: %(default)s)')

    args = parser.parse_args()
    
    if args.command == 'list-models': handle_list_models()
    elif args.command == 'set-model': handle_set_model(args)
    elif args.command == 'prepare': prepare_and_split_data()
    elif args.command == 'fine-tune': handle_finetune(args)
    elif args.command == 'list-finetuned': handle_list_finetuned()
    elif args.command == 'start-server': handle_server(args)
    elif args.command == 'server-status': handle_server_status(args)
    elif args.command == 'sync': handle_sync(args)
    elif args.command == 'test': handle_test(args)

if __name__ == '__main__':
    main() 

