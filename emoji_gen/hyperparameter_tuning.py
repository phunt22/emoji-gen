import os
import json
from pathlib import Path
import logging
from typing import Dict, Any, Literal

import torch
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch

from emoji_gen.models import EmojiFineTuner

def get_search_space(method: Literal["lora", "dreambooth", "full"]) -> Dict[str, Any]:
    """Get the search space for a specific fine-tuning method.
    
    Args:
        method: The fine-tuning method to get the search space for
        
    Returns:
        Dictionary defining the search space for the method
    """
    # common parameters for all methods
    common_space = {
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([4, 8, 16]),
        "gradient_accumulation_steps": tune.choice([1, 2, 4, 8]),
    }
    
    if method == "lora":
        return {
            **common_space,
            "lora_rank": tune.choice([2, 4, 8]),
            "lora_alpha": tune.choice([16, 32, 64]),
            "lora_dropout": tune.uniform(0.0, 0.1)
        }
    elif method == "dreambooth":
        return {
            **common_space,
            "instance_prompt": tune.choice([
                "a photo of sks emoji",
                "a photo of emoji style",
                "emoji style image"
            ]),
            "class_prompt": tune.choice([
                "a photo of emoji",
                "emoji",
                "emoji style"
            ]),
            "prior_loss_weight": tune.uniform(0.1, 1.0)
        }
    else:  # full
        return {
            **common_space,
            "weight_decay": tune.loguniform(1e-6, 1e-3),
            "warmup_steps": tune.choice([0, 100, 200, 500]),
            "scheduler": tune.choice(["linear", "cosine", "constant"])
        }

def tune_hyperparameters(
    train_data_path: str,
    val_data_path: str,
    base_model: str = "runwayml/stable-diffusion-v1-5", 
    method: Literal["lora", "dreambooth", "full"] = "lora",
    num_samples: int = 5,
    max_epochs: int = 3
) -> Dict[str, Any]:
    """Find optimal hyperparameters for fine-tuning.
    
    Args:
        train_data_path: Path to training data
        val_data_path: Path to validation data
        base_model: Base model to fine-tune
        method: Fine-tuning method to use
        num_samples: Number of hyperparameter trials
        max_epochs: Maximum epochs per trial
        
    Returns:
        Dictionary of best hyperparameters
    """
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            num_cpus=os.cpu_count(),
            num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            local_mode=False,  # Set to True for debugging/logs (False is way faster)
            ignore_reinit_error=True
        )
    
    # get method-specific search space
    config = get_search_space(method)
    
    # define training function
    def train_func(config):
        tuner = EmojiFineTuner(base_model)
        
        if method == "lora":
            tuner.train_lora(
                train_data_path=train_data_path,
                val_data_path=val_data_path,
                output_name=f"tuned_model_{tune.get_trial_id()}",
                num_epochs=max_epochs,
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                lora_rank=config["lora_rank"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"],
                gradient_accumulation_steps=config["gradient_accumulation_steps"]
            )
        elif method == "dreambooth":
            # Placeholder for Dreambooth implementation
            pass
        else:  # full
            # Placeholder for Full fine-tuning implementation
            pass
    
    # Create output directory
    output_dir = Path("ray_results").absolute() ## need to use absolute path since we are prefixing with file://
    output_dir.mkdir(exist_ok=True)
    
    # Run hyperparameter search
    analysis = tune.run(
        train_func,
        config=config,
        num_samples=num_samples,
        scheduler=ASHAScheduler(
            metric="val_loss",
            mode="min",
            max_t=max_epochs,
            grace_period=1
        ),
        search_alg=BayesOptSearch(metric="val_loss", mode="min"),
        resources_per_trial={"gpu": 1},
        storage_path=f"file://{output_dir}",  ## need to use URI format, so prefix with file://
        verbose=1
    )
    
    # Get best trial
    best_trial = analysis.get_best_trial("val_loss", "min", "last")
    best_params = best_trial.config
    
    # Save best parameters
    with open(output_dir / f"best_params_{method}.json", "w") as f:
        json.dump(best_params, f, indent=2)
    
    return best_params

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    best_params = tune_hyperparameters(
        train_data_path="data/splits/train_emoji_data.json",
        val_data_path="data/splits/val_emoji_data.json",
        method="lora"  # default to lora for now
    )
    print("Best parameters:", json.dumps(best_params, indent=2)) 