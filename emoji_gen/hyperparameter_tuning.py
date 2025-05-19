import os
import json
from pathlib import Path
import logging
from typing import Dict, Any, Literal

import torch
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import session

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
        "learning_rate": tune.loguniform(1e-6, 1e-4),  
        "batch_size": tune.choice([4, 8]),  
        "gradient_accumulation_steps": tune.choice([2, 4, 8]),  
    }
    
    if method == "lora":
        return {
            **common_space,
            "lora_rank": tune.choice([2, 4]),  
            "lora_alpha": tune.choice([16, 32]),  
            "lora_dropout": tune.uniform(0.0, 0.05)  
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
        try:
            tuner = EmojiFineTuner(base_model)
            
            if method == "lora":
                val_loss = tuner.train_lora(
                    train_data_path=train_data_path,
                    val_data_path=val_data_path,
                    model_name="tuned_model",
                    num_epochs=max_epochs,
                    batch_size=config["batch_size"],
                    learning_rate=config["learning_rate"],
                    lora_rank=config["lora_rank"],
                    lora_alpha=config["lora_alpha"],
                    lora_dropout=config["lora_dropout"],
                    gradient_accumulation_steps=config["gradient_accumulation_steps"]
                )
                # Report metrics to Ray Tune, handling NaN values
                if torch.isnan(torch.tensor(val_loss)) or torch.isinf(torch.tensor(val_loss)):
                    session.report({"val_loss": float('inf')})
                else:
                    session.report({"val_loss": val_loss})
            elif method == "dreambooth":
                # TODO PLACEHOLDER for Dreambooth implementation
                pass
            else:  # full
                # TODO PLACEHOLDER for Full fine-tuning implementation
                pass
        except Exception as e:
            # report the failure to ray tune
            session.report({"val_loss": float('inf')})
            raise e
    
    try:
        output_dir = Path("ray_results").absolute()
        output_dir.mkdir(exist_ok=True)
        
        # Run hyperparameter search
        tuner = tune.Tuner(
            tune.with_parameters(train_func),
            tune_config=tune.TuneConfig(
                num_samples=num_samples,
                scheduler=ASHAScheduler(
                    metric="val_loss",
                    mode="min",
                    max_t=max_epochs,
                    grace_period=1
                ),
                search_alg=HyperOptSearch(metric="val_loss", mode="min"), ## USE HYPEROPT
            ),
            run_config=ray.air.RunConfig(
                storage_path=str(output_dir),
                verbose=1,
            ),
            param_space=config,
        )
        
        # Run the tuning
        results = tuner.fit()
        
        # Get best trial
        best_trial = results.get_best_result(metric="val_loss", mode="min", filter_nan_and_inf=False)
        best_params = best_trial.config
        
        # Save best params
        with open(output_dir / f"best_params_{method}.json", "w") as f:
            json.dump(best_params, f, indent=2)
        
        return best_params
    finally:
        # Clean up Ray resources, even if fail
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    best_params = tune_hyperparameters(
        train_data_path="data/splits/train_emoji_data.json",
        val_data_path="data/splits/val_emoji_data.json",
        method="lora"  # default to lora for now
    )
    print("Best parameters:", json.dumps(best_params, indent=2)) 