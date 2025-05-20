import os
from pathlib import Path
from typing import Dict, Any
import logging
import torch
import json
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.optuna import OptunaSearch

from .fine_tuning import EmojiFineTuner


# THIS FILE IS NOT USED ANYMORE


logger = logging.getLogger(__name__)

class HyperparameterTuner:
    def __init__(
        self,
        base_model_id: str,
        train_data_path: str,
        val_data_path: str,
        output_dir: str,
        num_samples: int = 20,
        max_epochs: int = 100,
        gpu_per_trial: int = 1,
    ):
        self.base_model_id = base_model_id
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.max_epochs = max_epochs
        self.gpu_per_trial = gpu_per_trial
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def train_with_hyperparams(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Training function for hyperparameter tuning"""
        try:
            fine_tuner = EmojiFineTuner(base_model_id=self.base_model_id)
            
            # Use config values
            output_path = fine_tuner.train_lora(
                train_data_path=self.train_data_path,
                val_data_path=self.val_data_path,
                model_name=f"tuned_model_{tune.get_trial_id()}",
                learning_rate=config["learning_rate"],
                lora_rank=config["lora_rank"],
                num_epochs=config["num_epochs"],
                batch_size=config["batch_size"],
                gradient_accumulation_steps=config["gradient_accumulation_steps"],
                output_dir=self.output_dir / f"trial_{tune.get_trial_id()}",
            )
            
            # Return validation loss for tuning
            return {"val_loss": fine_tuner.best_val_loss}
            
        except Exception as e:
            self.logger.error(f"Error in trial {tune.get_trial_id()}: {str(e)}")
            return {"val_loss": float('inf')}

    def tune(self):
        """Run hyperparameter tuning"""
        # Define search space
        config = {
            "learning_rate": tune.loguniform(1e-5, 1e-3),
            "lora_rank": tune.choice([2, 4, 8, 16]),
            "num_epochs": tune.choice([50, 100, 200]),
            "batch_size": tune.choice([1, 2, 4]),
            "gradient_accumulation_steps": tune.choice([1, 2, 4, 8]),
        }

        # Create scheduler
        scheduler = ASHAScheduler(
            metric="val_loss",
            mode="min",
            max_t=self.max_epochs,
            grace_period=10,
            reduction_factor=2
        )

        # Create search algorithm
        search_alg = OptunaSearch(
            metric="val_loss",
            mode="min",
        )

        # Run tuning
        analysis = tune.run(
            self.train_with_hyperparams,
            config=config,
            num_samples=self.num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            resources_per_trial={"gpu": self.gpu_per_trial},
            local_dir=str(self.output_dir / "ray_results"),
            name="emoji_tuning",
        )

        # Get best trial
        best_trial = analysis.get_best_trial("val_loss", "min", "last")
        best_config = best_trial.config
        best_val_loss = best_trial.last_result["val_loss"]

        # Save best configuration
        best_config_path = self.output_dir / "best_config.json"
        with open(best_config_path, "w") as f:
            json.dump({
                "config": best_config,
                "val_loss": best_val_loss,
                "trial_id": best_trial.trial_id
            }, f, indent=2)

        self.logger.info(f"Best trial config: {best_config}")
        self.logger.info(f"Best trial validation loss: {best_val_loss}")
        self.logger.info(f"Best trial ID: {best_trial.trial_id}")
        self.logger.info(f"Best config saved to: {best_config_path}")

        return best_config, best_val_loss

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for emoji generation")
    parser.add_argument("--base-model", type=str, required=True, help="Base model to fine-tune")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for tuning results")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of trials to run")
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum number of epochs per trial")
    parser.add_argument("--gpu-per-trial", type=int, default=1, help="Number of GPUs per trial")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create tuner
    tuner = HyperparameterTuner(
        base_model_id=args.base_model,
        train_data_path="data/splits/train_emoji_data.json",
        val_data_path="data/splits/val_emoji_data.json",
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        max_epochs=args.max_epochs,
        gpu_per_trial=args.gpu_per_trial,
    )
    
    # Run tuning
    best_config, best_val_loss = tuner.tune()

if __name__ == "__main__":
    main() 