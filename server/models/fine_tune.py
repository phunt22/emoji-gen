import torch
from pathlib import Path
import logging
from typing import Optional, Dict, List
from .cache import model_cache


# TODO use google cloud here
class EmojiFineTuner:
    def __init__(self, base_model_id: str, output_dir: str = "fine_tuned_models"):
        self.base_model_id = base_model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    # fine tunes a model on the emoji dataset
    # returns the path to the saved model
    # TODO: implement actual fine-tuning logic here
    def train(self, dataset_path: str, model_name: str) -> str:
        # Placeholder for actual implementation
        output_path = self.output_dir / model_name
        output_path.mkdir(exist_ok=True)
        
        # TODO: Implement actual fine-tuning logic here
        # 1. Load base model
        # 2. Prepare dataset
        # 3. Set up training loop
        # 4. Train model
        # 5. Save fine-tuned model
        
        # Register the model with the cache
        model_cache.register_model(model_name, str(output_path))
        
        return str(output_path)
    
    def list_fine_tuned_models(self) -> List[str]:
        """List all fine-tuned models."""
        return [d.name for d in self.output_dir.iterdir() if d.is_dir()]


# fine_tuner = EmojiFineTuner([BASE MODEL])
# fine_tuner.train([EMOJI DATASET PATH], [MODEL NAME])