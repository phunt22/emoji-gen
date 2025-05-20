import torch
from pathlib import Path
import logging
from typing import Optional, Dict, List, Union, Literal, Tuple
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments
from diffusers.optimization import get_scheduler
from emoji_gen.models.cache import model_cache
from emoji_gen.config import DEVICE, DTYPE
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from accelerate import Accelerator
from tqdm.auto import tqdm
import math

class EmojiDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, image_size: int = 512):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.image_size = image_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get image
        response = requests.get(item['link'])
        image = Image.open(BytesIO(response.content))
        
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            # Create a white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            # Composite the image with alpha over the background
            background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize and convert to tensor
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 127.5 - 1.0
        
        # Get text
        text = item['processed']
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": image,
            "input_ids": text_inputs.input_ids[0],
            "attention_mask": text_inputs.attention_mask[0],
        }

class EmojiFineTuner:
    def __init__(self, base_model_id: str, output_dir: str = "fine_tuned_models"):
        self.base_model_id = base_model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def train_lora(
        self,
        train_data_path: str,
        val_data_path: str,
        model_name: str,
        lora_rank: int = 4,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        mixed_precision: str = "fp16",
        seed: int = 42,
    ) -> tuple[str, float]:
        """
        Fine-tune model using LoRA (Low-Rank Adaptation)
        
        Args:
            train_data_path: Path to training data JSON
            val_data_path: Path to validation data JSON
            model_name: Name for the fine-tuned model
            lora_rank: Rank for LoRA adaptation
            lora_alpha: Alpha parameter for LoRA scaling
            lora_dropout: Dropout probability for LoRA layers
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            gradient_accumulation_steps: Number of steps to accumulate gradients
            mixed_precision: Mixed precision training type
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (path to the saved model, best validation loss)
        """
        # Set random seed
        torch.manual_seed(seed)
        
        # Initialize accelerator
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )
        
        # Load base model
        self.logger.info(f"Loading base model: {self.base_model_id}")
        if "xl" in self.base_model_id.lower():
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.base_model_id,
                torch_dtype=DTYPE,
                use_safetensors=True,
                variant="fp16" if DEVICE == "cuda" else None
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                self.base_model_id,
                torch_dtype=DTYPE,
                use_safetensors=True,
            )
        
        # create lora config based on the model type
        if "xl" in self.base_model_id.lower():
            target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            # task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA to UNet
        pipe.unet = get_peft_model(pipe.unet, lora_config)
        
        # Create datasets
        train_dataset = EmojiDataset(train_data_path, pipe.tokenizer)
        val_dataset = EmojiDataset(val_data_path, pipe.tokenizer)
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            pipe.unet.parameters(),
            lr=learning_rate,
        )
        
        # Setup learning rate scheduler
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * num_epochs,
        )
        
        # Prepare for distributed training
        pipe.unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            pipe.unet, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
        
        # Training loop
        self.logger.info("Starting training...")
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            
            # Training
            pipe.unet.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(pipe.unet):
                    # make sure that data is on the correct device and dtype
                    batch["pixel_values"] = batch["pixel_values"].to(device=accelerator.device, dtype=DTYPE)
                    batch["input_ids"] = batch["input_ids"].to(device=accelerator.device)
                    batch["attention_mask"] = batch["attention_mask"].to(device=accelerator.device)
                    
                    # Forward pass
                    latents = pipe.vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * 0.18215
                    
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],))
                    noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                    
                    encoder_hidden_states = pipe.text_encoder(batch["input_ids"])[0]
                    
                    noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    
                    # Calculate loss
                    loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="none")
                    loss = loss.mean([1, 2, 3]).mean()
                    
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(pipe.unet.parameters(), 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                    
                progress_bar.update(1)
                global_step += 1
                
                if global_step % 100 == 0:
                    self.logger.info(f"Step {global_step}: Loss = {loss.item():.4f}")
            
            # Validation
            pipe.unet.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    # make sure data is on the correct device and dtype
                    batch["pixel_values"] = batch["pixel_values"].to(device=accelerator.device, dtype=DTYPE)
                    batch["input_ids"] = batch["input_ids"].to(device=accelerator.device)
                    batch["attention_mask"] = batch["attention_mask"].to(device=accelerator.device)
                    
                    latents = pipe.vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * 0.18215
                    
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],))
                    noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                    
                    encoder_hidden_states = pipe.text_encoder(batch["input_ids"])[0]
                    
                    noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    
                    loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="none")
                    loss = loss.mean([1, 2, 3]).mean()
                    val_loss += loss.item()
            
            val_loss /= len(val_dataloader)
            self.logger.info(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                output_path = self.output_dir / model_name
                output_path.mkdir(exist_ok=True)
                
                # Save LoRA weights
                pipe.unet.save_pretrained(output_path / "lora_weights")
                
                # Save full pipeline for inference
                pipe.save_pretrained(output_path)
                
                # Register model in cache
                model_cache.register_model(model_name, str(output_path))
                
                self.logger.info(f"Saved best model to {output_path}")
        
        return str(output_path), best_val_loss

    def train_dreambooth(
        self,
        train_data_path: str,
        val_data_path: str,
        model_name: str,
        instance_prompt: str = "emoji style",
        class_prompt: str = "emoji",
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        mixed_precision: str = "fp16",
        seed: int = 42,
    ) -> str:
        """
        Fine-tune model using Dreambooth
        
        Args:
            train_data_path: Path to training data JSON
            val_data_path: Path to validation data JSON
            model_name: Name for the fine-tuned model
            instance_prompt: Prompt for the specific style
            class_prompt: Prompt for the general class
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            gradient_accumulation_steps: Number of steps to accumulate gradients
            mixed_precision: Mixed precision training type
            seed: Random seed for reproducibility
            
        Returns:
            Path to the saved model
        """
        # Implementation will be similar to LoRA but with Dreambooth-specific modifications
        # This is a placeholder for now
        pass

    def train_full(
        self,
        train_data_path: str,
        val_data_path: str,
        model_name: str,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        mixed_precision: str = "fp16",
        seed: int = 42,
    ) -> str:
        """
        Full fine-tuning of the model
        
        Args:
            train_data_path: Path to training data JSON
            val_data_path: Path to validation data JSON
            model_name: Name for the fine-tuned model
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            gradient_accumulation_steps: Number of steps to accumulate gradients
            mixed_precision: Mixed precision training type
            seed: Random seed for reproducibility
            
        Returns:
            Path to the saved model
        """
        # Implementation will be similar to LoRA but without LoRA-specific modifications
        # This is a placeholder for now
        pass

    def list_fine_tuned_models(self) -> List[str]:
        """List all fine-tuned models."""
        return [d.name for d in self.output_dir.iterdir() if d.is_dir()] 