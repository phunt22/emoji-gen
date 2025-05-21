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
    def __init__(self, data_path: str, tokenizer, image_size: int = 512, is_sdxl: bool = False):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.is_sdxl = is_sdxl ## not actually needed now
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get image
        response = requests.get(item['link'])
        image = Image.open(BytesIO(response.content))
        
        
        # Convert RGBA to RGB if needed
        if image.mode == 'P':
            # if palette, paste the RGBA mask over the white backround on RGB
            image = image.convert('RGBA')
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
            image = background
        if image.mode == 'RGBA':
            # similar approach
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3]) 
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
        # Set random seed
        torch.manual_seed(seed)
        
        # Initialize accelerator
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )

        is_sdxl = "xl" in self.base_model_id.lower()

        if is_sdxl:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.base_model_id,
                torch_dtype=DTYPE,
                use_safetensors=True,
                variant="fp16" if DEVICE == "cuda" else None
            )
            pipe = pipe.to(DEVICE, dtype=DTYPE)

            # grab hidden size from SDXL
            text_encoder_hidden_size = pipe.text_encoder.config.hidden_size
            text_encoder_2_hidden_size = pipe.text_encoder_2.config.hidden_size
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                self.base_model_id,
                torch_dtype=DTYPE,
                use_safetensors=True,
            )
            pipe = pipe.to(DEVICE, dtype=DTYPE)
        
        if is_sdxl:
            target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )

        pipe.unet = get_peft_model(pipe.unet, lora_config)

        train_dataset = EmojiDataset(train_data_path, pipe.tokenizer, is_sdxl=is_sdxl)
        val_dataset = EmojiDataset(val_data_path, pipe.tokenizer, is_sdxl=is_sdxl)

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

        optimizer = torch.optim.AdamW(
            pipe.unet.parameters(),
            lr=learning_rate,
        )

        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * num_epochs,
        )

        pipe.unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            pipe.unet, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )

        # move encoders to right device after the accelerator is prepared
        if is_sdxl:
            pipe.text_encoder = pipe.text_encoder.to(accelerator.device, dtype=DTYPE)
            pipe.text_encoder_2 = pipe.text_encoder_2.to(accelerator.device, dtype=DTYPE)
            pipe.vae = pipe.vae.to(accelerator.device, dtype=DTYPE)
        else:
            pipe.text_encoder = pipe.text_encoder.to(accelerator.device, dtype=DTYPE)
            pipe.vae = pipe.vae.to(accelerator.device, dtype=DTYPE)
        

        self.logger.info("Starting training...")
        global_step = 0
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            # Training loop
            pipe.unet.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(pipe.unet):
                    # make sure that data is on the correct device and dtype
                    batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, dtype=DTYPE)
                    batch["input_ids"] = batch["input_ids"].to(accelerator.device)

                    latents = pipe.vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * 0.18215

                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=accelerator.device)
                    noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                    # forward pass
                    if is_sdxl:
                        encoder_hidden_states = pipe.text_encoder(batch["input_ids"])[0] ## [1, 77, 1280] -> [bs, seq_len, hidden_size]
                        encoder_output = pipe.text_encoder_2(batch["input_ids"])
                        # Get the pooled output by taking mean across sequence dimension
                        pooled_output = encoder_output[0].mean(dim=1) ## [bs, hidden_size] --> [1, 1280] (expected input dims)

                        bs = batch["input_ids"].shape[0]
                        target_size = (512,512)
                        time_ids = torch.tensor(
                            [
                                [target_size[0], target_size[1], 0, 0, target_size[0], target_size[1]]
                                for _ in range(bs)
                            ],
                            device=accelerator.device,
                            dtype=DTYPE 
                        )
                        
                        # Debug logging to understand shapes
                        print(f"DEBUG: pooled_output shape: {pooled_output.shape}")
                        print(f"DEBUG: time_ids shape: {time_ids.shape}")
                        
                        # DO NOT RESHAPE POOLED OUTPUT OR TIME IDS
                        # THEY ARE ALREADY CORRECT SHAPE
                        
                        added_cond_kwargs = {
                            "text_embeds": pooled_output,
                            "time_ids": time_ids
                        }

                        noise_pred = pipe.unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states,
                            added_cond_kwargs=added_cond_kwargs
                        ).sample
                    else:
                        encoder_hidden_states = pipe.text_encoder(batch["input_ids"])[0]
                        # TODO MIGHT HAVE to DEBUG THIS, see above for reference
                        noise_pred = pipe.unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states
                        ).sample

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
                    
            # validation loop
            pipe.unet.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, dtype=DTYPE)
                    batch["input_ids"] = batch["input_ids"].to(accelerator.device)

                    latents = pipe.vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * 0.18215
                    
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=accelerator.device)
                    noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                    if is_sdxl:
                        encoder_hidden_states = pipe.text_encoder(batch["input_ids"])[0]
                        pooled_output = pipe.text_encoder_2(batch["input_ids"])[0].mean(dim=1)

                        # correcting tensor size/shape
                        bs = batch["input_ids"].shape[0]
                        target_size = (512, 512)
                        time_ids = torch.tensor(
                            [
                                [target_size[0], target_size[1], 0, 0, target_size[0], target_size[1]]
                                for _ in range(bs)
                            ],
                            device=accelerator.device,
                            dtype=DTYPE
                        )

                        
                        # SDXL text_embeds should be of shape [batch_size, 1280]
                        # Fix for dimension mismatches in UNet
                        if hasattr(pipe.text_encoder_2.config, "projection_dim"):
                            # Get the expected embedding size from the model config
                            expected_dim = pipe.text_encoder_2.config.projection_dim
                            # Ensure pooled_output has the right shape without flattening everything
                            print(f"DEBUG-Val: pooled_output.shape={pooled_output.shape}, expected_dim={expected_dim}")
                            if pooled_output.shape[-1] != expected_dim and pooled_output.ndim > 1:
                                pooled_output = pooled_output.reshape(pooled_output.shape[0], expected_dim)
                                print(f"DEBUG-Val: reshaped to {pooled_output.shape}")

                        # Check time_ids for debugging
                        print(f"DEBUG-Val: time_ids.shape={time_ids.shape}")
                        
                        added_cond_kwargs = {
                            "text_embeds": pooled_output,
                            "time_ids": time_ids
                        }

                        noise_pred = pipe.unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states,
                            added_cond_kwargs=added_cond_kwargs
                        ).sample
                    else:
                        encoder_hidden_states = pipe.text_encoder(batch["input_ids"])[0]
                        noise_pred = pipe.unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states
                        ).sample
                    
                    loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="none")
                    loss = loss.mean([1, 2, 3]).mean()

                    val_loss += loss.item()

            val_loss /= len(val_dataloader)
            self.logger.info(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                output_path = self.output_dir / model_name
                output_path.mkdir(exist_ok=True)

                pipe.unet.save_pretrained(output_path / "lora_weights")
                pipe.save_pretrained(output_path)

                model_cache.register_model(model_name, str(output_path))

                self.logger.info(f"Saved best model to {output_path}")
        
        self.best_val_loss = best_val_loss
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