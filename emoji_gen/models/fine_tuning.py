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
import os

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
        
        # Debug PyTorch and CUDA setup
        print("\nDEBUG: PyTorch and CUDA Setup:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch path: {torch.__file__}")
        
        # Initialize CUDA if available
        if torch.cuda.is_available():
            # Make sure we're using the right device
            torch.cuda.set_device(0)
            # Clear any existing CUDA memory
            torch.cuda.empty_cache()
            # Force CUDA initialization
            torch.cuda.init()
            print("DEBUG: CUDA initialized and memory cleared")
            
            # Verify CUDA is working
            test_tensor = torch.zeros(1, device='cuda')
            print(f"DEBUG: CUDA test tensor created: {test_tensor.device}")
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
        print(f"CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        if torch.cuda.is_available():
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA current device: {torch.cuda.current_device()}")
            print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print("\n")
        
        # Initialize accelerator
        print("DEBUG: Initializing accelerator...")
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )

        
        print(f"DEBUG: TORCH_CUDA_AVAILABLE: {torch.cuda.is_available()}")
        print(f"DEBUG: CUDA_DEVICE_COUNT: {torch.cuda.device_count()}")
        print(f"DEBUG: CUDA_DEVICE_NAME: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        print(f"DEBUG: DEVICE from config: {DEVICE}")
        print(f"DEBUG: accelerator.device: {accelerator.device}")
        print(f"DEBUG: accelerator.state.device: {accelerator.state.device}")
        
        # Force GPU usage if available but not detected by accelerator
        if torch.cuda.is_available() and str(accelerator.device) == "cpu":
            print("WARNING: CUDA is available but accelerator is using CPU. Forcing GPU usage...")
            # Create a new accelerator with device_placement=False so we can manually handle device placement
            accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision=mixed_precision,
                device_placement=False,  # Disable automatic device placement
            )
            print(f"DEBUG: New accelerator created with device_placement=False")
            
            # Override accelerator device in its state
            if hasattr(accelerator, "state"):
                accelerator.state.device = torch.device("cuda:0")
                print(f"DEBUG: Forced accelerator.state.device to {accelerator.state.device}")
        
        is_sdxl = "xl" in self.base_model_id.lower()

        if is_sdxl:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.base_model_id,
                torch_dtype=DTYPE,
                use_safetensors=True,
                variant="fp16" if torch.cuda.is_available() else None,
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                self.base_model_id,
                torch_dtype=DTYPE,
                use_safetensors=True,
            )
        
        # Move pipeline to accelerator device
        pipe = pipe.to(accelerator.device, dtype=DTYPE)
        
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
            pin_memory=True,
            persistent_workers=True,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
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

        # # move encoders to right device after the accelerator is prepared
        # if is_sdxl:
        #     pipe.text_encoder = pipe.text_encoder.to(accelerator.device, dtype=DTYPE)
        #     pipe.text_encoder_2 = pipe.text_encoder_2.to(accelerator.device, dtype=DTYPE)
        #     pipe.vae = pipe.vae.to(accelerator.device, dtype=DTYPE)
        # else:
        #     pipe.text_encoder = pipe.text_encoder.to(accelerator.device, dtype=DTYPE)
        #     pipe.vae = pipe.vae.to(accelerator.device, dtype=DTYPE)

        # prep these with the accelerator
        pipe.unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            pipe.unet, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )

        # move encoders again to make sure
        # if is_sdxl:
        #     pipe.text_encoder = pipe.text_encoder.to(accelerator.device, dtype=DTYPE)
        #     pipe.text_encoder_2 = pipe.text_encoder_2.to(accelerator.device, dtype=DTYPE)
        #     pipe.vae = pipe.vae.to(accelerator.device, dtype=DTYPE)
        # else:
        #     pipe.text_encoder = pipe.text_encoder.to(accelerator.device, dtype=DTYPE)
        #     pipe.vae = pipe.vae.to(accelerator.device, dtype=DTYPE)
        
        print(f"DEBUG: pipe.unet.device: {pipe.unet.device}")
        print(f"DEBUG: pipe.text_encoder.device: {pipe.text_encoder.device}")
        print(f"DEBUG: pipe.text_encoder_2.device: {pipe.text_encoder_2.device}")

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
                        
                        # Debug the encoder_hidden_states shape
                        print(f"DEBUG: encoder_hidden_states shape: {encoder_hidden_states.shape}")
                        
                        # Get the expected hidden size for SDXL from the UNet config
                        expected_dim = None
                        if hasattr(pipe.unet, "config") and hasattr(pipe.unet.config, "cross_attention_dim"):
                            expected_dim = pipe.unet.config.cross_attention_dim
                        
                        # Check if we need to adjust the hidden states dimensions
                        if expected_dim is not None and encoder_hidden_states.shape[-1] != expected_dim:
                            # Create a projection layer if needed
                            projection = torch.nn.Linear(
                                encoder_hidden_states.shape[-1], 
                                expected_dim,
                                device=accelerator.device,
                                dtype=encoder_hidden_states.dtype
                            )
                            
                            # Apply the projection
                            encoder_hidden_states = projection(encoder_hidden_states)
                            print(f"DEBUG: Adjusted encoder_hidden_states shape: {encoder_hidden_states.shape}")
                            
                            # Verify shape is as expected
                            assert encoder_hidden_states.shape[0] == batch_size, "Batch size changed after projection!"
                            assert encoder_hidden_states.shape[1] == seq_len, "Sequence length changed after projection!"
                            assert encoder_hidden_states.shape[2] == expected_dim, "Hidden dimension not properly adjusted!"
                        
                        encoder_output = pipe.text_encoder_2(batch["input_ids"])
                        
                        # Add detailed debug information about encoder outputs
                        print(f"DEBUG: encoder_output type: {type(encoder_output)}")
                        print(f"DEBUG: encoder_output length: {len(encoder_output) if isinstance(encoder_output, tuple) else 'not tuple'}")
                        print(f"DEBUG: encoder_output[0] shape: {encoder_output[0].shape}")
                        
                        if isinstance(encoder_output, tuple) and len(encoder_output) > 1:
                            print(f"DEBUG: encoder_output[1] exists: {encoder_output[1] is not None}")
                            if encoder_output[1] is not None:
                                print(f"DEBUG: encoder_output[1] shape: {encoder_output[1].shape}")
                        
                        # Try to get pooled output from SDXL text_encoder_2
                        if isinstance(encoder_output, tuple) and len(encoder_output) > 1 and encoder_output[1] is not None:
                            # Some SDXL models have pooled output as second item
                            pooled_output = encoder_output[1]
                            print(f"DEBUG: Using pooled_output from encoder_output[1]: {pooled_output.shape}")
                        else:
                            # Check if we have a 2D tensor with shape [batch_size, hidden_dim]
                            if encoder_output[0].ndim == 2:
                                # This is already the right shape (batch_size, hidden_dim)
                                pooled_output = encoder_output[0]
                                print(f"DEBUG: Found 2D tensor with correct shape: {pooled_output.shape}")
                            elif encoder_output[0].ndim == 3:
                                # Get the pooled output by taking CLS token or using mean
                                if hasattr(pipe.text_encoder_2, "config") and hasattr(pipe.text_encoder_2.config, "projection_dim"):
                                    # Using CLS token (first token) is often better than mean
                                    pooled_output = encoder_output[0][:, 0, :]
                                    print(f"DEBUG: Using CLS token as pooled_output: {pooled_output.shape}")
                                else:
                                    # Fallback to mean with keepdim to preserve dimensions
                                    pooled_output = encoder_output[0].mean(dim=1, keepdim=True)
                            else:
                                pooled_output = encoder_output[0].reshape(1, -1)
                        
                        # Ensure we have the right shape [batch_size, hidden_dim]
                        if len(pooled_output.shape) == 1:
                            # Handle 1D tensor case
                            if hasattr(pipe.text_encoder_2, "config") and hasattr(pipe.text_encoder_2.config, "projection_dim"):
                                hidden_dim = pipe.text_encoder_2.config.projection_dim
                            else:
                                hidden_dim = 1280  # Default SDXL hidden dimension
                            
                            pooled_output = pooled_output.reshape(1, hidden_dim)
                            print(f"DEBUG: Reshaped 1D pooled_output to: {pooled_output.shape}")
                        
                        print(f"DEBUG: Final pooled_output shape: {pooled_output.shape}")
                        
                        # Get model's expected dimensions for debugging
                        if hasattr(pipe.unet, "config"):
                            if hasattr(pipe.unet.config, "addition_embed_dim"):
                                print(f"DEBUG: UNet expects addition_embed_dim: {pipe.unet.config.addition_embed_dim}")
                            if hasattr(pipe.unet.config, "addition_time_embed_dim"):
                                print(f"DEBUG: UNet expects addition_time_embed_dim: {pipe.unet.config.addition_time_embed_dim}")
                        
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
                        
                        print(f"DEBUG: time_ids shape: {time_ids.shape}")
                        
                        # Check tensor device and dtype
                        print(f"DEBUG: pooled_output device: {pooled_output.device}, dtype: {pooled_output.dtype}")
                        print(f"DEBUG: time_ids device: {time_ids.device}, dtype: {time_ids.dtype}")
                        print(f"DEBUG: encoder_hidden_states device: {encoder_hidden_states.device}, dtype: {encoder_hidden_states.dtype}")
                                                
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
                        
                        # Debug the encoder_hidden_states shape
                        print(f"DEBUG-Val: encoder_hidden_states shape: {encoder_hidden_states.shape}")
                        
                        # Get the expected hidden size for SDXL from the UNet config
                        expected_dim = None
                        if hasattr(pipe.unet, "config") and hasattr(pipe.unet.config, "cross_attention_dim"):
                            expected_dim = pipe.unet.config.cross_attention_dim
                        
                        # Check if we need to adjust the hidden states dimensions
                        if expected_dim is not None and encoder_hidden_states.shape[-1] != expected_dim:
                            # Create a projection layer if needed
                            projection = torch.nn.Linear(
                                encoder_hidden_states.shape[-1], 
                                expected_dim,
                                device=accelerator.device,
                                dtype=encoder_hidden_states.dtype
                            )
                            
                            # Apply the projection
                            encoder_hidden_states = projection(encoder_hidden_states)
                            print(f"DEBUG-Val: Adjusted encoder_hidden_states shape: {encoder_hidden_states.shape}")
                            
                            # Verify shape is as expected
                            assert encoder_hidden_states.shape[0] == batch_size, "Batch size changed after projection!"
                            assert encoder_hidden_states.shape[1] == seq_len, "Sequence length changed after projection!"
                            assert encoder_hidden_states.shape[2] == expected_dim, "Hidden dimension not properly adjusted!"
                        
                        encoder_output = pipe.text_encoder_2(batch["input_ids"])
                        
                        # Add detailed debug information about encoder outputs
                        print(f"DEBUG-Val: encoder_output type: {type(encoder_output)}")
                        print(f"DEBUG-Val: encoder_output length: {len(encoder_output) if isinstance(encoder_output, tuple) else 'not tuple'}")
                        print(f"DEBUG-Val: encoder_output[0] shape: {encoder_output[0].shape}")
                        
                        if isinstance(encoder_output, tuple) and len(encoder_output) > 1:
                            print(f"DEBUG-Val: encoder_output[1] exists: {encoder_output[1] is not None}")
                            if encoder_output[1] is not None:
                                print(f"DEBUG-Val: encoder_output[1] shape: {encoder_output[1].shape}")
                        
                        # Try to get pooled output from SDXL text_encoder_2
                        if isinstance(encoder_output, tuple) and len(encoder_output) > 1 and encoder_output[1] is not None:
                            # Some SDXL models have pooled output as second item
                            pooled_output = encoder_output[1]
                        else:
                            if encoder_output[0].ndim == 2:
                                pooled_output = encoder_output[0]
                            elif encoder_output[0].ndim == 3:
                                if hasattr(pipe.text_encoder_2, "config") and hasattr(pipe.text_encoder_2.config, "projection_dim"):
                                    pooled_output = encoder_output[0][:, 0, :]
                                else:
                                    pooled_output = encoder_output[0].mean(dim=1, keepdim=True)
                                    print(f"DEBUG-Val: Using mean with keepdim=True as pooled_output: {pooled_output.shape}")
                            else:
                                # Unexpected shape, try to adapt
                                print(f"DEBUG-Val: Unexpected tensor dimension: {encoder_output[0].ndim}")
                                # Reshape to expected format
                                pooled_output = encoder_output[0].reshape(1, -1)
                                print(f"DEBUG-Val: Reshaped to: {pooled_output.shape}")
                        
                        # Ensure we have the right shape [batch_size, hidden_dim]
                        if len(pooled_output.shape) == 1:
                            if hasattr(pipe.text_encoder_2, "config") and hasattr(pipe.text_encoder_2.config, "projection_dim"):
                                hidden_dim = pipe.text_encoder_2.config.projection_dim
                            else:
                                hidden_dim = 1280  # Default SDXL hidden dimension
                            
                            pooled_output = pooled_output.reshape(1, hidden_dim)
                            print(f"DEBUG-Val: Reshaped 1D pooled_output to: {pooled_output.shape}")
                        
                        print(f"DEBUG-Val: Final pooled_output shape: {pooled_output.shape}")
                        
                        # Get model's expected dimensions for debugging
                        if hasattr(pipe.unet, "config"):
                            if hasattr(pipe.unet.config, "addition_embed_dim"):
                                print(f"DEBUG-Val: UNet expects addition_embed_dim: {pipe.unet.config.addition_embed_dim}")
                            if hasattr(pipe.unet.config, "addition_time_embed_dim"):
                                print(f"DEBUG-Val: UNet expects addition_time_embed_dim: {pipe.unet.config.addition_time_embed_dim}")

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
                        
                        # Check tensor device and dtype
                        print(f"DEBUG-Val: time_ids shape: {time_ids.shape}")
                        print(f"DEBUG-Val: pooled_output device: {pooled_output.device}, dtype: {pooled_output.dtype}")
                        print(f"DEBUG-Val: time_ids device: {time_ids.device}, dtype: {time_ids.dtype}")
                        print(f"DEBUG-Val: encoder_hidden_states device: {encoder_hidden_states.device}, dtype: {encoder_hidden_states.dtype}")
                        
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