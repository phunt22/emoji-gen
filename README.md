# Emoji Gen

A project of generating emojis and fine-tuning open source models.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/phunt22/emoji-gen.git
   cd emoji-gen
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## GCP Setup and Management

0. Follow these instructions to set up your GCP VM:
   https://rayxsong.github.io/research/2024/WI/Nera/Nera-Weekly-Update-1

# Keep in mind that T4 GPUs are hard to get (think about an L4, etc.) and that Spot-Instances are much cheaper and work for this project

1. Initialize gcloud:

   ```bash
   gcloud init
   # Choose option 1 to re-initialize
   # Log into your GCP account
   # Select your project
   ```

2. Sanity check configuration:

   ```bash
   gcloud config list
   ```

3. VM Management:

   ```bash
   # List all compute instances
   gcloud compute instances list
   # RUNNING = being charged, TERMINATED = not being charged
   # Keep in mind of the instance name, which should be the format "2025****-******"

   # Stop a VM instance
   ## IMPORTANT: If you dont stop your VMs, you will get charged a lot. Make sure to stop when you are not using.
   gcloud compute instances stop instance-[INSTANCE_NAME]

   # Start a VM instance
   gcloud compute instances start instance-[INSTANCE_NAME]
   ```

4. Set up environment variables (for syncing images from VM to local):
   Create a `.env` file in the project root on your local computer:

   ```
   # these should be obtained from
   GCP_VM_EXTERNAL_IP=your_vm_ip
   GCP_INSTANCE_NAME=your_instance_name

   # The directory on your local computer where you want generated emojis to go
   EMOJI_LOCAL_SYNC_DIR=/path/to/local/sync/dir
   ```

Note: Installation steps must be completed on the GCP VM and your local computer.

## Usage

#### Note: Python virtual env must be activated

### User CLI (emoji-gen)

Generate custom emojis with text prompts:

```bash
# Basic usage
emoji-gen cat with sunglasses

# Advanced options
emoji-gen happy face with sunglasses --steps 30 --guidance 8.0

# You can also run a set of multiple prompts at once. Just make sure your server is started and run:

emoji-gen --benchmark

# To run all of the prompts in the emoji_gen/benchmarks/prompts.txt file
# Pass in the --name [BENCHMARK_DIR_NAME] to name your output folder (otherwise is benchmark[DATETIME])


```

Parameters:

# None needed

- `--steps`: Number of inference steps (default: 25)
- `--guidance`: Guidance scale (default: 7.5)
- `--output`: Custom output directory

### Developer CLI (emoji-dev)

Manage models and fine-tuning:

```bash
# List available models
emoji-dev list-models

# Set active model
emoji-dev set-model sd-v1.5

**Note on `set-model`**: The `emoji-dev set-model [MODEL]` command can sometimes encounter memory issues on the GPU, as the server might keep the previous model in memory. Currently working on fixing.
A more reliable approach to switch models if you encounter issues is to start (or restart) the server with the desired model preloaded:
`$ emoji-dev start-server --model [MODEL]`

# Prepare emoji dataset for training
emoji-dev prepare

# Fine-tune a model
emoji-dev fine-tune --model sd-v1.5 --output-name my_model

## More information about these flags is on the actual stable diffusion docs (https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md) but:
Key flags for `fine-tune` include:
*   `--model`: Base model ID or path (default: from `config.py`).
*   `--output`: Name for the output model directory (auto-generated otherwise).

* Training Parameters:*
*   `--max-train-steps`: (default: 500)
*   `--batch-size`: (default: 1)
*   `--learning-rate`: (default: 1e-4)
*   `--gradient-accumulation-steps`: (default: 4)
*   `--lora-rank`: (default: 32)
*   `--seed`: (default: 42)
*   `--resolution`: Training image resolution (e.g., 512, 1024; default depends on model).
*   `--mixed-precision`: 'no', 'fp16', 'bf16' (default: 'fp16').
*   `--lr-scheduler`: (default: 'constant')
*   `--lr-warmup-steps`: (default: 0)
*   `--checkpointing-steps`: Save checkpoint every X steps.
*   `--checkpoints-total-limit`: Max checkpoints to store (default: 2).

*Prompt and Validation Parameters:*
*   `--instance-prompt`: Your unique subject prompt (default: 'sks emoji').
*   `--class-prompt`: Regularization prompt for the general class.
*   `--validation-prompt`: Prompt for generating validation images.
*   `--validation-epochs`: Run validation every X epochs.

*SDXL-Specific Parameters:*
*   `--vae-path`: Path to SDXL VAE (default: 'madebyollin/sdxl-vae-fp16-fix').
*   `--enable-xformers`: (SDXL default: True)
*   `--gradient-checkpointing`: (SDXL default: True)
*   `--use-8bit-adam`: (SDXL default: True)

*Output and Tracking (all False by default):*
*   `--report-to`: 'wandb', 'tensorboard'.
*   `--push-to-hub`: Upload to Hugging Face Hub.
*   `--hub-model-id`: Repository ID for the Hub.

# List fine-tuned models
emoji-dev list-fine-tuned

# Sync generated images from local machine
# RUN THIS FROM YOUR LOCAL MACHINE
emoji-dev sync

# Start inference server
# This makes consecutive prompts (much) faster
emoji-dev start-server
```

### Available Models

Base models are defined in `emoji_gen/config.py`:

```python
MODEL_ID_MAP = {
    "sd-v1.5": "runwayml/stable-diffusion-v1-5",
    # Add more models here
}
```

Fine-tuned models are stored in the `fine_tuned_models` directory and are automatically detected by the system.

## Development

### Code Structure

```
emoji-gen/
├── emoji_gen/          # Core package
│   ├── models/         # Model management and fine-tuning
│   ├── data_utils/     # Dataset handling
│   └── generation.py   # Emoji generation logic
├── cli/               # Command-line Interfaces
├── data/              # Dataset storage
└── fine_tuned_models/ # Fine-tuned model storage
```

### Cleaning Python Cache

Remove cache files:

```bash
# Remove __pycache__ directories
find . -type d -name "__pycache__" -exec rm -r {} +

# Remove .pyc files
find . -name "*.pyc" -delete
```

VSCode settings to hide cache:

- Add `"**/__pycache__**"` to `files:exclude`
- Add `"**/*.egg_info"` to `files:exclude`

## Acknowledgments

- Stable Diffusion Team
- Evan Zhou
