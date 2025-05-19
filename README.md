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
   ## IMPORTANT: If you dont stop your VMs, you will get charged a lot. Make sure to stop when you are not using
   gcloud compute instances stop instance-[INSTANCE_NAME]

   # Start a VM instance
   gcloud compute instances start instance-[INSTANCE_NAME]
   ```

4. Set up environment variables:
   Create a `.env` file in the project root:

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
emoji-dev list

# Set active model
emoji-dev set-model sd-v1.5

# Prepare emoji dataset for training
emoji-dev prepare

# Fine-tune a model
emoji-dev fine-tune --model sd-v1.5 --output-name my_model

# List fine-tuned models
emoji-dev list-fine-tuned

# Sync generated images to local machine
emoji-dev sync

# Start inference server
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
