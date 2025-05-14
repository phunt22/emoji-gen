# Emoji Gen

A tool to generate emojis from a prompt using Stable Diffusion models.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/emoji-gen.git
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

## Usage

### Developer CLI

The developer CLI provides tools for preparing data for training and managing models:

```bash
# Run all preparation steps to grab the emoji data from Apple for fine-tuning
emoji-dev prepare

# List all available models
emoji-dev list-models

# Fine-tune a model on the emoji dataset
emoji-dev fine-tune --model sd-v1.5 --output-name my_fine_tuned_model

# List all fine-tuned models
emoji-dev list-fine-tuned
```

After running prepare, /data will contain /emoji, which has text-image pairs. /raw contains those images with the backgrounds removed, and are named corresponding to the .name field of each emoji in emojis.json. emojis.json connects each emoji to its name and image. emojisPruned.json is the same, but skin color is removed to make the fine-tuning data simpler.
All emoji photos are 160x160 pixels.

### User CLI

The user CLI allows generating custom emojis:

```bash
# Generate an emoji with default settings
emoji-gen "happy cat"

# Generate with specific model and parameters
emoji-gen "happy cat" --model sd-v1.5 --steps 30 --guidance 8.0

# Save to a specific output directory
emoji-gen "happy cat" --output ./my_emojis
```

## Running on a VM

This project is designed to run directly on a VM with GPU support. All operations are performed locally without a client-server architecture.

## Cleaning Python Cache Files

If you see `__pycache__` directories or `.pyc` files, you can clean them up with:

```bash
# Remove all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -r {} +

# Remove all .pyc files
find . -name "*.pyc" -delete
```

These files are ignored by git anyways, but are a little annoying in the code editor.
You can hide them in VSCode by going to Settings, searching "files:exclude", and adding the pattern "\_\_/pycache\*\*"
I also added "\*\*.egg_info"
