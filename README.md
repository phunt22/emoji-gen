# Emoji Gen

A tool to generate emojis from a prompt

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

The developer CLI provides tools for preparing data for training:

```bash
# Run all prepation steps to grab the emoji data from Apple for fine-tuning
emoji-dev prepare
```

After running prepare, /data will contain /emoji, which has text-image pairs. /raw contains those those images with the backgrounds removed, and are named corresponding to the .name field of each emoji in emojis.json. emojis.json connects each emoji to it's name and image. emojisPruned.json is the same, but skin color is removed to make the fine-tuning data simpler.
All emoji photos are 160x160 pixels

### User CLI

WORK IN PROGRESS AND ISNT FUNCTIONAL YET
Todo: put in options about which model to use (RAG? DISTILLED? etc.)

The user CLI allows generating custom emojis:

```bash
emoji-gen --gen "happy cat"
```

## Cleaning Python Cache Files

If you see `__pycache__` directories or `.pyc` files, you can clean them up with:

```bash
# Remove all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -r {} +

# Remove all .pyc files
find . -name "*.pyc" -delete
```

These files are ignored by git anyways, but are a little annoying in the code editor
You can hide them in VSCode by going to Settings, searching "files:exclude", and adding the pattern "\_\_/pycache\*\*"
I also added "\*\*.egg_info"
