from setuptools import setup, find_packages

# running some setup and configuring the CLIs
setup(
    name="emoji_gen",
    version="0.1.0",
    packages=find_packages(),
    # auto install dependencies
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio"
        "diffusers @ git+https://github.com/huggingface/diffusers.git", ## IMPORTANT from the git here
        "transformers",
        "accelerate",
        "safetensors",
        "pillow",
        "numpy",
        "tqdm",
        "shutil",
        
        # fine tuning dependencies
        "peft",  # For LoRA
        "xformers",
        "bitsandbytes", ## only needed for 8bit adam optim
        # web/server dependencies (ensure flask is listed once)
        "flask",
        "flask-cors",
        "python-dotenv",
        "requests",
        "beautifulsoup4",
        "protobuf", 
        "sentencepiece",
        
    ],
    # registers the dev_cli main function as "emoji-dev" in the CLI
    # client CLI is emoji-gen
    entry_points={
        "console_scripts": [
            "emoji-dev=cli.dev_cli:main",  # dev cli
            "emoji-gen=cli.client_cli:main"  # client cli
        ]
    },
    python_requires=">=3.8",
)