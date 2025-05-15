from setuptools import setup, find_packages

# running some setup and configuring the CLIs
setup(
    name="emoji-gen",
    version="0.1.0",
    packages=find_packages(),
    # auto install dependencies
    install_requires=[
        "torch",
        "diffusers",
        "Pillow",
        "python-dotenv",
        "flask",    
        "requests",
        "bs4",
        ## FLUX installation dependency
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
    }
)