from setuptools import setup, find_packages

# running some setup and configuring the CLIs
setup(
    name="emoji-gen-dev",
    version="0.1",
    packages=find_packages(),
    # auto install dependencies
    install_requires=[
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0",
        "Pillow>=9.0.0"
    ],
    # registers the dev_cli main function as "emoji-dev" in the CLI
    # client CLI is emoji-gen
    entry_points={
        "console_scripts": [
            "emoji-dev=cli.dev_cli:main", ## dev cli
            "emoji-gen=cli.client_cli:main" ## client cli
        ]
    }
)