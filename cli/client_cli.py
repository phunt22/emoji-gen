import argparse
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io
import requests
from pathlib import Path
import sys
from server.config import DEFAULT_MODEL, DEFAULT_PORT, DEFAULT_OUTPUT_PATH


def main():
    parser = argparse.ArgumentParser(description="EmojiGen CLI")
    parser.add_argument("prompt", nargs=argparse.REMAINDER, help="Prompt for emoji generation")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Model name to use")
    parser.add_argument("-o", "--output", help="Output directory for generated images")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port the server is running on")
    
    args = parser.parse_args()
    
    # Get the prompt
    prompt = " ".join(args.prompt)
    if not prompt:
        print("Error: Please provide a prompt")
        sys.exit(1) ## exit the CLI
    
    # Set up the request
    request_data = {
        "prompt": prompt,
        "model_choice": args.model or DEFAULT_MODEL,
        "output_path": args.output or str(DEFAULT_OUTPUT_PATH),
        "verbose": args.verbose
    }
    
    port = args.port or DEFAULT_PORT

    
    api_url = f"http://localhost:{port}"
    
    if args.verbose:
        print(f"Request data: {request_data}")
        print(f"Sending request to server at: {api_url}/generate")
    
    try:
        # check if sevrer running
        try:
            requests.get(f"{api_url}/status")
        except requests.exceptions.ConnectionError:
            print("Error: Server is not running!")
            print(f"Start the server with: python cli/dev_cli.py start [--port {args.port}]")
            sys.exit(1)
        
        # call API to generate
        response = requests.post(f"{api_url}/generate", json=request_data)
        response.raise_for_status()
        result = response.json()

        if result["status"] == "success":
            print(f"Emoji generated successfully! Image saved at: {result['image_path']}")
            
            # show image
            image_path = result["image_path"]
            image_response = requests.get(image_path)
            image_response.raise_for_status()
            
            image = Image.open(io.BytesIO(image_response.content))
            image.show()
            
            # save image
            output_path = Path(result["image_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(image_response.content)
            
            print(f"Generated emoji saved at: {output_path}")
        else:
            print(f"[ERROR] Failed to generate emoji: {result['error']}")
            
    except Exception as e:
        print(f"Error: {e}")
        print(f"Make sure the server is running with: python cli/dev_cli.py start --port {args.port}")

if __name__ == "__main__":
    main() 