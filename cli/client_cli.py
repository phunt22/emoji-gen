import argparse
from emoji_gen.generation import generate_emoji
from emoji_gen.server_client import is_server_running, generate_emoji_remote
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate emojis from text prompts")
    parser.add_argument("prompt", nargs=argparse.REMAINDER, help="Text prompt for emoji generation")
    parser.add_argument("--output", "-o", type=str, help="Output path for the generated image")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--local", action="store_true", help="Force local generation even if server is available")
    
    args = parser.parse_args()
    
    prompt = " ".join(args.prompt)
    if not prompt:
        print("Error: Prompt is required")
        return
    
    # check if server is running
    server_running, server_info = is_server_running()
    
    if server_running and not args.local:
        print(f"Using server with model: {server_info['model']}")
        result = generate_emoji_remote(
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            output_path=args.output
        )
        
        if result and "status" in result and result["status"] == "success":
            if "image_path" in result:
                print(f"Generated emoji saved to: {result['image_path']}")
            else:
                print("Successfully generated image but path not returned")
        else:
            error_msg = result.get("error", "Unknown error") if result else "Empty response from server"
            print(f"Error: {error_msg}")
            print("Falling back to local generation...")
            generate_locally(prompt, args)
    else:
        if args.local:
            print("Using local generation (--local flag specified)")
        else:
            print("Server not available, using local generation")
        generate_locally(prompt, args)

def generate_locally(prompt, args):
    """Generate emoji locally using the model directly."""
    print("Loading pipeline components...")
    result = generate_emoji(
        prompt=prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        output_path=args.output
    )
    
    if result and "status" in result:
        if result["status"] == "success" and "image_path" in result:
            print(f"Generated emoji saved to: {result['image_path']}")
        else:
            error_msg = result.get("error", "Unknown error")
            print(f"Error generating emoji: {error_msg}")
    else:
        print(f"Generated emoji saved to: {result}")

if __name__ == "__main__":
    main() 