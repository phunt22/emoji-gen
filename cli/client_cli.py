import argparse
import sys
from PIL import Image
from pathlib import Path
from emoji_gen.generation import generate_emoji
from emoji_gen.config import DEFAULT_MODEL, DEFAULT_OUTPUT_PATH

def main():
    parser = argparse.ArgumentParser(description="EmojiGen CLI")
    parser.add_argument("prompt", nargs=argparse.REMAINDER, help="Prompt for emoji generation")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Model name to use")
    parser.add_argument("-o", "--output", help="Output directory for generated images")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    
    args = parser.parse_args()
    
    # Get the prompt
    prompt = " ".join(args.prompt)
    if not prompt:
        print("Error: Please provide a prompt")
        sys.exit(1)
    
    if args.verbose:
        print(f"Generating emoji with prompt: {prompt}")
        print(f"Using model: {args.model}")
    
    try:
        # Generate the emoji directly
        result = generate_emoji(
            prompt=prompt,
            model_choice=args.model or DEFAULT_MODEL,
            output_path=args.output or str(DEFAULT_OUTPUT_PATH),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance
        )
        
        if result["status"] == "success":
            print(f"Emoji generated successfully! Image saved at: {result['image_path']}")
            
            # Display the image
            image_path = Path(result["image_path"])
            if image_path.exists():
                Image.open(image_path).show()
            
        else:
            print(f"[ERROR] Failed to generate emoji: {result['error']}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 