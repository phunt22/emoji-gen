import argparse
import sys
from pathlib import Path
from emoji_gen.generation import generate_emoji
from emoji_gen.config import DEFAULT_OUTPUT_PATH

def main():
    parser = argparse.ArgumentParser(description="Generate emojis from text prompts")
    parser.add_argument("prompt", help="Text prompt for emoji generation")
    parser.add_argument("--output", help="Output directory for generated images")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    args = parser.parse_args()

    # generates an emoji from a prompt with the active model selected by dev cli
    result = generate_emoji(
        prompt=args.prompt,
        output_path=args.output or str(DEFAULT_OUTPUT_PATH),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance
    )

    if result["status"] == "success":
        print(f"Generated emoji saved at: {result['image_path']}")
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    main() 