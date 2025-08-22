import argparse
from emoji_gen.generation import generate_emoji
from emoji_gen.server_client import is_server_running, generate_emoji_remote
import os
from pathlib import Path
from datetime import datetime

# Use relative path based on the script location, not absolute paths
BENCHMARK_PROMPTS_FILE = Path(__file__).parent / "prompts" / "benchmark_prompts.txt"

def main():
    parser = argparse.ArgumentParser(description="Generate emojis from text prompts")
    parser.add_argument("prompt", nargs=argparse.REMAINDER, help="Text prompt for emoji generation")
    parser.add_argument("--output", "-o", type=str, help="Output path for the generated image")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--local", action="store_true", help="Force local generation even if server is available")
    parser.add_argument("--benchmark", action="store_true", help="Run in benchmark mode")  # Changed to boolean flag
    parser.add_argument("--model", type=str, default="sd-v1.5", help="Model to use for generation")
    parser.add_argument("--name", type=str, help="Name of the output folder for the benchmark")
    parser.add_argument("--rag", action="store_true", help="Use RAG for generation")
    parser.add_argument("--llm", action="store_true", help="Use LLM to augment prompts for generation")
    args = parser.parse_args()

    if args.benchmark:
        name = args.name if args.name else f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_benchmark(args.model, name, args.steps, args.guidance, args.local, use_llm=args.llm, use_rag=args.rag)
        return
    
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
            use_rag=args.rag,
            use_llm=args.llm,
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
        use_rag=args.rag,
        use_llm=args.llm,
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

def run_benchmark(model_name, output_name, num_steps=40, guidance_scale=9, force_local=False, use_llm=False, use_rag=False):
    """Run inference on prompts from the benchmark prompts file."""
    # Create output directory structure with portable path
    benchmark_dir = Path('generated_emojis') / output_name
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize prompts list before try block
    prompts = []
    
    # Check if benchmark prompts file exists
    if not BENCHMARK_PROMPTS_FILE.exists():
        # Create the directory if it doesn't exist
        BENCHMARK_PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Benchmark prompts file not found: {BENCHMARK_PROMPTS_FILE}")
        print(f"Please create this file with your prompts (one per line)")
        return
    
    # Read prompts from file
    try:
        with open(BENCHMARK_PROMPTS_FILE, 'r') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"Error reading benchmark prompts file: {e}")
        return
    
    if not prompts:
        print("No prompts found in the benchmark file")
        return
    
    print(f"Running benchmark with {len(prompts)} prompts...")
    print(f"Saving results to: {benchmark_dir}")
    
    # Check if server is running
    server_running, server_info = is_server_running()
    
    # Generate emojis for each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"Generating emoji {i}/{len(prompts)}: {prompt}")
        
        if server_running and not force_local and server_info is not None:
            print(f"Using server with model: {server_info.get('model', 'unknown')}")
            result = generate_emoji_remote(
                prompt=prompt,
                num_inference_steps=num_steps,
                use_rag=use_rag,
                use_llm=use_llm,
                guidance_scale=guidance_scale,
                output_path=str(benchmark_dir)
            )
        else:
            if force_local:
                print("Using local generation (--local flag specified)")
            else:
                print("Server not available, using local generation")
            result = generate_emoji(
                prompt=prompt,
                num_inference_steps=num_steps,
                use_rag=use_rag,
                use_llm=use_llm,
                guidance_scale=guidance_scale,
                output_path=str(benchmark_dir)
            )
        
        if result and result.get("status") == "success":
            print(f"✓ Generated: {result['image_path']}")
        else:
            error_msg = result.get("error", "Unknown error") if result else "Empty response"
            print(f"✗ Failed to generate: {error_msg}")
    
    print(f"\nBenchmark complete! Results saved to: {benchmark_dir}")

if __name__ == "__main__":
    main() 