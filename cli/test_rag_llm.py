import argparse
import logging
from pathlib import Path
# PIL might be needed if we decide to do more with the image, but for now, just confirming retrieval.

# Key functions from the generation module
from emoji_gen.generation import (
    _load_emoji_metadata_for_rag,
    get_rag_ip_adapter_inputs,
    augment_prompt_with_llm
)

# Auxiliary model loaders (which handle their own model loading)
# These are implicitly called by the functions above, but good to be aware of them.
# from emoji_gen.utils.aux_models import get_clip_pipeline, get_llm_pipeline

# For logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test RAG retrieval and LLM augmentation.")
    parser.add_argument("prompt", type=str, help="The input prompt to test.")
    args = parser.parse_args()

    input_prompt = args.prompt
    logger.info(f"Testing with prompt: \"{input_prompt}\"")

    # Test LLM Augmentation
    print("\n--- Testing LLM Augmentation ---")
    logger.info("Attempting LLM prompt augmentation...")
    augmented_prompt = augment_prompt_with_llm(input_prompt)
    print(f"Original Prompt: {input_prompt}")
    print(f"Augmented Prompt: {augmented_prompt}")

    # Test RAG Retrieval
    print("\n--- Testing RAG Retrieval ---")
    logger.info("Attempting RAG input retrieval...")
    
    # Ensure RAG data and embeddings are loaded. 
    # _load_emoji_metadata_for_rag is called internally by get_rag_ip_adapter_inputs if not already loaded,
    # but calling it here explicitly can help debug loading issues if they arise.
    _load_emoji_metadata_for_rag() 

    retrieved_image, rag_scale = get_rag_ip_adapter_inputs(input_prompt)

    if retrieved_image:
        print(f"Retrieved reference image for RAG.")
        # The get_rag_ip_adapter_inputs function logs the path of the retrieved image.
        # We are just confirming that a PIL.Image object was returned.
        print(f"  - Retrieved Image Object: {'Yes' if retrieved_image else 'No'}")
        print(f"  - Calculated RAG Scale: {rag_scale:.4f}" if rag_scale is not None else "N/A")
    else:
        print("No reference image retrieved for RAG, or an error occurred.")
        logger.warning("RAG retrieval did not return an image. Check previous logs for details (e.g., embeddings not found, CLIP model issues).")

    print("\nTest script finished.")

if __name__ == "__main__":
    # This ensures that aux_models can find their configs if they rely on relative paths from project root implicitly
    # However, aux_models.py and config.py should handle paths robustly.
    # Adding a check or specific initialization for model_manager might be needed if aux_models don't self-initialize fully.
    # For now, assuming get_clip_pipeline and get_llm_pipeline are self-sufficient.
    main() 