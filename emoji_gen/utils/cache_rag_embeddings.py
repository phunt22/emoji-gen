import torch
import json
from pathlib import Path
import logging

from emoji_gen.utils.aux_models import get_clip_pipeline
from emoji_gen.config import RAG_DATA_PATH # To confirm RAG dir structure if needed, though not directly used for images here

logger = logging.getLogger(__name__)

# Define paths relative to the project structure
# Assumes this script is in emoji_gen/utils/ and data/ is at project_root/data/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EMOJIS_PRUNED_JSON_PATH = DATA_DIR / "emojisPruned.json"
OUTPUT_EMBEDDINGS_PATH = DATA_DIR / "rag_embeddings.pt"

def compute_and_cache_rag_embeddings():
    """ Computes all the RAG embeddings for the emoji list at once, then caches them """
    logger.info(f"Starting RAG caption embedding computation. Output to: {OUTPUT_EMBEDDINGS_PATH}")
    print(f"Starting RAG caption embedding computation. Output to: {OUTPUT_EMBEDDINGS_PATH}")

    if not EMOJIS_PRUNED_JSON_PATH.exists():
        logger.error(f"{EMOJIS_PRUNED_JSON_PATH} not found. Cannot compute RAG embeddings.")
        print(f"ERROR: {EMOJIS_PRUNED_JSON_PATH} not found. Cannot compute RAG embeddings.")
        return

    clip_model, clip_processor = get_clip_pipeline()
    if not clip_model or not clip_processor:
        logger.error("CLIP model or processor not available. Cannot compute RAG embeddings.")
        print("ERROR: CLIP model/processor not available for computing RAG embeddings.")
        return

    try:
        with open(EMOJIS_PRUNED_JSON_PATH, 'r', encoding='utf-8') as f:
            emoji_metadata_list = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {EMOJIS_PRUNED_JSON_PATH}: {e}", exc_info=True)
        print(f"ERROR: Failed to load {EMOJIS_PRUNED_JSON_PATH}.")
        return

    if not emoji_metadata_list:
        logger.warning(f"{EMOJIS_PRUNED_JSON_PATH} is empty. No embeddings to compute.")
        print(f"Warning: {EMOJIS_PRUNED_JSON_PATH} is empty. No RAG embeddings to compute.")
        
        # just save empty if we are empty, but we logged
        torch.save({
            'captions': [],
            'embeddings': torch.empty(0), 
            'image_filenames': []
        }, OUTPUT_EMBEDDINGS_PATH)
        logger.info(f"Saved empty RAG embeddings structure to {OUTPUT_EMBEDDINGS_PATH}")
        return

    captions_for_embedding = []
    image_filenames_for_rag = []

    for item in emoji_metadata_list:
        # use the built in fields from the json file
        caption = item.get("processed") ## name
        link = item.get("link") ## image
        if caption and link:
            captions_for_embedding.append(caption)
            image_filenames_for_rag.append(Path(link).name)
        else:
            logger.warning(f"Skipping item for RAG embedding due to missing caption or link: {item.get('name', 'N/A')}")

    if not captions_for_embedding:
        logger.error("No valid captions found after filtering. Cannot compute RAG embeddings.")
        print("ERROR: No valid captions found to compute RAG embeddings.")
        torch.save({
            'captions': [],
            'embeddings': torch.empty(0),
            'image_filenames': []
        }, OUTPUT_EMBEDDINGS_PATH)
        return

    logger.info(f"Computing embeddings for {len(captions_for_embedding)} captions...")
    print(f"Computing embeddings for {len(captions_for_embedding)} captions...")

    all_embeddings_list = []
    batch_size = 32 ## can adjust, naive start

    try:
        for i in range(0, len(captions_for_embedding), batch_size):
            batch_captions = captions_for_embedding[i:i+batch_size]
            inputs = clip_processor(text=batch_captions, return_tensors="pt", padding=True, truncation=True).to(clip_model.device)
            with torch.no_grad():
                batch_embeddings = clip_model.get_text_features(**inputs)
            all_embeddings_list.append(batch_embeddings.cpu()) ## save gpu memory, especially with 1900 emojis
            if (i // batch_size + 1) % 10 == 0: 
                logger.info(f"Processed batch {i // batch_size + 1} / {len(captions_for_embedding) // batch_size + 1}")
                print(f"Processed batch {i // batch_size + 1} of {len(captions_for_embedding) // batch_size + 1}")


        # grab the final vector embedding then normalize
        final_embeddings_tensor = torch.cat(all_embeddings_list, dim=0)
        final_embeddings_tensor = final_embeddings_tensor / final_embeddings_tensor.norm(p=2, dim=-1, keepdim=True)


        # this makes a file with the torch tensors that are the embeddings
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({
            'captions': captions_for_embedding, ## store the actaul captions
            'embeddings': final_embeddings_tensor,
            'image_filenames': image_filenames_for_rag ## store the actual file names
        }, OUTPUT_EMBEDDINGS_PATH)
        logger.info(f"Successfully computed and saved RAG embeddings to {OUTPUT_EMBEDDINGS_PATH}")
        print(f"Successfully saved RAG embeddings for {len(captions_for_embedding)} items to {OUTPUT_EMBEDDINGS_PATH}")

    except Exception as e:
        logger.error(f"Error during embedding computation or saving: {e}", exc_info=True)
        print(f"ERROR: An error occurred during RAG embedding computation: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    compute_and_cache_rag_embeddings() 