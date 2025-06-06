import logging
from pathlib import Path
from datetime import datetime
from PIL import Image
from typing import Optional, Tuple, List
import json
import torch 
from importlib import import_module ## want dynamic imports

from emoji_gen.models.model_manager import model_manager
from emoji_gen.config import (
    LLM_SYSTEM_PROMPT, RAG_DATA_PATH,
)
from emoji_gen.utils.aux_models import get_llm_pipeline, get_clip_pipeline

logger = logging.getLogger(__name__)

# file globals for RAG
_emoji_metadata_for_rag: List[dict] = []
_emoji_captions_for_rag: List[str] = []
_emoji_image_local_paths_for_rag: List[Path] = []
_rag_data_loaded = False

# global vars for RAG (only cache once)
_cached_rag_captions: List[str] = []
_cached_rag_embeddings: Optional[torch.Tensor] = None
_cached_rag_image_paths: List[Path] = []
_rag_data_loaded = False ## flag to see if we are ready to use

PROJECT_ROOT_GEN = Path(__file__).resolve().parent.parent # emoji_gen folder
DATA_DIR_GEN = PROJECT_ROOT_GEN / "data"
CACHED_EMBEDDINGS_PATH = DATA_DIR_GEN / "rag_embeddings.pt"

def _load_emoji_metadata_for_rag():
    """ Loads pre-computed RAG caption embeddings and corresponding image paths """
    global _cached_rag_captions, _cached_rag_embeddings, _cached_rag_image_paths, _rag_data_loaded

    if _rag_data_loaded:
        return

    if not CACHED_EMBEDDINGS_PATH.exists():
        logger.error(f"RAG embeddings cache file not found at {CACHED_EMBEDDINGS_PATH}. Please run the caching script.")
        _rag_data_loaded = False 

    try:
        cached_data = torch.load(CACHED_EMBEDDINGS_PATH)
        _cached_rag_captions = cached_data.get('captions', [])
        _cached_rag_embeddings = cached_data.get('embeddings') ## ALREADY TENSOR
        cached_image_filenames = cached_data.get('image_filenames', [])

        if not _cached_rag_captions or _cached_rag_embeddings is None or _cached_rag_embeddings.nelement() == 0 or not cached_image_filenames:
            logger.error(f"Loaded RAG cache {CACHED_EMBEDDINGS_PATH} is empty or malformed.")
            _rag_data_loaded = False

            # clear (potentially) partially loaded data
            _cached_rag_captions = []
            _cached_rag_embeddings = None
            _cached_rag_image_paths = []
            return

        # able to get the image paths
        base_image_path_for_rag = Path(RAG_DATA_PATH)
        if not base_image_path_for_rag.exists():
            logger.error(f"RAG image directory {base_image_path_for_rag} (from config RAG_DATA_PATH) does not exist. Cannot form image paths.")
            _rag_data_loaded = False
            return
        
        _cached_rag_image_paths = [base_image_path_for_rag / fname for fname in cached_image_filenames]
        
        # verify sample path as a sanity check, fail early if not okay
        if _cached_rag_image_paths and not _cached_rag_image_paths[0].exists():
             logger.warning(f"Sample RAG image path {_cached_rag_image_paths[0]} does not exist. Check RAG_DATA_PATH and data preparation.")

        logger.info(f"Successfully loaded {len(_cached_rag_captions)} items from RAG embeddings cache {CACHED_EMBEDDINGS_PATH}.")
        _rag_data_loaded = True

    except Exception as e:
        logger.error(f"Error loading RAG embeddings cache from {CACHED_EMBEDDINGS_PATH}: {e}", exc_info=True)
        # clear everything
        _cached_rag_captions = []
        _cached_rag_embeddings = None
        _cached_rag_image_paths = []
        _rag_data_loaded = False


def augment_prompt_with_llm(original_prompt: str) -> str:
    llm_model, llm_tokenizer = get_llm_pipeline()
    if not llm_model or not llm_tokenizer:
        logger.warning("LLM not available for prompt augmentation. Returning original prompt.")
        return original_prompt

    # config
    system_prompt = LLM_SYSTEM_PROMPT
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": original_prompt}
    ]

    try:
        # Use the chat template to format the input correctly for the model
        input_ids = llm_tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(llm_model.device)

        # 77 comes from the fact that model is fine tuned on max of 77 char
        outputs = llm_model.generate(input_ids, max_new_tokens=77, num_beams=4, early_stopping=True)
        
        # decode only newly generated tokens, not the input prompt
        augmented_prompt = llm_tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        
        logger.info(f"Original prompt: '{original_prompt}', Augmented prompt: '{augmented_prompt}'")
        return augmented_prompt
    except Exception as e:
        logger.error(f"Error during LLM prompt augmentation: {e}", exc_info=True)
        return original_prompt


def get_rag_ip_adapter_inputs(prompt: str) -> Tuple[Optional[Image.Image], Optional[float]]:
    """ Take the user's prompt, and find the image corresponding to the most similar prompt (and associated CLIP score)"""
    _load_emoji_metadata_for_rag() 
    clip_model, clip_processor = get_clip_pipeline()

    # ensure global vars
    if not _rag_data_loaded or not _cached_rag_captions or _cached_rag_embeddings is None or not clip_model or not clip_processor:
        logger.warning("RAG data, cached embeddings, or CLIP model not available. Skipping RAG.")
        return None, None

    try:
        # encode prompt and normalize
        prompt_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True, truncation=True).to(clip_model.device)
        prompt_embedding = clip_model.get_text_features(**prompt_inputs)
        prompt_embedding = prompt_embedding / prompt_embedding.norm(p=2, dim=-1, keepdim=True)

        # use the pre-computed embeddings
        all_caption_embeddings = _cached_rag_embeddings.to(prompt_embedding.device)
        
        # calculate similarities and find the best match
        similarities = torch.matmul(prompt_embedding, all_caption_embeddings.T).squeeze(0)
        
        if similarities.numel() == 0:
            logger.warning("No similarities calculated for RAG. Skipping.")
            return None, None

        best_match_idx = torch.argmax(similarities).item()
        best_clip_score = similarities[best_match_idx].item()
        retrieved_image_path = _cached_rag_image_paths[best_match_idx]
        retrieved_image: Optional[Image.Image] = None

        if retrieved_image_path.exists():
            retrieved_image = Image.open(retrieved_image_path).convert("RGB")
            logger.info(f"RAG: Found similar emoji '{_cached_rag_captions[best_match_idx]}' with score {best_clip_score:.4f} at {retrieved_image_path}")
        else:
            logger.warning(f"RAG: Local image not found at {retrieved_image_path}. Ensure images are downloaded correctly.")

        if retrieved_image is None:
            logger.warning("RAG: Failed to load image for IP-Adapter.")
            return None, None
        

        # TODO find a good function to map CLIP to IP-Adapter Scale
        # CLIP is [-1, 1]
        # IP-Adapter is [0, 1]
        # there are a few ways to map this.
        # 1. IP = ReLU(CLIP)
        # 2. IP = (CLIP + 1) / 2
        # 3. Non-linear clamp mapping
        # Likely want to bias towards a mid range unless we have a bad score or a really good one

        # map [-1, 1] to [0, 1]

        # however, we want some variation so drop below 1 to 
        # we also want the reference image to matter for styling, so dont drop too low
        # 0.5 is an even mix of style and 



        rag_scale = max(0.0, best_clip_score) ## ReLU
        # rag_scale = (best_clip_score + 1) / 2 
        rag_scale = max(0.0, min(1.0, rag_scale)) ## need to be [0,1], sanity check

        rag_scale = 0.5

        return retrieved_image, rag_scale

    except Exception as e:
        logger.error(f"Error during RAG input retrieval: {e}", exc_info=True)
        return None, None

# generates an emoji from a prompt using the active model (from model_manager specified in dev_cli)
def generate_emoji(
    prompt: str,
    output_path: Optional[str] = None,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    use_rag: bool = False,
    use_llm: bool = False
    # num_images: int = 1 ## functionality to save multiple images is not implemented
):
   
    try:
        # get model from the cache (inits default if needed)
        model_pipeline = model_manager.get_active_model()
        if not model_pipeline:
            logger.error("Failed to retrieve active model from model_manager.")
            return {"status": "error", "error": "Failed to initialize model"}
        
        final_prompt = prompt

        if use_llm:
            logger.info("Augmenting prompt with LLM...")
            final_prompt = augment_prompt_with_llm(prompt)
        
        model_call_params = {
            "prompt": final_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale
        }

        ## check if model is the sd3 ip adapter, not basically removing the use of the rag flag
        pipeline_class_name = model_pipeline.__class__.__name__
        is_sd3_ipadapter = "StableDiffusion3PipelineIPAdapter" in pipeline_class_name

        if use_rag and is_sd3_ipadapter:
            logger.info("Retrieving RAG inputs for IP-Adapter...")
            rag_reference_image, rag_scale_value = get_rag_ip_adapter_inputs(final_prompt)
            
            if rag_reference_image and rag_scale_value is not None:
                logger.info(f"Using RAG with retrieved image and calculated scale {rag_scale_value:.4f}")
                model_call_params["clip_image"] = rag_reference_image 
                model_call_params["ipadapter_scale"] = rag_scale_value
            else:
                logger.warning("RAG requested but could not retrieve necessary inputs. Proceeding without RAG.")
        
        elif use_rag and not is_sd3_ipadapter:
             logger.warning(f"RAG is only implemented for the custom SD3 IP-Adapter. The current model '{pipeline_class_name}' does not support it. Proceeding without RAG.")

        # make sure that sks emoji is in the prompt for ft models
        if "sks emoji" not in model_call_params["prompt"] and not is_sd3_ipadapter:
             # Check if it's a fine-tuned model by checking if the model_id is a path
             model_id = getattr(model_manager, '_model_id', '')
             if model_id and Path(model_manager.get_available_models().get(model_id, {}).get('path', '')).is_dir():
                  logger.info("Appending 'sks emoji' to prompt for fine-tuned model.")
                  model_call_params["prompt"] = f"{model_call_params['prompt']} sks emoji"

        # make sure in inference/eval mode
        if hasattr(model_pipeline, "eval"):
            model_pipeline.eval()

        # get the list of images from the pipeline output
        pipeline_output = model_pipeline(**model_call_params)
        generated_images_list = pipeline_output.images
        
        # make sure that we got at least one image (should only be one by default)
        if not generated_images_list or not isinstance(generated_images_list, list) or not generated_images_list[0]:
            logger.error("No images were generated by the pipeline.")
            return {"status": "error", "error": "No images were generated"}
        
        # save the first image
        image_to_save = generated_images_list[0]
        image_path = save_image(final_prompt, image_to_save, output_path)
        logger.info(f"Emoji generated successfully: {image_path}")
        return {"status": "success", "image_path": str(image_path)}
    
    except Exception as e:
        logger.error(f"Error during emoji generation: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e)}



def save_image(prompt, image, output_path: Optional[str] = None) -> Path:
    """Save the generated image to disk and return the path."""
    base_path = Path(output_path or "generated_emojis")
    base_path.mkdir(parents=True, exist_ok=True)
    
    image_name = "".join(c if c.isalnum() else "_" for c in prompt)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{image_name}_{timestamp}.png"
    image_path = base_path / file_name
    
    image.save(image_path)
    return image_path
