import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, CLIPModel, CLIPProcessor
import gc
import logging

logger = logging.getLogger(__name__)

# make these as global so we can access them from elsewhere
_llm_model = None
_llm_tokenizer = None
_clip_model = None
_clip_processor = None

_device = None
_llm_dtype = None
_clip_dtype = None

# TODO import these from config.py
DEFAULT_LLM_MODEL_ID = "google/gemma-2b-it"
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

def _initialize_device_and_dtypes():
    global _device, _llm_dtype, _clip_dtype
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"

        _llm_dtype = torch.float16 if _device == "cuda" else torch.float32
        _clip_dtype = torch.float16 if _device == "cuda" else torch.float32
        logger.info(f"Auxiliary models loaded on: {_device}")

def get_llm_pipeline():
    global _llm_model, _llm_tokenizer
    _initialize_device_and_dtypes() 

    # only load model if not already loaded
    if _llm_model is None:
        try:
            from transformers import AutoModelForCausalLM

            _llm_model = AutoModelForCausalLM.from_pretrained(
                DEFAULT_LLM_MODEL_ID,
                torch_dtype=_llm_dtype,
                low_cpu_mem_usage=True
            ).to(_device)
            _llm_model.eval() ## like inference mode

            _llm_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_LLM_MODEL_ID)

            # Gemma uses a specific chat template format that needs to be set.
            if "gemma" in DEFAULT_LLM_MODEL_ID and _llm_tokenizer.chat_template is None:
                logger.info("Setting Gemma chat template on tokenizer.")
                gemma_template = (
                    "{% for message in messages %}"
                    "{% if message['role'] == 'user' %}"
                    "{{ '<start_of_turn>user\\n' + message['content'] + '<end_of_turn>\\n' }}"
                    "{% elif message['role'] == 'assistant' %}"
                    "{{ '<start_of_turn>model\\n' + message['content'] + '<end_of_turn>\\n' }}"
                    "{% elif message['role'] == 'system' %}"
                    "{{ message['content'] + '\\n' }}"
                    "{% endif %}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}"
                    "{{ '<start_of_turn>model\\n' }}"
                    "{% endif %}"
                )
                _llm_tokenizer.chat_template = gemma_template

            logger.info(f"LLM model '{DEFAULT_LLM_MODEL_ID}' loaded successfully on {_device} with dtype {_llm_dtype}.")
        except Exception as e:
            logger.error(f"Failed to load LLM model '{DEFAULT_LLM_MODEL_ID}': {e}", exc_info=True)
            print(f"Failed to load LLM model '{DEFAULT_LLM_MODEL_ID}': {e}")

            # set to none
            _llm_model = None
            _llm_tokenizer = None
            return None, None
    else:
        logger.debug(f"Already loaded LLM model: {_llm_model.name_or_path if _llm_model else 'N/A'}")
        
    return _llm_model, _llm_tokenizer


def get_clip_pipeline(model_id: str = None):
    global _clip_model, _clip_processor
    _initialize_device_and_dtypes() 

    if _clip_model is None:
        try:
            _clip_model = CLIPModel.from_pretrained(
                DEFAULT_CLIP_MODEL_ID,
                torch_dtype=_clip_dtype
            ).to(_device)
            _clip_model.eval() ## like inference mode
            _clip_processor = CLIPProcessor.from_pretrained(DEFAULT_CLIP_MODEL_ID) ## TODO use_fast=True (get working first)
            logger.info(f"CLIP model '{DEFAULT_CLIP_MODEL_ID}' loaded successfully on {_device} with dtype {_clip_dtype}.")
        except Exception as e:
            logger.error(f"Failed to load CLIP model '{DEFAULT_CLIP_MODEL_ID}': {e}", exc_info=True)
            print(f"Failed to load CLIP model '{DEFAULT_CLIP_MODEL_ID}': {e}")

            _clip_model = None
            _clip_processor = None
            return None, None
    else:
        logger.debug(f"Already loaded CLIP model: {_clip_model.name_or_path if _clip_model else 'N/A'}")

    return _clip_model, _clip_processor


def cleanup_aux_models(llm_only=False, clip_only=False):
    global _llm_model, _llm_tokenizer, _clip_model, _clip_processor



    if not llm_only and not clip_only: # Full cleanup
        models_to_clean = [(_llm_model, "_llm_model"), (_clip_model, "_clip_model")]
    elif llm_only:
        models_to_clean = [(_llm_model, "_llm_model")]
    elif clip_only:
        models_to_clean = [(_clip_model, "_clip_model")]
    else:
        return
        
    for model_obj, model_name_str in models_to_clean:
        if model_obj is not None:
            logger.info(f"Cleaning up {model_name_str}...")
            del model_obj
            if model_name_str == "_llm_model":
                del _llm_tokenizer
                _llm_model = None
                _llm_tokenizer = None
            elif model_name_str == "_clip_model":
                del _clip_processor
                _clip_model = None
                _clip_processor = None
            logger.info(f"{model_name_str} cleaned up.")

    gc.collect()
    if _device == "cuda":
        logger.info("Emptying CUDA cache for auxiliary models.")
        torch.cuda.empty_cache()
        # torch.cuda.synchronize() # Not strictly necessary for empty_cache usually


# for local testing with python -m emoji_gen.utils.aux_models
# Run this if you want to sanity check
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("--- Testing LLM Pipeline ---")
    llm_model, llm_tokenizer = get_llm_pipeline()
    if llm_model and llm_tokenizer:
        logger.info(f"LLM Loaded: {llm_model.name_or_path}")
        # Second call to test lazy loading
        get_llm_pipeline() 
    else:
        logger.error("LLM pipeline test failed.")

    logger.info("\n--- Testing CLIP Pipeline ---")
    clip_model, clip_processor = get_clip_pipeline()
    if clip_model and clip_processor:
        logger.info(f"CLIP Loaded: {clip_model.name_or_path}")
        # Second call to test lazy loading
        get_clip_pipeline()
    else:
        logger.error("CLIP pipeline test failed.")

    logger.info("\n--- Testing Cleanup ---")
    cleanup_aux_models()
    logger.info("Cleanup called. Attempting to get models again (should reload).")
    get_llm_pipeline()
    get_clip_pipeline()
    logger.info("Aux models test complete.") 