# from pathlib import Path
# import json
# import torch
# from diffusers import StableDiffusionPipeline
# from typing import Optional, Dict
# from emoji_gen.config import (
#     DEVICE, DTYPE, MODEL_ID_MAP, MODEL_LIST_PATH,
#     get_model_path, get_available_models
# )

# class ModelCache:
#     def __init__(self):
#         self._model = None  
#         self._model_id = None
#         self.MODELS = {}
        
#         # load saved model list (if any)
#         self._load_model_list()

#     def _load_model_list(self):
#         if MODEL_LIST_PATH.exists():
#             with open(MODEL_LIST_PATH) as f:
#                 data = json.load(f)
#                 self.MODELS.update(data.get("models", {}))

#     def register_model(self, model_name: str, model_path: str):
#         # register the model in the cache
#         self.MODELS[model_name] = model_path
#         self._save_model_list()

#     def _save_model_list(self):
#         with open(MODEL_LIST_PATH, "w") as f:
#             json.dump({"models": self.MODELS}, f, indent=2)

#     def list_models(self) -> Dict[str, Dict]:
#         return get_available_models()
    
#     # get a model from the cache, loading it if needed
#     def get_model(self, model_name: str) -> Optional[StableDiffusionPipeline]:
#         # if already loaded, do not reload it 
#         if model_name == self._model_id:
#             return self._model
        
#         # not loaded, load the new model
#         try:
#             model_path = get_model_path(model_name)
#             print(f"Loading model {model_name} from {model_path}")

#             # clear the memory
#             if self._model is not None:
#                 del self._model 
#                 torch.cuda.empty_cache()

#             self._model = StableDiffusionPipeline.from_pretrained(
#                 model_path,
#                 torch_dtype=DTYPE
#             ).to(DEVICE)

#             self._model_id = model_name
#             return self._model

#         except Exception as e:
#             print(f"Error loading model {model_name}: {e}")
#             return None
    
#     def get_current_model_id(self) -> Optional[str]:
#         return self._model_id

#     def cleanup(self):
#         if self._model is not None:
#             del self._model
#             torch.cuda.empty_cache()
#         self._model = None
#         self._model_id = None

# # create a global model cache instance
# model_cache = ModelCache() 