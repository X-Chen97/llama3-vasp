import os
import gc
from .config import HF_CACHE_DIR
import torch
import re
from accelerate import PartialState

def setup_environment():
    if not os.path.exists(HF_CACHE_DIR):
        os.makedirs(HF_CACHE_DIR)
    
    gc.collect()
    torch.cuda.empty_cache()

def get_device_string():
    return PartialState().process_index

# extract the input text from the data set that includes the assistant response
def extract_input(text):
    match = re.search(r'(.*<\|im_end\|>)\s*<\|im_start\|>assistant', text, re.DOTALL)
    if match:
        return match.group(1).strip() + '\n<|im_start|>assistant'
    return None