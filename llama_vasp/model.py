from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, PeftModel
from trl import setup_chat_format
from .config import BASE_MODEL, BNB_CONFIG, ATTN_IMPLEMENTATION, TORCH_DTYPE, PEFT_CONFIG

def load_model_and_tokenizer(device_string):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    bnb_config = BitsAndBytesConfig(**BNB_CONFIG)
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map={'': device_string},
        attn_implementation=ATTN_IMPLEMENTATION,
        use_cache=False,
        torch_dtype=TORCH_DTYPE
    )

    model, tokenizer = setup_chat_format(model, tokenizer)
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def create_peft_model(model):
    peft_config = LoraConfig(**PEFT_CONFIG)
    model = PeftModel(model, peft_config)
    return model

def load_adapter_model(base_model, adapter_path):
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    return model