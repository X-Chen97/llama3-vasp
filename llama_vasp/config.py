import os
import torch

# Environment setup
HF_CACHE_DIR = './.hf_cache'
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["WANDB_NOTEBOOK_NAME"] = './test-llama.ipynb'
os.environ["WANDB_PROJECT"] = 'llama-dft'

# dataset configuration
DATASET = "MaterialsAI/robocr_poscar"

# Model configuration
ATTN_IMPLEMENTATION = "eager"
TORCH_DTYPE = torch.float16
BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
ADAPTER_MODEL = "dft-llama-3b-lora-4gpu" # your fine-tuned model name

# Quantization configuration
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": TORCH_DTYPE,
    "bnb_4bit_use_double_quant": True,
}

# LoRA configuration
PEFT_CONFIG = {
    "r": 64,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
}

# Training configuration
TRAINING_ARGS = {
    "output_dir": 'output',
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 12,
    "optim": "paged_adamw_8bit",
    "num_train_epochs": 5,
    "eval_strategy": "steps",
    "eval_steps": 0.1,
    "eval_accumulation_steps": 2,
    "logging_steps": 1,
    "warmup_steps": 10,
    "logging_strategy": "steps",
    "save_strategy": 'epoch',
    "save_total_limit": None,
    "load_best_model_at_end": False,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "group_by_length": True,
    "report_to": "wandb",
    "run_name": 'finetune-4gpu',
    "dataset_text_field": 'text',
    "max_seq_length": 2400,
    "fp16": True,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "ddp_find_unused_parameters": False
}

# Inference configuration
MAX_NEW_TOKENS = 500
DO_SAMPLE = True
