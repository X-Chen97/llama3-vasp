import gc
import os
if not os.path.exists('./.hf_cache'):
    os.makedirs('./.hf_cache')
os.environ["HF_HOME"] = './.hf_cache'

import torch
torch.cuda.empty_cache()

import wandb
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import SFTConfig, SFTTrainer, setup_chat_format
import re

from accelerate import PartialState
device_string = PartialState().process_index

os.environ["WANDB_NOTEBOOK_NAME"] = './test-llama.ipynb'
os.environ["WANDB_PROJECT"] = 'llama-dft'

attn_implementation = "eager"
torch_dtype = torch.float16
base_model = "meta-llama/Meta-Llama-3-8B"
new_model = "Qlora-Llama-3-8B"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# LoRA config
peft_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map={'':device_string},
    attn_implementation=attn_implementation,
    use_cache=False,
    torch_dtype=torch_dtype
)

model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)

dataset = load_dataset("MaterialsAI/robocr_poscar")

def round_poscar_numbers(text, decimal_places=2):
    def round_number(match):
        return f"{float(match.group()):.{decimal_places}f}"

    # Pattern to match floating point numbers
    pattern = r'\d+\.\d+'

    # Round numbers in the text
    rounded_text = re.sub(pattern, round_number, text)

    
    # Remove leading/trailing whitespace from each line
    rounded_text = '\n'.join(' '.join(line.split()) for line in rounded_text.splitlines())

    return rounded_text

def format_chat_template(row):
    rounded_out = round_poscar_numbers(row["output"], decimal_places=3)
    row_json = [{"role": "system", "content": row["instruction"]}, {"role": "user", "content": row["input"]},
               {"role": "assistant", "content": rounded_out}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc=4,
)

training_arguments = SFTConfig(
    output_dir='output/llama-4gpu',
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=12,
    optim="paged_adamw_8bit",
    num_train_epochs=5,
    eval_strategy="steps",
    eval_steps=0.1,
    eval_accumulation_steps=2,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    save_strategy='epoch',
    save_total_limit=None,
    load_best_model_at_end=False,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    group_by_length=True,
    report_to="wandb",
    run_name='finetune-4gpu',
    dataset_text_field='text',
    max_seq_length=2400,
    fp16=True,
    gradient_checkpointing = True,
    gradient_checkpointing_kwargs = {"use_reentrant": False}, #must be false for DDP
    ddp_find_unused_parameters=False # if use DDP is false, otherwise true
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model('dft-llama-3b-lora-4gpu')
