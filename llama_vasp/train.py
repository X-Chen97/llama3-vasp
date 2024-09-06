from transformers import TrainingArguments
from trl import SFTTrainer
from .config import TRAINING_ARGS

def setup_trainer(model, tokenizer, train_dataset, eval_dataset):
    training_arguments = TrainingArguments(**TRAINING_ARGS)

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    return trainer

def train_and_save_model(trainer, output_path):
    trainer.train()
    trainer.save_model(output_path)