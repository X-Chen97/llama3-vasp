import argparse
from .utils import setup_environment
from .model import load_base_model_and_tokenizer, create_peft_model, load_adapter_model
from .data import load_and_preprocess_dataset
from .train import setup_trainer, train_and_save_model
from .inference import inference
from .config import ADAPTER_MODEL

def main():
    parser = argparse.ArgumentParser(description="Run LLAMA VASP training or inference")
    parser.add_argument('mode', choices=['train', 'inference'], help='Mode to run: train or inference')
    args = parser.parse_args()

    setup_environment()

    if args.mode == 'train':
        run_training()
    elif args.mode == 'inference':
        run_inference()

def run_training():
    model, tokenizer = load_base_model_and_tokenizer()
    model = create_peft_model(model)
    dataset = load_and_preprocess_dataset(tokenizer)
    
    trainer = setup_trainer(model, tokenizer, dataset["train"], dataset["validation"])
    train_and_save_model(trainer, ADAPTER_MODEL)

def run_inference(input_text):
    model, tokenizer = load_base_model_and_tokenizer()
    model = load_adapter_model(model, ADAPTER_MODEL)
    response = inference(model, tokenizer, input_text)
    print("Model response:", response)

if __name__ == "__main__":
    main()