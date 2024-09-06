from .data import extract_input
from transformers import pipeline
from .config import MAX_NEW_TOKENS, DO_SAMPLE


def inference(model, tokenizer, input_text):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map='auto'
    )

    input_for_model = extract_input(input_text)
    
    result = pipe(input_for_model, do_sample=DO_SAMPLE, max_new_tokens=MAX_NEW_TOKENS)
    
    generated_text = result[0]['generated_text']
    assistant_response = generated_text.split('<|im_start|>assistant')[-1].strip()
    
    return assistant_response