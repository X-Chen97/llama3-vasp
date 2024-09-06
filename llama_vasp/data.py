import re
from datasets import load_dataset
from .config import DATASET

# round the positional coordinates in a POSCAR file to a given number of decimal places
# help to save the context space
def round_poscar_numbers(text, decimal_places=2):
    def round_number(match):
        return f"{float(match.group()):.{decimal_places}f}"

    pattern = r'\d+\.\d+'
    rounded_text = re.sub(pattern, round_number, text)
    rounded_text = '\n'.join(' '.join(line.split()) for line in rounded_text.splitlines())
    return rounded_text

def format_chat_template(row, tokenizer):
    rounded_out = round_poscar_numbers(row["output"], decimal_places=3)
    row_json = [
        {"role": "system", "content": row["instruction"]},
        {"role": "user", "content": row["input"]},
        {"role": "assistant", "content": rounded_out}
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

def load_and_preprocess_dataset(tokenizer):
    dataset = load_dataset(DATASET)
    return dataset.map(
        lambda row: format_chat_template(row, tokenizer),
        num_proc=4,
    )
