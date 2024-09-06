# llama3-vasp

To fine-tune the llama-vasp, run the following command:

```bash
accelerate launch --mixed_precision fp16 -m llama_vasp train
```

To do inference, run the following command:

```bash
python -m llama_vasp inference --input "Your crystal structure description here"
```