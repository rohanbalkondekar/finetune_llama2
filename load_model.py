import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel

# Path to saved model
model_path = "generate-json-agent-32e-llama2-chat-hf"

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("generate-json-agent-32e-llama2-chat-hf")


# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    load_in_4bit=True,    # changing this to load_in_8bit=True works on smaller models
    trust_remote_code=True,
    device_map="auto",    # finds GPU
)

while True:
    # Text generation
    prompt = input("Enter Prompt: ")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    # Generate text
    gen_tokens = model.generate(input_ids, do_sample=True, max_length=2000)
    generated_text = tokenizer.batch_decode(gen_tokens)[0]
    print("\n Generated Text: \n")
    print(generated_text)
    print("\n----------------------------------\n")
    