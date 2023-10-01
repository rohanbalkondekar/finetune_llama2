import re
import math
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# Path to saved model
model_path = "generate-json-agent-32e-llama2-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    trust_remote_code=True,
    device_map="auto",
)

def evaluate_json(json_data):
    function_name = json_data.get("function_name")
    parameter_1 = float(json_data.get("parameter_1", 0))
    parameter_2 = float(json_data.get("parameter_2", 0))

    if function_name == "add":
        result = parameter_1 + parameter_2
    elif function_name == "subtract":
        result = parameter_1 - parameter_2
    elif function_name == "multiply":
        result = parameter_1 * parameter_2
    elif function_name == "divide":
        result = parameter_1 / parameter_2
    elif function_name == "square_root":
        result = math.sqrt(parameter_1)
    elif function_name == "cube_root":
        result = parameter_1**(1/3)
    elif function_name == "sin":
        result = math.sin(math.radians(parameter_1))
    elif function_name == "cos":
        result = math.cos(math.radians(parameter_1))
    elif function_name == "tan":
        result = math.tan(math.radians(parameter_1))
    elif function_name == "log_base_2":
        result = math.log2(parameter_1)
    elif function_name == "ln":
        result = math.log(parameter_1)
    elif function_name == "power":
        result = parameter_1**parameter_2
    else:
        result = None

    return result



while True:
    prompt = input("Ask Question: ")
    formatted_prompt = f"<s>[INST] Reply with json for the following question: {prompt} [/INST] Here is your generated JSON: "

    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to("cuda")
    gen_tokens = model.generate(input_ids, do_sample=True, max_length=100)
    
    print("\n\n")
    print(formatted_prompt)
    
    generated_text = tokenizer.batch_decode(gen_tokens)[0]
    
    print("\n\n")
    print("*"*20)
    print("\033[94m" + f"\n\n {prompt} \n" + "\033[0m")
    print("\n\n")
    print("\033[90m" + generated_text + "\033[0m")
    print("\n")

    json_match = re.search(r'json\s*({.+?})\s*', generated_text, re.DOTALL)
    if json_match:
        json_string = json_match.group(1)
        try:
            json_data = json.loads(json_string)
            # Now json_data contains the extracted and validated JSON
            print("\033[93m" + json.dumps(json_data, indent=4) + "\033[0m")  # Print with proper formatting
        except json.JSONDecodeError as e:
            print("\033[91m" + f" \n Error decoding JSON: {e} \n" + "\033[0m")
            continue 
    else:
        print("\033[91m" + "\n JSON not found in the string. \n" + "\033[0m")
        continue 


    result = evaluate_json(json_data)
    print(f"\n\n \033[92mThe result is: {result} \033[0m \n\n")

    print("*"*20)
    print("\n\n")