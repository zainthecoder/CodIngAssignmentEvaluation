import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
import locale
import json
import sys

import pdb
import torch
import pprint
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json

# Open and read the JSON file
with open('token.json', 'r') as file:
    data = json.load(file)

access_token = data["access_token"]
model_names = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct"
]

model_singleton = {}

def get_multiple_models():
    if 'reader_models' not in model_singleton:
        reader_models={}
        for model_name in model_names:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            
            reader_models[model_name] = {
                "model": AutoModelForCausalLM.from_pretrained(
                    model_name,
                    token=access_token,
                    device_map={"": 0},
                    quantization_config=bnb_config,
                    torch_dtype="auto",
                    trust_remote_code=True,
                    do_sample=False,
                    temperature=0.1,
                    top_p=0.1
                ),
                "tokenizer": AutoTokenizer.from_pretrained(model_name, token=access_token,device_map="auto")
            }
        model_singleton['reader_models'] = reader_models
    return model_singleton['reader_models']