import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
import locale
import json
import sys

import pdb
from transformers import pipeline
import torch
import pprint
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json

# Open and read the JSON file
with open('token.json', 'r') as file:
    data = json.load(file)

access_token = data["access_token"]
model_name = "meta-llama/Llama-3.2-3B-Instruct"

model_singleton = {}

def get_reader_model():
    if 'reader_model' not in model_singleton:
        READER_MODEL_NAME =model_name
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        model_singleton['reader_model'] = AutoModelForCausalLM.from_pretrained(
            READER_MODEL_NAME,
            token=access_token,
            device_map={"": 0},
            quantization_config=bnb_config,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    return model_singleton['reader_model']

def get_tokenizer():
    if 'tokenizer' not in model_singleton:
        READER_MODEL_NAME = model_name
        model_singleton['tokenizer'] = AutoTokenizer.from_pretrained(READER_MODEL_NAME, token=access_token,device_map="auto")
    return model_singleton['tokenizer']