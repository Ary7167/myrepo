# script for instruction tuning LLM for answering user queries 
import wandb
import accelerate
import trl 
import Datasets
import os
import torch
from dataset import load_dataset
from trl import SFTTrainer
 
# we import our custom instruction tuning dataset and fine-tune with the help of the wandb server for convenience
batch_size = 16
num_workers = os.cpu_count()
max_steps = 3000
bf16 = False
fp16 = True
gradient_accumulation_steps = 2
context_length = 256
logging_steps = 500
save_steps = 500
learning_rate = 0.0001
model_name = 't'
out_dir = 'outputs/gpt2_alpaca_preprocess_fn'

#load the instruction tuning dataset
instruction_tune_data= Dataloader('instruction_tune.json')

# preprocess and tokenize the dataset using the byte pair tokenizer
     
