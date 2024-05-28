# script for instruction tuning LLM for answering user queries 
import wandb
import torch
import torch.nn as nn

# we import our custom instruction tuning dataset and fine-tune with the help of the wandb server for convenience

file_path= 'myrepo/Dataset/instruction_tuning.json'
with open(file_path,'r') as f:
     data= json.load(f)

with wandb.init(project='instruction-tuning'):
     at= wandb.artifact(
                 name="instruction_tuning"
                 type="dataset"
                 description="instruction tuning dataset for question answering ability "
                 metadata= {"https://github.com/Ary7167/myrepo/Dataset"}
            )
     at.add_file("instruction_tuning.json")

     
