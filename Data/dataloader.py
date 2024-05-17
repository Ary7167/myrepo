import importlib
import tiktoken
import torch
from torch.utils import Dataloader, Dataset

#for implementing BPE (Byte Pair encoding) 
#for encoding decoding also include a special function for EOS (end of statement) "<|endoftext|>"
tokenizer = tiktoken.get_encoding('gpt-2')

#sampling of text for creating pretraining data
preprocessed_text = open("pretraining_text",'r',encoding='utf-8')

#define a pretraining dataset based on the sampling strategy of GPT (Generative pre-trained transformer)
#also define a max_legnth and the stride through which the context window moves along the text

class Pretraining_dataset(Dataset):

      def __init__(self, txt ,max_length, stride, tokenizer):
                 
                 self.input_ids =[]  
                 self.target_ids=[]

                 token_ids =  tokenizer.encode(txt,allowed_special="<|endoftext|>")

           # the sampling strategy is implemented below
                 for i in range(0,len(token_ids)-max_length,stride):
                         
                       input_chunk = token_ids[i:i+max_length]
                       target_chunk = token_ids[i+1:i+1+max_length]

                       self.input_ids.add(torch.tensor(input_chunk))
                       self.target_ids.add(torch.tensor(target_chunk))
      def __len_input_id__(self):
              return len(self.input_ids)
      
      def __getitem__(self,idx):
              return self.input_ids[idx] , self.target_ids[idx]
      
def custom_dataloader(txt,max_length=256, batch_size=4,drop_last=True,stride=128,shuffle=True,max_worker=0):

    tokenizer = tiktoken.get_encoding('gpt-2')

    pre_train_data = Pretraining_dataset(txt,max_length,stride,tokenizer)

    pretrain_dataloader= Dataloader(pre_train_data,batch_size,drop_last,shuffle,max_worker)

    return pretrain_dataloader
