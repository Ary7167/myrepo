# We would be implementing the same methodology as developed for the original GPT
# the training parameters depends on the GPU compute available 
# this is a pre-training process involving training the model on a initial dataset to generate text
# Instruction fine tuning script would be developed later for downstream question answering capability 
# pre-training weights can be implemented but we would like to train the model specific to our data
import torch
import torch.nn as nn
from model import LLM

#implementing the text encoder and decoder to convert tokens to idx
def text_to_token_ids(text,tokenizer):
    encoded_ids= tokenizer.encode(text,allowed_special="<|endoftext|>")
    encoded_tensor= torch.tensor(encoded_ids).unsqueeze(0)

    return encoded_tensor

def token_ids_to_text(encoded_tensor,tokenizer):
    flat= encoded_tensor.squeeze(0)
    decoded_text= tokenizer.decode(flat.to_list())

    return decoded_text

def generate_text_sample(LLM, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
      for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = LLM(idx_cond)
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

      return idx