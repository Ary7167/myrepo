import torch
import torch.nn as nn
from MHA import MultiHeadAttention
# This would be the baseline working implementation of a lightweight version of an LLM which can be used for specific downstream tasks.
# The model architecture is built based on the work behind the GPT model. Involves a decoder only architecture with causal attention.
# The model is meant to be autoregressive for generation purposes, initially the model would be pre-trained on a large dataset.
# For fine-tuning we would be using the instruction based approach for question answering capability.

#configuration of model architecture
#the number of attention heads and the number of FC layers has to be re-assigned based on the compute constraints and inference latency.

LLM_config= {"vocab_size": 50257,    # Vocabulary size 
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate (drop_out for regularization of the model)
    "qkv_bias": False       # Query-Key-Value bias
    }

#implementing layer norm for ease in computation 

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

#to utilize state of the art activations for better peformance we implement GELU 

class GELU_activation(nn.module):
      def __init__(self):
           super().__init__()

      def forward(self,x):
           return 0.5*x*(1+torch.tanh(torch.sqrt(2.0/torch.pi))*(x+0.044714*torch.pow(x,3)))

#implementing the feed forward module to be used in the transformer module

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU_activation(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
    
# combining all the sub-modules into a single transformer block (decoder only architecture)
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
