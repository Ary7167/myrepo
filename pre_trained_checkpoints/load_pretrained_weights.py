# load pre-trained weights of the GPT-2 model in place of pretraining text
# Pre-training on unlabeled data might require significant amount of compute (atleast A100 GPUs). In such cases we adopt to the pre-trained weights of the GPT-2
# available open-source from the Open-AI repo.
import numpy as np
import torch
import torch.nn as nn
from .gpt_donwload_weights import download_and_load_gpt2


# import the weights using the donwloads script

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

#  load the weights into the current model instance 
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))
    
def load_weights_into_llm(llm, params):
    llm.pos_emb.weight = assign(llm.pos_emb.weight, params['wpe'])
    llm.tok_emb.weight = assign(llm.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        llm.trf_blocks[b].att.W_query.weight = assign(
            llm.trf_blocks[b].att.W_query.weight, q_w.T)
        llm.trf_blocks[b].att.W_key.weight = assign(
            llm.trf_blocks[b].att.W_key.weight, k_w.T)
        llm.trf_blocks[b].att.W_value.weight = assign(
            llm.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        llm.trf_blocks[b].att.W_query.bias = assign(
            llm.trf_blocks[b].att.W_query.bias, q_b)
        llm.trf_blocks[b].att.W_key.bias = assign(
            llm.trf_blocks[b].att.W_key.bias, k_b)
        llm.trf_blocks[b].att.W_value.bias = assign(
            llm.trf_blocks[b].att.W_value.bias, v_b)

        llm.trf_blocks[b].att.out_proj.weight = assign(
            llm.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        llm.trf_blocks[b].att.out_proj.bias = assign(
            llm.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        llm.trf_blocks[b].ff.layers[0].weight = assign(
            llm.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        llm.trf_blocks[b].ff.layers[0].bias = assign(
            llm.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        llm.trf_blocks[b].ff.layers[2].weight = assign(
            llm.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        llm.trf_blocks[b].ff.layers[2].bias = assign(
            llm.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        llm.trf_blocks[b].norm1.scale = assign(
            llm.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        llm.trf_blocks[b].norm1.shift = assign(
            llm.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        llm.trf_blocks[b].norm2.scale = assign(
            llm.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        llm.trf_blocks[b].norm2.shift = assign(
            llm.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    llm.final_norm.scale = assign(llm.final_norm.scale, params["g"])
    llmt.final_norm.shift = assign(llm.final_norm.shift, params["b"])
    llm.out_head.weight = assign(llm.out_head.weight, params["wte"])
