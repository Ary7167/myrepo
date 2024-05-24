# text2text-module
This is the implementation of the LLM based module that takes user queries and gives answers as the output.
The model architecture is based on GPT (Generative Pre-trained Transformer). It is a decoder only architecture with causal attention. The model size is about 124M parameters and has been chosent taking into 
consideration the compute constraints and inference latency.

# Training 
The training can be divided into the follwing steps:
1.) Pre-training (optional)
2.) Instruction-Fine Tuning 

Pre-trained weights and checkpoints of the GPT-2 model can be used for our architecture given the huge amount of corpus of text it has been trained on.
Instruction Fine tuning can be commenced once the data has been prepared (since it is supervised fine tuning).

# Notebooks for sample text generation 
To generate sample text the notebook link has been shared 
https://colab.research.google.com/drive/1wO_5Omp2rjddyoVjahDxPqIAAvkibhqp?usp=sharing

# Fine-tuning to generate relevant responses
