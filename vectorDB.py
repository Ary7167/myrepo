from vectordb import Memory

# Store all the content required to generate the text in the memory

memory= Memory(chunking_strategy={"mode":},embeddings={"best"})

# a function to load and chunk the relevant documents and store in the memory instance 
def load_data(filepath,memory):
    memory.save()


#define a function to generate an answer
def generate_answer(memory,model,query):
    output = memory.search(query,top_n=1)
  
   # augment the answer with the LLM model 

