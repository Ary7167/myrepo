import vectordb
from vectordb import Memory
import argparse

#create a vector store to keep all the relevant documents for usage
memory =Memory(chunking_strategy={'mode':'sliding_window', 'window_size': 128, 'overlap': 16},embeddings=best)

#extract the text from the documents (document files can be of the type pdf,txt,https)
with open(file_path,"r") as file:
     content = file.read()

# create arguments for the query and the filepath for uploading the documents
query= " "
texts.append(content)
metadat_list.append(file_path)

# save the text 
memory.save(memory.save(texts, metadata_list))

# based on the query recieved
results= memory.search(query,top_n=1,batch_results="flatten")

# use the result as context to generate the answer
