from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Create a simple document
documents = ["This is a test document.", "Another document for testing."]

# Create embeddings
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")

# Initialize ChromaDB
vectorstore = Chroma.from_texts(documents, embedding=embeddings, persist_directory="./chroma_db")

# Perform a similarity search
results = vectorstore.similarity_search("test")
print(results)