import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set up Grok API key
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"

# Configure LlamaIndex to use Grok as the LLM
llm = Groq(model="llama3-70b-8192", api_key=os.environ["GROQ_API_KEY"])
Settings.llm = llm

# Configure HuggingFace embeddings
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model

# Load documents from a directory (e.g., a PDF file)
documents = SimpleDirectoryReader("data").load_data()

# Create a vector index for retrieval using HuggingFace embeddings
index = VectorStoreIndex.from_documents(documents)

# Set up the query engine for conversational retrieval
query_engine = index.as_query_engine()

# Example chatbot interaction loop
def chatbot():
    print("Welcome to the RAG Chatbot with HuggingFace Embeddings! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        try:
            response = query_engine.query(user_input)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Create a data directory and place a sample PDF inside
    os.makedirs("data", exist_ok=True)
    chatbot()