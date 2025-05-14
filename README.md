# RAG Chatbot with LlamaIndex, Groq API, and HuggingFace Embeddings

This project implements a Retrieval-Augmented Generation (RAG) chatbot using [LlamaIndex](https://www.llamaindex.ai/), [groq API](https://x.ai/api), and [HuggingFace embeddings](https://huggingface.co/). The chatbot indexes documents (e.g., PDFs) and answers questions based on the content, leveraging groq for fast response generation and HuggingFace for efficient document embeddings.

## Features
- **Document Indexing**: Indexes PDF documents using LlamaIndex with HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
- **Context-Aware Responses**: Combines document retrieval with groq's LLM for accurate answers.
- **Simple CLI Interface**: Interactive command-line interface for chatting with the bot.
- **Extensible**: Easily adaptable for other document types or embedding models.

## Prerequisites
- Python 3.8 or higher
- A valid [groq API key](https://x.ai/api)
- A PDF document for indexing (placed in the `data` folder)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Vamsi-2203/Build-a-RAG-Chatbot-with-LlamaIndex-groq-API-and-HuggingFace-Embeddings.git
   cd rag-chatbot
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**:
   Install the required packages using the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional: GPU Support**:
   For faster embeddings with a GPU, install a CUDA-compatible PyTorch (if not included in `requirements.txt`):
   ```bash
   pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
   ```

## Setup

1. **Obtain a groq API Key**:
   - Sign up at [xAI API](https://x.ai/api) to get your groq API key.
   - Replace `"your_groq_api_key_here"` in `rag_chatbot_hf.py` with your key:
     ```python
     os.environ["GROQ_API_KEY"] = "your_actual_api_key"
     ```

2. **Prepare Documents**:
   - Create a `data` folder in the project directory:
     ```bash
     mkdir data
     ```
   - Place a PDF document in the `data` folder (e.g., `sample.pdf`).

## Usage

1. **Run the Chatbot**:
   Execute the script to start the chatbot:
   ```bash
   python chatbot.py
   ```

2. **Interact with the Chatbot**:
   - The chatbot will prompt for input.
   - Type questions related to the indexed document.
   - Type `exit` to quit.

   Example:
   ```
   Welcome to the RAG Chatbot with HuggingFace Embeddings! Type 'exit' to quit.
   You: What is the main topic of the document?
   Bot: The document discusses [summary based on content].
   You: exit
   Goodbye!
   ```

## Project Structure
```
rag-chatbot/
├── data/               # Folder for input documents (e.g., PDFs)
├── chatbot.py          # Main chatbot script
├── requirements.txt    # Dependency list
├── README.md           # Project documentation
└── venv/               # Virtual environment (if created)
```

## Customization
- **Change Embedding Model**: Modify the `model_name` in `HuggingFaceEmbedding` (e.g., `sentence-transformers/all-mpnet-base-v2` for higher quality).
- **Switch LLM Model**: Update the `model` in `Groq` (e.g., `mixtral-8x7b-32768` for a different groq model).
- **Add a Web UI**: Integrate with [Streamlit](https://streamlit.io/) for a browser-based interface:
  ```bash
  pip install streamlit
  ```

## Troubleshooting
- **API Key Errors**: Ensure the groq API key is valid and has sufficient quota.
- **Memory Issues**: Use a smaller embedding model or reduce document size.
- **Package Conflicts**: Verify `requirements.txt` versions and reinstall if needed:
  ```bash
  pip install -r requirements.txt --force-reinstall
  ```
- **PDF Parsing**: Ensure the PDF is text-based (not scanned) and not corrupted.
