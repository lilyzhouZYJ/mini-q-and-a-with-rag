# Mini Q&A with RAG

This repo includes two implementations of a tiny Q&A application using RAG.

1. **Raw RAG implementation** (`simple_rag/`) - a lightweight implementation built from scratch
2. **RAG using LangChain** (`rag_app/`) - a more production-ready approach using LangChain

Both implementations include:
- loading and chunking text content
- creating vector embeddings and storing them in vector store
- querying the content using natural language questions
- producing AI-generated answers based on the retrieved context

Currently, the raw RAG implementation supports retrieving text files, while the LangChain implementation supports retrieving webpages.

For more details on each implementation, see below documentations:
- Raw RAG: [`simple_rag/README.md`](simple_rag/README.md)
- RAG with LangChain: [`rag_app/README.md`](rag_app/README.md)

## Quick Start

### (1) Raw RAG Implementation

```bash
cd simple_rag
pip install -r requirements.txt

# Set up environment variables
# Create a .env file with OPENAI_API_KEY=your_key_here

# Run the application with a text file
python3 q_and_a_app.py --path test.txt
```

### (2) LangChain Implementation

```bash
cd rag_app
pip install -r requirements.txt

# Set up environment variables
# Create a .env file with OPENAI_API_KEY=your_key_here

# Run with a single URL
python main.py --url "https://example.com/article"

# Or run with multiple URLs from a file
python main.py --urls-file sample_urls.txt
```
