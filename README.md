# Mini Q&A with RAG

This repo includes two implementations of a small Q&A application using RAG.

1. **`simple_rag`** - a lightweight implementation built from scratch
2. **`rag_app`** - a more complex implementation, includes features such as chunk post-processing, sparse encoding, etc.

Both implementations include:
- loading and chunking text content
- creating vector embeddings and storing them in vector store
- querying the content using natural language questions
- producing AI-generated answers based on the retrieved context

Currently, `simple_rag` supports retrieving single text files, while `rag_app` supports retrieving single files or directories.

For more details on each implementation, see below documentations:
- Raw RAG: [`simple_rag/README.md`](simple_rag/README.md)
- RAG with LangChain: [`rag_app/README.md`](rag_app/README.md)

## Quick Start

### (1) `simple_rag`

```bash
cd simple_rag
pip install -r requirements.txt

# Set up environment variables
# Create a .env file with OPENAI_API_KEY=your_key_here

# Run the application with a text file
python3 q_and_a_app.py --path test.txt
```

### (2) `rag_app`

```bash
cd rag_app
pip install -r requirements.txt

# Set up environment variables
# Create a .env file with OPENAI_API_KEY=your_key_here

# Step 1: Ingest documents into the vector store
python ingest_documents.py --file "document.txt"
# Or ingest all files from a directory
python ingest_documents.py --files-dir "./documents"

# Step 2: Ask questions about the ingested documents
python run_q_and_a.py
```
