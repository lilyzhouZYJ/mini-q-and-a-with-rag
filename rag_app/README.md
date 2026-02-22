# Mini Q&A

The `rag_app` directory contains a RAG application that **ingests text files and answers questions about them**. The ingestion pipeline chunks source documents and performs a series of post-processing (such as noise removal, metadata generation) on the chunks before generating dense embeddings. The query process performs similarity search based on the dense embeddings.

For more detailed documentation, see [./docs](./docs/).

*todo:*
- add sparse embeddings
- improve query

## Getting started

### Requirements

```bash
cd rag_app
pip install -r requirements.txt
```

### Configuration

Create a `.env` file with the following variables:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (defaults shown)
MODEL_NAME=gpt-4o-mini
MODEL_PROVIDER=openai

# Optional LangSmith tracing
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=false
LANGSMITH_PROJECT=default
```

You can also run `cp .env.example .env` and fill it the config fields.

## Usage

The application is split into two separate scripts:

1. **`ingest_documents.py`** - Ingests documents into the vector store
2. **`run_q_and_a.py`** - Runs the interactive Q&A interface

### Step 1: Ingest Documents

First, ingest your documents into the vector store:

```bash
# Ingest a single text file
python ingest_documents.py --file "document.txt"

# Ingest all files from a directory
python ingest_documents.py --files-dir "./documents"

# Ingest without LLM-based transformation
python ingest_documents.py --file "document.txt" --no-transform
```

**Ingestion Options:**
- `--file PATH`: Ingest a single text file (.txt, .md)
- `--files-dir PATH`: Ingest all supported files from a directory (recursively scans subdirectories)
- `--no-transform`: Disable LLM-based transformation and enrichment

**Note**: When using `--files-dir`, the app will recursively scan the directory and all subdirectories for supported text files (.txt, .md).

### Step 2: Ask Questions

After ingesting documents, run the Q&A interface:

```bash
python run_q_and_a.py
```

The script runs in interactive mode - you can ask multiple questions. Type 'quit' or 'exit' to end the session.

## Examples

### Complete Workflow

```bash
# 1. Ingest documents
python ingest_documents.py --files-dir "./documents"

# 2. Ask questions interactively
python run_q_and_a.py
```

### Ingesting Multiple Documents

```bash
# Ingest all .txt and .md files in the directory and subdirectories
python ingest_documents.py --files-dir "./documents"
```

### Interactive Q&A Session

```bash
python run_q_and_a.py
# Then type your questions at the prompt
# Type 'quit' or 'exit' to end the session
```

## LangSmith Tracing

The app supports LangSmith tracing for monitoring and debugging your RAG pipeline:

1. **Enable Tracing**: Set `LANGSMITH_TRACING=true` in your `.env` file
2. **Get API Key**: Sign up at [LangSmith](https://smith.langchain.com/) and get your API key
3. **Set Project**: Configure `LANGSMITH_PROJECT=mini-q-and-a` (or your preferred project name)
4. **View Traces**: All LLM calls, retrievals, and graph executions will be logged to your LangSmith dashboard

This is especially useful for:
- Debugging retrieval quality
- Monitoring LLM performance
- Analyzing conversation flows
- Optimizing prompt engineering

## How It Works

1. **Content Loading**: The app loads text files (.txt, .md) from the filesystem
2. **Chunking**: Content is split into manageable chunks using RecursiveCharacterTextSplitter
3. **Vector Store**: Chunks are embedded and stored in a persistent Chroma vector store
4. **RAG Pipeline**: Uses LangGraph to implement a retrieval-augmented generation pipeline
5. **Question Answering**: Retrieves relevant chunks and generates answers using the configured LLM
