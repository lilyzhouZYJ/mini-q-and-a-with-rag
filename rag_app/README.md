# Mini Q&A using LangChain

The `rag_app` directory contains a LangChain-based implementation of a Q&A application. The application supports loading text files (single files or entire directories) and answering questions about them using RAG.

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

### Basic Usage

```bash
# Single text file with interactive mode
python main.py --file "document.txt"

# Multiple files from directory
python main.py --files-dir "./documents"

# Single question mode
python main.py --file "document.txt" --question "What is this about?"
```

### Command Line Options

- `--file PATH`: Load content from a single text file (.txt, .md)
- `--files-dir PATH`: Load all supported files from a directory (recursively scans subdirectories)
- `--interactive`: Run in interactive mode (default)
- `--question TEXT`: Ask a single question and exit
- `--no-transform`: Disable LLM-based transformation and enrichment
- `--help`: Show help message

**Note**: When using `--files-dir`, the app will recursively scan the directory and all subdirectories for supported text files (.txt, .md).

## Examples

### Interactive Mode with Single File
```bash
python main.py --file "document.txt"
```

### Batch Processing from Directory
```bash
# Process all .txt and .md files in the directory and subdirectories
python main.py --files-dir "./documents"
```

### Single Question Mode
```bash
python main.py --file "document.txt" --question "What are the main points?"
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
