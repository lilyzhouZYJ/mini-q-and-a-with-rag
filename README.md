# Mini Q&A CLI Application

A command-line application that loads web content and allows you to ask questions about it using a RAG (Retrieval-Augmented Generation) system powered by LangChain and LangGraph.

## Features

- Load content from single URLs or multiple URLs from a text file
- Configurable AI model via environment variables
- Interactive question-answering mode
- Single question mode for automation
- Robust error handling for failed URL loads

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

## Configuration

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

## Usage

### Basic Usage

```bash
# Single URL with interactive mode
python main.py --url "https://example.com/article"

# Multiple URLs from file
python main.py --urls-file "urls.txt"

# Single question mode
python main.py --url "https://example.com/article" --question "What is this article about?"
```

### Command Line Options

- `--url URL`: Load content from a single URL
- `--urls-file PATH`: Load URLs from a text file (one URL per line)
- `--interactive`: Run in interactive mode (default)
- `--question TEXT`: Ask a single question and exit
- `--help`: Show help message

### URL File Format

Create a text file with one URL per line. Lines starting with `#` are treated as comments:

```
# Sample URLs for testing
https://example.com/article1
https://example.com/article2
# This is a comment
https://example.com/article3
```

## Examples

### Interactive Mode with Single URL
```bash
python main.py --url "https://lilianweng.github.io/posts/2023-06-23-agent/"
```

### Batch Processing from File
```bash
python main.py --urls-file "sample_urls.txt"
```

### Single Question Mode
```bash
python main.py --url "https://example.com/article" --question "What are the main points?"
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

1. **Content Loading**: The app loads web content using LangChain's WebBaseLoader
2. **Chunking**: Content is split into manageable chunks using RecursiveCharacterTextSplitter
3. **Vector Store**: Chunks are embedded and stored in an in-memory vector store
4. **RAG Pipeline**: Uses LangGraph to implement a retrieval-augmented generation pipeline
5. **Question Answering**: Retrieves relevant chunks and generates answers using the configured LLM

## File Structure

```
mini-q-and-a/
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── sample_urls.txt        # Example URLs file
├── README.md              # This file
└── src/
    ├── config.py          # Configuration management
    ├── ingest.py          # Content loading and processing
    └── rag_graph.py       # RAG pipeline implementation
```
