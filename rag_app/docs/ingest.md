# Document Ingestion Documentation

## Overview

The ingestion process transforms raw text files into searchable vector embeddings stored in a persistent vector database. The pipeline processes documents through several stages: loading, chunking, post-processing, embedding generation, and storage.

## Ingestion Pipeline

The ingestion pipeline follows this flow:

```
Text Files → Loader → Chunker → PostProcessor → Embeddings → Vector Store
                      ↓
                 Ingestion DB (tracks history)
```

### Pipeline Stages

1. **Loader** - Reads text files and checks if they've been processed
2. **Chunker** - Splits documents into manageable chunks
3. **PostProcessor** (optional) - Refines chunks and extracts metadata using LLM
4. **Embeddings** - Generates semantic vector embeddings
5. **Vector Store** - Stores chunks and embeddings in ChromaDB
6. **Ingestion DB** - Tracks ingestion history to avoid reprocessing

## Components

### 1. Loader (`ingest/loader.py`)

The loader is responsible for reading text files from the filesystem.

#### Supported Formats
- `.txt` - Plain text files
- `.md` - Markdown files

#### Features
- **File Hash Calculation**: Uses SHA256 to calculate file hashes for deduplication
- **Duplicate Detection**: Checks ingestion history to skip already-processed files
- **Metadata Extraction**: Extracts file path, title, and document type

#### Classes

**`TextFileLoader`**
- Loads a single text file
- Returns a LangChain `Document` object with metadata
- Calculates file hash for tracking

**`LoaderFactory`**
- Factory class for creating appropriate loaders
- Determines file type and creates corresponding loader
- Validates file existence and type

### 2. Chunker (`ingest/chunker.py`)

Splits documents into smaller chunks for better retrieval and processing.

#### Configuration
- **`chunk_size`**: Default 1000 tokens per chunk
- **`chunk_overlap`**: Default 200 tokens overlap between chunks
- **`separators`**: Custom separators for splitting (optional)

#### Features
- Uses LangChain's `RecursiveCharacterTextSplitter`
- Preserves document metadata in each chunk
- Adds `chunk_index` to track chunk position
- Maintains `source_path` for traceability

#### Why Chunking?
- Large documents are split into manageable pieces
- Overlap prevents loss of context at boundaries
- Enables more precise retrieval of relevant sections

### 3. PostProcessor (`ingest/postprocessor.py`)

Optional step that uses LLM to refine and enrich chunks.

#### Two Main Functions

**Chunk Refinement**
- Removes noise (headers, footers, navigation elements)
- Fixes boundary issues (incomplete sentences/paragraphs)
- Makes chunks self-contained and understandable in isolation

**Metadata Extraction**
- Extracts semantic title for each chunk
- Generates summary of chunk content

#### Configuration
- **`max_retries`**: Number of retry attempts (default: 3)
- **`enable_refinement`**: Enable/disable chunk refinement
- **`enable_metadata`**: Enable/disable metadata extraction

#### LLM Usage
- Uses the model specified in `TRANSFORM_MODEL_NAME` config
- Implements exponential backoff for retries
- Falls back to original chunk if processing fails

#### Prompt Templates
- `refine_chunk.txt` - Template for chunk refinement
- `extract_metadata.txt` - Template for metadata extraction

### 4. Embeddings (`ingest/embeddings.py`)

Generates dense vector embeddings for semantic search.

#### Features
- **Incremental Processing**: Skips chunks that already exist (based on content hash)
- **Batch Processing**: Processes embeddings in configurable batches
- **Content Hashing**: Uses SHA256 hash of content + metadata to detect duplicates

#### Hash Calculation
The hash includes:
- Chunk content (`page_content`)
- Key metadata fields: `source_path`, `title`, `summary`

This ensures that if content or relevant metadata changes, new embeddings are generated.

#### Configuration
- **`EMBEDDING_MODEL_NAME`**: OpenAI embedding model (default: `text-embedding-3-small`)
- **`EMBEDDING_BATCH_SIZE`**: Number of chunks to embed per batch

### 5. Vector Store (`ingest/vector_store.py`)

Stores chunks and embeddings in ChromaDB for retrieval.

#### Features
- **Persistent Storage**: Data persists across sessions
- **Unique Chunk IDs**: Generated from source path, chunk index, and content hash
- **Metadata Preservation**: All chunk metadata is stored
- **Similarity Search**: Uses cosine similarity for retrieval

#### ChromaDB Configuration
- **Collection Name**: Configurable via `CHROMA_COLLECTION_NAME`
- **Persistence Directory**: Configurable via `CHROMA_PERSIST_DIR`
- **Similarity Metric**: Cosine similarity

#### Methods
- `upsert_chunks()` - Store or update chunks
- `similarity_search()` - Search for similar chunks
- `get_existing_chunk_hashes()` - Get hashes of existing chunks

### 6. Ingestion Database (`ingest/ingest_db.py`)

SQLite database that tracks ingestion history.

#### Purpose
- Prevents reprocessing of unchanged files
- Tracks ingestion status and errors
- Records chunk counts per file

#### Schema
```sql
ingestion_history (
    file_hash TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    status TEXT NOT NULL,  -- 'success', 'failed', 'processing'
    processed_at TIMESTAMP NOT NULL,
    chunk_count INTEGER DEFAULT 0
)
```

#### Functions
- `check_if_file_hash_exists()` - Check if file was already processed
- `record_ingestion()` - Record ingestion attempt and status

## Usage

### Basic Ingestion

```bash
# Ingest a single file
python ingest_documents.py --file "document.txt"

# Ingest all files from a directory
python ingest_documents.py --files-dir "./documents"
```

### Without Post-Processing

To skip LLM-based refinement and metadata extraction (faster, but lower quality):

```bash
python ingest_documents.py --file "document.txt" --no-transform
```

### Command Line Options

- `--file PATH`: Ingest a single text file (.txt, .md)
- `--files-dir PATH`: Ingest all supported files from a directory (recursively)
- `--no-transform`: Disable LLM-based transformation and enrichment

## Configuration

### Environment Variables

Key configuration options (set in `.env` file or environment):

- `OPENAI_API_KEY` - Required for embeddings and optional post-processing
- `EMBEDDING_MODEL_NAME` - Embedding model (default: `text-embedding-3-small`)
- `EMBEDDING_BATCH_SIZE` - Batch size for embedding generation
- `TRANSFORM_MODEL_NAME` - Model for post-processing (default: `gpt-4o-mini`)
- `CHROMA_PERSIST_DIR` - Directory for ChromaDB storage
- `CHROMA_COLLECTION_NAME` - Name of the ChromaDB collection
- `SQLITE_DB_PATH` - Path to ingestion history database

### Pipeline Parameters

Default values in `IngestPipeline`:
- `chunk_size`: 1000 tokens
- `chunk_overlap`: 200 tokens
- `enable_transform`: True

## Ingestion Flow Example

Here's what happens when you ingest a document:

1. **File Detection**: Script collects files from specified path(s)
2. **Hash Check**: For each file, calculate SHA256 hash
3. **Duplicate Check**: Query ingestion DB - if file was successfully processed, skip it
4. **Loading**: Read file content into LangChain Document
5. **Chunking**: Split document into chunks (e.g., 1000 tokens with 200 overlap)
6. **Post-Processing** (if enabled):
   - Refine each chunk using LLM
   - Extract title and summary
7. **Embedding Generation**:
   - Calculate content hash for each chunk
   - Check existing hashes in vector store
   - Generate embeddings only for new/modified chunks
8. **Storage**: Upsert chunks, embeddings, and metadata to ChromaDB
9. **History Recording**: Record successful ingestion in SQLite DB
