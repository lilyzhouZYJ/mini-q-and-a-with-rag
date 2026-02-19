"""
Main ingestion pipeline.
Stages: loader => splitter => transform => embeddings => storage
"""

from typing import List

from loader import LoaderFactory
from splitter import DocumentSplitter
from transform import transform_chunks
from embeddings import DenseEmbeddings
from vector_store import ChromaVectorStore
from ingest_db import record_ingestion

def process_single_source(
    source: str,
    vector_store: ChromaVectorStore,
    enable_transform: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> int:
    """
    Process a single source through the full pipeline.
    Returns the number of chunks processed.
    """
    print(f"\nProcessing: {source}")
    
    # (1) Load content from the source
    loader = LoaderFactory.create_loader(source)
    docs, source_hash, should_skip = loader.load()
    
    if should_skip:
        print(f"  Can skip already-processed source")
        return 0
    
    if not docs:
        print(f"  No content loaded")
        record_ingestion(source_hash, source, "failed", 0)
        return 0
    
    print(f"  Loaded {len(docs)} document(s)")
    
    # (2) Split the documents into chunks
    splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split(docs)
    print(f"  Split into {len(chunks)} chunks")
    
    # (3) Transform chunks
    if enable_transform:
        print(f"  Transforming chunks...")
        chunks = transform_chunks(chunks, enable_refinement=True, enable_metadata=True)
        print(f"  Transformation complete")
    
    # (4) Generate embeddings
    print(f"  Generating embeddings...")
    embedding_gen = DenseEmbeddings()
    
    # Get existing content hashes to avoid duplicate embeddings
    existing_hashes = vector_store.get_existing_content_hashes()
    
    # Generate embeddings
    dense_embeddings, content_hashes = embedding_gen.generate_embeddings(
        chunks, existing_hashes
    )
    
    print(f"  Generated {len([e for e in dense_embeddings if e is not None])} new embeddings")
    
    # (7) Storage
    print(f"  Storing in vector database...")
    vector_store.upsert_chunks(chunks, dense_embeddings, content_hashes)
    print(f"  Storage complete")
    
    # Record ingestion
    record_ingestion(source_hash, source, "success", len(chunks))
    
    return len(chunks)


def process_multiple_sources(
    sources: List[str],
    vector_store: ChromaVectorStore,
    enable_transform: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> int:
    """
    Process multiple sources through the pipeline.
    
    Args:
        sources: List of URLs or file paths
        vector_store: Chroma vector store instance
        enable_transform: Whether to enable LLM-based transformation
        chunk_size: Chunk size for splitting
        chunk_overlap: Chunk overlap for splitting
        
    Returns:
        Total number of chunks processed
    """
    total_chunks = 0
    
    for source in sources:
        try:
            chunks_count = process_single_source(
                source,
                vector_store,
                enable_transform,
                chunk_size,
                chunk_overlap
            )
            total_chunks += chunks_count
        except Exception as e:
            print(f"  Error processing {source}: {e}")
            continue
    
    return total_chunks


def load_urls_from_file(file_path: str) -> List[str]:
    """
    Load URLs from a text file (one URL per line).
    Lines starting with '#' are treated as comments and skipped.
    
    Args:
        file_path: Path to the text file containing URLs
        
    Returns:
        List of URLs (with empty lines and comments filtered out)
    """
    urls = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                urls.append(line)
    return urls


def build_vector_store(collection_name: str = "documents") -> ChromaVectorStore:
    """
    Build and return a Chroma vector store.
    
    Args:
        collection_name: Name of the Chroma collection
        
    Returns:
        ChromaVectorStore instance
    """
    return ChromaVectorStore(collection_name=collection_name)