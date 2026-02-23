"""
Main ingestion pipeline.
Steps: loader => chunker => postprocessor => embeddings => storage
"""

from typing import List, Optional
from loader import LoaderFactory
from chunker import Chunker
from postprocessor import PostProcessor
from embeddings import DenseEmbeddings
from vector_store import ChromaVectorStore
from ingest_db import record_ingestion

class IngestPipeline:    
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        config,
        enable_transform: Optional[bool] = None
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            vector_store: Chroma vector store instance
            config: Configuration object from config_loader
            enable_transform: Whether to enable LLM-based postprocessing (overrides config if provided)
        """
        self.vector_store = vector_store
        self.config = config
        self.enable_transform = enable_transform if enable_transform is not None else config.ingestion.enable_transform
        self.db_path = config.storage.sqlite_db_path
        
        # Initialize components (can be reused across multiple sources)
        if self.enable_transform:
            self.postprocessor = PostProcessor(
                model_name=config.llm.transform_model,
                model_provider=config.llm.provider,
                max_retries=config.ingestion.postprocessor_max_retries
            )
        else:
            self.postprocessor = None
        
        self.embedding_gen = DenseEmbeddings(
            api_key=config.openai_api_key,
            model=config.embedding.model,
            batch_size=config.embedding.batch_size
        )
    
    def process_single_source(self, source: str) -> int:
        """
        Process a single source through the full pipeline.
        
        Args:
            source: File path to process
            
        Returns:
            Number of chunks processed
        """
        print(f"\n[IngestPipeline] Processing: {source}")
        
        # (1) Load content from the source
        loader = LoaderFactory.create_loader(source)
        loader.db_path = self.db_path  # Set db_path for the loader
        docs, source_hash, should_skip = loader.load()
        
        if should_skip:
            print(f"[IngestPipeline] Can skip already-processed source")
            return 0
        
        if not docs:
            print(f"[IngestPipeline] No content loaded")
            record_ingestion(source_hash, source, "failed", self.db_path, 0)
            return 0
        
        print(f"[IngestPipeline] Loaded {len(docs)} document(s)")
        
        # (2) Split the documents into chunks
        chunker = Chunker(
            chunk_size=self.config.ingestion.chunk_size,
            chunk_overlap=self.config.ingestion.chunk_overlap
        )
        chunks = chunker.split(docs)
        print(f"[IngestPipeline] Split into {len(chunks)} chunks")
        
        # (3) Postprocess chunks (if enabled)
        if self.enable_transform and self.postprocessor:
            chunks = self.postprocessor.postprocess_chunks(chunks)
            print(f"[IngestPipeline] Postprocessing complete")
        
        # (4) Generate embeddings
        print(f"[IngestPipeline] Generating embeddings...")
        
        # Get existing content hashes to avoid duplicate embeddings
        existing_hashes = self.vector_store.get_existing_chunk_hashes(
            limit=self.config.ingestion.vector_store_hash_limit
        )
        
        # Generate embeddings
        dense_embeddings, content_hashes = self.embedding_gen.generate_embeddings(
            chunks, existing_hashes
        )
        
        print(f"[IngestPipeline] Generated {len([e for e in dense_embeddings if e is not None])} new embeddings")
        
        # (5) Storage
        print(f"[IngestPipeline] Storing in vector database...")
        
        # Delete old chunks from the same source before upserting new ones
        # This ensures we don't accumulate stale chunks when documents are modified
        if chunks:
            source_path = chunks[0].metadata.get('source_path')
            if source_path:
                self.vector_store.delete_chunks_by_source_path(source_path)
        
        self.vector_store.upsert_chunks(chunks, dense_embeddings, content_hashes)
        print(f"[IngestPipeline] Storage complete")
        
        # Record ingestion
        record_ingestion(source_hash, source, "success", self.db_path, len(chunks))
        print(f"[IngestPipeline] Ingestion complete")
        
        return len(chunks)
    
    def process_multiple_sources(self, sources: List[str]) -> int:
        """
        Process multiple sources through the pipeline.
        
        Args:
            sources: List of file paths
            
        Returns:
            Total number of chunks processed
        """
        total_chunks = 0
        
        for source in sources:
            try:
                chunks_count = self.process_single_source(source)
                total_chunks += chunks_count
                if chunks_count == 0:
                    print(f"[IngestPipeline] Warning: No chunks created for {source}")
            except Exception as e:
                print(f"[IngestPipeline] Error processing {source}: {e}")
                continue
        
        return total_chunks