"""
Step 4: Generate dense (semantic) embeddings for chunks.
Supports incremental processing based on content hashes.
"""

import hashlib
import json
from typing import List, Tuple
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings

class DenseEmbeddings:
    def __init__(self, api_key: str, model: str, batch_size: int):
        """
        Initialize the embeddings generator.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model name
            batch_size: Batch size for embedding generation
        """
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model=model
        )
        self.batch_size = batch_size
    
    @staticmethod
    def _calculate_chunk_hash(chunk: Document) -> str:
        """
        Calculate SHA256 hash of chunk content and metadata.
        We will use the hash for incremental processing.
        """
        # Combine content and key metadata for hash
        content_str = chunk.page_content
        metadata_str = json.dumps(
            {
                k: v for k, v in chunk.metadata.items()
                if k in ['source_path', 'title', 'summary']
            },
            sort_keys=True
        )
        combined = f"{content_str}|{metadata_str}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def generate_embeddings(
        self,
        chunks: List[Document],
        existing_hashes: set = None
    ) -> Tuple[List[List[float]], List[str]]:
        """
        Generate dense embeddings for the given chunks.
        - existing_hashes: set of content hashes that already exist in the vector store
        - returns: tuple of (dense_embeddings, chunk_hashes)
        """
        print(f"[DenseEmbeddings] Generating embeddings for {len(chunks)} chunks")
        if existing_hashes is None:
            existing_hashes = set()
        
        # Filter chunks that need embedding and track indices
        chunk_hashes = []
        chunks_to_embed = []
        
        for i, chunk in enumerate(chunks):
            chunk_hash = self._calculate_chunk_hash(chunk)
            chunk_hashes.append(chunk_hash)
            if chunk_hash not in existing_hashes:
                chunks_to_embed.append((chunk, i))
        print(f"[DenseEmbeddings] {len(chunks_to_embed)} chunks need embedding")
        
        # Generate dense embeddings
        dense_embeddings = [None] * len(chunks)
        if chunks_to_embed:
            all_dense_vecs = []
            texts = [chunk.page_content for chunk, _ in chunks_to_embed]
            
            # Generate dense embeddings in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_vecs = self.embeddings.embed_documents(batch_texts)
                all_dense_vecs.extend(batch_vecs)
            
            # Map embeddings in all_dense_vecs back to their original positions
            # in the full chunks list
            for idx, (_, orig_idx) in enumerate(chunks_to_embed):
                dense_embeddings[orig_idx] = all_dense_vecs[idx]
        
        return dense_embeddings, chunk_hashes
