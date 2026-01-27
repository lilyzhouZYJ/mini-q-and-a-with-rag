"""
Generate dense (semantic) embeddings for chunks.
Supports incremental processing based on content hashes.
"""

import hashlib
import json
from typing import List, Tuple
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings

from config import OPENAI_API_KEY, EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE

class DenseEmbeddings:
    def __init__(self, embedding_model: str = EMBEDDING_MODEL_NAME, batch_size: int = None):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=embedding_model
        )
        self.batch_size = batch_size if batch_size is not None else EMBEDDING_BATCH_SIZE
    
    def _calculate_content_hash(chunk: Document) -> str:
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
        - returns: tuple of (dense_embeddings, content_hashes)
        """
        if existing_hashes is None:
            existing_hashes = set()
        
        # Filter chunks that need embedding and track indices
        content_hashes = []
        chunks_to_embed = []
        
        for i, chunk in enumerate(chunks):
            content_hash = self._calculate_content_hash(chunk)
            content_hashes.append(content_hash)
            if content_hash not in existing_hashes:
                chunks_to_embed.append((chunk, i))
        
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
        
        return dense_embeddings, content_hashes
