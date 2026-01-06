
#!/usr/bin/env python3

import os
import json
from typing import List
from embeddings import BaseEmbeddings

class VectorStore:
    def __init__(self, chunks: List[str], embeddings: BaseEmbeddings) -> None:
        self.embeddings = embeddings
        self.chunks = chunks

    def build_store(self) -> None:
        """
        Build the vector store from the chunks.
        """
        self.vectors = [self.embeddings.get_embedding(c) for c in self.chunks]

    def persist_store(self, path: str = 'rag/vector_store') -> None:
        """
        Persist the vector store to a directory as json files.
        """
        # Create the storage directory
        if not os.path.exists(path):
            os.makedirs(path)

        # Save both chunks and vectors
        with open(os.path.join(path, 'chunks.json'), 'w') as f:
            json.dump(self.chunks, f)
        with open(os.path.join(path, 'vectors.json'), 'w') as f:
            json.dump(self.vectors, f)

    def load_store(self, path: str = 'rag/vector_store') -> bool:
        """
        Load the vector store from directory.
        Returns True if successful, False otherwise.
        """
        if not os.path.exists(path):
            return False

        with open(os.path.join(path, 'chunks.json'), 'r') as f:
            self.chunks = json.load(f)
        with open(os.path.join(path, 'vectors.json'), 'r') as f:
            self.vectors = json.load(f)

        return len(self.chunks) > 0 and len(self.vectors) > 0

    def query_store(self, query: str, top_k: int = 3) -> List[str]:
        """
        Query the vector store for the most similar chunks to the given query.
        """
        query_embedding = self.embeddings.get_embedding(query)
        similarities = [self.embeddings.cosine_similarity(query_embedding, v_embedding) for v_embedding in self.vectors]
        return [self.chunks[i] for i in sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]]