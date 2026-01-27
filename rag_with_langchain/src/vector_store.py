"""
Storage module for Chroma vector store.
"""
import hashlib
from typing import List
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings

from config import CHROMA_PERSIST_DIR, OPENAI_API_KEY, EMBEDDING_MODEL_NAME

class ChromaVectorStore:
    def __init__(self, embedding_model: str = EMBEDDING_MODEL_NAME):
        persist_directory = CHROMA_PERSIST_DIR
        
        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="documents", # collection name
            metadata={"hnsw:space": "cosine"})
        
        # Initialize embeddings for querying
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=embedding_model)
    
    def _generate_chunk_id(source_path: str, chunk_idx: str, content_hash: str) -> str:
        """
        Generate a unique chunk ID for the chunk.
        """
        combined = f"{source_path}|{chunk_idx}|{content_hash}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def upsert_chunks(
        self,
        chunks: List[Document],
        dense_embeddings: List[List[float]],
        content_hashes: List[str]
    ) -> None:
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            # Generate unique chunk ID
            source_path = chunk.metadata.get('source_path', 'unknown')
            chunk_index = chunk.metadata.get('chunk_index', i)
            chunk_id = self._generate_chunk_id(source_path, str(chunk_index), content_hashes[i])
            
            ids.append(chunk_id)
            documents.append(chunk.page_content)
            
            # Dense embedding
            if dense_embeddings[i] is not None:
                embeddings.append(dense_embeddings[i])
            else:
                # Generate on-the-fly if missing
                embeddings.append(self.embeddings.embed_query(chunk.page_content))
            
            # Metadata
            metadata = chunk.metadata.copy()
            metadata['content_hash'] = content_hashes[i]
            metadatas.append(metadata)
        
        # Batch upsert
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas)
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Perform similarity search using dense embeddings.
        Returns top k results.
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k)
        
        # Convert to Documents
        docs = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                doc = Document(
                    page_content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i])
                docs.append(doc)
        return docs
    
    def get_existing_content_hashes(self, limit: int = 10000) -> set:
        """
        Get set of existing content hashes in the vector store.
        This is used to avoid generating embeddings for chunks that already exist.
        """
        # Get all documents (with limit for large collections)
        results = self.collection.get(limit)
        
        hashes = set()
        if results['metadatas']:
            for metadata in results['metadatas']:
                if 'content_hash' in metadata:
                    hashes.add(metadata['content_hash'])
        
        return hashes
