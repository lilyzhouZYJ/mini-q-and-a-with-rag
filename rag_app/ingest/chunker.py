"""
Step 2: Chunk documents loaded by the loader.

Each chunk includes metadata like source_path, title, doc summary, etc.
"""

from typing import List, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Chunker:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Size of chunks
            chunk_overlap: Overlap between chunks
            separators: Optional list of separators for splitting
        """
        # Use LangChain's RecursiveCharacterTextSplitter
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
    
    def split(self, docs: List[Document]) -> List[Document]:
        """
        Split documents into chunks with metadata.
        """
        chunks = self.chunker.split_documents(docs)
        
        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            if not chunk.metadata:
                chunk.metadata = {}
            
            # Add index of the chunk
            chunk.metadata['chunk_index'] = i
            
            # source_path should have been set by the loader
            assert 'source_path' in chunk.metadata, "source_path should have been set by the loader"
        
        return chunks
