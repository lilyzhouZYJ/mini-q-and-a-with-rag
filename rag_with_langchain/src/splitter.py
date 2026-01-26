"""
Chunk documents.

Each chunk includes metadata like source_path, title, doc summary, etc.
"""

from typing import List, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        
        # We will use LangChain's RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
    
    def split(self, docs: List[Document]) -> List[Document]:
        """
        Split documents into chunks with metadata.
        """
        chunks = self.splitter.split_documents(docs)
        
        # Add chunk-related metadata
        for i, chunk in enumerate(chunks):
            if not chunk.metadata:
                chunk.metadata = {}
            
            # Add chunk positioning metadata
            chunk.metadata['chunk_index'] = i
            chunk.metadata['start_offset'] = 0  # Could calculate actual offset if needed
            chunk.metadata['end_offset'] = len(chunk.page_content)
            
            # source_path should have been set by the loader
            assert 'source_path' in chunk.metadata, "source_path should have been set by the loader"
        
        return chunks
