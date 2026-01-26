"""
Load content from various sources.

Supported formats:
- URLs (web pages)
- Text files
"""

import bs4
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader

from ingest_db import check_file_hash

class Loader(ABC):
    def __init__(self, source: str):
        """
        Initialize the loader.
        Source is the file path or the URL.
        """
        self.source = source
    
    @staticmethod
    def _calculate_file_hash(file_path: str) -> str:
        """
        Calculate the SHA256 hash of a file.
        Note that we don't load the file content into memory.
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    @staticmethod
    def _calculate_content_hash(content: bytes) -> str:
        """
        Calculate the SHA256 hash of content bytes.
        This function is used for webpage loading, which requires that
        we first read the content into memory.
        """
        return hashlib.sha256(content).hexdigest()
    
    def _check_early_exit(self, source_hash: str) -> bool:
        """
        Check if the source can be skipped (already processed).
        Return true if yes.
        """
        existing = check_file_hash(source_hash)
        return existing is not None and existing.get('status') == 'success'
    
    @abstractmethod
    def load(self) -> Tuple[List[Document], str, bool]:
        """
        Load content from the source.
        
        Returns:
            Tuple of (documents, hash, should_skip)
            - documents: List of Document objects (empty if should_skip is True)
            - hash: SHA256 hash of the source
            - should_skip: True if early exit (already processed)
        """
        pass

class WebPageLoader(Loader):
    def load(self) -> Tuple[List[Document], str, bool]:
        loader = WebBaseLoader(
            web_paths=(self.source,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()
        
        # Combine the fetched content and calculate the hash
        content = "\n\n".join(doc.page_content for doc in docs).encode('utf-8')
        content_hash = self._calculate_content_hash(content)
        
        # Check if the source can be skipped (already processed)
        if self._check_early_exit(content_hash):
            return [], content_hash, True
        
        # Include metadata for each document
        for doc in docs:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata['source_path'] = self.source
            doc.metadata['doc_type'] = 'webpage'
            if 'title' not in doc.metadata:
                doc.metadata['title'] = urlparse(self.source).path.split('/')[-1] or self.source
        
        return docs, content_hash, False

class TextFileLoader(Loader):
    def load(self) -> Tuple[List[Document], str, bool]:
        file_path = Path(self.source)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {self.source}")
        
        # Calculate file hash before loading content into memory
        file_hash = self._calculate_file_hash(str(file_path))
        if self._check_early_exit(file_hash):
            return [], file_hash, True
        
        # Read text file into memory
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        doc = Document(
            page_content=content,
            metadata={
                'source_path': str(file_path.absolute()),
                'doc_type': 'text',
                'title': file_path.stem
            }
        )
        
        return [doc], file_hash, False

class LoaderFactory:
    @staticmethod
    def _is_url(source: str) -> bool:
        parsed = urlparse(source)
        return parsed.scheme in ('http', 'https')
    
    @staticmethod
    def _get_file_type(file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix in {'.txt', '.md', '.text'}:
            return 'text'
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    @classmethod
    def create_loader(cls, source: str) -> Loader:
        """
        Create the appropriate loader for the given source.
        """
        if cls._is_url(source):
            return WebPageLoader(source)
        else:
            file_path = Path(source)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {source}")
            
            file_type = cls._get_file_type(file_path)
            if file_type == 'text':
                return TextFileLoader(source)