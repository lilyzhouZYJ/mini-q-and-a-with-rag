"""
Step 1: Load content from text files.

Supported formats:
- Text files (.txt, .md)
"""

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple
from langchain.schema import Document

from ingest_db import check_if_file_hash_exists

class Loader(ABC):
    def __init__(self, source: str):
        """
        Initialize the loader.
        Source is the file path.
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
    
    def _check_if_source_exists(self, source_hash: str) -> bool:
        """
        Check if the source can be skipped (already processed).
        Return true if yes.
        """
        existing = check_if_file_hash_exists(source_hash)
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

class TextFileLoader(Loader):
    def load(self) -> Tuple[List[Document], str, bool]:
        print(f"[Loader][TextFileLoader] Loading text file from {self.source}")
        
        # Make sure source file exists
        file_path = Path(self.source)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {self.source}")
        
        # Check if the source has already been loaded
        file_hash = self._calculate_file_hash(str(file_path))
        if self._check_if_source_exists(file_hash):
            print(f"[Loader][TextFileLoader] Source {self.source} has already been loaded; skip")
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

        print(f"[Loader][TextFileLoader] Loaded document from {self.source}")
        return [doc], file_hash, False

class LoaderFactory:
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.txt', '.md'}
    
    @staticmethod
    def _get_file_type(file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix in LoaderFactory.SUPPORTED_EXTENSIONS:
            return 'text'
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    @classmethod
    def create_loader(cls, source: str) -> Loader:
        """
        Create the appropriate loader for the given file path.
        """
        print(f"[Loader][LoaderFactory] Creating loader for source {source}")
        
        file_path = Path(source)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        
        if file_path.is_dir():
            raise ValueError(f"Expected a file path, but got a directory: {source}")
        
        file_type = cls._get_file_type(file_path)
        if file_type == 'text':
            return TextFileLoader(source)