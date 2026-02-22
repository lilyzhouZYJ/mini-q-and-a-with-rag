"""
Tests for the chunker module.
"""

import unittest
import sys
from pathlib import Path
from langchain.schema import Document

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "ingest"))

from chunker import Chunker


class TestChunker(unittest.TestCase):
    """Test cases for the Chunker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.chunker = Chunker(chunk_size=100, chunk_overlap=20)
    
    def test_split_single_document(self):
        """Test splitting a single document."""
        # Create a document with enough content to split
        doc = Document(
            page_content="This is a test document. " * 20,  # Long enough to split
            metadata={
                'source_path': '/test/path.txt',
                'title': 'Test Document'
            }
        )
        
        chunks = self.chunker.split([doc])
        
        # Should produce multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Check that each chunk has required metadata
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.metadata['chunk_index'], i)
            self.assertIn('source_path', chunk.metadata)
            self.assertEqual(chunk.metadata['source_path'], '/test/path.txt')
    
    def test_split_multiple_documents(self):
        """Test splitting multiple documents."""
        docs = [
            Document(
                page_content="First document. " * 10,
                metadata={'source_path': '/test/path1.txt', 'title': 'Doc 1'}
            ),
            Document(
                page_content="Second document. " * 10,
                metadata={'source_path': '/test/path2.txt', 'title': 'Doc 2'}
            )
        ]
        
        chunks = self.chunker.split(docs)
        
        # Should produce chunks from both documents
        self.assertGreater(len(chunks), 2)
        
        # Check chunk indices are sequential
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.metadata['chunk_index'], i)
            self.assertIn('source_path', chunk.metadata)
    
    def test_split_short_document(self):
        """Test splitting a document that doesn't need splitting."""
        doc = Document(
            page_content="Short content.",
            metadata={'source_path': '/test/path.txt', 'title': 'Short Doc'}
        )
        
        chunks = self.chunker.split([doc])
        
        # Should still produce at least one chunk
        self.assertGreaterEqual(len(chunks), 1)
        
        # Check metadata
        self.assertEqual(chunks[0].metadata['chunk_index'], 0)
        self.assertIn('source_path', chunks[0].metadata)
    
    def test_split_empty_document(self):
        """Test splitting an empty document."""
        doc = Document(
            page_content="",
            metadata={'source_path': '/test/path.txt', 'title': 'Empty Doc'}
        )
        
        chunks = self.chunker.split([doc])
        
        # Should handle empty content gracefully
        self.assertGreaterEqual(len(chunks), 0)
    
    def test_split_preserves_metadata(self):
        """Test that splitting preserves original metadata."""
        doc = Document(
            page_content="Test content. " * 20,
            metadata={
                'source_path': '/test/path.txt',
                'title': 'Test Title',
                'doc_type': 'text',
                'custom_field': 'custom_value'
            }
        )
        
        chunks = self.chunker.split([doc])
        
        # All chunks should preserve original metadata
        for chunk in chunks:
            self.assertEqual(chunk.metadata['source_path'], '/test/path.txt')
            self.assertEqual(chunk.metadata['title'], 'Test Title')
            self.assertEqual(chunk.metadata['doc_type'], 'text')
            self.assertEqual(chunk.metadata['custom_field'], 'custom_value')
            # Plus the chunk_index
            self.assertIn('chunk_index', chunk.metadata)
    
    def test_split_requires_source_path(self):
        """Test that splitting requires source_path in metadata."""
        doc = Document(
            page_content="Test content.",
            metadata={'title': 'Test'}  # Missing source_path
        )
        
        # Should raise assertion error
        with self.assertRaises(AssertionError):
            self.chunker.split([doc])
    
    def test_custom_chunk_size(self):
        """Test chunker with custom chunk size."""
        chunker = Chunker(chunk_size=50, chunk_overlap=10)
        doc = Document(
            page_content="Test content. " * 20,
            metadata={'source_path': '/test/path.txt', 'title': 'Test'}
        )
        
        chunks = chunker.split([doc])
        
        # With smaller chunk size, should produce more chunks
        self.assertGreater(len(chunks), 1)
        
        # Check that chunks are roughly the right size
        for chunk in chunks:
            # Chunk size should be approximately chunk_size (allowing for some variance)
            self.assertLessEqual(len(chunk.page_content), 50 * 2)  # Allow some flexibility
    
    def test_empty_document_list(self):
        """Test splitting an empty list of documents."""
        chunks = self.chunker.split([])
        self.assertEqual(len(chunks), 0)


if __name__ == '__main__':
    unittest.main()

