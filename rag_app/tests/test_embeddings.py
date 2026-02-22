"""
Tests for the embeddings module.
"""

import unittest
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "ingest"))

from embeddings import DenseEmbeddings


class TestDenseEmbeddings(unittest.TestCase):
    """Test cases for the DenseEmbeddings class."""
    
    @patch('embeddings.OpenAIEmbeddings')
    def test_init(self, mock_openai_embeddings_class):
        """Test DenseEmbeddings initialization."""
        mock_embeddings = Mock()
        mock_openai_embeddings_class.return_value = mock_embeddings
        
        dense_emb = DenseEmbeddings()
        
        # Should initialize OpenAIEmbeddings
        mock_openai_embeddings_class.assert_called_once()
        self.assertEqual(dense_emb.embeddings, mock_embeddings)
        self.assertIsNotNone(dense_emb.batch_size)
    
    def test_calculate_chunk_hash(self):
        """Test chunk hash calculation."""
        chunk = Document(
            page_content="Test content",
            metadata={
                'source_path': '/test/path.txt',
                'title': 'Test Title',
                'summary': 'Test summary',
                'chunk_index': 0,
                'other_field': 'ignored'
            }
        )
        
        hash1 = DenseEmbeddings._calculate_chunk_hash(chunk)
        hash2 = DenseEmbeddings._calculate_chunk_hash(chunk)
        
        # Same chunk should produce same hash
        self.assertEqual(hash1, hash2)
        # Hash should be a hex string
        self.assertEqual(len(hash1), 64)
    
    def test_calculate_chunk_hash_different_content(self):
        """Test that different content produces different hashes."""
        chunk1 = Document(
            page_content="Content 1",
            metadata={'source_path': '/test/path.txt', 'title': 'Title'}
        )
        chunk2 = Document(
            page_content="Content 2",
            metadata={'source_path': '/test/path.txt', 'title': 'Title'}
        )
        
        hash1 = DenseEmbeddings._calculate_chunk_hash(chunk1)
        hash2 = DenseEmbeddings._calculate_chunk_hash(chunk2)
        
        # Different content should produce different hashes
        self.assertNotEqual(hash1, hash2)
    
    def test_calculate_chunk_hash_different_metadata(self):
        """Test that different metadata produces different hashes."""
        chunk1 = Document(
            page_content="Same content",
            metadata={'source_path': '/test/path1.txt', 'title': 'Title'}
        )
        chunk2 = Document(
            page_content="Same content",
            metadata={'source_path': '/test/path2.txt', 'title': 'Title'}
        )
        
        hash1 = DenseEmbeddings._calculate_chunk_hash(chunk1)
        hash2 = DenseEmbeddings._calculate_chunk_hash(chunk2)
        
        # Different metadata should produce different hashes
        self.assertNotEqual(hash1, hash2)
    
    def test_calculate_chunk_hash_only_relevant_metadata(self):
        """Test that only relevant metadata fields are used in hash."""
        chunk1 = Document(
            page_content="Same content",
            metadata={
                'source_path': '/test/path.txt',
                'title': 'Title',
                'chunk_index': 0  # This should be ignored
            }
        )
        chunk2 = Document(
            page_content="Same content",
            metadata={
                'source_path': '/test/path.txt',
                'title': 'Title',
                'chunk_index': 1  # This should be ignored
            }
        )
        
        hash1 = DenseEmbeddings._calculate_chunk_hash(chunk1)
        hash2 = DenseEmbeddings._calculate_chunk_hash(chunk2)
        
        # Same content and relevant metadata should produce same hash
        # (chunk_index is not in ['source_path', 'title', 'summary'])
        self.assertEqual(hash1, hash2)
    
    def test_calculate_chunk_hash_missing_metadata(self):
        """Test hash calculation with missing metadata fields."""
        chunk = Document(
            page_content="Test content",
            metadata={'source_path': '/test/path.txt'}  # Missing title and summary
        )
        
        hash_value = DenseEmbeddings._calculate_chunk_hash(chunk)
        
        # Should still produce a valid hash
        self.assertEqual(len(hash_value), 64)
        self.assertIsInstance(hash_value, str)
    
    @patch('embeddings.OpenAIEmbeddings')
    def test_generate_embeddings_new_chunks(self, mock_openai_embeddings_class):
        """Test generating embeddings for new chunks."""
        # Mock OpenAIEmbeddings
        mock_embeddings = Mock()
        mock_embeddings.embed_documents.return_value = [
            [0.1, 0.2, 0.3],  # Embedding for chunk 1
            [0.4, 0.5, 0.6]   # Embedding for chunk 2
        ]
        mock_openai_embeddings_class.return_value = mock_embeddings
        
        dense_emb = DenseEmbeddings()
        dense_emb.batch_size = 100  # Set batch size for testing
        
        chunks = [
            Document(
                page_content="Content 1",
                metadata={'source_path': '/test/path1.txt', 'title': 'Title 1'}
            ),
            Document(
                page_content="Content 2",
                metadata={'source_path': '/test/path2.txt', 'title': 'Title 2'}
            )
        ]
        
        embeddings, chunk_hashes = dense_emb.generate_embeddings(chunks)
        
        # Should generate embeddings for all chunks
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(chunk_hashes), 2)
        self.assertIsNotNone(embeddings[0])
        self.assertIsNotNone(embeddings[1])
        self.assertEqual(embeddings[0], [0.1, 0.2, 0.3])
        self.assertEqual(embeddings[1], [0.4, 0.5, 0.6])
        # Should call embed_documents once with both texts
        mock_embeddings.embed_documents.assert_called_once()
        call_args = mock_embeddings.embed_documents.call_args[0][0]
        self.assertEqual(call_args, ["Content 1", "Content 2"])
    
    @patch('embeddings.OpenAIEmbeddings')
    def test_generate_embeddings_existing_chunks(self, mock_openai_embeddings_class):
        """Test skipping embeddings for existing chunks."""
        # Mock OpenAIEmbeddings
        mock_embeddings = Mock()
        # Set return value for embed_documents (only second chunk will be embedded)
        mock_embeddings.embed_documents.return_value = [
            [0.4, 0.5, 0.6]  # Embedding for chunk 2
        ]
        mock_openai_embeddings_class.return_value = mock_embeddings
        
        dense_emb = DenseEmbeddings()
        
        chunks = [
            Document(
                page_content="Content 1",
                metadata={'source_path': '/test/path1.txt', 'title': 'Title 1'}
            ),
            Document(
                page_content="Content 2",
                metadata={'source_path': '/test/path2.txt', 'title': 'Title 2'}
            )
        ]
        
        # Calculate hash for first chunk
        existing_hash = DenseEmbeddings._calculate_chunk_hash(chunks[0])
        existing_hashes = {existing_hash}
        
        embeddings, chunk_hashes = dense_emb.generate_embeddings(chunks, existing_hashes)
        
        # Should generate embeddings only for second chunk
        self.assertEqual(len(embeddings), 2)
        self.assertIsNone(embeddings[0])  # First chunk skipped
        self.assertIsNotNone(embeddings[1])  # Second chunk embedded
        self.assertEqual(embeddings[1], [0.4, 0.5, 0.6])
        # Should call embed_documents for the second chunk only
        mock_embeddings.embed_documents.assert_called_once()
        call_args = mock_embeddings.embed_documents.call_args[0][0]
        self.assertEqual(call_args, ["Content 2"])
    
    @patch('embeddings.OpenAIEmbeddings')
    def test_generate_embeddings_batch_processing(self, mock_openai_embeddings_class):
        """Test that embeddings are generated in batches."""
        # Mock OpenAIEmbeddings
        mock_embeddings = Mock()
        # Return different embeddings for each batch
        mock_embeddings.embed_documents.side_effect = [
            [[0.1, 0.2], [0.3, 0.4]],  # First batch
            [[0.5, 0.6], [0.7, 0.8]]   # Second batch
        ]
        mock_openai_embeddings_class.return_value = mock_embeddings
        
        dense_emb = DenseEmbeddings()
        dense_emb.batch_size = 2  # Small batch size for testing
        
        # Create 4 chunks
        chunks = [
            Document(
                page_content=f"Content {i}",
                metadata={'source_path': f'/test/path{i}.txt', 'title': f'Title {i}'}
            )
            for i in range(4)
        ]
        
        embeddings, chunk_hashes = dense_emb.generate_embeddings(chunks)
        
        # Should generate embeddings for all chunks
        self.assertEqual(len(embeddings), 4)
        self.assertIsNotNone(embeddings[0])
        self.assertIsNotNone(embeddings[3])
        # Should call embed_documents twice (2 batches of 2)
        self.assertEqual(mock_embeddings.embed_documents.call_count, 2)
    
    @patch('embeddings.OpenAIEmbeddings')
    def test_generate_embeddings_empty_chunks(self, mock_openai_embeddings_class):
        """Test generating embeddings for empty chunk list."""
        mock_embeddings = Mock()
        mock_openai_embeddings_class.return_value = mock_embeddings
        
        dense_emb = DenseEmbeddings()
        embeddings, chunk_hashes = dense_emb.generate_embeddings([])
        
        # Should return empty lists
        self.assertEqual(len(embeddings), 0)
        self.assertEqual(len(chunk_hashes), 0)
        # Should not call embed_documents
        mock_embeddings.embed_documents.assert_not_called()
    
    @patch('embeddings.OpenAIEmbeddings')
    def test_generate_embeddings_none_existing_hashes(self, mock_openai_embeddings_class):
        """Test generating embeddings with None existing_hashes."""
        mock_embeddings = Mock()
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_openai_embeddings_class.return_value = mock_embeddings
        
        dense_emb = DenseEmbeddings()
        
        chunks = [
            Document(
                page_content="Content",
                metadata={'source_path': '/test/path.txt', 'title': 'Title'}
            )
        ]
        
        # Pass None for existing_hashes
        embeddings, chunk_hashes = dense_emb.generate_embeddings(chunks, None)
        
        # Should treat None as empty set and generate embeddings
        self.assertEqual(len(embeddings), 1)
        self.assertIsNotNone(embeddings[0])
        mock_embeddings.embed_documents.assert_called_once()
    
    @patch('embeddings.OpenAIEmbeddings')
    def test_generate_embeddings_all_existing(self, mock_openai_embeddings_class):
        """Test when all chunks already exist."""
        mock_embeddings = Mock()
        mock_openai_embeddings_class.return_value = mock_embeddings
        
        dense_emb = DenseEmbeddings()
        
        chunks = [
            Document(
                page_content="Content 1",
                metadata={'source_path': '/test/path1.txt', 'title': 'Title 1'}
            ),
            Document(
                page_content="Content 2",
                metadata={'source_path': '/test/path2.txt', 'title': 'Title 2'}
            )
        ]
        
        # All chunks exist
        existing_hashes = {
            DenseEmbeddings._calculate_chunk_hash(chunks[0]),
            DenseEmbeddings._calculate_chunk_hash(chunks[1])
        }
        
        embeddings, chunk_hashes = dense_emb.generate_embeddings(chunks, existing_hashes)
        
        # Should return None for all embeddings
        self.assertEqual(len(embeddings), 2)
        self.assertIsNone(embeddings[0])
        self.assertIsNone(embeddings[1])
        # Should not call embed_documents
        mock_embeddings.embed_documents.assert_not_called()
    
    @patch('embeddings.OpenAIEmbeddings')
    def test_generate_embeddings_preserves_order(self, mock_openai_embeddings_class):
        """Test that embeddings are returned in the same order as input chunks."""
        mock_embeddings = Mock()
        mock_embeddings.embed_documents.return_value = [
            [0.1, 0.2],  # For chunk 0
            [0.3, 0.4],  # For chunk 2 (chunk 1 is skipped)
            [0.5, 0.6]   # For chunk 3
        ]
        mock_openai_embeddings_class.return_value = mock_embeddings
        
        dense_emb = DenseEmbeddings()
        
        chunks = [
            Document(
                page_content="Content 0",
                metadata={'source_path': '/test/path0.txt', 'title': 'Title 0'}
            ),
            Document(
                page_content="Content 1",
                metadata={'source_path': '/test/path1.txt', 'title': 'Title 1'}
            ),
            Document(
                page_content="Content 2",
                metadata={'source_path': '/test/path2.txt', 'title': 'Title 2'}
            ),
            Document(
                page_content="Content 3",
                metadata={'source_path': '/test/path3.txt', 'title': 'Title 3'}
            )
        ]
        
        # Skip chunk 1
        existing_hash = DenseEmbeddings._calculate_chunk_hash(chunks[1])
        existing_hashes = {existing_hash}
        
        embeddings, chunk_hashes = dense_emb.generate_embeddings(chunks, existing_hashes)
        
        # Should preserve order: [embedding, None, embedding, embedding]
        self.assertEqual(len(embeddings), 4)
        self.assertIsNotNone(embeddings[0])
        self.assertIsNone(embeddings[1])
        self.assertIsNotNone(embeddings[2])
        self.assertIsNotNone(embeddings[3])
        # Check that embeddings are in correct positions
        self.assertEqual(embeddings[0], [0.1, 0.2])
        self.assertEqual(embeddings[2], [0.3, 0.4])
        self.assertEqual(embeddings[3], [0.5, 0.6])


if __name__ == '__main__':
    unittest.main()

