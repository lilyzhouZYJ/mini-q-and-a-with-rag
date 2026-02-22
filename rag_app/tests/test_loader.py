"""
Tests for the loader module.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

# Add parent directory to path to import modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "ingest"))

from loader import Loader, WebPageLoader, TextFileLoader, LoaderFactory


class TestLoader(unittest.TestCase):
    """Test cases for the base Loader class."""
    
    def test_calculate_file_hash(self):
        """Test file hash calculation."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name
        
        try:
            hash1 = Loader._calculate_file_hash(temp_path)
            hash2 = Loader._calculate_file_hash(temp_path)
            # Same content should produce same hash
            self.assertEqual(hash1, hash2)
            # Hash should be a hex string
            self.assertEqual(len(hash1), 64)
        finally:
            os.unlink(temp_path)
    
    def test_calculate_content_hash(self):
        """Test content hash calculation."""
        content = b"test content"
        hash1 = Loader._calculate_content_hash(content)
        hash2 = Loader._calculate_content_hash(content)
        # Same content should produce same hash
        self.assertEqual(hash1, hash2)
        # Hash should be a hex string
        self.assertEqual(len(hash1), 64)


class TestTextFileLoader(unittest.TestCase):
    """Test cases for TextFileLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary text file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        self.temp_file.write("This is test content for the loader.")
        self.temp_file.close()
        self.temp_path = self.temp_file.name
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)
    
    @patch('loader.check_if_file_hash_exists')
    def test_load_text_file(self, mock_check_hash):
        """Test loading a text file."""
        mock_check_hash.return_value = None  # File not processed yet
        
        loader = TextFileLoader(self.temp_path)
        docs, file_hash, should_skip = loader.load()
        
        # Should not skip
        self.assertFalse(should_skip)
        # Should return one document
        self.assertEqual(len(docs), 1)
        # Check document content
        self.assertEqual(docs[0].page_content, "This is test content for the loader.")
        # Check metadata
        self.assertEqual(docs[0].metadata['doc_type'], 'text')
        self.assertIn('source_path', docs[0].metadata)
        self.assertIn('title', docs[0].metadata)
        # Hash should be calculated
        self.assertIsNotNone(file_hash)
    
    @patch('loader.check_if_file_hash_exists')
    def test_load_text_file_already_processed(self, mock_check_hash):
        """Test skipping already processed file."""
        mock_check_hash.return_value = {'status': 'success'}  # Already processed
        
        loader = TextFileLoader(self.temp_path)
        docs, file_hash, should_skip = loader.load()
        
        # Should skip
        self.assertTrue(should_skip)
        # Should return empty documents
        self.assertEqual(len(docs), 0)
        # Hash should still be calculated
        self.assertIsNotNone(file_hash)
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        loader = TextFileLoader("/nonexistent/file.txt")
        with self.assertRaises(FileNotFoundError):
            loader.load()


class TestWebPageLoader(unittest.TestCase):
    """Test cases for WebPageLoader."""
    
    @patch('loader.WebBaseLoader')
    @patch('loader.check_if_file_hash_exists')
    def test_load_webpage(self, mock_check_hash, mock_web_loader_class):
        """Test loading a webpage."""
        # Mock WebBaseLoader
        mock_loader = MagicMock()
        mock_doc = Document(
            page_content="Test webpage content",
            metadata={'title': 'Test Page'}
        )
        mock_loader.load.return_value = [mock_doc]
        mock_web_loader_class.return_value = mock_loader
        
        mock_check_hash.return_value = None  # Not processed yet
        
        loader = WebPageLoader("https://example.com/test")
        docs, content_hash, should_skip = loader.load()
        
        # Should not skip
        self.assertFalse(should_skip)
        # Should return documents
        self.assertGreater(len(docs), 0)
        # Check metadata
        self.assertEqual(docs[0].metadata['doc_type'], 'webpage')
        self.assertEqual(docs[0].metadata['source_path'], "https://example.com/test")
        # Hash should be calculated
        self.assertIsNotNone(content_hash)
    
    @patch('loader.WebBaseLoader')
    @patch('loader.check_if_file_hash_exists')
    def test_load_webpage_already_processed(self, mock_check_hash, mock_web_loader_class):
        """Test skipping already processed webpage."""
        # Mock WebBaseLoader
        mock_loader = MagicMock()
        mock_doc = Document(page_content="Test content")
        mock_loader.load.return_value = [mock_doc]
        mock_web_loader_class.return_value = mock_loader
        
        mock_check_hash.return_value = {'status': 'success'}  # Already processed
        
        loader = WebPageLoader("https://example.com/test")
        docs, content_hash, should_skip = loader.load()
        
        # Should skip
        self.assertTrue(should_skip)
        # Should return empty documents
        self.assertEqual(len(docs), 0)


class TestLoaderFactory(unittest.TestCase):
    """Test cases for LoaderFactory."""
    
    def test_is_url(self):
        """Test URL detection."""
        self.assertTrue(LoaderFactory._is_url("https://example.com"))
        self.assertTrue(LoaderFactory._is_url("http://example.com"))
        self.assertFalse(LoaderFactory._is_url("/path/to/file.txt"))
        self.assertFalse(LoaderFactory._is_url("file.txt"))
    
    def test_get_file_type(self):
        """Test file type detection."""
        self.assertEqual(LoaderFactory._get_file_type(Path("test.txt")), 'text')
        self.assertEqual(LoaderFactory._get_file_type(Path("test.md")), 'text')
        self.assertEqual(LoaderFactory._get_file_type(Path("test.text")), 'text')
        
        with self.assertRaises(ValueError):
            LoaderFactory._get_file_type(Path("test.pdf"))
    
    @patch('loader.WebPageLoader')
    def test_create_loader_for_url(self, mock_web_loader):
        """Test creating loader for URL."""
        mock_web_loader.return_value = Mock()
        loader = LoaderFactory.create_loader("https://example.com")
        self.assertIsInstance(loader, Mock)
        mock_web_loader.assert_called_once_with("https://example.com")
    
    def test_create_loader_for_text_file(self):
        """Test creating loader for text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test")
            temp_path = f.name
        
        try:
            loader = LoaderFactory.create_loader(temp_path)
            self.assertIsInstance(loader, TextFileLoader)
        finally:
            os.unlink(temp_path)
    
    def test_create_loader_for_nonexistent_file(self):
        """Test creating loader for non-existent file."""
        with self.assertRaises(FileNotFoundError):
            LoaderFactory.create_loader("/nonexistent/file.txt")


if __name__ == '__main__':
    unittest.main()

