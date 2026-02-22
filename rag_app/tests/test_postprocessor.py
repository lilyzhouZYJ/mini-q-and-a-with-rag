"""
Tests for the postprocessor module.
"""

import unittest
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "ingest"))

from postprocessor import PostProcessor


class TestPostProcessor(unittest.TestCase):
    """Test cases for the PostProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = PostProcessor(max_retries=1)
        self.test_chunk = Document(
            page_content="This is a test chunk with some content.",
            metadata={
                'source_path': '/test/path.txt',
                'chunk_index': 0,
                'title': 'Test Chunk'
            }
        )
    
    def test_init(self):
        """Test PostProcessor initialization."""
        processor = PostProcessor(max_retries=5)
        self.assertEqual(processor.max_retries, 5)
        self.assertIsNotNone(processor.prompts_dir)
        self.assertIsNone(processor._llm)
    
    @patch('postprocessor.init_chat_model')
    def test_llm_lazy_initialization(self, mock_init_model):
        """Test that LLM is initialized lazily."""
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm
        
        processor = PostProcessor()
        
        # LLM should not be initialized yet
        self.assertIsNone(processor._llm)
        
        # Accessing llm property should initialize it
        llm = processor.llm
        
        # Should be initialized now
        self.assertIsNotNone(processor._llm)
        self.assertEqual(llm, mock_llm)
        mock_init_model.assert_called_once()
    
    def test_load_prompt_template(self):
        """Test loading a prompt template."""
        # This test assumes the prompt files exist
        try:
            template = self.processor._load_prompt_template("refine_chunk.txt")
            self.assertIsInstance(template, str)
            self.assertGreater(len(template), 0)
            # Should contain the placeholder
            self.assertIn("{chunk_text}", template)
        except FileNotFoundError:
            self.skipTest("Prompt files not found")
    
    def test_load_nonexistent_prompt_template(self):
        """Test loading a non-existent prompt template."""
        with self.assertRaises(FileNotFoundError):
            self.processor._load_prompt_template("nonexistent.txt")
    
    def test_refine_chunk_prompt(self):
        """Test generating refine chunk prompt."""
        try:
            prompt = self.processor._refine_chunk_prompt("test content")
            self.assertIsInstance(prompt, str)
            self.assertIn("test content", prompt)
        except FileNotFoundError:
            self.skipTest("Prompt files not found")
    
    def test_extract_metadata_prompt(self):
        """Test generating extract metadata prompt."""
        try:
            prompt = self.processor._extract_metadata_prompt("test content")
            self.assertIsInstance(prompt, str)
            self.assertIn("test content", prompt)
        except FileNotFoundError:
            self.skipTest("Prompt files not found")
    
    @patch('postprocessor.init_chat_model')
    def test_refine_chunk_success(self, mock_init_model):
        """Test successful chunk refinement."""
        # Mock LLM response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Refined test content"
        mock_llm.invoke.return_value = mock_response
        mock_init_model.return_value = mock_llm
        
        processor = PostProcessor()
        refined_chunk = processor.refine_chunk(self.test_chunk)
        
        # Should return a refined chunk
        self.assertIsInstance(refined_chunk, Document)
        self.assertEqual(refined_chunk.page_content, "Refined test content")
        # Metadata should be preserved
        self.assertEqual(refined_chunk.metadata['chunk_index'], 0)
        self.assertEqual(refined_chunk.metadata['source_path'], '/test/path.txt')
    
    @patch('postprocessor.init_chat_model')
    def test_refine_chunk_failure(self, mock_init_model):
        """Test chunk refinement failure returns original chunk."""
        # Mock LLM to raise exception
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        mock_init_model.return_value = mock_llm
        
        processor = PostProcessor(max_retries=1)
        refined_chunk = processor.refine_chunk(self.test_chunk)
        
        # Should return original chunk on failure
        self.assertEqual(refined_chunk.page_content, self.test_chunk.page_content)
        self.assertEqual(refined_chunk.metadata, self.test_chunk.metadata)
    
    @patch('postprocessor.init_chat_model')
    def test_extract_metadata_success(self, mock_init_model):
        """Test successful metadata extraction."""
        # Mock LLM response with JSON
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "title": "Test Title",
            "summary": "Test summary"
        })
        mock_llm.invoke.return_value = mock_response
        mock_init_model.return_value = mock_llm
        
        processor = PostProcessor()
        metadata = processor.extract_metadata(self.test_chunk)
        
        # Should return valid metadata
        self.assertEqual(metadata['title'], "Test Title")
        self.assertEqual(metadata['summary'], "Test summary")
    
    @patch('postprocessor.init_chat_model')
    def test_extract_metadata_with_code_blocks(self, mock_init_model):
        """Test metadata extraction with JSON in code blocks."""
        # Mock LLM response with JSON in markdown code block
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "```json\n" + json.dumps({
            "title": "Test Title",
            "summary": "Test summary"
        }) + "\n```"
        mock_llm.invoke.return_value = mock_response
        mock_init_model.return_value = mock_llm
        
        processor = PostProcessor()
        metadata = processor.extract_metadata(self.test_chunk)
        
        # Should extract JSON from code block
        self.assertEqual(metadata['title'], "Test Title")
        self.assertEqual(metadata['summary'], "Test summary")
    
    @patch('postprocessor.init_chat_model')
    def test_extract_metadata_failure(self, mock_init_model):
        """Test metadata extraction failure returns default metadata."""
        # Mock LLM to raise exception
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        mock_init_model.return_value = mock_llm
        
        processor = PostProcessor(max_retries=1)
        metadata = processor.extract_metadata(self.test_chunk)
        
        # Should return default metadata
        self.assertEqual(metadata['title'], 'Test Chunk')  # From chunk metadata
        self.assertIn('summary', metadata)
    
    @patch('postprocessor.init_chat_model')
    def test_postprocess_chunks_refinement_only(self, mock_init_model):
        """Test postprocessing with only refinement enabled."""
        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Refined content"
        mock_llm.invoke.return_value = mock_response
        mock_init_model.return_value = mock_llm
        
        processor = PostProcessor()
        chunks = [self.test_chunk]
        
        result = processor.postprocess_chunks(
            chunks,
            enable_refinement=True,
            enable_metadata=False
        )
        
        # Should return processed chunks
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].page_content, "Refined content")
    
    @patch('postprocessor.init_chat_model')
    def test_postprocess_chunks_metadata_only(self, mock_init_model):
        """Test postprocessing with only metadata extraction enabled."""
        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "title": "Extracted Title",
            "summary": "Extracted summary"
        })
        mock_llm.invoke.return_value = mock_response
        mock_init_model.return_value = mock_llm
        
        processor = PostProcessor()
        chunks = [self.test_chunk]
        
        result = processor.postprocess_chunks(
            chunks,
            enable_refinement=False,
            enable_metadata=True
        )
        
        # Should return chunks with extracted metadata
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].metadata['title'], "Extracted Title")
        self.assertEqual(result[0].metadata['summary'], "Extracted summary")
    
    @patch('postprocessor.init_chat_model')
    def test_postprocess_chunks_both_enabled(self, mock_init_model):
        """Test postprocessing with both refinement and metadata enabled."""
        # Mock LLM - first call for refinement, second for metadata
        mock_llm = Mock()
        mock_response1 = Mock()
        mock_response1.content = "Refined content"
        mock_response2 = Mock()
        mock_response2.content = json.dumps({
            "title": "Extracted Title",
            "summary": "Extracted summary"
        })
        mock_llm.invoke.side_effect = [mock_response1, mock_response2]
        mock_init_model.return_value = mock_llm
        
        processor = PostProcessor()
        chunks = [self.test_chunk]
        
        result = processor.postprocess_chunks(
            chunks,
            enable_refinement=True,
            enable_metadata=True
        )
        
        # Should return processed chunks with both refinement and metadata
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].page_content, "Refined content")
        self.assertEqual(result[0].metadata['title'], "Extracted Title")
    
    def test_postprocess_chunks_empty_list(self):
        """Test postprocessing empty chunk list."""
        result = self.processor.postprocess_chunks([])
        self.assertEqual(len(result), 0)
    
    @patch('postprocessor.init_chat_model')
    def test_postprocess_chunks_error_handling(self, mock_init_model):
        """Test that errors in processing don't crash the whole batch."""
        # Mock LLM to raise exception
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("Processing error")
        mock_init_model.return_value = mock_llm
        
        processor = PostProcessor(max_retries=1)
        chunks = [self.test_chunk, Document(
            page_content="Another chunk",
            metadata={'source_path': '/test/path2.txt', 'chunk_index': 1}
        )]
        
        # Should handle errors gracefully and continue
        result = processor.postprocess_chunks(chunks)
        
        # Should return all chunks (even if processing failed)
        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()

