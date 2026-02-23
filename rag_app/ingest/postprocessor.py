"""
Step 3: Post-processing, i.e. transform / enrich chunks.

We implement the following steps:

(1) Chunk refinement:
- remove noise, i.e. headers, footers, etc.
- fix boundary issues, i.e. sentences or paragraphs were awkwardly split across chunk boundaries
- make each chunk self-contained, i.e. understandable in isolation

(2) Metadata extraction:
- extract information from each chunk, like title and summary
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.chat_models import init_chat_model

class PostProcessor:
    """
    Post-processor for transforming and enriching document chunks.
    Handles chunk refinement and metadata extraction using LLM.
    """
    def __init__(self, model_name: str, model_provider: str, max_retries: int = 3):
        """
        Initialize the post-processor.
        
        Args:
            model_name: Model name for transformation
            model_provider: Model provider (e.g., 'openai')
            max_retries: Maximum number of retries for LLM calls
        """
        self.model_name = model_name
        self.model_provider = model_provider
        self.max_retries = max_retries
        self.prompts_dir = Path(__file__).parent / "prompts"
        self._llm: Optional[Any] = None
    
    @property
    def llm(self):
        """Lazy initialization of LLM using config values."""
        if self._llm is None:
            self._llm = init_chat_model(self.model_name, model_provider=self.model_provider)
        return self._llm
    
    def _load_prompt_template(self, prompt_file: str) -> str:
        """
        Load a prompt template from a file.
        
        Args:
            prompt_file: Name of the prompt file (e.g., "refine_chunk.txt")
            
        Returns:
            The prompt template as a string
        """
        prompt_path = self.prompts_dir / prompt_file
        if not prompt_path.exists():
            raise FileNotFoundError(f"[PostProcessor] Prompt file not found: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def _refine_chunk_prompt(self, chunk_text: str) -> str:
        """
        Generate prompt for chunk refinement by loading template and inserting chunk text.
        """
        template = self._load_prompt_template("refine_chunk.txt")
        return template.format(chunk_text=chunk_text)
    
    def refine_chunk(self, chunk: Document) -> Document:
        """
        Refine a chunk using LLM to remove noise and ensure self-containment.
        
        Args:
            chunk: Document chunk to refine
            
        Returns:
            Refined Document chunk
        """
        print(f"[PostProcessor] Refining chunk with index {chunk.metadata.get('chunk_index', 'unknown')}")
        prompt = self._refine_chunk_prompt(chunk.page_content)
        
        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(prompt)
                refined_text = response.content if hasattr(response, 'content') else str(response)
                
                # Create new document with refined content; keep metadata
                refined_chunk = Document(
                    page_content=refined_text.strip(),
                    metadata=chunk.metadata.copy()
                )
                return refined_chunk
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # exponential backoff
                    print(f"  Retry {attempt + 1}/{self.max_retries} after {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"  Failed to refine chunk after {self.max_retries} attempts: {e}")
                    # Return original chunk on failure
                    return chunk
        
        return chunk
    
    def _extract_metadata_prompt(self, chunk_text: str) -> str:
        """
        Generate prompt for metadata extraction by loading template and inserting chunk text.
        """
        template = self._load_prompt_template("extract_metadata.txt")
        return template.format(chunk_text=chunk_text)
    
    def extract_metadata(self, chunk: Document) -> Dict[str, Any]:
        """
        Extract metadata information from a chunk using LLM.
        
        Args:
            chunk: Document chunk to extract metadata from
            
        Returns:
            Dictionary with title and summary
        """
        print(f"[PostProcessor] Extracting metadata from chunk with index {chunk.metadata.get('chunk_index', 'unknown')}")
        prompt = self._extract_metadata_prompt(chunk.page_content)
        
        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
                
                # Extract JSON from markdown code blocks if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                metadata = json.loads(content)
                
                # Validate structure
                if not all(k in metadata for k in ['title', 'summary']):
                    raise ValueError("Missing required metadata fields")
                
                return metadata
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # exponential backoff
                    print(f"  Retry {attempt + 1}/{self.max_retries} after {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"  Failed to extract metadata after {self.max_retries} attempts: {e}")
                    # Return default metadata on failure
                    return {
                        'title': chunk.metadata.get('title', 'Untitled'),
                        'summary': chunk.page_content[:100] + '...'
                    }
        
        # Shouldn't get here
        return {
            'title': chunk.metadata.get('title', 'Untitled'),
            'summary': chunk.page_content[:100] + '...'
        }
    
    def postprocess_chunks(
        self,
        chunks: List[Document],
        enable_refinement: bool = True,
        enable_metadata: bool = True
    ) -> List[Document]:
        """
        Main entry-point for post-processing chunks.
        
        Args:
            chunks: List of Document chunks to transform
            enable_refinement: Whether to refine chunks (default: True)
            enable_metadata: Whether to extract metadata (default: True)
            
        Returns:
            List of transformed Document chunks
        """
        if not chunks:
            return chunks
        
        transformed = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Refine chunk if enabled
                if enable_refinement:
                    chunk = self.refine_chunk(chunk)
                
                # Extract metadata if enabled
                if enable_metadata:
                    semantic_metadata = self.extract_metadata(chunk)
                    # Merge semantic metadata into chunk metadata
                    chunk.metadata.update(semantic_metadata)
                
                transformed.append(chunk)
                
                if (i + 1) % 10 == 0:
                    print(f"  Transformed {i + 1}/{len(chunks)} chunks")
                    
            except Exception as e:
                print(f"  Error transforming chunk {i + 1}: {e}")
                # Continue with original chunk on error
                transformed.append(chunk)
        
        return transformed