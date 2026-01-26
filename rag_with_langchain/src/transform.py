"""
Transform and enrich chunks, using LLM.

We implement the following:

(1) Chunk refinement (post-processing):
- remove noise, i.e. headers, footers, etc.
- fix boundary issues, i.e. sentences or paragraphs were awkwardly split across chunk boundaries
- make each chunk self-contained, i.e. understandable in isolation.

(2) Metadata extraction:
- extract information from each chunk, like title, summary, tags.
"""

import time
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.chat_models import init_chat_model

from config import TRANSFORM_MODEL, MODEL_PROVIDER

def _refine_chunk_prompt(chunk_text: str) -> str:
    return f"""You are a text refinement assistant. Your task is to clean and refine the following text chunk to make it self-contained and semantically coherent.

Text chunk:
{chunk_text}

Instructions:
1. Remove any page headers, footers, or navigation elements
2. Merge related fragments that were incorrectly split
3. Ensure the chunk is self-contained and makes sense on its own
4. Preserve all important information
5. Do not add information that wasn't in the original

Return only the refined text, without any additional commentary."""

def refine_chunk(chunk: Document, llm, max_retries: int = 3) -> Document:
    """
    Refine a chunk using LLM to remove noise and ensure self-containment.
    """
    prompt = _refine_chunk_prompt(chunk.page_content)
    
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            refined_text = response.content if hasattr(response, 'content') else str(response)
            
            # Create new document with refined content; keep metadata
            refined_chunk = Document(
                page_content=refined_text.strip(),
                metadata=chunk.metadata.copy()
            )
            return refined_chunk
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # exponential backoff
                print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"  Failed to refine chunk after {max_retries} attempts: {e}")
                # Return original chunk on failure
                return chunk
    
    return chunk

def _extract_metadata_prompt(chunk_text: str) -> str:
    """Generate prompt for metadata extraction."""
    return f"""Analyze the following text chunk and extract semantic metadata.

Text chunk:
{chunk_text}

Extract the following information:
1. Title: A concise title (3-8 words) that summarizes the main topic
2. Summary: A brief summary (1-2 sentences) of the chunk's content
3. Tags: 3-5 relevant tags/keywords that describe the content

Format your response as JSON:
{{
    "title": "title here",
    "summary": "summary here",
    "tags": ["tag1", "tag2", "tag3"]
}}"""

def extract_metadata(chunk: Document, llm, max_retries: int = 3) -> Dict[str, Any]:
    """
    Extract metadata information from a chunk using LLM.
    Returns: dictionary with title, summary, and tags.
    """
    prompt = _extract_metadata_prompt(chunk.page_content)
    
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse JSON from response
            import json
            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            metadata = json.loads(content)
            
            # Validate structure
            if not all(k in metadata for k in ['title', 'summary', 'tags']):
                raise ValueError("Missing required metadata fields")
            
            return metadata
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # exponential backoff
                print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"  Failed to extract metadata after {max_retries} attempts: {e}")
                # Return default metadata on failure
                return {
                    'title': chunk.metadata.get('title', 'Untitled'),
                    'summary': chunk.page_content[:100] + '...',
                    'tags': []
                }
    
    return {
        'title': chunk.metadata.get('title', 'Untitled'),
        'summary': chunk.page_content[:100] + '...',
        'tags': []
    }

def transform_chunks(
    chunks: List[Document],
    enable_refinement: bool = True,
    enable_metadata: bool = True
) -> List[Document]:
    """
    Main entry-point for refinement and metadata extraction.
    """
    if not chunks:
        return chunks
    
    # Initialize LLM for transformations
    llm = init_chat_model(TRANSFORM_MODEL, model_provider=MODEL_PROVIDER)
    
    transformed = []
    
    for i, chunk in enumerate(chunks):
        try:
            # Refine chunk if enabled
            if enable_refinement:
                chunk = refine_chunk(chunk, llm)
            
            # Extract metadata if enabled
            if enable_metadata:
                semantic_metadata = extract_metadata(chunk, llm)
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