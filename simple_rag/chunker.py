#!/usr/bin/env python3

import tiktoken
from typing import List

class BaseChunker:
    def __init__(self, path: str, chunk_size: int = 1000, chunk_overlap: int = 0) -> None:
        self.path = path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_content(self) -> str:
        """
        Read the content of the file.
        """
        raise NotImplementedError

    def get_chunks(self, content: str) -> List[str]:
        """
        Split the content into chunks. The length of each chunk should
        be less than or equal to chunk_size, plus the overlap.
        Note that chunk_size and chunk_overlap are the number of tokens,
        not the number of characters.
        """
        raise NotImplementedError

class FileChunker(BaseChunker):
    def __init__(self, path: str, chunk_size: int = 1000, chunk_overlap: int = 0) -> None:
        super().__init__(path, chunk_size, chunk_overlap)

    def read_content(self) -> str:
        with open(self.path, 'r') as file:
            return file.read()

    def get_chunks(self, content: str) -> None:
        # Encode entire content into tokens
        enc = tiktoken.encoding_for_model("gpt-4")
        tokens = enc.encode(content)
        
        chunks = []
        start_idx = 0
        actual_chunk_size = self.chunk_size - self.chunk_overlap

        while start_idx < len(tokens):
            # Get end of current chunk
            end_idx = start_idx + self.chunk_size

            # Tokens of this chunk
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode back to text and append to chunks
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move start index forward while accounting for overlap;
            # next chunk will start at (current_start + chunk_size - overlap)
            start_idx += actual_chunk_size

        return chunks