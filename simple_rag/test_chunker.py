#!/usr/bin/env python3

from chunker import FileChunker

import os

file_name = "test.txt"

# Get absolute path to the file
abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file_name))
chunker = FileChunker(abs_path, chunk_size=100, chunk_overlap=20)

content = chunker.read_content()
print(f"Content length: {len(content)}")

chunks = chunker.get_chunks(content)
print(f"Number of chunks: {len(chunks)}")

for i in range(3):
    print(f"=== Chunk {i+1} ===")
    print(chunks[i])
    print()