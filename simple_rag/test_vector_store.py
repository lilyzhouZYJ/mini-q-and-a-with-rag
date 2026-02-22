#!/usr/bin/env python3

import os
from vector_store import VectorStore
from embeddings import OpenAIEmbeddings
from chunker import FileChunker

from config import OPENAI_API_KEY, EMBEDDING_MODEL_NAME

file_name = "test.txt"
abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file_name))

# Chunk the file
chunker = FileChunker(abs_path, chunk_size=100, chunk_overlap=20)
content = chunker.read_content()
chunks = chunker.get_chunks(content)

# Build vector store
embeddings = OpenAIEmbeddings(OPENAI_API_KEY, EMBEDDING_MODEL_NAME)
vector_store = VectorStore(chunks, embeddings)

# Query the vector store
query = "What did Mrs. Bennett want for her daughters?"
results = vector_store.query_store(query)
for i in range(len(results)):
    print(f"=== Result {i+1} ===")
    print(results[i])
    print()