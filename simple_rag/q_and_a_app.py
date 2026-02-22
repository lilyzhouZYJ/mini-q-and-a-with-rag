#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from chunker import FileChunker
from embeddings import OpenAIEmbeddings
from vector_store import VectorStore
from llm_model import OpenAIModel
from config import MODEL_PROVIDER, OPENAI_API_KEY, EMBEDDING_MODEL_NAME, MODEL_NAME

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Mini Q&A using RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single URL
  python q_and_a_app.py --path "path/to/file.txt"
        """
    )
    
    # Input options (mutually exclusive)
    parser.add_argument(
        '--path', 
        type=str, 
        help='Path to file to load and process'
    )
    
    args = parser.parse_args()

    if MODEL_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            print("Error: OPENAI_API_KEY not found in environment variables.")
            print("Please set your OpenAI API key in a .env file or environment variable.")
            sys.exit(1)
    else:
        print(f"Error: Unsupported model provider: {MODEL_PROVIDER}")
        sys.exit(1)

    # Load and process content
    print("(1) Loading content...")
    chunker = FileChunker(args.path, chunk_size=1000, chunk_overlap=200)
    chunks = chunker.get_chunks(chunker.read_content())
    if not chunks:
        print("No content was loaded. Please check your path and try again.")
        sys.exit(1)
    print(f" - loaded {len(chunks)} chunks total")
    
    # Build vector store
    print("(2) Building vector store...")
    embeddings = OpenAIEmbeddings(OPENAI_API_KEY, EMBEDDING_MODEL_NAME)
    vector_store = VectorStore(chunks, embeddings)

    # Try to load the vector store if it exists;
    # if not, we build it from scratch
    if vector_store.load_store():
        print(" - vector store loaded from storage")
    else:
        vector_store.build_store()
        vector_store.persist_store()
        print(" - vector store built and persisted")

    # (3) Initialize LLM model
    print("(3) Initializing LLM...")
    llm = OpenAIModel(MODEL_NAME, OPENAI_API_KEY)

    # (4) Main loop
    history = []
    while True:
        user_input = input("Question: ")
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not user_input:
            continue
        
        context = vector_store.query_store(user_input, top_k=10)
        response = llm.generate_response(user_input, "\n".join(context), history)
        print(f"Answer: {response}")
        print()
        history.append({"role": "assistant", "content": response})