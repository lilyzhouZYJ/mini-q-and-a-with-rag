#!/usr/bin/env python3
"""
Interactive Q&A Script

This script allows you to ask questions about documents that have been ingested
into the vector store. Run this after ingesting documents with ingest_documents.py.
"""

import sys
import os
from langchain.chat_models import init_chat_model

# Add root directory and ingest directory to path so we can import our modules
root_dir = os.path.dirname(__file__)
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'ingest'))

from rag_graph import build_langgraph
from ingest.vector_store import ChromaVectorStore
from config import MODEL_NAME, MODEL_PROVIDER, OPENAI_API_KEY


def main():
    
    # Check if API key is set
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in a .env file or environment variable.")
        sys.exit(1)
    
    try:
        # Load vector store (persistent Chroma)
        print("Loading vector store...")
        vector_store = ChromaVectorStore()
        
        # Check if vector store has any documents
        try:
            doc_count = vector_store.collection.count()
            if doc_count == 0:
                print("Warning: Vector store is empty.")
                print("Please ingest documents first using ingest_documents.py")
                sys.exit(1)
            print(f"Found {doc_count} document(s) in vector store.")
        except Exception as e:
            print(f"Warning: Could not access vector store: {e}")
            print("Please ingest documents first using ingest_documents.py")
            sys.exit(1)
        
        # Initialize LLM
        print(f"Initializing {MODEL_NAME} model...")
        llm = init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER)
        
        # Build LangGraph
        print("Building RAG graph...")
        graph = build_langgraph(vector_store, llm)
        
        print("\n" + "="*50)
        print("Ready to answer questions!")
        print("Ask questions about the ingested content (type 'quit' to exit)")
        print("="*50)
        
        # Interactive mode
        while True:
            try:
                query = input("\nQuestion: ").strip()
                if query.lower() in {"quit", "exit", "q"}:
                    print("Goodbye!")
                    break
                if not query:
                    continue
                
                print("Thinking...")
                answer = graph.invoke({"question": query})
                print(f"Answer: {answer['answer']}")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

