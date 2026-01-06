#!/usr/bin/env python3
"""
Mini Q&A CLI Application

This application allows you to load web content and ask questions about it.
You can provide either a single URL or a file containing multiple URLs.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src directory to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import config before langchain imports so that it picks up USER_AGENT
from config import MODEL_NAME, MODEL_PROVIDER, OPENAI_API_KEY

from langchain.chat_models import init_chat_model
from rag_graph import build_langgraph
from ingest import load_and_chunk_webpage, load_urls_from_file, load_and_chunk_multiple_webpages, build_vector_store

def main():
    parser = argparse.ArgumentParser(
        description="Mini Q&A CLI Application - Load web content and ask questions about it",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single URL
  python main.py --url "https://example.com/article"
  
  # Multiple URLs from file
  python main.py --urls-file "urls.txt"
  
  # Interactive mode with single URL
  python main.py --url "https://example.com/article" --interactive
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--url', 
        type=str, 
        help='Single URL to load and process'
    )
    input_group.add_argument(
        '--urls-file', 
        type=str, 
        help='Path to text file containing URLs (one per line)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--interactive', 
        action='store_true', 
        help='Run in interactive mode to ask questions (default: true)'
    )
    parser.add_argument(
        '--question', 
        type=str, 
        help='Ask a single question and exit'
    )
    
    args = parser.parse_args()
    
    # Check if API key is set
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in a .env file or environment variable.")
        sys.exit(1)
    
    try:
        # Load and process content
        print("Loading content...")
        if args.url:
            print(f"Loading single URL: {args.url}")
            chunks = load_and_chunk_webpage(args.url)
        else:
            print(f"Loading URLs from file: {args.urls_file}")
            urls = load_urls_from_file(args.urls_file)
            if not urls:
                print(f"No valid URLs found in {args.urls_file}")
                sys.exit(1)
            print(f"Found {len(urls)} URLs to process")
            chunks = load_and_chunk_multiple_webpages(urls)
        
        if not chunks:
            print("No content was loaded. Please check your URLs and try again.")
            sys.exit(1)
        
        print(f"Loaded {len(chunks)} chunks total")
        
        # Build vector store
        print("Building vector store...")
        vector_store = build_vector_store(chunks)
        
        # Initialize LLM
        print(f"Initializing {MODEL_NAME} model...")
        llm = init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER)
        
        # Build LangGraph
        print("Building RAG graph...")
        graph = build_langgraph(vector_store, llm)
        
        # Handle questions
        if args.question:
            # Single question mode
            print(f"Question: {args.question}")
            answer = graph.invoke({"question": args.question})
            print(f"Answer: {answer['answer']}")
        else:
            # Interactive mode (default)
            print("\n" + "="*50)
            print("Q&A Interactive Mode")
            print("Ask questions about the loaded content (type 'quit' to exit)")
            print("="*50)
            
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
                    
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
