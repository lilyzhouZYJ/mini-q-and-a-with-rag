#!/usr/bin/env python3
"""
Mini Q&A CLI Application

This application allows you to load web content or text files (.txt, .md) and ask questions about them.
You can provide URLs, file paths, or directories containing files.
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
from ingest import (
    process_single_source,
    process_multiple_sources,
    build_vector_store,
    load_urls_from_file
)

def _collect_files_from_dir(directory: str) -> list:
    """Collect all supported files from a directory."""
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Directory not found: {directory}")
    
    supported_extensions = {'.txt', '.md', '.text'}
    files = []
    
    for file_path in dir_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            files.append(str(file_path))
    
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Mini Q&A CLI Application - Load web content or text files (.txt, .md) and ask questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single URL
  python main.py --url "https://example.com/article"
  
  # Multiple URLs from file
  python main.py --urls-file "urls.txt"
  
  # Single text file
  python main.py --file "document.txt"
  
  # Multiple files from directory
  python main.py --files-dir "./documents"
  
  # Single question mode
  python main.py --file "document.txt" --question "What is this about?"
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
    input_group.add_argument(
        '--file',
        type=str,
        help='Single file path (.txt, .md) to load and process'
    )
    input_group.add_argument(
        '--files-dir',
        type=str,
        help='Directory containing files to process'
    )
    
    # Optional arguments
    parser.add_argument(
        '--interactive', 
        action='store_true', 
        default=True,
        help='Run in interactive mode to ask questions (default: true)'
    )
    parser.add_argument(
        '--no-interactive',
        dest='interactive',
        action='store_false',
        help='Disable interactive mode'
    )
    parser.add_argument(
        '--question', 
        type=str, 
        help='Ask a single question and exit'
    )
    parser.add_argument(
        '--no-transform',
        action='store_true',
        help='Disable LLM-based transformation and enrichment'
    )
    
    args = parser.parse_args()
    
    # Check if API key is set
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in a .env file or environment variable.")
        sys.exit(1)
    
    try:
        # Determine sources to process
        sources = []
        
        if args.url:
            sources = [args.url]
        elif args.urls_file:
            sources = load_urls_from_file(args.urls_file)
            if not sources:
                print(f"No valid URLs found in {args.urls_file}")
                sys.exit(1)
        elif args.file:
            sources = [args.file]
        elif args.files_dir:
            sources = _collect_files_from_dir(args.files_dir)
            if not sources:
                print(f"No supported files found in {args.files_dir}")
                sys.exit(1)
        
        print(f"Processing {len(sources)} source(s)...")
        
        # Build vector store (persistent Chroma)
        vector_store = build_vector_store()
        
        # Process sources through pipeline
        enable_transform = not args.no_transform
        if len(sources) == 1:
            process_single_source(sources[0], vector_store, enable_transform)
        else:
            process_multiple_sources(sources, vector_store, enable_transform)
        
        print("\n" + "="*50)
        print("Processing complete!")
        print("="*50)
        
        # Initialize LLM
        print(f"\nInitializing {MODEL_NAME} model...")
        llm = init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER)
        
        # Build LangGraph
        print("Building RAG graph...")
        graph = build_langgraph(vector_store, llm)
        
        # Handle questions
        if args.question:
            # Single question mode
            print(f"\nQuestion: {args.question}")
            print("Thinking...")
            answer = graph.invoke({"question": args.question})
            print(f"Answer: {answer['answer']}")
        elif args.interactive:
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
        else:
            print("\nNo question provided and interactive mode disabled.")
            print("Use --question to ask a question or --interactive to enable interactive mode.")
                    
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
