#!/usr/bin/env python3
"""
Document Ingestion Script

This script ingests text files (.txt, .md) into the vector store.
You can provide a single file or a directory containing files.
"""

import argparse
import sys
import os
from pathlib import Path

# Add root directory and ingest directory to path so we can import our modules
root_dir = os.path.dirname(__file__)
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'ingest'))

from ingest.ingest_pipeline import IngestPipeline
from ingest.vector_store import ChromaVectorStore
from ingest.loader import LoaderFactory


def _collect_files_from_dir(directory: str) -> list:
    """Collect all supported files from a directory (recursively)."""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {directory}")
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    supported_extensions = LoaderFactory.SUPPORTED_EXTENSIONS
    files = []
    
    for file_path in dir_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            files.append(str(file_path))
    
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Ingest text files (.txt, .md) into the vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single text file
  python ingest_documents.py --file "document.txt"
  
  # Ingest all files from a directory
  python ingest_documents.py --files-dir "./documents"
  
  # Ingest without LLM-based transformation
  python ingest_documents.py --file "document.txt" --no-transform
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--file',
        type=str,
        help='Single file path (.txt, .md) to ingest'
    )
    input_group.add_argument(
        '--files-dir',
        type=str,
        help='Directory containing files to ingest'
    )
    
    # Optional arguments
    parser.add_argument(
        '--no-transform',
        action='store_true',
        help='Disable LLM-based transformation and enrichment'
    )
    
    args = parser.parse_args()
    
    try:
        # Determine sources to process
        sources = []
        
        if args.file:
            sources = [args.file]
        elif args.files_dir:
            sources = _collect_files_from_dir(args.files_dir)
            if not sources:
                print(f"No supported files found in {args.files_dir}")
                sys.exit(1)
        
        print(f"Processing {len(sources)} source(s)...")
        
        # Build vector store (persistent Chroma)
        vector_store = ChromaVectorStore()
        
        # Create ingestion pipeline
        enable_transform = not args.no_transform
        pipeline = IngestPipeline(
            vector_store=vector_store,
            enable_transform=enable_transform
        )
        
        # Process sources through pipeline
        if len(sources) == 1:
            pipeline.process_single_source(sources[0])
        else:
            pipeline.process_multiple_sources(sources)
        
        print("\n" + "="*50)
        print("Ingestion complete!")
        print("="*50)
        print(f"Successfully ingested {len(sources)} source(s) into the vector store.")
                    
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

