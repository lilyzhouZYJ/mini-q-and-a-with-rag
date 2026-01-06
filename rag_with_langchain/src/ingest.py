import bs4
from typing import List
from langchain.schema import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

from config import OPENAI_API_KEY

def load_and_chunk_webpage(url: str) -> List[Document]:
    """
    Load a web page and split it into chunks.
    """
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return chunks

def load_urls_from_file(file_path: str) -> List[str]:
    """
    Load URLs from a text file (one URL per line).
    """
    urls = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            url = line.strip()
            if url and not url.startswith('#'):  # Skip empty lines and comments
                urls.append(url)
    return urls

def load_and_chunk_multiple_webpages(urls: List[str]) -> List[Document]:
    """
    Load multiple web pages and split them into chunks.
    """
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for url in urls:
        try:
            print(f"Loading: {url}")
            loader = WebBaseLoader(
                web_paths=(url,),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header")
                    )
                ),
            )
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
            print(f"Loaded {len(chunks)} chunks from {url}")
        except Exception as e:
            print(f"Error loading {url}: {e}")
            continue
    
    return all_chunks

def build_vector_store(docs: List[Document] = None) -> InMemoryVectorStore:
    """
    Build an in-memory vector store from provided documents.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = InMemoryVectorStore(embeddings)

    if docs:
        vector_store.add_documents(docs)

    return vector_store