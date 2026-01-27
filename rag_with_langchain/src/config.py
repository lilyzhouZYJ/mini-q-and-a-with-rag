import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env into os.environ
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")
TRANSFORM_MODEL_NAME = os.getenv("TRANSFORM_MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

# Storage configuration
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(Path(__file__).parent.parent / "chroma_db"))
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", str(Path(__file__).parent.parent / "ingestion_history.db"))

# Processing configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
