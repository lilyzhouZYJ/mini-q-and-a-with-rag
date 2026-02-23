"""Configuration loading and validation for the RAG application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv

# Load environment variables from .env into os.environ (if file exists)
# This is optional - environment variables can be set directly
try:
    load_dotenv()
except Exception:
    # If .env file doesn't exist or can't be read, continue without it
    pass

# ---------------------------------------------------------------------------
# Repo root & path resolution
# ---------------------------------------------------------------------------
# Anchored to this file's location: <repo>/rag_app/config/config_loader.py → parent.parent
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Default absolute path to config.yaml
DEFAULT_CONFIG_PATH: Path = REPO_ROOT / "config" / "config.yaml"


def resolve_path(relative: Union[str, Path]) -> Path:
    """Resolve a repo-relative path to an absolute path.

    If *relative* is already absolute it is returned as-is.  Otherwise
    it is resolved against :data:`REPO_ROOT`.

    >>> resolve_path("config/config.yaml")  # doctest: +SKIP
    PosixPath('/home/user/rag_app/config/config.yaml')
    """
    p = Path(relative)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


class ConfigError(ValueError):
    """Raised when config validation fails."""


def _require_mapping(data: Dict[str, Any], key: str, path: str) -> Dict[str, Any]:
    value = data.get(key)
    if value is None:
        raise ConfigError(f"Missing required field: {path}.{key}")
    if not isinstance(value, dict):
        raise ConfigError(f"Expected mapping for field: {path}.{key}")
    return value


def _require_value(data: Dict[str, Any], key: str, path: str) -> Any:
    if key not in data or data.get(key) is None:
        raise ConfigError(f"Missing required field: {path}.{key}")
    return data[key]


def _require_str(data: Dict[str, Any], key: str, path: str) -> str:
    value = _require_value(data, key, path)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Expected non-empty string for field: {path}.{key}")
    return value


def _require_int(data: Dict[str, Any], key: str, path: str) -> int:
    value = _require_value(data, key, path)
    if not isinstance(value, int):
        raise ConfigError(f"Expected integer for field: {path}.{key}")
    return value


def _require_number(data: Dict[str, Any], key: str, path: str) -> float:
    value = _require_value(data, key, path)
    if not isinstance(value, (int, float)):
        raise ConfigError(f"Expected number for field: {path}.{key}")
    return float(value)


def _require_bool(data: Dict[str, Any], key: str, path: str) -> bool:
    value = _require_value(data, key, path)
    if not isinstance(value, bool):
        raise ConfigError(f"Expected boolean for field: {path}.{key}")
    return value


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    transform_model: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    batch_size: int


@dataclass(frozen=True)
class VectorStoreConfig:
    provider: str
    persist_directory: str
    collection_name: str


@dataclass(frozen=True)
class StorageConfig:
    sqlite_db_path: str


@dataclass(frozen=True)
class IngestionConfig:
    chunk_size: int
    chunk_overlap: int
    splitter: str
    enable_transform: bool
    postprocessor_max_retries: int
    vector_store_hash_limit: int


@dataclass(frozen=True)
class RetrievalConfig:
    similarity_search_k: int


@dataclass(frozen=True)
class Config:
    llm: LLMConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    storage: StorageConfig
    ingestion: IngestionConfig
    retrieval: RetrievalConfig
    # API keys (loaded from env vars, not from YAML)
    openai_api_key: Optional[str] = None
    langsmith_api_key: Optional[str] = None
    langsmith_tracing: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        if not isinstance(data, dict):
            raise ConfigError("Config root must be a mapping")

        llm = _require_mapping(data, "llm", "config")
        embedding = _require_mapping(data, "embedding", "config")
        vector_store = _require_mapping(data, "vector_store", "config")
        storage = _require_mapping(data, "storage", "config")
        ingestion = _require_mapping(data, "ingestion", "config")
        retrieval = _require_mapping(data, "retrieval", "config")

        # Load API keys from environment variables (override YAML)
        openai_api_key = os.getenv("OPENAI_API_KEY") or data.get("api_keys", {}).get("openai_api_key")
        langsmith_api_key = os.getenv("LANGSMITH_API_KEY") or data.get("api_keys", {}).get("langsmith_api_key")
        langsmith_tracing = (
            os.getenv("LANGSMITH_TRACING", "").lower() == "true"
            or data.get("langsmith", {}).get("tracing", False)
        )

        # Resolve relative paths to absolute paths
        vector_store_persist_dir = vector_store.get("persist_directory", "")
        if vector_store_persist_dir and not Path(vector_store_persist_dir).is_absolute():
            vector_store_persist_dir = str(REPO_ROOT / vector_store_persist_dir)

        storage_db_path = storage.get("sqlite_db_path", "")
        if storage_db_path and not Path(storage_db_path).is_absolute():
            storage_db_path = str(REPO_ROOT / storage_db_path)

        config = cls(
            llm=LLMConfig(
                provider=_require_str(llm, "provider", "llm"),
                model=_require_str(llm, "model", "llm"),
                transform_model=llm.get("transform_model") or _require_str(llm, "model", "llm"),
                temperature=_require_number(llm, "temperature", "llm"),
                max_tokens=_require_int(llm, "max_tokens", "llm"),
            ),
            embedding=EmbeddingConfig(
                provider=_require_str(embedding, "provider", "embedding"),
                model=_require_str(embedding, "model", "embedding"),
                batch_size=_require_int(embedding, "batch_size", "embedding"),
            ),
            vector_store=VectorStoreConfig(
                provider=_require_str(vector_store, "provider", "vector_store"),
                persist_directory=vector_store_persist_dir or _require_str(vector_store, "persist_directory", "vector_store"),
                collection_name=_require_str(vector_store, "collection_name", "vector_store"),
            ),
            storage=StorageConfig(
                sqlite_db_path=storage_db_path or _require_str(storage, "sqlite_db_path", "storage"),
            ),
            ingestion=IngestionConfig(
                chunk_size=_require_int(ingestion, "chunk_size", "ingestion"),
                chunk_overlap=_require_int(ingestion, "chunk_overlap", "ingestion"),
                splitter=_require_str(ingestion, "splitter", "ingestion"),
                enable_transform=_require_bool(ingestion, "enable_transform", "ingestion"),
                postprocessor_max_retries=_require_int(ingestion, "postprocessor_max_retries", "ingestion"),
                vector_store_hash_limit=_require_int(ingestion, "vector_store_hash_limit", "ingestion"),
            ),
            retrieval=RetrievalConfig(
                similarity_search_k=_require_int(retrieval, "similarity_search_k", "retrieval"),
            ),
            openai_api_key=openai_api_key,
            langsmith_api_key=langsmith_api_key,
            langsmith_tracing=langsmith_tracing,
        )

        return config


def validate_config(config: Config) -> None:
    """Validate config and raise ConfigError if invalid."""

    if not config.llm.provider:
        raise ConfigError("Missing required field: llm.provider")
    if not config.embedding.provider:
        raise ConfigError("Missing required field: embedding.provider")
    if not config.vector_store.provider:
        raise ConfigError("Missing required field: vector_store.provider")
    if config.ingestion.chunk_size <= 0:
        raise ConfigError("ingestion.chunk_size must be positive")
    if config.ingestion.chunk_overlap < 0:
        raise ConfigError("ingestion.chunk_overlap must be non-negative")
    if config.retrieval.similarity_search_k <= 0:
        raise ConfigError("retrieval.similarity_search_k must be positive")


def load_config(path: str | Path | None = None) -> Config:
    """Load config from a YAML file and validate required fields.

    Args:
        path: Path to config YAML.  Defaults to
            ``<repo>/config/config.yaml`` (absolute, CWD-independent).
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = resolve_path(config_path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    config = Config.from_dict(data or {})
    validate_config(config)
    return config

