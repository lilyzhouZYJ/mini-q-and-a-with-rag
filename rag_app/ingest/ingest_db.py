"""
Database module for tracking ingestion history.

We use SQLite database to record file hashes and ingestion status;
if a file has already been processed, we skip it.

We store below information in the database:
- file_hash: SHA256 hash of the file
- source_path: Path of the source file
- status: Status of ingestion ('success', 'failed', 'processing')
- processed_at: Timestamp of the ingestion
- chunk_count: Number of chunks created
"""

import sqlite3
from datetime import datetime
from typing import Optional
from pathlib import Path

def get_db_connection(db_path: str) -> sqlite3.Connection:
    """
    Get a connection to the SQLite database, creating it if needed.
    
    Args:
        db_path: Path to the SQLite database file
    """
    # Ensure directory exists
    db_path_obj = Path(db_path)
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    return conn

def _init_db(conn: sqlite3.Connection) -> None:
    """
    Initialize the database schema if it doesn't exist.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_history (
            file_hash TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            status TEXT NOT NULL,
            processed_at TIMESTAMP NOT NULL,
            chunk_count INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_source_path ON ingestion_history(source_path)
    """)
    conn.commit()

def check_if_file_hash_exists(file_hash: str, db_path: str) -> Optional[dict]:
    """
    Check if the file hash exists in the ingestion history.
    Return the ingestion record if found, None otherwise.
    
    Args:
        file_hash: SHA256 hash of the file
        db_path: Path to the SQLite database file
    """
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM ingestion_history WHERE file_hash = ?",
            (file_hash,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()

def record_ingestion(
    file_hash: str,
    source_path: str,
    status: str,
    db_path: str,
    chunk_count: int = 0
) -> None:
    """
    Record an ingestion attempt in the database.
    
    Args:
        file_hash: SHA256 hash of the file
        source_path: Path of the source file
        status: Status of ingestion ('success', 'failed', 'processing')
        db_path: Path to the SQLite database file
        chunk_count: Number of chunks created (default: 0)
    """
    conn = get_db_connection(db_path)
    try:
        conn.execute("""
            INSERT OR REPLACE INTO ingestion_history 
            (file_hash, source_path, status, processed_at, chunk_count)
            VALUES (?, ?, ?, ?, ?)
        """, (file_hash, source_path, status, datetime.now(), chunk_count))
        conn.commit()
    finally:
        conn.close()

