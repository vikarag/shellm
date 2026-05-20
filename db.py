"""Shared SQLite connection for memory_manager and rag_engine.

The database is created on first use. Schema covers `memories` (with FTS5)
and `rag_docs`/`rag_chunks` (with FTS5 over chunk text). Triggers keep the
FTS shadow tables in sync.
"""

import os
import sqlite3
import threading

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shellm.db")
_conn = None
_lock = threading.Lock()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    source      TEXT    NOT NULL DEFAULT 'unknown',
    content     TEXT    NOT NULL,
    tags        TEXT    NOT NULL DEFAULT '[]',
    archived    INTEGER NOT NULL DEFAULT 0,
    archived_at TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TABLE IF NOT EXISTS rag_docs (
    doc_id      TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    timestamp   TEXT NOT NULL,
    tags        TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS rag_chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES rag_docs(doc_id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5(
    text,
    content='rag_chunks',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS rag_chunks_ai AFTER INSERT ON rag_chunks BEGIN
    INSERT INTO rag_chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS rag_chunks_ad AFTER DELETE ON rag_chunks BEGIN
    INSERT INTO rag_chunks_fts(rag_chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS rag_chunks_au AFTER UPDATE ON rag_chunks BEGIN
    INSERT INTO rag_chunks_fts(rag_chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO rag_chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


def get_connection() -> sqlite3.Connection:
    """Return the process-wide connection, creating and initializing it lazily."""
    global _conn
    if _conn is not None:
        return _conn
    with _lock:
        if _conn is None:
            conn = sqlite3.connect(_DB_PATH, check_same_thread=False, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(_SCHEMA)
            _conn = conn
    return _conn
