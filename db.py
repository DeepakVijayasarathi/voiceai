"""
Persistent storage for stories and characters - SQLite, file-based, no
external database service needed. This is the actual missing prerequisite
for "Infinite Story Universe" / "Character Resurrection" / "Story Time
Machine" style features: nothing about story or character identity
persisted anywhere in this system before this file existed.

Scope note: this stores story/character records and lets a new story
reference an existing character or branch from an existing story. It does
NOT attempt cross-story consistency enforcement (checking that a revived
character's established facts don't contradict a new story) - that would
need the new story's generation to be constrained against the full prior
canon, which needs a much larger context-management system than a single
OpenAI call. What's here is real and useful (persistence + retrieval +
generation seeded with prior context) without pretending to guarantee
consistency it can't check.
"""
import json
import os
import sqlite3
import threading

DB_PATH = os.environ.get("INDIC_TTS_DB_PATH", "/root/indic-tts-fast/api/stories.db")

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA foreign_keys = ON")
    return _conn


def init_db() -> None:
    conn = _get_conn()
    with _lock:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS stories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                language TEXT NOT NULL,
                text TEXT NOT NULL,
                parent_story_id INTEGER REFERENCES stories(id),
                branch_note TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                personality TEXT,
                backstory TEXT,
                language TEXT NOT NULL,
                origin_story_id INTEGER REFERENCES stories(id),
                voice_profile TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS story_characters (
                story_id INTEGER NOT NULL REFERENCES stories(id),
                character_id INTEGER NOT NULL REFERENCES characters(id),
                PRIMARY KEY (story_id, character_id)
            );
        """)
        conn.commit()
        # CREATE TABLE IF NOT EXISTS above is a no-op against an existing
        # DB file from before voice_profile existed - add it explicitly so
        # upgrading doesn't require dropping saved stories/characters.
        existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(characters)")}
        if "voice_profile" not in existing_columns:
            conn.execute("ALTER TABLE characters ADD COLUMN voice_profile TEXT")
            conn.commit()


def save_story(title: str, language: str, text: str, parent_story_id: int | None = None, branch_note: str | None = None) -> int:
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "INSERT INTO stories (title, language, text, parent_story_id, branch_note) VALUES (?, ?, ?, ?, ?)",
            (title, language, text, parent_story_id, branch_note),
        )
        conn.commit()
        return cur.lastrowid


def get_story(story_id: int) -> dict | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM stories WHERE id = ?", (story_id,)).fetchone()
    return dict(row) if row else None


def list_stories(limit: int = 50) -> list[dict]:
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT id, title, language, parent_story_id, branch_note, created_at FROM stories ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def save_character(
    name: str, personality: str, backstory: str, language: str,
    origin_story_id: int | None = None, voice_profile: str | None = None,
) -> int:
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "INSERT INTO characters (name, personality, backstory, language, origin_story_id, voice_profile) VALUES (?, ?, ?, ?, ?, ?)",
            (name, personality, backstory, language, origin_story_id, voice_profile),
        )
        conn.commit()
        return cur.lastrowid


def link_character_to_story(story_id: int, character_id: int) -> None:
    conn = _get_conn()
    with _lock:
        conn.execute(
            "INSERT OR IGNORE INTO story_characters (story_id, character_id) VALUES (?, ?)",
            (story_id, character_id),
        )
        conn.commit()


def get_character(character_id: int) -> dict | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM characters WHERE id = ?", (character_id,)).fetchone()
    return dict(row) if row else None


def list_characters(limit: int = 50) -> list[dict]:
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT id, name, personality, language, origin_story_id, voice_profile, created_at FROM characters ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_characters_for_story(story_id: int) -> list[dict]:
    """Characters linked to a story either as its origin (extracted from
    it) or via story_characters (e.g. revived into it) - used to give
    /stories/{id}/branch a known-character list for speaker/voice
    attribution on the branch's continuation."""
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            """
            SELECT DISTINCT c.id, c.name, c.personality, c.language, c.voice_profile
            FROM characters c
            WHERE c.origin_story_id = ?
               OR c.id IN (SELECT character_id FROM story_characters WHERE story_id = ?)
            """,
            (story_id, story_id),
        ).fetchall()
    return [dict(r) for r in rows]
