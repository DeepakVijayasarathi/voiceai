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
                image BLOB,
                audio BLOB,
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
                avatar BLOB,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS story_characters (
                story_id INTEGER NOT NULL REFERENCES stories(id),
                character_id INTEGER NOT NULL REFERENCES characters(id),
                PRIMARY KEY (story_id, character_id)
            );
            CREATE TABLE IF NOT EXISTS story_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                story_id INTEGER NOT NULL REFERENCES stories(id),
                scene_index INTEGER NOT NULL,
                start_time_s REAL NOT NULL,
                end_time_s REAL NOT NULL,
                excerpt TEXT,
                prompt TEXT,
                url TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS quality_backlog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                story_id INTEGER NOT NULL REFERENCES stories(id),
                axis_scores_json TEXT NOT NULL,
                rewritten INTEGER NOT NULL DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );
        """)
        conn.commit()
        # CREATE TABLE IF NOT EXISTS above is a no-op against an existing
        # DB file from before these columns existed - add them explicitly
        # so upgrading doesn't require dropping saved stories/characters.
        existing_char_columns = {row["name"] for row in conn.execute("PRAGMA table_info(characters)")}
        if "voice_profile" not in existing_char_columns:
            conn.execute("ALTER TABLE characters ADD COLUMN voice_profile TEXT")
            conn.commit()
        existing_story_columns = {row["name"] for row in conn.execute("PRAGMA table_info(stories)")}
        if "image" not in existing_story_columns:
            conn.execute("ALTER TABLE stories ADD COLUMN image BLOB")
            conn.commit()
        if "audio" not in existing_story_columns:
            conn.execute("ALTER TABLE stories ADD COLUMN audio BLOB")
            conn.commit()
        if "variant_of_story_id" not in existing_story_columns:
            # Language-switching feature: a story generated as a same-
            # story/same-characters adaptation into another language
            # points back at the story it's a variant of (see
            # story_gen.adapt_story_language, POST /stories/{id}/switch-language).
            conn.execute("ALTER TABLE stories ADD COLUMN variant_of_story_id INTEGER REFERENCES stories(id)")
            conn.commit()
        if "avatar" not in existing_char_columns:
            conn.execute("ALTER TABLE characters ADD COLUMN avatar BLOB")
            conn.commit()


def save_story(
    title: str, language: str, text: str, parent_story_id: int | None = None, branch_note: str | None = None,
    variant_of_story_id: int | None = None,
) -> int:
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "INSERT INTO stories (title, language, text, parent_story_id, branch_note, variant_of_story_id) VALUES (?, ?, ?, ?, ?, ?)",
            (title, language, text, parent_story_id, branch_note, variant_of_story_id),
        )
        conn.commit()
        return cur.lastrowid


def get_story(story_id: int) -> dict | None:
    """Excludes the raw image/audio BLOBs (use get_story_image/
    get_story_audio for those) - a JSON story record shouldn't carry
    multi-hundred-KB payloads inline; has_image/has_audio tell the caller
    whether they exist (has_audio also means GET .../video can compose a
    video from them)."""
    conn = _get_conn()
    with _lock:
        row = conn.execute(
            """
            SELECT id, title, language, text, parent_story_id, branch_note,
                   variant_of_story_id, created_at, (image IS NOT NULL) AS has_image,
                   (audio IS NOT NULL) AS has_audio
            FROM stories WHERE id = ?
            """,
            (story_id,),
        ).fetchone()
    return dict(row) if row else None


def list_stories(limit: int = 50) -> list[dict]:
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            """
            SELECT id, title, language, parent_story_id, branch_note, variant_of_story_id,
                   created_at, (image IS NOT NULL) AS has_image, (audio IS NOT NULL) AS has_audio
            FROM stories ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def save_story_image(story_id: int, image_bytes: bytes) -> None:
    """Sets the story's single AI-generated cover/scene image (see
    image_gen.py, POST /dream-to-story's visual=true) - NOT the same
    thing as save_scene_image below, which records one of several
    time-mapped storyboard images for a story (see images.py)."""
    conn = _get_conn()
    with _lock:
        conn.execute("UPDATE stories SET image = ? WHERE id = ?", (image_bytes, story_id))
        conn.commit()


def get_story_image(story_id: int) -> bytes | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT image FROM stories WHERE id = ?", (story_id,)).fetchone()
    return bytes(row["image"]) if row and row["image"] is not None else None


def save_story_audio(story_id: int, audio_bytes: bytes) -> None:
    """Only populated when a caller requested video composition (see
    app.py's /dream-to-story `video` field) - routine narration-only
    requests don't persist audio, to avoid bloating the DB with
    hundred-KB-to-MB blobs nobody asked to keep."""
    conn = _get_conn()
    with _lock:
        conn.execute("UPDATE stories SET audio = ? WHERE id = ?", (audio_bytes, story_id))
        conn.commit()


def get_story_audio(story_id: int) -> bytes | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT audio FROM stories WHERE id = ?", (story_id,)).fetchone()
    return bytes(row["audio"]) if row and row["audio"] is not None else None


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
    """Excludes the raw avatar BLOB (use get_character_avatar for that) -
    same has_x reporting pattern as get_story's has_image/has_audio."""
    conn = _get_conn()
    with _lock:
        row = conn.execute(
            """
            SELECT id, name, personality, backstory, language, origin_story_id,
                   voice_profile, created_at, (avatar IS NOT NULL) AS has_avatar
            FROM characters WHERE id = ?
            """,
            (character_id,),
        ).fetchone()
    return dict(row) if row else None


def list_characters(limit: int = 50) -> list[dict]:
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            """
            SELECT id, name, personality, language, origin_story_id, voice_profile,
                   created_at, (avatar IS NOT NULL) AS has_avatar
            FROM characters ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def save_character_avatar(character_id: int, image_bytes: bytes) -> None:
    conn = _get_conn()
    with _lock:
        conn.execute("UPDATE characters SET avatar = ? WHERE id = ?", (image_bytes, character_id))
        conn.commit()


def get_character_avatar(character_id: int) -> bytes | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT avatar FROM characters WHERE id = ?", (character_id,)).fetchone()
    return bytes(row["avatar"]) if row and row["avatar"] is not None else None


def save_scene_image(
    story_id: int, scene_index: int, start_time_s: float, end_time_s: float,
    excerpt: str, prompt: str, url: str,
) -> int:
    """Records one of several time-mapped storyboard images for a story
    (see images.py, POST /tts or /dream-to-story's images=true) - distinct
    from save_story_image above, which is a single cover image stored
    inline as a BLOB. `excerpt` is the native-script story text this scene
    came from (used as a comic panel's caption - see
    export.build_comic_panels); `prompt` is the English image-generation
    prompt (kept for reference/debugging, not shown to a reader)."""
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "INSERT INTO story_images (story_id, scene_index, start_time_s, end_time_s, excerpt, prompt, url) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (story_id, scene_index, start_time_s, end_time_s, excerpt, prompt, url),
        )
        conn.commit()
        return cur.lastrowid


def get_images_for_story(story_id: int) -> list[dict]:
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT id, scene_index, start_time_s, end_time_s, excerpt, prompt, url FROM story_images WHERE story_id = ? ORDER BY scene_index",
            (story_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def save_quality_entry(story_id: int, axis_scores: dict, rewritten: bool) -> int:
    """Records a story whose final (post-rewrite-attempt, if any) quality
    score was still below quality.QUALITY_THRESHOLD - a visible backlog of
    weak spots rather than a log of every generation (most stories score
    fine and never land here)."""
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "INSERT INTO quality_backlog (story_id, axis_scores_json, rewritten) VALUES (?, ?, ?)",
            (story_id, json.dumps(axis_scores), int(rewritten)),
        )
        conn.commit()
        return cur.lastrowid


def list_quality_backlog(limit: int = 50) -> list[dict]:
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT id, story_id, axis_scores_json, rewritten, created_at FROM quality_backlog ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    results = []
    for row in rows:
        entry = dict(row)
        entry["axis_scores"] = json.loads(entry.pop("axis_scores_json"))
        entry["rewritten"] = bool(entry["rewritten"])
        results.append(entry)
    return results


def get_variants(story_id: int) -> list[dict]:
    """Other-language adaptations of this exact story (see
    story_gen.adapt_story_language) - lets a client offer 'listen in...'
    switching between them. Does not include story_id itself."""
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT id, title, language, created_at FROM stories WHERE variant_of_story_id = ? ORDER BY id",
            (story_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_child_branches(story_id: int) -> list[dict]:
    """Direct branches of this story (see POST /stories/{id}/branch) - used
    to build a navigable choice-tree (see export.build_game_tree)."""
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT id, title, branch_note FROM stories WHERE parent_story_id = ? ORDER BY id",
            (story_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_story_lineage(story_id: int) -> list[dict]:
    """Walks parent_story_id back to the root, returning ancestors OLDER
    than story_id itself (root first, story_id's direct parent last) -
    story_id's own text isn't included, since callers already have that.
    Used to give /stories/{id}/branch a condensed sense of a branch's full
    history, not just its one direct parent (see story_gen.
    generate_continuation's ancestor_summaries param). Each get_story()
    call below fully acquires and releases _lock before the next, rather
    than holding it across the walk - _lock is a plain (non-reentrant)
    threading.Lock, so nesting acquisitions here would deadlock."""
    ancestors = []
    current = get_story(story_id)
    while current and current.get("parent_story_id"):
        current = get_story(current["parent_story_id"])
        if not current:
            break
        ancestors.append(current)
    ancestors.reverse()
    return ancestors


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
