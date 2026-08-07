# auth_db.py
"""
Permanent storage for user accounts and conversation records.

SQLite — free, file-based, zero external services — holds two tables:
  users:          hashed credentials
  conversations:  per-user session metadata (title, timestamps)

Passwords are hashed with PBKDF2-HMAC-SHA256 (stdlib hashlib, no extra
dependency) using a random per-user salt. Plaintext passwords are never
stored or logged.

This module is intentionally independent of Vault/agents — it only manages
"who is this user" and "which conversations do they own". The actual
conversation content (chat turns) lives in ChromaDB, added in Phase B.
"""
import sqlite3
import hashlib
import os
import uuid
from datetime import datetime, timedelta
from contextlib import contextmanager

DB_PATH = "users.db"

PBKDF2_ITERATIONS = 200_000
SALT_BYTES = 16
SESSION_EXPIRY_DAYS = 7


class AuthError(Exception):
    """Raised for user-facing auth failures (bad credentials, duplicate username, etc.)."""
    pass


@contextmanager
def _get_connection(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path=DB_PATH):
    """Create tables if they don't already exist. Safe to call on every app start."""
    with _get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                pinned INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # Migration: DBs created before "pinned" existed won't have the
        # column — CREATE TABLE IF NOT EXISTS above is a no-op for them,
        # so add it explicitly if missing.
        existing_cols = [row["name"] for row in conn.execute("PRAGMA table_info(conversations)")]
        if "pinned" not in existing_cols:
            conn.execute("ALTER TABLE conversations ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")


# =================================================
# PASSWORD HASHING
# =================================================
def _hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(SALT_BYTES).hex()
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        PBKDF2_ITERATIONS
    ).hex()
    return digest, salt


# =================================================
# USER ACCOUNTS
# =================================================
def register_user(username, password, db_path=DB_PATH):
    """Create a new user account. Raises AuthError on bad input or duplicate username."""
    username = username.strip()
    if not username or not password:
        raise AuthError("Username and password are required")
    if len(password) < 6:
        raise AuthError("Password must be at least 6 characters")

    password_hash, salt = _hash_password(password)

    with _get_connection(db_path) as conn:
        existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            raise AuthError(f"Username '{username}' is already taken")

        conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username, password_hash, salt, datetime.now().isoformat(timespec="seconds"))
        )
        user_id = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()["id"]

    return user_id


def authenticate_user(username, password, db_path=DB_PATH):
    """Verify credentials. Returns user_id on success, raises AuthError on failure."""
    with _get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT id, password_hash, salt FROM users WHERE username = ?",
            (username.strip(),)
        ).fetchone()

    if not row:
        raise AuthError("Invalid username or password")

    computed_hash, _ = _hash_password(password, salt=row["salt"])
    if computed_hash != row["password_hash"]:
        raise AuthError("Invalid username or password")

    return row["id"]


# =================================================
# PERSISTENT LOGIN (survives a browser refresh)
# =================================================
def create_session(user_id, db_path=DB_PATH, expiry_days=SESSION_EXPIRY_DAYS):
    """Create a login token stored server-side (sessions table) and meant
    to be echoed back in the URL via st.query_params. A page refresh keeps
    the URL, so the app can look this token up and silently restore login
    instead of dropping back to the login screen."""
    token = uuid.uuid4().hex
    now = datetime.now()
    expires_at = (now + timedelta(days=expiry_days)).isoformat(timespec="seconds")
    with _get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now.isoformat(timespec="seconds"), expires_at)
        )
    return token


def validate_session(token, db_path=DB_PATH):
    """Returns {'user_id':, 'username':} if the token is valid and not
    expired, otherwise None (expired tokens are cleaned up automatically)."""
    if not token:
        return None
    with _get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT sessions.user_id AS user_id, users.username AS username, sessions.expires_at AS expires_at "
            "FROM sessions JOIN users ON sessions.user_id = users.id "
            "WHERE sessions.token = ?",
            (token,)
        ).fetchone()

    if not row:
        return None
    if row["expires_at"] < datetime.now().isoformat(timespec="seconds"):
        delete_session(token, db_path)
        return None

    return {"user_id": row["user_id"], "username": row["username"]}


def delete_session(token, db_path=DB_PATH):
    with _get_connection(db_path) as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


# =================================================
# CONVERSATIONS (metadata only — content lives in ChromaDB, Phase B)
# =================================================
def create_conversation(user_id, title="New Chat", db_path=DB_PATH):
    """Start a new conversation record for a user. Returns the new session_id."""
    session_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")
    with _get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO conversations (session_id, user_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, title, now, now)
        )
    return session_id


def list_conversations(user_id, db_path=DB_PATH):
    """Return this user's conversations — pinned first, then most recently
    updated within each group."""
    with _get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT session_id, title, pinned, created_at, updated_at FROM conversations "
            "WHERE user_id = ? ORDER BY pinned DESC, updated_at DESC",
            (user_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def set_conversation_pinned(session_id, pinned, db_path=DB_PATH):
    with _get_connection(db_path) as conn:
        conn.execute(
            "UPDATE conversations SET pinned = ? WHERE session_id = ?",
            (1 if pinned else 0, session_id)
        )


def touch_conversation(session_id, db_path=DB_PATH):
    """Update a conversation's last-updated timestamp — call after each message
    so the sidebar ordering reflects recent activity."""
    with _get_connection(db_path) as conn:
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE session_id = ?",
            (datetime.now().isoformat(timespec="seconds"), session_id)
        )


def rename_conversation(session_id, new_title, db_path=DB_PATH):
    with _get_connection(db_path) as conn:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE session_id = ?",
            (new_title, session_id)
        )


def delete_conversation(session_id, db_path=DB_PATH):
    with _get_connection(db_path) as conn:
        conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))