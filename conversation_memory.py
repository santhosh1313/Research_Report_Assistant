# conversation_memory.py
"""
Per-user, per-conversation chat memory, stored in ChromaDB — kept in a
separate collection ("conversation_history") from Vault's research-facts
collection ("research_assistant") so a user's chat turns never mix with
research content, and one user's turns never mix with another's.

Every chat turn (a user message or an assistant reply) is embedded and
stored tagged with user_id + session_id. Two read paths are provided:
  - get_conversation_history(): the full ordered transcript of one
    conversation (used to reconstruct chat history on load / refresh)
  - query_conversation_memory(): semantic search over a user's past
    turns (used by the orchestration layer to recall relevant context
    when answering a follow-up)
"""
import os
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "conversation_history"


class ConversationMemoryError(Exception):
    """Raised for conversation-memory failures the caller explicitly needs
    to know about. Most functions here fail soft (log + return empty),
    matching Vault's long-term-memory error handling."""
    pass


_embeddings = None
_store = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    return _embeddings


def _get_store():
    global _store
    if _store is None:
        _store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=_get_embeddings(),
            persist_directory=PERSIST_DIRECTORY
        )
    return _store


def save_turn(user_id, session_id, role, content):
    """Persist one chat turn. role is 'user' or 'assistant'."""
    if not content or not content.strip():
        return
    try:
        store = _get_store()
        store.add_texts(
            texts=[content],
            metadatas=[{
                "user_id": str(user_id),
                "session_id": session_id,
                "role": role,
                "timestamp": datetime.now().isoformat(timespec="seconds")
            }]
        )
    except Exception as e:
        print(f"[ConversationMemory] Failed to save turn: {e}")


def get_conversation_history(user_id, session_id, limit=100):
    """Return the full transcript of one conversation, in chronological
    order. Uses a metadata filter (not similarity search) since we want
    every turn in order, not a relevance-ranked subset."""
    try:
        store = _get_store()
        results = store.get(
            where={
                "$and": [
                    {"user_id": str(user_id)},
                    {"session_id": session_id}
                ]
            }
        )
    except Exception as e:
        print(f"[ConversationMemory] Failed to load conversation history: {e}")
        return []

    docs = results.get("documents", []) or []
    metas = results.get("metadatas", []) or []

    turns = [
        {
            "role": meta.get("role"),
            "content": doc,
            "timestamp": meta.get("timestamp", "")
        }
        for doc, meta in zip(docs, metas)
    ]
    turns.sort(key=lambda t: t["timestamp"])
    return turns[-limit:]


def query_conversation_memory(user_id, query, session_id=None, k=5):
    """Semantic search over a user's conversation history — scoped to one
    session if session_id is given, otherwise across all of that user's
    conversations. Used by the orchestration layer to recall relevant past
    exchanges when answering a follow-up question."""
    if not query:
        return []

    where_filter = {"user_id": str(user_id)}
    if session_id:
        where_filter = {"$and": [{"user_id": str(user_id)}, {"session_id": session_id}]}

    try:
        store = _get_store()
        results = store.similarity_search_with_relevance_scores(query, k=k, filter=where_filter)
    except Exception as e:
        print(f"[ConversationMemory] Query failed: {e}")
        return []

    return [
        {
            "content": doc.page_content,
            "role": doc.metadata.get("role"),
            "session_id": doc.metadata.get("session_id"),
            "score": round(score, 3)
        }
        for doc, score in results
    ]


def delete_conversation_memory(user_id, session_id):
    """Remove all stored turns for one conversation — call this alongside
    auth_db.delete_conversation() so ChromaDB and the permanent DB stay
    in sync when a user deletes a chat."""
    try:
        store = _get_store()
        store.delete(
            where={
                "$and": [
                    {"user_id": str(user_id)},
                    {"session_id": session_id}
                ]
            }
        )
    except Exception as e:
        print(f"[ConversationMemory] Failed to delete conversation memory: {e}")