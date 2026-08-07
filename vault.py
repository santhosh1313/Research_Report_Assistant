# vault.py
import os
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


class VaultMemoryError(Exception):
    """Raised when a long-term memory (ChromaDB) operation fails in a way
    the caller explicitly needs to know about. Most call sites treat
    long-term memory as best-effort and won't raise this — see the
    try/except handling inside save_to_long_term_memory / query_long_term_memory."""
    pass


class Vault:

    def __init__(self):

        # Input
        self.mode = None
        self.input_data = None

        # ---- Short-term memory (run-scoped, in-memory, cleared each run) ----
        self.subtasks = []            # from Pathfinder
        self.facts = []               # from Harvester: [{text, source}]

        # ---- Long-term memory (persists across runs via ChromaDB) ----
        # Vault is the interface every agent talks to; ChromaDB is the
        # storage detail underneath it, lazily initialized on first use so
        # a run that never touches memory doesn't pay any embedding-API cost.
        self._embeddings = None
        self._vector_store = None
        self.persist_directory = "./chroma_db"
        self.collection_name = "research_assistant"
        self.retrieved_memory = []    # results of the most recent long-term query

        # ---- Agent coordination ----
        # A visible record of handoffs between agents. Agents don't message
        # each other directly (LangGraph edges + Vault state are what actually
        # move execution and data forward) — this log exists so that
        # coordination, which is otherwise implicit in shared-state mutation,
        # can be inspected and demonstrated after a run.
        self.comm_log = []

        # Outputs
        self.synthesis = None
        self.final_report = None

    # =================================================
    # SHORT-TERM MEMORY
    # =================================================
    def log_fact(self, text, source):
        self.facts.append(
            {
                "text": text,
                "source": source
            }
        )

    # =================================================
    # AGENT COORDINATION
    # =================================================
    def log_message(self, from_agent, to_agent, action, detail=""):
        """Record a handoff between agents. Called by an agent when it
        starts work it received from another agent, and again when it
        completes and hands off to the next one."""
        self.comm_log.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "from": from_agent,
                "to": to_agent,
                "action": action,
                "detail": detail
            }
        )

    # =================================================
    # LONG-TERM MEMORY (ChromaDB)
    # =================================================
    def _get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-2",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        return self._embeddings

    def _get_vector_store(self):
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._get_embeddings(),
                persist_directory=self.persist_directory
            )
        return self._vector_store

    def save_to_long_term_memory(self, texts, metadatas):
        """Embed and persist texts to ChromaDB so future runs (even in a
        different process) can recall them. Long-term memory is an
        enhancement, not a hard pipeline dependency — a failure here is
        logged and swallowed rather than raised, so a Gemini quota hiccup
        on the embeddings call never takes down an otherwise-good run."""
        if not texts:
            return
        try:
            store = self._get_vector_store()
            store.add_texts(texts=texts, metadatas=metadatas)
            print(f"[Vault] Persisted {len(texts)} item(s) to long-term memory")
        except Exception as e:
            print(f"[Vault] Failed to save to long-term memory: {e}")

    def query_long_term_memory(self, query, k=5):
        """Retrieve the top-k most relevant facts from prior runs, each with
        a relevance score. Returns [] (not an error) on an empty/missing
        store or a failed embedding call, since 'no prior memory yet' is
        the normal state for a first-ever run."""
        if not query:
            return []
        try:
            store = self._get_vector_store()
            results = store.similarity_search_with_relevance_scores(query, k=k)
        except Exception as e:
            print(f"[Vault] Long-term memory query failed: {e}")
            return []

        retrieved = [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": round(score, 3)
            }
            for doc, score in results
        ]
        self.retrieved_memory = retrieved
        return retrieved