# orchestrator.py
"""
Conversational orchestration layer — the "session loop" from your Phase 2
architecture doc. This is deliberately a plain Python function, not a
LangGraph node: LangGraph fits a fixed "run once per report" pipeline,
while a chat loop needs to branch per-message in ways that would mean
introducing graph cycles for no real benefit.

Per incoming message, this decides one of two paths:
  1. New research request -> runs your existing Atlas -> Pathfinder ->
     Harvester -> Synthesizer -> Scribe pipeline unchanged.
  2. Follow-up question -> answered directly: ChromaDB first (this user's
     conversation history + the shared research-facts memory), and only
     falls back to a live Tavily search if memory has nothing relevant.

Nothing about the existing milestone-3 pipeline (main.py, the agents,
Vault) is modified here — this module only calls into it.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from vault import Vault
from agents.atlas import atlas_route
from agents.pathfinder import pathfinder_plan, PathfinderError
from agents.harvester import (
    harvester_web_search,
    harvester_parse_single,
    harvester_embed_multi,
    HarvesterError,
    tavily,
)
from agents.synthesizer import synthesizer_run, SynthesizerError
from agents.scribe import scribe_write, ScribeError

import conversation_memory as convo_mem

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Below this relevance score, a memory hit is treated as not relevant
# enough to answer a follow-up from — same threshold Synthesizer uses,
# for consistency across the project.
FOLLOW_UP_MEMORY_THRESHOLD = 0.5
FOLLOW_UP_MEMORY_K = 5

# How much of each recalled memory snippet to include in the prompt.
CONTEXT_SNIPPET_CHAR_LIMIT = 800

# How many recent chat turns to include for conversational continuity.
RECENT_HISTORY_TURNS = 6
# Older turns get a modest allowance...
OLDER_TURN_CHAR_LIMIT = 1000
# ...but the single most recent turn gets a generous one. Follow-ups very
# often refer directly to "the above" / "that report" — i.e. the last
# thing said, not something semantic search happens to score highly. A
# full research report can run 5,000-8,000+ characters, so a small limit
# here silently cuts off the exact content being asked about.
LAST_TURN_CHAR_LIMIT = 8000

# Typing one of these prefixes mid-conversation forces a fresh research run
# instead of a follow-up answer, e.g. "research: quantum computing".
RESEARCH_TRIGGER_PREFIXES = ("research:", "topic:", "analyze:")


class OrchestratorError(Exception):
    """Raised when the orchestration layer itself fails (not a specific
    agent — those raise their own named errors, caught and surfaced here)."""
    pass


# =================================================
# ROUTING: research request vs follow-up
# =================================================
def _is_new_research_request(message, prior_history):
    """The first message in a conversation is always treated as a research
    request. After that, only an explicit prefix triggers a fresh pipeline
    run — otherwise the message is answered as a follow-up. This is a
    simple, predictable rule rather than LLM-based intent classification,
    kept deliberately lightweight for now."""
    if not prior_history:
        return True
    return message.strip().lower().startswith(RESEARCH_TRIGGER_PREFIXES)


def _strip_trigger_prefix(message):
    stripped = message.strip()
    lowered = stripped.lower()
    for prefix in RESEARCH_TRIGGER_PREFIXES:
        if lowered.startswith(prefix):
            return stripped[len(prefix):].strip()
    return stripped


# =================================================
# PATH 1: full research pipeline (unchanged agents)
# =================================================
def _run_research_pipeline(input_data):
    """input_data: topic string, or a list of one/more PDF paths — same
    contract as main.py's --topic / --docs args. Reuses your existing
    agent functions exactly as they are for the milestone-3 pipeline."""
    vault = Vault()
    vault.input_data = input_data

    atlas_route(vault)
    pathfinder_plan(vault)

    if vault.mode == "topic":
        harvester_web_search(vault)
    elif vault.mode == "single_doc":
        harvester_parse_single(vault)
    else:
        harvester_embed_multi(vault)

    synthesizer_run(vault)
    scribe_write(vault)

    return vault.final_report


# =================================================
# PATH 2: follow-up answer (ChromaDB-first, Tavily fallback)
# =================================================
def _answer_follow_up(user_id, session_id, message, prior_history):
    # A throwaway Vault instance here is just a way to reach the shared
    # research-facts long-term memory — it never runs the pipeline.
    memory_vault = Vault()

    convo_hits = [
        h for h in convo_mem.query_conversation_memory(
            user_id, message, session_id=session_id, k=FOLLOW_UP_MEMORY_K
        )
        if h["score"] >= FOLLOW_UP_MEMORY_THRESHOLD
    ]
    research_hits = [
        h for h in memory_vault.query_long_term_memory(message, k=FOLLOW_UP_MEMORY_K)
        if h["score"] >= FOLLOW_UP_MEMORY_THRESHOLD
    ]

    used_fallback = False
    if not convo_hits and not research_hits:
        # Nothing relevant in memory — fall back to a live web search
        # rather than making something up.
        used_fallback = True
        try:
            search_results = tavily.search(query=message, max_results=3)
            result_list = search_results.get("results", []) if isinstance(search_results, dict) else []
        except Exception as e:
            print(f"[Orchestrator] Tavily fallback failed: {e}")
            result_list = []

        fallback_texts, fallback_metadatas = [], []
        for r in result_list:
            content, url = r.get("content"), r.get("url")
            if content and url:
                fallback_texts.append(content)
                fallback_metadatas.append({"source": url, "mode": "followup"})

        if fallback_texts:
            # Persist so the next follow-up (this user's or another's) on
            # a related question doesn't need another Tavily call.
            memory_vault.save_to_long_term_memory(fallback_texts, fallback_metadatas)
            research_hits = [
                {"text": t, "source": m["source"], "score": 1.0}
                for t, m in zip(fallback_texts, fallback_metadatas)
            ]

    context_parts = [f"- (earlier in this chat) {h['content'][:CONTEXT_SNIPPET_CHAR_LIMIT]}" for h in convo_hits]
    context_parts += [f"- (research memory, source: {h['source']}) {h['text'][:CONTEXT_SNIPPET_CHAR_LIMIT]}" for h in research_hits]
    context_block = "\n".join(context_parts) if context_parts else "No directly relevant prior context found via semantic search."

    recent_turns = prior_history[-RECENT_HISTORY_TURNS:]
    history_lines = []
    for i, turn in enumerate(recent_turns):
        is_most_recent = (i == len(recent_turns) - 1)
        char_limit = LAST_TURN_CHAR_LIMIT if is_most_recent else OLDER_TURN_CHAR_LIMIT
        history_lines.append(f"{turn['role']}: {turn['content'][:char_limit]}")
    history_block = "\n".join(history_lines) or "(no earlier messages)"

    fallback_note = " (a live web search was also run since memory had nothing relevant)" if used_fallback else ""
    prompt = f"""You are a helpful research assistant continuing an ongoing conversation.

Recent conversation:
{history_block}

Relevant context recalled from memory{fallback_note}:
{context_block}

User's new message: {message}

Answer conversationally and directly. If the context doesn't actually answer
the question, say so honestly rather than guessing."""

    try:
        response = llm.invoke(prompt)
    except Exception as e:
        raise OrchestratorError(f"Groq LLM call failed while answering follow-up: {e}") from e

    content = getattr(response, "content", None)
    if not content or not content.strip():
        raise OrchestratorError("Follow-up LLM returned an empty response")

    return content


# =================================================
# ENTRY POINT — call this from Streamlit
# =================================================
def handle_message(user_id, session_id, message, file_paths=None):
    """Main entry point for one chat turn.
    - message: the user's typed text
    - file_paths: optional list of uploaded PDF paths. If given, always
      triggers the research pipeline in single/multi-doc mode, regardless
      of the text-based routing rule.
    Returns the assistant's reply text. Saves both the user's message and
    the assistant's reply to conversation_memory before returning.
    """
    if not file_paths and (not message or not message.strip()):
        raise OrchestratorError("Empty message")

    display_message = message.strip() if message else f"[Uploaded {len(file_paths)} document(s)]"
    prior_history = convo_mem.get_conversation_history(user_id, session_id)

    trigger_research = bool(file_paths) or _is_new_research_request(message or "", prior_history)

    convo_mem.save_turn(user_id, session_id, "user", display_message)

    try:
        if trigger_research:
            input_data = file_paths if file_paths else _strip_trigger_prefix(message)
            reply = _run_research_pipeline(input_data)
        else:
            reply = _answer_follow_up(user_id, session_id, message, prior_history)
    except (PathfinderError, HarvesterError, SynthesizerError, ScribeError, ValueError) as e:
        reply = f"I couldn't complete that research request: {e}"
    except FileNotFoundError as e:
        reply = f"I couldn't find that file: {e}"
    except OrchestratorError as e:
        reply = f"Something went wrong answering that: {e}"

    convo_mem.save_turn(user_id, session_id, "assistant", reply)
    return reply