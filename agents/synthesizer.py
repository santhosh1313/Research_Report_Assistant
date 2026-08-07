# agents/synthesizer.py
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Below this relevance score, a long-term memory hit is treated as noise
# rather than genuinely related prior research and is dropped before it
# reaches the prompt.
LONG_TERM_MEMORY_RELEVANCE_THRESHOLD = 0.5
LONG_TERM_MEMORY_TOP_K = 5


class SynthesizerError(Exception):
    """Raised when synthesis fails or has no input material to work with."""
    pass


def _format_prior_context(prior_hits):
    lines = [
        f"- {p['text'][:400]} (Source: {p['source']}, relevance: {p['score']})"
        for p in prior_hits
    ]
    return "\n".join(lines)


def synthesizer_run(vault):
    if not vault.facts and vault.mode != "multi_doc":
        raise SynthesizerError("No facts available in Vault — Harvester may have failed silently upstream")

    facts_text = "\n".join([f"- {f['text'][:1500]} (Source: {f['source']})" for f in vault.facts])

    if vault.mode == "topic":
        # Long-term memory: pull in relevant findings from prior research runs
        # on related topics, so the report can build on past work instead of
        # only ever seeing this run's fresh Tavily results.
        prior_hits = [
            p for p in vault.query_long_term_memory(str(vault.input_data), k=LONG_TERM_MEMORY_TOP_K)
            if p["score"] >= LONG_TERM_MEMORY_RELEVANCE_THRESHOLD
        ]

        prompt = f"""Synthesize the following research facts into clear themes,
        noting any contradictions between sources. Keep every source reference intact.
        Facts:\n{facts_text}"""

        if prior_hits:
            prompt += f"""\n\nRelated findings recalled from prior research sessions
            (long-term memory — use only if genuinely relevant, and keep source
            references intact):\n{_format_prior_context(prior_hits)}"""

    elif vault.mode == "single_doc":
        # Query long-term memory using a snippet of this document as the query,
        # to surface prior papers/facts relevant to the same subject matter.
        memory_query = (vault.document_text or "")[:300]
        prior_hits = [
            p for p in vault.query_long_term_memory(memory_query, k=LONG_TERM_MEMORY_TOP_K)
            if p["score"] >= LONG_TERM_MEMORY_RELEVANCE_THRESHOLD
        ]

        prompt = f"""Extract the key findings, methodology, and limitations from
        this paper's content. Note whether the results support the claims made.
        Content:\n{facts_text[:6000]}"""

        if prior_hits:
            prompt += f"""\n\nRelated prior research recalled from long-term memory
            (use only for context, this paper's own content is the primary source):
            \n{_format_prior_context(prior_hits)}"""

    else:  # multi_doc
        # Synthesizer already has full context across all documents here, so
        # long-term memory retrieval isn't needed for this mode.
        dims = "\n".join(vault.subtasks)

        context = vault.multi_doc_text

        prompt = f"""
        Compare the following research papers across these dimensions:

        {dims}

        Research Papers:
        {context}

        Identify:

        - Similarities
        - Differences
        - Contradictions
        - Strengths
        - Weaknesses
        - Research gaps

        Mention the paper names whenever possible.
        """

    try:
        response = llm.invoke(prompt)
    except Exception as e:
        raise SynthesizerError(f"Gemini LLM call failed during synthesis: {e}") from e

    content = getattr(response, "content", None)
    if not content or not content.strip():
        raise SynthesizerError("Synthesizer LLM returned an empty response")

    vault.synthesis = content
    print("[Synthesizer] Synthesis complete")

    vault.log_message(
        from_agent="Synthesizer",
        to_agent="Scribe",
        action="synthesis_ready",
        detail=f"{len(content)} chars"
    )
    return vault.synthesis