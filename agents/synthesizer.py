# agents/synthesizer.py
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


class SynthesizerError(Exception):
    """Raised when synthesis fails or has no input material to work with."""
    pass


def synthesizer_run(vault):
    if not vault.facts and vault.mode != "multi_doc":
        raise SynthesizerError("No facts available in Vault — Harvester may have failed silently upstream")

    facts_text = "\n".join([f"- {f['text'][:1500]} (Source: {f['source']})" for f in vault.facts])

    if vault.mode == "topic":
        prompt = f"""Synthesize the following research facts into clear themes,
        noting any contradictions between sources. Keep every source reference intact.
        Facts:\n{facts_text}"""

    elif vault.mode == "single_doc":
        prompt = f"""Extract the key findings, methodology, and limitations from
        this paper's content. Note whether the results support the claims made.
        Content:\n{facts_text[:6000]}"""

    else:  # multi_doc
        if vault.vector_store is None:
            raise SynthesizerError("No vector store available for multi-doc synthesis — Harvester step likely failed")

        dims = "\n".join(vault.subtasks)
        try:
            retrieved = vault.vector_store.similarity_search(dims, k=8)
        except Exception as e:
            raise SynthesizerError(f"ChromaDB similarity_search failed: {e}") from e

        if not retrieved:
            raise SynthesizerError("similarity_search returned no chunks for the comparison dimensions")

        context = "\n".join([f"- {d.page_content[:300]} (Source: {d.metadata['source']})" for d in retrieved])
        prompt = f"""Compare the following paper excerpts across these dimensions:
        {dims}
        Identify agreements, contradictions, and gaps across papers.
        Excerpts:\n{context}"""

    try:
        response = llm.invoke(prompt)
    except Exception as e:
        raise SynthesizerError(f"Gemini LLM call failed during synthesis: {e}") from e

    content = getattr(response, "content", None)
    if not content or not content.strip():
        raise SynthesizerError("Synthesizer LLM returned an empty response")

    vault.synthesis = content
    print("[Synthesizer] Synthesis complete")
    return vault.synthesis