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
    return vault.synthesis