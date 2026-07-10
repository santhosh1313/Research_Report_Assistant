# agents/synthesizer.py
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
 
load_dotenv()
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def synthesizer_run(vault):
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
        retrieved = vault.vector_store.similarity_search(dims, k=8)
        context = "\n".join([f"- {d.page_content[:300]} (Source: {d.metadata['source']})" for d in retrieved])
        prompt = f"""Compare the following paper excerpts across these dimensions:
        {dims}
        Identify agreements, contradictions, and gaps across papers.
        Excerpts:\n{context}"""
 
    response = llm.invoke(prompt)
    vault.synthesis = response.content
    print("[Synthesizer] Synthesis complete")
    return vault.synthesis
