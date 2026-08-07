# agents/harvester.py
from tavily import TavilyClient
from dotenv import load_dotenv
import fitz  # PyMuPDF
import os

load_dotenv()

# Tavily Client
tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


class HarvesterError(Exception):
    """Raised when a Harvester tool invocation fails in a way the pipeline cannot recover from."""
    pass


# =================================================
# MODE 1: Topic Research
# =================================================
def harvester_web_search(vault):
    texts = []
    metadatas = []

    if not vault.subtasks:
        raise HarvesterError("No subtasks from Pathfinder — cannot run web search")

    failed_queries = []

    for question in vault.subtasks:
        question = question.strip()
        if not question:
            continue
        try:
            results = tavily.search(
                query=question,
                max_results=3
            )
        except Exception as e:
            # One bad query/rate-limit shouldn't kill the whole research run.
            print(f"[Harvester] Tavily search failed for '{question}': {e}")
            failed_queries.append(question)
            continue

        result_list = results.get("results", []) if isinstance(results, dict) else []
        if not result_list:
            print(f"[Harvester] No results returned for '{question}'")
            continue

        for r in result_list:
            content = r.get("content")
            url = r.get("url")
            if not content or not url:
                continue
            vault.log_fact(
                text=content,
                source=url
            )
            texts.append(content)
            metadatas.append(
                {
                    "source": url,
                    "mode": "topic"
                }
            )

    if not texts:
        raise HarvesterError(
            f"Tavily search returned no usable results for any of {len(vault.subtasks)} subtasks "
            f"({len(failed_queries)} query calls failed outright)"
        )

    vault.web_results = texts

    # Long-term memory: persist so a future run on a related topic can recall these.
    vault.save_to_long_term_memory(texts, metadatas)

    print(
        f"[Harvester] Collected {len(texts)} web results")

    vault.log_message(
        from_agent="Harvester",
        to_agent="Synthesizer",
        action="web_results_ready",
        detail=f"{len(texts)} facts from {len(vault.subtasks)} subtasks ({len(failed_queries)} queries failed)"
    )



# =================================================
# MODE 2: Single Document
# =================================================
def harvester_parse_single(vault):
    path = vault.input_data[0]
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        doc = fitz.open(path)
    except Exception as e:
        raise HarvesterError(f"Failed to open PDF '{path}': {e}") from e

    texts = []
    metadatas = []
    full_text = ""

    for page_num, page in enumerate(doc):
        try:
            text = page.get_text()
        except Exception as e:
            print(f"[Harvester] Failed to extract text from page {page_num + 1}: {e}")
            continue

        if not text.strip():
            continue

        full_text += text
        vault.log_fact(
            text=text,
            source=f"{path} page {page_num + 1}"
        )
        texts.append(text)
        metadatas.append(
            {
                "source": path,
                "page": page_num + 1,
                "mode": "single_document"
            }
        )

    if not texts:
        raise HarvesterError(f"No extractable text found in '{path}' — file may be a scanned image PDF")

    vault.document_text = full_text

    # Long-term memory: persist so this paper's content is recallable in future runs.
    vault.save_to_long_term_memory(texts, metadatas)

    print(
        f"[Harvester] Extracted {len(texts)} pages"
    )

    vault.log_message(
        from_agent="Harvester",
        to_agent="Synthesizer",
        action="document_parsed",
        detail=f"{len(texts)} pages extracted from {path}"
    )
    return full_text


# =================================================
# MODE 3: Multiple Documents
# =================================================
def harvester_embed_multi(vault):
    texts = []
    metadatas = []
    skipped_files = []

    for path in vault.input_data:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        try:
            doc = fitz.open(path)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
        except Exception as e:
            print(f"[Harvester] Failed to parse '{path}': {e}")
            skipped_files.append(path)
            continue

        if not full_text.strip():
            print(f"[Harvester] No extractable text in '{path}' — skipping")
            skipped_files.append(path)
            continue

        texts.append(
            f"\n\n===== {os.path.basename(path)} =====\n\n{full_text}"
        )
        metadatas.append(
            {
                "source": path,
                "mode": "multi_document"
            }
        )

    if not texts:
        raise HarvesterError(
            f"None of the {len(vault.input_data)} documents produced usable text "
            f"(skipped: {skipped_files})"
        )

    vault.multi_doc_text = "\n\n".join(texts)

    # Long-term memory: persist per-document text. Note Synthesizer still uses
    # the full multi_doc_text directly for this mode (it already has complete
    # context), so retrieval isn't wired into synthesis here — this just makes
    # these documents recallable by topic/single-doc runs later.
    vault.save_to_long_term_memory(texts, metadatas)

    print(
        f"[Harvester] Parsed {len(texts)} documents"
    )

    vault.log_message(
        from_agent="Harvester",
        to_agent="Synthesizer",
        action="documents_parsed",
        detail=f"{len(texts)} documents parsed, {len(skipped_files)} skipped"
    )

    return vault.multi_doc_text