# agents/harvester.py
from tavily import TavilyClient
from dotenv import load_dotenv
import fitz  # PyMuPDF
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

# Tavily Client
tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

# Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Text Splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)


class HarvesterError(Exception):
    """Raised when a Harvester tool invocation fails in a way the pipeline cannot recover from."""
    pass


# Common Vector DB function for all modes
def save_to_vector_db(texts, metadatas):
    if not texts:
        raise ValueError("No text found to store in Vector DB")
    try:
        vector_store = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            persist_directory="./chroma_db",
            collection_name="research_assistant"
        )
    except Exception as e:
        raise HarvesterError(f"Failed to write to ChromaDB: {e}") from e
    return vector_store


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

    vault.vector_store = save_to_vector_db(
        texts,
        metadatas
    )
    print(
        f"[Harvester] Stored {len(texts)} web results in ChromaDB"
        + (f" ({len(failed_queries)} queries failed)" if failed_queries else "")
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

    vault.vector_store = save_to_vector_db(
        texts,
        metadatas
    )
    print(
        f"[Harvester] Stored {len(texts)} PDF pages in ChromaDB"
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

        chunks = splitter.split_text(full_text)
        for chunk in chunks:
            texts.append(chunk)
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

    vault.vector_store = save_to_vector_db(
        texts,
        metadatas
    )
    print(
        f"[Harvester] Stored {len(texts)} chunks "
        f"from {len(vault.input_data)} papers in ChromaDB"
        + (f" ({len(skipped_files)} files skipped)" if skipped_files else "")
    )
    return vault.vector_store