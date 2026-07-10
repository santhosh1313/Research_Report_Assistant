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


# Common Vector DB function for all modes
def save_to_vector_db(texts, metadatas):
    if not texts:
        raise ValueError("No text found to store in Vector DB")

    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="./chroma_db",
        collection_name="research_assistant"
    )

    return vector_store


# =================================================
# MODE 1: Topic Research
# =================================================

def harvester_web_search(vault):
    texts = []
    metadatas = []

    for question in vault.subtasks:
        results = tavily.search(
            query=question,
            max_results=3
        )

        for r in results["results"]:
            content = r["content"]
            url = r["url"]

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

    vault.vector_store = save_to_vector_db(
        texts,
        metadatas
    )

    print(
        f"[Harvester] Stored {len(texts)} web results in ChromaDB"
    )


# =================================================
# MODE 2: Single Document
# =================================================

def harvester_parse_single(vault):
    path = vault.input_data[0]

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    doc = fitz.open(path)

    texts = []
    metadatas = []
    full_text = ""

    for page_num, page in enumerate(doc):
        text = page.get_text()
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

    for path in vault.input_data:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        doc = fitz.open(path)

        full_text = ""

        for page in doc:
            full_text += page.get_text()

        chunks = splitter.split_text(full_text)

        for chunk in chunks:
            texts.append(chunk)

            metadatas.append(
                {
                    "source": path,
                    "mode": "multi_document"
                }
            )

    vault.vector_store = save_to_vector_db(
        texts,
        metadatas
    )

    print(
        f"[Harvester] Stored {len(texts)} chunks "
        f"from {len(vault.input_data)} papers in ChromaDB"
    )

    return vault.vector_store