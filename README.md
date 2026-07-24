# Multi-Agent Research & Report Assistant

A multi-agent AI system that turns either a **research topic** or a **set of research papers** into a polished, structured, citation-backed report — built entirely with free-tier tools as a capstone project for **Infosys Springboard 7.0**.

Five specialized agents are orchestrated with **LangGraph**, coordinated through a single shared memory object (**Vault**) that is passed through the pipeline end to end.

---

## Table of Contents

- [Overview](#overview)
- [Operating Modes](#operating-modes)
- [Architecture](#architecture)
- [Agents](#agents)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Output Format](#output-format)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)

---

## Overview

Given either **(a)** a topic to research, or **(b)** one or more research papers, the system produces a polished, structured report using a mode-appropriate template.

A coordinator agent (`Atlas`) inspects the input and routes it to the correct pipeline. All three paths converge on the same final stage — a well-structured report drafted by `Scribe`.

## Operating Modes

| Mode | Trigger | What it does |
|---|---|---|
| **Topic Mode** | A topic string is provided | Performs live web research (Tavily) and synthesizes findings into a themed, cited report |
| **Single-Doc Mode** | One PDF is provided | Produces a structured summary of the paper — overview, methodology, contributions, results, and an internal-consistency check (no comparison) |
| **Multi-Doc Mode** | Two or more PDFs are provided | Chunks & embeds every paper, retrieves relevant content per comparison dimension, and produces a literature-review-style comparison |

Mode detection is automatic, based on the shape of the input (`atlas.py`):

```python
str            -> "topic"
[one path]     -> "single_doc"
[many paths]   -> "multi_doc"
```

## Architecture

```
User Input (topic string OR PDF path[s])
        │
        ▼
     Atlas ──────────────┐   (detects mode, routes pipeline)
        │                │
        ▼                │
   Pathfinder             │   (plans sub-questions / comparison dimensions)
        │                │
        ▼                │
   Harvester ◄────────────┤   (mode-specific retrieval)
   ├─ Topic:      Tavily web search
   ├─ Single-Doc: PyMuPDF parse (1 file)
   └─ Multi-Doc:  chunk + embed (Gemini Embeddings) into a vector store
        │                │
        ▼                │
  Synthesizer ◄───────────┤   (synthesizes themes / comparisons, Gemini 2.5-flash)
        │                │
        ▼                │
     Scribe ◄─────────────┘   (drafts final report, Groq)
        │
        ▼
   Final Report

        ▲▲▲▲▲
        │││││
      VAULT — shared runtime memory (GraphState)
      every agent above reads from / writes to it
```

`Vault` is **not an agent** — it's the shared storage layer (mode, input data, planned subtasks, logged facts with sources, synthesis output, final report) that every LangGraph node reads from and writes to. It's what prevents hallucinated citations: every fact `Harvester` pulls is logged with its source, so `Synthesizer` and `Scribe` only ever cite what was actually retrieved.

## Agents

| Agent | Role | Responsibility | Model / Tool |
|---|---|---|---|
| **Atlas** | Coordinator | Detects input type and routes the pipeline to the correct mode | Pure Python (mode-routing logic) |
| **Pathfinder** | Planner | Breaks a topic into sub-questions, or outlines comparison dimensions for documents | Groq · `llama-3.1-8b-instant` |
| **Harvester** | Researcher | Runs web search, parses a PDF, or chunks & embeds documents, depending on mode | Tavily · PyMuPDF · Gemini Embeddings |
| **Synthesizer** | Analyst | Synthesizes findings into themes, flags contradictions and research gaps | Gemini `2.5-flash` |
| **Scribe** | Writer | Drafts the final structured, citation-backed report from a mode-specific template | Groq · `llama-3.1-8b-instant` |

The hybrid dual-provider strategy (Groq for planning/writing, Gemini for retrieval-heavy reasoning and embeddings) keeps the pipeline fast and comfortably inside both providers' free-tier limits.

## Tech Stack

- **Language / IDE:** Python, VS Code
- **Orchestration:** LangChain, LangGraph
- **LLMs:** Google Gemini (`gemini-2.5-flash`), Groq (`llama-3.1-8b-instant`)
- **Embeddings:** Gemini Embeddings (`gemini-embedding-2`)
- **Vector store:** ChromaDB (persistent, used internally by `Harvester`/`Synthesizer` for multi-doc retrieval)
- **Web search:** Tavily API (free tier)
- **PDF parsing:** PyMuPDF (`fitz`)
- **Report export:** `python-docx`
- **Config:** `python-dotenv`

Everything runs locally against free-tier APIs — no paid subscriptions anywhere in the pipeline.

## Project Structure

```
.
├── main.py                 # CLI entry point, LangGraph wiring & conditional routing
├── vault.py                 # Shared memory object (GraphState) passed through the graph
├── agents/
│   ├── atlas.py              # Coordinator — mode detection & routing
│   ├── pathfinder.py         # Planner — sub-questions / comparison dimensions
│   ├── harvester.py          # Researcher — web search / PDF parse / chunk+embed
│   ├── synthesizer.py        # Analyst — synthesis & comparison
│   └── scribe.py             # Writer — final report drafting (3 prompt templates)
├── chroma_db/                # Persistent vector store (created at runtime, multi-doc mode)
├── .env                      # API keys (not committed)
├── requirements.txt
└── README.md
```

## Setup

1. **Clone the repo and create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys** — create a `.env` file in the project root (see [Environment Variables](#environment-variables)).

All three keys used below have generous free tiers and require no credit card.

## Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
GROQ_API_KEY=your_groq_api_key
```

`main.py` checks for all three keys at startup and exits early with a clear message if any are missing.

## Usage

**Topic Mode**
```bash
python main.py --topic "Impact of remote work on employee productivity" --out report.txt
```

**Single-Doc Mode**
```bash
python main.py --docs paper.pdf --out summary.txt
```

**Multi-Doc Mode**
```bash
python main.py --docs paper1.pdf paper2.pdf paper3.pdf --out review.txt
```

The final report is printed to the console and saved to the file passed via `--out` (defaults to `report.txt`).

## Output Format

Each mode drives a dedicated `Scribe` prompt template, producing a structured Markdown report:

**Topic Mode**
1. Executive Summary
2. Introduction
3. Key Research Themes
4. Perspectives & Contradictions
5. Trends / Applications
6. Conclusion
7. References `[1] [2] ...`

*Narrative report with inline citations linking to a numbered reference list.*

**Single-Doc Mode**
1. Overview
2. Methodology Explained
3. Key Contributions
4. Experimental Setup
5. Results & Findings
6. Internal Consistency Analysis
7. Limitations
8. Conclusion

*Pure per-section summary of one paper — no comparison table, no agreement/contradiction section.*

**Multi-Doc Mode**
1. Introduction
2. Individual Paper Summaries
3. Comparative Analysis Table
4. Common Findings
5. Contradictions / Differences
6. Research Gaps
7. Final Evaluation
8. Conclusion

*Literature-review-style comparison with a methodology/results table across all papers.*

## Roadmap

- **Phase 1 (in progress):** Upgrade `Synthesizer` from full-context stuffing to proper RAG — `similarity_search()` against ChromaDB with top-k retrieval and relevance scoring, plus a synthesis prompt that enforces grounding and citations
- **Phase 2 (in progress):** Conversational follow-up interface after report generation — ChromaDB-first retrieval with a relevance threshold, Tavily fallback when context is insufficient, write-back of new information into ChromaDB, and session `chat_history` held in Vault
- **Future / enterprise upgrade:** FastAPI wrapper, Docker packaging, LangSmith observability, pytest CI, async concurrency, PostgreSQL for run history & audit trails, object storage for uploaded documents

## Acknowledgements

Built as a capstone project for **Infosys Springboard 7.0**.
