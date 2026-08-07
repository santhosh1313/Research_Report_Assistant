# Multi-Agent Research & Report Assistant

A multi-agent AI system that researches a topic, summarizes a single paper, or compares
multiple papers into a literature review — built as a capstone project for **Infosys
Springboard 7.0**, categorized under **AI Coordination & Decision Engine**.

The system runs entirely on **free-tier services** (Groq, Google Gemini, Tavily, ChromaDB,
SQLite) — no paid infrastructure required.

Two ways to use it:
- **CLI** (`main.py`) — one-shot: give it a topic or PDFs, get a report.
- **Chat app** (`app.py`, Streamlit) — a ChatGPT-style conversational interface with
  accounts, persistent conversations, and follow-up Q&A.

---

## How it works

Five specialized agents, each with one job, coordinated through a shared memory object
called **Vault** and orchestrated with **LangGraph**:

| Agent | Role |
|---|---|
| **Atlas** | Detects the mode (topic / single document / multiple documents) and routes the run |
| **Pathfinder** | Plans the work — breaks a topic into sub-questions, or picks comparison dimensions for papers |
| **Harvester** | Gathers material — live web search (Tavily), PDF parsing (PyMuPDF), or multi-document parsing |
| **Synthesizer** | Analyzes and synthesizes the gathered material into themes, findings, and comparisons |
| **Scribe** | Writes the final structured report |

### Three modes

- **Topic Mode** — Pathfinder generates sub-questions, Harvester searches the live web via
  Tavily, Synthesizer builds themes from the results.
- **Single-Doc Mode** — Harvester extracts text per page with PyMuPDF, Synthesizer pulls out
  methodology/findings/limitations from one paper.
- **Multi-Doc Mode** — Harvester parses every document, Synthesizer produces a comparative
  literature review across all of them.

---

## Agent coordination & memory

This project treats "agents talking to each other" and "agents remembering things" as two
distinct systems, both visible and testable rather than implicit:

**Coordination.** Every agent logs a handoff to `vault.comm_log` when it passes work to the
next agent (e.g. `Pathfinder -> Harvester: plan_ready`). After each CLI run,
`workflow_validator.py` checks that the expected agent sequence
(Atlas → Pathfinder → Harvester → Synthesizer → Scribe) actually happened, and saves a
`*_trace.json` file with the full handoff log and a PASS/FAIL summary — demonstrable proof
of collaborative execution, not just console prints.

**Memory** is split into two clearly separated systems:

- **Short-term (Vault)** — run-scoped, in-memory, cleared after every run. Holds the current
  subtasks, gathered facts, synthesis, and final report.
- **Long-term (ChromaDB)** — persists across runs and processes. Two collections live in the
  same on-disk store (`./chroma_db`):
  - `research_assistant` — raw research facts (web results, paper text), **shared across all
    users**. Harvester checks this before running a fresh Tavily search — if it already has a
    highly relevant result for a sub-question, it reuses it instead of spending a search call.
    Synthesizer also queries it to fold relevant findings from *past* runs into a new report.
  - `conversation_history` — chat turns from the Streamlit app, scoped per `user_id` +
    `session_id` so conversations stay private between users.

---

## The conversational app

`app.py` wraps the same five agents in a chat interface:

- **Accounts** (`auth_db.py`, SQLite) — registration and login with PBKDF2-hashed passwords
  (no plaintext, no extra dependency). Login persists across a page refresh via a session
  token stored in the URL and validated against a `sessions` table.
- **Conversations** — each chat is a row in SQLite (title, timestamps, pinned flag) plus its
  full message history in ChromaDB. The sidebar lists a user's conversations, pinned ones
  first, each with a **⋮** menu for rename / pin / delete (deleting a chat cleans up both the
  SQLite record and its ChromaDB history).
- **Orchestration** (`orchestrator.py`) — a plain Python session loop (not a LangGraph node —
  a chat loop doesn't need graph cycles). Per message, it decides:
  - **First message in a conversation**, or a message prefixed with `research:` / `topic:` /
    `analyze:` → runs the full Atlas → Scribe pipeline.
  - **Everything else** → answered as a follow-up: ChromaDB first (this user's conversation
    history + the shared research memory), Tavily only as a fallback if memory has nothing
    relevant. The most recent turn in the conversation is given generous room (up to 8,000
    characters) rather than being aggressively truncated, since follow-ups often refer
    directly to "the above" report.
- **Attachments** — the chat bar uses Streamlit's native `+` file-attach button
  (`st.chat_input(accept_file="multiple")`) for PDFs, one bar for everything, no separate
  upload section.
- **Theme** — a custom light theme (`.streamlit/config.toml`) instead of the Streamlit
  default dark palette.

---

## Project structure

```
research-report-assistant/
├── main.py                  # CLI entry point (one-shot pipeline)
├── app.py                   # Streamlit chat app entry point
├── vault.py                 # Shared state: short-term + long-term (ChromaDB) memory
├── orchestrator.py          # Conversational session loop (chat app only)
├── auth_db.py                # User accounts, sessions, conversation metadata (SQLite)
├── conversation_memory.py    # Per-user chat history (ChromaDB)
├── workflow_validator.py     # Validates + traces agent handoffs (CLI runs)
├── agents/
│   ├── atlas.py               # Mode detection / routing
│   ├── pathfinder.py          # Planning (sub-questions / comparison dimensions)
│   ├── harvester.py           # Web search, PDF parsing, memory-first search
│   ├── synthesizer.py         # Analysis + synthesis (RAG-augmented for topic/single-doc)
│   └── scribe.py               # Final report writing
├── .streamlit/
│   └── config.toml            # UI theme
├── requirements.txt
└── .env                      # API keys (not committed)
```

---

## Setup

**1. Clone and enter the project, create a virtual environment:**
```bash
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Add your API keys** in a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
TAVILY_API_KEY=your_tavily_key
```
All three services have free tiers. Groq and Gemini are used together deliberately — a
dual-provider strategy to stay within free-tier quota limits rather than relying on one
provider alone.

---

## Usage

### CLI (one-shot report)
```bash
# Topic research
python main.py --topic "Research on RAG" --out report.txt

# Single document summary
python main.py --docs "path/to/paper.pdf" --out summary.txt

# Multi-document literature review
python main.py --docs "paper1.pdf" "paper2.pdf" --out review.txt
```
Each run prints a live agent-by-agent trace, saves the report, and saves a
`<out>_trace.json` workflow validation file alongside it.

### Chat app
```bash
streamlit run app.py
```
Opens in your browser (usually `http://localhost:8501`). Register an account, start a new
chat, and either type a topic (first message always runs full research) or attach PDFs with
the `+` button in the chat bar. After that, ask follow-up questions naturally, or prefix a
message with `research:` to force a fresh report mid-conversation.

---

## Tech stack

- **Language / orchestration:** Python, LangChain, LangGraph
- **LLMs:** Groq (`llama-3.1-8b-instant`), Google Gemini (`gemini-2.5-flash`,
  `gemini-embedding-2`)
- **Search:** Tavily
- **Vector memory:** ChromaDB (long-term research memory + per-user conversation memory)
- **Document parsing:** PyMuPDF (PDF), python-docx
- **Permanent storage:** SQLite (accounts, sessions, conversation metadata)
- **UI:** Streamlit

---

## Milestone context

Built for Infosys Springboard 7.0. Milestone 3 (Agent Coordination & Memory Systems)
objectives and how this project meets them:

| Objective | Where |
|---|---|
| Specialized agents with defined roles | `agents/` — Atlas, Pathfinder, Harvester, Synthesizer, Scribe |
| Agent communication & coordination | `vault.comm_log`, populated by every agent on handoff |
| Short-term & long-term memory | `vault.py` (short-term) + ChromaDB via `save_to_long_term_memory` / `query_long_term_memory` (long-term) |
| Validate collaborative workflow execution | `workflow_validator.py` — PASS/FAIL trace saved per run |

---

## Roadmap

- LLM-based intent classification for chat routing (currently a simple first-message /
  explicit-prefix rule)
- FastAPI wrapper for programmatic access
- Docker containerization
