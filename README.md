# Prerequisite Course Planning Assistant

An AI-powered Agentic RAG system that helps students plan their courses, check prerequisites, and build term plans — all grounded in their actual course catalog documents.

Built with **CrewAI** for multi-agent orchestration, **Ollama** for local LLM inference, **ChromaDB** for vector search, **FastAPI** for the backend API, and **Next.js** for the frontend UI.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Next.js Frontend                          │
│   Upload Catalog | Ask Questions | Plan My Term                 │
└───────────────────┬─────────────────────────────────────────────┘
                    │ REST API (JSON)
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend (api.py)                   │
│  POST /api/upload  │  POST /api/plan  │  GET /api/status        │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CrewAI Pipeline (src/crew.py)                │
│                                                                 │
│   1. Intake Validation (profile completeness check)             │
│        ↓ (if complete)                                          │
│   2. Retriever Agent ─→ CatalogRetrievalTool + ProgramTool      │
│        ↓                          ↓                             │
│   3. Planner Agent ─────── ChromaDB (vector search)             │
│        ↓                                                        │
│   4. Verifier Agent (citation audit)                            │
│        ↓                                                        │
│   Final cited course plan                                       │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ChromaDB + Ollama                              │
│   Embedding: nomic-embed-text  │  LLM: llama3.1:8b              │
└─────────────────────────────────────────────────────────────────┘
```

### CrewAI Agents

| Agent | Role | Tools |
|-------|------|-------|
| **Retriever Agent** | Searches ChromaDB for catalog chunks relevant to the student's profile and query | `CatalogRetrievalTool`, `ProgramRequirementsTool` |
| **Planner Agent** | Generates a structured, citation-backed term plan from retrieved chunks | — (uses context from Retriever) |
| **Verifier Agent** | Audits every citation in the plan; checks format, validity, and prerequisite logic | `CatalogRetrievalTool` |

All agents use a local Ollama LLM via a custom LangChain `BaseLLM` wrapper with automatic retry on transient errors.

### Key Design Decisions

- **Sequential CrewAI Process**: Retriever → Planner → Verifier ensures each agent's output feeds the next.
- **Intake validation** runs before the Crew to short-circuit with clarifying questions if the student profile is incomplete.
- **Citation-first planning**: Planner is forbidden from recommending courses without catalog citations.
- **Word-based chunking**: Ingestion uses word-count chunking (no external tokenizer downloads needed).
- **Ollama embedding proxy**: Custom `__call__` method ensures ChromaDB always receives `List[List[float]]`.

---

## Setup

### Prerequisites

- Python 3.12+
- Node.js 18+
- [Ollama](https://ollama.ai/) running locally with:
  - `ollama pull llama3.1:8b`
  - `ollama pull nomic-embed-text`

### 1. Clone the repository

```bash
git clone <repo-url>
cd Prerequisite-Course-Planning-Assistant
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` if your Ollama host differs from `http://localhost:11434`.

### 4. Start the FastAPI backend

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

### 5. Start the Next.js frontend

```bash
cd frontend
npm install
npm run dev
```

The UI will be available at `http://localhost:3000`.

---

## Usage

1. **Upload Catalog** tab — Upload your course catalog PDF(s) to index them into ChromaDB.
2. **Ask Questions** tab — Fill in your profile and ask specific questions (e.g. "What are the prerequisites for CS301?").
3. **Plan My Term** tab — Fill in your profile to get a complete, verified term plan with citations.

---

## Project Structure

```
.
├── api.py                    # FastAPI server (replaces Gradio app.py)
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── src/
│   ├── crew.py               # CrewAI agents, tools, and pipeline (replaces LangGraph)
│   ├── ingestion.py          # PDF/HTML document loading and chunking
│   ├── vectorstore.py        # ChromaDB operations (index, retrieve, clear)
│   ├── state.py              # Pydantic models and TypedDict state
│   ├── prompts.py            # System prompts for each agent
│   └── utils.py              # Shared helper functions
├── evaluation/
│   ├── evaluate.py           # Evaluation script (citation coverage, etc.)
│   └── test_queries.py       # 25 test queries across query types
├── material/                 # Sample course catalog PDFs
└── frontend/                 # Next.js UI
    ├── app/
    │   ├── page.tsx          # Main page with 3-tab layout
    │   ├── layout.tsx        # Root layout
    │   └── globals.css       # Global styles
    └── lib/
        └── api.ts            # API client for the FastAPI backend
```

---

## Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Chunk size** | 500 words | Covers a full course description + prerequisites |
| **Overlap** | 50 words | Prevents information loss at chunk boundaries |
| **Top-k** | 5 per query | Two queries (main + program requirements) = up to 10 unique chunks |
| **Similarity** | Cosine | Standard for normalized sentence-transformer embeddings |
| **Embeddings** | nomic-embed-text (Ollama) | Local inference, no API key required |

---

## Evaluation

```bash
python evaluation/evaluate.py
```

Metrics computed:
- **Citation Coverage Rate** — fraction of responses with ≥1 citation
- **Eligibility Correctness** — correct eligible/ineligible recommendations
- **Abstention Accuracy** — correct abstention for out-of-catalog queries

---

## Key Failure Modes

1. **Sparse Catalogs**: Missing prerequisite details cause the planner to abstain.
2. **Chunk Boundary Splits**: Prerequisite chains spanning page breaks may be split.
3. **Ollama Unavailable**: All LLM-dependent operations fail gracefully with error messages in the UI.
