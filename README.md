# Course Planning Assistant

An AI-powered Agentic RAG (Retrieval-Augmented Generation) system that helps students plan their courses, check prerequisites, and build term plans — all grounded in their actual course catalog documents.

Built with **LangGraph** for agent orchestration, **Ollama** for local LLM inference, **ChromaDB** for vector search, and **Gradio** for the user interface.

---

## Setup

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd course_planning_assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   ```

4. **Configure Ollama** in `.env`:
   ```
   OLLAMA_HOST=http://localhost:11434
   OLLAMA_CHAT_MODEL=llama3.1:8b
   OLLAMA_EMBED_MODEL=nomic-embed-text
   ```
   Make sure Ollama is running and the models are pulled.

5. **Launch the app**
   ```bash
   python app.py
   ```
   The Gradio UI will be available at `http://localhost:7860`.

---

## Architecture

### LangGraph Pipeline

The system is built as a 4-node LangGraph `StateGraph` with conditional routing. Each node is a plain Python function that reads/writes a shared `PlannerState` (TypedDict). The graph enforces a strict flow: profile validation → retrieval → planning → verification, with an automatic retry loop for failed citation checks.

### Node Responsibilities

| Node | Responsibility |
|------|---------------|
| **Intake Node** | Validates the student profile is complete (courses, major, year, term, credits). If fields are missing, generates clarifying questions via Claude and short-circuits to END. |
| **Retriever Node** | Builds composite search queries from the student's profile + question, retrieves top-k chunks from ChromaDB, deduplicates results from both primary and program-requirement queries. |
| **Planner Node** | Generates a structured course plan using *only* the retrieved catalog chunks. Every recommendation must carry a `[Source, Chunk, Section]` citation. |
| **Verifier Node** | Audits every claim: checks citation existence, format, and prerequisite logic soundness. Failed verifications are sent back to the Planner for revision (max 3 retries). |

### Graph Flow

```
START
  │
  ▼
┌──────────────┐
│  intake_node │
└──────┬───────┘
       │
       ├── is_profile_complete = False
       │     → return clarifying questions → END
       │
       └── is_profile_complete = True
             │
             ▼
      ┌───────────────┐
      │ retriever_node│
      └───────┬───────┘
              │
              ▼
      ┌───────────────┐ ◄──── retry (max 3) ────┐
      │  planner_node │                          │
      └───────┬───────┘                          │
              │                                  │
              ▼                                  │
      ┌───────────────┐                          │
      │ verifier_node │                          │
      └───────┬───────┘                          │
              │                                  │
              ├── verified = True  ───────► END  │
              │                                  │
              ├── verified = False               │
              │   retry_count < 3 ───────────────┘
              │
              └── verified = False
                  retry_count >= 3 ───► END (with warning)
```

---

## Chunking & Retrieval Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Chunk size** | 500 tokens | Large enough for a full course description + prerequisites, small enough to avoid diluting relevance |
| **Overlap** | 50 tokens | Prevents information loss at chunk boundaries, especially for prerequisite chains that span paragraphs |
| **Top-k** | 5 | Balances recall with precision; combined with a second program-requirements query, total context is up to ~10 unique chunks |
| **Similarity** | Cosine | Standard metric for sentence-transformer embeddings; works well with normalized vectors from all-MiniLM-L6-v2 |
| **Embeddings** | all-MiniLM-L6-v2 | Fast, lightweight (80 MB), strong performance on semantic similarity benchmarks |

---

## Prompts Overview

| Prompt | Key Rules |
|--------|-----------|
| **Intake** | Check 5 required fields; generate ≤5 clarifying questions; never guess missing info |
| **Retriever** | Only use catalog chunks; always cite source + chunk_id; say "NOT FOUND IN CATALOG" if absent |
| **Planner** | Strict citation format `[Source, Chunk, Section]` for every claim; forbidden from using general knowledge; structured output format enforced |
| **Verifier** | Three-check audit per course (citation exists, format valid, prereq logic correct); overall PASS/FAIL; zero tolerance for uncited claims |

---

## Sources

| URL | Date Accessed | What It Covers |
|-----|--------------|----------------|
| [Google Generative AI Docs](https://ai.google.dev/docs) | 2024-12 | Gemini API usage, model configuration, free tier |
| [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) | 2024-12 | StateGraph, conditional edges, node functions |
| [ChromaDB Docs](https://docs.trychroma.com) | 2024-12 | Persistent client, embedding functions, cosine similarity |
| [Sentence-Transformers](https://www.sbert.net) | 2024-12 | all-MiniLM-L6-v2 model for embeddings |
| [Gradio Docs](https://www.gradio.app/docs) | 2024-12 | Blocks API, tabs, file upload, event handlers |

---

## Evaluation Results

| Metric | Score |
|--------|-------|
| Citation Coverage | — |
| Eligibility Correctness | — |
| Abstention Accuracy | — |

> **Note:** Scores depend on the catalog documents loaded. Run `python evaluation/evaluate.py` after indexing your catalog to fill in these values.

---

## Key Failure Modes

1. **Sparse Catalogs / Missing Sections**: If the uploaded catalog lacks prerequisite details for certain courses, the planner may produce over-cautious responses (abstaining even for courses that exist) because it finds no supporting chunks.

2. **Chunk Boundary Splits**: Prerequisite chains that span across page breaks or sections may be split into different chunks, causing the verifier to flag valid claims when the cited chunk only contains half the information.

3. **Citation Format Sensitivity**: The verifier is strict about citation format. Minor formatting deviations from the planner (e.g., extra spaces, different quote styles) cause false-negative verification failures, triggering unnecessary retry loops.

---

## Next Improvements

1. **Hybrid Retrieval**: Combine dense embeddings with BM25 keyword search to improve recall for exact course codes (e.g., "CS301") which dense models sometimes overlook.

2. **Conversation Memory**: Add multi-turn memory so students can ask follow-up questions without re-entering their profile, and the system can reference earlier answers.

3. **Graph Visualization**: Add a tab to the Gradio UI showing real-time pipeline progress — which node is active, how many retries have been attempted, and what chunks were retrieved.
