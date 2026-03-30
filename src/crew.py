"""
CrewAI pipeline for the Course Planning Assistant.

Defines four specialized agents and their tasks:
    1. Intake Agent    — validates the student profile and requests missing info
    2. Retriever Agent — searches ChromaDB for relevant catalog chunks
    3. Planner Agent   — produces a cited term plan from retrieved chunks
    4. Verifier Agent  — audits every citation and flags unsupported claims

All agents use a local Ollama model via a custom LangChain-compatible LLM wrapper.
The Crew runs sequentially so each agent's output feeds the next.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Iterator, List, Optional

import requests
from dotenv import load_dotenv
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from pydantic import Field

from src.prompts import INTAKE_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT
from src.state import PlannerState
from src.utils import (
    format_chunks_for_prompt,
    format_student_profile,
    safe_get,
    extract_citations,
    extract_assumptions,
    extract_section,
)
from src.vectorstore import retrieve as vs_retrieve

load_dotenv()

# ── Ollama configuration ──────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
MAX_RETRIES = 5
INITIAL_BACKOFF = 5  # seconds


# ══════════════════════════════════════════════════════════════
# Custom Ollama LLM wrapper (LangChain BaseLLM)
# ══════════════════════════════════════════════════════════════

class OllamaLLM(BaseLLM):
    """LangChain-compatible wrapper around the local Ollama /api/generate endpoint."""

    model: str = MODEL
    host: str = OLLAMA_HOST
    max_tokens: int = 4096
    temperature: float = 0.2

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call_with_retry(self, prompt: str) -> str:
        url = f"{self.host.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        }
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(url, json=payload, timeout=180)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
            except requests.RequestException as exc:
                wait_time = INITIAL_BACKOFF * (2 ** attempt)
                print(
                    f"[OllamaLLM] ⏳ Attempt {attempt + 1}/{MAX_RETRIES} failed: {exc}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
        raise RuntimeError(
            f"Ollama request failed after {MAX_RETRIES} retries. "
            "Is Ollama running and the model pulled?"
        )

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call_with_retry(prompt)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        yield self._call_with_retry(prompt)


def _make_llm() -> OllamaLLM:
    return OllamaLLM(model=MODEL, host=OLLAMA_HOST)


# ══════════════════════════════════════════════════════════════
# CrewAI Custom Tools
# ══════════════════════════════════════════════════════════════

class CatalogRetrievalTool(BaseTool):
    """Tool that searches ChromaDB for relevant catalog chunks."""

    name: str = "catalog_retrieval"
    description: str = (
        "Search the course catalog vector store. "
        "Input: a search query string. "
        "Returns: formatted catalog chunks with source, chunk_id, and section."
    )
    k: int = Field(default=5)

    def _run(self, query: str) -> str:
        try:
            chunks = vs_retrieve(query, k=self.k)
            if not chunks:
                return "NO CATALOG CHUNKS FOUND for query: " + query
            return format_chunks_for_prompt(chunks)
        except Exception as exc:
            return f"Retrieval error: {exc}"


class ProgramRequirementsTool(BaseTool):
    """Tool that fetches program/degree requirement chunks from the catalog."""

    name: str = "program_requirements"
    description: str = (
        "Fetch degree-requirement chunks for a specific major. "
        "Input: major name (e.g. 'Computer Science'). "
        "Returns: program requirement catalog chunks."
    )
    k: int = Field(default=5)

    def _run(self, major: str) -> str:
        try:
            query = f"{major} program requirements degree plan curriculum"
            chunks = vs_retrieve(query, k=self.k)
            if not chunks:
                return "NO PROGRAM REQUIREMENTS FOUND for major: " + major
            return format_chunks_for_prompt(chunks)
        except Exception as exc:
            return f"Requirements retrieval error: {exc}"


# ══════════════════════════════════════════════════════════════
# Helper: direct LLM call (used for quick profile checks)
# ══════════════════════════════════════════════════════════════

def _llm_call(system_prompt: str, user_message: str, max_tokens: int = 2048) -> str:
    """Direct Ollama /api/chat call without the LangChain wrapper."""
    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, timeout=180)
            resp.raise_for_status()
            return (resp.json().get("message") or {}).get("content", "")
        except requests.RequestException as exc:
            wait_time = INITIAL_BACKOFF * (2 ** attempt)
            print(f"[_llm_call] Retry {attempt + 1}/{MAX_RETRIES} in {wait_time}s: {exc}")
            time.sleep(wait_time)
    raise RuntimeError("Ollama /api/chat failed after retries.")


# ── Allowed terms ─────────────────────────────────────────────
ALLOWED_TERMS = ("Fall", "Spring")


# ══════════════════════════════════════════════════════════════
# Intake validation (runs before CrewAI Crew to short-circuit)
# ══════════════════════════════════════════════════════════════

def _validate_profile(profile: dict, user_query: str) -> dict | None:
    """
    Check if the student profile has all required fields.

    Returns None if profile is complete, otherwise returns a dict with
    clarifying questions and a short-circuit final_output.
    """
    missing: List[str] = []
    courses = safe_get(profile, "completed_courses", [])
    if not courses or (isinstance(courses, list) and len(courses) == 0):
        missing.append("completed_courses")
    if not safe_get(profile, "target_major"):
        missing.append("target_major")
    if not safe_get(profile, "catalog_year"):
        missing.append("catalog_year")
    target_term = safe_get(profile, "target_term")
    if not target_term or target_term not in ALLOWED_TERMS:
        missing.append("target_term")
    max_credits = safe_get(profile, "max_credits")
    if not max_credits or max_credits <= 0:
        missing.append("max_credits")

    if not missing:
        return None  # Profile is complete

    print(f"  [Intake] Missing fields: {missing}")
    user_message = (
        f"Missing fields: {', '.join(missing)}\n\n"
        f"Current profile:\n{format_student_profile(profile)}\n\n"
        f"Student question: {user_query}\n\n"
        "Generate up to 5 numbered clarifying questions to collect the missing info. "
        "Output only the numbered list."
    )
    try:
        questions_text = _llm_call(INTAKE_PROMPT, user_message)
        questions = re.findall(r"\d+[\.\)]\s*(.+)", questions_text)
        if not questions:
            questions = [q.strip() for q in questions_text.strip().split("\n") if q.strip()]
        questions = questions[:5]
    except Exception as exc:
        print(f"  [Intake] LLM failed: {exc}")
        questions = [f"Please provide your {f.replace('_', ' ')}." for f in missing]

    output = "Profile incomplete. Please answer the following:\n\n" + "\n".join(
        f"{i + 1}. {q}" for i, q in enumerate(questions)
    )
    return {
        "is_profile_complete": False,
        "clarifying_questions": questions,
        "missing_fields": missing,
        "final_output": output,
    }


# ══════════════════════════════════════════════════════════════
# CrewAI Pipeline
# ══════════════════════════════════════════════════════════════

def _build_crew(profile_text: str, user_query: str) -> tuple[Crew, dict]:
    """
    Build a CrewAI Crew with four agents and their tasks.

    Returns the compiled Crew and a context dict for interpolating task inputs.
    """
    llm = _make_llm()
    retrieval_tool = CatalogRetrievalTool(k=5)
    program_tool = ProgramRequirementsTool(k=5)

    # ── Agent 1: Retriever ────────────────────────────────────
    retriever_agent = Agent(
        role="Academic Catalog Retriever",
        goal=(
            "Search the course catalog to find all relevant prerequisite, "
            "course description, and program requirement information for the student."
        ),
        backstory=(
            "You are an expert in navigating academic catalogs. "
            "You use vector search to retrieve the most relevant course and "
            "prerequisite information. You always cite sources precisely."
        ),
        tools=[retrieval_tool, program_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    # ── Agent 2: Planner ──────────────────────────────────────
    planner_agent = Agent(
        role="Academic Course Planner",
        goal=(
            "Create a detailed, citation-backed term plan strictly from the "
            "retrieved catalog chunks. Never recommend courses without citations."
        ),
        backstory=(
            "You are a meticulous academic advisor who builds term plans only "
            "from verified catalog evidence. Every recommendation includes a "
            "citation in the format [Source: <file>, Chunk: <id>, Section: <section>]."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )

    # ── Agent 3: Verifier ─────────────────────────────────────
    verifier_agent = Agent(
        role="Academic Citation Auditor",
        goal=(
            "Audit every course recommendation in the plan. "
            "Flag any claim that lacks a valid citation or has incorrect prerequisite logic."
        ),
        backstory=(
            "You are a strict academic citation auditor. You verify that every "
            "course recommendation has a citation that exactly matches a retrieved "
            "catalog chunk. Zero tolerance for uncited claims."
        ),
        tools=[retrieval_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    context = {
        "profile_text": profile_text,
        "user_query": user_query,
    }

    # ── Task 1: Retrieval ─────────────────────────────────────
    retrieval_task = Task(
        description=(
            "Search the course catalog for information relevant to this student.\n\n"
            f"STUDENT PROFILE:\n{profile_text}\n\n"
            f"STUDENT QUESTION:\n{user_query}\n\n"
            "Steps:\n"
            "1. Use catalog_retrieval with a query combining the student question + major + completed courses.\n"
            "2. Use program_requirements with the student's target major.\n"
            "3. Combine and return all retrieved chunks, labelled clearly by source, chunk_id, and section.\n"
            "4. If no chunks are found, state: NO CATALOG CHUNKS AVAILABLE."
        ),
        expected_output=(
            "All relevant catalog chunks formatted as:\n"
            "--- Chunk N ---\n"
            "Source: <filename>\nChunk ID: <id>\nSection: <section>\nContent:\n<text>\n"
        ),
        agent=retriever_agent,
    )

    # ── Task 2: Planning ──────────────────────────────────────
    planning_task = Task(
        description=(
            "Generate a term course plan strictly from the retrieved catalog chunks.\n\n"
            f"STUDENT PROFILE:\n{profile_text}\n\n"
            f"STUDENT QUESTION:\n{user_query}\n\n"
            f"{PLANNER_PROMPT}\n\n"
            "Use ONLY the catalog chunks provided by the Retriever. "
            "Every course must have a citation: [Source: <file>, Chunk: <id>, Section: <section>]. "
            "If you cannot cite a prerequisite, do NOT recommend that course."
        ),
        expected_output=(
            "A structured course plan with:\n"
            "ANSWER / PLAN:\n[courses with credits]\n\n"
            "WHY (Requirements/Prerequisites Satisfied):\n[justifications with citations]\n\n"
            "CITATIONS:\n- [Source: ..., Chunk: ..., Section: ...]\n\n"
            "ASSUMPTIONS / NOT IN CATALOG:\n[anything unverifiable]"
        ),
        agent=planner_agent,
        context=[retrieval_task],
    )

    # ── Task 3: Verification ──────────────────────────────────
    verification_task = Task(
        description=(
            "Audit the course plan produced by the Planner. "
            "For every course recommendation check:\n"
            "1. Does it have a citation? (YES/NO)\n"
            "2. Is the citation format correct [Source: ..., Chunk: ..., Section: ...]? (YES/NO)\n"
            "3. Is the prerequisite logic sound per the cited chunk? (YES/NO)\n\n"
            "Output a structured verification report ending with OVERALL: PASS or OVERALL: FAIL.\n"
            "List all FAILED CLAIMS if any."
        ),
        expected_output=(
            "VERIFICATION REPORT:\n"
            "Course: <name>\n"
            "- Has citation: YES/NO\n"
            "- Citation valid: YES/NO\n"
            "- Prereq logic correct: YES/NO\n"
            "- Status: PASS/FAIL\n\n"
            "OVERALL: PASS/FAIL\n"
            "FAILED CLAIMS: [list or NONE]"
        ),
        agent=verifier_agent,
        context=[retrieval_task, planning_task],
    )

    crew = Crew(
        agents=[retriever_agent, planner_agent, verifier_agent],
        tasks=[retrieval_task, planning_task, verification_task],
        process=Process.sequential,
        verbose=True,
    )
    return crew, context


# ══════════════════════════════════════════════════════════════
# Public API: run_pipeline
# ══════════════════════════════════════════════════════════════

def run_pipeline(user_query: str, student_profile: dict) -> dict:
    """
    Run the full CrewAI pipeline.

    Validates the student profile first (short-circuits with clarifying questions
    if incomplete), then dispatches a 3-agent Crew (Retriever → Planner → Verifier).

    Args:
        user_query: The student's natural-language question.
        student_profile: Dictionary of profile fields.

    Returns:
        PlannerState-compatible dict with final_output and all intermediate fields.
    """
    print("\n" + "█" * 60)
    print("██  COURSE PLANNING ASSISTANT — CrewAI Pipeline Start")
    print("█" * 60)
    print(f"  Query  : {user_query[:100]}")
    print(f"  Profile: {list(student_profile.keys())}")

    # ── Step 1: Validate profile ──────────────────────────────
    incomplete = _validate_profile(student_profile, user_query)
    if incomplete:
        print("  → Profile incomplete. Returning clarifying questions.")
        return {
            **_empty_state(user_query, student_profile),
            **incomplete,
        }

    # ── Step 2: Build and run CrewAI Crew ────────────────────
    profile_text = format_student_profile(student_profile)
    try:
        crew, _ = _build_crew(profile_text, user_query)
        crew_output = crew.kickoff()
        # crew_output.raw holds the final task output as a string
        final_text: str = getattr(crew_output, "raw", str(crew_output))
    except Exception as exc:
        print(f"\n✗ CrewAI pipeline error: {exc}")
        return {
            **_empty_state(user_query, student_profile),
            "is_profile_complete": True,
            "final_output": f"Pipeline error: {exc}",
        }

    # ── Step 3: Parse the final plan ──────────────────────────
    citations = extract_citations(final_text)
    assumptions = extract_assumptions(final_text)

    why_section = extract_section(final_text, "WHY (Requirements/Prerequisites Satisfied)")
    justifications = []
    if why_section:
        for line in why_section.split("\n"):
            line = line.strip().lstrip("- ")
            if line:
                justifications.append({"justification": line})

    # Determine verification status from the verifier task output
    verified = "OVERALL: PASS" in final_text.upper()
    failed_claims: List[str] = []
    if "FAILED CLAIMS:" in final_text.upper():
        after = final_text.upper().split("FAILED CLAIMS:")[1]
        for line in after.split("\n"):
            line = line.strip().lstrip("- ")
            if line and line not in ("NONE", "N/A", "[]"):
                failed_claims.append(line)

    print("\n" + "█" * 60)
    print("██  Pipeline Complete")
    print("█" * 60)

    return {
        "user_query": user_query,
        "student_profile": student_profile,
        "is_profile_complete": True,
        "clarifying_questions": [],
        "missing_fields": [],
        "retrieved_chunks": [],      # chunks held internally by agents
        "retrieval_query": user_query,
        "course_plan": final_text,
        "plan_justifications": justifications,
        "verified": verified,
        "verification_report": [{"report": final_text, "overall_pass": verified}],
        "failed_citations": failed_claims,
        "retry_count": 0,
        "final_output": final_text,
        "citations": citations,
        "assumptions": assumptions,
    }


def _empty_state(user_query: str, student_profile: dict) -> dict:
    return {
        "user_query": user_query,
        "student_profile": student_profile,
        "is_profile_complete": False,
        "clarifying_questions": [],
        "missing_fields": [],
        "retrieved_chunks": [],
        "retrieval_query": "",
        "course_plan": "",
        "plan_justifications": [],
        "verified": False,
        "verification_report": [],
        "failed_citations": [],
        "retry_count": 0,
        "final_output": "",
        "citations": [],
        "assumptions": [],
    }
