"""
Agent node functions for the Course Planning Assistant LangGraph pipeline.

Each function is a LangGraph node that takes PlannerState and returns
an updated PlannerState dictionary. All LLM calls use a local Ollama model
via the Ollama HTTP API with automatic retry on transient errors.
"""

import os
import time
from typing import List

from dotenv import load_dotenv

load_dotenv()

import requests

from src.state import PlannerState
from src.prompts import INTAKE_PROMPT, RETRIEVER_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT
from src.vectorstore import retrieve as vs_retrieve
from src.utils import (
    format_chunks_for_prompt,
    format_student_profile,
    extract_citations,
    extract_section,
    extract_assumptions,
    safe_get,
)

# ── Ollama client config ──────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")

# Rate-limit retry settings
MAX_RETRIES = 5
INITIAL_BACKOFF = 10  # seconds


def _call_llm(system_prompt: str, user_message: str, max_tokens: int = 2048) -> str:
    """
    Call a local Ollama chat model with automatic retry on transient errors.

    Args:
        system_prompt: The system-level instruction prompt.
        user_message: The user-level message/query.
        max_tokens: Maximum tokens to generate.

    Returns:
        The text response from the Ollama model.

    Raises:
        Exception: If the API call fails after all retries.
    """
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
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return (data.get("message") or {}).get("content", "")
        except requests.RequestException as e:
            wait_time = INITIAL_BACKOFF * (2 ** attempt)
            print(f"  [Ollama] ⏳ Request error: {e}. Retrying in {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait_time)
        except ValueError as e:
            print(f"  [Ollama] ✗ Invalid JSON response: {e}")
            raise

    raise Exception(
        f"Ollama request failed after {MAX_RETRIES} retries. Please check that Ollama is running."
    )


# ══════════════════════════════════════════════════════════════
# NODE 1 — INTAKE
# ══════════════════════════════════════════════════════════════
def intake_node(state: PlannerState) -> PlannerState:
    """
    Check if the student profile is complete.

    If any required field is missing, generates clarifying questions
    via Claude. If complete, marks the profile as ready for retrieval.

    Required fields:
        - completed_courses (non-empty list)
        - target_major (not None)
        - catalog_year (not None)
        - target_term (Fall or Spring)
        - max_credits (positive integer)
    """
    print("\n" + "=" * 50)
    print("▶ NODE: intake_node")
    print("=" * 50)

    profile = state.get("student_profile", {}) or {}
    missing_fields: List[str] = []

    # Check completed_courses
    courses = safe_get(profile, "completed_courses", [])
    if not courses or (isinstance(courses, list) and len(courses) == 0):
        missing_fields.append("completed_courses")

    # Check target_major
    if not safe_get(profile, "target_major"):
        missing_fields.append("target_major")

    # Check catalog_year
    if not safe_get(profile, "catalog_year"):
        missing_fields.append("catalog_year")

    # Check target_term
    target_term = safe_get(profile, "target_term")
    if not target_term or target_term not in ("Fall", "Spring"):
        missing_fields.append("target_term")

    # Check max_credits
    max_credits = safe_get(profile, "max_credits")
    if max_credits is None or (isinstance(max_credits, int) and max_credits <= 0):
        missing_fields.append("max_credits")

    if missing_fields:
        print(f"  Missing fields: {missing_fields}")

        # Generate clarifying questions via Claude
        user_message = (
            f"The student wants to plan courses but is missing the following information:\n"
            f"Missing fields: {', '.join(missing_fields)}\n\n"
            f"Current profile information:\n{format_student_profile(profile)}\n\n"
            f"Student's question: {state.get('user_query', 'No question provided')}\n\n"
            f"Generate up to 5 clear, specific clarifying questions to collect the missing information. "
            f"Output only the numbered list of questions."
        )

        try:
            questions_text = _call_llm(INTAKE_PROMPT, user_message)
            # Parse numbered questions
            import re
            questions = re.findall(r'\d+[\.\)]\s*(.+)', questions_text)
            if not questions:
                questions = [q.strip() for q in questions_text.strip().split('\n') if q.strip()]
            questions = questions[:5]
        except Exception as e:
            print(f"  ✗ Failed to generate clarifying questions: {e}")
            questions = [f"Please provide your {field.replace('_', ' ')}." for field in missing_fields]

        print(f"  Profile INCOMPLETE — {len(questions)} clarifying question(s) generated.")
        return {
            "is_profile_complete": False,
            "clarifying_questions": questions,
            "missing_fields": missing_fields,
            "final_output": "Profile incomplete. Please answer the following questions:\n\n"
                           + "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions)),
        }
    else:
        print("  ✓ Profile is complete.")
        return {
            "is_profile_complete": True,
            "clarifying_questions": [],
            "missing_fields": [],
        }


# ══════════════════════════════════════════════════════════════
# NODE 2 — RETRIEVER
# ══════════════════════════════════════════════════════════════
def retriever_node(state: PlannerState) -> PlannerState:
    """
    Build retrieval queries from the student profile and user query,
    then retrieve relevant chunks from ChromaDB.

    Steps:
        1. Build a combined retrieval query from user_query + target_major + courses
        2. Retrieve top-k chunks for the main query
        3. Retrieve program requirement chunks separately
        4. Combine and deduplicate results
    """
    print("\n" + "=" * 50)
    print("▶ NODE: retriever_node")
    print("=" * 50)

    profile = state.get("student_profile", {}) or {}
    user_query = state.get("user_query", "")

    # Build primary retrieval query
    target_major = safe_get(profile, "target_major", "")
    completed_courses = safe_get(profile, "completed_courses", [])
    courses_str = ", ".join(completed_courses) if completed_courses else ""

    retrieval_query = f"{user_query}"
    if target_major:
        retrieval_query += f" {target_major} major requirements"
    if courses_str:
        retrieval_query += f" prerequisites for students who completed {courses_str}"

    print(f"  Retrieval query: {retrieval_query[:120]}...")

    # Retrieve main chunks
    try:
        main_chunks = vs_retrieve(retrieval_query, k=5)
    except Exception as e:
        print(f"  ✗ Main retrieval failed: {e}")
        main_chunks = []

    # Retrieve program requirement chunks separately
    program_query = f"{target_major} program requirements degree plan" if target_major else "program requirements"
    try:
        program_chunks = vs_retrieve(program_query, k=5)
    except Exception as e:
        print(f"  ✗ Program retrieval failed: {e}")
        program_chunks = []

    # Combine and deduplicate
    seen_ids = set()
    combined_chunks = []
    for chunk in main_chunks + program_chunks:
        cid = chunk.get("chunk_id", "")
        if cid not in seen_ids:
            seen_ids.add(cid)
            combined_chunks.append(chunk)

    # Print retrieved sources
    print(f"\n  Retrieved {len(combined_chunks)} unique chunk(s):")
    for chunk in combined_chunks:
        print(f"    • [{chunk.get('source', '?')}] Section: {chunk.get('section', '?')} "
              f"(Chunk: {chunk.get('chunk_id', '?')[:8]}...)")

    return {
        "retrieved_chunks": combined_chunks,
        "retrieval_query": retrieval_query,
    }


# ══════════════════════════════════════════════════════════════
# NODE 3 — PLANNER
# ══════════════════════════════════════════════════════════════
def planner_node(state: PlannerState) -> PlannerState:
    """
    Generate a course plan strictly from retrieved chunks.

    Every course recommendation must have a citation. Builds a
    detailed prompt with the student profile and retrieved chunks,
    then calls Claude to produce a structured plan.
    """
    print("\n" + "=" * 50)
    print("▶ NODE: planner_node")
    print("=" * 50)

    profile = state.get("student_profile", {}) or {}
    chunks = state.get("retrieved_chunks", [])
    user_query = state.get("user_query", "")
    retry_count = state.get("retry_count", 0)
    failed_citations = state.get("failed_citations", [])

    # Build the user message with all context
    chunks_text = format_chunks_for_prompt(chunks)
    profile_text = format_student_profile(profile)

    revision_note = ""
    if retry_count > 0 and failed_citations:
        revision_note = (
            f"\n\n⚠ REVISION REQUIRED (attempt {retry_count + 1}/3):\n"
            f"The following claims failed verification:\n"
            + "\n".join(f"- {fc}" for fc in failed_citations)
            + "\n\nPlease fix these claims by adding proper citations from the retrieved chunks, "
            "or remove them if they cannot be supported."
        )

    user_message = (
        f"STUDENT PROFILE:\n{profile_text}\n\n"
        f"STUDENT QUESTION:\n{user_query}\n\n"
        f"RETRIEVED CATALOG CHUNKS:\n{chunks_text}\n\n"
        f"Based ONLY on the retrieved catalog chunks above, create a course plan "
        f"for this student. Follow the output format exactly as specified in your instructions."
        f"{revision_note}"
    )

    try:
        plan_text = _call_llm(PLANNER_PROMPT, user_message, max_tokens=4096)
    except Exception as e:
        print(f"  ✗ Planner LLM call failed: {e}")
        return {
            "course_plan": "Error generating plan. Please try again.",
            "plan_justifications": [],
            "citations": [],
            "assumptions": ["Plan generation failed due to an API error."],
            "final_output": "Error generating course plan. Please try again.",
        }

    # Parse the structured output
    citations = extract_citations(plan_text)
    assumptions = extract_assumptions(plan_text)

    # Extract justifications
    why_section = extract_section(plan_text, "WHY (Requirements/Prerequisites Satisfied)")
    justifications = []
    if why_section:
        for line in why_section.split("\n"):
            line = line.strip().lstrip("- ")
            if line:
                justifications.append({"justification": line})

    print(f"  Plan generated with {len(citations)} citation(s)")
    print(f"  Assumptions: {len(assumptions)}")

    return {
        "course_plan": plan_text,
        "plan_justifications": justifications,
        "citations": citations,
        "assumptions": assumptions,
        "final_output": plan_text,
    }


# ══════════════════════════════════════════════════════════════
# NODE 4 — VERIFIER
# ══════════════════════════════════════════════════════════════
def verifier_node(state: PlannerState) -> PlannerState:
    """
    Audit every claim in the course plan by checking citations
    against actual retrieved chunks.

    If any citation fails, sets verified=False and sends back
    to planner_node for revision. Max 3 retries.
    """
    print("\n" + "=" * 50)
    print("▶ NODE: verifier_node")
    print("=" * 50)

    course_plan = state.get("course_plan", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    retry_count = state.get("retry_count", 0)

    # Build set of valid chunk IDs
    valid_chunk_ids = {chunk.get("chunk_id", "") for chunk in retrieved_chunks}

    # Build verification prompt
    chunks_text = format_chunks_for_prompt(retrieved_chunks)
    user_message = (
        f"COURSE PLAN TO VERIFY:\n{course_plan}\n\n"
        f"AVAILABLE CATALOG CHUNKS:\n{chunks_text}\n\n"
        f"Verify every claim in this course plan. For each course recommendation:\n"
        f"1. Does it have a citation? (YES/NO)\n"
        f"2. Is the citation format correct: [Source: <filename>, Chunk: <chunk_id>, Section: <section>]? (YES/NO)\n"
        f"3. Is the prerequisite logic correct per the cited chunk? (YES/NO)\n\n"
        f"Output your verification report in the exact format specified in your instructions."
    )

    try:
        verification_text = _call_llm(VERIFIER_PROMPT, user_message, max_tokens=2048)
    except Exception as e:
        print(f"  ✗ Verifier LLM call failed: {e}")
        # On API failure, force pass to avoid infinite loop
        return {
            "verified": True,
            "verification_report": [{"error": str(e)}],
            "failed_citations": [],
            "retry_count": retry_count,
            "final_output": course_plan + "\n\n⚠ Verification skipped due to API error.",
        }

    print(f"\n  Verification Report:\n{verification_text[:500]}...")

    # Parse verification result
    overall_pass = "OVERALL: PASS" in verification_text.upper()

    # Extract failed claims
    failed_claims = []
    if "FAILED CLAIMS:" in verification_text.upper():
        failed_section = verification_text.upper().split("FAILED CLAIMS:")[1]
        for line in failed_section.split("\n"):
            line = line.strip().lstrip("- ")
            if line and line != "NONE" and line != "N/A" and line != "[]":
                failed_claims.append(line)

    # Also check extracted citations against valid chunk IDs
    citations = extract_citations(course_plan)
    import re
    for citation in citations:
        # Extract chunk_id from the citation string
        match = re.search(r'Chunk:\s*([^,\]]+)', citation)
        if match:
            cited_id = match.group(1).strip()
            if cited_id not in valid_chunk_ids:
                failed_claims.append(f"Invalid chunk_id in citation: {cited_id[:16]}...")

    verification_report = [{"report": verification_text, "overall_pass": overall_pass}]

    if overall_pass and not failed_claims:
        print("  ✓ VERIFICATION PASSED — all claims supported.")
        return {
            "verified": True,
            "verification_report": verification_report,
            "failed_citations": [],
            "retry_count": retry_count,
            "final_output": course_plan,
        }
    else:
        new_retry = retry_count + 1
        if new_retry >= 3:
            print(f"  ⚠ VERIFICATION FAILED — max retries ({new_retry}) reached. Forcing output with warning.")
            warning = (
                "\n\n⚠ WARNING: This plan could not be fully verified after 3 attempts. "
                "Some claims may not be properly supported by catalog evidence. "
                "Please verify with your academic advisor."
            )
            return {
                "verified": True,  # Force pass
                "verification_report": verification_report,
                "failed_citations": failed_claims,
                "retry_count": new_retry,
                "final_output": course_plan + warning,
            }
        else:
            print(f"  ✗ VERIFICATION FAILED — {len(failed_claims)} issue(s). Retry {new_retry}/3.")
            return {
                "verified": False,
                "verification_report": verification_report,
                "failed_citations": failed_claims,
                "retry_count": new_retry,
            }
