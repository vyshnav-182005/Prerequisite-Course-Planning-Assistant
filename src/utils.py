"""
Shared utility functions for the Course Planning Assistant.

Provides helper functions for text parsing, safe dictionary access,
citation extraction, and other common operations used across modules.
"""

import re
from typing import List, Optional


def safe_get(d: dict, key: str, default=None):
    """Safely get a value from a dictionary, returning default if missing or None."""
    if d is None:
        return default
    value = d.get(key, default)
    return value if value is not None else default


def parse_list_string(text: str) -> List[str]:
    """
    Parse a comma-separated string into a list of stripped strings.

    Args:
        text: Comma-separated string (e.g., "CS101, CS201, MATH120").

    Returns:
        List of trimmed, non-empty strings.
    """
    if not text or not text.strip():
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def extract_citations(text: str) -> List[str]:
    """
    Extract citation strings from a response text.

    Looks for patterns like [Source: <filename>, Chunk: <id>, Section: <section>].

    Args:
        text: The full response text from the planner node.

    Returns:
        List of citation strings found in the text.
    """
    pattern = r'\[Source:\s*[^,\]]+,\s*Chunk:\s*[^,\]]+,\s*Section:\s*[^\]]+\]'
    return re.findall(pattern, text)


def extract_section(text: str, section_name: str) -> str:
    """
    Extract a named section from structured output text.

    Looks for content between a section header and the next section header
    or end of text.

    Args:
        text: The full structured response text.
        section_name: The name of the section to extract (e.g., "ANSWER / PLAN").

    Returns:
        The content of the section, or empty string if not found.
    """
    pattern = rf'{re.escape(section_name)}:\s*\n(.*?)(?=\n[A-Z][A-Z /]+:\s*\n|\Z)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_assumptions(text: str) -> List[str]:
    """
    Extract assumptions from the ASSUMPTIONS / NOT IN CATALOG section.

    Args:
        text: The full structured response text.

    Returns:
        List of assumption strings.
    """
    section = extract_section(text, "ASSUMPTIONS / NOT IN CATALOG")
    if not section:
        return []
    lines = [line.strip().lstrip("- ").strip() for line in section.split("\n")]
    return [line for line in lines if line]


def format_chunks_for_prompt(chunks: List[dict]) -> str:
    """
    Format retrieved chunks into a string suitable for inclusion in an LLM prompt.

    Args:
        chunks: List of chunk dictionaries with 'content', 'source',
                'chunk_id', and 'section' keys.

    Returns:
        Formatted string with all chunks labeled by source and ID.
    """
    if not chunks:
        return "No catalog chunks available."

    formatted = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        chunk_id = chunk.get("chunk_id", "unknown")
        section = chunk.get("section", "unknown")
        content = chunk.get("content", "")
        formatted.append(
            f"--- Chunk {i} ---\n"
            f"Source: {source}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Section: {section}\n"
            f"Content:\n{content}\n"
        )
    return "\n".join(formatted)


def format_student_profile(profile: dict) -> str:
    """
    Format a student profile dictionary into a readable string for LLM prompts.

    Args:
        profile: Dictionary with student profile fields.

    Returns:
        Formatted string describing the student's profile.
    """
    if not profile:
        return "No student profile provided."

    lines = []
    fields = [
        ("completed_courses", "Completed Courses"),
        ("target_major", "Target Major"),
        ("catalog_year", "Catalog Year"),
        ("target_term", "Target Term"),
        ("max_credits", "Max Credits"),
        ("grades", "Grades"),
        ("transfer_credits", "Transfer Credits"),
    ]
    for key, label in fields:
        value = profile.get(key)
        if value is not None:
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value) if value else "None"
            lines.append(f"- {label}: {value}")
    return "\n".join(lines) if lines else "No student profile fields set."


def has_abstention_phrase(text: str) -> bool:
    """
    Check if a response text contains an abstention / 'not found' phrase.

    Args:
        text: The response text to check.

    Returns:
        True if the text indicates the system abstained from answering.
    """
    abstention_phrases = [
        "not found in catalog",
        "i don't have that information",
        "not in the provided catalog",
        "please check with your academic advisor",
        "information is not available",
        "cannot determine from the provided",
        "not found in the chunks",
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in abstention_phrases)
