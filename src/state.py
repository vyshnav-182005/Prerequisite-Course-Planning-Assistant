"""
State definitions for the Course Planning Assistant LangGraph pipeline.

Defines the shared state (PlannerState) used across all graph nodes,
plus Pydantic models for structured data validation.
"""

from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field


class StudentProfile(BaseModel):
    """Pydantic model representing a student's academic profile."""

    completed_courses: List[str] = Field(
        default_factory=list,
        description="List of course codes the student has completed."
    )
    target_major: Optional[str] = Field(
        default=None,
        description="The student's intended major or program."
    )
    catalog_year: Optional[str] = Field(
        default=None,
        description="The catalog year the student is following (e.g., '2024-2025')."
    )
    target_term: Optional[str] = Field(
        default=None,
        description="The term the student is planning for (Fall or Spring)."
    )
    max_credits: Optional[int] = Field(
        default=None,
        description="Maximum number of credits the student wants to take."
    )
    grades: Optional[dict] = Field(
        default=None,
        description="Dictionary mapping course codes to grades earned."
    )
    transfer_credits: Optional[List[str]] = Field(
        default=None,
        description="List of courses transferred from another institution."
    )


class RetrievedChunk(BaseModel):
    """Pydantic model representing a single retrieved document chunk."""

    chunk_id: str = Field(
        description="Unique identifier for this chunk (UUID4)."
    )
    source: str = Field(
        description="Source filename or URL this chunk came from."
    )
    section: str = Field(
        description="Detected section heading for this chunk."
    )
    content: str = Field(
        description="The actual text content of this chunk."
    )
    page: Optional[int] = Field(
        default=None,
        description="Page number in the source document (PDF only)."
    )


class PlannerState(TypedDict):
    """
    Shared LangGraph state passed between all nodes in the pipeline.

    This TypedDict defines every field that nodes can read/write.
    """

    # --- Input ---
    user_query: str
    student_profile: dict

    # --- Intake node output ---
    is_profile_complete: bool
    clarifying_questions: List[str]
    missing_fields: List[str]

    # --- Retriever node output ---
    retrieved_chunks: List[dict]
    retrieval_query: str

    # --- Planner node output ---
    course_plan: str
    plan_justifications: List[dict]

    # --- Verifier node output ---
    verified: bool
    verification_report: List[dict]
    failed_citations: List[str]
    retry_count: int

    # --- Final output ---
    final_output: str
    citations: List[str]
    assumptions: List[str]
