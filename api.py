"""
FastAPI backend for the Course Planning Assistant.

Exposes three endpoints consumed by the Next.js frontend:
    POST /api/upload   — ingest catalog documents into ChromaDB
    POST /api/plan     — run the CrewAI planning pipeline
    GET  /api/status   — return the number of indexed catalog chunks
    DELETE /api/index  — clear the ChromaDB index
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

from src.ingestion import load_documents
from src.vectorstore import build_index, clear_index, get_chunk_count
from src.crew import run_pipeline

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Course Planning Assistant API",
    version="2.0.0",
    description="AI-powered course planning using CrewAI agents and ChromaDB RAG.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
# Request / Response models
# ══════════════════════════════════════════════════════════════

class StudentProfile(BaseModel):
    completed_courses: List[str] = Field(default_factory=list)
    target_major: Optional[str] = None
    catalog_year: Optional[str] = None
    target_term: Optional[str] = None          # "Fall" or "Spring"
    max_credits: Optional[int] = None
    grades: Optional[dict] = None
    transfer_credits: Optional[List[str]] = None


class PlanRequest(BaseModel):
    user_query: str
    student_profile: StudentProfile


class PlanResponse(BaseModel):
    is_profile_complete: bool
    clarifying_questions: List[str]
    missing_fields: List[str]
    course_plan: str
    final_output: str
    citations: List[str]
    assumptions: List[str]
    verified: bool
    failed_citations: List[str]


class StatusResponse(BaseModel):
    chunk_count: int
    ready: bool


class UploadResponse(BaseModel):
    message: str
    chunks_indexed: int


# ══════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════

@app.get("/api/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    """Return the number of catalog chunks currently indexed."""
    count = get_chunk_count()
    return StatusResponse(chunk_count=count, ready=count > 0)


@app.post("/api/upload", response_model=UploadResponse)
async def upload_catalog(
    files: List[UploadFile] = File(...),
    clear_existing: bool = Form(default=False),
) -> UploadResponse:
    """
    Upload one or more PDF/HTML catalog documents.

    The files are saved to a temporary directory, ingested into chunks,
    and indexed into ChromaDB. Returns the number of chunks indexed.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # Optionally clear the existing index
    if clear_existing:
        try:
            clear_index()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to clear index: {exc}")

    # Save uploaded files to a temp directory
    tmp_dir = tempfile.mkdtemp(prefix="catalog_upload_")
    saved_paths: List[str] = []
    try:
        for upload in files:
            filename = upload.filename or "document"
            dest = os.path.join(tmp_dir, filename)
            content = await upload.read()
            with open(dest, "wb") as fh:
                fh.write(content)
            saved_paths.append(dest)

        # Ingest and index
        chunks = load_documents(saved_paths)
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No content could be extracted from the uploaded files.",
            )
        build_index(chunks)
        return UploadResponse(
            message=f"Successfully indexed {len(chunks)} chunk(s) from {len(saved_paths)} file(s).",
            chunks_indexed=len(chunks),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload processing error: {exc}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/plan", response_model=PlanResponse)
def plan_courses(request: PlanRequest) -> PlanResponse:
    """
    Run the CrewAI planning pipeline for the given student profile and query.

    Returns a structured plan with citations, verification status, and any
    clarifying questions if the profile is incomplete.
    """
    if not request.user_query.strip():
        raise HTTPException(status_code=400, detail="user_query must not be empty.")

    profile_dict = request.student_profile.model_dump(exclude_none=False)

    try:
        result = run_pipeline(
            user_query=request.user_query,
            student_profile=profile_dict,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    return PlanResponse(
        is_profile_complete=result.get("is_profile_complete", False),
        clarifying_questions=result.get("clarifying_questions", []),
        missing_fields=result.get("missing_fields", []),
        course_plan=result.get("course_plan", ""),
        final_output=result.get("final_output", ""),
        citations=result.get("citations", []),
        assumptions=result.get("assumptions", []),
        verified=result.get("verified", False),
        failed_citations=result.get("failed_citations", []),
    )


@app.delete("/api/index")
def delete_index() -> dict:
    """Clear all documents from the ChromaDB index."""
    try:
        clear_index()
        return {"message": "Index cleared successfully."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to clear index: {exc}")


@app.get("/")
def root() -> dict:
    return {"message": "Course Planning Assistant API v2.0 (CrewAI)", "docs": "/docs"}
