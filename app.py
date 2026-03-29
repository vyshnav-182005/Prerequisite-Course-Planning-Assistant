"""
Gradio UI for the Course Planning Assistant.

Provides three tabs:
    1. Upload Catalog — Upload and process PDF/HTML catalog documents
    2. Ask Questions — Ask course-related questions with student profile
    3. Plan My Term — Generate a full term plan

All tabs call the LangGraph pipeline via run_pipeline().
"""

import os
import sys
import traceback

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

import gradio as gr

from src.ingestion import load_documents
from src.vectorstore import build_index, clear_index, get_chunk_count
from src.graph import run_pipeline
from src.utils import parse_list_string


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _build_profile(
    completed_courses: str,
    target_major: str,
    catalog_year: str,
    target_term: str,
    max_credits: str,
    grades: str = "",
) -> dict:
    """
    Build a student profile dictionary from UI form inputs.

    Args:
        completed_courses: Comma-separated course codes.
        target_major: Student's intended major.
        catalog_year: e.g., '2024-2025'.
        target_term: 'Fall' or 'Spring'.
        max_credits: Max credits as string number.
        grades: Optional comma-separated 'COURSE:GRADE' pairs.

    Returns:
        Student profile dictionary.
    """
    profile = {}

    courses = parse_list_string(completed_courses)
    if courses:
        profile["completed_courses"] = courses
    else:
        profile["completed_courses"] = []

    if target_major and target_major.strip():
        profile["target_major"] = target_major.strip()

    if catalog_year and catalog_year.strip():
        profile["catalog_year"] = catalog_year.strip()

    if target_term and target_term.strip():
        profile["target_term"] = target_term.strip()

    if max_credits and max_credits.strip():
        try:
            profile["max_credits"] = int(max_credits.strip())
        except ValueError:
            pass

    if grades and grades.strip():
        grade_dict = {}
        for pair in grades.split(","):
            parts = pair.strip().split(":")
            if len(parts) == 2:
                grade_dict[parts[0].strip()] = parts[1].strip()
        if grade_dict:
            profile["grades"] = grade_dict

    return profile


# ══════════════════════════════════════════════════════════════
# TAB 1 — Upload Catalog
# ══════════════════════════════════════════════════════════════

def process_documents(files):
    """
    Process uploaded PDF/HTML files: ingest, chunk, and index.

    Args:
        files: List of uploaded file objects from Gradio.

    Returns:
        Status message string.
    """
    if not files:
        return "⚠ No files uploaded. Please upload at least one PDF or HTML file."

    try:
        # Get file paths
        file_paths = []
        for f in files:
            if hasattr(f, "name"):
                file_paths.append(f.name)
            elif isinstance(f, str):
                file_paths.append(f)
            else:
                file_paths.append(str(f))

        if not file_paths:
            return "⚠ No valid file paths found."

        # Clear existing index
        clear_index()

        # Load and chunk documents
        chunks = load_documents(file_paths)

        if not chunks:
            return "⚠ No text content could be extracted from the uploaded files."

        # Build vector index
        build_index(chunks)

        count = get_chunk_count()
        return (
            f"✓ Done! {count} chunks indexed from {len(file_paths)} document(s).\n\n"
            f"Files processed:\n"
            + "\n".join(f"  • {os.path.basename(p)}" for p in file_paths)
        )

    except Exception as e:
        traceback.print_exc()
        return f"✗ Error processing documents: {str(e)}"


# ══════════════════════════════════════════════════════════════
# TAB 2 — Ask Questions
# ══════════════════════════════════════════════════════════════

def ask_question(
    question: str,
    completed_courses: str,
    target_major: str,
    catalog_year: str,
    target_term: str,
    max_credits: str,
    grades: str,
):
    """
    Process a student's question through the LangGraph pipeline.

    Returns:
        Tuple of (full_response, citations_text).
    """
    if not question or not question.strip():
        return "⚠ Please enter a question.", ""

    # Check if documents are indexed
    if get_chunk_count() == 0:
        return (
            "⚠ No catalog documents are indexed yet.\n"
            "Please go to the 'Upload Catalog' tab and upload your catalog documents first.",
            "",
        )

    try:
        profile = _build_profile(
            completed_courses, target_major, catalog_year,
            target_term, max_credits, grades,
        )

        result = run_pipeline(question.strip(), profile)

        final_output = result.get("final_output", "No response generated.")
        citations = result.get("citations", [])
        citations_text = "\n".join(citations) if citations else "No citations found."

        return final_output, citations_text

    except Exception as e:
        traceback.print_exc()
        return f"✗ Error: {str(e)}", ""


# ══════════════════════════════════════════════════════════════
# TAB 3 — Plan My Term
# ══════════════════════════════════════════════════════════════

def generate_plan(
    completed_courses: str,
    target_major: str,
    catalog_year: str,
    target_term: str,
    max_credits: str,
    grades: str,
):
    """
    Generate a full term plan through the LangGraph pipeline.

    Returns:
        Tuple of (full_plan, citations_text, assumptions_text).
    """
    # Check if documents are indexed
    if get_chunk_count() == 0:
        return (
            "⚠ No catalog documents are indexed yet.\n"
            "Please go to the 'Upload Catalog' tab and upload your catalog documents first.",
            "",
            "",
        )

    try:
        profile = _build_profile(
            completed_courses, target_major, catalog_year,
            target_term, max_credits, grades,
        )

        query = (
            f"Generate a complete term plan for {target_term} "
            f"for a {target_major} major. "
            f"Maximum {max_credits} credits. "
            f"Courses already completed: {completed_courses}."
        )

        result = run_pipeline(query, profile)

        final_output = result.get("final_output", "No plan generated.")
        citations = result.get("citations", [])
        assumptions = result.get("assumptions", [])

        citations_text = "\n".join(citations) if citations else "No citations found."
        assumptions_text = "\n".join(f"• {a}" for a in assumptions) if assumptions else "No assumptions."

        return final_output, citations_text, assumptions_text

    except Exception as e:
        traceback.print_exc()
        return f"✗ Error: {str(e)}", "", ""


# ══════════════════════════════════════════════════════════════
# BUILD GRADIO APP
# ══════════════════════════════════════════════════════════════

def create_app() -> gr.Blocks:
    """
    Build and return the Gradio Blocks application with three tabs.

    Returns:
        Configured Gradio Blocks app.
    """
    with gr.Blocks(
        title="Course Planning Assistant",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="sky",
        ),
    ) as app:
        gr.Markdown(
            "# 🎓 Course Planning Assistant\n"
            "An AI-powered academic advisor using Retrieval-Augmented Generation (RAG) "
            "to help you plan your courses based on your actual course catalog."
        )

        # ── Tab 1: Upload Catalog ──────────────────────────
        with gr.Tab("📄 Upload Catalog"):
            gr.Markdown(
                "### Upload your course catalog documents\n"
                "Supported formats: **PDF** and **HTML**"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    file_upload = gr.File(
                        label="Select catalog files",
                        file_count="multiple",
                        file_types=[".pdf", ".html", ".htm"],
                        type="filepath",
                    )
                    process_btn = gr.Button("🔄 Process Documents", variant="primary")
                with gr.Column(scale=3):
                    upload_status = gr.Textbox(
                        label="Processing Status",
                        lines=8,
                        interactive=False,
                        placeholder="Upload documents and click 'Process Documents' to begin...",
                    )

            process_btn.click(
                fn=process_documents,
                inputs=[file_upload],
                outputs=[upload_status],
            )

        # ── Tab 2: Ask Questions ───────────────────────────
        with gr.Tab("❓ Ask Questions"):
            gr.Markdown(
                "### Ask questions about courses, prerequisites, and requirements\n"
                "Fill in your student profile and ask your question."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Student Profile**")
                    q_courses = gr.Textbox(
                        label="Completed Courses (comma-separated)",
                        placeholder="CS101, CS201, MATH120",
                    )
                    q_major = gr.Textbox(
                        label="Target Major",
                        placeholder="Computer Science",
                    )
                    q_year = gr.Textbox(
                        label="Catalog Year",
                        placeholder="2024-2025",
                    )
                    q_term = gr.Dropdown(
                        label="Target Term",
                        choices=["Fall", "Spring"],
                        value="Fall",
                    )
                    q_credits = gr.Textbox(
                        label="Max Credits",
                        placeholder="15",
                    )
                    q_grades = gr.Textbox(
                        label="Grades (optional, e.g. CS101:A,CS201:B+)",
                        placeholder="CS101:A, CS201:B+",
                    )

                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Your Question",
                        lines=3,
                        placeholder="Can I take CS301 if I've completed CS101 and MATH120?",
                    )
                    ask_btn = gr.Button("🔍 Get Answer", variant="primary")

                    answer_output = gr.Textbox(
                        label="Response",
                        lines=15,
                        interactive=False,
                    )
                    citations_output = gr.Textbox(
                        label="📎 Citations",
                        lines=5,
                        interactive=False,
                    )

            ask_btn.click(
                fn=ask_question,
                inputs=[question_input, q_courses, q_major, q_year, q_term, q_credits, q_grades],
                outputs=[answer_output, citations_output],
            )

        # ── Tab 3: Plan My Term ────────────────────────────
        with gr.Tab("📋 Plan My Term"):
            gr.Markdown(
                "### Generate a full term course plan\n"
                "Fill in your profile and click 'Generate Plan'."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Student Profile**")
                    p_courses = gr.Textbox(
                        label="Completed Courses (comma-separated)",
                        placeholder="CS101, CS201, MATH120",
                    )
                    p_major = gr.Textbox(
                        label="Target Major",
                        placeholder="Computer Science",
                    )
                    p_year = gr.Textbox(
                        label="Catalog Year",
                        placeholder="2024-2025",
                    )
                    p_term = gr.Dropdown(
                        label="Target Term",
                        choices=["Fall", "Spring"],
                        value="Fall",
                    )
                    p_credits = gr.Textbox(
                        label="Max Credits",
                        placeholder="15",
                    )
                    p_grades = gr.Textbox(
                        label="Grades (optional, e.g. CS101:A,CS201:B+)",
                        placeholder="CS101:A, CS201:B+",
                    )
                    plan_btn = gr.Button("📝 Generate Plan", variant="primary")

                with gr.Column(scale=2):
                    plan_output = gr.Textbox(
                        label="Course Plan",
                        lines=18,
                        interactive=False,
                    )
                    plan_citations = gr.Textbox(
                        label="📎 Citations",
                        lines=5,
                        interactive=False,
                    )
                    plan_assumptions = gr.Textbox(
                        label="⚠ Assumptions / Risks",
                        lines=5,
                        interactive=False,
                    )

            plan_btn.click(
                fn=generate_plan,
                inputs=[p_courses, p_major, p_year, p_term, p_credits, p_grades],
                outputs=[plan_output, plan_citations, plan_assumptions],
            )

        gr.Markdown(
            "---\n"
            "*Powered by Ollama + LangGraph + ChromaDB | "
            "Always verify recommendations with your academic advisor.*"
        )

    return app


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🎓 Starting Course Planning Assistant...")
    print("   Make sure your .env file has OLLAMA_HOST set if it is not default.\n")

    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7861")),
        share=False,
    )
