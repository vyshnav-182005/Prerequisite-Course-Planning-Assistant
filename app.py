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
# CUSTOM CSS FOR PROFESSIONAL STYLING
# ══════════════════════════════════════════════════════════════

CUSTOM_CSS = """
    :root {
        --primary-color: #1e40af;
        --secondary-color: #0284c7;
        --success-color: #16a34a;
        --warning-color: #ea580c;
        --danger-color: #dc2626;
        --bg-light: #f9fafb;
        --border-color: #e5e7eb;
    }

    body {
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
        background: linear-gradient(135deg, #f0f4f8 0%, #f8fafc 100%);
    }

    .gradio-container {
        max-width: 1400px;
        margin: 0 auto;
    }

    .gr-markdown {
        font-family: 'Segoe UI', Arial, sans-serif !important;
    }

    .title-section {
        background: linear-gradient(135deg, #1e40af 0%, #0284c7 100%);
        color: white;
        padding: 40px 20px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(30, 64, 175, 0.2);
    }

    .title-section h1 {
        font-size: 2.5em;
        margin: 0 0 10px 0;
        font-weight: 700;
    }

    .title-section p {
        margin: 0;
        font-size: 1.1em;
        opacity: 0.95;
    }

    .instruction-box {
        background-color: #eff6ff;
        border-left: 4px solid #0284c7;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
        font-size: 0.95em;
        line-height: 1.6;
    }

    .instruction-box h4 {
        color: #0284c7;
        margin-top: 0;
        font-weight: 600;
    }

    .instruction-box ol, .instruction-box ul {
        margin: 10px 0;
        padding-left: 20px;
    }

    .instruction-box li {
        margin: 5px 0;
    }

    .example-box {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        overflow-x: auto;
    }

    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #ea580c;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        font-size: 0.95em;
    }

    .gr-tabs {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .gr-tab-nav button {
        font-size: 1.05em;
        font-weight: 500;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }

    .gr-tab-nav button.selected {
        border-bottom-color: #1e40af;
        color: #1e40af;
    }

    .gr-box {
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        background-color: white;
    }

    .gr-button-primary {
        background: linear-gradient(135deg, #1e40af 0%, #0284c7 100%) !important;
        border: none !important;
        font-weight: 600;
        padding: 10px 20px;
        border-radius: 8px;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .gr-button-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.3);
    }

    .gr-textbox input, .gr-textbox textarea {
        font-family: 'Segoe UI', Arial, sans-serif !important;
        font-size: 0.95em;
        border-radius: 6px;
    }

    .gr-label {
        font-weight: 600;
        color: #1f2937;
        font-size: 0.95em;
    }

    .response-section {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #e5e7eb;
        line-height: 1.7;
    }

    .response-header {
        color: #1e40af;
        font-weight: 700;
        font-size: 1.1em;
        margin-bottom: 10px;
    }

    .course-list {
        background-color: #f9fafb;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
    }

    .footer-text {
        text-align: center;
        color: #6b7280;
        font-size: 0.9em;
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #e5e7eb;
    }
"""

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


def format_response(response_text: str) -> str:
    """
    Format AI response for better readability.

    Args:
        response_text: Raw response from the AI model.

    Returns:
        Formatted response with better structure.
    """
    formatted = response_text.strip()

    # Add visual separators for sections
    formatted = formatted.replace("ANSWER / PLAN:", "📋 **RECOMMENDED COURSES:**")
    formatted = formatted.replace("WHY (Requirements/Prerequisites Satisfied):", "✅ **WHY THIS WORKS:**")
    formatted = formatted.replace("CITATIONS:", "📚 **SOURCES:**")
    formatted = formatted.replace("CLARIFYING QUESTIONS (if needed):", "❓ **QUESTIONS FOR YOU:**")
    formatted = formatted.replace("ASSUMPTIONS / NOT IN CATALOG:", "⚠️ **ASSUMPTIONS:**")

    return formatted


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
        Tuple of (formatted_response, citations_text).
    """
    if not question or not question.strip():
        return "⚠️ **Please enter a question** to get started!", ""

    # Check if documents are indexed
    if get_chunk_count() == 0:
        return (
            "⚠️ **No catalog documents indexed yet**\n\n"
            "Please go to the **'📄 Upload Catalog'** tab and upload your course catalog PDF first.\n\n"
            "Once uploaded, come back here and ask your question!",
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

        # Format the main response
        formatted_output = format_response(final_output)

        citations_text = "\n".join(f"• {c}" for c in citations) if citations else "No citations found."

        return formatted_output, citations_text

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ **Error processing your question:** {str(e)}", ""


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
        Tuple of (formatted_plan, citations_text, assumptions_text).
    """
    # Check if documents are indexed
    if get_chunk_count() == 0:
        return (
            "⚠️ **No catalog documents indexed yet**\n\n"
            "Please go to the **'📄 Upload Catalog'** tab and upload your course catalog PDF first.",
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

        # Format the main plan
        formatted_plan = format_response(final_output)

        citations_text = "\n".join(f"• {c}" for c in citations) if citations else "No citations found."
        assumptions_text = "\n".join(f"• {a}" for a in assumptions) if assumptions else "No assumptions."

        return formatted_plan, citations_text, assumptions_text

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ **Error generating plan:** {str(e)}", "", ""


# ══════════════════════════════════════════════════════════════
# BUILD GRADIO APP
# ══════════════════════════════════════════════════════════════

def create_app() -> gr.Blocks:
    """
    Build and return the Gradio Blocks application with three tabs.

    Returns:
        Configured Gradio Blocks app.
    """
    with gr.Blocks(title="Course Planning Assistant") as app:

        # ═══════════════════════════════════════════════════════════
        # HEADER SECTION
        # ═══════════════════════════════════════════════════════════
        with gr.Column(elem_classes="title-section"):
            gr.Markdown(
                "# 🎓 Course Planning Assistant\n"
                "Your AI-powered academic advisor using Retrieval-Augmented Generation (RAG)"
            )
            gr.Markdown(
                "Get personalized course recommendations based on your actual course catalog, "
                "prerequisites, and degree requirements."
            )

        # ═══════════════════════════════════════════════════════════
        # MAIN TABS
        # ═══════════════════════════════════════════════════════════

        # ── Tab 1: Upload Catalog ──────────────────────────────────
        with gr.Tab("📄 Upload Catalog"):
            gr.Markdown(
                "## Upload Your Course Catalog\n"
                "Start by uploading your institution's course catalog document."
            )

            with gr.Column(elem_classes="instruction-box"):
                gr.Markdown(
                    "### 📋 How to Use\n"
                    "1. **Download or prepare** your course catalog as a PDF or HTML file\n"
                    "2. **Upload the file** using the button below\n"
                    "3. **Wait** for the system to process and index all courses\n"
                    "4. Once complete, go to **'Ask Questions'** or **'Plan My Term'** tabs"
                )

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📁 Select Files")
                    file_upload = gr.File(
                        label="Choose catalog files (PDF or HTML)",
                        file_count="multiple",
                        file_types=[".pdf", ".html", ".htm"],
                        type="filepath",
                    )
                    process_btn = gr.Button(
                        "🔄 Process Documents",
                        variant="primary",
                        size="lg",
                    )

                with gr.Column(scale=3):
                    gr.Markdown("### ✅ Status")
                    upload_status = gr.Textbox(
                        label="Processing Status",
                        lines=10,
                        interactive=False,
                        placeholder="Upload your catalog and click 'Process Documents'...",
                    )

            with gr.Column(elem_classes="example-box"):
                gr.Markdown(
                    "**✨ Pro Tip:** For best results, use a comprehensive catalog that includes:\n"
                    "• All available courses with course codes\n"
                    "• Prerequisites for each course\n"
                    "• Degree requirements by major\n"
                    "• Credit hours per course"
                )

            process_btn.click(
                fn=process_documents,
                inputs=[file_upload],
                outputs=[upload_status],
            )

        # ── Tab 2: Ask Questions ───────────────────────────────────
        with gr.Tab("❓ Ask Questions"):
            gr.Markdown(
                "## Ask Questions About Your Courses\n"
                "Get answers to specific questions about prerequisites, course availability, "
                "and course planning."
            )

            with gr.Column(elem_classes="instruction-box"):
                gr.Markdown(
                    "### 🎯 About This Tab\n"
                    "**What to do:**\n"
                    "1. Fill in your **Student Profile** (left side)\n"
                    "2. Type your specific question (middle)\n"
                    "3. Click **'Get Answer'** to receive a response\n\n"
                    "**Example Questions:**\n"
                    "• \"What courses can I take in Semester 2?\"\n"
                    "• \"What are the prerequisites for CS301?\"\n"
                    "• \"Can I take Database and Web Development together?\""
                )

            with gr.Row():
                # Left Column: Student Profile
                with gr.Column(scale=1):
                    gr.Markdown("### 👤 Your Profile")
                    with gr.Group():
                        q_courses = gr.Textbox(
                            label="✓ Completed Courses",
                            placeholder="e.g., 23MAT106, 23AID101, 23AID103",
                            info="Comma-separated course codes you've finished",
                        )
                        q_major = gr.Textbox(
                            label="🎓 Target Major",
                            placeholder="e.g., Computer Science",
                            info="Your degree program",
                        )
                        q_year = gr.Textbox(
                            label="📅 Catalog Year",
                            placeholder="e.g., 2024-2025",
                            info="Academic year of your catalog",
                        )
                        q_term = gr.Dropdown(
                            label="🗓️ Target Term",
                            choices=["Fall", "Spring", "Summer"],
                            value="Fall",
                            info="Which semester you're planning for",
                        )
                        q_credits = gr.Textbox(
                            label="📊 Max Credits",
                            placeholder="e.g., 15",
                            info="Maximum credits you can take",
                        )
                        q_grades = gr.Textbox(
                            label="⭐ Grades (Optional)",
                            placeholder="e.g., 23MAT106:A, 23AID101:B+",
                            info="Your grades help with prerequisite analysis",
                        )

                # Middle Column: Question & Button
                with gr.Column(scale=1):
                    gr.Markdown("### ❓ Your Question")
                    question_input = gr.Textbox(
                        label="What would you like to know?",
                        lines=4,
                        placeholder="Ask anything about courses, prerequisites, or planning...",
                    )
                    ask_btn = gr.Button(
                        "🔍 Get Answer",
                        variant="primary",
                        size="lg",
                        scale=1,
                    )

                # Right Column: Responses
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Response")
                    answer_output = gr.Markdown(
                        label="Answer",
                        value="Your answer will appear here...",
                    )

                    gr.Markdown("### 📚 Sources")
                    citations_output = gr.Textbox(
                        label="Citations from Catalog",
                        lines=6,
                        interactive=False,
                        value="Citations will appear here...",
                    )

            ask_btn.click(
                fn=ask_question,
                inputs=[question_input, q_courses, q_major, q_year, q_term, q_credits, q_grades],
                outputs=[answer_output, citations_output],
            )

        # ── Tab 3: Plan My Term ────────────────────────────────────
        with gr.Tab("📋 Plan My Term"):
            gr.Markdown(
                "## Generate Your Perfect Term Plan\n"
                "Get a complete course recommendation for your upcoming term, with all "
                "prerequisites verified."
            )

            with gr.Column(elem_classes="instruction-box"):
                gr.Markdown(
                    "### 📝 Create Your Plan\n"
                    "1. **Enter your profile** with completed courses and constraints\n"
                    "2. **Click 'Generate Plan'** to create your ideal schedule\n"
                    "3. **Review recommendations** with full citing and prerequisites\n"
                    "4. **Verify with your advisor** before registering"
                )

            with gr.Row():
                # Left Column: Student Profile
                with gr.Column(scale=1):
                    gr.Markdown("### 👤 Your Profile")
                    with gr.Group():
                        p_courses = gr.Textbox(
                            label="✓ Completed Courses",
                            placeholder="e.g., 23MAT106, 23AID101, 23AID103",
                            info="Comma-separated course codes",
                        )
                        p_major = gr.Textbox(
                            label="🎓 Target Major",
                            placeholder="e.g., Computer Science",
                        )
                        p_year = gr.Textbox(
                            label="📅 Catalog Year",
                            placeholder="e.g., 2024-2025",
                        )
                        p_term = gr.Dropdown(
                            label="🗓️ Target Term",
                            choices=["Fall", "Spring", "Summer"],
                            value="Fall",
                        )
                        p_credits = gr.Textbox(
                            label="📊 Max Credits",
                            placeholder="e.g., 15",
                        )
                        p_grades = gr.Textbox(
                            label="⭐ Grades (Optional)",
                            placeholder="e.g., 23MAT106:A, 23AID101:B+",
                        )

                    plan_btn = gr.Button(
                        "📝 Generate Plan",
                        variant="primary",
                        size="lg",
                    )

                # Right Column: Results
                with gr.Column(scale=2):
                    gr.Markdown("### 📚 Your Course Plan")
                    plan_output = gr.Markdown(
                        label="Course Plan",
                        value="Your personalized plan will appear here...",
                    )

                    gr.Markdown("### 📚 Sources & Citations")
                    plan_citations = gr.Textbox(
                        label="Catalog Sources",
                        lines=5,
                        interactive=False,
                        value="Citations will appear here...",
                    )

                    gr.Markdown("### ⚠️ Important Notes")
                    plan_assumptions = gr.Textbox(
                        label="Assumptions & Risks",
                        lines=5,
                        interactive=False,
                        value="Any assumptions or risks will be noted here...",
                    )

            plan_btn.click(
                fn=generate_plan,
                inputs=[p_courses, p_major, p_year, p_term, p_credits, p_grades],
                outputs=[plan_output, plan_citations, plan_assumptions],
            )

        # ═══════════════════════════════════════════════════════════
        # FOOTER
        # ═══════════════════════════════════════════════════════════
        with gr.Column(elem_classes="footer-text"):
            gr.Markdown(
                "---\n"
                "### ⚠️ Important Disclaimer\n"
                "**Always verify all recommendations with your academic advisor before registering.**\n\n"
                "*Powered by Ollama • LangGraph • ChromaDB • Gradio*"
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
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="sky",
        ),
    )
