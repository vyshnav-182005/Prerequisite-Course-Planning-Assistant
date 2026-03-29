"""
LangGraph pipeline for the Course Planning Assistant.

Builds a StateGraph with four nodes (intake, retriever, planner, verifier)
and conditional edges for routing based on profile completeness and
verification results. Includes a retry loop for failed verifications.
"""

from langgraph.graph import StateGraph, END

from src.state import PlannerState
from src.agents import (
    intake_node,
    retriever_node,
    planner_node,
    verifier_node,
)


# ── Routing Functions ─────────────────────────────────────────

def route_after_intake(state: PlannerState) -> str:
    """
    Route after the intake node.

    If the student profile is complete, proceed to the retriever node.
    If incomplete, go to END and return clarifying questions.

    Args:
        state: Current pipeline state.

    Returns:
        "retriever" or END.
    """
    if state.get("is_profile_complete", False):
        print("  → Routing to: retriever")
        return "retriever"
    print("  → Routing to: END (profile incomplete)")
    return END


def route_after_verifier(state: PlannerState) -> str:
    """
    Route after the verifier node.

    If verification passed, go to END with the final plan.
    If not verified but retries remain, go back to the planner for revision.
    If retries exhausted, go to END with a warning.

    Args:
        state: Current pipeline state.

    Returns:
        "planner" or END.
    """
    if state.get("verified", False):
        print("  → Routing to: END (verified)")
        return END
    if state.get("retry_count", 0) >= 3:
        print("  → Routing to: END (max retries reached)")
        return END
    print("  → Routing to: planner (revision needed)")
    return "planner"


# ── Graph Builder ─────────────────────────────────────────────

def build_graph():
    """
    Build and compile the full LangGraph pipeline.

    Graph flow:
        START → intake → (complete?) → retriever → planner → verifier
                            ↓ (incomplete)                     ↓
                           END                    ├── (pass) → END
                                                  └── (fail, retries < 3) → planner
                                                  └── (fail, retries ≥ 3) → END

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    graph = StateGraph(PlannerState)

    # Add nodes
    graph.add_node("intake", intake_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("planner", planner_node)
    graph.add_node("verifier", verifier_node)

    # Set entry point
    graph.set_entry_point("intake")

    # Conditional: intake → retriever | END
    graph.add_conditional_edges(
        "intake",
        route_after_intake,
        {
            "retriever": "retriever",
            END: END,
        },
    )

    # Edge: retriever → planner
    graph.add_edge("retriever", "planner")

    # Edge: planner → verifier
    graph.add_edge("planner", "verifier")

    # Conditional: verifier → planner (retry) | END
    graph.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "planner": "planner",
            END: END,
        },
    )

    return graph.compile()


# ── Compiled graph (singleton) ────────────────────────────────
planning_graph = build_graph()


# ── Pipeline Runner ───────────────────────────────────────────

def run_pipeline(user_query: str, student_profile: dict) -> dict:
    """
    Run the full LangGraph pipeline.

    Initializes the state with the user's query and student profile,
    then invokes the compiled graph. Returns the final state with
    all outputs.

    Args:
        user_query: The student's question or request.
        student_profile: Dictionary with student profile fields.

    Returns:
        Final PlannerState dictionary with all pipeline outputs.
    """
    initial_state: PlannerState = {
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

    print("\n" + "█" * 60)
    print("██  COURSE PLANNING ASSISTANT — Pipeline Start")
    print("█" * 60)
    print(f"  Query: {user_query[:80]}...")
    print(f"  Profile keys: {list(student_profile.keys())}")

    try:
        result = planning_graph.invoke(initial_state)
    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        result = {**initial_state, "final_output": f"Pipeline error: {e}"}

    print("\n" + "█" * 60)
    print("██  Pipeline Complete")
    print("█" * 60)

    return result
