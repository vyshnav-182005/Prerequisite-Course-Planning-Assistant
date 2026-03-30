"""
Evaluation script for the Course Planning Assistant.

Runs all 25 test queries through the LangGraph pipeline and computes:
    - Citation Coverage Rate
    - Eligibility Correctness
    - Abstention Accuracy

Also prints example transcripts for each query type.
"""

import os
import sys
import json
import time
from typing import List

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

from src.crew import run_pipeline
from src.utils import extract_citations, has_abstention_phrase
from evaluation.test_queries import ALL_QUERIES


# ── Individual Check Functions ────────────────────────────────

def check_has_citation(response: dict) -> bool:
    """
    Check if response contains at least one citation.

    Args:
        response: The pipeline result dictionary.

    Returns:
        True if at least one citation is found.
    """
    # Check citations list in state
    citations = response.get("citations", [])
    if citations:
        return True

    # Also check the final_output text for citation patterns
    final_output = response.get("final_output", "")
    found = extract_citations(final_output)
    return len(found) > 0


def check_abstained(response: dict) -> bool:
    """
    Check if response correctly abstained from answering.

    Looks for abstention keywords in the final output.

    Args:
        response: The pipeline result dictionary.

    Returns:
        True if the response indicates abstention.
    """
    final_output = response.get("final_output", "")
    return has_abstention_phrase(final_output)


def extract_decision(response: dict) -> str:
    """
    Extract eligible/not_eligible/abstain decision from response.

    Args:
        response: The pipeline result dictionary.

    Returns:
        One of: 'eligible', 'not_eligible', 'abstain', 'unknown'.
    """
    final_output = response.get("final_output", "").lower()

    if has_abstention_phrase(response.get("final_output", "")):
        return "abstain"

    eligibility_positive = [
        "you are eligible",
        "you meet the prerequisites",
        "you can take",
        "eligible to enroll",
        "prerequisites are met",
        "you have completed the required",
        "you satisfy",
    ]
    eligibility_negative = [
        "you are not eligible",
        "not eligible",
        "you do not meet",
        "cannot take",
        "missing prerequisite",
        "you have not completed",
        "prerequisite not met",
        "you need to complete",
    ]

    for phrase in eligibility_negative:
        if phrase in final_output:
            return "not_eligible"

    for phrase in eligibility_positive:
        if phrase in final_output:
            return "eligible"

    return "unknown"


# ── Metric Computation ────────────────────────────────────────

def compute_citation_coverage(results: list) -> float:
    """
    Compute the percentage of responses that have at least one citation.

    Args:
        results: List of result dictionaries.

    Returns:
        Citation coverage percentage (0-100).
    """
    if not results:
        return 0.0
    cited = sum(1 for r in results if r.get("has_citation", False))
    return round((cited / len(results)) * 100, 1)


def compute_eligibility(results: list) -> float:
    """
    Compute the percentage of prereq checks with correct eligibility decision.

    Only evaluates queries of type 'prereq_check'.

    Args:
        results: List of result dictionaries.

    Returns:
        Eligibility correctness percentage (0-100).
    """
    prereq_results = [r for r in results if r.get("type") == "prereq_check"]
    if not prereq_results:
        return 0.0

    correct = 0
    for r in prereq_results:
        decision = r.get("decision", "unknown")
        expected = r.get("expected_decision", "")

        # If expected is "eligible_or_not_eligible", any non-abstain answer counts
        if expected == "eligible_or_not_eligible":
            if decision in ("eligible", "not_eligible"):
                correct += 1
        elif expected == "not_eligible":
            if decision == "not_eligible":
                correct += 1
        elif expected == "eligible":
            if decision == "eligible":
                correct += 1
        elif expected == "abstain":
            if decision == "abstain":
                correct += 1

    return round((correct / len(prereq_results)) * 100, 1)


def compute_abstention(results: list) -> float:
    """
    Compute the percentage of not-in-docs queries that correctly abstained.

    Only evaluates queries of type 'not_in_docs'.

    Args:
        results: List of result dictionaries.

    Returns:
        Abstention accuracy percentage (0-100).
    """
    trick_results = [r for r in results if r.get("type") == "not_in_docs"]
    if not trick_results:
        return 0.0

    correct = sum(1 for r in trick_results if r.get("abstained", False))
    return round((correct / len(trick_results)) * 100, 1)


# ── Transcript Printing ──────────────────────────────────────

def print_example_transcript(results: list, query_type: str) -> None:
    """
    Print one full example transcript for the given query type.

    Args:
        results: List of result dictionaries.
        query_type: The type of query to find an example for.
    """
    for r in results:
        if r.get("type") == query_type:
            print(f"\n{'─' * 60}")
            print(f"EXAMPLE TRANSCRIPT — Type: {query_type}")
            print(f"{'─' * 60}")
            print(f"  Query ID: {r.get('id', 'N/A')}")
            print(f"  Query: {r.get('query', 'N/A')}")
            print(f"  Has Citation: {r.get('has_citation', False)}")
            print(f"  Abstained: {r.get('abstained', False)}")
            print(f"  Decision: {r.get('decision', 'unknown')}")
            print(f"\n  Response (first 500 chars):")

            response = r.get("response", {})
            final_output = response.get("final_output", "No output")
            print(f"  {final_output[:500]}")
            print(f"{'─' * 60}\n")
            return

    print(f"\n  [No example found for type: {query_type}]")


# ── Main Evaluation Runner ───────────────────────────────────

def run_evaluation(queries: list = None, graph_runner=None) -> dict:
    """
    Run all test queries through the pipeline and compute metrics.

    Args:
        queries: List of query dicts. Defaults to ALL_QUERIES.
        graph_runner: Pipeline function. Defaults to run_pipeline.

    Returns:
        Dictionary with metrics and full results.
    """
    if queries is None:
        queries = ALL_QUERIES
    if graph_runner is None:
        graph_runner = run_pipeline

    results = []
    total = len(queries)

    print("\n" + "═" * 60)
    print("   COURSE PLANNING ASSISTANT — EVALUATION")
    print("═" * 60)
    print(f"   Running {total} test queries...\n")

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{total}] Running query {query['id']}: {query['query'][:60]}...")

        start_time = time.time()
        try:
            response = graph_runner(
                query["query"],
                query["student_profile"],
            )
        except Exception as e:
            print(f"  ✗ Query {query['id']} failed: {e}")
            response = {"final_output": f"Error: {e}", "citations": [], "assumptions": []}

        elapsed = time.time() - start_time
        print(f"  ✓ Completed in {elapsed:.1f}s")

        result = {
            "id": query["id"],
            "type": query["type"],
            "query": query["query"],
            "expected_decision": query.get("expected_decision", ""),
            "expected_has_citation": query.get("expected_has_citation", True),
            "response": response,
            "has_citation": check_has_citation(response),
            "abstained": check_abstained(response),
            "decision": extract_decision(response),
            "time_seconds": round(elapsed, 1),
        }
        results.append(result)

    # ── Compute metrics ───────────────────────────────────
    citation_coverage = compute_citation_coverage(results)
    eligibility_correctness = compute_eligibility(results)
    abstention_accuracy = compute_abstention(results)

    # ── Print report ──────────────────────────────────────
    print("\n" + "═" * 60)
    print("   EVALUATION REPORT")
    print("═" * 60)
    print(f"   Citation Coverage Rate:    {citation_coverage}%")
    print(f"   Eligibility Correctness:   {eligibility_correctness}%")
    print(f"   Abstention Accuracy:       {abstention_accuracy}%")
    print("═" * 60)

    # ── Per-query breakdown ───────────────────────────────
    print("\n   Per-Query Results:")
    print(f"   {'ID':<8} {'Type':<15} {'Citation':<10} {'Decision':<15} {'Time':<8}")
    print(f"   {'─'*8} {'─'*15} {'─'*10} {'─'*15} {'─'*8}")
    for r in results:
        print(
            f"   {r['id']:<8} {r['type']:<15} "
            f"{'✓' if r['has_citation'] else '✗':<10} "
            f"{r['decision']:<15} {r['time_seconds']}s"
        )

    # ── Example transcripts ──────────────────────────────
    print("\n\n" + "═" * 60)
    print("   EXAMPLE TRANSCRIPTS")
    print("═" * 60)
    print_example_transcript(results, "prereq_check")
    print_example_transcript(results, "chain")
    print_example_transcript(results, "not_in_docs")

    return {
        "citation_coverage": citation_coverage,
        "eligibility_correctness": eligibility_correctness,
        "abstention_accuracy": abstention_accuracy,
        "results": results,
    }


# ── CLI Entry Point ──────────────────────────────────────────

if __name__ == "__main__":
    from src.vectorstore import get_chunk_count

    count = get_chunk_count()
    if count == 0:
        print("\n⚠ No documents indexed in ChromaDB.")
        print("  Please upload and process catalog documents first via the Gradio UI,")
        print("  or run the ingestion + indexing pipeline before evaluating.")
        print("  Evaluation requires catalog data to produce meaningful results.\n")
    else:
        print(f"\n✓ Found {count} indexed chunks. Starting evaluation...\n")

    metrics = run_evaluation()

    # Save results to file
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    serializable_results = []
    for r in metrics["results"]:
        sr = {k: v for k, v in r.items() if k != "response"}
        sr["final_output_preview"] = r.get("response", {}).get("final_output", "")[:200]
        serializable_results.append(sr)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "citation_coverage": metrics["citation_coverage"],
                "eligibility_correctness": metrics["eligibility_correctness"],
                "abstention_accuracy": metrics["abstention_accuracy"],
                "results": serializable_results,
            },
            f,
            indent=2,
        )

    print(f"\n✓ Results saved to: {output_path}")
