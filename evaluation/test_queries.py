"""
Test queries for evaluating the Course Planning Assistant.

Defines 25 test queries across 4 categories:
    - prereq_checks (10): Prerequisite eligibility questions
    - chain_questions (5): Multi-hop prerequisite chain questions
    - program_questions (5): Program/degree requirement questions
    - trick_questions (5): Questions about info NOT in the catalog
"""

# ══════════════════════════════════════════════════════════════
# CATEGORY 1: Prerequisite Checks (10 queries)
# ══════════════════════════════════════════════════════════════

prereq_checks = [
    {
        "id": "PC001",
        "query": "Can I take CS301 if I completed CS101 and MATH120?",
        "type": "prereq_check",
        "student_profile": {
            "completed_courses": ["CS101", "MATH120"],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 15,
        },
        "expected_decision": "eligible_or_not_eligible",
        "expected_has_citation": True,
    },
    {
        "id": "PC002",
        "query": "Am I eligible to enroll in CS401 Data Structures II?",
        "type": "prereq_check",
        "student_profile": {
            "completed_courses": ["CS101", "CS201", "CS301"],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 15,
        },
        "expected_decision": "eligible_or_not_eligible",
        "expected_has_citation": True,
    },
    {
        "id": "PC003",
        "query": "Can I take MATH301 Differential Equations without MATH201?",
        "type": "prereq_check",
        "student_profile": {
            "completed_courses": ["MATH101", "MATH120"],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Spring",
            "max_credits": 15,
        },
        "expected_decision": "not_eligible",
        "expected_has_citation": True,
    },
    {
        "id": "PC004",
        "query": "I have completed CS101, CS201, and MATH201. Can I take CS350 Software Engineering?",
        "type": "prereq_check",
        "student_profile": {
            "completed_courses": ["CS101", "CS201", "MATH201"],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 18,
        },
        "expected_decision": "eligible_or_not_eligible",
        "expected_has_citation": True,
    },
    {
        "id": "PC005",
        "query": "Do I meet the prerequisites for CS460 Machine Learning?",
        "type": "prereq_check",
        "student_profile": {
            "completed_courses": ["CS101", "CS201", "CS301", "MATH201", "STAT301"],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 15,
        },
        "expected_decision": "eligible_or_not_eligible",
        "expected_has_citation": True,
    },
    {
        "id": "PC006",
        "query": "Can I take PHYS201 without completing PHYS101 first?",
        "type": "prereq_check",
        "student_profile": {
            "completed_courses": ["MATH101", "MATH120"],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Spring",
            "max_credits": 15,
        },
        "expected_decision": "not_eligible",
        "expected_has_citation": True,
    },
    {
        "id": "PC007",
        "query": "I got a D in CS201. Can I still take CS301?",
        "type": "prereq_check",
        "student_profile": {
            "completed_courses": ["CS101", "CS201"],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 15,
            "grades": {"CS201": "D"},
        },
        "expected_decision": "eligible_or_not_eligible",
        "expected_has_citation": True,
    },
    {
        "id": "PC008",
        "query": "Am I eligible for the Senior Capstone CS490?",
        "type": "prereq_check",
        "student_profile": {
            "completed_courses": ["CS101", "CS201", "CS301", "CS350"],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Spring",
            "max_credits": 15,
        },
        "expected_decision": "eligible_or_not_eligible",
        "expected_has_citation": True,
    },
    {
        "id": "PC009",
        "query": "Can I take ENG201 Advanced Writing? I completed ENG101.",
        "type": "prereq_check",
        "student_profile": {
            "completed_courses": ["ENG101", "CS101"],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 15,
        },
        "expected_decision": "eligible_or_not_eligible",
        "expected_has_citation": True,
    },
    {
        "id": "PC010",
        "query": "I want to take CS301 and CS350 together. Do I meet the prerequisites for both?",
        "type": "prereq_check",
        "student_profile": {
            "completed_courses": ["CS101", "CS201", "MATH120", "MATH201"],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 18,
        },
        "expected_decision": "eligible_or_not_eligible",
        "expected_has_citation": True,
    },
]

# ══════════════════════════════════════════════════════════════
# CATEGORY 2: Chain / Multi-hop Questions (5 queries)
# ══════════════════════════════════════════════════════════════

chain_questions = [
    {
        "id": "CQ001",
        "query": "What is the full prerequisite chain to reach Advanced Algorithms?",
        "type": "chain",
        "student_profile": {
            "completed_courses": [],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 15,
        },
        "expected_has_citation": True,
        "expected_hops": 2,
    },
    {
        "id": "CQ002",
        "query": "What courses do I need to complete before I can take CS490 Senior Capstone, starting from scratch?",
        "type": "chain",
        "student_profile": {
            "completed_courses": [],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 15,
        },
        "expected_has_citation": True,
        "expected_hops": 3,
    },
    {
        "id": "CQ003",
        "query": "What is the prerequisite path from introductory math to Machine Learning?",
        "type": "chain",
        "student_profile": {
            "completed_courses": ["MATH101"],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 15,
        },
        "expected_has_citation": True,
        "expected_hops": 2,
    },
    {
        "id": "CQ004",
        "query": "Show me the full course sequence to reach Operating Systems starting from CS101.",
        "type": "chain",
        "student_profile": {
            "completed_courses": ["CS101"],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 15,
        },
        "expected_has_citation": True,
        "expected_hops": 2,
    },
    {
        "id": "CQ005",
        "query": "What prerequisite chain do I need for Database Systems, and how many semesters will it take?",
        "type": "chain",
        "student_profile": {
            "completed_courses": [],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 15,
        },
        "expected_has_citation": True,
        "expected_hops": 2,
    },
]

# ══════════════════════════════════════════════════════════════
# CATEGORY 3: Program Requirement Questions (5 queries)
# ══════════════════════════════════════════════════════════════

program_questions = [
    {
        "id": "PQ001",
        "query": "How many elective credits do I need for a CS major?",
        "type": "program_req",
        "student_profile": {
            "completed_courses": [],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 15,
        },
        "expected_has_citation": True,
    },
    {
        "id": "PQ002",
        "query": "What are the core required courses for a Computer Science degree?",
        "type": "program_req",
        "student_profile": {
            "completed_courses": [],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 15,
        },
        "expected_has_citation": True,
    },
    {
        "id": "PQ003",
        "query": "How many total credit hours are required to graduate with a CS degree?",
        "type": "program_req",
        "student_profile": {
            "completed_courses": [],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 15,
        },
        "expected_has_citation": True,
    },
    {
        "id": "PQ004",
        "query": "What math courses are required for the CS program?",
        "type": "program_req",
        "student_profile": {
            "completed_courses": [],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 15,
        },
        "expected_has_citation": True,
    },
    {
        "id": "PQ005",
        "query": "Are there any general education requirements I need to complete for the CS major?",
        "type": "program_req",
        "student_profile": {
            "completed_courses": [],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 15,
        },
        "expected_has_citation": True,
    },
]

# ══════════════════════════════════════════════════════════════
# CATEGORY 4: Trick / Not-in-Docs Questions (5 queries)
# ══════════════════════════════════════════════════════════════

trick_questions = [
    {
        "id": "TQ001",
        "query": "Is CS301 offered in Summer 2026?",
        "type": "not_in_docs",
        "student_profile": {
            "completed_courses": ["CS101"],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 15,
        },
        "expected_decision": "abstain",
        "expected_has_citation": False,
    },
    {
        "id": "TQ002",
        "query": "What is the average GPA of students who take CS460?",
        "type": "not_in_docs",
        "student_profile": {
            "completed_courses": ["CS101", "CS201"],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 15,
        },
        "expected_decision": "abstain",
        "expected_has_citation": False,
    },
    {
        "id": "TQ003",
        "query": "Who is the instructor for CS301 next semester?",
        "type": "not_in_docs",
        "student_profile": {
            "completed_courses": ["CS101", "CS201"],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Spring",
            "max_credits": 15,
        },
        "expected_decision": "abstain",
        "expected_has_citation": False,
    },
    {
        "id": "TQ004",
        "query": "Can I get credit for an internship at Google toward my CS degree?",
        "type": "not_in_docs",
        "student_profile": {
            "completed_courses": ["CS101", "CS201", "CS301"],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 15,
        },
        "expected_decision": "abstain",
        "expected_has_citation": False,
    },
    {
        "id": "TQ005",
        "query": "What is the tuition cost per credit hour for CS courses?",
        "type": "not_in_docs",
        "student_profile": {
            "completed_courses": [],
            "target_major": "Computer Science",
            "catalog_year": "2024-2025",
            "target_term": "Fall",
            "max_credits": 15,
        },
        "expected_decision": "abstain",
        "expected_has_citation": False,
    },
]

# ══════════════════════════════════════════════════════════════
# ALL QUERIES
# ══════════════════════════════════════════════════════════════

ALL_QUERIES = prereq_checks + chain_questions + program_questions + trick_questions
