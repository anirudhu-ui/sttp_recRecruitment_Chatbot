"""
RecruitFlow AI - Member 1 independent test.

Runs the full Member 1 slice of the graph on its own, with no
dependency on the other three members' nodes:

    load_jd_node -> parse_jd_node -> load_resumes_node -> screen_candidates_node

Usage:
    cd recruitflow_member1
    python -m tests.test_member1
"""

from __future__ import annotations

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, START, END

from models.state import RecruitmentState
from nodes.data_loader import load_jd_node, load_resumes_node
from agents.jd_parser import parse_jd_node
from agents.screening_agent import screen_candidates_node

REQUIRED_OUTPUT_KEYS = [
    "jd_text",
    "jd_role",
    "jd_skills",
    "jd_experience",
    "resume_count",
    "resumes_loaded",
    "ranked_candidates",
    "agent_logs",
]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")


def build_member1_graph():
    builder = StateGraph(RecruitmentState)
    builder.add_node("load_jd_node", load_jd_node)
    builder.add_node("parse_jd_node", parse_jd_node)
    builder.add_node("load_resumes_node", load_resumes_node)
    builder.add_node("screen_candidates_node", screen_candidates_node)

    builder.add_edge(START, "load_jd_node")
    builder.add_edge("load_jd_node", "parse_jd_node")
    builder.add_edge("parse_jd_node", "load_resumes_node")
    builder.add_edge("load_resumes_node", "screen_candidates_node")
    builder.add_edge("screen_candidates_node", END)

    return builder.compile()


def main():
    graph = build_member1_graph()

    initial_state: RecruitmentState = {
        "jd_path": os.path.join(DATA_DIR, "jd_sample.txt"),
        "resume_directory": os.path.join(DATA_DIR, "resumes"),
        "agent_logs": [],
    }

    print("=" * 70)
    print("Running Member 1 graph: load_jd -> parse_jd -> load_resumes -> screen")
    print("=" * 70)

    result = graph.invoke(initial_state)

    print("\n--- agent_logs ---")
    for line in result.get("agent_logs", []):
        print(f"  - {line}")

    if result.get("error_message"):
        print(f"\n!! error_message: {result['error_message']}")

    print("\n--- Structured JD ---")
    print(f"Role:       {result.get('jd_role')}")
    print(f"Skills:     {result.get('jd_skills')}")
    print(f"Experience: {result.get('jd_experience')}")

    print(f"\n--- Resumes loaded: {result.get('resume_count')} ---")
    for r in result.get("resumes_loaded", []):
        print(f"  - {r['filename']:30s} {r['candidate_name']}")

    print("\n--- Ranked candidates (highest match first) ---")
    ranked = result.get("ranked_candidates", [])
    for i, c in enumerate(ranked, start=1):
        print(f"{i:2d}. {c['candidate_name']:20s} "
              f"score={c['match_score']:3d}  source={c['resume_source']}")
        print(f"      matched: {c['matched_skills']}")
        print(f"      missing: {c['missing_skills']}")
        print(f"      reason:  {c['match_reason']}")

    print("\n--- Output key verification ---")
    missing_keys = [k for k in REQUIRED_OUTPUT_KEYS if k not in result]
    if missing_keys:
        print(f"FAIL - missing keys: {missing_keys}")
        sys.exit(1)
    else:
        print("PASS - all required state keys present:")
        print(f"  {REQUIRED_OUTPUT_KEYS}")

    print("\n--- Full result (JSON) ---")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
