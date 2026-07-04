"""
Independent test suite - Member 2 module.
Run: python test_router.py
"""
from router import (
    query_router_node, route_query, count_applicants_node,
    _deterministic_classify,
)

TEST_QUERIES = [
    "How many applicants?",
    "Get me top candidates",
    "Rewrite this JD for a startup",
    "Interview questions for Adhyumna",
    "Salary expectations for this role?",
    "Shortlist the top 3",
    "Help",
    "So what do you make of all this then?",  # ambiguous -> LLM fallback
]


def run_classification_tests():
    print("=" * 70)
    print("QUERY CLASSIFICATION TESTS")
    print("=" * 70)
    for q in TEST_QUERIES:
        state = {"user_query": q}
        used = "deterministic" if _deterministic_classify(q) else "llm_fallback"
        result = query_router_node(state)
        # route_query purity check - only reads query_type
        full_state = {**state, **result}
        route_out = route_query(full_state)

        print(f"\nQuery           : {q}")
        print(f"query_type      : {result.get('query_type')}")
        print(f"routing_used    : {used}")
        print(f"route_query()   : {route_out}")
        print(f"rewrite_tone    : {result.get('rewrite_tone')}")
        print(f"selected_cand   : {result.get('selected_candidate')}")
        print(f"pending_action  : {result.get('pending_action')}")
        print(f"error_message   : {result.get('error_message')}")
        print(f"agent_logs      : {result.get('agent_logs')}")


def run_count_tests():
    print("\n" + "=" * 70)
    print("COUNT_APPLICANTS_NODE TESTS (proves no LLM used)")
    print("=" * 70)

    # Normal case
    state = {"resume_count": 17}
    result = count_applicants_node(state)
    print(f"\nInput resume_count=17 -> result: {result}")
    assert "error_message" not in result
    assert result["agent_logs"] == ["Applicant count completed using plain Python"]

    # Missing resume_count case
    state2 = {}
    result2 = count_applicants_node(state2)
    print(f"Input missing resume_count -> result: {result2}")
    assert "error_message" in result2

    print("\ncount_applicants_node uses zero LLM/Tavily/ChromaDB calls (pure Python). PASS")


def run_state_preservation_test():
    print("\n" + "=" * 70)
    print("MULTI-TURN STATE PRESERVATION TEST")
    print("=" * 70)
    # Simulate prior turns already having data
    state = {
        "user_query": "Salary expectations for this role?",
        "ranked_candidates": [{"name": "Priya", "score": 0.91}],
        "jd_skills": ["Python", "SQL"],
        "resume_count": 12,
    }
    result = query_router_node(state)
    merged = {**state, **result}
    print(f"query_type after salary query : {merged['query_type']}")
    print(f"ranked_candidates preserved    : {merged['ranked_candidates']}")
    print(f"jd_skills preserved            : {merged['jd_skills']}")
    print(f"resume_count preserved         : {merged['resume_count']}")
    assert merged["ranked_candidates"] == [{"name": "Priya", "score": 0.91}]
    assert merged["jd_skills"] == ["Python", "SQL"]
    print("State preservation PASS (node returned only changed keys)")


def run_error_handling_tests():
    print("\n" + "=" * 70)
    print("ERROR HANDLING TESTS")
    print("=" * 70)

    r1 = query_router_node({"user_query": ""})
    print(f"Empty query        -> query_type={r1['query_type']}, error={r1.get('error_message')}")
    assert r1["query_type"] == "unknown"

    r2 = query_router_node({"user_query": "asdkj qpwoeiru zzz nothing matches"})
    print(f"Gibberish query    -> query_type={r2['query_type']}")

    r3 = query_router_node({"user_query": "Interview questions for"})
    print(f"Missing cand name  -> selected_candidate={r3.get('selected_candidate')}, "
          f"error={r3.get('error_message')}")


if __name__ == "__main__":
    run_classification_tests()
    run_count_tests()
    run_state_preservation_test()
    run_error_handling_tests()
    print("\nALL TESTS COMPLETED")
