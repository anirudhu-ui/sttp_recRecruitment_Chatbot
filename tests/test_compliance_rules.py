"""
Rule-compliance regression checks for the STTP recruitment chatbot brief.

Run:
    python tests/test_compliance_rules.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.salary_agent import _extract_location, _fallback_summary
from agents.screening_agent import _retrieve_local_chunks
from tools.resume_rag import ResumeRecord


def test_live_salary_results_are_summarized_before_cached_fallback():
    results = [
        {
            "title": "Backend Python Developer salary India",
            "url": "https://example.com/backend-python-salary",
            "content": (
                "Backend Python Developer salaries in India commonly show "
                "INR 8 LPA, with listings ranging from 5 LPA - 14 LPA."
            ),
        }
    ]

    summary = _fallback_summary(
        role="Backend Python Developer",
        experience_level="2 to 4 years",
        location="India",
        results=results,
    )

    assert not summary.startswith("[CACHED FALLBACK")
    assert "8 LPA" in summary
    assert "5 LPA - 14 LPA" in summary


def test_salary_location_parser_accepts_trailing_punctuation():
    assert _extract_location("Salary expectations for this role in India?") == "India"
    assert _extract_location("salary range in Bengaluru.") == "Bengaluru"


def test_india_salary_summary_prefers_lpa_over_dollars():
    results = [
        {
            "title": "Backend Python Developer salary India",
            "url": "https://example.com/backend-python-salary-india",
            "content": "India snippets mention $80,000 globally but 8 LPA - 14 LPA locally.",
        }
    ]

    summary = _fallback_summary(
        role="Backend Python Developer",
        experience_level="2 to 4 years",
        location="India",
        results=results,
    )

    assert "8 LPA - 14 LPA" in summary
    assert "$80,000" not in summary


def test_local_retrieval_selects_jd_relevant_resume_chunks():
    record = ResumeRecord(
        filename="candidate.txt",
        candidate_name="Candidate One",
        text=(
            "Name: Candidate One\n"
            "Education: B.Tech.\n\n"
            "Project: Built a Flask REST API with SQL and Docker for a hiring app.\n\n"
            "Other: Campus event coordination and public speaking."
        ),
    )

    chunks = _retrieve_local_chunks(
        record,
        query="Backend Python Developer Flask REST APIs SQL Docker",
        k=1,
    )

    assert len(chunks) == 1
    assert "Flask REST API" in chunks[0]
    assert "SQL" in chunks[0]


if __name__ == "__main__":
    test_live_salary_results_are_summarized_before_cached_fallback()
    test_salary_location_parser_accepts_trailing_punctuation()
    test_india_salary_summary_prefers_lpa_over_dollars()
    test_local_retrieval_selects_jd_relevant_resume_chunks()
    print("PASS - STTP compliance regression checks")
