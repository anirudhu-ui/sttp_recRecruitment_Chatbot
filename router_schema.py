"""
QueryClassification - structured output schema for LLM fallback router.
Used ONLY when deterministic rules can't confidently classify a query.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field

QueryType = Literal[
    "load_data",
    "count_applicants",
    "screen_candidates",
    "rewrite_jd",
    "interview_questions",
    "salary_search",
    "shortlist_action",
    "help",
    "unknown",
]


class QueryClassification(BaseModel):
    query_type: QueryType = Field(description="Exact route label for the recruiter query")
    reason: Optional[str] = Field(default=None, description="One-line reason for the classification")
