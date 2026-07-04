"""
RecruitFlow AI - Structured output schemas (Member 3)
Used with .with_structured_output() so LLM output is validated, not free text.
"""
from typing import List
from pydantic import BaseModel, Field


class JDRewriteResult(BaseModel):
    """Structured output for the JD rewrite agent."""
    rewritten_jd: str = Field(
        description="The full rewritten job description text, in the requested tone, "
        "preserving the original role purpose, critical skills, and experience requirements."
    )
    preserved_elements: List[str] = Field(
        default_factory=list,
        description="Short bullet list of which critical skills/requirements from the "
        "original JD were explicitly preserved.",
    )


class InterviewQuestionSet(BaseModel):
    """Structured output for the interview question agent."""
    technical_questions: List[str] = Field(
        description="Technical questions testing the JD's required skills that the "
        "candidate's resume claims to have."
    )
    candidate_specific_questions: List[str] = Field(
        description="Questions referencing specific projects, experience, or resume "
        "evidence unique to this candidate."
    )
    skill_gap_questions: List[str] = Field(
        description="Questions probing skills the JD requires but the candidate's "
        "resume does not clearly show."
    )
    role_specific_questions: List[str] = Field(
        description="Questions about role responsibilities/scope, grounded in jd_role."
    )
