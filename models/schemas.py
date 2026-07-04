"""
RecruitFlow AI - Member 1 structured-output schemas.

These Pydantic models are passed to `llm.with_structured_output(...)`
(the same pattern used in Day 4 - 04_structured_output.py and the
Day 5 router demo, demo5_conditional_llm.py) so the LLM's response comes
back as a validated Python object instead of raw text we'd have to parse
ourselves.
"""

from typing import List
from pydantic import BaseModel, Field


class JobDescriptionData(BaseModel):
    """Structured fields extracted from a raw JD text file.

    Required fields per the hackathon brief: role, skills, experience.
    A couple of extra fields are included because they're cheap for the
    LLM to extract from the same JD text and are useful context for
    screening/rewriting later - they are NOT written into the shared
    graph state (which only exposes jd_role / jd_skills / jd_experience),
    they're only used internally by this module.
    """

    role: str = Field(description="The job title / role being hired for.")
    skills: List[str] = Field(
        description="Individual required or preferred skills, technologies, "
        "or tools mentioned in the JD, as a flat list of short strings."
    )
    experience: str = Field(
        description="The experience requirement as stated in the JD, e.g. "
        "'3-5 years' or 'Fresher / 0-1 years'. Keep the original phrasing."
    )
    education: str = Field(
        default="",
        description="Minimum education qualification mentioned in the JD, "
        "if any (e.g. 'B.Tech in CSE or related field'). Empty string if "
        "not mentioned.",
    )
    responsibilities: List[str] = Field(
        default_factory=list,
        description="Key responsibilities/duties listed in the JD, as a "
        "flat list of short strings. Empty list if not mentioned.",
    )


class CandidateMatch(BaseModel):
    """One candidate's RAG-grounded screening result against a JD."""

    candidate_name: str = Field(description="Candidate's full name.")
    resume_source: str = Field(
        description="Filename the resume was loaded from, so the result "
        "can be traced back to the exact source document."
    )
    match_score: int = Field(
        ge=0,
        le=100,
        description="Overall match score from 0 to 100, grounded in how well "
        "the retrieved resume context covers the JD's required skills and "
        "experience level. 0 = no relevant overlap, 100 = ideal match.",
    )
    matched_skills: List[str] = Field(
        default_factory=list,
        description="JD skills that the candidate's resume evidence "
        "clearly demonstrates.",
    )
    missing_skills: List[str] = Field(
        default_factory=list,
        description="JD skills that the retrieved resume context does not "
        "show any evidence of.",
    )
    match_reason: str = Field(
        description="One or two sentence justification for the score, "
        "grounded in the specific resume evidence retrieved."
    )


class CandidateMatchList(BaseModel):
    """Wrapper so a single structured-output call can return one
    candidate's result (used internally - the node itself assembles the
    final ranked list in Python across multiple per-candidate calls)."""

    matches: List[CandidateMatch] = Field(default_factory=list)
