"""
RecruitFlow AI - Member 1, Responsibility 2.

parse_jd_node -> reads state['jd_text'], produces a structured
JobDescriptionData object via ChatGroq.with_structured_output (same
pattern as Day 4 - 04_structured_output.py), and writes jd_role /
jd_skills / jd_experience into shared state.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate

from models.schemas import JobDescriptionData
from models.state import RecruitmentState
from tools.resume_rag import VectorStoreError, get_llm

_JD_PARSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured hiring data from a raw job description. "
            "Read the JD text carefully and fill out every field of the "
            "schema as accurately as possible using only information "
            "present in the text. If a field genuinely isn't mentioned, "
            "leave it empty rather than inventing details.",
        ),
        ("user", "Job description text:\n\n{jd_text}"),
    ]
)


DEFAULT_SKILLS = [
    "Python",
    "Django",
    "Flask",
    "REST APIs",
    "SQL",
    "PostgreSQL",
    "MySQL",
    "Git",
    "pytest",
    "unittest",
    "Docker",
    "AWS",
    "Redis",
    "CI/CD",
    "React",
    "Celery",
    "RabbitMQ",
    "Kafka",
]


def _parse_jd_without_llm(jd_text: str) -> JobDescriptionData:
    role = ""
    role_match = re.search(r"^\s*(?:Job Title|Role)\s*:\s*(.+)$", jd_text, re.I | re.M)
    if role_match:
        role = role_match.group(1).strip()

    found_skills = []
    jd_lower = jd_text.lower()
    for skill in DEFAULT_SKILLS:
        if skill.lower() in jd_lower:
            found_skills.append(skill)

    experience = ""
    exp_match = re.search(
        r"(?:Experience Required|Experience)\s*:\s*([^\n]+(?:\n(?!\w+\s*:)[^\n]+)*)",
        jd_text,
        re.I,
    )
    if exp_match:
        experience = " ".join(exp_match.group(1).split())
    else:
        years_match = re.search(r"\b\d+\s*(?:to|-)\s*\d+\s+years\b", jd_text, re.I)
        if years_match:
            experience = years_match.group(0)

    return JobDescriptionData(
        role=role or "Backend Python Developer",
        skills=found_skills,
        experience=experience,
    )


def parse_jd_node(state: RecruitmentState) -> dict:
    """Reads state['jd_text'] and writes jd_role, jd_skills, jd_experience."""
    jd_text = state.get("jd_text")

    if not jd_text or not jd_text.strip():
        return {
            "error_message": "parse_jd_node: state['jd_text'] is empty - "
            "run load_jd_node first.",
            "agent_logs": ["parse_jd_node: no jd_text to parse"],
        }

    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(JobDescriptionData)
        chain = _JD_PARSE_PROMPT | structured_llm
        parsed: JobDescriptionData = chain.invoke({"jd_text": jd_text})
    except VectorStoreError as e:
        parsed = _parse_jd_without_llm(jd_text)
        return {
            "jd_role": parsed.role,
            "jd_skills": parsed.skills,
            "jd_experience": parsed.experience,
            "agent_logs": [
                f"parse_jd_node: LLM setup failed ({e}); used deterministic fallback"
            ],
        }
    except Exception as e:
        parsed = _parse_jd_without_llm(jd_text)
        return {
            "jd_role": parsed.role,
            "jd_skills": parsed.skills,
            "jd_experience": parsed.experience,
            "agent_logs": [
                f"parse_jd_node: structured-output failed ({e}); used deterministic fallback"
            ],
        }

    return {
        "jd_role": parsed.role,
        "jd_skills": parsed.skills,
        "jd_experience": parsed.experience,
        "agent_logs": ["Job description parsed into structured fields"],
    }
