"""
RecruitFlow AI - Member 1, Responsibility 5.

screen_candidates_node -> retrieves each candidate's most JD-relevant
resume evidence from the ChromaDB vector store, scores them with a
structured-output LLM call grounded in that retrieved evidence, and
writes a ranked list of CandidateMatch results into shared state.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate

from models.schemas import CandidateMatch
from models.state import RecruitmentState
from tools.resume_rag import (
    VectorStoreError,
    build_resume_vectorstore,
    get_candidate_chunks,
    get_llm,
    load_resume_files,
)

_SCREEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a recruiting analyst. Score ONE candidate against a job "
            "description using ONLY the retrieved resume evidence provided - "
            "do not assume skills or experience that aren't shown in the "
            "evidence. Be strict: a skill only counts as matched if the "
            "evidence clearly demonstrates it.",
        ),
        (
            "user",
            "Job description:\n"
            "Role: {jd_role}\n"
            "Required skills: {jd_skills}\n"
            "Required experience: {jd_experience}\n\n"
            "Candidate: {candidate_name}\n"
            "Resume source file: {resume_source}\n"
            "Retrieved resume evidence (most relevant sections only):\n"
            "{evidence}\n\n"
            "Score this candidate from 0-100 and fill out the schema.",
        ),
    ]
)


def _build_query(jd_role: str, jd_skills: list[str], jd_experience: str) -> str:
    skills_text = ", ".join(jd_skills) if jd_skills else ""
    return f"Role: {jd_role}. Required skills: {skills_text}. Experience: {jd_experience}."


def _tokenize_for_local_retrieval(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+#.]{1,}", text or "")
        if token.lower() not in {"and", "the", "for", "with", "this", "that", "role"}
    }


def _retrieve_local_chunks(record, query: str, k: int = 3) -> list[str]:
    """Small no-network retrieval fallback used only when embeddings/Chroma
    are unavailable. It still retrieves JD-relevant chunks per candidate
    instead of scoring the whole resume blindly."""
    text = record.text or ""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    query_tokens = _tokenize_for_local_retrieval(query)
    scored: list[tuple[int, int, str]] = []
    for idx, paragraph in enumerate(paragraphs):
        paragraph_tokens = _tokenize_for_local_retrieval(paragraph)
        overlap = len(query_tokens & paragraph_tokens)
        scored.append((overlap, -idx, paragraph))

    scored.sort(reverse=True)
    selected = [paragraph for overlap, _idx, paragraph in scored[:k] if overlap > 0]
    return selected or paragraphs[:k]


def _score_without_llm(
    record,
    jd_skills: list[str],
    jd_role: str,
    jd_experience: str,
    evidence_text: str | None = None,
) -> dict:
    scoring_text = evidence_text or record.text
    resume_lower = scoring_text.lower()
    matched = [skill for skill in jd_skills if skill.lower() in resume_lower]
    missing = [skill for skill in jd_skills if skill not in matched]

    if jd_skills:
        skill_score = int((len(matched) / len(jd_skills)) * 85)
    else:
        skill_score = 35

    role_bonus = 10 if jd_role and any(part in resume_lower for part in jd_role.lower().split()) else 0
    experience_bonus = 5 if jd_experience and "year" in resume_lower else 0
    match_score = min(100, skill_score + role_bonus + experience_bonus)

    if matched:
        reason = (
            f"Local retrieval fallback matched {', '.join(matched[:6])} "
            "in JD-relevant resume evidence."
        )
    else:
        reason = "Local retrieval fallback found little direct overlap with the JD skills."

    return {
        "candidate_name": record.candidate_name,
        "resume_source": record.filename,
        "match_score": match_score,
        "matched_skills": matched,
        "missing_skills": missing,
        "match_reason": reason,
    }


def screen_candidates_node(state: RecruitmentState) -> dict:
    """Reads jd_role/jd_skills/jd_experience/jd_text + resume_directory,
    retrieves per-candidate evidence from the resume vector store, scores
    every candidate, and writes ranked_candidates (highest score first)."""

    jd_role = state.get("jd_role")
    jd_skills = state.get("jd_skills") or []
    jd_experience = state.get("jd_experience")
    resume_directory = state.get("resume_directory")

    if not jd_role and not jd_skills and not jd_experience:
        return {
            "error_message": "screen_candidates_node: no structured JD fields "
            "found in state - run parse_jd_node first.",
            "agent_logs": ["screen_candidates_node: missing JD fields"],
        }

    resumes_loaded = state.get("resumes_loaded")
    try:
        if resumes_loaded:
            candidates = [
                (r["filename"], r.get("candidate_name", r["filename"]))
                for r in resumes_loaded
            ]
        else:
            # Fall back to re-loading from disk if resumes_loaded wasn't
            # populated in this run (e.g. node called standalone).
            records = load_resume_files(resume_directory)
            candidates = [(r.filename, r.candidate_name) for r in records]
    except Exception as e:
        return {
            "error_message": f"screen_candidates_node: could not determine "
            f"candidate list - {e}",
            "agent_logs": ["screen_candidates_node: no candidates available"],
        }

    if not candidates:
        return {
            "error_message": "screen_candidates_node: zero candidates to screen.",
            "ranked_candidates": [],
            "agent_logs": ["screen_candidates_node: empty candidate list"],
        }

    try:
        records = load_resume_files(resume_directory)
    except Exception as e:
        return {
            "error_message": f"screen_candidates_node: could not load resumes - {e}",
            "agent_logs": ["screen_candidates_node: resume load failed"],
        }

    records_by_filename = {record.filename: record for record in records}
    vectorstore = None
    retrieval_error = ""
    try:
        vectorstore = build_resume_vectorstore(records, resume_directory=resume_directory)
    except Exception as e:
        retrieval_error = str(e)

    query = _build_query(jd_role, jd_skills, jd_experience)

    chain = None
    llm_error = ""
    try:
        llm = get_llm(temperature=0.2)
        structured_llm = llm.with_structured_output(CandidateMatch)
        chain = _SCREEN_PROMPT | structured_llm
    except VectorStoreError as e:
        llm_error = str(e)

    results: list[CandidateMatch | dict] = []
    logs: list[str] = []
    errors: list[str] = []

    for filename, candidate_name in candidates:
        if vectorstore is None or chain is None:
            record = records_by_filename.get(filename)
            if record is None:
                errors.append(f"{filename}: resume record missing")
                continue
            local_chunks = _retrieve_local_chunks(record, query=query, k=3)
            local_evidence = "\n---\n".join(local_chunks)
            results.append(
                _score_without_llm(
                    record,
                    jd_skills,
                    jd_role or "",
                    jd_experience or "",
                    evidence_text=local_evidence,
                )
            )
            continue

        try:
            chunks = get_candidate_chunks(
                vectorstore, query=query, source_filename=filename, k=3
            )
        except VectorStoreError as e:
            errors.append(f"{filename}: retrieval failed - {e}")
            continue

        evidence = (
            "\n---\n".join(c.text for c in chunks)
            if chunks
            else "(no relevant resume evidence retrieved)"
        )

        try:
            match: CandidateMatch = chain.invoke(
                {
                    "jd_role": jd_role or "",
                    "jd_skills": ", ".join(jd_skills),
                    "jd_experience": jd_experience or "",
                    "candidate_name": candidate_name,
                    "resume_source": filename,
                    "evidence": evidence,
                }
            )
        except Exception as e:
            errors.append(f"{filename}: scoring failed - {e}")
            continue

        if not (0 <= match.match_score <= 100):
            errors.append(
                f"{filename}: invalid match_score {match.match_score} discarded"
            )
            continue

        # Trust the retrieval layer's identity over whatever the LLM echoed
        # back, so results always trace to the correct source file.
        match.resume_source = filename
        match.candidate_name = candidate_name
        results.append(match)

    if not results and errors:
        for filename, _candidate_name in candidates:
            record = records_by_filename.get(filename)
            if record is not None:
                local_chunks = _retrieve_local_chunks(record, query=query, k=3)
                results.append(
                    _score_without_llm(
                        record,
                        jd_skills,
                        jd_role or "",
                        jd_experience or "",
                        evidence_text="\n---\n".join(local_chunks),
                    )
                )
        logs.append(
            "screen_candidates_node: all LLM scoring attempts failed; "
            "used local retrieval fallback for ranking"
        )
        errors = []

    if results and isinstance(results[0], dict):
        results.sort(key=lambda m: m["match_score"], reverse=True)
        ranked_candidates = results
    else:
        results.sort(key=lambda m: m.match_score, reverse=True)
        ranked_candidates = [m.model_dump() for m in results]

    if vectorstore is None or chain is None:
        logs.append(
            f"Resume screening completed with local retrieval fallback - "
            f"{len(results)}/{len(candidates)} candidates scored"
        )
        if retrieval_error:
            logs.append(f"screen_candidates_node: RAG unavailable - {retrieval_error}")
        if llm_error:
            logs.append(f"screen_candidates_node: LLM unavailable - {llm_error}")
    else:
        logs.append(
            f"Resume RAG screening completed - {len(results)}/{len(candidates)} "
            "candidates scored"
        )
    if errors:
        logs.append(f"screen_candidates_node: {len(errors)} candidate(s) had errors")

    output: dict = {
        "ranked_candidates": ranked_candidates,
        "agent_logs": logs,
    }
    if errors and not results:
        output["error_message"] = "; ".join(errors)
    return output
