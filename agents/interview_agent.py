"""
RecruitFlow AI - Interview Question Agent (Member 3, Responsibility 2)
Node: interview_questions_node

Grounds questions in JD skills AND the selected candidate's resume record from
ranked_candidates. No Tavily access - pure LLM generation grounded in state.
"""
from state import RecruitmentState
from agents.member3_schemas import InterviewQuestionSet
from agents.llm_client import get_llm

SYSTEM_PROMPT = """You are an interview-question generator for an HR recruitment \
assistant.

Rules you MUST follow:
1. Every question must be grounded in either the JD's required skills/role or the
   specific candidate record you are given. NEVER produce generic, boilerplate
   interview questions unrelated to this JD or this candidate.
2. technical_questions: test the JD's required skills, prioritizing skills the
   candidate's resume claims to have (verify depth) and skills central to the role.
3. candidate_specific_questions: reference the candidate's actual listed projects,
   employers, or resume evidence by name/topic. If resume evidence is thin, ask
   questions that would surface more detail about what is listed.
4. skill_gap_questions: target JD-required skills that are missing or unclear in the
   candidate's resume, to assess how big the actual gap is.
5. role_specific_questions: about the responsibilities/scope of jd_role itself,
   not tied to a specific skill.
6. If recruiter feedback is given, adjust the mix/content of ALL categories to honor it
   (e.g. "add more LangGraph questions" -> add/weight LangGraph-specific items).
"""


def _find_candidate(ranked_candidates, selected_candidate):
    if not selected_candidate:
        return None
    target = selected_candidate.strip().lower()
    for cand in ranked_candidates or []:
        name = str(cand.get("candidate_name") or cand.get("name") or "").strip().lower()
        cand_id = str(
            cand.get("id") or cand.get("resume_source") or cand.get("filename") or ""
        ).strip().lower()
        if target == name or target == cand_id or target in name or target in cand_id:
            return cand
    return None


def _candidate_display_name(candidate, selected_candidate):
    return (
        candidate.get("candidate_name")
        or candidate.get("name")
        or selected_candidate
        or "Selected candidate"
    )


def _build_human_prompt(candidate, jd_role, jd_skills, jd_experience, jd_text, feedback):
    skills_str = ", ".join(jd_skills) if jd_skills else "(not specified)"
    matched = candidate.get("matched_skills", [])
    missing = candidate.get("missing_skills", [])
    projects = candidate.get("projects") or candidate.get("resume_summary") or "(none listed)"

    parts = [
        f"JD ROLE: {jd_role or '(not specified)'}",
        f"JD REQUIRED EXPERIENCE: {jd_experience or '(not specified)'}",
        f"JD CRITICAL SKILLS: {skills_str}",
        f"JD TEXT (for extra context): {jd_text or '(not provided)'}",
        "",
        f"CANDIDATE RECORD: {candidate}",
        f"CANDIDATE MATCHED SKILLS: {matched}",
        f"CANDIDATE MISSING SKILLS: {missing}",
        f"CANDIDATE PROJECTS/EVIDENCE: {projects}",
    ]
    if feedback:
        parts.append(f"\nRecruiter feedback (apply now): {feedback}")
    parts.append("\nGenerate the grounded interview question set now.")
    return "\n".join(parts)


def _fallback_question_set(candidate, jd_role, jd_skills, human_feedback):
    name = _candidate_display_name(candidate, "")
    matched = candidate.get("matched_skills", []) or []
    missing = candidate.get("missing_skills", []) or []
    focus_skill = matched[0] if matched else (jd_skills[0] if jd_skills else "the core stack")
    gap_skill = missing[0] if missing else (jd_skills[-1] if jd_skills else "a production challenge")
    feedback_suffix = f" Consider this recruiter note: {human_feedback}" if human_feedback else ""

    return {
        "technical_questions": [
            f"Can you walk through a production problem you solved using {focus_skill}?",
            f"How would you design and test a key service for the {jd_role or 'target'} role?",
        ],
        "candidate_specific_questions": [
            f"Your resume match highlights {', '.join(matched[:3]) or 'relevant experience'}. Which project best proves that depth?",
            f"What part of your past work would transfer most directly to this role?{feedback_suffix}",
        ],
        "skill_gap_questions": [
            f"The resume is less clear on {gap_skill}. How would you approach work that requires it?",
            "Which missing or weaker skill would you ramp up first for this job, and how?",
        ],
        "role_specific_questions": [
            f"What does success look like in the first 90 days for you as a {jd_role or 'new hire'}?",
            "How do you collaborate with product, frontend, and DevOps teams when backend priorities compete?",
        ],
    }


def interview_questions_node(state: RecruitmentState) -> dict:
    logs = []

    selected_candidate = state.get("selected_candidate", "")
    ranked_candidates = state.get("ranked_candidates", [])
    jd_role = state.get("jd_role", "")
    jd_skills = state.get("jd_skills", [])
    jd_experience = state.get("jd_experience", "")
    jd_text = state.get("jd_text", "")
    human_feedback = state.get("human_feedback", "")

    if not selected_candidate:
        return {
            "error_message": "interview_questions_node: selected_candidate is empty.",
            "agent_logs": ["interview_questions_node: aborted, no selected_candidate"],
        }

    if not ranked_candidates:
        return {
            "error_message": "interview_questions_node: ranked_candidates is empty - run screening first.",
            "agent_logs": ["interview_questions_node: aborted, empty ranked_candidates"],
        }

    if not jd_skills:
        logs.append(
            "interview_questions_node: jd_skills empty - technical/skill-gap "
            "questions will be weaker without them"
        )

    candidate = _find_candidate(ranked_candidates, selected_candidate)
    if candidate is None:
        return {
            "error_message": (
                f"interview_questions_node: candidate '{selected_candidate}' not found "
                "in ranked_candidates."
            ),
            "agent_logs": logs + ["interview_questions_node: candidate lookup failed"],
        }

    human_prompt = _build_human_prompt(
        candidate, jd_role, jd_skills, jd_experience, jd_text, human_feedback
    )

    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(InterviewQuestionSet)
        result = structured_llm.invoke(
            [("system", SYSTEM_PROMPT), ("human", human_prompt)]
        )
        question_set = result.model_dump()
    except Exception as e:
        question_set = _fallback_question_set(candidate, jd_role, jd_skills, human_feedback)
        logs.append(
            f"interview_questions_node: LLM call failed ({e}); used deterministic fallback"
        )

    total_q = sum(len(v) for v in question_set.values())
    if total_q == 0:
        return {
            "error_message": "interview_questions_node: LLM returned zero questions.",
            "agent_logs": logs + ["interview_questions_node: empty question set, not accepted"],
        }

    logs.append(
        f"interview_questions_node: generated {total_q} questions for "
        f"'{selected_candidate}' (feedback_applied={bool(human_feedback)})."
    )

    candidate_key = _candidate_display_name(candidate, selected_candidate)
    existing_questions = state.get("interview_questions") or {}
    if not isinstance(existing_questions, dict):
        existing_questions = {}
    questions_by_candidate = dict(existing_questions)
    questions_by_candidate[candidate_key] = question_set

    return {
        "interview_questions": questions_by_candidate,
        "agent_logs": logs,
    }
