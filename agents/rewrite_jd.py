"""
RecruitFlow AI - JD Rewrite Agent (Member 3, Responsibility 1)
Node: rewrite_jd_node

Grounds the rewrite in the structured JD fields (role, skills, experience).
Does not invent a new role. Adapts tone. Incorporates human_feedback if present.
No Tavily access - pure LLM generation grounded in JD text/fields.
"""
from state import RecruitmentState
from agents.member3_schemas import JDRewriteResult
from agents.llm_client import get_llm

VALID_TONES = {"startup", "professional", "concise", "inclusive", "technical"}
DEFAULT_TONE = "professional"

SYSTEM_PROMPT = """You are a job-description rewriting specialist for an HR recruitment \
assistant.

Rules you MUST follow:
1. You are given a STRUCTURED job description (role, skills, experience, full text).
   Rewrite it - do not invent a different role, seniority, or unrelated skill set.
2. Preserve, in the rewritten JD:
   - the role's core purpose
   - all critical/required skills given to you
   - the stated experience requirement
3. Adapt only the tone, structure, and phrasing to match the requested tone.
4. If recruiter feedback is provided, apply it as a targeted revision of the previous
   rewrite - do not discard the earlier grounding to satisfy the feedback.
5. Never fabricate skills, benefits, salary, or company details that were not given to you.

Tone definitions:
- startup: energetic, informal, mission-driven, short paragraphs.
- professional: neutral corporate tone, standard JD structure.
- concise: shortest possible faithful version, bullet-heavy, no filler.
- inclusive: bias-reduced language, welcoming tone, avoids gendered/exclusionary phrasing.
- technical: precise engineering register, assumes a technical reader, detailed on stack/skills.
"""


def _build_human_prompt(jd_text, jd_role, jd_skills, jd_experience, tone, feedback):
    skills_str = ", ".join(jd_skills) if jd_skills else "(not specified)"
    parts = [
        f"ROLE: {jd_role or '(not specified)'}",
        f"REQUIRED EXPERIENCE: {jd_experience or '(not specified)'}",
        f"CRITICAL SKILLS: {skills_str}",
        f"ORIGINAL JD TEXT:\n{jd_text}",
        f"\nRequested tone: {tone}",
    ]
    if feedback:
        parts.append(
            f"\nRecruiter feedback on the previous rewrite (apply this now): {feedback}"
        )
    parts.append(
        "\nProduce the rewritten JD now, grounded strictly in the fields above."
    )
    return "\n".join(parts)


def _fallback_rewrite(jd_text, jd_role, jd_skills, jd_experience, tone, feedback):
    skills = ", ".join(jd_skills) if jd_skills else "the required skills"
    heading = jd_role or "Open Role"
    lines = [
        f"{heading}",
        "",
        "About the role",
        jd_text.strip(),
        "",
        "What we are looking for",
        f"- Core skills: {skills}",
        f"- Experience: {jd_experience or 'as described in the original JD'}",
    ]
    if tone == "startup":
        lines.insert(2, "Join a fast-moving team and help build practical, reliable products.")
    elif tone == "concise":
        lines = [
            f"{heading}",
            f"Skills: {skills}",
            f"Experience: {jd_experience or 'See original JD'}",
            jd_text.strip(),
        ]
    if feedback:
        lines.extend(["", f"Recruiter revision note: {feedback}"])
    return "\n".join(lines)


def rewrite_jd_node(state: RecruitmentState) -> dict:
    logs = []

    jd_text = state.get("jd_text", "")
    jd_role = state.get("jd_role", "")
    jd_skills = state.get("jd_skills", [])
    jd_experience = state.get("jd_experience", "")
    human_feedback = state.get("human_feedback", "")
    rewrite_tone = state.get("rewrite_tone", "")

    if not jd_text or not jd_text.strip():
        return {
            "error_message": "rewrite_jd_node: jd_text is empty - load/parse a JD before rewriting.",
            "agent_logs": ["rewrite_jd_node: aborted, empty jd_text"],
        }

    if not rewrite_tone or rewrite_tone.lower() not in VALID_TONES:
        logs.append(
            f"rewrite_jd_node: rewrite_tone missing/invalid "
            f"('{rewrite_tone}'), defaulting to '{DEFAULT_TONE}'"
        )
        rewrite_tone = DEFAULT_TONE
    else:
        rewrite_tone = rewrite_tone.lower()

    human_prompt = _build_human_prompt(
        jd_text, jd_role, jd_skills, jd_experience, rewrite_tone, human_feedback
    )

    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(JDRewriteResult)
        result = structured_llm.invoke(
            [("system", SYSTEM_PROMPT), ("human", human_prompt)]
        )
    except Exception as e:
        fallback = _fallback_rewrite(
            jd_text, jd_role, jd_skills, jd_experience, rewrite_tone, human_feedback
        )
        return {
            "rewritten_jd": fallback,
            "agent_logs": logs + [
                f"rewrite_jd_node: LLM call failed ({e}); used deterministic fallback"
            ],
        }

    if not result.rewritten_jd or not result.rewritten_jd.strip():
        return {
            "error_message": "rewrite_jd_node: LLM returned an empty rewritten_jd.",
            "agent_logs": logs + ["rewrite_jd_node: empty rewrite from LLM, not accepted"],
        }

    logs.append(
        f"rewrite_jd_node: rewrote JD (tone={rewrite_tone}, "
        f"feedback_applied={bool(human_feedback)}, "
        f"preserved={result.preserved_elements})"
    )

    return {
        "rewritten_jd": result.rewritten_jd,
        "agent_logs": logs,
    }
