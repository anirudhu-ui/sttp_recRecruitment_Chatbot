"""
RecruitFlow AI - Salary Research Agent

Uses Tavily live web search for current salary evidence.
The LLM summarizes only Tavily evidence.

If the LLM is unavailable, the system returns the actual
Tavily source snippets instead of blindly extracting random
salary numbers.
"""

import json
import os
import re
from typing import Optional

from state import RecruitmentState
from tools.salary_search import salary_search_tool, TavilyNotConfigured
from agents.llm_client import get_llm


FALLBACK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "salary_fallback.json",
)


SUMMARY_SYSTEM_PROMPT = """
You are a recruitment salary research analyst.

Use ONLY the supplied LIVE WEB SEARCH EVIDENCE.

You are estimating salary for one specific job role,
experience level, and location.

IMPORTANT RULES:

1. Never invent salary figures.

2. Never invent salary ranges.

3. Never invent source names.

4. Ignore salary figures that clearly refer to:
   - unrelated job roles
   - senior leadership roles
   - company-wide maximum salaries
   - total company salary ranges
   - different countries
   - different currencies
   - monthly salaries when annual salary is requested

5. Prefer evidence matching:
   - the target role
   - the target experience level
   - the target location

6. If multiple reliable results provide different salary figures,
   explain that the sources vary.

7. Do NOT simply list isolated salary numbers.

8. Do NOT call a number an average unless the evidence explicitly
   describes it as an average.

9. Treat salary figures as market benchmarks, not guaranteed offers.

Return the answer in this exact recruiter-friendly structure:

Estimated Market Salary:
<Give a supported annual salary range or clearly say the evidence
does not support one reliable combined range.>

Market Benchmark:
<Explain the most useful salary benchmark found in the evidence.>

Experience Context:
<Explain how the candidate's experience level affects the salary.>

Compensation Factors:
<Briefly mention relevant technical skills or market factors that may
increase compensation.>

Market Note:
<Explain source disagreement, uncertainty, or market variation.>

Keep the complete response concise.
"""


def _extract_location(salary_query: str) -> str:
    if not salary_query:
        return ""

    cleaned = salary_query.strip().rstrip("?.!,;:")

    match = re.search(
        r"\bin\s+([A-Za-z][A-Za-z\s]{2,30})$",
        cleaned,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    return ""


def _format_search_evidence(results: list[dict]) -> str:
    evidence_blocks = []

    for index, result in enumerate(results, start=1):
        title = result.get("title", "").strip()
        url = result.get("url", "").strip()
        content = result.get("content", "").strip()

        if not content:
            continue

        evidence_blocks.append(
            f"""
SOURCE {index}
Title: {title}
URL: {url}
Evidence:
{content}
""".strip()
        )

    return "\n\n".join(evidence_blocks)


def _source_based_fallback(
    role: str,
    experience_level: str,
    location: str,
    results: list[dict],
) -> str:
    """
    Safe fallback when the LLM is unavailable.

    We deliberately do NOT use regex to combine random salary figures.
    Instead, we show the recruiter the actual live Tavily evidence.
    """

    if not results:
        return (
            "Live salary search returned no usable market evidence. "
            "A salary benchmark cannot be stated without supporting data."
        )

    location_text = location or "unspecified market"
    experience_text = experience_level or "unspecified experience"

    lines = [
        "Estimated Market Salary:",
        (
            "A reliable combined salary range could not be generated because "
            "the salary summarization model is currently unavailable."
        ),
        "",
        "Market Benchmark:",
        (
            f"Live Tavily evidence was retrieved for {role} in "
            f"{location_text}."
        ),
        "",
        "Experience Context:",
        f"Target experience: {experience_text}.",
        "",
        "Live Search Evidence:",
    ]

    for index, result in enumerate(results[:3], start=1):
        title = result.get("title", "Untitled source").strip()
        content = " ".join(result.get("content", "").split())

        if len(content) > 350:
            content = content[:347] + "..."

        lines.append(f"{index}. {title}")
        lines.append(f"   {content}")

    lines.extend(
        [
            "",
            "Market Note:",
            (
                "The live sources should be reviewed before stating a final "
                "salary range because the automated LLM summarizer is "
                "currently unavailable. No salary figure has been invented."
            ),
        ]
    )

    return "\n".join(lines)


def _summarize_with_llm(
    role: str,
    experience_level: str,
    location: str,
    results: list[dict],
) -> str:
    if not results:
        return (
            "No usable live salary results were returned. "
            "A salary figure cannot be stated without supporting evidence."
        )

    evidence = _format_search_evidence(results)

    if not evidence:
        return (
            "The live search returned results, but no usable salary evidence "
            "was available in the returned snippets."
        )

    human_prompt = f"""
TARGET ROLE:
{role}

TARGET EXPERIENCE:
{experience_level or "Not specified"}

TARGET LOCATION:
{location or "Not specified"}

LIVE WEB SEARCH EVIDENCE:

{evidence}

Analyze only the evidence above.

Create the recruiter-facing salary research summary now.
"""

    try:
        llm = get_llm()

        response = llm.invoke(
            [
                ("system", SUMMARY_SYSTEM_PROMPT),
                ("human", human_prompt),
            ]
        )

        summary = response.content.strip()

        if not summary:
            raise ValueError("LLM returned an empty salary summary.")

        return summary

    except Exception:
        return _source_based_fallback(
            role,
            experience_level,
            location,
            results,
        )


def _load_fallback(
    role: str,
    experience_level: str,
) -> Optional[dict]:

    if not os.path.exists(FALLBACK_PATH):
        return None

    try:
        with open(FALLBACK_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        return None

    role_lower = (role or "").strip().lower()
    experience_lower = (experience_level or "").strip().lower()

    entries = data.get("entries", [])

    best_match = None

    for entry in entries:
        entry_role = entry.get("role", "").strip().lower()
        entry_experience = (
            entry.get("experience_level", "").strip().lower()
        )

        if entry_role == role_lower:
            best_match = entry

            if entry_experience == experience_lower:
                return entry

    if best_match is not None:
        return best_match

    role_aliases = [
        (
            ("backend", "python", "developer", "software"),
            "Software Engineer",
        ),
        (
            ("machine learning", "ml ", "llm", "ai "),
            "AI Engineer",
        ),
        (
            ("data", "analytics"),
            "Data Scientist",
        ),
    ]

    for keywords, fallback_role in role_aliases:
        if any(keyword in role_lower for keyword in keywords):
            for entry in entries:
                if (
                    entry.get("role", "").strip().lower()
                    == fallback_role.lower()
                ):
                    return entry

    return None


def salary_search_node(state: RecruitmentState) -> dict:
    logs = []

    jd_role = state.get("jd_role", "")
    jd_experience = state.get("jd_experience", "")
    salary_query = state.get("salary_query", "")

    if not jd_role and not salary_query:
        return {
            "error_message": (
                "salary_search_node: no jd_role or salary_query "
                "available to search for."
            ),
            "agent_logs": [
                "salary_search_node: aborted, nothing to search"
            ],
        }

    role = jd_role or salary_query
    experience_level = jd_experience or ""
    location = _extract_location(salary_query)

    try:
        results = salary_search_tool.invoke(
            {
                "role": role,
                "location": location,
                "experience_level": experience_level,
            }
        )

        if results:
            summary = _summarize_with_llm(
                role,
                experience_level,
                location,
                results,
            )

            sources = []

            for result in results:
                url = result.get("url", "")

                if url and url not in sources:
                    sources.append(url)

            logs.append(
                "salary_search_node: used LIVE Tavily search "
                f"({len(results)} results, role='{role}')"
            )

            return {
                "salary_summary": summary,
                "salary_sources": sources,
                "error_message": "",
                "agent_logs": logs,
            }

        logs.append(
            "salary_search_node: Tavily returned zero usable results"
        )

    except TavilyNotConfigured as error:
        logs.append(
            "salary_search_node: TAVILY_API_KEY missing "
            f"({error})"
        )

    except Exception as error:
        logs.append(
            "salary_search_node: Tavily live search failed "
            f"({error})"
        )

    entry = _load_fallback(
        role,
        experience_level,
    )

    if entry is None:
        return {
            "error_message": (
                "Salary research failed: live Tavily data was unavailable "
                "and no matching cached benchmark exists."
            ),
            "salary_summary": "",
            "salary_sources": [],
            "agent_logs": logs,
        }

    summary = (
        "[CACHED FALLBACK - NOT LIVE DATA]\n"
        + entry.get("cached_summary", "")
    )

    logs.append(
        "salary_search_node: used FALLBACK CACHE "
        f"for role='{entry.get('role', '')}'"
    )

    return {
        "salary_summary": summary,
        "salary_sources": entry.get("cached_sources", []),
        "error_message": "",
        "agent_logs": logs,
    }
