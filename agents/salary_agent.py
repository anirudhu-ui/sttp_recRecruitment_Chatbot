"""
RecruitFlow AI - Salary Research Agent (Member 3, Responsibility 3 continued)
Node: salary_search_node

Primary path: Tavily live web search (salary_search_tool).
Fallback path: data/salary_fallback.json, ONLY on Tavily failure/missing key/no
results, and always clearly labeled as cached, not live.

This is the ONLY node in Member 3's module that touches Tavily.
"""
import json
import os
import re
from typing import Optional

from state import RecruitmentState
from tools.salary_search import salary_search_tool, TavilyNotConfigured
from agents.llm_client import get_llm

FALLBACK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "salary_fallback.json"
)

SUMMARY_SYSTEM_PROMPT = """You summarize live web search results into a short salary \
benchmark summary for a recruiter.

Rules:
1. Only use information present in the provided search results. Never invent numbers,
   ranges, or claims not supported by the given content.
2. If the results disagree with each other (different ranges/figures), say so
   explicitly and give the range of what was found rather than picking one number.
3. If the results are thin or vague, say that clearly instead of guessing.
4. Keep it to a short paragraph (3-5 sentences), recruiter-facing, no fluff.
"""


def _extract_location(salary_query: str) -> str:
    if not salary_query:
        return ""
    cleaned = salary_query.strip().rstrip("?.!,;:")
    m = re.search(r"\bin\s+([A-Za-z][A-Za-z\s]{2,30})$", cleaned)
    if m:
        return m.group(1).strip()
    return ""


def _extract_salary_figures(results) -> list[str]:
    figures: list[str] = []
    patterns = [
        r"(?:INR|Rs\.?)\s?[\d,.]+\s?(?:LPA|lakhs?|lakh|per annum|pa)?(?:\s?-\s?(?:INR|Rs\.?)?\s?[\d,.]+\s?(?:LPA|lakhs?|lakh|per annum|pa)?)?",
        r"\$[\d,.]+(?:\s?-\s?\$?[\d,.]+)?\s?(?:per year|annually|/yr|a year)?",
        r"[\d,.]+\s?(?:LPA|lakhs?|lakh)\s?(?:-\s?[\d,.]+\s?(?:LPA|lakhs?|lakh))?",
    ]
    for result in results or []:
        text = " ".join(
            str(result.get(key, "")) for key in ("title", "content") if result.get(key)
        )
        for pattern in patterns:
            for match in re.findall(pattern, text, flags=re.IGNORECASE):
                cleaned = " ".join(str(match).split())
                if cleaned and cleaned not in figures:
                    figures.append(cleaned)

    filtered: list[str] = []
    for figure in figures:
        figure_l = figure.lower()
        if any(figure_l != other.lower() and figure_l in other.lower() for other in figures):
            continue
        filtered.append(figure)
    return filtered[:5]


def _live_results_summary(role, experience_level, location, results) -> str:
    figures = _extract_salary_figures(results)
    if location and location.strip().lower() == "india":
        india_figures = [
            figure
            for figure in figures
            if re.search(r"\b(?:inr|rs\.?|lpa|lakh|lakhs)\b", figure, re.IGNORECASE)
        ]
        if india_figures:
            figures = india_figures
    location_text = f" in {location}" if location else ""
    exp_text = f" for {experience_level}" if experience_level else ""
    source_titles = []
    for result in results or []:
        title = result.get("title", "").strip()
        if title and title not in source_titles:
            source_titles.append(title)

    if figures:
        return (
            f"Live Tavily salary search for {role}{exp_text}{location_text} "
            f"found these salary figures in returned snippets: {', '.join(figures)}. "
            "Use this as a market benchmark, and mention that figures vary by company, "
            "city, and exact backend stack."
        )

    titles = "; ".join(source_titles[:3]) if source_titles else "salary benchmark sources"
    return (
        f"Live Tavily search ran for {role}{exp_text}{location_text}, but the returned "
        f"snippets did not expose a clean numeric salary range. Sources returned: {titles}. "
        "A valid GROQ_KEY can summarize richer page snippets, but no salary number is "
        "being invented here."
    )


def _fallback_summary(role, experience_level, location, results) -> str:
    if results:
        return _live_results_summary(role, experience_level, location, results)

    entry = _load_fallback(role, experience_level)
    if entry is not None:
        cached = entry.get("cached_summary", "")
        return "[CACHED FALLBACK - NOT LIVE DATA] " + cached

    return (
        f"No cached salary fallback is available for {role}. Configure TAVILY_API_KEY "
        "for live salary search, or add this role to data/salary_fallback.json."
    )


def _summarize_with_llm(role, experience_level, location, results) -> str:
    if not results:
        return (
            "No usable live results were returned for this query - cannot state a "
            "salary figure without fabricating one."
        )
    content_block = "\n\n".join(
        f"- {r['title']}: {r['content']}" for r in results if r.get("content")
    )
    human = (
        f"Role: {role}\nExperience level: {experience_level}\nLocation: {location or '(unspecified)'}\n\n"
        f"Search results:\n{content_block}\n\n"
        "Write the recruiter-facing salary summary now."
    )
    try:
        llm = get_llm()
        response = llm.invoke([("system", SUMMARY_SYSTEM_PROMPT), ("human", human)])
        return response.content.strip()
    except Exception:
        return _fallback_summary(role, experience_level, location, results)


def _load_fallback(role: str, experience_level: str) -> Optional[dict]:
    if not os.path.exists(FALLBACK_PATH):
        return None
    try:
        with open(FALLBACK_PATH, "r") as f:
            data = json.load(f)
    except Exception:
        return None

    role_l = (role or "").strip().lower()
    exp_l = (experience_level or "").strip().lower()
    best = None
    entries = data.get("entries", [])
    for entry in data.get("entries", []):
        if entry.get("role", "").strip().lower() == role_l:
            best = entry
            if entry.get("experience_level", "").strip().lower() == exp_l:
                return entry
    if best is not None:
        return best

    role_aliases = [
        (("backend", "python", "developer", "software"), "Software Engineer"),
        (("machine learning", "ml ", "llm", "ai "), "AI Engineer"),
        (("data", "analytics"), "Data Scientist"),
    ]
    for keywords, fallback_role in role_aliases:
        if any(keyword in role_l for keyword in keywords):
            for entry in entries:
                if entry.get("role", "").strip().lower() == fallback_role.lower():
                    return entry

    if entries:
        return entries[0]
    return best


def salary_search_node(state: RecruitmentState) -> dict:
    logs = []

    jd_role = state.get("jd_role", "")
    jd_experience = state.get("jd_experience", "")
    salary_query = state.get("salary_query", "")

    if not jd_role and not salary_query:
        return {
            "error_message": "salary_search_node: no jd_role or salary_query available to search for.",
            "agent_logs": ["salary_search_node: aborted, nothing to search"],
        }

    role = jd_role or salary_query
    experience_level = jd_experience or ""
    location = _extract_location(salary_query)

    # ---- Primary path: live Tavily ----
    try:
        results_a = salary_search_tool.invoke(
            {"role": role, "location": location, "experience_level": experience_level}
        )
        results_b = []
        if location:
            # second, broader query without location, in case the narrow one is thin
            results_b = salary_search_tool.invoke(
                {"role": role, "location": "", "experience_level": experience_level}
            )
        all_results = results_a + [r for r in results_b if r not in results_a]

        if all_results:
            summary_results = all_results
            if location and _extract_salary_figures(results_a):
                summary_results = results_a
            summary = _summarize_with_llm(role, experience_level, location, summary_results)
            sources = [r["url"] for r in all_results]
            logs.append(
                f"salary_search_node: used LIVE Tavily search "
                f"({len(all_results)} results, role='{role}')"
            )
            return {
                "salary_summary": summary,
                "salary_sources": sources,
                "agent_logs": logs,
            }

        logs.append("salary_search_node: Tavily returned zero usable results, trying fallback cache")

    except TavilyNotConfigured as e:
        logs.append(f"salary_search_node: TAVILY_API_KEY missing ({e}), trying fallback cache")
    except Exception as e:
        logs.append(f"salary_search_node: Tavily live search failed ({e}), trying fallback cache")

    # ---- Fallback path: cached JSON, clearly labeled ----
    entry = _load_fallback(role, experience_level)
    if entry is None:
        return {
            "error_message": (
                "salary_search_node: Tavily unavailable and no matching fallback cache "
                f"entry for role='{role}', experience='{experience_level}'."
            ),
            "salary_summary": "",
            "salary_sources": [],
            "agent_logs": logs + ["salary_search_node: fallback cache miss, no data to return"],
        }

    logs.append(
        f"salary_search_node: used FALLBACK CACHE (not live) for role='{entry.get('role')}'"
    )
    summary = "[CACHED FALLBACK - NOT LIVE DATA] " + entry.get("cached_summary", "")
    return {
        "salary_summary": summary,
        "salary_sources": entry.get("cached_sources", []),
        "agent_logs": logs,
    }
