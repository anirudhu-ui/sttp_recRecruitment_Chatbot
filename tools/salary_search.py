"""
RecruitFlow AI - Live Salary Search Tool (Member 3, Responsibility 3)
Tool: salary_search_tool

This is the ONLY tool in Member 3's module with Tavily/live-web access.
rewrite_jd_node and interview_questions_node never call this. Tool permission
is isolated by construction: the tool function lives here, and only
salary_search_agent.py imports it.
"""
import os
from typing import List, Dict
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

_client = None


class TavilyNotConfigured(Exception):
    """Raised when TAVILY_API_KEY is missing - forces caller to fall back, not fabricate."""


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("TAVILY_API_KEY") or os.getenv("TAVILY_KEY")
        if not api_key:
            raise TavilyNotConfigured(
                "TAVILY_API_KEY missing from environment (.env)."
            )
        try:
            from tavily import TavilyClient
        except ModuleNotFoundError as exc:
            raise TavilyNotConfigured(
                "tavily-python is not installed; install requirements or use fallback cache."
            ) from exc
        _client = TavilyClient(api_key=api_key)
    return _client


@tool
def salary_search_tool(role: str, location: str = "", experience_level: str = "") -> List[Dict]:
    """Search the live web for current salary benchmarks for a given job role,
    location, and experience level, using the Tavily search API.

    Args:
        role: Job role/title to search salary data for, e.g. "AI Engineer".
        location: Optional location/market, e.g. "India", "Bangalore", "Remote".
        experience_level: Optional experience band, e.g. "entry level", "5 years".

    Returns:
        A list of result dicts with keys: title, url, content, score. Only results
        actually returned by Tavily are included - never fabricated.

    Raises:
        TavilyNotConfigured: if TAVILY_API_KEY is not set.
        Exception: propagated from the Tavily client on API failure.
    """
    client = _get_client()

    query_parts = [role, experience_level, location, "salary"]
    query = " ".join(p for p in query_parts if p).strip()
    if not query:
        query = "average tech salary benchmark"

    response = client.search(
        query=query,
        search_depth="basic",
        max_results=5,
        include_answer=False,
    )

    results = []
    for r in response.get("results", []):
        url = r.get("url", "")
        if not url or not url.startswith(("http://", "https://")):
            continue  # never keep an invalid/fabricated-looking URL
        results.append(
            {
                "title": r.get("title", ""),
                "url": url,
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
            }
        )
    return results
