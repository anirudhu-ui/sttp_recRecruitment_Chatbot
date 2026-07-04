"""
RecruitFlow AI - Shared LLM client (Member 3)
Same provider/pattern as Member 2's router.py: lazy-init ChatGroq singleton,
key from .env (GROQ_KEY). No hardcoded keys.
"""
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

_llm = None


def get_llm():
    """Lazy init - built only on first real call, shared across all Member 3 agents."""
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_KEY missing from environment. Add it to your .env file."
            )
        _llm = ChatGroq(model="openai/gpt-oss-120b", api_key=api_key, temperature=0.3)
    return _llm
