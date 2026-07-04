"""RecruitFlow AI - Unknown query fallback node (Member 4). No LLM, never crashes."""
from state import RecruitmentState

FALLBACK_TEXT = """I couldn't confidently identify that recruitment action.

Try:
  - Get me top candidates
  - Rewrite this JD for a startup
  - Salary expectations for this role?"""


def fallback_node(state: RecruitmentState) -> dict:
    return {"agent_logs": ["Fallback: unknown query, showed suggestions"]}
